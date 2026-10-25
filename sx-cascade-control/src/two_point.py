"""Two-point control: two measurements inside the cascade, each held by one handle.

The way distillation columns are often run, and the simplest structure that holds an SX cascade's
split without a model of it. The scrub flow (strip tied to it) holds measurement A, usually a
composition near the product end of the scrub; the organic flow holds measurement B, usually a
composition in the extraction section. Each loop is an integral controller: every control step it
moves its handle by a fraction of the error, so it keeps moving until its measurement is back at
the setpoint, whatever the plant's numbers are. A PI level loop on every settler holds the
interfaces underneath (loop.LevelLoops).

The only model information used is the long-run gain of each measurement to its handle at the
design point (finite differences on the nominal model's steady state): its sign, and its size to
set the loop speed. A plant whose gains are two or three times off still settles, only slower or
faster. (Notebook 22 picks A and B; notebook 23 runs this controller.)

Measurements (aqueous phase of one stage):
    lace_frac   La+Ce share of the metal        pr        Pr, mol/L
    pr_ratio    Pr / (Pr + Nd)                  ndpr_frac (Pr + Nd) share of the metal
    metal       total metal, mol/L

    loop i:   u_i(k+1) = u_i(k) + t_step / (g_ii * T_cl) * (target_i - y_i(k)),  clipped to the rate
              limit and the handle's bounds (the MPC's), g_ii = d y_i / d u_i at steady state

Each loop works on the logarithm of its measurement (log_scale=True): an impurity near zero
flattens out as the product gets purer, and the loop would lose its signal (it can't tell 99.9 %
from 99.99 %). Log compositions keep the gain about the same across that range, the usual fix in
distillation composition control (Skogestad and Morari).

With swap=True the pairing is the other way round: the organic holds A and the scrub holds B.
Which one is right is what the relative gain says: lambda for (scrub-A, organic-B), 1 - lambda for
the swap; a pairing with a negative relative gain makes the two integral loops fight.

Judgement call: T_cl (the closed-loop time constant asked of each loop) defaults to 20 h; the
slowest open-loop mode of this circuit is hundreds of hours, so this is already asking a lot. The
loops aren't decoupled; the pairing (scrub-A, organic-B) is checked by the relative gain at the
design point, printed when the controller is built.
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, Inputs
from .simulate import steady_state

KINDS = ("lace_frac", "pr", "pr_ratio", "ndpr_frac", "metal")


def measure(circuit: Circuit, x: np.ndarray, kind: str, stage: int) -> float:
    """One measurement of the aqueous phase of `stage` in state x (model layout)."""
    c = np.asarray(circuit.unpack(x)[0][stage], float).ravel()
    names = list(circuit.p.names)
    la, pr, nd = c[names.index("LaCe")], c[names.index("Pr")], c[names.index("Nd")]
    tot = max(float(c.sum()), 1e-12)
    return {"lace_frac": la / tot, "pr": pr, "pr_ratio": pr / max(pr + nd, 1e-12),
            "ndpr_frac": (pr + nd) / tot, "metal": tot}[kind]


class TwoPointController:
    """Scrub holds measurement A, organic holds measurement B. Satisfies loop.Controller.

    Args:
        model: the nominal model (for the gains and to read measurements from the state).
        x_model, u_model: the model at rest at the operating point (targets are taken there).
        cut: the cut ("LaCe").
        hold_a, hold_b: (kind, stage) for each loop.
        t_step: the control interval, h.
        bounds, rate_per_step: the MPC's handle bounds and move limits (dicts by handle name).
        net_product: strip = net_product + scrub, L/h.
        closed_loop_h: T_cl, h.
        swap: True pairs the organic with A and the scrub with B (see the module docstring).
        log_scale: True runs each loop on the log of its measurement.

    The relative gain of the pairing (scrub-A, organic-B) is kept in `relative_gain`.
    """

    def __init__(self, model: Circuit, x_model: np.ndarray, u_model: Inputs, cut: str, hold_a: tuple,
                 hold_b: tuple, t_step: float, bounds: dict, rate_per_step: dict, net_product: float,
                 closed_loop_h: float = 20.0, swap: bool = False, log_scale: bool = True, level_loops=None):
        """Take the targets at the operating point, measure the gains there, and set each loop's
        integral gain so it closes in about closed_loop_h. Arguments as in the class docstring.

        Why fourteen parameters: the model and its operating point, the two loops' targets, and tuning keywords with defaults from the run file; registry._two_point builds it.
        """
        self.model, self.u_nominal, self.cut = model, u_model, cut
        self.level_loops = level_loops      # the settler loops under it (loop.LevelLoops), or None
        self.log_scale = bool(log_scale)
        self._t_step, self.hold = t_step, (tuple(hold_a), tuple(hold_b))
        self.bounds, self.rate, self.net = bounds, rate_per_step, float(net_product)
        self.targets = self._read_both(x_model)
        self.G = self._gains(x_model, u_model)                   # rows: A, B; columns: scrub, organic
        self.swap = bool(swap)
        # gain from each handle (scrub, organic) to the measurement it holds
        g_loop = (self.G[1, 0], self.G[0, 1]) if self.swap else (self.G[0, 0], self.G[1, 1])
        self.ki = np.array([t_step / (g_loop[0] * closed_loop_h), t_step / (g_loop[1] * closed_loop_h)])
        self.u = np.array([float(u_model.Q_scrub), float(u_model.Q_org)])
        self.last = {}
        self.relative_gain = _relative_gain(self.G)

    def _gains(self, x0: np.ndarray, u0: Inputs) -> np.ndarray:
        """Long-run gain of A and B to scrub (+2 L/h, strip tied) and organic (+20 L/h), nominal model.

        Each column is a finite difference between two steady states. Returns G (2, 2): rows A, B;
        columns scrub, organic.
        """
        y0 = self._read_both(x0)
        G = np.zeros((2, 2))
        for col, (field, du) in enumerate((("Q_scrub", 2.0), ("Q_org", 20.0))):
            x = steady_state(self.model, self._stepped(u0, field, du), x0=x0)
            G[:, col] = (self._read_both(x) - y0) / du
        return G

    def _stepped(self, u0: Inputs, field: str, du: float) -> Inputs:
        """u0 with one handle moved by du; a scrub move takes the strip with it (strip tied)."""
        u = u0.replace(**{field: float(getattr(u0, field)) + du})
        if field == "Q_scrub":
            u = u.replace(Q_strip=float(u.Q_scrub) + self.net)
        return u

    def _read(self, x: np.ndarray, hold: tuple) -> float:
        """One loop's measurement from state x, as the loop sees it (its log, with log_scale)."""
        v = measure(self.model, x, *hold)
        return float(np.log(max(v, 1e-9))) if self.log_scale else v

    def _read_both(self, x: np.ndarray) -> np.ndarray:
        """Both loops' measurements from state x, as an array (A, B)."""
        return np.array([self._read(x, h) for h in self.hold])

    @property
    def t_step(self) -> float:
        """The control interval, h."""
        return self._t_step

    def step(self, obs) -> dict:
        """Read A and B from the state as known, move each handle by its integral law."""
        y = self._read_both(np.asarray(obs.x, float))
        err = self.targets - y                                   # (A, B)
        du = self.ki * (err[::-1] if self.swap else err)         # (scrub, organic)
        limits = np.array([self.rate["Q_scrub"], self.rate["Q_org"]])
        du = np.clip(du, -limits, limits)
        lo = np.array([self.bounds["Q_scrub"][0], self.bounds["Q_org"][0]])
        hi = np.array([self.bounds["Q_scrub"][1], self.bounds["Q_org"][1]])
        self.u = np.clip(self.u + du, lo, hi)
        self.last = {"a_seen": float(y[0]), "b_seen": float(y[1]), "a_target": float(self.targets[0]),
                     "b_target": float(self.targets[1])}          # in log units with log_scale
        return {"Q_scrub": float(self.u[0]), "Q_org": float(self.u[1])}

    def apply(self, handles: dict) -> Inputs:
        """Nominal inputs with the two handles written in, strip tied to scrub."""
        u = self.u_nominal.replace(**handles)
        return u.replace(Q_strip=self.net + float(u.Q_scrub))


def _relative_gain(G: np.ndarray) -> float:
    """Relative gain of the pairing (scrub-A, organic-B) from the 2x2 gain matrix G.

    lambda = G_AS * G_BO / (G_AS * G_BO - G_AO * G_BS). Near 1 the loops barely interact; below 0
    the pairing makes the two integral loops fight (the swap then has 1 - lambda).
    """
    return float(G[0, 0] * G[1, 1] / (G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]))
