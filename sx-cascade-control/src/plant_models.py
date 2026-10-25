"""Plants with mechanisms the controller's model does not have (structural mismatch).

Where this sits
---------------
The controller plans on circuit.Circuit. To test it fairly, the "plant" it controls in a
simulation should not be an exact copy of its model. Some plants differ only in numbers (a
tired mixer, a biased flowmeter: mismatch.py and disturbances.py). The two here differ in
structure: they have physics the model leaves out (ARCHITECTURE.md, decision 25;
APPENDIX_A_model.md, A.12). Both are subclasses of Circuit and reuse its pieces (inflows,
transfer, saponification, interface), so only what's new is written here.

The model's stage, and a real one
---------------------------------
In the model, each stage is one well-mixed pool per phase, mixer and settler together, with metal
moving between the phases anywhere in that pool. A real stage is a small mixer where the transfer
happens and a settler four times its size where the phases only separate and flow on, close to
plug flow (the settler hydrodynamics in dynamic models of copper SX circuits, e.g. Komulainen et
al. 2006, are a mixed zone followed by plug flow).

SettlerCircuit builds that: per stage, a mixer (well mixed, transfer here only) followed by a
settler split into `cells` tanks in series per phase. The total hold-up is the model's; only
where it sits and how it flows differ. At steady state the settler passes on what the mixer
makes, so the steady states agree and only the dynamics differ: a pure test of whether the
controller copes with dynamics its model does not have.

    mixer of stage j:   V_aq,mix  * dc_aq/dt  = aq in  - Q_aq,j * c_aq  - r
                        V_org,mix * dc_org/dt = org in - Q_org  * c_org + r
                        V_aq,mix  * dh/dt     = acid in - Q_aq,j * h + 3 sum(r) - dose
    settler cell k:     V_cell * dc/dt = Q * (c in cell k-1 - c in cell k), each phase on its own

with V_aq,mix = V_mix * phi_j and V_org,mix = V_mix * (1 - phi_j), the mixer split by the
aqueous share of the flow, as in the model. What leaves the stage is the last cell's contents.

Entrainment
-----------
EntrainmentCircuit keeps the model's state and adds what imperfect settlers do: each phase
carries droplets of the other into the next stage, more of them as the interface strays toward
a weir, so some metal is pushed back the wrong way. The steady state differs a little from
the model's (500 ppm of entrainment, chosen); the rise past the window is what matters.

State layout
------------
SettlerCircuit's state: the model's own layout first, holding what leaves each stage (the last
settler cell of each phase, and the interface), then the mixers, then the inner settler cells.
So sensors, metrics and commissioning read the first part exactly as they read the model;
`observable` hands that part to anyone acting as an instrument.

    [ model layout: outlet cells and z | mixers | settler cell 1 | ... | settler cell cells-1 ]

Each block after the model layout is [c_aq (N*n) | c_org (N*n) | h (N)].
"""
from __future__ import annotations

from typing import NamedTuple

import casadi as ca
import numpy as np

from .circuit import Circuit, Inputs, _StageOutlets


class _MixerFeed(NamedTuple):
    """What comes into the mixer of one stage, and the aqueous flow through it (all symbolic)."""
    aq_mol_in: ca.SX     # (n,) mol/h of metal arriving in the aqueous
    acid_in: ca.SX       # mol/h of H+ arriving in the aqueous
    org_mol_in: ca.SX    # (n,) mol/h of metal arriving in the organic
    base_dose: ca.SX     # mol/h of saponification base spent in this stage
    Q_aq: ca.SX          # aqueous flow through the stage, L/h


def _stage(block, j: int) -> tuple:
    """Stage j's (c_aq, c_org, h) out of a block of per-stage lists (c_aq, c_org, h)."""
    c_aq, c_org, h = block
    return c_aq[j], c_org[j], h[j]


def _store(rates, j: int, d) -> None:
    """Put stage j's three derivatives d = (d c_aq, d c_org, d h) into the per-stage lists `rates`."""
    rates[0][j], rates[1][j], rates[2][j] = d


def _between_stages(e) -> bool:
    """True for an edge (circuit.Edge) from one stage to another; False for a source or a sink."""
    return isinstance(e.src, int) and isinstance(e.dst, int)


class SettlerCircuit(Circuit):
    """Mixer plus a settler of `cells` tanks in series per phase, for every stage.

    cells: number of tanks each phase's settler hold-up is split into (3 in the robustness tests).
    The other arguments are Circuit's.
    Raises ValueError for fewer than one cell.

    Why: a well-mixed mixer followed by near plug flow in the settler is the structure used in
    dynamic models of copper SX circuits (Komulainen et al. 2006, as cited above).
    Judgement call: three cells per phase, and the settler volume is the model's (see
    Circuit._holdups on why that settler is small for these flows). More cells means closer to
    plug flow and a sharper delay. A tracer test on a plant settler would settle the number.

    Example:
        plant = SettlerCircuit(cmap, p, cells=3, scrub_reflux=True)
        y = plant.observable(x)        # the part of x laid out as the model's state
    """

    def __init__(self, *args, cells: int = 3, **kwargs):
        """cells: tanks per phase in each settler; everything else goes to Circuit.

        Raises ValueError for fewer than one cell.
        """
        if cells < 1:
            raise ValueError("a settler needs at least one cell")
        self.cells = cells
        super().__init__(*args, **kwargs)

    # ---- sizes and packing -------------------------------------------------
    @property
    def block(self) -> int:
        """Size of one copy of the per-stage concentrations: aqueous, organic, acid."""
        return 2 * self.N * self.n + self.N

    @property
    def nx(self) -> int:
        """Length of the state: the model's layout, then `cells` more blocks (mixers and inner cells)."""
        return self.nx_model + self.cells * self.block

    def observable(self, x) -> np.ndarray:
        """The first part of the state, laid out as the model's: what leaves each stage, and z."""
        return np.asarray(x, float)[: self.nx_model]

    def _unpack_block(self, v):
        """Split one block [c_aq (N*n) | c_org (N*n) | h (N)] into per-stage lists (c_aq, c_org, h)."""
        N, n = self.N, self.n
        c_aq = [v[j * n:(j + 1) * n] for j in range(N)]
        c_org = [v[N * n + j * n: N * n + (j + 1) * n] for j in range(N)]
        h = [v[2 * N * n + j] for j in range(N)]
        return c_aq, c_org, h

    def _blocks(self, x) -> list:
        """[mixer, settler cell 1, ..., settler cell cells-1, outlet cell]; each (c_aq, c_org, h).

        In flow order: the mixer first, the outlet cell last. The mixer and inner cells come from
        the extra blocks after the model layout; the outlet cell is the model layout itself.
        """
        start, b = self.nx_model, self.block
        inner = [self._unpack_block(x[start + k * b: start + (k + 1) * b]) for k in range(self.cells)]
        c_aq, c_org, h, _ = self.unpack(x)
        return inner + [(c_aq, c_org, h)]

    def initial_state(self, u: Inputs) -> np.ndarray:
        """The model's initial state, with every mixer and inner cell starting the same as the outlet."""
        x0 = super().initial_state(u)
        return np.concatenate([x0, np.tile(x0[: self.block], self.cells)])

    # ---- the equations -----------------------------------------------------
    def _build(self):
        """Build the right-hand side and outlet function for the mixer-plus-settler plant.

        Same steps as Circuit._build, except that the inflows come from each stage's outlet cell,
        transfer happens in the mixer only, and each settler cell just passes on what the cell
        before it holds.
        """
        N = self.N

        # Symbols, and the state split into blocks in flow order (mixer first, outlet cell last).
        x = ca.SX.sym("x", self.nx)
        uv = ca.SX.sym("u", Inputs.size(self.n, N))
        u = Inputs.from_vector(uv, self.n, N)
        blocks = self._blocks(x)
        out_aq, out_org, out_h = blocks[-1]
        z = self.unpack(x)[3]

        # What enters each stage comes from the neighbours' outlet cells.
        Q_aq = self.aqueous_flows(u)
        outlets = _StageOutlets(Q_aq, out_aq, out_h, out_org)
        inflows = self._inflows(u, outlets)
        doses = self._saponification_doses(u, blocks[0][2])       # base meets the mixer's acid
        rates = self._block_rates(u, blocks, inflows, doses, Q_aq)

        # The interface follows the densities of what leaves the stage, as in the model.
        dz = [self._interface_rate(u, j, out_aq[j], out_org[j], z[j]) for j in range(N)]
        self._make_rhs_functions(x, uv, self._stack_rates(rates, dz))
        self._make_outlet_function(x, uv, u, outlets)

    def _block_rates(self, u: Inputs, blocks, inflows, doses, Q_aq) -> list:
        """Derivatives of every block, stage by stage: the mixer, then each settler cell in turn.

        u: the inputs. blocks: the state in flow order (see _blocks). inflows: (aq_in, acid_in,
        org_in) per stage, mol/h. doses: base spent per stage, mol/h. Q_aq: aqueous flow out of
        each stage, L/h.
        Returns rates, one per block: rates[k] = (d c_aq, d c_org, d h), lists of N entries.

        Why five parameters: as Circuit._stage_balances, pieces already worked out per stage by the caller.
        """
        aq_in, acid_in, org_in = inflows
        V_cells = self._cell_volumes()
        rates = [([None] * self.N, [None] * self.N, [None] * self.N) for _ in blocks]
        for j in range(self.N):
            feed = _MixerFeed(aq_in[j], acid_in[j], org_in[j], doses[j], Q_aq[j])
            _store(rates[0], j, self._mixer(u, j, _stage(blocks[0], j), feed))
            self._store_settler_rates(rates, blocks, j, (Q_aq[j], u.Q_org), V_cells)
        return rates

    def _store_settler_rates(self, rates, blocks, j: int, flows, volumes) -> None:
        """Put the derivatives of stage j's settler cells into `rates` (blocks 1 onward).

        Each cell is fed by the block before it: the first cell by the mixer, the outlet cell by
        the last inner cell. flows: (Q_aq through the stage, Q_org), L/h. volumes: one cell's
        (V_aq, V_org), L.

        Why five parameters: where to write, the block layout, which stage, and that stage's flows and volumes.
        """
        for k in range(1, len(blocks)):
            _store(rates[k], j, self._cell(_stage(blocks[k], j), _stage(blocks[k - 1], j), flows, volumes))

    @staticmethod
    def _stack_rates(rates, dz):
        """All the derivatives as one column, in the state's order: the outlet cell and dz first
        (the model's layout), then the mixer, then the inner cells."""
        outlet, inner = rates[-1], rates[:-1]
        parts = [*outlet[0], *outlet[1], *outlet[2], *dz]
        for r in inner:
            parts += [*r[0], *r[1], *r[2]]
        return ca.vertcat(*parts)

    def _cell_volumes(self) -> tuple[float, float]:
        """Volume of one settler cell per phase, L: (aqueous, organic).

        The settler is split between the phases at the interface setpoint (aqueous below, organic
        above), and each phase's share is split evenly over the cells.
        """
        geo = self.p.raw["stage_geometry"]
        share_aq = geo["interface_setpoint_fraction"]
        V_set = geo["settler_volume_L"]
        return V_set * share_aq / self.cells, V_set * (1 - share_aq) / self.cells

    def _mixer(self, u: Inputs, j: int, mixer_j, feed: _MixerFeed):
        """The mixer of stage j: inflows from the neighbours' outlets, transfer, base dose.

        The same balances as the model's stage, but over the mixer only: hold-up V_mix split by the
        aqueous share phi_aq of the flow through it.

        u: the inputs. j: stage index. mixer_j: the mixer's (c_aq (n,), c_org (n,), h), mol/L.
        feed: what comes in, and the aqueous flow.
        Returns (d c_aq, d c_org, d h), mol/L/h.
        """
        p = self.p
        V_mix = p.raw["stage_geometry"]["mixer_volume_L"]
        c_aq, c_org, h = mixer_j
        Q_aq_j, Q_org = feed.Q_aq, u.Q_org
        phi_aq = Q_aq_j / (Q_aq_j + Q_org)                        # aqueous share of the mixer
        r = self._transfer(u, j, mixer_j, phi_aq)                 # mol/h, aqueous -> organic
        d_aq = (feed.aq_mol_in - Q_aq_j * c_aq - r) / (V_mix * phi_aq)
        d_org = (feed.org_mol_in - Q_org * c_org + r) / (V_mix * (1 - phi_aq))
        # acid: 3 H+ released per metal ion extracted, minus the base the organic spends here
        d_h = ((feed.acid_in - Q_aq_j * h + p.ligands_per_metal * ca.sum1(r) - feed.base_dose)
               / (V_mix * phi_aq))
        return d_aq, d_org, d_h

    @staticmethod
    def _cell(cell_j, upstream_j, flows, volumes):
        """One settler cell of stage j: each phase flows through from the cell before it.

        No transfer here; each phase is a stirred tank: V * dc/dt = Q * (c upstream - c here).

        cell_j, upstream_j: (c_aq (n,), c_org (n,), h) of this cell and of the one before it, mol/L.
        flows: (Q_aq through the stage, Q_org), L/h. volumes: (V_aq, V_org) of one cell, L.
        Returns (d c_aq, d c_org, d h), mol/L/h.
        """
        c_aq, c_org, h = cell_j
        u_aq, u_org, u_h = upstream_j
        Q_aq_j, Q_org = flows
        V_aq, V_org = volumes
        d_aq = Q_aq_j * (u_aq - c_aq) / V_aq
        d_org = Q_org * (u_org - c_org) / V_org
        d_h = Q_aq_j * (u_h - h) / V_aq
        return d_aq, d_org, d_h


class EntrainmentCircuit(Circuit):
    """Each phase leaves a settler carrying droplets of the other, more of them as the interface
    nears that phase's weir (aqueous in the organic overflow when the interface is high, organic in
    the aqueous underflow when it is low). The droplets join their own phase in the next stage and
    come back with that phase's flow, so the net effect is back-mixing between neighbouring stages:
    metal pushed back the wrong way. The model has none; it takes the settler's separation as
    perfect at any interface height.

    Sizes (chosen): a base of 500 ppm by volume (published normal figures run from about 100 ppm
    to the low thousands), rising to 2 % over a few centimetres past `onset_m` beyond the
    controller's window. Steady state differs from the model's by the base entrainment; the rise
    is what the controller has to keep away from.

    base:    entrained volume fraction at any interface height (5e-4 = 500 ppm)
    peak:    extra fraction the rise adds, far past the window (0.02 = 2 %)
    onset_m: m past the window edge where the rise is halfway
    width_m: m, how sharp the rise is (it goes from 27 % to 73 % of `peak` over 2 * width_m)
    The other arguments are Circuit's.

    Judgement call: entrainment here depends on the interface height only. In a plant it also
    rises with throughput per settler area, with crud at the interface, and as the phases get
    harder to separate (temperature, high loading). At the ndpr_plant flows the settlers run
    well above usual design loading (see Circuit._holdups), so a real settler of this size would
    likely sit above 500 ppm all the time. Entrainment samples from a pilot settler at design
    flow would settle it.
    """

    def __init__(self, *args, base: float = 5e-4, peak: float = 0.02, onset_m: float = 0.025,
                 width_m: float = 0.01, **kwargs):
        """Store the entrainment sizes (see the class docstring), then build the circuit."""
        self.base, self.peak, self.onset_m, self.width_m = base, peak, onset_m, width_m
        super().__init__(*args, **kwargs)

    def entrained_fractions(self, z):
        """(aqueous in organic, organic in aqueous), volume fractions, at interface height z (m).

        Aqueous in the organic overflow rises as the interface goes up past the top of the window;
        organic in the aqueous underflow rises as it goes down past the bottom:

            eps_ao(z) = base + rise(z - z_hi - onset)
            eps_oa(z) = base + rise(z_lo - onset - z)

        z_lo and z_hi are the window edges, m (interface_window_fraction times the settler height).
        Works on numbers and CasADi symbols.
        """
        geo = self.p.raw["stage_geometry"]
        lo, hi = (f * geo["settler_height_m"] for f in geo["interface_window_fraction"])
        return self.base + self._rise(z - hi - self.onset_m), self.base + self._rise(lo - self.onset_m - z)

    def _rise(self, d):
        """The smooth step on top of the base: peak / (1 + exp(-d / width)), with d in m.

        Half the peak at d = 0, close to zero well below it, close to the peak well above.
        """
        return self.peak / (1 + ca.exp(-d / self.width_m))

    def _carry_over(self, u: Inputs, Q_aq, state, inflows):
        """Add the entrainment exchange between neighbouring stages to the inflows, in place.

        For a pair (j, k) with the organic going j -> k and the aqueous coming back k -> j:
          - the organic leaving j carries aqueous droplets at the flow q_a = eps_ao(z_j) * Q_org.
            They go to k and come back to j with the aqueous, so it's an exchange of aqueous
            between j and k at q_a. Metal and acid both move.
          - the aqueous leaving k carries organic droplets at q_o = eps_oa(z_k) * Q_aq,k. They go to
            j and come back to k with the organic: an exchange of organic at q_o.
        An exchange at flow q adds q * (c over there - c here) to each side, mol/h.

        u: the inputs. Q_aq: aqueous flow out of each stage, L/h.
        state: (c_aq, c_org, h, z) per stage. inflows: (aq_in, acid_in, org_in), mol/h, changed here.
        """
        c_aq, c_org, h, z = state
        aq_in, acid_in, org_in = inflows
        for j, k in self._neighbours():                      # organic j -> k, aqueous k -> j
            aq_in_org, _ = self.entrained_fractions(z[j])
            _, org_in_aq = self.entrained_fractions(z[k])
            q_a = aq_in_org * u.Q_org                        # aqueous carried j -> k, back with the aqueous
            q_o = org_in_aq * Q_aq[k]                        # organic carried k -> j, back with the organic
            aq_in[j] += q_a * (c_aq[k] - c_aq[j])
            aq_in[k] += q_a * (c_aq[j] - c_aq[k])
            acid_in[j] += q_a * (h[k] - h[j])
            acid_in[k] += q_a * (h[j] - h[k])
            org_in[k] += q_o * (c_org[j] - c_org[k])
            org_in[j] += q_o * (c_org[k] - c_org[j])

    def _neighbours(self) -> list:
        """Stage pairs (j, k) where the organic goes j -> k and the aqueous comes back k -> j.

        Pairs where only one phase flows between the stages are left out: at the boundary between
        the scrub and the strip the aqueous paths don't connect, so there's no exchange there.

        Judgement call: leaving them out drops the one-way paths a plant worries about most.
        Scrub aqueous carried in the loaded organic into the first strip stage puts La and Ce
        straight into the product. Strip acid carried with the recycled organic goes to stage 0.
        Organic carried out with the raffinate and the product is lost. As one-way flows these
        would make this plant harder on purity, and the first one most of all. Adding them is a
        few lines here; a product assay against the scrub liquor's La+Ce would show how much it
        matters on a real plant.
        """
        org, aq = self._stage_links("org"), self._stage_links("aq")
        return sorted((j, k) for j, k in org if (k, j) in aq)

    def _stage_links(self, phase: str) -> set:
        """(from, to) for every `phase` stream that runs from one stage to another.

        Streams that come in from outside or leave the circuit are left out.
        """
        return {(e.src, e.dst) for e in self.cmap.edges if e.phase == phase and _between_stages(e)}


# ==============================================================================================
# A plant whose chemistry drifts under the controller (FINDINGS_why_hard.md, phases 3 and 4)
# ==============================================================================================
DRIFT_SPLIT_AMP = 0.03               # chosen: La+Ce against the rest, log amplitude (see DriftingCircuit)
ACID_FLOOR = 1e-3                    # mol/L, the base law's floor (parameters.toml acid_floor_mol_per_L)
HIDDEN_RELAX_H = 1e6                 # h, the hidden drift states' slow pull to zero (see DriftingCircuit)


class DriftingCircuit(Circuit):
    """The model's equations, with every distribution coefficient multiplied by a slowly drifting
    factor the controller doesn't know about: the plant's chemistry moving over days.

    What drifts, and why (all sizes chosen, to be checked against a real plant):
      common   all elements together, +-10 % (log amplitude 0.10). The extractant's strength
               (degradation, make-up additions: D goes as the cube of the free extractant, so 3 %
               of extractant is ~9 % of D) and the temperature (a few degrees).
      split    Pr+Nd against La+Ce, +-3 % on La+Ce: the separation factor itself wandering
               (composition along the cascade, temperature, impurities). Small, but the purity
               goes as this to the power of the stage count.
    Each is a sum of three sine waves with periods of 1 to 7 days and phases drawn from `seed`:
    smooth, slow, and the same every run with the same seed.

    How: n hidden states after the model's layout hold the log-multipliers, one per element. They
    relax toward zero with a 1e6 h time constant: nothing over a run, but enough that a steady-state
    solve pins them at zero instead of treating them as free (with a zero derivative the solver moved
    them anywhere, 1e10 and more, during commissioning). The closed loop calls hidden_at(x, t) once per control step, which writes the
    multipliers for time t into the state. observable(x) is the model's part, so instruments and
    controllers never see them.
    """

    def __init__(self, *args, seed: int = 1, common_amp: float = 0.10, split_amp: float | None = None,
                 periods_h: tuple = (24.0, 72.0, 168.0), trend_per_week: float = 0.0, **kwargs):
        """Draw the drift's phases and weights, then build the circuit.

        seed: picks the phases and weights of the sine waves (its own generator, seed + 2000).
        common_amp: log amplitude of the drift all elements share (0.10 = about +-10 % on D).
        split_amp: log amplitude of La+Ce against the rest; None = DRIFT_SPLIT_AMP.
        periods_h: periods of the sine waves, h.
        trend_per_week: a steady slide of La+Ce against the rest, log units per week (0 = none).
        The other arguments are Circuit's.

        Why five parameters: the drift's random seed and four sizes from the run file.
        """
        rng = np.random.default_rng(seed + 2000)
        self._phases = rng.uniform(0.0, 2 * np.pi, size=(2, len(periods_h)))
        self._weights = rng.uniform(0.5, 1.0, size=(2, len(periods_h)))
        self._weights /= self._weights.sum(axis=1, keepdims=True)
        self.common_amp, self.periods_h = common_amp, tuple(periods_h)
        self.split_amp = DRIFT_SPLIT_AMP if split_amp is None else split_amp
        # a steady slide of La+Ce against the rest, log units per week (0: none): a separation factor
        # that keeps falling, as with an extractant that degrades between make-ups
        self.trend_per_week = trend_per_week
        super().__init__(*args, **kwargs)

    @property
    def nx(self) -> int:
        """The model's layout, then n log-multipliers on D."""
        return self.nx_model + self.n

    def observable(self, x) -> np.ndarray:
        """The model's part of the state; the drift stays hidden from instruments."""
        return np.asarray(x, float)[: self.nx_model]

    def initial_state(self, u: Inputs) -> np.ndarray:
        """The model's initial state, with no drift yet."""
        return np.concatenate([super().initial_state(u), np.zeros(self.n)])

    def _read_hidden(self, x):
        """Keep the log-multiplier symbols, which sit after the model's layout in x."""
        self._logmult = x[self.nx_model: self.nx_model + self.n]

    def _hidden_rates(self) -> list:
        """The slow pull of each log-multiplier toward zero (see the class docstring)."""
        return [-self._logmult[i] / HIDDEN_RELAX_H for i in range(self.n)]

    def _distribution_multiplier(self, j=None, mixer_j=None):
        """The drift's factor on every element's D, the same in every stage."""
        return ca.exp(self._logmult)

    def log_multipliers(self, t_h: float) -> np.ndarray:
        """The drift at time t (h): log of the factor on each element's D."""
        waves = [np.sum(w * np.sin(2 * np.pi * t_h / np.array(self.periods_h) + ph))
                 for w, ph in zip(self._weights, self._phases)]
        common = self.common_amp * waves[0]
        out = np.full(self.n, common)
        names = list(self.p.names)
        if "LaCe" in names:                                  # the split: La+Ce against the rest
            out[names.index("LaCe")] += self.split_amp * waves[1] + self.trend_per_week * t_h / 168.0
        return out

    def hidden_at(self, x: np.ndarray, t_h: float) -> np.ndarray:
        """x with the drift for time t written into its hidden states (t = 0 gives no drift)."""
        x = np.asarray(x, float).copy()
        x[self.nx_model:] = self.log_multipliers(t_h) - self.log_multipliers(0.0)
        return x


# ==============================================================================================
# A plant built closer to the book (Qi, Hydrometallurgy of Rare Earths, 2018, ch. 2-5)
# ==============================================================================================
class BookCircuit(DriftingCircuit):
    """The drifting perturbed plant, with four things the controller's model doesn't have, each
    sized from the book; the figure, table and page behind each one is given with it below.

    1. The La+Ce/Pr separation factor is weaker where one side is a trace. Industrial Pr/Ce,
       measured stage by stage, is 2.0-2.2 in the middle of a cascade and 1.6-1.74 toward the ends
       (Fig. 2.56, p. 254). Here the La+Ce extraction constant is multiplied by
           m_trace(x) = (beta_model / beta_mid) / (1 - trace_drop * (1 - 4 x (1 - x)))
       where x is the (Pr+Nd) share of the stage's aqueous metal: beta_mid (2.1) at x = 0.5, about
       20 % lower when x goes to 0 or 1.
    2. The scrub section separates a little worse than the extraction section (the theory carries
       beta' < beta; the worked example uses 1.70 and 1.50, p. 502). Here the La+Ce constant in scrub
       stages is divided by scrub_beta_factor (0.92, about half of that example's gap).
    3. A weaker acid dependence: with saponified P507 the slope of lg D on lg [H+] is about -2.4
       instead of -3 (p. 246). The model's K were fitted with n = 3, so the plant keeps them and
           multiplies D by (h / h_ref) ** (n_model - n_plant)
       so both agree at h_ref = 0.05 mol/L, a typical extraction-section acidity.
    4. Slower transfer in acid. P507 reaches equilibrium in about 5 min at pH 4 and in 20-60 min
       at 1.5 mol/L H+ (p. 258); the rate goes as 1/[H+] (p. 224). Here, in the extraction and scrub
       stages, the transfer rate is multiplied by
           r(h) = (1 + h_k_ref / h_half) / (1 + h / h_half),   h_half = 0.2 mol/L
       which is 1 at h_k_ref = 0.02 mol/L (where the model's rate was set, the Idaho rig's
       extraction) and about 1/8 at 1.5 mol/L. The strip stages keep the model's rate (the book
       gives no strip kinetics for light rare earths).
    Plus the slow chemistry drift of DriftingCircuit. The perturbed plant's own acid-exponent shift
    is left out here (registry builds this plant with dn_acid = 0), since item 3 replaces it.
    """

    def __init__(self, *args, beta_mid: float = 2.1, trace_drop: float = 0.20, scrub_beta_factor: float = 0.92,
                 n_plant: float = 2.5, h_ref: float = 0.05, h_half: float = 0.2, h_k_ref: float = 0.02, **kwargs):
        """Store the sizes of the four items in the class docstring, then build the drifting plant.

        beta_mid: La+Ce/Pr separation factor mid-cascade (item 1).
        trace_drop: how much weaker that factor gets where one side is a trace, 0..1 (item 1).
        scrub_beta_factor: divides the La+Ce constant in scrub stages (item 2).
        n_plant: the plant's acid exponent (item 3).
        h_ref: acid where the plant's D and the model's agree, mol/L (item 3).
        h_half: acid where the transfer rate is halved from its low-acid value, mol/L (item 4).
        h_k_ref: acid where the plant's rate equals the model's, mol/L (item 4).
        The other arguments are DriftingCircuit's and Circuit's.

        Why seven parameters: the book's relations, each with its own number from the run file.
        """
        self.beta_mid, self.trace_drop, self.scrub_beta_factor = beta_mid, trace_drop, scrub_beta_factor
        self.n_plant, self.h_ref, self.h_half, self.h_k_ref = n_plant, h_ref, h_half, h_k_ref
        super().__init__(*args, **kwargs)

    def _elements(self):
        """Positions of La+Ce, Pr and Nd in the component arrays."""
        names = list(self.p.names)
        return names.index("LaCe"), names.index("Pr"), names.index("Nd")

    def _distribution_multiplier(self, j=None, mixer_j=None):
        """The drift's factor on D, times items 1-3: the La+Ce factor and the acid factor.

        j: stage index. mixer_j: (c_aq, c_org, h) in the stage's mixer. Without mixer_j only the
        drift applies. Returns a column with one factor per element.
        """
        base = super()._distribution_multiplier(j, mixer_j)            # the drift
        if mixer_j is None:
            return base
        c_aq, _, h = mixer_j
        i_la = self._elements()[0]
        m_la = self._lace_factor(j, c_aq)
        acid = self._acid_factor(h)
        per_element = [acid * (m_la if i == i_la else 1) for i in range(self.n)]
        return base * ca.vertcat(*per_element)

    def _lace_factor(self, j: int, c_aq):
        """Items 1 and 2: the factor on the La+Ce constant in stage j, from the aqueous metal c_aq.

        Weaker separation where Pr+Nd or La+Ce is a trace, and weaker again in the scrub.
        """
        i_la, i_pr, i_nd = self._elements()
        x = (c_aq[i_pr] + c_aq[i_nd]) / (ca.sum1(c_aq) + 1e-9)          # (Pr+Nd) share of the metal
        beta_model = float(self.p.K[i_pr] / self.p.K[i_la])
        m_la = (beta_model / self.beta_mid) / (1 - self.trace_drop * (1 - 4 * x * (1 - x)))
        if self.cmap.section(j) == "scrub":
            m_la = m_la / self.scrub_beta_factor
        return m_la

    def _acid_factor(self, h):
        """Item 3: the factor that turns the model's acid exponent into the plant's, 1 at h_ref."""
        # above the base law's own acid floor (1e-3, chemistry.distribution): below it D flattens there,
        # and a factor that kept shrinking with h would make D fall as the acid runs out (a D that
        # rises then falls has several steady states; the first version did that)
        return (ca.fmax(h, ACID_FLOOR) / self.h_ref) ** (float(self.p.n_acid) - self.n_plant)

    def _rate_multiplier(self, j=None, mixer_j=None):
        """Item 4: slower transfer in acid, in the extraction and scrub stages only.

        Returns 1 for the strip stages and when the mixer contents aren't given.
        """
        if mixer_j is None or j >= self.cmap.n_ext + self.cmap.n_scrub:
            return 1
        h = ca.fmax(mixer_j[2], 1e-6)
        return (1 + self.h_k_ref / self.h_half) / (1 + h / self.h_half)
