"""The first MPC tried: a nonlinear controller built with do-mpc. Kept as a record only.

This file isn't used. Nothing imports it, do-mpc isn't installed in the project's
environments, and its interface is older than loop.Controller: its step() takes the state and
the feed directly instead of an Observation, and it brings its own run_closed_loop. It's kept
because the numbers it gave are the reason the controller in use is built the way it is.

What it did. One controller held purity, recovery, reagent and every interface in a single
nonlinear problem. do-mpc took the circuit's own CasADi equations (decision 1), discretized
them by orthogonal collocation over a 12-step horizon, and handed the whole thing to IPOPT
every step. Purity under spec, recovery under target and interfaces outside their window were
soft constraints with large penalties; reagent use and handle moves were in the objective.
This version (v1) sees the full state and the current feed composition (an analyser on the
feed line). The plant it drives is a separate call to simulate(), with the disturbances and
any parameter mismatch applied there (decision 2).

Why it was dropped. About 1.4 s per IPOPT iteration and no convergence in 200 iterations on
the 18-stage circuit, so 90 s per step at best, out of the question for 64 stages. The
controller in use (mpc.py) splits the job in two (ARCHITECTURE.md, decision 16): a nonlinear
steady-state optimizer sets the operating point once, and a linear MPC solves a QP each step.

The code below is left as it was when it was set aside; only comments and docstrings have
been added.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import casadi as ca
import do_mpc
import numpy as np

from .circuit import Circuit, Inputs
from .metrics import heavy_mask
from .simulate import Trajectory, simulate

# The names of every field of circuit.Inputs, in order: flows, feed, acids, jacklegs ...
INPUT_FIELDS = [f.name for f in fields(Inputs)]


# ==============================================================================================
# Settings
# ==============================================================================================
# Flows in L/h, acidity in mol/L, saponification as a fraction, jackleg heights in m, time in
# h. The bounds and rate limits are the ones later reused by the RL environment (rl_env.py).

@dataclass
class MPCSettings:
    """The do-mpc controller's settings: handles, limits, horizon, weights and solver options."""
    handles: tuple = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
    jacklegs: bool = True                       # every settler's jackleg is a handle too
    bounds: dict = field(default_factory=lambda: {
        "Q_org": (600.0, 4000.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6), "H_jackleg": (0.30, 0.48)})
    rate_per_step: dict = field(default_factory=lambda: {       # how far a handle may move per step
        "Q_org": 150.0, "Q_scrub": 40.0, "h_scrub": 0.1, "saponified_fraction": 0.02, "H_jackleg": 0.01})
    horizon: int = 12                           # steps the controller looks ahead
    t_step: float = 0.5                         # h
    w_purity: float = 1e4                       # penalty on purity below spec (soft constraint)
    w_recovery: float = 1e3                     # penalty on recovery below target (soft constraint)
    w_interface: float = 1e3                    # penalty per metre outside the window (soft constraint)
    w_reagent: float = 1e-4                     # on reagent, mol/h
    w_move: dict = field(default_factory=lambda: {                # move suppression, per handle
        "Q_org": 1e-5, "Q_scrub": 1e-4, "h_scrub": 1.0, "saponified_fraction": 10.0, "H_jackleg": 10.0})
    strip_tied_to_scrub: float | None = 300.0   # net product flow; None = strip flow fixed
    ipopt_max_iter: int = 200
    scale: bool = True                          # tell the solver the typical size of each variable
    symvar: str = "MX"                          # do-mpc symbol type; MX keeps the NLP graph small for big circuits
    hessian: str = "exact"                      # or "limited-memory"
    max_cpu_s: float = 120.0                    # IPOPT gives up on a step after this; the last iterate is used


# ==============================================================================================
# The controller
# ==============================================================================================

class CircuitMPC:
    """The nonlinear MPC on the full circuit model, built with do-mpc.

    Args:
        circuit: the circuit model the controller plans on.
        u_nominal: the inputs at the operating point; handles not controlled stay here.
        cut: the cut the circuit makes, such as "LaCe" (which components are the product).
        spec: the product purity floor (fraction).
        recovery_target: the recovery floor (fraction).
        settings: MPCSettings; the defaults if None.

    The constructor is long (about 85 lines) because do-mpc is set up in one go: the model,
    then the optimizer. It was never split up, since the file was set aside first.
    """

    def __init__(self, circuit: Circuit, u_nominal: Inputs, cut: str, spec: float,
                 recovery_target: float, settings: MPCSettings | None = None):
        """Build the do-mpc model and optimizer once; step() then solves it each control step.

        In order: the do-mpc model (the state, the handles, and the feed as a time-varying
        parameter), the full input vector with those symbols written in, the circuit's equations
        and the outputs (purity, recovery, reagent, interfaces), then the optimizer (collocation
        and IPOPT options, the reagent objective, move penalties, bounds and scaling), the soft
        constraints, and the feed held over the horizon. Arguments are in the class docstring.
        """
        self.circuit, self.u_nominal, self.cut = circuit, u_nominal, cut
        self.s = s = settings or MPCSettings()
        p, n, N = circuit.p, circuit.n, circuit.N        # parameters, components, stages
        geo = p.raw["stage_geometry"]
        H = geo["settler_height_m"]
        z_lo, z_hi = (f * H for f in geo["interface_window_fraction"])     # interface window, m

        # ---- the do-mpc model: state, handles, and the feed as a time-varying parameter ----
        model = do_mpc.model.Model("continuous", s.symvar)
        x = model.set_variable("_x", "x", (circuit.nx, 1))
        uh = {name: model.set_variable("_u", name, (1, 1)) for name in s.handles}
        if s.jacklegs:
            uh["H_jackleg"] = model.set_variable("_u", "H_jackleg", (N, 1))
        c_feed = model.set_variable("_tvp", "c_feed", (n, 1))
        h_feed = model.set_variable("_tvp", "h_feed", (1, 1))

        # The full input vector: the operating point, with the handles and the feed replaced
        # by the symbols above. The strip flow is tied to the scrub flow when set to.
        vals = {k: getattr(u_nominal, k) for k in INPUT_FIELDS}
        if vals["k_factor"] is None:
            vals["k_factor"] = np.ones(N)
        vals.update(uh)
        vals["c_feed"], vals["h_feed"] = c_feed, h_feed
        if s.strip_tied_to_scrub is not None:
            vals["Q_strip"] = s.strip_tied_to_scrub + vals["Q_scrub"]
        u_full = Inputs(**vals).to_vector()

        # The circuit's own equations, and the outputs the controller cares about, as symbols.
        model.set_rhs("x", circuit.casadi_rhs()(x, u_full))
        out = circuit._out(x=x, u=u_full)
        heavy = [i for i, m in enumerate(heavy_mask(p, cut)) if m]          # the product's components
        purity = ca.sum1(out["c_prod"][heavy]) / ca.sum1(out["c_prod"])
        recovery = out["Q_prod"] * ca.sum1(out["c_prod"][heavy]) / (vals["Q_feed"] * ca.sum1(c_feed[heavy]))
        reagent = (vals["saponified_fraction"] * p.raw["extractant"]["concentration_monomer"] * vals["Q_org"]
                   + vals["Q_strip"] * vals["h_strip"] + vals["Q_scrub"] * vals["h_scrub"])   # mol/h
        z = x[2 * N * n + N: 2 * N * n + 2 * N]                              # the interfaces, m
        model.set_expression("purity", purity)
        model.set_expression("recovery", recovery)
        model.set_expression("reagent", reagent)
        model.set_expression("z", z)
        model.setup()
        # after setup, do-mpc wants its own registered symbols, not the ones above
        x = model.x["x"]
        purity, recovery, reagent, z = (model.aux[k] for k in ("purity", "recovery", "reagent", "z"))

        # ---- the optimizer: collocation, IPOPT options, objective, bounds, scaling ----
        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(n_horizon=s.horizon, t_step=s.t_step, n_robust=0, store_full_solution=False,
                      collocation_deg=2, collocation_ni=1,
                      nlpsol_opts={"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": s.ipopt_max_iter,
                                   "ipopt.tol": 1e-6, "ipopt.acceptable_tol": 1e-4, "ipopt.mu_strategy": "adaptive",
                                   "ipopt.max_cpu_time": s.max_cpu_s, "ipopt.hessian_approximation": s.hessian})
        # Reagent over the horizon is the running cost; there's no terminal cost (zero times x[0]).
        mpc.set_objective(mterm=ca.DM(0) * x[0], lterm=s.w_reagent * reagent)
        rterm = {name: s.w_move[name] for name in uh}                       # penalty on each move
        mpc.set_rterm(**rterm)
        for name, var in uh.items():
            lo, hi = s.bounds[name]
            mpc.bounds["lower", "_u", name] = lo
            mpc.bounds["upper", "_u", name] = hi
        # No negative concentrations; acid at least 1e-4 mol/L; interfaces at or above 0.
        mpc.bounds["lower", "_x", "x"] = np.concatenate([np.zeros(2 * N * n), np.full(N, 1e-4), np.zeros(N)])
        if s.scale:
            mpc.scaling["_x", "x"] = np.concatenate([np.full(2 * N * n, 0.1), np.full(N, 0.01), np.full(N, 0.25)])
            typical = {"Q_org": 1000.0, "Q_scrub": 300.0, "h_scrub": 1.0, "saponified_fraction": 0.5, "H_jackleg": 0.4}
            for name in uh:
                mpc.scaling["_u", name] = typical[name]

        # ---- the floors and the interface window, as soft constraints ----
        mpc.set_nl_cons("purity_below_spec", spec - purity, ub=0, soft_constraint=True,
                        penalty_term_cons=s.w_purity, maximum_violation=1.0)
        mpc.set_nl_cons("recovery_below_target", recovery_target - recovery, ub=0, soft_constraint=True,
                        penalty_term_cons=s.w_recovery, maximum_violation=1.0)
        mpc.set_nl_cons("interface_low", z_lo - z, ub=0, soft_constraint=True,
                        penalty_term_cons=s.w_interface, maximum_violation=H)
        mpc.set_nl_cons("interface_high", z - z_hi, ub=0, soft_constraint=True,
                        penalty_term_cons=s.w_interface, maximum_violation=H)

        # ---- the feed over the horizon: the latest measurement, held constant ----
        self._tvp = mpc.get_tvp_template()
        self._feed_now = (np.asarray(u_nominal.c_feed, float), float(u_nominal.h_feed))

        def tvp_fun(t_now):
            """The feed over the horizon for do-mpc: the last measured feed (set by step()) at
            every horizon step. t_now isn't used; do-mpc passes it."""
            c, h = self._feed_now
            for k in range(s.horizon + 1):
                self._tvp["_tvp", k, "c_feed"] = c.reshape(-1, 1)
                self._tvp["_tvp", k, "h_feed"] = h
            return self._tvp
        mpc.set_tvp_fun(tvp_fun)
        mpc.setup()
        self.mpc, self.model = mpc, model
        self._u_names = list(uh)
        self._started = False

    # ---- one controller step ----------------------------------------------
    def step(self, x: np.ndarray, measured_feed: np.ndarray, measured_h_feed: float) -> dict:
        """Solve the problem from state x and return the handle values to apply until the next step.

        Args:
            x: the circuit's full state.
            measured_feed: the feed composition, mol/L per component.
            measured_h_feed: the feed acidity, mol/L.

        Returns:
            {handle name: value}; H_jackleg (if a handle) is an array, one per settler.
        """
        self._feed_now = (np.asarray(measured_feed, float), float(measured_h_feed))
        # The first step starts IPOPT from the operating point's handles.
        if not self._started:
            self.mpc.x0 = x.reshape(-1, 1)
            u0 = self.mpc.u0
            for name in self._u_names:
                u0[name] = np.atleast_1d(np.asarray(getattr(self.u_nominal, name), float)).reshape(-1, 1)
            self.mpc.u0 = u0
            self.mpc.set_initial_guess()
            self._started = True
        u = np.asarray(self.mpc.make_step(x.reshape(-1, 1)), float).ravel()
        # Unpack the flat solution vector into named handles.
        out, i = {}, 0
        for name in self._u_names:
            k = self.circuit.N if name == "H_jackleg" else 1
            out[name] = u[i:i + k].ravel() if k > 1 else float(u[i])
            i += k
        return out

    def apply(self, handles: dict) -> Inputs:
        """Nominal inputs with the controller's handles written in."""
        u = self.u_nominal.replace(**handles)
        if self.s.strip_tied_to_scrub is not None:
            u = u.replace(Q_strip=self.s.strip_tied_to_scrub + u.Q_scrub)
        return u


# ==============================================================================================
# The closed loop it was run in
# ==============================================================================================
# The forerunner of loop.ClosedLoop. It takes seven arguments and branches more than the
# guidelines like, and it was left that way: nothing calls it now.

def run_closed_loop(plant: Circuit, x0: np.ndarray, controller: CircuitMPC, scenario, t_end: float,
                    t_step: float | None = None, verbose: bool = True) -> tuple[Trajectory, list[dict]]:
    """Drive `plant` with the controller. The scenario's disturbances act on the plant inputs
    after the controller's handles are written in (a flowmeter bias scales what was commanded, a
    hydraulic drift adds to the jackleg that was set). The controller sees the plant state and
    the plant's feed, not the bias and not the drift.

    Args:
        plant: the circuit being driven.
        x0: its starting state.
        controller: the CircuitMPC above.
        scenario: a disturbances.Scenario, or None for no upsets.
        t_end: how long to run, h.
        t_step: the control interval, h; the controller's own if None.
        verbose: print the time every 10 steps.

    Returns:
        (the trajectory, the handles commanded at each step with the step's start time).
    """
    t_step = t_step or controller.s.t_step
    u_of_t = scenario.u_of_t(controller.u_nominal) if scenario is not None else (lambda t: controller.u_nominal)
    ts, xs, us, handles_log = [0.0], [x0], [], []
    x, t = np.asarray(x0, float), 0.0
    while t < t_end - 1e-9:
        feed_now = u_of_t(t)
        handles = controller.step(x, feed_now.c_feed, feed_now.h_feed)
        u_cmd = controller.apply(handles)
        u_plant = scenario.u_of_t(u_cmd)(t) if scenario is not None else u_cmd
        uv = u_plant.numpy()
        seg = simulate(plant, x, lambda _t: uv, t_step, dt_out=t_step)
        x = seg.x[-1]
        t += t_step
        ts.append(t); xs.append(x); us.append(uv); handles_log.append(dict(t=t - t_step, **handles))
        if verbose and int(round(t / t_step)) % 10 == 0:
            print(f"  t = {t:5.1f} h", flush=True)
    us.append(us[-1])
    return Trajectory(np.array(ts), np.array(xs), np.array(us)), handles_log
