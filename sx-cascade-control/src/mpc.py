"""Linear MPC, re-linearized as it goes, with the nonlinear model behind it.

Two layers, the way plant MPC is built (ARCHITECTURE.md, decision 16):

  1. The operating point (nominal handles) comes from the nonlinear
     steady-state optimizer (design.py, through the run files), then a trim
     under control on the plant itself (operating_point.py). Re-running the
     optimizer when the feed changes for good is the natural next layer; it
     isn't built.
  2. This controller linearizes the circuit's own CasADi equations on its own
     model state and the current handles (every step by default), takes
     an exact discretization, and solves a small QP: hold purity and recovery
     above their targets, keep every interface in its window, spend as little
     reagent as that allows, move the handles gently. Measured feed
     composition and the estimator's flow-offset estimate are feedforward.

Re-linearizing each step (successive linearization) is what lets one linear
QP follow the circuit through a feed-strength step, where a model fixed at
the operating point sent the moves the wrong way (decision 18). The QP is
assembled numerically and handed to OSQP through CasADi's numeric interface,
so a re-linearization costs a Jacobian, a matrix exponential and some matrix
products, not a symbolic rebuild.

The model is never the plant. The controller keeps its own model state,
run forward with the model under the inputs it applied (implicit Euler
with the model's own Jacobian, a fixed cost every step), and linearizes
there, on the model's own trajectory, where the linearization is good.
What the plant (or the estimator) reports enters as an output bias: the
difference between the purity, recovery and interfaces the model state
gives and the ones reported, filtered, added to every prediction. That is
the integral action, the way industrial MPC has done it since DMC: a
persistent gap between model and plant becomes a constant the controller
plans around. Linearizing the model at the plant's own state was tried
first and is wrong in kind: a few percent off the model's manifold the
model's internal rates are far from zero and its Jacobian there has
neither the slow inventory mode nor the steady-state gains of the real
circuit (ARCHITECTURE.md, 21).

One control step, in order (CircuitMPC.step):
  measure      the disturbance vector and the offsets read back from the plant
  bias         update the output bias: reported outputs minus the model state's
  linearize    on the model state (every step by default)
  predict      the outputs over the horizon with the handles held, plus the bias
  solve        the QP for the moves
  advance      the model state under what is now applied

The closed loop around it (plant, scenario, sensors) is in loop.py. The first
attempt was a nonlinear do-mpc controller (mpc_nonlinear.py); it needed about
90 s per step on an 18-stage circuit, so it stays as a record only.

What the controller works with
------------------------------
The product is NdPr as one product: the cut is "LaCe", so La and Ce go one way and
Pr and Nd the other. (The Pr | Nd cut is there for validation only.) The handles are
the organic flow, the scrub flow, the scrub acidity, the saponified fraction, and one
jackleg per settler. With a reflux scrub the strip flow is tied to the scrub flow:
Q_strip = 300 L/h + Q_scrub, so moving the scrub flow moves the strip flow too.

The outputs the controller holds, in this order, make the vector y:
  y = [purity, recovery, z_0, ..., z_{N-1}]
purity and recovery are fractions (0.995 is 99.5 %), and z_j is the interface height in
settler j, in metres. Time is in hours; one control step is t_step = 0.5 h.

The math, step by step (APPENDIX_B_control.md, B.2 and B.3)
-------------------------------------------------------------
Linearize at the model state x_c, the handles u_c and the measured disturbances d_c,
keeping the affine term (the model state isn't at rest, so dx/dt there isn't zero):
    dxi/dt = A xi + B du + E dd + f_c          xi = x - x_c,  du = u - u_c,  dd = d - d_c
    y      = y_c + C xi + Du du + Dd dd
Discretize exactly over t_step, for inputs held over the step, with one matrix
exponential of the augmented matrix [[A, B, E, f_c], [0, 0, 0, 0]]:
    xi[k+1] = Ad xi[k] + Bd du[k] + Ed dd + fd
Stack the outputs over the horizon (Hp = 24 steps) and they're affine in the moves:
    Y = Y0 + M dU      Y0 = the outputs if the handles are held, plus the bias
The QP picks the moves dU (Hu = 8 moves; one for each jackleg) to keep purity and
recovery above their floors and every interface in its window, each with a slack,
and to spend little reagent. Each move is at most its rate limit, each handle stays
inside its bounds. The handles the plan ends on must also hold the floors at steady
state, as far as the linear model knows (the steady-state floors, decision 21).

Units in the QP. Everything is scaled so the weights are of order 0.01 to 100:
purity slack in thousandths, recovery slack in hundredths, interface slack in cm,
a move as a fraction of its rate limit, reagent in kmol/h.

The QP's variables, in this order, make the vector w:
  w = [moves (n_du) | s_spec (Hp) | s_target (Hp) | s_rec (Hp) | s_lo (n_ic*N) | s_hi (n_ic*N)
       | s_ss_purity | s_ss_recovery]
  s_spec     purity below the spec, per horizon step, in thousandths
  s_target   purity below the purity target, per horizon step, in thousandths
  s_rec      recovery below the recovery target, per horizon step, in hundredths
  s_lo, s_hi interface below / above its window, at the checked steps, in cm
  s_ss_*     purity (thousandths) and recovery (hundredths) below their targets at steady state
n_ic is the number of horizon steps where the interfaces are checked (5 by default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import casadi as ca
import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from . import chemistry
from .circuit import Circuit, Inputs
# run_closed_loop isn't used here: callers import it from this module.
from .loop import Observation, run_closed_loop      # noqa: F401
from .metrics import heavy_mask
from .simulate import implicit_euler

# Why (two layers, decision 16): a steady-state optimizer above a dynamic MPC is how industrial MPC
# is laid out (Qin and Badgwell's survey). Judgement call: the optimizer runs once, offline, and is
# not re-run when the feed changes for good; the reagent term and the floors do the local economics.
# Re-running design.py on the new feed and comparing its reagent with where the MPC settles would
# show what an online optimizer would add.
# Why (a hand-built linear QP, decision 16): measured. A nonlinear do-mpc controller took about
# 1.4 s per IPOPT iteration and did not converge in 200 on the 18-stage circuit, so 90 s a step at
# best; this QP solves in under a second, and the studies needed thousands of solves.


# =====================================================================================================
# Settings
# =====================================================================================================
@dataclass
class MPCSettings:
    """Every choice the controller makes, in one place.

    The defaults are what every reported run uses. Bounds, rate limits and weights are chosen,
    not measured (ARCHITECTURE.md, decisions 19 and 21; APPENDIX_B_control.md, B.3).

    Units of the handles: flows in L/h, scrub acidity in mol/L, the saponified fraction as a
    fraction of the extractant (0 to 1), jackleg heights in m. Time in hours.

    Example:
        settings = MPCSettings(horizon=12, relinearize_every=0)   # shorter plan, one linearization
        ctl = CircuitMPC(circuit, x_ss, u_ss, rc.cut, spec, 0.995, rec, settings)
    """
    # ---- the handles ---------------------------------------------------------------------------
    handles: tuple = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
    jacklegs: bool = True                       # every settler's jackleg is a handle too
    # Lowest and highest value of each handle (chosen). Q_org's 4700 L/h is the organic pump's
    # maximum (chosen); the design uses 85 % of it, so the controller has room to add organic.
    # Why (the headroom): measured, decision 19. A design at the pump's ceiling left the controller
    # nothing to add on a +20 % feed. Judgement call (the other bounds): set wide, not from equipment.
    bounds: dict = field(default_factory=lambda: {
        "Q_org": (600.0, 4700.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6), "H_jackleg": (0.30, 0.48)})
    # Largest move of each handle in one step (chosen: gentle). The plant circuit's long-run
    # purity moves about 13 points per 100 L/h of organic; big steps on a model that's only
    # roughly right is how the first version lost the plant.
    # Why (that they are tight): decision 21; they went to a third of the first version's limits
    # after that version lost the plant. Judgement call (the numbers): no sweep was done. Too
    # tight slows the answer to a real feed change. A sweep at half and double on the feed switch
    # and the pump stop would settle it.
    rate_per_step: dict = field(default_factory=lambda: {
        "Q_org": 50.0, "Q_scrub": 10.0, "h_scrub": 0.05, "saponified_fraction": 0.01,
        "H_jackleg": 0.01})

    # ---- timing and horizons -------------------------------------------------------------------
    # Judgement call: half an hour is short next to the circuit's hours-long responses. The
    # quarter-hour analysers are only read on this grid. Not compared against 15 min or 1 h steps.
    t_step: float = 0.5                         # h, one control step
    # Judgement call: 12 h covers the purity response to a move but not the ~400 h inventory mode;
    # the steady-state floors cover the rest. Hp = 12 and 48 on the catalogue would show the cost.
    horizon: int = 24                           # prediction steps (Hp): 24 x 0.5 h = 12 h
    # Judgement call: 8 moves (4 h) keeps the QP small. With the rate limits, 8 moves is at most
    # 400 L/h of organic in one plan. Not compared against other values.
    control_horizon: int = 8                    # moves (Hu); the last one is held to the horizon end
    # Why: a settler's interface settles in minutes, well inside one 0.5 h step, so a second
    # jackleg move in the plan adds variables and changes nothing the plan can see.
    jackleg_moves: int = 1                      # jacklegs settle in minutes: one move per step is enough
    # Judgement call: interfaces are fast, so five points (0.5, 1.5, 3, 6 and 12 h) stand in for
    # all 24, which keeps the QP small. An interface could leave its window between checks
    # without the plan seeing it. A run checking every step would show whether that happens.
    interface_check_steps: tuple = (0, 2, 5, 11, 23)   # horizon steps at which interfaces are constrained
    # Why: measured, decision 18. Fixed at the operating point, the moves went the wrong way on a
    # +20 % feed. Successive linearization is a standard family (Rawlings, Mayne and Diehl).
    relinearize_every: int = 1                  # steps; 0 = linearize once at the operating point

    # ---- integral action and the long run ------------------------------------------------------
    # Why (an output bias): DMC's way of getting offset-free control (Cutler and Ramaker, 1980),
    # still the most common one in industrial MPC (Qin and Badgwell's survey); decision 21.
    # Judgement call (0.5): a filter to keep analyser noise out of the plan. It slows the bias by
    # about one step. No sweep; 0.2, 0.5 and 1.0 with a noisy sensor set would settle it.
    bias_filter: float = 0.5                    # share of each step's (reported - model) output difference
                                                # taken into the output bias (0 = off); decision 21
    # Why: measured, decision 21 (3). Without it the product rate read 160 % of the feed for hours
    # while the held-up metal drained, the 12 h horizon saw recovery holding, and the long run lost.
    steady_state_floors: bool = True            # the handles the plan ends on must also hold the purity and
                                                # recovery floors at steady state, as far as the linear model
                                                # knows (a reflux circuit's inventory settles over days; the
                                                # horizon alone lets the controller borrow from it)

    # ---- weights -------------------------------------------------------------------------------
    # Weights on scaled errors so the QP is well conditioned: purity in thousandths, recovery in
    # hundredths, interface slack in cm, a move as a fraction of its rate limit, reagent in kmol/h.
    # Targets are floors: being above them is never penalized.
    # Why (the scaling): every term in its natural unit is standard tuning practice, and it keeps
    # OSQP well conditioned. Judgement call (the values): picked by hand until the catalogue ran
    # clean; no sensitivity study. Each weight at half and double on the catalogue would settle it.
    w_purity: float = 1.0                       # slack below the purity target, per thousandth^2
    w_recovery: float = 1.0                     # slack below the recovery target, per hundredth^2
    # Judgement call: 100x the target's weight, so the spec gives way last. One thousandth under
    # spec costs as much as ten under the target.
    w_purity_floor: float = 100.0               # slack below the spec, per thousandth^2
    w_interface: float = 1.0                    # slack outside the window, per cm^2
    # An acidity floor in the extraction section (decision 33). Off by default (None). Saponifying
    # the organic spends acid; push it far enough and the extraction stages run out of acid. Below
    # about pH 3 the extraction law isn't valid (parameters.toml, acid_floor_mol_per_L), and in a
    # plant the rare earths start to come down as hydroxides. The design step keeps every stage at
    # or above 1e-3 mol/L; this gives the controller the same floor, checked at the interface
    # check steps, as a soft constraint with its slack in mmol/L. A plant reads it with a pH probe
    # on the raffinate and the stages next to the feed.
    acid_floor: float | None = None             # mol/L, on the aqueous acid of every extraction stage
    w_acid: float = 10.0                        # slack below the acid floor, per (mmol/L)^2
    # Fixed on Oct 13 (ARCHITECTURE.md, decision 26): until then the offset was divided by 100
    # instead of multiplied, so this term was 1e8 weaker than meant and did almost nothing.
    w_center: float = 0.02                      # pull of each interface toward its setpoint, per cm^2 (so the
                                                # jacklegs bring interfaces back after an upset, not just inside)
    # Judgement call: a small linear price on reagent so the controller saves where the floors
    # allow. It isn't tied to real prices, so the recovery floor is held at any reagent cost
    # (ARCHITECTURE.md, Open). Prices for NdPr, caustic and acid would settle it.
    w_reagent: float = 0.01                     # reagent, per kmol/h, linear
    w_move: dict = field(default_factory=lambda: {
        "Q_org": 1.0, "Q_scrub": 1.0, "h_scrub": 1.0, "saponified_fraction": 1.0, "H_jackleg": 1.0})
    # Judgement call: added when the reagent term and an active floor made the moves chatter.
    # The value was the first that stopped it.
    w_hold: float = 0.02                        # on accumulated moves over the horizon (damps chatter between
                                                # the reagent term and an active floor), per rate-limit^2

    # ---- the circuit and the solver ------------------------------------------------------------
    strip_tied_to_scrub: float | None = 300.0   # net product flow, L/h: Q_strip = this + Q_scrub;
                                                # None = strip flow fixed
    # Why (decision 30): OSQP retunes its step size on a schedule that by default depends on how long
    # its own setup took, so on a badly scaled problem the same QP could solve or be called
    # "infeasible" from one run to the next. A fixed interval makes it deterministic.
    osqp: dict = field(default_factory=lambda: {"verbose": False, "eps_abs": 1e-6, "eps_rel": 1e-6,
                                                "max_iter": 20000, "polish": True,
                                                "adaptive_rho_interval": 25})
    # Why (decision 30): every output constraint has a slack, so the QP always has a solution; when
    # OSQP still reports failure (seen on plants far off spec, where the slack costs run into the
    # millions) the same problem goes to qpOASES, an active-set solver, before the handles are held.
    fallback_solver: str | None = "qpoases"
    # Why (decision 30): ranked constraints. The purity spec comes first: a first QP finds the
    # smallest shortfall below spec the handles can reach over the horizon, and the full QP is then
    # not allowed a larger one. Without it, purity and recovery are both soft floors with fixed
    # weights, and a plant that can't make both had its purity traded away for recovery.
    ranked: bool = False                        # tried on Oct 14 (decision 30): hurt plants that could make spec
    ranked_tolerance: float = 0.05              # thousandths of purity the second QP may add to the first's
    # Exact penalty on the spec (decision 30): a straight-line cost per thousandth below spec, on top
    # of the squared one. With squared costs alone a small shortfall is nearly free, so purity could
    # be traded for recovery a little at a time; a straight-line cost larger than what a thousandth
    # of purity could ever buy elsewhere keeps the spec first in one QP (the usual way industrial
    # MPC softens a constraint without letting it be traded away). 0 = off.
    w_purity_floor_linear: float = 0.0          # per thousandth below spec, summed over the horizon


# What a reported purity, recovery and bias may be (chosen): an estimate built on a wrong model, or a
# feed that has stopped, can report a recovery of 2260 %; the QP must never plan on that (decision 23).
# Why (clipping at all): measured, decision 23; a stopped feed made an estimate say 2260 %.
# Judgement call (the numbers): 150 % and +-0.2 / +-0.5 were picked to be loose. A tighter bias
# limit would hide a real model error. Logging how often each clip is hit would show if they matter.
PURITY_RANGE, RECOVERY_RANGE = (0.0, 1.0), (0.0, 1.5)
BIAS_LIMIT_PURITY, BIAS_LIMIT_RECOVERY = 0.2, 0.5
# A linearization with entries this large is numerically useless (chosen): the move-to-output map M
# above USELESS_GAIN, or the affine part of the prediction Gf above USELESS_AFFINE.
# Judgement call: the thresholds are orders of magnitude, not measured. Counting re-anchors per
# run, and the largest healthy entries, would show how much room there is either side.
USELESS_GAIN, USELESS_AFFINE = 1e8, 1e6


# =====================================================================================================
# Small containers for one linearization
# =====================================================================================================
class _Jacobians(NamedTuple):
    """The circuit linearized at one point, in continuous time, with the inputs split into the
    handles the controller moves and the disturbances it's told about.

        dxi/dt = A xi + B du + E dd + f_c
        y      = y_c + C xi + Du du + Dd dd
    """
    f_c: np.ndarray        # dx/dt at the point (the affine term), length nx
    y_c: np.ndarray        # outputs at the point, [purity, recovery, z_0 .. z_{N-1}], length ny
    A: np.ndarray          # df/dx, nx by nx
    B: np.ndarray          # df/du for the handles, nx by nu
    E: np.ndarray          # df/dd for the measured disturbances, nx by nd
    C: np.ndarray          # dy/dx, ny by nx
    Du: np.ndarray         # dy/du for the handles, ny by nu
    Dd: np.ndarray         # dy/dd, ny by nd
    g_reagent: np.ndarray  # d(reagent rate)/du for the handles, mol/h per unit of each handle


class _Discrete(NamedTuple):
    """The same model over one control step, exact for inputs held over the step:
    xi[k+1] = Ad xi[k] + Bd du + Ed dd + fd."""
    Ad: np.ndarray
    Bd: np.ndarray
    Ed: np.ndarray
    fd: np.ndarray


class _QP(NamedTuple):
    """One QP as the solver takes it:

        minimize 1/2 w'Hw + g'w   subject to  lba <= A w <= uba,  lbw <= w <= ubw

    The lower bounds lbw on w never change, so the controller keeps them; the upper bounds
    change when the spec is ranked first (see CircuitMPC._ranked_bounds)."""
    H: sp.csc_matrix
    g: np.ndarray
    A: sp.csc_matrix
    lba: np.ndarray
    uba: np.ndarray
    ubw: np.ndarray


# =====================================================================================================
# Sparse-matrix helpers for the QP
# =====================================================================================================
def _csc_full(block: np.ndarray) -> sp.csc_matrix:
    """A csc matrix that keeps every entry of a dense block, zeros included, so the sparsity
    pattern handed to the QP solver is the same at every step.

    OSQP is set up once with a sparsity pattern; if a Jacobian entry happened to be exactly zero
    on one step and scipy dropped it, the pattern would change and the solver would refuse it.
    """
    m = sp.csc_matrix(np.ones_like(block, dtype=float))
    m.data = np.asarray(block, float).ravel(order="F")
    return m


def _zeros(rows: int, cols: int) -> sp.csc_matrix:
    """An all-zero sparse block, rows by cols (it stores no entries)."""
    return sp.csc_matrix((rows, cols))


# =====================================================================================================
# The controller
# =====================================================================================================
class CircuitMPC:
    """The linear MPC on the circuit, re-linearized on its own model state every step.

    It satisfies loop.Controller: the closed loop calls step(obs) once per control interval and
    apply(handles) to turn the handle values into the plant's inputs.

    Example:
        ctl = CircuitMPC(model, x_ss, u_ss, rc.cut, spec, 0.995, rec, MPCSettings())
        traj, log = run_closed_loop(plant, x0, ctl, scenario, t_end=40.0)
        # or one step by hand:
        handles = ctl.step(obs)          # e.g. {"Q_org": 3990.0, ..., "H_jackleg": array([...])}
        u = ctl.apply(handles)           # the full Inputs to give the plant
    """

    def __init__(self, circuit: Circuit, x_ss: np.ndarray, u_ss: Inputs, cut: str, spec: float,
                 purity_target: float, recovery_target: float, settings: MPCSettings | None = None,
                 level_loops=None):
        """Set the controller up at an operating point.

        Args:
            circuit: the controller's model (it's never the plant).
            x_ss: the model's rest point under u_ss (state vector).
            u_ss: the inputs at the operating point; the handles start here.
            cut: where the product is split, "LaCe" for NdPr as one product ("Pr" for validation).
            spec: the purity below which product is off spec (fraction, e.g. 0.99).
            purity_target, recovery_target: the floors the controller holds (fractions; the purity
                target above spec, as plants run).
            settings: MPCSettings; None means the defaults.
            level_loops: the PI loops on the settlers under this controller (loop.LevelLoops), or
                None when the controller moves the jacklegs itself. Part of loop.Controller.

        Eight parameters (nine with level_loops) is more than the guidelines like. They're kept: the run machinery
        (registry.py), the notebooks and the tests all build the controller this way, and each one
        is a separate fact about the job (the model, the operating point, the cut, the floors, the
        settings).
        """
        self.level_loops = level_loops      # the settler loops under it (loop.LevelLoops), or None
        # What the controller is for: the model, the operating point, the floors.
        self.circuit, self.u_nominal, self.cut = circuit, u_ss, cut
        self.spec, self.purity_target, self.recovery_target = spec, purity_target, recovery_target
        self.s = settings or MPCSettings()
        self._read_interface_window()

        # The shape of the problem: which inputs move, which outputs are held, the operating
        # point, then the parts of the QP that never change.
        self._lay = Inputs.layout(circuit.n, circuit.N)
        self._select_handles()
        self._select_disturbances()
        self._build_output_map(u_ss)
        self._set_operating_point(x_ss, u_ss)
        self._build_qp_structure()
        self._reset_running_state()

        # relinearize_every = 0: one linearization, at the operating point, kept for good.
        if self.s.relinearize_every == 0:
            self._linearize(self.x_ss, self.d_nominal)

    @property
    def t_step(self) -> float:
        """The control interval, h."""
        return self.s.t_step

    # =================================================================================================
    # Construction
    # =================================================================================================
    def _read_interface_window(self):
        """The interface window and setpoint, in metres from the settler floor, from the settler
        height and the fractions in the parameter file."""
        geo = self.circuit.p.raw["stage_geometry"]
        H = geo["settler_height_m"]
        self.z_lo, self.z_hi = (f * H for f in geo["interface_window_fraction"])
        self.z_set = geo["interface_setpoint_fraction"] * H

    def _select_handles(self):
        """Which entries of the full input vector the controller moves.

        Sets:
            _u_names, _u_sizes: the handle names and how many entries each has (N for the jacklegs).
            _cols: their positions in the full input vector; nu: how many there are.
            _jack_slice: where the jacklegs sit in u_prev (None without jackleg handles).
            _h_scrub_pos: where the scrub acidity sits in u_prev (None if it isn't a handle).
        """
        s, N, lay = self.s, self.circuit.N, self._lay
        names = list(s.handles) + (["H_jackleg"] if s.jacklegs else [])
        cols, self._u_names, self._u_sizes = [], names, []
        for name in names:
            sl = lay[name]
            cols += list(range(sl.start, sl.stop))
            self._u_sizes.append(sl.stop - sl.start)
        self.nu = len(cols)
        self._cols = np.array(cols)
        self._jack_slice = slice(self.nu - N, self.nu) if s.jacklegs else None   # jacklegs in u_prev
        self._jackleg_offset = np.zeros(N)
        # The scrub liquor's acidity is read back (a pH probe on the line is a given, like the feed
        # flowmeter): what the plant gets minus what was commanded is carried like the flow offset.
        self._h_scrub_pos = cols.index(lay["h_scrub"].start) if "h_scrub" in names else None
        self._h_scrub_offset = 0.0

    def _select_disturbances(self):
        """Which entries of the full input vector the controller is told about but doesn't move.

        The measured disturbances d: feed composition (n entries, mol/L), feed acidity (mol/L),
        feed flow (L/h; a feed flowmeter is a given), and, as the last entry, the estimator's
        organic flow OFFSET (L/h, delivered minus commanded). The offset's column is Q_org's,
        so in the linearization it acts exactly like extra organic flow.
        """
        # Why: measured disturbances go in as feedforward, as in industrial MPC (Qin and Badgwell's
        # survey). Measured cost: with the ECJTU pair plus a feed analyser the feedforward, acted on
        # through a model a few percent wrong, overreacted on the feed switch (REPORT.md section 6).
        lay = self._lay
        self._d_cols = np.array(list(range(lay["c_feed"].start, lay["c_feed"].stop))
                                + [lay["h_feed"].start, lay["Q_feed"].start, lay["Q_org"].start])
        self.nd = len(self._d_cols)

    def _build_output_map(self, u_ss: Inputs):
        """The outputs the controller holds (purity, recovery, every interface), the reagent rate,
        and all their Jacobians, as one CasADi function of (state, inputs).

        Sets:
            _lin(x, u) -> f, y, df/dx, df/du, dy/dx, dy/du, d(reagent)/du   (all on the full u)
            _y_fun(x, u) -> y
            ny = 2 + N
        """
        circuit, N, n = self.circuit, self.circuit.N, self.circuit.n
        x = ca.SX.sym("x", circuit.nx)
        uv = ca.SX.sym("u", Inputs.size(n, N))
        out = circuit._out(x=x, u=uv)
        purity, recovery = self._purity_and_recovery(out, uv, u_ss)
        # The state is [c_aq (N*n), c_org (N*n), acid (N), z (N)]: the interfaces are the last N.
        z = x[2 * N * n + N: 2 * N * n + 2 * N]
        reagent = self._reagent_rate(uv)
        y = ca.vertcat(purity, recovery, z)
        self.n_acid = circuit.cmap.n_ext if self.s.acid_floor is not None else 0
        if self.n_acid:                         # acid in the extraction stages, after the interfaces
            y = ca.vertcat(y, x[2 * N * n: 2 * N * n + self.n_acid])
        self.ny = 2 + N + self.n_acid
        f = circuit.casadi_rhs()(x, uv)
        self._lin = ca.Function("lin", [x, uv], [f, y, ca.jacobian(f, x), ca.jacobian(f, uv),
                                                 ca.jacobian(y, x), ca.jacobian(y, uv),
                                                 ca.jacobian(reagent, uv)])
        self._y_fun = ca.Function("y", [x, uv], [y])

    def _purity_and_recovery(self, out: dict, uv: ca.SX, u_ss: Inputs) -> tuple:
        """Product purity and recovery as CasADi expressions of the circuit's outlets.

        purity   = (heavy metal in the product) / (all metal in the product)
        recovery = Q_prod * (heavy metal in the product) / (Q_feed * heavy metal in the feed)
        "Heavy" is the product side of the cut: Pr and Nd when cut = "LaCe".
        Both are fractions.
        """
        lay = self._lay
        heavy = [i for i, m in enumerate(heavy_mask(self.circuit.p, self.cut)) if m]
        eps = 1e-9                         # an estimate with nothing in the product stage must not give 0/0
        purity = ca.sum1(out["c_prod"][heavy]) / (ca.sum1(out["c_prod"]) + eps)
        # Recovery against what is being fed; floored at 5 % of the nominal feed so a stopped feed
        # (Q_feed = 0) makes the ratio large and the floor inactive instead of blowing the QP up.
        # Judgement call: recovery is the ratio of product now to feed now. On a pump stop it reads
        # nonsense for a while, and that is why the controller does worse than open loop there
        # (REPORT.md section 5). Two fixes were tried and each hurt another case. A hold on loss of
        # feed, as a plant does, tested across the whole catalogue, is what would settle it.
        fed_nominal = float(u_ss.Q_feed * np.sum(np.asarray(u_ss.c_feed, float)[heavy]))
        fed = uv[lay["Q_feed"]] * ca.sum1(uv[lay["c_feed"]][heavy])
        recovery = out["Q_prod"] * ca.sum1(out["c_prod"][heavy]) / ca.fmax(fed, 0.05 * fed_nominal)
        return purity, recovery

    def _reagent_rate(self, uv: ca.SX) -> ca.SX:
        """Reagent spent, mol/h, as a CasADi expression of the inputs:

            saponified_fraction * extractant concentration * Q_org     (base, to saponify the organic)
            + Q_strip * h_strip + Q_scrub * h_scrub                     (acid, to strip and to scrub)
        """
        lay, p = self._lay, self.circuit.p
        return (uv[lay["saponified_fraction"]] * p.raw["extractant"]["concentration_monomer"]
                * uv[lay["Q_org"]]
                + uv[lay["Q_strip"]] * uv[lay["h_strip"]] + uv[lay["Q_scrub"]] * uv[lay["h_scrub"]])

    def _set_operating_point(self, x_ss: np.ndarray, u_ss: Inputs):
        """Where the controller starts: the model's rest point, the full input vector at the
        operating point, the handles there (u0_h, and u_prev, the handles held now), and the
        nominal disturbances (the flow offset's nominal value is 0, not the organic flow)."""
        self.x_ss, self.u_ss_v = np.asarray(x_ss, float), u_ss.numpy()
        self.u0_h = self.u_ss_v[self._cols].copy()
        self.u_prev = self.u0_h.copy()
        self.d_nominal = self.u_ss_v[self._d_cols].copy()
        self.d_nominal[-1] = 0.0
        self._d_last = self.d_nominal.copy()          # the disturbances last measured (resync uses them)

    def _reset_running_state(self):
        """What changes from step to step: the solver (built on the first solve), the last QP
        solution (the warm start), the record, the counters, the model state and the bias."""
        self._solver = None                     # OSQP for the full QP (built on first use)
        self._solvers = {}                       # the other solvers, by name (built on first use)
        self._w_prev = np.zeros(self.nw)
        self.last = {}
        self.failures = 0
        self._k = 0
        self.x_m = self.x_ss.copy()              # the controller's own model state, on the model's manifold
        self.bias = np.zeros(self.ny)            # reported outputs minus the model state's, filtered
        self._lin_point = None

    # ---- the fixed part of the QP ------------------------------------------------------------------
    def _build_qp_structure(self):
        """The parts of the QP that never change: which moves exist, how they add up to handle
        values over the horizon, the bounds on the moves, and the fixed part of the cost.

        The key matrix is T. Stack the handle values over the horizon, u_0 .. u_{Hp-1}; then
            u_k - u_prev = T_k @ dU          (T_k = rows k*nu .. (k+1)*nu of T)
        because each handle at step k is where it is now plus every move made at or before k.
        """
        s = self.s
        self.Hp, self.Hu = s.horizon, s.control_horizon
        self._spread_handle_limits()
        self._lay_out_moves()
        n_slack = self._count_slacks()
        self.nw = self._n_du + n_slack
        self._locate_output_rows()
        self._H = self._fixed_hessian(self._per_entry(s.w_move) / self._rate ** 2, n_slack)
        self._lbw, self._ubw = self._variable_bounds(n_slack)

    def _spread_handle_limits(self):
        """Rate limits and bounds, given per handle in the settings, spread out to one number per
        entry of u (a jackleg setting goes to all N jacklegs).

        Sets _rate (largest move per step), _lb_u and _ub_u (lowest and highest handle values).
        """
        bounds = self.s.bounds
        self._rate = self._per_entry(self.s.rate_per_step)
        self._lb_u = self._per_entry({k: v[0] for k, v in bounds.items()})
        self._ub_u = self._per_entry({k: v[1] for k, v in bounds.items()})

    def _lay_out_moves(self):
        """The move variables and the map T from moves to handle values over the horizon.

        Sets _du_cols (the moves, as (move index, handle entry)), _n_du (how many), _T, and
        _sum_T (the sum over k of T_k, which the reagent term needs).
        """
        self._du_cols = self._move_columns()
        self._n_du = len(self._du_cols)
        self._T = self._move_to_handle_map()
        self._sum_T = self._T.reshape(self.Hp, self.nu, self._n_du).sum(axis=0)

    def _count_slacks(self) -> int:
        """How many slack variables follow the moves in w (order in the module docstring).

        Also sets where the interfaces are checked (_ichk, _n_ic) and how many acid-floor slacks
        there are (_n_as; they come after all the others). The two slacks before the acid ones
        are purity and recovery at steady state.
        """
        self._ichk = [k for k in self.s.interface_check_steps if k < self.Hp]
        self._n_ic = len(self._ichk)
        self._n_as = self._n_ic * self.n_acid
        return 3 * self.Hp + 2 * self._n_ic * self.circuit.N + 2 + self._n_as

    def _variable_bounds(self, n_slack: int) -> tuple:
        """Bounds on w: each move within +-its rate limit, every slack >= 0.

        Returns:
            (lbw, ubw), both of length nw.
        """
        rate_du = np.array([self._rate[j] for _, j in self._du_cols])
        return (np.concatenate([-rate_du, np.zeros(n_slack)]),
                np.concatenate([rate_du, np.full(n_slack, np.inf)]))

    def _per_entry(self, per_handle: dict) -> np.ndarray:
        """A per-handle setting (by name) as one number per entry of u, in u's order."""
        return np.concatenate([np.full(k, per_handle[name]) for name, k in zip(self._u_names, self._u_sizes)])

    def _move_columns(self) -> list:
        """The move variables: every handle may move at steps 0..Hu-1, except the jacklegs, which
        get `jackleg_moves` moves (one, by default). Listed as (move index, handle entry)."""
        is_jack = self._jackleg_entries()
        return [(i, j) for i in range(self.Hu) for j in range(self.nu)
                if self._may_move(i, is_jack[j])]

    def _jackleg_entries(self) -> np.ndarray:
        """A mask over the entries of u: True where the entry is a jackleg."""
        is_jack = np.zeros(self.nu, bool)
        if self._jack_slice is not None:
            is_jack[self._jack_slice] = True
        return is_jack

    def _may_move(self, i: int, is_jackleg: bool) -> bool:
        """Whether a handle entry gets a move variable at move index i. Why the jacklegs get
        fewer: an interface settles in minutes, so later jackleg moves change nothing the plan
        can see (MPCSettings.jackleg_moves)."""
        return (not is_jackleg) or i < self.s.jackleg_moves

    def _move_to_handle_map(self) -> np.ndarray:
        """T, (Hp*nu) by n_du: a move of handle entry j at step i shows in that entry at every
        step k >= i (the move is held after it's made). Entries are 0 or 1."""
        Hp, nu = self.Hp, self.nu
        T = np.zeros((Hp * nu, self._n_du))                  # u_k - u_prev = T_k @ du
        for col, (i, j) in enumerate(self._du_cols):
            for k in range(i, Hp):
                T[k * nu + j, col] = 1.0
        return T

    def _locate_output_rows(self):
        """Rows of the stacked outputs Y (Hp blocks of ny, one per horizon step) that the QP uses:
        purity and recovery at every step, and the interfaces at the checked steps."""
        self._pur = np.arange(self.Hp) * self.ny
        self._rec = self._pur + 1
        N = self.circuit.N
        self._zrows = np.concatenate([np.arange(k * self.ny + 2, k * self.ny + 2 + N) for k in self._ichk])
        self._arows = np.concatenate([np.arange(k * self.ny + 2 + N, (k + 1) * self.ny) for k in self._ichk]) \
            if self.n_acid else np.zeros(0, int)

    def _fixed_hessian(self, move_weight: np.ndarray, n_slack: int) -> sp.csc_matrix:
        """The Hessian of the fixed part of the cost. The solver minimizes 1/2 w'Hw + g'w, so H
        carries the factor 2.

        Moves:  sum of w_move * (du / rate)^2       -> move_weight (= w_move / rate^2) on the diagonal
        Hold:   w_hold * sum over k of ((u_k - u_prev) / rate)^2 / Hp
                = w_hold * du' T' W T du / Hp,  W = diag(1 / rate^2), so it couples the moves
        Slacks: their weights on the diagonal, in the order of w.

        Args:
            move_weight: per entry of u, w_move / rate^2.
            n_slack: how many slack variables follow the moves.
        """
        s, Hp, T = self.s, self.Hp, self._T
        H_du = np.diag(np.array([move_weight[j] for _, j in self._du_cols]))
        W = np.diag(np.tile(1.0 / self._rate ** 2, Hp))
        H_du = H_du + s.w_hold * (T.T @ W @ T) / Hp
        slack_w = np.concatenate([np.full(Hp, s.w_purity_floor), np.full(Hp, s.w_purity),
                                  np.full(Hp, s.w_recovery),
                                  np.full(2 * self._n_ic * self.circuit.N, s.w_interface),
                                  [s.w_purity, s.w_recovery], np.full(self._n_as, s.w_acid)])
        return sp.block_diag([sp.csc_matrix(2.0 * H_du), sp.diags(2.0 * slack_w)]).tocsc()

    # =================================================================================================
    # The full input vector the plant is believed to see
    # =================================================================================================
    def _u_full(self, d) -> np.ndarray:
        """The full input vector as the controller believes the plant sees it: the operating
        point, with the handles held now, the measured disturbances, and the offsets read back
        from the plant written in.

        Args:
            d: the disturbance vector (see _select_disturbances).
        """
        u_full = self.u_ss_v.copy()
        u_full[self._cols] = self.u_prev
        u_full[self._d_cols[:-1]] = d[:-1]
        u_full[self._d_cols[-1]] += d[-1]                      # flow offset adds to the commanded flow
        # hydraulic drift: effective jackleg = commanded + offset
        # Why (both offsets): measured. The jackleg offset took the drift case from 0.365 to 0.006 m*h
        # (decision 18b); the scrub acid read-back took a strong scrub liquor from 2240 mol lost to 56
        # (decision 23 (3)). Judgement call: a pH probe that reads 0.05-2 M acid is assumed to exist.
        u_full[self._lay["H_jackleg"]] += self._jackleg_offset
        u_full[self._lay["h_scrub"].start] += self._h_scrub_offset   # scrub acid as read back on the line
        if self.s.strip_tied_to_scrub is not None:
            u_full[self._lay["Q_strip"].start] = self.s.strip_tied_to_scrub + u_full[self._lay["Q_scrub"].start]
        return u_full

    def _infer_jackleg_offsets(self, x: np.ndarray) -> np.ndarray:
        """Where each interface actually sits says what jackleg height the settler is really seeing
        (it settles in minutes, so between control steps it is at rest). The difference from the
        commanded height is a hydraulic drift the model does not know about, and it is fed
        forward exactly like the flow offset (decision 18b). Without it the controller believes
        the interface will fall back on its own and leaves it where it is.

        The balance at rest: the aqueous column in the jackleg (height H, density rho_aq) weighs
        the same as the aqueous layer (height z) plus the organic above it up to the weir:
            H_seen = (z * (rho_aq - rho_org) + rho_org * weir) / rho_aq
        The offset is H_seen - H_commanded, in m, one per settler.

        Args:
            x: the plant state as the controller is told it (truth or estimate).
        """
        p, N = self.circuit.p, self.circuit.N
        weir = p.raw["stage_geometry"]["organic_weir_height_m"]
        c_aq, c_org, _, z = self.circuit.unpack(x)
        H_cmd = self.u_prev[self._jack_slice] if self.s.jacklegs else self.u_ss_v[self._lay["H_jackleg"]]
        off = np.zeros(N)
        for j in range(N):
            ra = float(chemistry.rho_aqueous(np.asarray(c_aq[j], float).ravel(), p))
            ro = float(chemistry.rho_organic(np.asarray(c_org[j], float).ravel(), p))
            off[j] = (float(z[j]) * (ra - ro) + ro * weir) / ra - float(H_cmd[j])
        return off

    # =================================================================================================
    # Linearize at (x_c, handles u_prev, disturbances d_c); build the predictor
    # =================================================================================================
    def _linearize(self, x_c: np.ndarray, d_c: np.ndarray):
        """Linearize the model at the state x_c, the handles held now and the disturbances d_c,
        discretize it, and build everything the prediction and the QP need. The result is kept in
        self._lin_point (a dict) until the next linearization.

        Args:
            x_c: the state to linearize at (the controller's own model state x_m, normally).
            d_c: the disturbance vector at the linearization point.
        """
        jac = self._jacobians(x_c, d_c)
        y_ss0, G_ss = self._steady_state_gains(jac)
        disc = self._discretize(jac)
        Phi, Gu, Gd, Gf = self._prediction_matrices(disc, jac)
        # M = Gu T: the map from the moves straight to the stacked outputs, Y = Y0 + M dU.
        self._lin_point = dict(x_c=np.asarray(x_c, float).copy(), d_c=np.asarray(d_c, float).copy(),
                               u_c_h=self.u_prev.copy(), y_c=jac.y_c, C=jac.C, Phi=Phi, Gu=Gu, Gd=Gd,
                               Gf=Gf, M=Gu @ self._T, g_reagent=jac.g_reagent, y_ss0=y_ss0, G_ss=G_ss,
                               Ad=disc.Ad, Bd=disc.Bd, Ed=disc.Ed, fd=disc.fd)

    def _jacobians(self, x_c: np.ndarray, d_c: np.ndarray) -> _Jacobians:
        """Evaluate the model and its Jacobians at the point and pick out the handle columns and
        the disturbance columns.

        With the strip flow tied to the scrub flow (Q_strip = net + Q_scrub), a scrub move also
        moves the strip flow, so the strip column is added onto the scrub column:
        d/dQ_scrub = df/dQ_scrub + df/dQ_strip.
        """
        f_c, y_c, A, Fu, C, Yu, Gr = (np.array(m) for m in self._lin(x_c, self._u_full(d_c)))
        f_c, y_c = f_c.ravel(), y_c.ravel()
        if self.s.strip_tied_to_scrub is not None:        # strip flow moves with the scrub flow
            scrub, strip = self._lay["Q_scrub"].start, self._lay["Q_strip"].start
            Fu[:, scrub] += Fu[:, strip]
            Yu[:, scrub] += Yu[:, strip]
            Gr[:, scrub] += Gr[:, strip]
        return _Jacobians(f_c=f_c, y_c=y_c, A=A, B=Fu[:, self._cols], E=Fu[:, self._d_cols], C=C,
                          Du=Yu[:, self._cols], Dd=Yu[:, self._d_cols], g_reagent=Gr[0, self._cols])

    def _steady_state_gains(self, jac: _Jacobians) -> tuple:
        """Where the handles would leave the plant in the long run, as far as the linear model
        knows: the model's rest point at the current handles, and the steady-state gain of a handle
        move. The inventory mode of a reflux circuit is slow; this sees past the horizon.

        Set dxi/dt = 0:  xi = -A^-1 (f_c + B du), so
            y_ss = (y_c - C A^-1 f_c) + (Du - C A^-1 B) du = y_ss0 + G_ss du

        Returns:
            y_ss0 (ny), G_ss (ny by nu). If A is singular: y_c and zero gains.
        """
        # Judgement call: the long-run gain comes from one linearization, so it is only right near
        # the model state. With the ~400 h inventory mode, small errors in A can move -C A^-1 B a lot.
        # Comparing these gains with full-model steady states after single moves would settle it.
        A, B, C, Du, f_c, y_c = jac.A, jac.B, jac.C, jac.Du, jac.f_c, jac.y_c
        try:
            return y_c - C @ np.linalg.solve(A, f_c), Du - C @ np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            return y_c.copy(), np.zeros((self.ny, self.nu))

    def _discretize(self, jac: _Jacobians) -> _Discrete:
        """Exact discretization over one control step, the affine term f_c carried as a constant
        input. Build the augmented matrix

            Maug = [[A, B, E, f_c],
                    [0, 0, 0, 0  ]]

        and take expm(Maug * t_step). Its top row of blocks is [Ad, Bd, Ed, fd]: Ad = e^(A t_step),
        and Bd, Ed, fd are the integrals of e^(A s) over the step times B, E and f_c.
        """
        # Why: exact zero-order-hold discretization by the augmented matrix exponential is the
        # textbook method (Rawlings, Mayne and Diehl), and the handles really are held over a step.
        nx, nu, nd = self.circuit.nx, self.nu, self.nd
        M = np.zeros((nx + nu + nd + 1,) * 2)
        M[:nx, :nx], M[:nx, nx:nx + nu], M[:nx, nx + nu:nx + nu + nd], M[:nx, -1] = \
            jac.A, jac.B, jac.E, jac.f_c
        Md = expm(M * self.s.t_step)
        return _Discrete(Ad=Md[:nx, :nx], Bd=Md[:nx, nx:nx + nu], Ed=Md[:nx, nx + nu:nx + nu + nd],
                         fd=Md[:nx, -1])

    def _prediction_matrices(self, disc: _Discrete, jac: _Jacobians) -> tuple:
        """Stacked outputs over the horizon: Y = Phi xi0 + Gu U + Gd dd + Gf.

        Step the discrete model forward k times from xi0:
            xi_k = Ad^k xi0 + sum over j < k of Ad^(k-1-j) (Bd u_j + Ed dd + fd)
            y_k  = C xi_k + Du u_k + Dd dd                    (y_c is added later, in _free_response)
        for k = 1..Hp. Block row k-1 of each matrix is y_k. U = [u_0 .. u_{Hp-1}] (deviations from
        the linearization point); y_Hp uses u_{Hp-1}, since the handles are held after the horizon.

        Returns:
            Phi (Hp*ny by nx), Gu (Hp*ny by Hp*nu), Gd (Hp*ny by nd), Gf (Hp*ny).
        """
        Hp, ny, nu, nd = self.Hp, self.ny, self.nu, self.nd
        CA, CAB, CAE, CAf = self._output_powers(disc, jac.C)
        Phi = np.zeros((Hp * ny, self.circuit.nx))
        Gu = np.zeros((Hp * ny, Hp * nu))
        Gd = np.zeros((Hp * ny, nd))
        Gf = np.zeros(Hp * ny)
        for k in range(1, Hp + 1):
            r = slice((k - 1) * ny, k * ny)
            Phi[r] = CA[k]
            for j in range(k):
                Gu[r, j * nu:(j + 1) * nu] += CAB[k - 1 - j]
                Gd[r] += CAE[k - 1 - j]
                Gf[r] += CAf[k - 1 - j]
            kk = min(k, Hp - 1)                                   # y_k uses u_k, held after the horizon
            Gu[r, kk * nu:(kk + 1) * nu] += jac.Du
            Gd[r] += jac.Dd
        return Phi, Gu, Gd, Gf

    def _output_powers(self, disc: _Discrete, C: np.ndarray) -> tuple:
        """C Ad^m for m = 0..Hp, and the same times Bd, Ed and fd: the building blocks of the
        prediction (how a state, a handle, a disturbance and the affine term show in the
        outputs m steps later)."""
        CA = [C]
        for _ in range(self.Hp):
            CA.append(CA[-1] @ disc.Ad)
        return CA, [m @ disc.Bd for m in CA], [m @ disc.Ed for m in CA], [m @ disc.fd for m in CA]

    # =================================================================================================
    # One controller step
    # =================================================================================================
    def step(self, obs: Observation) -> dict:
        """Handle values to apply until the next step, given what the controller is told (see
        loop.Observation): what is measured is used as measured, and the estimate stands in only
        for what is not.

        Args:
            obs: the state as known, the feed as measured, the offsets read back, and a product
                analyser's reading if there is one.

        Returns:
            The handle values by name: floats, and an array of N for "H_jackleg". After the call,
            self.last holds what the controller wants recorded for this step (QP status, bias,
            plan); a failed or unusable step holds the handles where they were.
        """
        # measure: the disturbances and the offsets the plant reveals
        d = self._disturbances(obs)
        self._d_last = d
        self._update_offsets(obs)
        u_full = self._u_full(d)
        # bias: reported outputs minus the model state's, filtered
        self._update_bias(obs, u_full)
        # linearize: on the model's own trajectory, never at the plant's state
        # Why: measured, decision 21. At the plant's state the Jacobian had the slow mode 15x too
        # fast and the purity gain of the wrong sign; on the model's own trajectory it matches the
        # full model to four decimals for small moves over 12 h.
        if self._due_for_linearization():
            self._linearize(self.x_m, d)
        lin_point = self._lin_point
        if not self._usable(lin_point):
            return self._hold_and_reanchor()
        # predict, solve, advance
        Y0 = self._free_response(lin_point, d, u_full)
        self._solve(lin_point, Y0)
        self._advance_model_state(d)
        return self.handles()

    def apply(self, handles: dict) -> Inputs:
        """Nominal inputs with the controller's handles written in (and the strip flow set from
        the scrub flow, when they're tied)."""
        u = self.u_nominal.replace(**handles)
        if self.s.strip_tied_to_scrub is not None:
            u = u.replace(Q_strip=self.s.strip_tied_to_scrub + u.Q_scrub)
        return u

    def _due_for_linearization(self) -> bool:
        """True on the steps where the model is re-linearized: every relinearize_every-th step,
        never when it's 0 (then the one linearization from construction is kept)."""
        return self.s.relinearize_every > 0 and self._k % self.s.relinearize_every == 0

    # ---- measure --------------------------------------------------------------------------------
    def _disturbances(self, obs: Observation) -> np.ndarray:
        """Feed composition (mol/L), feed acid (mol/L), feed flow (L/h), and the organic flow
        offset (L/h), as one vector d."""
        return np.concatenate([np.asarray(obs.c_feed, float).ravel(),
                               [float(obs.h_feed), float(obs.Q_feed), float(obs.qorg_offset)]])

    def _update_offsets(self, obs: Observation):
        """Hydraulic drift from where the interfaces sit; scrub acid as read back on the line
        (what the plant gets minus what was commanded, mol/L)."""
        self._jackleg_offset = self._infer_jackleg_offsets(np.asarray(obs.x, float))
        if obs.h_scrub is None:
            return
        commanded = (self.u_prev[self._h_scrub_pos] if self._h_scrub_pos is not None
                     else self.u_ss_v[self._lay["h_scrub"].start])
        self._h_scrub_offset = float(obs.h_scrub) - float(commanded)

    # ---- bias -----------------------------------------------------------------------------------
    def _update_bias(self, obs: Observation, u_full: np.ndarray):
        """The output bias: what is reported (true state, estimate, or a product analyser) minus
        what the model state gives, filtered. Both sides clipped to what a purity and a recovery
        can be, and the bias to its limits, so the QP never plans on nonsense.

            b <- (1 - a) b + a (y_reported - y_model),     a = bias_filter (0.5)

        The bias is added to every predicted output, which is what gives the controller integral
        action (DMC style; decision 21, APPENDIX_B_control.md B.2). Also records the purity and
        recovery the controller was told, for the log.
        """
        # Judgement call (output bias against a disturbance model in the MPC): a constant output
        # bias assumes the plant-model gap acts like a step on the outputs. A gap that enters at
        # the inputs and works through the slow inventory mode is then followed late (the point
        # made by Muske and Badgwell, and by Pannocchia and Rawlings). The estimator's feed and
        # flow offsets cover the input side when there are sensors; with the full state, the
        # bias is the only integral action. Putting an input-disturbance model into the prediction
        # and comparing on the feed switch would settle it.
        y_rep = self._reported_outputs(obs, u_full)
        self._seen = (float(y_rep[0]), float(y_rep[1]))
        if self.s.bias_filter <= 0:
            return
        y_mod = np.array(self._y_fun(self.x_m, u_full)).ravel()
        if not (np.all(np.isfinite(y_rep)) and np.all(np.isfinite(y_mod))):
            return
        _clip_purity_recovery(y_mod)
        a = self.s.bias_filter
        self.bias = (1 - a) * self.bias + a * (y_rep - y_mod)
        self.bias[0] = float(np.clip(self.bias[0], -BIAS_LIMIT_PURITY, BIAS_LIMIT_PURITY))
        self.bias[1] = float(np.clip(self.bias[1], -BIAS_LIMIT_RECOVERY, BIAS_LIMIT_RECOVERY))

    def _reported_outputs(self, obs: Observation, u_full: np.ndarray) -> np.ndarray:
        """The outputs y from the reported state, with a product analyser's purity (and recovery,
        if it's finite, which means the reading was prompt) taken as measured in place of the
        estimate's (decision 23). Purity and recovery clipped to what they can be."""
        y_rep = np.array(self._y_fun(np.asarray(obs.x, float), u_full)).ravel()
        # Why: a measured controlled variable is used as measured, the DMC way. Measured, decision 23:
        # on the perturbed plant the estimate alone said 99.98 % when the truth was 99.77 %.
        for i, v in enumerate(obs.product[:2] if obs.product is not None else ()):
            if np.isfinite(v):
                y_rep[i] = float(v)
        _clip_purity_recovery(y_rep)
        return y_rep

    # ---- predict --------------------------------------------------------------------------------
    @staticmethod
    def _usable(lin_point: dict) -> bool:
        """False when the linearization is no use: anything not finite, or the move-to-output map
        or the affine term out of all proportion (USELESS_GAIN, USELESS_AFFINE)."""
        finite = all(np.all(np.isfinite(lin_point[k]))
                     for k in ("y_c", "Phi", "Gu", "Gd", "Gf", "y_ss0", "G_ss"))
        return (finite and np.abs(lin_point["M"]).max() < USELESS_GAIN
                and np.abs(lin_point["Gf"]).max() < USELESS_AFFINE)

    def _hold_and_reanchor(self) -> dict:
        """The model state is somewhere the model cannot be linearized to any use: hold the handles,
        and put the model state back on the model's own rest point. (Putting it where the estimate
        says was tried: an estimate built on a wrong model sits off the model's manifold, and
        every linearization after that was useless too.) The bias re-learns from there."""
        # Judgement call: the rest point x_ss belongs to the operating-point handles, not the
        # handles held now, so for a while the model state and the handles disagree and the bias
        # carries a large step. Re-anchoring to the rest point under the current handles, and
        # counting how often this branch runs in the matrix, would show whether it matters.
        self.failures += 1
        self.x_m = self.x_ss.copy()
        self._k += 1
        self.last = {"cost": float("nan"), "purity_slack": float("nan"),
                     "qp_ok": False, "qp_status": "not finite",
                     "bias_purity": float(self.bias[0]), "bias_recovery": float(self.bias[1])}
        return self.handles()

    def _free_response(self, lin_point: dict, d: np.ndarray, u_full: np.ndarray) -> np.ndarray:
        """Outputs over the horizon with the handles held where they are, from the model state,
        plus the bias:

            Y0 = y_c (each step) + Phi xi0 + Gu (u_prev - u_c, each step) + Gd (d - d_c) + Gf
                 + bias (each step),          xi0 = x_m - x_c

        When the model wasn't linearized on this step, the linear model's y at x_m is off from
        the nonlinear one; that difference is added at every step so the prediction starts from
        what the model says now.
        """
        Hp, L = self.Hp, lin_point
        xi0 = self.x_m - L["x_c"]
        Y0 = (np.tile(L["y_c"], Hp) + L["Phi"] @ xi0 + L["Gu"] @ np.tile(self.u_prev - L["u_c_h"], Hp)
              + L["Gd"] @ (d - L["d_c"]) + L["Gf"] + np.tile(self.bias, Hp))
        if self.s.relinearize_every != 1:      # not linearized here: correct by what the model says now
            y_now = np.array(self._y_fun(self.x_m, u_full)).ravel()
            Y0 += np.tile(y_now - (L["y_c"] + L["C"] @ xi0), Hp)
        return Y0

    # ---- solve ----------------------------------------------------------------------------------
    def _solve(self, lin_point: dict, Y0: np.ndarray):
        """Solve the QP and take the first move:

            minimize 1/2 w'Hw + g'w   subject to  lba <= A w <= uba,  lbw <= w <= ubw
            w = [moves, slacks] (order in the module docstring)

        A failed solve holds the handles. Sets u_prev, du_plan (the planned moves), Y_plan (the
        outputs the controller expects from its plan) and last (the record for this step).
        """
        H, g = self._qp_cost(lin_point, Y0)
        A, lba, uba = self._qp_constraints(lin_point, Y0)
        ubw, spec_floor = self._ranked_bounds(A, lba, uba)
        sol, stats, ok, used = self._solve_qp("full", _QP(H, g, A, lba, uba, ubw))
        w = np.array(sol["x"]).ravel()
        self._take_first_move(w, ok)
        self.Y_plan = Y0 + lin_point["M"] @ self.du_plan      # what the controller expects from its plan
        self.last = self._step_record(sol, stats, w, ok)
        self.last.update(qp_solver=used, spec_shortfall_best=spec_floor)
        self.failures += 0 if ok else 1
        self._k += 1

    def _ranked_bounds(self, A, lba, uba) -> tuple:
        """Upper bounds on w for the full QP, and the smallest purity shortfall below spec found
        (thousandths, summed over the horizon; nan if not ranked or the first QP failed).

        Ranked constraints (decision 30): a first QP with the same constraints but a cost on the
        spec slacks alone (plus a tiny cost on everything, so it has one answer) finds how close to
        spec the handles can bring purity over the horizon. The full QP then gets each spec slack
        capped at what the first QP needed, plus ranked_tolerance. Recovery, the purity target,
        interfaces and reagent then share what's left.

        Args:
            A, lba, uba: the full QP's constraints (the first QP uses the same ones).
        """
        if not self.s.ranked:
            return self._ubw, float("nan")
        Hp, n_du = self.Hp, self._n_du
        spec = slice(n_du, n_du + Hp)                           # the spec slacks, thousandths
        diag = np.full(self.nw, 1e-6)
        diag[spec] = 1.0
        H1 = sp.diags(diag).tocsc()
        sol, _, ok, _ = self._solve_qp("spec_first", _QP(H1, np.zeros(self.nw), A, lba, uba, self._ubw))
        if not ok:
            return self._ubw, float("nan")
        s1 = np.maximum(np.array(sol["x"]).ravel()[spec], 0.0)
        ubw = self._ubw.copy()
        ubw[spec] = s1 + self.s.ranked_tolerance
        return ubw, float(np.sum(s1))

    def _solve_qp(self, key: str, qp: _QP) -> tuple:
        """Solve one QP with OSQP, and with the fallback solver if OSQP fails.

        Args:
            key: which solver to use: "full" for the full QP's own OSQP (warm started), any other
                name for a solver kept under that name (built on first use).
            qp: the problem.

        Returns:
            (solution, solver stats, ok, the name of the solver that answered).
        """
        osqp = self._qp_solver(qp.H, qp.A) if key == "full" else self._named_solver(key, "osqp", qp.H, qp.A)
        sol = osqp(**self._solver_args(qp), x0=self._w_prev)
        stats = osqp.stats()
        if _succeeded(stats) or self.s.fallback_solver is None:
            return sol, stats, _succeeded(stats), "osqp"
        return self._solve_with_fallback(key + "_fallback", qp)

    def _solve_with_fallback(self, key: str, qp: _QP) -> tuple:
        """Solve the QP with the fallback solver (qpOASES by default), kept under `key`.
        Returns the same tuple as _solve_qp."""
        backup = self._named_solver(key, self.s.fallback_solver, qp.H, qp.A)
        sol = backup(**self._solver_args(qp))
        stats = backup.stats()
        ok = _succeeded(stats)
        if not ok:
            # Why: qpOASES keeps its active set from one call to the next, and after a failed
            # solve every later call fails too ("unable to perform homotopy as previous QP is not
            # solved"), so one bad step became a permanent failure (found by notebook 36's
            # watchdog test). A fresh solver next time starts clean.
            del self._solvers[key]
        return sol, stats, ok, self.s.fallback_solver

    def _solver_args(self, qp: _QP) -> dict:
        """The QP in the keyword form CasADi's conic solvers take."""
        return dict(h=ca.DM(qp.H), g=qp.g, a=ca.DM(qp.A), lba=qp.lba, uba=qp.uba, lbx=self._lbw, ubx=qp.ubw)

    def _named_solver(self, key: str, plugin: str, H, A):
        """A QP solver built once per (key) from the sparsity of H and A, then reused."""
        if key not in self._solvers:
            opts = {"osqp": self.s.osqp} if plugin == "osqp" else {"printLevel": "none"}
            self._solvers[key] = ca.conic("qp", plugin, {"h": ca.DM(H).sparsity(), "a": ca.DM(A).sparsity()},
                                          {**opts, "error_on_fail": False})
        return self._solvers[key]

    def _qp_solver(self, H: sp.csc_matrix, A: sp.csc_matrix):
        """The OSQP solver, through CasADi's conic interface. Built on the first call from the
        sparsity patterns of H and A (they're the same every step), then reused; warm started
        from the last solution."""
        if self._solver is None:
            self._solver = ca.conic("qp", "osqp", {"h": ca.DM(H).sparsity(), "a": ca.DM(A).sparsity()},
                                    {"osqp": self.s.osqp, "error_on_fail": False, "warm_start_primal": True})
        return self._solver

    def _take_first_move(self, w: np.ndarray, ok: bool):
        """Apply the moves at step 0 (the first nu entries of w, with the default one jackleg move
        per step) and clip the handles to their bounds. A failed solve moves nothing, and the plan
        becomes all zeros. A good solution is kept as the next warm start."""
        n_du = self._n_du
        # Why: holding the handles on a failed solve is the usual fallback in plant MPC; the
        # receding horizon tries again next step. It is counted, and the studies report it.
        if ok:
            self._w_prev = w
        self.u_prev = np.clip(self.u_prev + (w[:self.nu] if ok else np.zeros(self.nu)),
                              self._lb_u, self._ub_u)
        self.du_plan = w[:n_du] if ok else np.zeros(n_du)

    def _step_record(self, sol: dict, stats: dict, w: np.ndarray, ok: bool) -> dict:
        """What the controller wants recorded for this step: the QP's cost and status, the purity
        slack below spec at the first horizon step (thousandths), the bias, what it was told, and
        the purity and recovery its plan ends on."""
        n_du = self._n_du
        return {"cost": float(sol["cost"]) if ok else float("nan"),
                "purity_slack": float(w[n_du]) if ok else float("nan"),
                "qp_ok": ok, "qp_status": str(stats.get("return_status", "")),
                "bias_purity": float(self.bias[0]), "bias_recovery": float(self.bias[1]),
                "purity_seen": self._seen[0], "recovery_seen": self._seen[1],
                "purity_plan": float(self.Y_plan[self._pur[-1]]),
                "recovery_plan": float(self.Y_plan[self._rec[-1]])}

    def _qp_cost(self, lin_point: dict, Y0: np.ndarray):
        """Reagent (linear in the moves) and the pull of each interface toward its setpoint
        (quadratic), on top of the fixed move and slack weights.

        Reagent: w_reagent * sum over k of (g_reagent . (u_k - u_prev)) * t_step / 1000
                 = w_reagent * t_step * (g_reagent / 1000) @ sum_T @ dU
          (g_reagent in mol/h per unit of handle; / 1000 gives kmol/h. The part that doesn't
          depend on the moves is a constant and is left out.)
        Centering: w_center * sum over checked steps of (Mz dU + (Y0_z - z_set) * 100)^2, with
          Mz = M_z * 100, so the offset is in cm like the window slacks. Expanded into the
          1/2 w'Hw + g'w form: H += 2 w_center Mz'Mz, g += 2 w_center Mz' (Y0_z - z_set) * 100.
          (Before decision 26 the code divided by 100 here, which made the term 1e8 too weak.)
        """
        # Judgement call (reagent as a linear term): an economic term in the dynamic QP, in place
        # of an LP or QP target layer above it. Simple, and it does pull toward cheaper handles.
        # It also pushes against the floors, and with quadratic-only slacks a floor then gives way
        # by a small amount; w_hold was added for the chatter. Not compared with a target layer.
        s, n_du, L = self.s, self._n_du, lin_point
        g = np.zeros(self.nw)
        g[n_du:n_du + self.Hp] = s.w_purity_floor_linear       # exact penalty on the spec slacks
        g[:n_du] = s.w_reagent * s.t_step * (L["g_reagent"] / 1000.0) @ self._sum_T
        H = self._H.copy()
        if s.w_center > 0:                     # w_center * sum over checked steps of ((z - z_set) in cm)^2
            Mz = L["M"][self._zrows] * 100.0
            Hc = sp.csc_matrix(2.0 * s.w_center * (Mz.T @ Mz))
            H = (H + sp.block_diag([Hc, _zeros(self.nw - n_du, self.nw - n_du)])).tocsc()
            g[:n_du] += 2.0 * s.w_center * Mz.T @ ((Y0[self._zrows] - self.z_set) * 100.0)
        return H, g

    def _qp_constraints(self, lin_point: dict, Y0: np.ndarray):
        """Rows, in order: purity >= spec, purity >= target, recovery >= target (every horizon
        step); interfaces inside their window (checked steps); handles inside their bounds; purity
        and recovery at steady state for the handles the plan ends on. Each with its own slack.

        Every output row reads: (predicted output) = Y0 + M dU, so a floor y >= y_min becomes
        M dU + slack / scale >= y_min - Y0.

        Returns:
            A (sparse), lba, uba.
        """
        G_end, y_ss = self._steady_state_rows(lin_point)
        A = self._constraint_matrix(lin_point["M"], G_end)
        lba, uba = self._constraint_bounds(Y0, y_ss)
        return A, lba, uba

    def _constraint_matrix(self, M: np.ndarray, G_end: np.ndarray) -> sp.csc_matrix:
        """The constraint matrix, one block row per kind of constraint. Each block is
        [the moves' part | the slack columns], with the block's own slack scaled back to the
        output's units (1/1000 for purity, 1/100 for recovery, 1/100 for an interface in m).

            purity_k   + s_spec_k / 1000   >= spec
            purity_k   + s_target_k / 1000 >= purity_target
            recovery_k + s_rec_k / 100     >= recovery_target
            z_k + s_lo / 100 >= z_lo,       z_k - s_hi / 100 <= z_hi
            u_k inside its bounds (k < Hu; after Hu the handles are held)
            purity_ss + s_ss_p / 1000 >= purity_target,   recovery_ss + s_ss_r / 100 >= recovery_target
        """
        # Why (soft outputs, hard inputs): output floors with slacks keep the QP feasible after any
        # upset, and handle bounds stay hard because a pump can't exceed them. Standard in industrial
        # MPC (Qin and Badgwell's survey).
        # Judgement call (quadratic-only slack cost): with no linear term on the slacks a floor
        # gives way by a little whenever another term pulls. A linear slack term next to the
        # quadratic one, compared on the feed switch, would settle it.
        Hp, Hu, nu, N, n_ic = self.Hp, self.Hu, self.nu, self.circuit.N, self._n_ic
        pur, rec, zrows = self._pur, self._rec, self._zrows
        n_zs, n_iface = 2 * n_ic * N + 2, n_ic * N
        A_pur = sp.hstack([_csc_full(M[pur]), sp.eye(Hp) / 1e3, _zeros(Hp, 2 * Hp + n_zs)])
        A_pt = sp.hstack([_csc_full(M[pur]), _zeros(Hp, Hp), sp.eye(Hp) / 1e3, _zeros(Hp, Hp + n_zs)])
        A_rec = sp.hstack([_csc_full(M[rec]), _zeros(Hp, 2 * Hp), sp.eye(Hp) / 1e2, _zeros(Hp, n_zs)])
        A_lo = sp.hstack([_csc_full(M[zrows]), _zeros(n_iface, 3 * Hp), sp.eye(n_iface) / 100.0,
                          _zeros(n_iface, n_iface + 2)])
        A_hi = sp.hstack([_csc_full(M[zrows]), _zeros(n_iface, 3 * Hp + n_iface), -sp.eye(n_iface) / 100.0,
                          _zeros(n_iface, 2)])
        A_u = sp.hstack([_csc_full(self._T[: Hu * nu]), _zeros(Hu * nu, self.nw - self._n_du)])
        A_ss = sp.hstack([_csc_full(G_end), _zeros(2, 3 * Hp + 2 * n_iface), sp.diags([1e-3, 1e-2])])
        blocks = [A_pur, A_pt, A_rec, A_lo, A_hi, A_u, A_ss]
        if self._n_as:
            # acid_k + s_acid / 1000 >= acid_floor (slack in mmol/L); every other block gets zero
            # columns for the acid slacks (A_u already spans all of w)
            n_as = self._n_as
            blocks = [b if b is A_u else sp.hstack([b, _zeros(b.shape[0], n_as)]) for b in blocks]
            blocks.append(sp.hstack([_csc_full(M[self._arows]), _zeros(n_as, 3 * Hp + 2 * n_iface + 2),
                                     sp.eye(n_as) / 1e3]))
        return sp.vstack(blocks).tocsc()

    def _constraint_bounds(self, Y0: np.ndarray, y_ss: np.ndarray) -> tuple:
        """Lower and upper bounds for the rows of _constraint_matrix, in the same order. Floors
        have no upper bound; the interface's high side has no lower bound. For the handles,
        u_min - u_prev <= T_k dU <= u_max - u_prev."""
        # Why (targets as floors): zone or range control on controlled variables is the usual form in
        # industrial MPC (Qin and Badgwell's survey). Purity above target is good product, so it isn't
        # pushed back down; only the reagent term pulls the other way.
        Hp, Hu, n_iface = self.Hp, self.Hu, self._n_ic * self.circuit.N
        pur, rec, zrows = self._pur, self._rec, self._zrows
        lba = np.concatenate([self.spec - Y0[pur], self.purity_target - Y0[pur],
                              self.recovery_target - Y0[rec],
                              self.z_lo - Y0[zrows], np.full(n_iface, -np.inf),
                              np.tile(self._lb_u, Hu) - np.tile(self.u_prev, Hu),
                              [self.purity_target - y_ss[0], self.recovery_target - y_ss[1]]])
        uba = np.concatenate([np.full(3 * Hp, np.inf), np.full(n_iface, np.inf), self.z_hi - Y0[zrows],
                              np.tile(self._ub_u, Hu) - np.tile(self.u_prev, Hu), [np.inf, np.inf]])
        if self._n_as:
            lba = np.concatenate([lba, self.s.acid_floor - Y0[self._arows]])
            uba = np.concatenate([uba, np.full(self._n_as, np.inf)])
        return lba, uba

    def _steady_state_rows(self, lin_point: dict):
        """Purity and recovery at steady state, linear in the moves: y_ss + G_end @ du.

            y_ss  = y_ss0 + G_ss (u_prev - u_c) + bias       (purity and recovery only)
            G_end = G_ss T_end,   T_end = the last block of T, so u_end - u_prev = T_end dU

        With steady_state_floors off, the rows are zero and their floors are -inf (never active).
        """
        L = lin_point
        if not self.s.steady_state_floors:
            return np.zeros((2, self._n_du)), np.array([np.inf, np.inf])
        T_end = self._T[-self.nu:]                                      # u_end - u_prev = T_end @ du
        y_ss = L["y_ss0"][:2] + L["G_ss"][:2] @ (self.u_prev - L["u_c_h"]) + self.bias[:2]
        return L["G_ss"][:2] @ T_end, y_ss

    # ---- advance --------------------------------------------------------------------------------
    def _advance_model_state(self, d: np.ndarray):
        """The model state moves on under what is now applied: the nonlinear model, integrated by
        implicit Euler with the model's own Jacobian in a fixed number of sub-steps (six, in
        simulate.implicit_euler). The adaptive integrator was used here first; after a pump stop it
        could crawl for minutes in its retry ladder, and a controller must come back on time. One
        step of the linearization instead was not accurate enough over a run (the model state
        drifted off the model)."""
        # Why: measured, decision 23 (4). The adaptive integrator could take minutes after a pump
        # stop; a fixed cost per step is what a controller on a clock needs.
        x_m = implicit_euler(self.circuit, self.x_m, self._u_full(d), self.s.t_step)
        if np.all(np.isfinite(x_m)):
            self.x_m = x_m
        else:
            self.failures += 1                   # the model state stays; the bias will carry the difference

    def resync(self):
        """Start again from the model's rest point for the handles held now and the feed last
        measured, with no output bias, re-linearizing on the next step. Used when the controller
        takes over again after something else held the plant (interlock.py, deployment.py).

        Why the inputs come from _u_full: it writes in the measured feed and, with reflux, ties the
        strip flow to the scrub flow. Building them from the operating point instead (as the first
        re-arm did) put the model on the design feed and the design strip flow; after a feed
        switch the next QP couldn't be solved (notebook 36's watchdog test).
        """
        from .simulate import steady_state
        self.x_m = steady_state(self.circuit, self._u_full(self._d_last), x0=self.x_m)
        self.bias = np.zeros_like(self.bias)
        self._lin_point = None
        self._k = 0                                     # re-linearize on the next step

    def handles(self) -> dict:
        """The handle values the controller is holding now, by name: a float for each plant-wide
        handle, an array of N for "H_jackleg"."""
        out, i = {}, 0
        for name, k in zip(self._u_names, self._u_sizes):
            out[name] = self.u_prev[i:i + k].copy() if k > 1 else float(self.u_prev[i])
            i += k
        return out


def _succeeded(stats: dict) -> bool:
    """Whether a CasADi QP solver reported success (a solver that doesn't say counts as success)."""
    return bool(stats.get("success", True))


def _clip_purity_recovery(y: np.ndarray):
    """Clip purity (y[0]) and recovery (y[1]) in place to what they can be (PURITY_RANGE,
    RECOVERY_RANGE)."""
    y[0] = float(np.clip(y[0], *PURITY_RANGE))
    y[1] = float(np.clip(y[1], *RECOVERY_RANGE))
