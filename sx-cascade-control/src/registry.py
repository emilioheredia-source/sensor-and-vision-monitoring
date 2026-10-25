"""The named things a run is assembled from: plants, controllers and sensor sets.

A RunSpec (runs.py) picks one of each by name. This file is where the names live. Each one is
an `Entry` with a stable key (what a RunSpec stores), a short title, a paragraph explaining
what it is, and a `build` function that makes it. A front end can list the entries and say
what each one is without knowing anything else about it, and adding a plant or a controller
means adding one entry here (ARCHITECTURE.md, decision 24). The upsets are named the same way,
in scenarios.py.

Plants, simplest first:
  matched     the controller's own model (the reference: a controller that knows the plant exactly)
  perturbed   the same equations with different numbers, drawn from a seed (mismatch.py, decision 20)
  settler     a mechanism the model lacks: transfer in a small mixer only, then a settler the
              phases flow through close to plug flow (plant_models.py, decision 25)
  entrainment another mechanism the model lacks: droplets of each phase carried into the next
              stage, more of them when an interface strays toward a weir (plant_models.py, decision 25)
  drifting    the perturbed plant whose extraction constants then drift over days (plant_models.py)

Controllers, simplest first:
  open_loop               handles held where they are
  mpc_fixed               the MPC linearized once at the operating point, no output bias
  mpc                     the MPC as built: re-linearized on its own model state, output bias,
                          steady-state floors (mpc.py, decisions 16, 18 and 21)
  open_loop_level_loops   handles held, but a PI level loop on every settler moves its jackleg
  mpc_level_loops         the MPC without the jacklegs, over those level loops (a cascade)
  mpc_interlock           the MPC over level loops with a feed-loss interlock (interlock.py)
  mpc_interlock_base      the same, saponification stopped while the feed is off
  mpc_shadow              mpc_interlock in shadow mode: it advises, nothing reaches the plant (deployment.py)
  mpc_advisory            its advice applied by an operator every 2 h, inside move limits
  mpc_guarded             it moves the handles itself, inside move limits, with a watchdog
  two_point               no model of the split: scrub and organic each hold one composition (two_point.py)

Sensor sets go from full_state (the controller sees the true state, the reference) down to
nothing (no readings at all, the estimator runs the model alone).

Each table has its own build signature:
  PLANTS:       build(rc) -> (plant circuit, its stage-efficiency multipliers)
  CONTROLLERS:  build(op, **settings) -> a controller that satisfies loop.Controller
  SENSOR_SETS:  build(circuit, rc) -> a list of sensors.Sensor, or None for the full state

Usage:

    plant, plant_k = PLANTS["perturbed"].build(rc)
    for key, title, explanation in options(SENSOR_SETS): print(key, title)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .circuit import Circuit, Inputs
from .config import RunConfig, build_circuit
from .mismatch import Mismatch, perturbed_params, stage_efficiency_factors
from .plant_models import EntrainmentCircuit, SettlerCircuit
from .sensors import Sensor


@dataclass(frozen=True)
class Entry:
    """One named choice in a table.

    Attributes:
        key: the stable name a RunSpec stores, such as "perturbed".
        title: a short title for a list or a figure legend.
        explanation: a paragraph saying what it is, for anyone choosing it.
        build: the function that makes it (the signature depends on the table; see above).
    """
    key: str
    title: str
    explanation: str
    build: Callable


# ==============================================================================================
# Plants: build(rc) -> (plant circuit, its stage efficiency multipliers)
# ==============================================================================================
# The plant is what the controller drives. The controller always plans on the nominal model
# (config.build_circuit); the plant may differ from it in its numbers or in its physics.
#
# The second thing each builder returns is one multiplier per stage on the stage's transfer
# rate. The controller doesn't know these (decision 20). Only the perturbed plant has
# multipliers other than 1.

def _matched(rc: RunConfig):
    """The plant is the controller's own model."""
    plant = build_circuit(rc)
    return plant, np.ones(plant.N)


def _perturbed(rc: RunConfig):
    """The model's equations with numbers drawn from the run file's [plant] seed and strength."""
    mismatch = Mismatch(seed=int(rc["plant"]["seed"]), strength=float(rc["plant"]["strength"]),
                        generator=str(rc["plant"]["generator"]))
    model = build_circuit(rc)
    plant = Circuit(model.cmap, perturbed_params(rc.params, mismatch), model=model.model,
                    scrub_reflux=model.scrub_reflux)
    return plant, stage_efficiency_factors(model.N, mismatch)


def _drifting(rc: RunConfig):
    """The perturbed plant (same numbers as "perturbed" for this seed), whose chemistry then drifts
    over days: every extraction constant together +-10 %, La+Ce against the rest +-3 %."""
    from .plant_models import DriftingCircuit
    mismatch = Mismatch(seed=int(rc["plant"]["seed"]), strength=float(rc["plant"]["strength"]),
                        generator=str(rc["plant"]["generator"]))
    model = build_circuit(rc)
    plant = DriftingCircuit(model.cmap, perturbed_params(rc.params, mismatch), model=model.model,
                            scrub_reflux=model.scrub_reflux, seed=int(rc["plant"]["seed"]),
                            split_amp=float(rc["plant"]["drift_split"]),
                            trend_per_week=float(rc["plant"]["drift_trend"]))
    return plant, stage_efficiency_factors(model.N, mismatch)


def _book(rc: RunConfig):
    """The drifting perturbed plant with four mechanisms from the book (plant_models.BookCircuit):
    separation weaker at trace ends and in the scrub, a weaker acid dependence, slower transfer in
    acid. The perturbed numbers' own acid-exponent shift is replaced by the book's."""
    from .plant_models import BookCircuit
    mismatch = Mismatch(seed=int(rc["plant"]["seed"]), strength=float(rc["plant"]["strength"]),
                        generator=str(rc["plant"]["generator"]), dn_acid=0.0)
    model = build_circuit(rc)
    plant = BookCircuit(model.cmap, perturbed_params(rc.params, mismatch), model=model.model,
                        scrub_reflux=model.scrub_reflux, seed=int(rc["plant"]["seed"]),
                        split_amp=float(rc["plant"]["drift_split"]),
                            trend_per_week=float(rc["plant"]["drift_trend"]))
    return plant, stage_efficiency_factors(model.N, mismatch)


def _settler(rc: RunConfig):
    """Mixer then settler cells per stage; the number of cells comes from the run file's [plant]."""
    model = build_circuit(rc)
    plant = SettlerCircuit(model.cmap, model.p, model=model.model, scrub_reflux=model.scrub_reflux,
                           cells=int(rc["plant"]["settler_cells"]))
    return plant, np.ones(plant.N)


def _entrainment(rc: RunConfig):
    """Each phase carries droplets of the other into the next stage."""
    model = build_circuit(rc)
    plant = EntrainmentCircuit(model.cmap, model.p, model=model.model, scrub_reflux=model.scrub_reflux)
    return plant, np.ones(plant.N)


PLANTS = {e.key: e for e in [
    Entry("matched", "Matched: the controller's own model",
          "The plant is the model the controller plans on, upsets and sensors apart. A controller that "
          "knows the plant exactly always looks good; this is the reference, not a test.", _matched),
    Entry("perturbed", "Perturbed: same equations, different numbers",
          "Extraction constants off by a random factor with a 10 % spread (20 % for La+Ce), the acid "
          "exponent 3.3 instead of 3, each stage's transfer rate between 85 and 100 % of nominal, densities "
          "and hold-up off by a few percent, drawn from the run "
          "file's [plant] seed. Tests the controller's tuning; it still shares the model's laws.", _perturbed),
    Entry("drifting", "Drifting: the perturbed plant, with chemistry that moves over days",
          "The perturbed plant for this seed, with every distribution coefficient then drifting: all "
          "together by up to about 10 % (extractant strength, temperature) and La+Ce against the rest by "
          "up to 3 % (the separation factor itself), as slow waves with periods of one to seven days. The "
          "controller's model stays put while the plant moves under it.", _drifting),
    Entry("book", "Book: closer to a real plant (Qi 2018)",
          "The drifting perturbed plant with four mechanisms from the book: the La+Ce/Pr separation "
          "factor about 20 % weaker toward the cascade ends and a little weaker in the scrub, a weaker "
          "acid dependence (slope 2.5, not 3), and transfer up to eight times slower in acidic stages.",
          _book),
    Entry("settler", "Settler: dynamics the model does not have",
          "Every stage is a mixer, where the metal moves between the phases, followed by a settler four "
          "times its size that the phases only flow through, as three tanks in a row per phase. The model "
          "lumps mixer and settler into one well-mixed pool. Same numbers, same hold-up and same steady "
          "state; only the dynamics differ.", _settler),
    Entry("entrainment", "Entrainment: settlers that do not separate perfectly",
          "Each phase leaves its settler carrying droplets of the other, which ride into the next stage "
          "and come back with their own phase: metal pushed back the wrong way. 500 ppm normally, up to "
          "2 % when an interface goes a few centimetres past its window toward a weir. The model takes "
          "separation as perfect at any interface height.", _entrainment),
]}


# ==============================================================================================
# Controllers: build(op: OperatingPoint, **settings) -> a loop.Controller
# ==============================================================================================
# `op` is the operating point (operating_point.py): the model, the plant on spec and the
# handles there. `settings` are MPCSettings fields to change, from the RunSpec's
# controller_settings. Every builder takes the control interval (t_step) from those settings.

class HoldController:
    """Open loop: the operating point's handles, held. Satisfies loop.Controller.

    Args:
        u_nominal: the inputs at the operating point; apply() always returns these.
        t_step: the control interval, h.
        cut: the cut the circuit makes (such as "LaCe"); loop.Controller asks for it.
        level_loops: a PI loop on each settler (loop.LevelLoops), or None for none.
    """

    def __init__(self, u_nominal: Inputs, t_step: float, cut: str, level_loops=None):
        """Hold the operating point's handles (arguments in the class docstring)."""
        self.u_nominal, self._t_step, self.cut = u_nominal, t_step, cut
        self.level_loops = level_loops
        self.last = {}                   # nothing to record per interval

    @property
    def t_step(self) -> float:
        """The control interval, h."""
        return self._t_step

    def step(self, obs) -> dict:
        """Look at nothing and command nothing: no handles change."""
        return {}

    def apply(self, handles: dict) -> Inputs:
        """The operating point's inputs, whatever was commanded."""
        return self.u_nominal


def _open_loop(op, level_loops=None, **settings):
    """The handles held at the operating point. The settings only set the control interval."""
    from .operating_point import controller_settings
    return HoldController(op.u_model, controller_settings(op.rc, **settings).t_step, op.rc.cut, level_loops)


def _mpc(op, level_loops=None, **settings):
    """The MPC as built (mpc.py), with this run file's settings and any edits on top; level_loops,
    if given, are the settler loops under it."""
    from .operating_point import controller_settings
    return op.controller(controller_settings(op.rc, **settings), level_loops=level_loops)


def _mpc_fixed(op, **settings):
    """The MPC with re-linearization and the output bias turned off: the first version built.

    The run's own settings are written last, so they win over these two.
    """
    return _mpc(op, **{"relinearize_every": 0, "bias_filter": 0.0, **settings})


def _settler_loops(rc, jacklegs, settings):
    """A PI level loop on every settler (loop.LevelLoops), holding each interface at the setpoint.
    The one place they're built, from the run file's stage geometry.

    Args:
        rc: the run file (RunConfig); its stage geometry gives the setpoint and the time constant.
        jacklegs: the jackleg heights the loops start from, m (an operating point's, or a model's).
        settings: the controller settings (MPCSettings); their jackleg bounds limit the loops.
    """
    from .loop import LevelLoops
    geo = rc.params.raw["stage_geometry"]
    return LevelLoops(z_set=geo["interface_setpoint_fraction"] * geo["settler_height_m"],
                      H0=np.asarray(jacklegs, float).ravel(), tau_h=geo["jackleg_time_constant_h"],
                      bounds=settings.bounds["H_jackleg"])


def mpc_over_level_loops(rc, model, x_model, u_model, level_loops=None, **settings):
    """The main case's controller (mpc_level_loops) on a given model: for studies that rebuild
    the controller on a fitted model (notebooks 31 to 37). level_loops: the running loops, to keep
    them through the rebuild; None builds new ones at u_model's jacklegs.

    Why five parameters: the run file, the model's three pieces (model, rest state, inputs) and the running level loops to keep through a rebuild; settings come as keywords.
    """
    from .mpc import CircuitMPC
    from .operating_point import controller_settings
    s = controller_settings(rc, **{"jacklegs": False, "w_center": 0.0, "w_interface": 1e-4, **settings})
    if level_loops is None:
        level_loops = _settler_loops(rc, u_model.H_jackleg, s)
    return CircuitMPC(model, x_model, u_model, rc.cut, rc.targets()[0], rc.design_purity, rc.targets()[1], s,
                      level_loops=level_loops)


def _mpc_level_loops(op, **settings):
    """The MPC without the jacklegs, over a PI level loop on every settler (a cascade, as plants
    are built: local loops hold the interfaces, the MPC moves the flows and the acid).

    The interface terms of the MPC are switched off: w_center = 0, and w_interface down to 1e-4
    (kept a hair above zero so the slacks stay in a well-posed QP). The run's settings win."""
    from .operating_point import controller_settings
    loops = _settler_loops(op.rc, op.u_model.H_jackleg, controller_settings(op.rc, **settings))
    return _mpc(op, level_loops=loops,
                **{"jacklegs": False, "w_center": 0.0, "w_interface": 1e-4, **settings})


def _two_point(op, hold_a=("lace_frac", 26), hold_b=("ndpr_frac", 4), closed_loop_h=20.0, swap=False, **settings):
    """Two-point control (two_point.py): scrub holds hold_a, organic holds hold_b, each an integral
    loop, over a PI level loop on every settler. hold_a / hold_b: (kind, stage).

    Why five parameters: the registry's builder interface (the operating point) plus the two-point controller's keyword settings.
    """
    from .operating_point import controller_settings
    from .two_point import TwoPointController
    s = controller_settings(op.rc, **settings)
    net = s.strip_tied_to_scrub if s.strip_tied_to_scrub is not None else float(op.u_model.Q_strip - op.u_model.Q_scrub)
    return TwoPointController(op.model, op.x_model, op.u_model, op.rc.cut, tuple(hold_a), tuple(hold_b), s.t_step,
                              s.bounds, s.rate_per_step, net, closed_loop_h, swap,
                              level_loops=_settler_loops(op.rc, op.u_model.H_jackleg, s))


def _mpc_interlock(op, stop_base=False, **settings):
    """The MPC over level loops, wrapped in a feed-loss interlock (interlock.py): freeze the handles
    when the feed stops, re-sync and hand back two hours after it returns."""
    from .interlock import FeedLossInterlock
    return FeedLossInterlock(_mpc_level_loops(op, **settings), stop_base=stop_base)


def _mpc_interlock_base(op, **settings):
    """The same, and saponification stopped while the feed is off."""
    return _mpc_interlock(op, stop_base=True, **settings)


def _guarded(op, mode, guard_max_move=None, guard_max_offset=None, **settings):
    """mpc_interlock inside a DeploymentGuard (deployment.py) in the given mode. guard_max_move and
    guard_max_offset (fractions of the commissioned value) change the guard's limits; the other
    settings go to the controller."""
    from dataclasses import replace
    from .deployment import DeploymentGuard, Limits
    changes = {k: v for k, v in (("max_move", guard_max_move), ("max_offset", guard_max_offset)) if v is not None}
    return DeploymentGuard(_mpc_interlock(op, **settings), mode=mode, limits=replace(Limits(), **changes))


def _open_loop_level_loops(op, **settings):
    """No controller on the flows and acid, but a PI level loop on every settler."""
    from .operating_point import controller_settings
    loops = _settler_loops(op.rc, op.u_model.H_jackleg, controller_settings(op.rc, **settings))
    return _open_loop(op, level_loops=loops, **settings)


CONTROLLERS = {e.key: e for e in [
    Entry("open_loop", "Open loop: nobody touches anything",
          "The handles stay at the operating point. What doing nothing costs.", _open_loop),
    Entry("mpc_fixed", "MPC, linearized once, no bias",
          "The first controller built: one linearization at the operating point, no output bias. "
          "Kept to show what the later fixes buy.", _mpc_fixed),
    Entry("mpc", "MPC as built",
          "Keeps its own model state, takes what the plant reports as an output bias (integral "
          "action), re-linearizes every half hour, holds purity and recovery floors over a 12 h "
          "horizon and at steady state, keeps every interface in its window.", _mpc),
    Entry("open_loop_level_loops", "Open loop, local level loops",
          "The flows and acid stay at the operating point; a PI loop on each settler moves its "
          "jackleg to hold the interface. A plant with level control and nothing above it.",
          _open_loop_level_loops),
    Entry("mpc_level_loops", "MPC over local level loops",
          "The MPC as built, but without the jacklegs: a PI loop on each settler holds its "
          "interface, and the MPC moves only the flows and the acid (a cascade).", _mpc_level_loops),
    Entry("mpc_interlock", "MPC over level loops, with a feed-loss interlock",
          "When the feed flow drops below 20 % of nominal the handles freeze; two hours after it's back "
          "the controller is re-synced to the model's rest point and takes over again.", _mpc_interlock),
    Entry("mpc_interlock_base", "The same, saponification stopped while the feed is off",
          "As mpc_interlock, and the base dose goes to zero while tripped: with no feed there's no acid "
          "coming in to neutralise.", _mpc_interlock_base),
    Entry("mpc_shadow", "MPC in shadow mode: advice only",
          "The MPC with the interlock runs on the live readings and says what it would do; nothing it "
          "says reaches the plant, which runs as if no controller were there. The first step of a "
          "deployment.", lambda op, **s: _guarded(op, "shadow", **s)),
    Entry("mpc_advisory", "MPC in advisory mode: an operator applies it every 2 h",
          "Its advice is applied at each operator review, every 2 h, inside the move limits (2 % of "
          "the commissioned value per interval, 20 % in all).", lambda op, **s: _guarded(op, "advisory", **s)),
    Entry("mpc_guarded", "MPC closed loop, inside limits, with a watchdog",
          "It moves the handles itself inside the same limits; if it fails three solves in a row or "
          "raises, the handles are held and it stays out until re-armed. The level loops run "
          "throughout.", lambda op, **s: _guarded(op, "closed", **s)),
    Entry("two_point", "Two-point control over local level loops",
          "No model of the split: the scrub flow holds one composition inside the cascade and the "
          "organic flow holds another, each with an integral loop, as distillation columns are "
          "often run; PI level loops hold the interfaces.", _two_point),
]}


# ==============================================================================================
# Sensor sets: build(circuit, rc) -> list of Sensors, or None for the full state
# ==============================================================================================
# A sensor set is a list of instruments (sensors.py). None is different from an empty list:
# None means no instruments and no estimator, the controller is handed the true state; an
# empty list means an estimator with nothing to correct it.
#
# Stages are counted from 0 at the raffinate end, so stage 0 is where the raffinate leaves.
# An "analyser" here is an on-stream composition analyser on a stage's aqueous outlet; it reads
# every component (decision 23).

def _ecjtu(rc: RunConfig) -> tuple[int, int]:
    """One analyser mid-extraction and one mid-scrub, the layout of the ECJTU patent.

    Returns:
        The two stage numbers: (middle of extraction, middle of scrub).
    """
    n_ext, n_scrub = rc["circuit"]["n_ext"], rc["circuit"]["n_scrub"]
    return n_ext // 2, n_ext + n_scrub // 2


def _analysers(*stages) -> list:
    """An aqueous composition analyser on each of the given stages."""
    return [Sensor("composition_aq", stage=j) for j in stages]


def _full_state(circuit, rc):
    """No instruments: the controller is handed the true state."""
    return None


def _product_raffinate(circuit, rc):
    """Analysers on the product stage and on stage 0, where the raffinate leaves."""
    return _analysers(circuit.cmap.product_stage, 0)


def _product_raffinate_feed(circuit, rc):
    """The product and raffinate analysers, plus an analyser on the feed."""
    return _analysers(circuit.cmap.product_stage, 0) + [Sensor("feed_analyser")]


def _ecjtu_feed(circuit, rc):
    """The ECJTU pair, plus an analyser on the feed."""
    return _analysers(*_ecjtu(rc)) + [Sensor("feed_analyser")]


def _ecjtu_only(circuit, rc):
    """The ECJTU pair alone. (The key is "ecjtu"; the name _ecjtu is taken by the stage picker.)"""
    return _analysers(*_ecjtu(rc))


def _pair_interfaces(circuit, rc):
    """The ECJTU pair, a feed analyser, and an interface (level) reading on every settler."""
    return (_analysers(*_ecjtu(rc)) + [Sensor("feed_analyser")]
            + [Sensor("interface", stage=j) for j in range(circuit.N)])


def _lab_assay(circuit, rc):
    """A lab assay of the product only."""
    return [Sensor("product_assay")]


def _nothing(circuit, rc):
    """An estimator with no readings at all."""
    return []


SENSOR_SETS = {e.key: e for e in [
    Entry("full_state", "Full state (no instruments)",
          "The controller sees the true state of every stage. The reference.", _full_state),
    Entry("product_raffinate", "Product + raffinate analysers",
          "On-stream analysers on the two streams that leave the circuit.", _product_raffinate),
    Entry("product_raffinate_feed", "Product + raffinate + feed analysers",
          "The same, plus an analyser on the feed for feedforward.", _product_raffinate_feed),
    Entry("ecjtu_feed", "Patent pair + feed analyser",
          "One analyser mid-extraction, one mid-scrub (the ECJTU patent's layout), and a feed analyser.",
          _ecjtu_feed),
    Entry("ecjtu", "Patent pair",
          "One analyser mid-extraction, one mid-scrub, nothing on the feed.", _ecjtu_only),
    Entry("pair_interfaces", "Patent pair + feed analyser + every interface",
          "The patent pair and a feed analyser, plus a level reading on every settler.", _pair_interfaces),
    Entry("lab_assay", "Lab assay only",
          "A sample of the product every two hours, back four hours later.", _lab_assay),
    Entry("nothing", "No readings",
          "The estimator runs the model and nothing corrects it.", _nothing),
]}


# ==============================================================================================
# Listing
# ==============================================================================================

def options(table: dict) -> list[tuple[str, str, str]]:
    """(key, title, explanation) for every entry of PLANTS, CONTROLLERS or SENSOR_SETS, in order.

    This is what a front end lists, and what `python -m src.cli list` prints.
    """
    return [(e.key, e.title, e.explanation) for e in table.values()]
