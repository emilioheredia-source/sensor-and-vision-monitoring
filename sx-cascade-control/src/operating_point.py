"""The starting point of every study: the plant commissioned and trimmed onto spec.

A plant is not run on the handles a model designed. It is started on them (commission), then
the controller, with full information, trims it until the product is on spec and the handles
stop moving (trim_under_control). Where the handles end is the plant's operating point, and
every scenario starts there. The controller keeps the nominal model throughout; the plant may
differ from it (mismatch.py, or any plant in the registry). (ARCHITECTURE.md decision 21,
APPENDIX_B_control.md B.9.)

Two circuits are carried side by side:

  model   what the controller and the estimator plan with: nominal parameters, every stage
          at nominal efficiency (k_factor = 1).
  plant   what they drive. It can have other parameters, its own stage efficiencies
          (plant_k), and even a longer state; plant.observable(x) gives the part in the
          model's layout.

prepare() does the whole sequence and returns an OperatingPoint. controller_settings() gives
the controller settings a run file implies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuit import Circuit, Inputs
from .commission import commission, trim_under_control
from .config import RunConfig, build_circuit, nominal_inputs
from .mpc import CircuitMPC, MPCSettings
from .simulate import steady_state


# --------------------------------------------------------------------------
# The operating point
# --------------------------------------------------------------------------
@dataclass
class OperatingPoint:
    """Everything a study needs to start: the two circuits, the plant at rest, and the inputs.

    Usage:
        op = prepare(rc, plant, plant_k)
        ctl = op.controller()                     # a fresh controller at this point
        traj = simulate(op.plant, op.x_plant, lambda t: op.u_plant, 20.0)
    """
    rc: RunConfig
    model: Circuit                # the controller's model
    plant: Circuit                # what it drives
    plant_k: np.ndarray           # the plant's own stage efficiencies (multipliers), unknown to the controller
    x_plant: np.ndarray           # the plant at rest at the operating point
    u_model: Inputs               # the operating point's inputs as the controller knows them
                                  # (nominal efficiencies)
    x_model: np.ndarray           # the model's own rest point under those inputs
    u_design: Inputs              # the model's design handles, before the trim

    @property
    def u_plant(self) -> Inputs:
        """The operating point's inputs as the plant really gets them (its own efficiencies)."""
        return self.u_model.replace(k_factor=self.plant_k)

    @property
    def spec(self) -> float:
        """The product purity spec, as a fraction (from the run file's targets)."""
        return self.rc.targets()[0]

    @property
    def recovery_target(self) -> float:
        """The recovery floor, as a fraction (from the run file's targets)."""
        return self.rc.targets()[1]

    def controller(self, settings: MPCSettings | None = None, level_loops=None) -> CircuitMPC:
        """A fresh controller at this operating point (settings: None = controller_settings(rc);
        level_loops: the settler loops under it, or None if it moves the jacklegs itself)."""
        rc = self.rc
        # Why (two purity floors): the hard-weighted spec (99 %) and a soft target at the design
        # purity (99.5 %), half a point above spec, as plants run (decision 15). REPORT.md 5.1 hints
        # the margin matters: the matched plant, started with none, put 215 mol off spec on the
        # feed switch; the trimmed perturbed plant, at 99.77 %, none. Different plants, so a hint.
        return CircuitMPC(self.model, self.x_model, self.u_model, rc.cut, self.spec, rc.design_purity,
                          self.recovery_target, settings or controller_settings(rc), level_loops=level_loops)


# --------------------------------------------------------------------------
# Controller settings from the run file
# --------------------------------------------------------------------------
def controller_settings(rc: RunConfig, **fields) -> MPCSettings:
    """Controller settings for this run file: the strip flow tied to the scrub flow when the scrub
    is a reflux, the organic flow inside the pump's range; any MPCSettings field may be given.

    Parameters:
        rc: the run file.
        **fields: MPCSettings fields to override. A dict for a dict field (bounds, for instance)
            changes only the keys it names; a list given where the setting holds tuples
            becomes a tuple.
    Returns:
        the MPCSettings.
    Raises:
        KeyError if a field name isn't an MPCSettings field.

    Usage:
        settings = controller_settings(rc, bounds={"Q_scrub": [100.0, 900.0]})
    """
    design = rc.design
    # with a reflux scrub: Q_strip = net product flow + Q_scrub (L/h); the strip flow moves with the scrub flow
    settings = MPCSettings(strip_tied_to_scrub=design["net_product_L_per_h"] if rc.reflux else None)
    # Why: decision 19, measured. The design sits at 85 % of the pump maximum and the controller
    # gets the whole range, so on a stronger feed it can add organic instead of doubling the scrub.
    settings.bounds["Q_org"] = (design["organic_min_L_per_h"], design["pump_max_L_per_h"])     # L/h
    # the run file's own [controller_settings] first, then this run's edits on top
    for key, value in {**rc.cfg.get("controller_settings", {}), **fields}.items():
        if not hasattr(settings, key):
            raise KeyError(f"MPCSettings has no field {key!r}")
        setattr(settings, key, _merged_setting(getattr(settings, key), value))
    return settings


def _merged_setting(current, value):
    """The new value for one setting. An edit to one bound keeps the others; lists become tuples.

    Why: run files are TOML, which has arrays but no tuples.
    """
    if isinstance(current, dict) and isinstance(value, dict):     # an edit to one bound keeps the others
        return {**current, **_lists_as_tuples(value)}
    return _tuple_if_expected(current, value)


def _lists_as_tuples(entries: dict) -> dict:
    """A copy of the dict with every list value turned into a tuple (bounds are held as tuples)."""
    return {k: tuple(v) if isinstance(v, list) else v for k, v in entries.items()}


def _tuple_if_expected(current, value):
    """value as a tuple when the setting holds a tuple and a list was given; else value as is."""
    return tuple(value) if isinstance(current, tuple) and isinstance(value, list) else value


# --------------------------------------------------------------------------
# prepare: commission, trim, and the model's rest point
# --------------------------------------------------------------------------
def prepare(rc: RunConfig, plant: Circuit, plant_k: np.ndarray, trim_hours: float = 30.0) -> OperatingPoint:
    """Commission `plant` on the run file's design handles, then trim it under control.
    plant / plant_k: the plant and its stage efficiencies (config.build_plant, or a registry plant).

    The steps:
      1. Commission the plant on the design handles, with its own efficiencies: x_rest, u_design.
      2. Find the model's own rest point on the same handles at nominal efficiency, starting from
         the plant's state (in the model's layout).
      3. Build a controller there and let it trim the plant for trim_hours (hours): x_plant, u_model.
      4. Find the model's rest point again under the trimmed inputs: x_model. The controller of
         every study starts from this point.

    Parameters:
        rc: the run file.
        plant: the plant circuit.
        plant_k: the plant's stage efficiencies, one multiplier per stage.
        trim_hours: how long the trim runs, hours.
    Returns:
        the OperatingPoint.
    """
    model = build_circuit(rc)
    nominal_k = np.ones(model.N)                    # the model's stage efficiencies: all nominal
    # 1. start-up on the design handles
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    # 2. the model at rest on the same handles
    x_model = steady_state(model, u_design.replace(k_factor=nominal_k), x0=plant.observable(x_rest))
    # 3. the trim, by a controller built at the model's rest point
    # Why: decision 21 (2). A plant is started on the design handles and trimmed onto spec, and
    # every scenario, open loop included, starts from the same trimmed point, so the comparison
    # is fair. The extra reagent it takes is what the model's error costs.
    # Judgement call: 30 h with full information, then stop. "Handles have stopped moving" is not
    # checked; on the perturbed plant purity settled within 10 h. Logging handle movement over the
    # last hours of the trim, for every plant, would confirm it each time.
    trim = CircuitMPC(model, x_model, u_design.replace(k_factor=nominal_k), rc.cut, rc.targets()[0],
                      rc.design_purity, rc.targets()[1], controller_settings(rc))
    x_plant, u_model = trim_under_control(plant, x_rest, plant_k, trim, t_settle=trim_hours)
    # 4. the model at rest on the trimmed handles
    x_model = steady_state(model, u_model, x0=plant.observable(x_plant))
    return OperatingPoint(rc, model, plant, np.asarray(plant_k, float), x_plant, u_model, x_model, u_design)
