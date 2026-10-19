"""What the controller is allowed to see.

A Sensor is a quantity at a location, with a noise, a delay and a sample
interval. A SensorSet does two jobs with a list of them:

  - For the simulation, it turns the plant's state history into readings:
    for each sensor that has a sample due, take the plant state at
    (now - delay), map it to what the instrument reads, and add noise.
  - For the estimator, it gives the matrix that maps a state deviation to a
    reading deviation (rows), the reading's fixed offset, and its noise.

A reading of sensor m at time t is (APPENDIX_B_control.md B.4):

    y_m(t) = C_m @ x(t - delay_m) + offset_m + noise,   noise ~ N(0, sd_m^2)

and it only exists on the sensor's sample ticks.

Kinds (where they sit and what they read):
  composition_aq   aqueous concentration of chosen components at a stage (on-stream analyser)
  composition_org  organic concentration at a stage
  acid             aqueous H+ at a stage (pH probe)
  density_aq / density_org   phase density at a stage (vibrating-tube meter)
  interface        interface height at a stage (the vision project)
  product_assay    aqueous composition at the product stage, lab-style: slow and delayed
  feed_analyser    the feed composition (what the MPC uses as feedforward)
Flows are not listed: the controller commands them and a meter would read the
command back. A flowmeter bias is exactly the thing a meter cannot see.
(ARCHITECTURE.md decision 17: sensors are data. Noise, delay and interval
per kind are chosen and labelled in parameters.toml [sensors].)

State layout (stages numbered from 0 at the raffinate end, N stages, n components):

    x = [ c_aq (N*n) | c_org (N*n) | h (N) | z (N) ]
          mol/L        mol/L         mol/L   m

A richer plant can carry a longer state; it starts with this layout, and
`plant.observable(x)` gives the part the instruments see.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .circuit import Circuit
from .params import Params

# A reading's noise is never below this, in the reading's units (chosen; APPENDIX_B B.4).
# Why: nothing reads exactly zero, and the estimator needs R without zeros on its diagonal.
MIN_NOISE_SD = 1e-4
# Judgement call (the instrument model as a whole): white Gaussian noise, a fixed delay and a fixed
# interval, with numbers chosen from typical instruments (parameters.toml [sensors]; ARCHITECTURE.md,
# Open). Real analysers also drift, lose calibration, give outliers and go offline. A pass with
# vendor datasheets, and runs with a slow analyser bias and dropped readings, would settle it.
# Where the feed composition sits in the input vector: after Q_org, Q_feed, Q_scrub, Q_strip.
FEED_COMPOSITION_START = 4


# --------------------------------------------------------------------------
# One instrument
# --------------------------------------------------------------------------
@dataclass
class Sensor:
    """One instrument: what it reads, where, and how well.

    Fields:
        kind: one of the kinds in the module docstring.
        stage: the stage it sits on (0 = raffinate end). None for the feed analyser;
            set automatically for product_assay.
        components: component names a composition sensor reads; None = all of them.
        noise: one standard deviation, relative (fraction of the reading) or absolute
            (g/mL for density, m for interface) depending on the kind. None = the kind's default.
        delay_h: hours between taking the sample and having the result. 0 = the kind's default.
        interval_h: hours between samples. 0 = the kind's default (which may be every step).
        name: an optional label; by default "kind@stage".
    """
    kind: str
    stage: int | None = None
    # components: for composition kinds; None = every component the circuit carries
    # (an XRF or ICP reads them all; the impurity is what purity is about)
    # Why: decision 23 (1). The Pr/Nd-only default was a leftover, and purity is about the impurity.
    components: tuple | None = None
    noise: float | None = None              # None: the default for the kind, from parameters.toml
    delay_h: float = 0.0
    interval_h: float = 0.0                 # 0: every control step
    name: str = ""

    def label(self) -> str:
        """The name readings are filed under: `name` if given, else "kind@stage" (or just "kind")."""
        return self.name or (f"{self.kind}@{self.stage}" if self.stage is not None else self.kind)


def defaults_from(p: Params) -> dict:
    """Default noise, delay and sample interval per sensor kind, from parameters.toml [sensors].

    Returns {kind: dict(noise, relative, delay_h, interval_h)}. relative=True means the noise is a
    fraction of the reading; False means it is in the reading's own units (g/mL, m).
    """
    s = p.raw["sensors"]
    return {
        "composition_aq": dict(noise=s["onstream_noise_relative"], relative=True, delay_h=s["onstream_delay_h"],
                               interval_h=s["onstream_interval_h"]),
        "composition_org": dict(noise=s["onstream_noise_relative"], relative=True,
                                delay_h=s["onstream_delay_h"], interval_h=s["onstream_interval_h"]),
        "acid": dict(noise=s["acid_noise_relative"], relative=True, delay_h=0.0, interval_h=0.0),
        "density_aq": dict(noise=s["density_noise_g_per_mL"], relative=False, delay_h=0.0, interval_h=0.0),
        "density_org": dict(noise=s["density_noise_g_per_mL"], relative=False, delay_h=0.0, interval_h=0.0),
        "interface": dict(noise=s["interface_noise_m"], relative=False, delay_h=0.0, interval_h=0.0),
        "product_assay": dict(noise=s["lab_noise_relative"], relative=True, delay_h=s["lab_delay_h"],
                              interval_h=s["lab_interval_h"]),
        "feed_analyser": dict(noise=s["onstream_noise_relative"], relative=True, delay_h=s["onstream_delay_h"],
                              interval_h=s["onstream_interval_h"]),
    }


def _default_if_zero(given: float, default: float) -> float:
    """A sensor's delay or interval: the kind's default when the sensor left it at 0 (the field's
    "not given" value), else the sensor's own."""
    return default if given == 0.0 and default else given


# --------------------------------------------------------------------------
# A set of instruments on one circuit
# --------------------------------------------------------------------------
@dataclass
class SensorSet:
    """The instruments on a circuit, and the random-number stream for their noise.

    Fields:
        sensors: list of Sensor. Their missing settings are filled in from the kind's defaults.
        circuit: the circuit (the model layout) the sensors read.
        seed: seed for the noise; the same seed gives the same noise draws.

    Usage:
        ss = SensorSet([Sensor("composition_aq", stage=7), Sensor("interface", stage=3)], circuit)
        readings = ss.readings(t, history)        # {label: (value, t_measured)}
    """
    sensors: list
    circuit: Circuit
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        """Start the noise stream and fill in each sensor's missing settings."""
        self._rng = np.random.default_rng(self.seed)
        self._defaults = defaults_from(self.circuit.p)
        for s in self.sensors:
            self._fill_defaults(s)

    def _fill_defaults(self, s: Sensor):
        """Fill in what the sensor left open from its kind's defaults (the Sensor is changed in place).
        A zero delay or interval counts as "not given". A product assay always sits on the product stage."""
        d = self._defaults[s.kind]
        if s.components is None:
            s.components = tuple(self.circuit.p.names)
        if s.noise is None:
            s.noise = d["noise"]
        s.delay_h = _default_if_zero(s.delay_h, d["delay_h"])
        s.interval_h = _default_if_zero(s.interval_h, d["interval_h"])
        if s.kind == "product_assay":
            s.stage = self.circuit.cmap.product_stage

    # ======================================================================
    # Rows of the linear measurement map, for state-based sensors
    # ======================================================================
    # Each sensor reads a linear function of the state:  reading = rows @ x + offset.
    # rows has one row per number the sensor gives (one per component for an analyser, one
    # otherwise) and one column per model state. Most rows pick a single state (a 1.0 in that
    # column); a density meter adds up the components with their molar masses.
    def rows(self, sensor: Sensor) -> np.ndarray | None:
        """(m, nx) matrix: reading deviation = rows @ state deviation. None for feed_analyser.

        Raises ValueError for an unknown kind.
        """
        builders = {
            "composition_aq": self._rows_composition_aq,
            "product_assay": self._rows_composition_aq,
            "composition_org": self._rows_composition_org,
            "acid": self._rows_acid,
            "interface": self._rows_interface,
            "density_aq": self._rows_density_aq,
            "density_org": self._rows_density_org,
        }
        if sensor.kind == "feed_analyser":
            return None                           # it reads an input, not the state
        if sensor.kind not in builders:
            raise ValueError(sensor.kind)
        return builders[sensor.kind](sensor)

    def _rows_composition_aq(self, sensor: Sensor) -> np.ndarray:
        """One row per component read: picks c_aq[stage, component], at index stage*n + component."""
        c, p = self.circuit, self.circuit.p
        n, nx, j = c.n, c.nx_model, sensor.stage
        R = np.zeros((len(sensor.components), nx))
        for r, name in enumerate(sensor.components):
            R[r, j * n + p.index(name)] = 1.0
        return R

    def _rows_composition_org(self, sensor: Sensor) -> np.ndarray:
        """One row per component read: picks c_org[stage, component], at index N*n + stage*n + component."""
        c, p = self.circuit, self.circuit.p
        N, n, nx, j = c.N, c.n, c.nx_model, sensor.stage
        R = np.zeros((len(sensor.components), nx))
        for r, name in enumerate(sensor.components):
            R[r, N * n + j * n + p.index(name)] = 1.0
        return R

    def _rows_acid(self, sensor: Sensor) -> np.ndarray:
        """One row: picks the acid h[stage] (mol/L), at index 2*N*n + stage."""
        c = self.circuit
        R = np.zeros((1, c.nx_model))
        R[0, 2 * c.N * c.n + sensor.stage] = 1.0
        return R

    def _rows_interface(self, sensor: Sensor) -> np.ndarray:
        """One row: picks the interface height z[stage] (m), at index 2*N*n + N + stage."""
        c = self.circuit
        R = np.zeros((1, c.nx_model))
        R[0, 2 * c.N * c.n + c.N + sensor.stage] = 1.0
        return R

    def _rows_density_aq(self, sensor: Sensor) -> np.ndarray:
        """One row: aqueous density (g/mL) as a linear sum of the aqueous concentrations (APPENDIX_A A.9).

            rho_aq = rho_water + sum_i (M_i + 3 M_Cl - rho_water * phi_V) * c_aq_i / 1000

        Each metal comes in as its chloride RECl3 (molar masses in g/mol), and the salt takes up an
        apparent volume phi_V that would otherwise hold water. rho_water is the offset (see offset).
        """
        c, p = self.circuit, self.circuit.p
        n, j = c.n, sensor.stage
        R = np.zeros((1, c.nx_model))
        R[0, j * n:(j + 1) * n] = (p.M + 3 * p.M_cl - p.rho_water * p.phi_v) / 1000.0
        return R

    def _rows_density_org(self, sensor: Sensor) -> np.ndarray:
        """One row: organic density (g/mL) = rho_org0 + sum_i M_i * c_org_i / 1000 (APPENDIX_A A.9).
        The loaded metal adds mass at unchanged volume. rho_org0 is the offset (see offset)."""
        c, p = self.circuit, self.circuit.p
        N, n, j = c.N, c.n, sensor.stage
        R = np.zeros((1, c.nx_model))
        R[0, N * n + j * n: N * n + (j + 1) * n] = p.M / 1000.0
        return R

    # ======================================================================
    # True value, fixed offset and noise of a reading
    # ======================================================================
    def true_value(self, sensor: Sensor, x: np.ndarray, u_vec: np.ndarray) -> np.ndarray:
        """What a perfect instrument would read: rows @ x + offset, or the feed composition for the
        feed analyser (taken from the input vector u_vec, mol/L)."""
        if sensor.kind == "feed_analyser":
            n = self.circuit.n
            return u_vec[FEED_COMPOSITION_START:FEED_COMPOSITION_START + n].copy()
        return self.rows(sensor) @ x + self.offset(sensor)

    def offset(self, sensor: Sensor) -> float:
        """The constant part of the reading: the clean-phase density (g/mL) for density meters, else 0."""
        p = self.circuit.p
        return {"density_aq": p.rho_water, "density_org": p.rho_org0}.get(sensor.kind, 0.0)

    def noise_sd(self, sensor: Sensor, value: np.ndarray) -> np.ndarray:
        """One standard deviation of the noise on each number of the reading `value`.

        Relative kinds: sd = noise * |value|, never below 1e-4 (nothing reads exactly zero).
        Absolute kinds (density, interface): sd = noise, in the reading's units.
        """
        rel = self._defaults[sensor.kind]["relative"]
        # relative noise, with an absolute floor: nothing reads exactly zero
        # (chosen: 1e-4 in the reading's units)
        if rel:
            return np.maximum(sensor.noise * np.abs(value), MIN_NOISE_SD)
        return np.full_like(np.atleast_1d(value), sensor.noise, dtype=float)

    # ======================================================================
    # Readings at time t from the plant's history
    # ======================================================================
    def readings(self, t: float, history: list) -> dict:
        """The readings available at time t.

        Parameters:
            t: the present time, hours.
            history: list of (t, x, u_vec), oldest first, on the control grid. x is the state in the
                model layout, u_vec the input vector applied then.
        Returns:
            {label: (value, t_measured)} for the sensors that have a fresh sample at t; a sensor off
            its sample tick gives nothing. value is an array in the reading's units, t_measured the
            time (h) of the stored sample it was read from.

        For each sensor, in the order of the list (the noise draws follow that order, so the same
        seed gives the same readings):
          1. skip it if t is not on its sample tick, or if its sample time is before the history;
          2. take the stored state nearest to t - delay;
          3. reading = true value + noise_sd * a standard normal draw.
        """
        # Judgement call: readings are taken only on the control grid and from the stored state
        # nearest to t - delay. A delay under half a step (the on-stream analysers' 0.1 h) rounds
        # to zero, and the quarter-hour analysers give one reading per half-hour step. Fine for the
        # lab assay's 4 h; for the on-stream analysers the study sees them a little better than they
        # are. Storing sub-step history, or a run with the delay rounded up, would show the cost.
        out = {}
        for s in self.sensors:
            if not self._sample_due(s, t):
                continue
            t_meas = t - s.delay_h
            if t_meas < history[0][0] - 1e-9:      # the sample would predate the history
                continue
            k = self._nearest_sample(history, t_meas)
            _, x, uv = history[k]
            v = np.atleast_1d(self.true_value(s, x, uv))
            v = v + self._rng.normal(0.0, 1.0, size=v.shape) * self.noise_sd(s, v)
            out[s.label()] = (v, history[k][0])
        return out

    @staticmethod
    def _sample_due(s: Sensor, t: float) -> bool:
        """True if t (hours) is on the sensor's sample tick: a whole number of intervals, or every
        step when the interval is 0."""
        return not (s.interval_h > 0 and (t / s.interval_h) % 1 > 1e-6)

    @staticmethod
    def _nearest_sample(history: list, t_meas: float) -> int:
        """Index of the stored sample nearest to t_meas (histories are on the control grid)."""
        return min(range(len(history)), key=lambda i: abs(history[i][0] - t_meas))

    # ======================================================================
    # Which sensors read what
    # ======================================================================
    @property
    def feed_sensor(self) -> Sensor | None:
        """The feed analyser, if the set has one."""
        return next((s for s in self.sensors if s.kind == "feed_analyser"), None)

    def state_sensors(self) -> list:
        """Every sensor that reads the state (all but the feed analyser): what the estimator uses."""
        return [s for s in self.sensors if s.kind != "feed_analyser"]
