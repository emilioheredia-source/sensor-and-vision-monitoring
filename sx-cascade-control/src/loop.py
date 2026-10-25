"""A plant, a controller and a scenario, stepped together one control interval at a time.

The closed loop is a generator: each `step()` advances the plant by one control interval and
returns what happened. Batch studies collect the steps into a Trajectory; a GUI can draw each
step as it arrives. Nothing here knows about any front end.

One interval, in order:
  1. the controller is told what it can know (an Observation): the true state, or with sensors
     the estimator's view, plus the feed as measured and the offsets read back from the plant
  2. it returns handle values; the scenario's upsets act on the plant inputs after the handles
     are written in (a flowmeter bias scales what was commanded, a hydraulic drift adds to the
     jackleg that was set), and the plant's own stage efficiencies are applied
  3. the plant is integrated over the interval
  4. with sensors: readings are taken, the estimator updates, and the next Observation is built
     from what the instruments said

Two ways to run it. Without sensors the controller sees the true plant state and the true feed
(full information). With a sensor set and an estimator, it sees only what the instruments and
the estimator say (ARCHITECTURE.md, decision 8). The plant is a separate simulate() call each
interval, so the plant can be a different model from the controller's (decisions 20 and 25).

What the controller sees is kept in one Observation object, updated in place each interval.
Time is in hours; flows in L/h; concentrations and acidities in mol/L.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

from .circuit import Circuit, Inputs
from .metrics import heavy_mask
from .simulate import Trajectory, simulate


# =====================================================================================================
# What passes between the loop and the controller
# =====================================================================================================
@dataclass
class Observation:
    """What a controller is told at the start of a control interval."""
    x: np.ndarray                         # the plant state as known: the truth, or the estimator's view
    c_feed: np.ndarray                    # feed composition as measured (analyser) or estimated, mol/L
    h_feed: float                         # feed acidity, mol/L
    Q_feed: float                         # feed flow, L/h (a feed flowmeter is a given)
    qorg_offset: float = 0.0              # estimated organic delivered minus commanded, L/h
    h_scrub: float | None = None          # scrub liquor acidity read back on the line (None: as commanded)
    product: tuple | None = None          # (purity, recovery) from a product-stream analyser, if any


class Controller(Protocol):
    """What the closed loop needs from a controller."""
    u_nominal: Inputs                     # the inputs at the operating point
    cut: str                              # where the product is split ("LaCe" for NdPr)
    last: dict                            # what the controller wants recorded for this interval
    level_loops: LevelLoops | None        # a PI loop on each settler under it, or None (it moves the jacklegs)

    @property
    def t_step(self) -> float:
        """The control interval, h."""

    def step(self, obs: Observation) -> dict:
        """Handle values for this interval."""

    def apply(self, handles: dict) -> Inputs:
        """Handle values written into the full inputs."""


@dataclass
class StepRecord:
    """One control interval: the interval started at `t - t_step` and ended at `t`."""
    t: float                              # h
    x: np.ndarray                         # plant state at the end of the interval
    u: np.ndarray                         # plant inputs over the interval, as applied (upsets included)
    handles: dict                         # what the controller commanded
    info: dict = field(default_factory=dict)    # the controller's record (QP status, bias, plan ...)
    x_seen: np.ndarray | None = None      # what the controller will be told next (truth or estimate)


# =====================================================================================================
# The closed loop
# =====================================================================================================
class ClosedLoop:
    """A plant driven by a controller through a scenario, one control interval per step().

    Example:
        loop = ClosedLoop(plant, x0, ctl, scenario)
        for rec in loop.steps(t_end=40.0):            # one StepRecord per interval, as it happens
            print(rec.t, rec.handles["Q_org"])
        # or all at once:
        traj, log = ClosedLoop(plant, x0, ctl, scenario).run(40.0)
    """

    def __init__(self, plant: Circuit, x0: np.ndarray, controller: Controller, scenario=None,
                 sensors=None, estimator=None, plant_k_factor=None, plant_drift: bool = True):
        """Set the loop up at time 0.

        plant_drift: let a drifting plant (plant_models.DriftingCircuit) drift. False during
        commissioning, so every run starts from the same trimmed plant whatever the drift.

        Args:
            plant: the circuit that plays the plant (it may be a different model from the
                controller's).
            x0: the plant's state at t = 0.
            controller: anything that satisfies Controller (CircuitMPC, or the open-loop holder).
            scenario: the upsets (a disturbances.Scenario), or None for none.
            sensors, estimator: a SensorSet and an Estimator. They go together: without them the
                controller sees the true state and the true feed.
            plant_k_factor: the plant's own stage efficiencies, one per stage (multipliers on the
                transfer rate the controller doesn't know about), or None.
            plant_drift: see above.

        Why eight parameters: the same as run_closed_loop's: four required, the rest optional keywords with defaults.
        """
        self.plant, self.controller, self.scenario = plant, controller, scenario
        self.sensors, self.estimator = sensors, estimator
        self.plant_drift = plant_drift
        self.kf = None if plant_k_factor is None else np.asarray(plant_k_factor, float)
        self.t_step = controller.t_step
        # The feed over time, as the scenario has it (the nominal feed if there's no scenario).
        nominal = controller.u_nominal
        self.u_of_t = scenario.u_of_t(nominal) if scenario is not None else (lambda t: nominal)
        self.t = 0.0
        self.x = np.asarray(x0, float)
        # What the controller is told first: the plant's state and the nominal feed.
        self.obs = Observation(x=plant.observable(self.x).copy(), c_feed=np.asarray(nominal.c_feed, float),
                               h_feed=float(nominal.h_feed), Q_feed=float(nominal.Q_feed))
        # The recent history the sensors read from (late readings look back in it).
        self._history = [(0.0, plant.observable(self.x).copy(), self.u_of_t(0.0).numpy())]
        self._heavy = heavy_mask(plant.p, controller.cut)          # which components are the product
        self._product_sensor = self._find_product_sensor()
        self.observer = self._make_observer()

    # ---- the loop -------------------------------------------------------------------------
    def steps(self, t_end: float) -> Iterator[StepRecord]:
        """One StepRecord per control interval, until t_end (h)."""
        while self.t < t_end - 1e-9:
            yield self.step()

    def run(self, t_end: float) -> tuple[Trajectory, list[dict]]:
        """All the steps to t_end, collected: the plant trajectory and one log row per interval.

        Each log row is the interval's start time t, the handles, and the controller's record.
        The trajectory has one more input row than intervals (the last one repeated), so that
        times, states and inputs line up.
        """
        ts, xs, us, log = [self.t], [self.x.copy()], [], []
        for rec in self.steps(t_end):
            ts.append(rec.t)
            xs.append(rec.x)
            us.append(rec.u)
            log.append(dict(t=rec.t - self.t_step, **rec.handles, **rec.info))
        us.append(us[-1])
        return Trajectory(np.array(ts), np.array(xs), np.array(us)), log

    def step(self) -> StepRecord:
        """One control interval: tell the controller, take its handles, run the plant, read the
        instruments. Returns what happened."""
        feed_now = self.u_of_t(self.t)
        # 1. what the controller is told (with sensors, it was built at the end of the last step)
        # Why (full information without sensors): decision 8. It's the reference the sensor study
        # degrades from, so a change in results comes only from what the controller can see.
        if self.sensors is None:
            self._observe_truth(feed_now)
        self.obs.Q_feed = float(feed_now.Q_feed)
        # 2. the controller's handles, and the inputs the plant really gets
        handles = self.controller.step(self.obs)
        u_cmd = self.controller.apply(handles)
        # 3. the plant over the interval, inputs held (or, under local level loops, the jacklegs
        #    moved by those loops every few minutes inside the interval)
        if self.plant_drift:                       # a drifting plant sets its drift; any other returns x as it is
            self.x = self.plant.hidden_at(self.x, self.t)
        levels = self.controller.level_loops
        if levels is None:
            uv = self._plant_inputs(u_cmd)
            self.x = simulate(self.plant, self.x, lambda _t: uv, self.t_step, dt_out=self.t_step).x[-1]
        else:
            uv = self._run_under_level_loops(u_cmd, levels)
        self.t += self.t_step
        # 4. with sensors: readings, the estimator, and the next Observation
        if self.sensors is not None:
            self._observe_instruments(u_cmd, feed_now, uv)
        return StepRecord(t=self.t, x=self.x, u=uv, handles=handles, info=dict(self.controller.last),
                          x_seen=self.obs.x)

    # ---- the pieces -----------------------------------------------------------------------
    def _run_under_level_loops(self, u_cmd: Inputs, levels: "LevelLoops") -> np.ndarray:
        """One control interval with the settlers' own level loops underneath: the interval is cut
        into the loops' sub-steps; at each one every loop reads its interface and moves its jackleg,
        and the plant runs the sub-step with that jackleg. Returns the last sub-step's inputs."""
        n_sub = max(1, int(round(self.t_step / levels.dt_h)))
        dt = self.t_step / n_sub
        N = self.plant.N
        uv = None
        for k in range(n_sub):
            z = self.plant.observable(self.x)[-N:]                 # each settler's interface probe
            u_sub = u_cmd.replace(H_jackleg=levels.update(z))
            uv = self._plant_inputs(u_sub, self.t + k * dt)
            u_held = uv
            self.x = simulate(self.plant, self.x, lambda _t: u_held, dt, dt_out=dt).x[-1]
        return uv

    def _plant_inputs(self, u_cmd: Inputs, t: float | None = None) -> np.ndarray:
        """The scenario's upsets on top of the commanded inputs, and the plant's own stage
        efficiencies. Also records the scrub acidity the plant really gets (it is read back).
        t: the time the upsets are taken at (default: the start of the interval)."""
        # Why (upsets after the handles): decision 16. A flowmeter bias has to scale what was
        # commanded and a hydraulic drift add to the jackleg that was set, or the controller
        # would be told about the upset through its own commands.
        t = self.t if t is None else t
        u_plant = self.scenario.u_of_t(u_cmd)(t) if self.scenario is not None else u_cmd
        self.obs.h_scrub = float(u_plant.h_scrub)
        return self._with_plant_efficiencies(u_plant).numpy()

    def _with_plant_efficiencies(self, u_plant: Inputs) -> Inputs:
        """The inputs with the plant's own stage efficiencies multiplied in (on top of any an upset
        set), or unchanged when the plant has none."""
        if self.kf is None:
            return u_plant
        base = np.ones(self.plant.N) if u_plant.k_factor is None else np.asarray(u_plant.k_factor, float)
        return u_plant.replace(k_factor=base * self.kf)

    def _observe_truth(self, feed_now: Inputs):
        """Full information: the controller is told the plant's state and the feed as it is."""
        self.obs.x = self.plant.observable(self.x)
        self.obs.c_feed, self.obs.h_feed = feed_now.c_feed, feed_now.h_feed

    def _observe_instruments(self, u_cmd: Inputs, feed_now: Inputs, uv: np.ndarray):
        """With sensors: take the readings, update the estimator, and build the next Observation
        from what the instruments and the estimator say.

        Args:
            u_cmd: the inputs the controller commanded this interval.
            feed_now: the feed as it was this interval (the feed flowmeter reads its flow).
            uv: the plant's inputs as applied, upsets included (vector).
        """
        # The last 200 intervals are kept for late readings.
        self._history = (self._history + [(self.t, self.plant.observable(self.x).copy(), uv)])[-200:]
        readings = self.sensors.readings(self.t, self._history)
        product = self._product_sensor
        q_prod = (float(self.plant.outlets(self.x, uv)["Q_prod"])          # the product flowmeter
                  if product is not None and product.label() in readings else None)
        self.observer.update(self.t, u_cmd, float(feed_now.Q_feed), readings, q_prod)

    def _make_observer(self):
        """The Observer that turns readings into what the controller is told, or None without sensors."""
        if self.sensors is None:
            return None
        return Observer(self.obs, self.estimator, self.sensors.feed_sensor, self._product_sensor, self.plant,
                        self._heavy, self.t_step)

    def _find_product_sensor(self):
        """The analyser on the product stage (an online one, or a lab assay), or None."""
        if self.sensors is None:
            return None
        product_stage = self.plant.cmap.product_stage
        return next((s for s in self.sensors.sensors if _reads_product(s, product_stage)), None)


def _reads_product(sensor, product_stage: int) -> bool:
    """True for an analyser (online, or a lab assay) on the product stage."""
    return sensor.kind in ("composition_aq", "product_assay") and sensor.stage == product_stage


def _label_or_none(sensor) -> str | None:
    """The sensor's label, or None when there's no sensor."""
    return None if sensor is None else sensor.label()


class Observer:
    """What the controller is told, from the instrument readings: the estimator's state, the feed
    as analysed (or estimated), the organic flow offset, and the product as measured.

    It needs only numbers that could come over a network: the time, the inputs commanded, the feed
    flowmeter, the readings ({label: (value, t_measured)}) and the product flowmeter. That's what
    lets the same code run in this loop and on the far side of a plant interface (plant_server.py).
    What it needs to know about the plant (which components the product analyser reports, which
    of them are product) is worked out once, here.

    Args:
        obs: the Observation to update in place.
        estimator: the Kalman filter (estimator.Estimator).
        feed_sensor, product_sensor: the feed analyser and the product analyser (sensors.Sensor),
            or None for either.
        plant: the circuit whose component order the readings follow.
        heavy: which components are product (metrics.heavy_mask).
        t_step: the control interval, h.
    """

    def __init__(self, obs, estimator, feed_sensor, product_sensor, plant, heavy, t_step: float):
        """Keep what update() needs (the arguments are in the class docstring), and work out once
        which of the product analyser's components are product.

        Why seven parameters: each is a separate source of what the controller is told (the class docstring lists them), and they come from different places in ClosedLoop.
        """
        self.obs, self.estimator, self.t_step = obs, estimator, t_step
        self.heavy = heavy
        self.feed_label = _label_or_none(feed_sensor)
        self.product_label = _label_or_none(product_sensor)
        if product_sensor is not None:
            self.product_heavy = np.array([heavy[plant.p.index(name)] for name in product_sensor.components])
            self.product_delay_h = product_sensor.delay_h

    def update(self, t: float, u_cmd: Inputs, q_feed: float, readings: dict, q_prod: float | None):
        """Fold one interval's readings into the Observation.

        Why five parameters: this is the Observer interface, which plant_server.PlantServer.update also serves: one interval's time, commands, feed flow, readings and product flow.
        """
        # the estimator predicts with what was commanded, plus the feed flowmeter's reading
        # Why: that's all a real controller knows; the gap to what the plant got is what the
        # estimator's offset states are for (decision 18).
        self.obs.x = self.estimator.step(t, u_cmd.replace(Q_feed=q_feed), readings)
        self._update_feed_seen(readings)
        self.obs.qorg_offset = self.estimator.qorg_offset
        if self.product_label is not None and self.product_label in readings:
            self.obs.product = self._measured_product(readings, q_feed, q_prod)

    def _update_feed_seen(self, readings: dict):
        """The feed composition the controller is told: the feed analyser's reading when there is
        one this interval (the last one stands otherwise); with no analyser, the estimator's
        feed estimate."""
        if self.feed_label is None:
            self.obs.c_feed = self.estimator.feed_estimate         # no analyser: the estimator's view
            return
        if self.feed_label in readings:
            self.obs.c_feed = readings[self.feed_label][0]

    def _measured_product(self, readings: dict, q_feed: float, q_prod: float) -> tuple:
        """Purity as measured on the product stream; recovery as measured too when the reading is
        prompt (flows are known), since a reading hours old against the feed of now is no recovery
        at all. A lab assay gives purity only; the estimate stands in for the rest (decision 23).

            purity   = heavy metal / all metal, in the product reading
            recovery = Q_prod * heavy metal in the product / (Q_feed * heavy metal in the feed)

        Returns:
            (purity, recovery) as fractions; recovery is NaN when the reading isn't prompt.
        """
        c = np.asarray(readings[self.product_label][0], float)
        heavy = self.product_heavy
        fed = q_feed * float(np.sum(np.asarray(self.obs.c_feed, float)[self.heavy]))
        purity = float(np.sum(c[heavy]) / max(np.sum(c), 1e-12))
        # Why (purity always, recovery only when prompt): decision 23 (2). A reading four hours old
        # against the feed of now is no recovery. Judgement call: "prompt" means within one control
        # step; the line was drawn there, not studied.
        prompt = self.product_delay_h <= self.t_step
        recovery = q_prod * float(np.sum(c[heavy])) / max(fed, 1e-9) if prompt else np.nan
        return purity, recovery


# =====================================================================================================
# Local level loops: a separate, faster layer under the controller
# =====================================================================================================
class LevelLoops:
    """One PI loop per settler, the way plants hold interfaces: each loop reads its own settler's
    interface probe every dt_h hours and moves that settler's jackleg to hold the interface at its
    setpoint. A controller that carries a LevelLoops (attribute level_loops) leaves the jacklegs to
    it; the closed loop runs the loops inside each control interval.

    Tuning, IMC for a first-order process. The interface answers a jackleg move with gain
    k = rho_aq / (rho_aq - rho_org) (4 to 7 at the ndpr_plant point, circuit._interface_rate) and the
    jackleg time constant tau (6 min). For a closed-loop time constant lam:
        Kc = tau / (k lam),   Ti = tau
    Velocity form, so a jackleg at its stop doesn't wind up:
        H_next = clip(H + Kc (e - e_prev) + Kc (dt / Ti) e),   e = z_set - z

    Judgement call: one gain for every settler (k = 5.5, mid-range), lam = tau, a probe that reads
    the interface exactly. A plant tunes each loop on a bump test, and its probes have a dead band.
    """

    def __init__(self, z_set: float, H0: np.ndarray, tau_h: float, bounds: tuple,
                 process_gain: float = 5.5, lam_h: float | None = None, dt_h: float = 0.025):
        """Tune the loops and start them at the jacklegs H0.

        Args:
            z_set: the interface setpoint, m, the same for every settler.
            H0: each settler's jackleg height at the start, m.
            tau_h: the jackleg time constant, h.
            bounds: (low, high), the jackleg's travel, m.
            process_gain: k, the interface's answer to a jackleg move (see above).
            lam_h: the closed-loop time constant, h; None means tau_h.
            dt_h: how often the loops read their probes and move, h.

        Why seven parameters: each is a separate tuning fact from the run file (setpoint, starting jacklegs, time constant, bounds, gain, closed-loop time, step).
        """
        self.z_set, self.bounds, self.dt_h = float(z_set), bounds, float(dt_h)
        self.H = np.asarray(H0, float).copy()
        lam = tau_h if lam_h is None else lam_h
        self.Kc, self.Ti = tau_h / (process_gain * lam), tau_h
        self._e_prev = None

    def update(self, z: np.ndarray) -> np.ndarray:
        """Each loop's new jackleg height (m) from its interface reading z (m)."""
        e = self.z_set - np.asarray(z, float)
        if self._e_prev is None:
            self._e_prev = e
        self.H = np.clip(self.H + self.Kc * (e - self._e_prev) + self.Kc * (self.dt_h / self.Ti) * e,
                         *self.bounds)
        self._e_prev = e
        return self.H.copy()


def run_closed_loop(plant: Circuit, x0: np.ndarray, controller: Controller, scenario, t_end: float,
                    sensors=None, estimator=None, plant_k_factor=None,
                    plant_drift: bool = True) -> tuple[Trajectory, list[dict]]:
    """Drive `plant` with `controller` to t_end (h); the trajectory and one log row per interval.

    t_end is the run's length in hours of plant time. Every other argument is ClosedLoop's, with
    the same meaning (see ClosedLoop.__init__). The long argument list is kept because notebooks,
    tests and the baseline call it with these names.

    Returns:
        (Trajectory, log), as ClosedLoop.run.

    Example:
        traj, log = run_closed_loop(plant, x0, ctl, Scenario([FeedStep(t=1.0, factor=1.3)]), 12.0)

    Why nine parameters: a plant, a start, a controller and an end time, plus optional keywords with defaults (scenario, sensors, estimator, the plant's stage rates, drift); callers pass only the ones they need.
    """
    loop = ClosedLoop(plant, x0, controller, scenario, sensors, estimator, plant_k_factor, plant_drift)
    return loop.run(t_end)
