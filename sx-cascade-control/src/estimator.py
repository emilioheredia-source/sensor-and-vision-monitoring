"""State estimate from what the sensors give.

The controller can't see the plant's state. It sees readings: an analyser
here, a density meter there, an interface camera, a lab assay four hours
late. This file turns those readings into a best guess of the whole state,
step by step, with a Kalman filter.

How one step works:

  1. Predict. Run the controller's model (nominal parameters, no knowledge
     of any bias or drift) from the last estimate to now, under the inputs
     that were commanded plus the offsets estimated so far.
  2. Correct. Compare each reading with what the model says it should be
     (the "innovation"), multiply by the Kalman gain K, and add that to the
     estimate. K is computed once per sensor subset, from the model
     linearized at the operating point and the sensor set's noise.

The model is augmented with **disturbance states**: an offset on the feed
composition and an offset on the organic flow, both random walks. Without
them a filter blames every surprise on white noise in each state and spreads
the correction wrongly; with them it attributes a persistent surprise to a
feed that is not what it was told, or a flow that is not what was commanded.
Those two estimates are handed to the controller: the feed estimate as
feedforward when there is no feed analyser, the flow offset as a correction
for a flowmeter bias it could not otherwise see. (ARCHITECTURE.md decision 18;
the equations are in APPENDIX_B_control.md B.5.)

A delayed reading is compared with the estimate the filter had at the time
the sample was taken (the fresh prediction included), and the correction is
applied to the present estimate: the usual fixed-lag approximation, fine for
a process whose time constants are hours.

Every correction is gated (no state moves more than 30 % of its size in one
step) and the result is kept physical (no negative concentrations, offsets
within caps), so a surprise can't throw the estimate into a corner the
integrator can't leave. The offset caps are decision 23, item 4.

With no sensors at all the estimator is a pure model run and the offsets
stay at zero, which is the honest baseline.

State layout, as everywhere in the project (stages numbered from 0 at the
raffinate end, N stages, n components):

    x = [ c_aq (N*n) | c_org (N*n) | h (N) | z (N) ]
          mol/L        mol/L         mol/L   m
          aqueous      organic       acid    interface height
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from .circuit import Circuit, Inputs
from .sensors import SensorSet
from .simulate import implicit_euler, simulate


# --------------------------------------------------------------------------
# Chosen numbers used below (all in the units of the quantity they guard)
# --------------------------------------------------------------------------
# Smallest "size" a state is given when scaling noise and gates, so a state
# that sits at zero still gets some room: 1e-3 mol/L for concentrations and
# acid, 0.01 m for interface heights.
CONCENTRATION_FLOOR = 1e-3
INTERFACE_FLOOR_M = 0.01
# Smallest size of a disturbance's nominal value (L/h or mol/L), for the same reason.
DISTURBANCE_FLOOR = 1e-3
# A reading's noise is never taken below this, in the reading's units (APPENDIX_B B.4).
# Why (a floor at all): a component the feed doesn't carry reads zero with zero relative noise,
# which puts a zero on the diagonal of R and makes the Riccati equation singular (decision 23).
# Judgement call (1e-4): small next to any reading that matters; not varied.
MIN_READING_SD = 1e-4
# Acid is kept above this, in mol/L, after a correction.
MIN_ACID = 1e-4
# How far each offset may wander, as a fraction of its nominal value (decision 23, item 4).
# Why (caps by kind): measured. One 50 % cap on both let the mid-cascade pair invent 850 L/h of
# extra organic; a 100 % feed cap fed the controller a doubled feed and saturated its model.
# Judgement call (50 % and 15 %): picked from those two failures, not swept. The cost is known:
# without a feed analyser the estimator can't learn a feed whose NdPr has doubled (REPORT.md 10).
FLOW_OFFSET_CAP = 0.15
FEED_OFFSET_CAP = 0.5
# How many past estimates are kept for matching late readings.
HISTORY_LENGTH = 200


class _OutOfEvaluations(Exception):
    """The adaptive integrator used up its evaluation budget for one prediction (see
    Estimator.max_prediction_evaluations)."""


class _BudgetedCircuit:
    """A circuit that counts how often its right-hand side is evaluated and raises
    _OutOfEvaluations past `max_calls`. Everything else is passed through to the circuit."""

    def __init__(self, circuit, max_calls: int):
        """Wrap `circuit`, allowing at most `max_calls` right-hand-side evaluations."""
        self._circuit, self._max_calls, self._calls = circuit, max_calls, 0

    def rhs(self, x, u):
        """The circuit's dx/dt, counted."""
        self._calls += 1
        if self._calls > self._max_calls:
            raise _OutOfEvaluations(f"more than {self._max_calls} evaluations")
        return self._circuit.rhs(x, u)

    def __getattr__(self, name):
        """Anything else is the wrapped circuit's own."""
        return getattr(self._circuit, name)


@dataclass
class Estimator:
    """Kalman filter on the controller's model, with feed and organic-flow offsets as extra states.

    Fields:
        circuit: the controller's model (nominal parameters).
        sensors: the SensorSet whose readings come in each step.
        x_ss: the model's steady state at the operating point (state layout above).
        u_ss: the inputs at the operating point.
        t_step: the control step, in hours (the gain is built for this step).
        process_noise_relative: how much the filter distrusts its own model per step, as a
            fraction of each state's size. None: the value in parameters.toml [sensors].
        disturbance_noise_relative: how fast the offsets may wander per step, as a fraction of
            their nominal value.
        disturbances: which inputs get offset states: "c_feed" (one per component), "Q_org".
        max_disturbances: cap on how many offset states are used (None = no extra cap).
        integrator: "adaptive" or "euler" for the prediction (see the comments on the field).
        max_step_fraction: the gate; a correction moves a state by at most this fraction of its size.

    Usage:
        est = Estimator(model, sensors, x_ss, u_ss, t_step=0.5)
        x_hat = est.step(t_new, u_applied, sensors.readings(t_new, plant_history))
        c_feed_guess, flow_offset = est.feed_estimate, est.qorg_offset
    """
    circuit: Circuit
    sensors: SensorSet
    x_ss: np.ndarray
    u_ss: Inputs
    t_step: float
    # Judgement call: Q and the offsets' wander are relative sizes picked by hand (0.01 from
    # parameters.toml, 0.005 here), not identified from data. They set how fast the filter believes
    # a real change. With plant data, they'd be fitted from the innovations; here a sweep would do.
    process_noise_relative: float | None = None
    disturbance_noise_relative: float = 0.005    # chosen: how fast the offsets may wander, per half-hour step
                                                 # (0.02 let the flow-offset estimate swing ±80 L/h a step)
    disturbances: tuple = ("c_feed", "Q_org")
    max_disturbances: int | None = None          # cap on the disturbance states used (the placement study
                                                 # sets it so every sensor set is judged on the same model)
    # Why (adaptive here, implicit Euler in the controller): measured, decision 23 (4). With one
    # analyser late in the scrub, the cruder prediction gave a 15-point purity error; adaptive, 5
    # thousandths. The RL environment's "euler" is for speed, and its MPC comparison uses it too.
    integrator: str = "adaptive"                 # "adaptive" (the plant's integrator; exact enough for the
                                                 # Kalman gain, see step) or "euler" (fixed cost: for the RL
                                                 # environment, where an exploring agent drives the plant to
                                                 # corners the adaptive one crawls in, and whose analysers sit
                                                 # where the two agree to a few percent)
    # Judgement call: the gate keeps a surprise from throwing the estimate somewhere the integrator
    # can't leave. 30 % was not swept. It also slows the estimate on a real large change, and it
    # makes the filter no longer a Kalman filter on those steps. A sweep at 0.1, 0.3 and 1.0 on the
    # feed switch with the ECJTU pair would settle it.
    max_step_fraction: float = 0.3               # chosen: a correction may move a state by at most this
                                                 # fraction of its size per step
    # Why (decision 27): a controller has to come back on time. On some estimates the adaptive
    # integrator crawls, millions of tiny steps for one half hour (found with the ECJTU pair after
    # decision 26). After this many evaluations of the model it gives up for that step and the
    # prediction is made by implicit Euler instead (fixed cost, a little cruder); the step is counted
    # in slow_predictions. Judgement call: 20,000 is a few hundred times what a normal step needs.
    max_prediction_evaluations: int = 20000
    _gains: dict = field(default_factory=dict, repr=False)   # Kalman gains already computed, per sensor subset

    # ======================================================================
    # Set-up: which offsets to carry, the augmented model, the noise
    # ======================================================================
    def __post_init__(self):
        """Build everything the filter needs once: the offset columns and their caps, the
        discrete-time augmented model A_aug, the process noise Q_aug, and the starting estimate."""
        model = self.circuit
        self._lay = Inputs.layout(model.n, model.N)      # where each input sits in the input vector
        self.u_ss_v = self.u_ss.numpy()                  # the operating-point inputs as one vector

        self._d_cols = self._disturbance_columns()
        self.nd, self.nx = len(self._d_cols), model.nx   # number of offset states, number of model states
        self._d_cap = self._disturbance_caps()

        self.A_aug = self._augmented_transition()
        self.Q_aug = self._augmented_process_noise()

        # Start at the operating point, with no offsets. history holds (time h, estimate) pairs,
        # oldest first, so a late reading can be compared with the estimate of its own time.
        self.x_hat = self.x_ss.copy()
        self.d_hat = np.zeros(self.nd)
        self.history = [(0.0, self.x_hat.copy())]
        # Every sensor but the feed analyser reads the state. The feed analyser reads an input and
        # goes straight to the controller as feedforward (loop.py), so the filter never sees it.
        self._state_sensors = self.sensors.state_sensors()

    def _disturbance_columns(self) -> np.ndarray:
        """Which entries of the input vector get an offset state, in priority order.

        The order: feed components actually present in the feed (largest first), then the
        organic flow, then the feed components that are absent. A sensor set can only support as
        many offset states as it has measurement rows, so the gain uses the first few of this list
        (see _gain). Returns indices into the input vector (Inputs.layout).
        """
        present, absent = self._feed_columns() if "c_feed" in self.disturbances else ([], [])
        flow = [self._lay["Q_org"].start] if "Q_org" in self.disturbances else []
        return np.array(present + flow + absent)

    def _feed_columns(self) -> tuple[list, list]:
        """The input-vector columns of the feed components, split in two: the ones in the feed,
        largest concentration first, and the ones the feed doesn't carry, in component order."""
        feed_start = self._lay["c_feed"].start
        c_feed = np.asarray(self.u_ss.c_feed, float)              # mol/L, one per component
        largest_first = np.argsort(-c_feed)
        present = [feed_start + int(i) for i in largest_first[c_feed[largest_first] > 0]]
        absent = [feed_start + int(i) for i in range(self.circuit.n) if c_feed[i] <= 0]
        return present, absent

    def _disturbance_caps(self) -> np.ndarray:
        """How far each offset may wander, as a fraction of its nominal value (chosen).

        A feed offset may go to 50 %, a flow offset to 15 % (a flowmeter is not 50 % wrong).
        Without the flow cap the mid-cascade analyser pairs explained what a wrong model could
        not with an organic flow 850 L/h above what was commanded; with a feed cap of 100 % they
        fed the controller's model a doubled feed, which saturated it and made its linearization
        useless. A feed that really doubles is what a feed analyser is for. (Decision 23, item 4.)
        """
        q_org_col = self._lay["Q_org"].start
        return np.array([FLOW_OFFSET_CAP if col == q_org_col else FEED_OFFSET_CAP for col in self._d_cols])

    def _augmented_transition(self) -> np.ndarray:
        """The model over one control step, linearized at the operating point, with the offsets.

        Continuous time, near the operating point:  dx/dt = A x + F d,  where
            A = d(rhs)/dx                 (nx, nx)  the circuit's Jacobian
            F = d(rhs)/d(inputs), only the columns that have an offset  (nx, nd)
        Over one step of t_step hours, with d held constant (zero-order hold), the exact answer is
            expm([[A, F], [0, 0]] * t_step) = [[Ad, Ed], [0, I]]
        so x_next = Ad x + Ed d. The offsets are random walks: d_next = d. Together:
            [x; d]_next = A_aug [x; d],   A_aug = [[Ad, Ed], [0, I]]
        """
        # Why (random-walk offsets): the disturbance-model approach to offset-free estimation (Muske
        # and Badgwell, 2002; Pannocchia and Rawlings, 2003). Measured, decision 18: without them the
        # filter got worse the more it trusted the readings.
        # Judgement call (one linearization, at the operating point): the MPC re-linearizes every
        # step but this gain doesn't. Far from the point (a doubled-NdPr feed) the gain is for the
        # wrong model. Recomputing it every few steps and comparing purity error on the feed switch
        # would settle it.
        nx, nd = self.circuit.nx, len(self._d_cols)
        A = self.circuit.jac(self.x_ss, self.u_ss_v)
        F = self.circuit.jac_u(self.x_ss, self.u_ss_v)[:, self._d_cols]
        M = np.zeros((nx + nd, nx + nd))
        M[:nx, :nx], M[:nx, nx:] = A, F
        Md = expm(M * self.t_step)
        Ad, Ed = Md[:nx, :nx], Md[:nx, nx:]
        return np.block([[Ad, Ed], [np.zeros((nd, nx)), np.eye(nd)]])

    def _augmented_process_noise(self) -> np.ndarray:
        """Process noise covariance Q_aug for [x; d], diagonal.

        Each state's noise standard deviation is a fraction q of its size at the operating point:
            sd_x = q * max(|x_ss|, floor)
        and each offset's is disturbance_noise_relative of its nominal input:
            sd_d = disturbance_noise_relative * max(|u_ss|, 1e-3)
        The floors (1e-3 mol/L for concentrations and acid, 0.01 m for interfaces) give a state
        sitting at zero some room to move. They are also used by the gate in _apply_gated_correction.
        """
        N, n = self.circuit.N, self.circuit.n
        if self.process_noise_relative is None:
            q = self.circuit.p.raw["sensors"]["process_noise_relative"]
        else:
            q = self.process_noise_relative
        floors = np.concatenate([np.full(2 * N * n, CONCENTRATION_FLOOR), np.full(N, CONCENTRATION_FLOOR),
                                 np.full(N, INTERFACE_FLOOR_M)])
        self._floors = floors
        scale_x = np.maximum(np.abs(self.x_ss), floors)
        scale_d = np.maximum(np.abs(self.u_ss_v[self._d_cols]), DISTURBANCE_FLOOR)
        return np.diag(np.concatenate([(q * scale_x) ** 2, (self.disturbance_noise_relative * scale_d) ** 2]))

    # ======================================================================
    # What the controller gets
    # ======================================================================
    def _offsets(self) -> np.ndarray:
        """The estimated offsets laid out as a full input vector (zero where there's no offset state)."""
        off = np.zeros(Inputs.size(self.circuit.n, self.circuit.N))
        off[self._d_cols] = self.d_hat
        return off

    @property
    def feed_estimate(self) -> np.ndarray:
        """The feed composition the filter believes, mol/L per component: nominal plus offset.
        The controller uses it as feedforward when there is no feed analyser."""
        return np.asarray(self.u_ss.c_feed, float) + self._offsets()[self._lay["c_feed"]]

    @property
    def qorg_offset(self) -> float:
        """The estimated organic flow offset, L/h: how far the plant's organic flow seems to be from
        what was commanded (a flowmeter bias the meter itself can't see)."""
        return float(self._offsets()[self._lay["Q_org"].start])

    # ======================================================================
    # The Kalman gain for the sensors that reported this step (cached per subset)
    # ======================================================================
    # Sensors report on different ticks (an analyser every quarter hour, a lab assay every two
    # hours), so the set of readings changes from step to step. Each distinct set gets its own
    # gain, computed the first time it's seen and kept in self._gains.
    def _gain(self, labels: tuple):
        """Steady-state Kalman gain for the sensors named in `labels`.

        Parameters:
            labels: sorted tuple of sensor labels (Sensor.label()) that reported this step.
        Returns:
            (K, C_aug, P_full):
              K       (nx + nd, m)        gain: correction = K @ innovations
              C_aug   (m, nx + nd_used)   measurement map on the state and the offsets in use
              P_full  (nx + nd, nx + nd)  steady-state error covariance (zeros for unused offsets)
        Raises:
            RuntimeError if no gain can be computed even with no offset states.

        Usage:
            K, _, _ = est._gain(("composition_aq@3", "density_aq@5"))
        """
        if labels in self._gains:
            return self._gains[labels]
        C, R = self._measurement_model(labels)
        # As many offset states as the readings can support (one per measurement row at most);
        # fewer, down to none, until the Riccati equation can be solved (the pair is detectable).
        nd_top = self._offsets_supported(C.shape[0])
        # Why (no more offsets than measurement rows): the detectability condition for an augmented
        # disturbance model (Muske and Badgwell; Pannocchia and Rawlings). Judgement call: a failed
        # Riccati solve is used as the test for detectability instead of a rank check.
        for nd_use in range(nd_top, -1, -1):
            result = self._riccati_gain(C, R, nd_use)
            if result is not None:
                self._gains[labels] = result
                return self._gains[labels]
        raise RuntimeError("no Kalman gain could be computed for this sensor set")

    def _measurement_model(self, labels: tuple) -> tuple[np.ndarray, np.ndarray]:
        """The measurement map C and the reading noise covariance R for these sensors.

        reading deviation = C @ state deviation   (C stacks each sensor's rows, see SensorSet.rows)
        R = diag(sd^2), sd = each reading's noise at the operating point, never below 1e-4.

        An instrument has an absolute floor as well as a relative noise: a component that is not
        there reads zero with "zero" relative noise, and a zero on the diagonal of R turns the
        Riccati equation singular and the gain enormous (chosen: 1e-4 in the reading's units).
        """
        reporting = [s for s in self._state_sensors if s.label() in labels]
        C = np.vstack([self.sensors.rows(s) for s in reporting])
        sd = np.concatenate([self.sensors.noise_sd(s, self.sensors.true_value(s, self.x_ss, self.u_ss_v))
                             for s in reporting])
        R = np.diag(np.maximum(sd, MIN_READING_SD) ** 2)
        return C, R

    def _offsets_supported(self, n_rows: int) -> int:
        """Most offset states to try: no more than the measurement rows, the offsets carried, or the cap."""
        cap = self.max_disturbances if self.max_disturbances is not None else self.nd
        return min(n_rows, self.nd, cap)

    def _riccati_gain(self, C: np.ndarray, R: np.ndarray, nd_use: int):
        """Kalman gain using the model states plus the first `nd_use` offset states.

        Solves the discrete algebraic Riccati equation for the steady-state error covariance P of
        the pair (A, C_aug), with C_aug = [C, 0] (readings see the state, not the offsets directly):
            P = A P A' - A P C' (C P C' + R)^-1 C P A' + Q
        then  K = P C' (C P C' + R)^-1.
        Returns (K, C_aug, P_full) padded to the full [x; d] size, or None if the equation has no
        solution for this pair (scipy raises LinAlgError).
        """
        nx = self.nx
        idx = np.r_[np.arange(nx), nx + np.arange(nd_use)]       # the rows/columns of [x; d] in use
        A = self.A_aug[np.ix_(idx, idx)]
        Q = self.Q_aug[np.ix_(idx, idx)]
        C_aug = np.hstack([C, np.zeros((C.shape[0], nd_use))])
        try:
            P = solve_discrete_are(A.T, C_aug.T, Q, R)
        except np.linalg.LinAlgError:
            return None
        K_small = P @ C_aug.T @ np.linalg.inv(C_aug @ P @ C_aug.T + R)
        K = np.zeros((nx + self.nd, C.shape[0]))
        K[idx] = K_small
        P_full = np.zeros((nx + self.nd, nx + self.nd))
        P_full[np.ix_(idx, idx)] = P
        return (K, C_aug, P_full)

    # ======================================================================
    # One estimator step: predict, set interfaces, correct, record
    # ======================================================================
    def step(self, t_new: float, u_applied: Inputs, readings: dict) -> np.ndarray:
        """Advance the estimate to t_new and correct it with the readings taken.

        Parameters:
            t_new: the new time, hours.
            u_applied: the inputs commanded since the last step (the feed flowmeter's reading
                included). The estimated offsets are added on top.
            readings: {label: (value, t_measured)} from SensorSet.readings; value is an array in
                the reading's units, t_measured the time (h) the sample was taken.
        Returns:
            the state estimate at t_new (state layout above). Also kept in self.x_hat, and the
            offsets in self.d_hat. Each prediction or correction that had to be thrown away adds
            one to self.failures.
        """
        self._predict(t_new, u_applied)
        self._set_interfaces(readings)
        labels = self._gain_labels(readings)
        if labels:
            self._correct(labels, readings, t_new)
        self._record(t_new)
        return self.x_hat

    def _count_failure(self):
        """One more step whose prediction or correction had to be thrown away."""
        self.failures = getattr(self, "failures", 0) + 1

    # ---- 1. predict -------------------------------------------------------
    def _predict(self, t_new: float, u_applied: Inputs):
        """Run the model from the last estimate to t_new under the commanded inputs plus offsets.

        The adaptive integrator, not the controller's implicit Euler: the Kalman gain is built for
        the exact model, and at the positions where it is largest (one analyser late in the scrub)
        a cruder prediction each step was amplified into a 15-point purity error; the adaptive
        integrator gave 5 thousandths at the same position, and faster. (Decision 23.)
        If the integrator can't carry the estimate (an unphysical state), the estimate stays where
        it was and the failure is counted.
        """
        u_with_offsets = u_applied.numpy()
        u_with_offsets[self._d_cols] += self.d_hat      # commanded input + estimated offset
        t_old = self.history[-1][0]
        try:
            self.x_hat = self._integrate_estimate(u_with_offsets, t_new - t_old)
        except RuntimeError:                     # an unphysical estimate the integrator cannot carry
            self._count_failure()
            self.x_hat = self.history[-1][1].copy()

    def _integrate_estimate(self, u_vec: np.ndarray, dt: float) -> np.ndarray:
        """The model state self.x_hat moved forward by dt hours under the constant input vector u_vec.
        Raises RuntimeError if the integrator fails or the result isn't finite."""
        if self.integrator == "euler":
            return self._euler_prediction(u_vec, dt)
        budgeted = _BudgetedCircuit(self.circuit, self.max_prediction_evaluations)
        try:
            return simulate(budgeted, self.x_hat, lambda _t: u_vec, dt, dt_out=dt).x[-1]
        except _OutOfEvaluations:
            self.slow_predictions = getattr(self, "slow_predictions", 0) + 1
            return self._euler_prediction(u_vec, dt)

    def _euler_prediction(self, u_vec: np.ndarray, dt: float) -> np.ndarray:
        """The same prediction by implicit Euler (fixed cost). Raises RuntimeError if it isn't finite."""
        x_pred = implicit_euler(self.circuit, self.x_hat, u_vec, dt)
        if not np.all(np.isfinite(x_pred)):
            raise RuntimeError("non-finite prediction")
        return x_pred

    # ---- 2. interface readings go straight in --------------------------------
    def _set_interfaces(self, readings: dict):
        """An interface reading is taken as the interface: the z state of that stage is set to it.

        Through the Riccati gain it would also rewrite the compositions (the model ties the interface
        to the densities, tightly, and a 2 mm reading against a 6-minute settler is a very confident
        instrument), and that threw the estimate into corners the integrator could not leave.
        Hydraulics say little about chemistry on a real settler; here they say nothing.
        """
        # Why: measured, decision 23 (5). Through the gain, a 2 mm reading rewrote the compositions.
        # Judgement call: this throws away what an interface says about the phase densities, which
        # is real if weak information on loading. Cost unknown; a small, separately tuned gain from
        # the interface to the densities, tried on the pump stop, would show if it's worth having.
        N, n = self.circuit.N, self.circuit.n
        z_start = 2 * N * n + N                         # index of z for stage 0 in the state
        for s in self._state_sensors:
            if s.kind == "interface" and s.label() in readings:
                self.x_hat[z_start + s.stage] = float(np.atleast_1d(readings[s.label()][0])[0])

    def _gain_labels(self, readings: dict) -> tuple:
        """Sorted labels of the readings that go through the Kalman gain (everything but interfaces)."""
        return tuple(sorted(k for k in readings
                            if any(s.label() == k and s.kind != "interface" for s in self._state_sensors)))

    # ---- 3. correct with the other readings ----------------------------------
    def _correct(self, labels: tuple, readings: dict, t_new: float):
        """Kalman correction:  [x; d] += K @ innovations, gated, then kept physical.

        A correction that comes out non-finite (a reading that makes no sense) is skipped and
        counted as a failure.
        """
        K, _, _ = self._gain(labels)
        dz = K @ np.concatenate(self._innovations(labels, readings, t_new))
        if not np.all(np.isfinite(dz)):          # a reading that makes no sense: skip the correction
            self._count_failure()
            dz = np.zeros_like(dz)
        self._apply_gated_correction(dz)
        self._keep_physical()

    def _innovations(self, labels: tuple, readings: dict, t_new: float) -> list:
        """Each reading minus what the estimate says it should read, one array per sensor.

        innovation = reading - (rows @ x_then + offset)
        x_then is the estimate nearest to the time the sample was taken, from the stored history plus
        the fresh prediction at t_new. A sensor with a 4-hour delay is compared with the estimate of
        4 hours ago, and the correction is applied to the present estimate (fixed-lag approximation).
        The order follows the sensor set, the same order as the rows of the gain.
        """
        # Judgement call (fixed-lag shortcut): the innovation is formed at the sample time but the
        # gain and the correction act on the present state, as if the state hadn't moved since.
        # Fine for slow changes; for the 4 h lab assay after a feed switch it isn't exact. The exact
        # way is to carry delayed states in the filter; comparing the two on the lab-assay set would
        # show the cost.
        candidates = self.history + [(t_new, self.x_hat)]
        innov = []
        for s in self._state_sensors:
            if s.label() not in labels:
                continue
            value, t_meas = readings[s.label()]
            x_then = min(candidates, key=lambda h: abs(h[0] - t_meas))[1]
            innov.append(np.atleast_1d(value) - (self.sensors.rows(s) @ x_then + self.sensors.offset(s)))
        return innov

    def _apply_gated_correction(self, dz: np.ndarray):
        """Add the correction dz = [dx; dd], each entry clipped to max_step_fraction of its size.

        Gate the step: no state moves by more than a fraction of its own size in one correction,
        so a surprise cannot throw the estimate into a corner the integrator cannot leave.
            limit_x = max_step_fraction * max(|x_hat|, floor)
            limit_d = max_step_fraction * max(|u_ss| of the offset's input, 1e-3)
        """
        lim_x = self.max_step_fraction * np.maximum(np.abs(self.x_hat), self._floors)
        lim_d = self.max_step_fraction * np.maximum(np.abs(self.u_ss_v[self._d_cols]), DISTURBANCE_FLOOR)
        self.x_hat = self.x_hat + np.clip(dz[: self.nx], -lim_x, lim_x)
        self.d_hat = self.d_hat + np.clip(dz[self.nx:], -lim_d, lim_d)

    def _keep_physical(self):
        """Keep things physical after a correction.

        - concentrations and acid >= 0, and acid >= 1e-4 mol/L
        - no concentration past 3x the largest one at the operating point (at least 0.1 mol/L)
        - each offset within its cap: |d| <= cap * max(|nominal input|, 1e-3)
        """
        N, n = self.circuit.N, self.circuit.n
        n_conc = 2 * N * n + N                          # concentrations and acid: everything but z
        self.x_hat[:n_conc] = np.maximum(self.x_hat[:n_conc], 0.0)
        self.x_hat[2 * N * n: n_conc] = np.maximum(self.x_hat[2 * N * n: n_conc], MIN_ACID)
        # no concentration past 3x the largest nominal one (mol/L)
        # Judgement call: 3x is a loose physical guard, not measured; it should never bind.
        cap = 3.0 * max(float(np.max(self.x_ss[:2 * N * n])), 0.1)
        self.x_hat[:2 * N * n] = np.minimum(self.x_hat[:2 * N * n], cap)
        lim = self._d_cap * np.maximum(np.abs(self.u_ss_v[self._d_cols]), DISTURBANCE_FLOOR)
        self.d_hat = np.clip(self.d_hat, -lim, lim)

    # ---- 4. record ------------------------------------------------------------
    def _record(self, t_new: float):
        """Store (t_new, a copy of the estimate) and keep only the last HISTORY_LENGTH entries."""
        self.history.append((t_new, self.x_hat.copy()))
        if len(self.history) > HISTORY_LENGTH:
            self.history = self.history[-HISTORY_LENGTH:]

    # ======================================================================
    # For the sensor placement study
    # ======================================================================
    def purity_uncertainty(self, g_purity: np.ndarray, labels: tuple | None = None) -> float:
        """Standard deviation of product purity implied by the steady-state covariance, for a sensor set.

        sd_purity = sqrt(g' P g), with g = d purity / d x at the operating point and P the steady-state
        error covariance from the Riccati equation (APPENDIX_B B.7). It counts precision and sampling,
        not delay.

        Parameters:
            g_purity: (nx,) gradient of product purity with respect to the state.
            labels: which sensors; None = every state sensor except the interfaces.
        Returns:
            sd of purity, as a fraction (0.001 = 0.1 percentage point).
        """
        labels = labels or tuple(sorted(s.label() for s in self._state_sensors if s.kind != "interface"))
        # Why: the steady-state Kalman covariance projected on the output is the standard
        # model-based way to rank sensor positions before any closed loop. Judgement call: it trusts
        # the model and ignores delay; notebook 07 checks it on the perturbed plant, where the
        # numbers came out forty times larger but the best positions stayed late in the scrub.
        _, _, P = self._gain(labels)
        g = np.concatenate([g_purity, np.zeros(self.nd)])
        return float(np.sqrt(g @ P @ g))
