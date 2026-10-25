"""A Gymnasium environment around the simulator, for the RL phase.

Reinforcement learning needs the plant wrapped in a standard shape: reset() starts an episode
and returns what the agent sees; step(action) moves the handles, runs the plant for one control
interval and returns what the agent sees next, a reward and whether the episode is over.
notebooks/09_rl_feed_switch.py trains a PPO policy on it (stable-baselines3) and compares it
with the MPC on the same episodes.

The task:

- The plant is the one the run file's [plant] section names, commissioned and trimmed onto
  spec once, when the environment is made. Every episode starts there.
- Each episode draws its own upsets: a feed switch at a random time between 2 and 10 h, to
  the Yangibana feed or to Eneabba (the nominal feed, so about half the episodes have no
  real switch); a flowmeter bias on the organic flow in about 70 % of episodes; and in about
  half, one stage's transfer rate cut to between 30 % and 90 %.
- The agent sees the plant through a realistic sensor set: analysers on the product and the
  raffinate, a feed analyser and every interface, read through an estimator.
- The agent moves the same four handles the MPC moves (not the jacklegs), with its own rate
  limits (RATE, about three times the MPC's) and an organic ceiling of 4000 L/h, and is scored
  on the same losses as metrics.transition_cost.

The constants below (HANDLES, RATE, BOUNDS) and the reward weights in _reward() are what the
trained policy in runs/ was trained with. Changing them means training again.

Needs the `rl` pixi environment:  pixi run -e rl python ...

Usage:

    env = CascadeEnv("ndpr_plant", seed=0)
    obs, info = env.reset(seed=3)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

import numpy as np

from .circuit import Inputs
from .commission import commission, trim_under_control
from .config import build_circuit, build_plant, load_config, nominal_inputs
from .disturbances import EfficiencyLoss, FeedSwitch, FlowmeterBias, Scenario
from .estimator import Estimator
from .mpc import CircuitMPC, MPCSettings
from .metrics import heavy_mask, product_purity
from .sensors import Sensor, SensorSet
from .simulate import simulate, steady_state

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as err:          # the default env has no gymnasium; the rl env does
    raise ImportError("rl_env needs gymnasium: run under `pixi run -e rl`") from err


# The handles the agent moves, in the order of the action vector.
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
# How far each handle moves per step at an action of +1 or -1 (L/h, L/h, mol/L, fraction).
# Judgement call: these are the first controller's limits (the do-mpc settings), about three
# times the MPC's, and the organic ceiling is 4000 L/h against the MPC's 4700. The comparison in
# notebook 09 runs the MPC with its own settings, so the two don't have the same room to move.
# Retraining with the MPC's rates and bounds would make it like for like; it means training again.
RATE = {"Q_org": 150.0, "Q_scrub": 40.0, "h_scrub": 0.1, "saponified_fraction": 0.02}
# Where each handle is kept, (low, high), in the same units.
BOUNDS = {"Q_org": (600.0, 4000.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
          "saponified_fraction": (0.0, 0.6)}


class CascadeEnv(gym.Env):
    """The plant as a Gymnasium environment: one step is one control interval.

    Args:
        config_name: a run file in configs/.
        t_step: the control interval, h.
        episode_h: how long an episode lasts, h.
        purity_target: the purity below which the reward adds a steep penalty (fraction).
        seed: the seed for the episodes' random upsets; reset(seed=...) replaces it.

    Making the environment commissions and trims the plant, which takes a while. After that,
    reset() is quick.

    Attributes used outside this file (notebooks/09): rc, spec, rec_target, purity_target,
    circ (the model), plant, kf (the plant's stage-efficiency multipliers), x_ss (the plant
    on spec), x_ss_model, u_model, t_step, n_steps, sensor_list, scenario, x, t, history.
    """
    metadata = {"render_modes": []}

    def __init__(self, config_name: str = "ndpr_plant", t_step: float = 0.5, episode_h: float = 40.0,
                 purity_target: float = 0.995, seed: int | None = None):
        """Load the run file, build the model and the plant, commission and trim the plant, and set
        up the instruments and the spaces. The arguments are in the class docstring.

        Why five parameters: the environment's settings, each a keyword with a default, the way gym environments are built.
        """
        super().__init__()
        self.rc = load_config(config_name)
        self.p = self.rc.params
        self.spec, self.rec_target = self.rc.targets()       # product purity spec, recovery floor
        self.purity_target = purity_target
        self.circ = build_circuit(self.rc)                    # the model the estimator uses
        self.plant, self.kf = build_plant(self.rc)            # what the agent drives
        self._commission_and_trim()
        self.t_step, self.n_steps = t_step, int(round(episode_h / t_step))
        self._choose_instruments()
        self.heavy = heavy_mask(self.p, self.rc.cut)          # True for the product's components
        self._build_spaces()
        self._rng = np.random.default_rng(seed)

    # ==========================================================================================
    # Setting up, once
    # ==========================================================================================

    def _commission_and_trim(self):
        """Bring the plant onto spec, much as operating_point.prepare does (decision 21).

        First commission: start the plant on the run file's design handles and let it settle.
        Then trim: an MPC with full information moves the handles until the product is on spec
        and they stop moving. Sets x_ss and u_model (the plant on spec and its handles), u_ss
        (the same handles with the plant's own stage efficiencies) and x_ss_model (the model's
        rest point under the trimmed handles, where the estimator starts).
        """
        ones = np.ones(self.circ.N)                           # nominal stage efficiencies
        x_rest, u_design = commission(self.plant, nominal_inputs(self.rc, self.plant).replace(k_factor=self.kf))
        # the model's own rest point under the design handles, for the trim controller
        self.x_ss_model = steady_state(self.circ, u_design.replace(k_factor=ones), x0=x_rest)
        trim = CircuitMPC(self.circ, self.x_ss_model, u_design.replace(k_factor=ones), self.rc.cut, self.spec,
                          self.purity_target, self.rec_target, MPCSettings(handles=HANDLES, jacklegs=False))
        # on spec, on the real plant
        self.x_ss, self.u_model = trim_under_control(self.plant, x_rest, self.kf, trim)
        self.u_ss = self.u_model.replace(k_factor=self.kf)
        # the model's rest point under the trimmed handles
        self.x_ss_model = steady_state(self.circ, self.u_model, x0=self.x_ss)

    def _choose_instruments(self):
        """The sensor list, and the interface window (m) the reward checks against.

        Measure what is controlled: analysers on the product and the raffinate, the feed
        analyser, and every interface. An analyser pair mid-cascade would infer purity through
        the model, and the model isn't the plant (decision 23).
        """
        # Why: the same instruments the MPC gets in notebook 09, so the two see the same plant.
        N = self.circ.N
        self.sensor_list = [Sensor("composition_aq", stage=self.circ.cmap.product_stage),   # [0]: product
                            Sensor("composition_aq", stage=0),                              # [1]: raffinate
                            Sensor("feed_analyser")] + [Sensor("interface", stage=j) for j in range(N)]
        geo = self.p.raw["stage_geometry"]
        self.z_lo, self.z_hi = (f * geo["settler_height_m"] for f in geo["interface_window_fraction"])

    def _build_spaces(self):
        """The observation and action spaces.

        Observation (kept small on purpose; see _obs() for the scaling): product composition (n),
        raffinate composition (n), the feed composition seen (n), every interface (N), the four
        handles (4) and the time (1). n is the number of components, N the number of stages.

        Action: one number per handle in HANDLES, from -1 to +1, a fraction of that handle's RATE.
        """
        n_obs = 2 * self.circ.n + self.circ.n + self.circ.N + len(HANDLES) + 1
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(HANDLES),), dtype=np.float32)

    # ==========================================================================================
    # An episode
    # ==========================================================================================

    def reset(self, *, seed=None, options=None):
        """Start an episode at the plant on spec, with freshly drawn upsets and instrument noise.

        Args:
            seed: if given, restarts the random draws from this seed (the same seed gives the
                same episode).
            options: not used; Gymnasium's signature has it.

        Returns:
            (observation, {}).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.scenario = self._draw_scenario()
        # the instruments' noise has its own seed, drawn after the upsets
        self.sensors = SensorSet(self.sensor_list, self.plant, seed=int(self._rng.integers(1 << 30)))
        # Why (implicit Euler): speed, and an exploring agent drives the plant into corners where the
        # adaptive integrator crawls (decision 23). The MPC in notebook 09 is given the same estimator.
        self.est = Estimator(self.circ, self.sensors, self.x_ss_model, self.u_model, self.t_step,
                             integrator="euler")
        self.x, self.t, self.k = self.x_ss.copy(), 0.0, 0     # plant state, time (h), step count
        self.readings = {}
        self.handles = {h: float(getattr(self.u_ss, h)) for h in HANDLES}
        self.feed_seen = np.asarray(self.u_ss.c_feed, float)   # the feed composition last read, mol/L
        self.history = [(0.0, self.x.copy(), self.u_ss.numpy())]   # (t, x, u) for the delayed instruments
        self.lost0 = self._lost_rate(self.x, self.u_ss.numpy())    # mol/h lost at the start, the baseline
        return self._obs(), {}

    def _draw_scenario(self) -> Scenario:
        """This episode's upsets, drawn from the environment's random generator.

        The order of the draws matters: the same seed must give the same episode, so a trained
        policy can be checked against the MPC on identical episodes.
        """
        # Why (drawn upsets, seeded): a policy trained on one fixed upset learns that upset; the
        # seed lets the MPC be run on the identical episode. Judgement call: the mix (half with a
        # switch, 70 % with a bias, half with a slow mixer) and the sizes were picked, not studied.
        t_switch = float(self._rng.uniform(2.0, 10.0))                  # h
        to = "yangibana" if self._rng.random() < 0.5 else "eneabba"     # eneabba is the nominal feed
        items = [FeedSwitch(t=t_switch, ramp_h=1.0, to=to, params=self.p,
                            removed_upstream=tuple(self.rc["feed"]["removed_upstream"]))]
        if self._rng.random() < 0.7:
            items.append(FlowmeterBias(t=0.0, stream="organic", bias=float(self._rng.uniform(-0.05, 0.05))))
        if self._rng.random() < 0.5:
            stage = int(self._rng.integers(0, self.circ.N))
            items.append(EfficiencyLoss(t=0.0, stages=(stage,), factor=float(self._rng.uniform(0.3, 0.9))))
        return Scenario(items)

    def _apply(self) -> Inputs:
        """The inputs the agent commands: the operating point with its handles written in.

        The strip flow follows the scrub flow, 300 L/h above it.
        """
        u = self.u_model.replace(**self.handles)                     # what the agent commands
        return u.replace(Q_strip=300.0 + u.Q_scrub)

    def _lost_rate(self, x, uv) -> float:
        """Product metal leaving in the raffinate, mol/h, at plant state x and inputs uv."""
        out = self.plant.outlets(x, uv)
        return out["Q_raff"] * float(np.sum(out["c_raff"][self.heavy]))

    def step(self, action):
        """Move the handles, run the plant one interval, read the instruments, score it.

        Args:
            action: one number per handle in HANDLES; clipped to [-1, 1] and scaled by RATE.

        Returns:
            (observation, reward, terminated, truncated, info). The episode never ends early
            (terminated is always False); it's truncated after n_steps. info has the product
            purity and the excess lost rate (mol/h) at the end of the interval.
        """
        a = np.clip(np.asarray(action, float), -1, 1)
        self._move_handles(a)
        u_cmd = self._apply()
        uv = self._advance_plant(u_cmd)
        self._read_instruments(u_cmd)
        reward, purity, lost_excess = self._reward(a, u_cmd, uv)
        terminated = False
        truncated = self.k >= self.n_steps
        return self._obs(), float(reward), terminated, truncated, {"purity": purity, "lost_excess": lost_excess}

    # ---- the pieces of one step ----------------------------------------------------------------

    def _move_handles(self, a: np.ndarray):
        """Each handle moves by its action times its RATE, and stays inside its BOUNDS."""
        for h, ai in zip(HANDLES, a):
            lo, hi = BOUNDS[h]
            self.handles[h] = float(np.clip(self.handles[h] + ai * RATE[h], lo, hi))

    def _advance_plant(self, u_cmd: Inputs) -> np.ndarray:
        """Run the plant one interval and return the inputs it really got, as a vector.

        The upsets act on what was commanded (a flowmeter bias scales the commanded flow), and
        the plant's own stage efficiencies are written in. Keeps the last 200 points of history
        for the instruments' delays.
        """
        u_plant = self.scenario.u_of_t(u_cmd)(self.t)
        base = np.ones(self.plant.N) if u_plant.k_factor is None else np.asarray(u_plant.k_factor, float)
        u_plant = u_plant.replace(k_factor=base * self.kf)           # the plant's own efficiencies
        uv = u_plant.numpy()
        self.x = simulate(self.plant, self.x, lambda _t: uv, self.t_step, dt_out=self.t_step).x[-1]
        self.t += self.t_step
        self.k += 1
        self.history.append((self.t, self.x.copy(), uv))
        self.history = self.history[-200:]
        return uv

    def _read_instruments(self, u_cmd: Inputs):
        """Take the readings due now, note the feed analyser's, and update the estimate.

        The estimator predicts with what was commanded, not with what the plant really got.
        """
        readings = self.sensors.readings(self.t, self.history)
        self.readings = readings
        feed_sensor = self.sensors.feed_sensor
        if feed_sensor is not None and feed_sensor.label() in readings:
            self.feed_seen = readings[feed_sensor.label()][0]
        self.est.step(self.t, u_cmd, readings)

    def _reward(self, a: np.ndarray, u_cmd: Inputs, uv: np.ndarray):
        """The transition cost of this interval, negated: the same terms as metrics.transition_cost.

        Each term is taken at the end of the interval; they're weighted, summed and multiplied by
        the interval length (h):
            off_spec      product made while purity is under spec, mol/h        weight 1e-2
            lost_excess   product metal lost beyond the starting rate, mol/h    weight 1e-2
            viol          how far the interfaces sit outside the window, m      weight 10
            reagent       base + strip acid + scrub acid, mol/h, less 4000      weight 1e-4
            purity        how far purity is under purity_target (fraction)      weight 1e3
            action        the sum of the squared actions (moving costs a bit)   weight 0.05

        Returns:
            (reward, product purity, lost_excess in mol/h).
        """
        # Why (the same terms as transition_cost): the policy is trained on what it is scored on.
        # Judgement call (the weights): metal is priced 100x reagent and a purity shortfall steeply;
        # chosen by hand. One hour of training on one seed shows where the policy got to, not where
        # it could get. Several seeds, longer training and a reward-weight sweep would settle it.
        out = self.plant.outlets(self.x, uv)
        purity = product_purity(out["c_prod"], self.p, self.rc.cut)
        prod_rate = out["Q_prod"] * float(np.sum(out["c_prod"]))
        off_spec = prod_rate if purity < self.spec else 0.0
        lost_excess = self._lost_rate(self.x, uv) - self.lost0
        viol = self._interface_violation()
        reagent = self._reagent_rate(u_cmd)
        reward = -(1e-2 * off_spec + 1e-2 * lost_excess + 10.0 * viol + 1e-4 * (reagent - 4000.0)
                   + 1e3 * max(0.0, self.purity_target - purity) + 0.05 * float(np.sum(a ** 2))) * self.t_step
        return reward, purity, lost_excess

    def _interface_violation(self) -> float:
        """How far the plant's interfaces sit outside the window, summed over the settlers, m."""
        z = np.asarray(self.plant.unpack(self.x)[3], float)                  # interface heights, m
        return float(np.sum(np.maximum(0, self.z_lo - z) + np.maximum(0, z - self.z_hi)))

    def _reagent_rate(self, u_cmd: Inputs) -> float:
        """Base for saponification plus strip and scrub acid, mol/h, as metrics.transition_cost."""
        return (u_cmd.saponified_fraction * self.p.raw["extractant"]["concentration_monomer"] * u_cmd.Q_org
                + u_cmd.Q_strip * u_cmd.h_strip + u_cmd.Q_scrub * u_cmd.h_scrub)

    # ==========================================================================================
    # What the agent sees
    # ==========================================================================================

    def _obs(self) -> np.ndarray:
        """The observation vector (float32), each part scaled to roughly order 1.

        Parts, in order:
            product composition / 0.5      mol/L per component, as the product analyser reads it
            raffinate composition / 0.5    the same, for the raffinate analyser
            feed composition / 0.3         mol/L, the feed analyser's last reading
            (interfaces - 0.25) / 0.1      m, from the estimate
            handles                        each as a fraction of its range in BOUNDS (0 to 1)
            t / 40                         time in the episode, h over 40
        Where an analyser has no reading this step, the estimate fills in.
        """
        # Judgement call: a small observation (outlet and feed compositions, interfaces, handles,
        # time), scaled by hand. No estimated offsets and no history, so a flowmeter bias is only
        # visible through its effect on the outlets. Adding the estimator's offsets would test that.
        x_hat = self.est.x_hat
        uv = self._apply().numpy()
        out = self.circ.outlets(x_hat, uv)
        z = np.asarray(self.circ.unpack(x_hat)[3], float)
        readings = getattr(self, "readings", {})
        c_prod = self._as_read(self.sensor_list[0], readings, out["c_prod"])
        c_raff = self._as_read(self.sensor_list[1], readings, out["c_raff"])
        obs = np.concatenate([
            c_prod / 0.5, c_raff / 0.5, self.feed_seen / 0.3, (z - 0.25) / 0.1,
            [(self.handles[h] - BOUNDS[h][0]) / (BOUNDS[h][1] - BOUNDS[h][0]) for h in HANDLES],
            [self.t / 40.0],
        ])
        return np.asarray(obs, np.float32)

    @staticmethod
    def _as_read(sensor: Sensor, readings: dict, estimate: np.ndarray) -> np.ndarray:
        """The sensor's reading if it has one this step, otherwise the estimate."""
        if sensor.label() in readings:
            return np.asarray(readings[sensor.label()][0], float)
        return estimate
