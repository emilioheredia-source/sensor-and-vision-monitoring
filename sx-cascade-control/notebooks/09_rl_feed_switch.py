"""RL on the feed switch: train a PPO policy against the simulator, then score it
on the same episodes and the same transition cost as the MPC.

The question: a learned policy, given the same plant, handles and sensors as
the controller, how close does it get?

The point is the simulator as the safe place to let a policy learn and be
stress-tested before anything touches a plant. The environment is
src/rl_env.py: the plant the run file names (perturbed, for the ndpr_* files),
commissioned and trimmed once; each episode draws its own upsets (a feed switch
in about half, a flowmeter bias in most, sometimes a slow mixer); the agent
reads product, raffinate and feed analysers and every interface through an
estimator, and moves the same four handles as the MPC, not the jacklegs.

What came out, in the report's run (README, "A learned policy"): the policy
holds purity about as well, uses a quarter less reagent, and loses three to
fourteen times more NdPr than the MPC.

Method: train PPO (stable-baselines3) for a fixed number of minutes, then for
each evaluation episode run the policy and the MPC on the identical upsets
from the same start, and score both.

Needs the rl pixi environment:
    pixi run -e rl python notebooks/09_rl_feed_switch.py [config_name] [train_minutes] [n_eval]
    config_name defaults to ndpr_small, train_minutes to 30, n_eval to 5.
Writes runs/rl_<config>/ppo_feed_switch.zip (the trained policy) and runs/rl_eval_<config>.csv.
The results in the report: ndpr_small, 60 training minutes, 5 evaluation episodes.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.estimator import Estimator                                     # noqa: E402
from src.metrics import transition_cost                                 # noqa: E402
from src.mpc import CircuitMPC, MPCSettings, run_closed_loop            # noqa: E402
from src.rl_env import HANDLES, CascadeEnv                              # noqa: E402
from src.sensors import SensorSet                                       # noqa: E402
from src.simulate import Trajectory                                     # noqa: E402

# ---- 1. the command line ----
name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_small"
TRAIN_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0     # training time, minutes of wall clock
N_EVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 5             # evaluation episodes
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "runs" / f"rl_{name}"
OUT.mkdir(parents=True, exist_ok=True)

from stable_baselines3 import PPO                                       # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback             # noqa: E402


# ---- 2. train for a fixed time ----
# PPO is told to run 10 million steps, far more than it gets to: the callback below stops it when
# the time budget runs out. The environment and the policy are both seeded with 0.
class TimeBudget(BaseCallback):
    """Stops training once a set number of minutes has passed."""

    def __init__(self, minutes):
        """minutes: the training budget, in minutes of wall clock from now."""
        super().__init__()
        self.deadline = time.time() + 60 * minutes

    def _on_step(self):
        """Called by stable-baselines3 after every step; returning False ends training."""
        return time.time() < self.deadline


env = CascadeEnv(name, seed=0)
model = PPO("MlpPolicy", env, n_steps=256, batch_size=64, learning_rate=3e-4, gamma=0.99, verbose=0,
            tensorboard_log=None, seed=0)
t0 = time.time()
model.learn(total_timesteps=10_000_000, callback=TimeBudget(TRAIN_MIN))
model.save(OUT / "ppo_feed_switch")
print(f"trained {model.num_timesteps} steps in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- 3. evaluation: same episodes for the policy and for the MPC (same sensors, same estimator) ----
# Episode ep uses seed 1000 + ep, so its upsets are the same for both. The policy acts on what the
# environment shows it; the MPC is run on the same plant, scenario and sensor set, with an estimator
# that integrates the same way as the agent's (implicit Euler, for speed).
rows = []
for ep in range(N_EVAL):
    seed = 1000 + ep
    # RL: step the environment with the trained policy until the episode ends, keeping the plant's
    # state (xs), the inputs it got (us) and the times (ts), to score like any other run
    obs, _ = env.reset(seed=seed)
    xs, us, ts = [env.x.copy()], [], [0.0]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        xs.append(env.x.copy())
        us.append(env.history[-1][2])
        ts.append(env.t)
        done = term or trunc
    us.append(us[-1])                              # one input per state: repeat the last for the final state
    traj_rl = Trajectory(np.array(ts), np.array(xs), np.array(us))
    c_rl = transition_cost(env.plant, traj_rl, env.rc.cut, env.spec)
    # MPC on the identical scenario and plant, same sensor set (its noise drawn separately)
    env.reset(seed=seed)
    ss = SensorSet(env.sensor_list, env.plant, seed=seed)
    est = Estimator(env.circ, ss, env.x_ss_model, env.u_model, env.t_step, integrator="euler")   # as the agent's
    ctl = CircuitMPC(env.circ, env.x_ss_model, env.u_model, env.rc.cut, env.spec, env.purity_target, env.rec_target,
                     MPCSettings(handles=HANDLES, jacklegs=False))
    traj_mpc, _ = run_closed_loop(env.plant, env.x_ss, ctl, env.scenario, env.n_steps * env.t_step,
                                  sensors=ss, estimator=est, plant_k_factor=env.kf)
    c_mpc = transition_cost(env.plant, traj_mpc, env.rc.cut, env.spec)
    # one row per episode: each cost twice, with _rl and _mpc on the end
    row = {"episode": ep}
    for k in ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol"):
        row[k + "_rl"], row[k + "_mpc"] = c_rl[k], c_mpc[k]
    row["purity_min_rl"], row["purity_min_mpc"] = float(c_rl["purity_series"].min()), float(c_mpc["purity_series"].min())
    rows.append(row)
    print(f"episode {ep}: purity min RL {row['purity_min_rl']:.4f} MPC {row['purity_min_mpc']:.4f} | "
          f"lost RL {c_rl['lost_excess_mol']:.0f} MPC {c_mpc['lost_excess_mol']:.0f} | "
          f"reagent RL {c_rl['reagent_mol']:.0f} MPC {c_mpc['reagent_mol']:.0f}", flush=True)

# ---- 4. save the table ----
with open(HERE / "runs" / f"rl_eval_{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", HERE / "runs" / f"rl_eval_{name}.csv")
