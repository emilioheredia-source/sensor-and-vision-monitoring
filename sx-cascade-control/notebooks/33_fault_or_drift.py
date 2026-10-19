"""Drift or fault? Watch the trend of what the daily fit estimates.

The question. Notebook 32 refits the model once a day and tracks a slowly drifting plant. A plant
also has faults: crud building up in a block of mixers cuts how close those stages get to
equilibrium. To the controller, both just look like the product moving. The daily fit separates
them, because they move different numbers in different ways: drift moves the separation factor
slowly; fouled mixers move the stage-rate factor, and suddenly.

The method. Notebook 32's plant (seed 4, separation factor sliding about 6 % a week) and its
daily refit, plus a fault on day 7: eight scrub mixers (stages 10 to 17) drop to 40 % of their
transfer rate over six hours, as if crud had built up there. Each day the fit gives the two
separation factors and the stage-rate factor. The flag: the stage-rate factor more than 25 % away
from the median of the previous five days.

Run:  pixi run python notebooks/33_fault_or_drift.py [seed]
Writes runs/fault_or_drift_ndpr_plant.csv (one row per day)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
_src = open(HERE / "notebooks" / "31_identify_then_control.py", encoding="utf-8").read().split("done = set()")[0]
NB31 = {"__file__": str(HERE / "notebooks" / "31_identify_then_control.py")}
exec(_src, NB31)
readings, noise_sd, model_with = NB31["readings"], NB31["noise_sd"], NB31["model_with"]

from src.disturbances import EfficiencyLoss, Scenario                   # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.mpc import CircuitMPC                                          # noqa: E402
from src.registry import mpc_over_level_loops                         # noqa: E402
from src.operating_point import controller_settings                     # noqa: E402
from src.runs import RunSession, RunSpec                                # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
MAIN = CONFIG == "ndpr_main"            # the main case: the MPC over level loops, as notebook 14 runs it
ARGS = [a for i, a in enumerate(sys.argv[1:]) if a != "--config" and sys.argv[i] != "--config"]
SEED = int(ARGS[0]) if len(ARGS) > 0 else 4
HOURS, REFIT_EVERY_H, TREND = 336.0, 24.0, 0.06
FAULT_DAY, FAULT_STAGES, FAULT_FACTOR = 7, tuple(range(10, 18)), 0.4
FLAG_JUMP = 0.25
OUT = HERE / "runs" / f"fault_or_drift_{CONFIG}.csv"
FIELDS = ["seed", "day", "true_beta_pr_lace", "fit_beta_pr_lace", "fit_beta_nd_pr", "fit_k", "flag", "purity",
          "recovery", "seconds"]

spec = RunSpec(config=CONFIG, plant="drifting", controller="mpc_level_loops" if MAIN else "mpc", upset="none", hours=HOURS,
               run_file={"plant": {"seed": SEED, "generator": "ratios", "drift_trend": TREND}},
               controller_settings={"acid_floor": 1e-3})
s = RunSession(spec)
loop, plant, rc = s.loop, s.op.plant, s.op.rc


def new_controller(model, x_model, u_model):
    """The controller rebuilt on a (refitted) model: on the main case the MPC over the level loops
    already running; otherwise the MPC with its jacklegs, as first run."""
    if MAIN:
        return mpc_over_level_loops(rc, model, x_model, u_model, level_loops=loop.controller.level_loops)
    return CircuitMPC(model, x_model, u_model, rc.cut, rc.targets()[0], rc.design_purity,
                      rc.targets()[1], controller_settings(rc, acid_floor=1e-3))
loop.scenario = Scenario([EfficiencyLoss(t=FAULT_DAY * 24.0, ramp_h=6.0, stages=FAULT_STAGES, factor=FAULT_FACTOR)])
names = list(plant.p.names)
i_la, i_pr = names.index("LaCe"), names.index("Pr")
heavy = np.array(heavy_mask(plant.p, rc.cut))
fed = float(s.op.u_model.Q_feed) * np.asarray(s.op.u_model.c_feed, float)[heavy].sum()
rng = np.random.default_rng(SEED + 9000)
theta = np.zeros(3)
k_history = []


def true_beta(t):
    lm = plant.log_multipliers(t) - plant.log_multipliers(0.0)
    return plant.p.K[i_pr] / plant.p.K[i_la] * np.exp(lm[i_pr] - lm[i_la])


def refit(x_plant, u_now):
    global theta
    y_true = readings(plant, x_plant, u_now, names)
    sd = noise_sd(y_true)
    y_meas = y_true + rng.normal(0.0, sd)
    u_model = u_now.replace(k_factor=np.ones(plant.N))
    state = {"x": plant.observable(x_plant)}

    def residuals(th):
        model, _, _ = model_with(rc, th)
        x = steady_state(model, u_model, x0=state["x"])
        state["x"] = x
        return (readings(model, x, u_model, names) - y_meas) / sd

    theta = least_squares(residuals, theta, bounds=(-0.8, 0.5), diff_step=0.02, max_nfev=40).x
    model, beta_pl, beta_np = model_with(rc, theta)
    return model, beta_pl, beta_np, float(np.exp(theta[2])), u_model


with open(OUT, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()
t0 = time.time()
for rec in loop.steps(HOURS):
    if abs(rec.t % REFIT_EVERY_H) > 1e-6:
        continue
    day = int(round(rec.t / REFIT_EVERY_H))
    u_now = loop.controller.apply(loop.controller.handles())
    model, beta_pl, beta_np, k_fit, u_model = refit(rec.x, u_now)
    recent = k_history[-5:]
    flag = bool(recent) and abs(k_fit / np.median(recent) - 1) > FLAG_JUMP
    k_history.append(k_fit)
    x_model = steady_state(model, u_model, x0=plant.observable(rec.x))
    loop.controller = new_controller(model, x_model, u_model)
    o = plant.outlets(rec.x, rec.u)
    cp = np.atleast_1d(o["c_prod"])
    row = {"seed": SEED, "day": day, "true_beta_pr_lace": true_beta(rec.t), "fit_beta_pr_lace": beta_pl,
           "fit_beta_nd_pr": beta_np, "fit_k": k_fit, "flag": int(flag), "purity": product_purity(cp, plant.p, rc.cut),
           "recovery": float(o["Q_prod"]) * cp[heavy].sum() / fed, "seconds": round(time.time() - t0)}
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"day {day:2d}{' (fault starts)' if day == FAULT_DAY else '':15s} beta true {row['true_beta_pr_lace']:.3f} "
          f"fit {beta_pl:.3f}   stage-rate factor {k_fit:.2f}{'   <- FLAG: not drift' if flag else ''}   "
          f"purity {100 * row['purity']:6.2f} %  recovery {100 * row['recovery']:6.2f} %", flush=True)
