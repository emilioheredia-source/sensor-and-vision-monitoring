"""Supervised model updates: the same drift and fault as notebook 33, with every fit reviewed first.

The question. Notebook 33 put every daily fit straight into the controller. After eight scrub
mixers fouled on day 7, the fit's stage-rate factor jumped and sat on the limit the fit allows,
and the controller ran on that model: purity settled at 99.45 to 99.5 %, on or under the spec.
What changes if each fit is a proposal, checked and reviewed before the controller uses it?

The method. Notebook 33's run exactly (seed 4, drift of 6 % a week, the fault on day 7, the same
instruments and noise draws, the MPC rebuilt every day), except that each day's fit goes into the
model registry (src/model_registry.py) as a proposal with four checks: the step from the model in
use, any fitted number on its bound, the fit residual, the range of the separation factors. A
scripted reviewer stands in for the engineer: it approves and puts in use a version whose deciding
checks (step, bounds, range) all pass, and rejects the rest, keeping the model in use. A first run
let the residual decide too; it rejected every fit (runs/supervised_updates_*_first_gate.*). The registry, with every version and
every decision, is saved as JSON.

Run:  pixi run python notebooks/37_supervised_updates.py [seed]
Writes runs/supervised_updates_ndpr_plant.csv (one row per day) and runs/supervised_updates_registry.json
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
from src.model_registry import ModelRegistry              # noqa: E402
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
OUT = HERE / "runs" / f"supervised_updates_{CONFIG}.csv"
REGISTRY = HERE / "runs" / ("supervised_updates_registry.json" if not MAIN else f"supervised_updates_registry_{CONFIG}.json")
BOUNDS = (-0.8, 0.5)
FIELDS = ["seed", "day", "true_beta_pr_lace", "fit_beta_pr_lace", "fit_beta_nd_pr", "fit_k", "flag", "purity",
          "recovery", "decision", "failed_checks", "active_version", "seconds"]

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
    """The drifting plant's Pr/(La+Ce) separation factor at time t."""
    lm = plant.log_multipliers(t) - plant.log_multipliers(0.0)
    return plant.p.K[i_pr] / plant.p.K[i_la] * np.exp(lm[i_pr] - lm[i_la])


def refit(x_plant, u_now):
    """Fit the model to noisy readings of the plant now, from the last answer: (model, Pr/(La+Ce) factor,
    Nd/Pr factor, stage-rate factor, the model's inputs, the fit's mean squared scaled residual)."""
    global theta
    y_true = readings(plant, x_plant, u_now, names)
    sd = noise_sd(y_true)
    y_meas = y_true + rng.normal(0.0, sd)
    u_model = u_now.replace(k_factor=np.ones(plant.N))
    state = {"x": plant.observable(x_plant)}

    def residuals(th):
        """Noise-scaled reading errors of the model with these numbers (warm-started steady state)."""
        model, _, _ = model_with(rc, th)
        x = steady_state(model, u_model, x0=state["x"])
        state["x"] = x
        return (readings(model, x, u_model, names) - y_meas) / sd

    fit = least_squares(residuals, theta, bounds=BOUNDS, diff_step=0.02, max_nfev=40)
    theta = fit.x
    model, beta_pl, beta_np = model_with(rc, theta)
    return model, beta_pl, beta_np, float(np.exp(theta[2])), u_model, 2 * float(fit.cost) / len(y_meas)


base_model, base_pl, base_np = model_with(rc, np.zeros(3))
registry = ModelRegistry(dict(zip(("log_beta_pr_lace", "log_beta_nd_pr", "log_k"), np.zeros(3))), (base_pl, base_np), BOUNDS,
                         path=REGISTRY)


def review(v, day):
    """The scripted reviewer: approve and put in use when every check passes, reject otherwise.
    Day 1 is the plant's first identification against the nominal model, reviewed at commissioning,
    so its step check is waived."""
    failed = [k for k, c in v.checks.items() if c["decides"] and not c["passed"] and not (day == 1 and k == "step")]
    if failed:
        registry.reject(v.id, "reviewer (scripted)", "checks failed: " + "; ".join(v.checks[k]["detail"] for k in failed))
        return "rejected", failed
    registry.approve(v.id, "reviewer (scripted)", "first identification" if day == 1 else "all checks pass")
    registry.activate(v.id)
    return "approved", failed


with open(OUT, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=FIELDS).writeheader()
t0 = time.time()
for rec in loop.steps(HOURS):
    if abs(rec.t % REFIT_EVERY_H) > 1e-6:
        continue
    day = int(round(rec.t / REFIT_EVERY_H))
    u_now = loop.controller.apply(loop.controller.handles())
    _, beta_pl, beta_np, k_fit, u_model, residual = refit(rec.x, u_now)
    version = registry.propose(rec.t, theta, f"daily fit, day {day}", residual)
    decision, failed = review(version, day)
    active = registry.active.params
    model, _, _ = model_with(rc, [active["log_beta_pr_lace"], active["log_beta_nd_pr"], active["log_k"]])
    recent = k_history[-5:]
    flag = bool(recent) and abs(k_fit / np.median(recent) - 1) > FLAG_JUMP
    k_history.append(k_fit)
    x_model = steady_state(model, u_model, x0=plant.observable(rec.x))
    loop.controller = new_controller(model, x_model, u_model)
    o = plant.outlets(rec.x, rec.u)
    cp = np.atleast_1d(o["c_prod"])
    row = {"seed": SEED, "day": day, "true_beta_pr_lace": true_beta(rec.t), "fit_beta_pr_lace": beta_pl,
           "fit_beta_nd_pr": beta_np, "fit_k": k_fit, "flag": int(flag), "purity": product_purity(cp, plant.p, rc.cut),
           "recovery": float(o["Q_prod"]) * cp[heavy].sum() / fed, "decision": decision,
           "failed_checks": " ".join(failed), "active_version": registry.active_id, "seconds": round(time.time() - t0)}
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"day {day:2d}{' (fault starts)' if day == FAULT_DAY else '':15s} beta true {row['true_beta_pr_lace']:.3f} "
          f"fit {beta_pl:.3f}   stage-rate factor {k_fit:.2f}{'   <- FLAG: not drift' if flag else ''}   "
          f"purity {100 * row['purity']:6.2f} %  recovery {100 * row['recovery']:6.2f} %  {decision} {' '.join(failed)}",
          flush=True)
