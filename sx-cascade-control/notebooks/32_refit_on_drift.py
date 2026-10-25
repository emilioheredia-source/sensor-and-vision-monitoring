"""Track a drifting plant: refit the model from the instruments once a day.

The question. Notebook 31 fitted the model once, at rest, and the controller then worked on plants
where the nominal model failed. A real plant keeps changing: an extractant that degrades between
make-ups lowers the separation factor week by week. Does refitting once a day keep the controller
on spec while the plant slides, where a controller that never refits loses it?

The method. The first plant (30 L mixers, edge targets 99.5 % / 95 %, the acidity floor on), a
plant drawn with its Pr/(La+Ce) separation factor near the model's 2.0, made to drift
(plant "drifting": the usual slow waves, plus La+Ce's constant rising 6 % a week, so the
separation factor slides from about 2.0 toward 1.77 over two weeks, into the range where the
nominal model failed). Two weeks, no other upset:
  fixed      the MPC on the nominal model, never updated
  refit      every 24 h: read XRF on product and raffinate and UV-Vis Pr and Nd at stages 2, 4, 6
             (with noise, notebook 31's instruments), fit the model's two separation factors and
             stage-rate factor to them, and rebuild the MPC on the fitted model at the current flows
Scored on NdPr made off spec and lost beyond the start, purity and recovery over time, and how
close each daily fit came to the plant's true separation factor.

Run:  pixi run python notebooks/32_refit_on_drift.py [seed] [fixed|refit]
Writes runs/refit_on_drift_ndpr_plant.csv (one row per day per run)
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
exec(_src, NB31)                                 # notebook 31's readings, noise and fitted-model builder
readings, noise_sd, model_with = NB31["readings"], NB31["noise_sd"], NB31["model_with"]

from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.mpc import CircuitMPC                                          # noqa: E402
from src.registry import mpc_over_level_loops                         # noqa: E402
from src.operating_point import controller_settings                     # noqa: E402
from src.runs import RunSession, RunSpec                                # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
MAIN = CONFIG == "ndpr_main"            # the main case: the MPC over level loops, as notebook 14 runs it
ARGS = [a for i, a in enumerate(sys.argv[1:]) if a != "--config" and sys.argv[i] != "--config"]
SEED = int(ARGS[0]) if len(ARGS) > 0 else 11
MODE = ARGS[1] if len(ARGS) > 1 else "refit"
HOURS, REFIT_EVERY_H = 336.0, 24.0
TREND = 0.06                                     # log units per week on La+Ce's constant
OUT = HERE / "runs" / f"refit_on_drift_{CONFIG}.csv"
FIELDS = ["seed", "mode", "day", "true_beta_pr_lace", "fit_beta_pr_lace", "purity", "recovery",
          "min_purity_day", "Q_org", "Q_scrub", "seconds"]

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
names = list(plant.p.names)
i_la, i_pr = names.index("LaCe"), names.index("Pr")
heavy = np.array(heavy_mask(plant.p, rc.cut))
fed = float(s.op.u_model.Q_feed) * np.asarray(s.op.u_model.c_feed, float)[heavy].sum()
rng = np.random.default_rng(SEED + 7000)
theta = np.zeros(3)


def true_beta(t):
    """The drifting plant's Pr/(La+Ce) separation factor at time t."""
    lm = plant.log_multipliers(t) - plant.log_multipliers(0.0)
    return plant.p.K[i_pr] / plant.p.K[i_la] * np.exp(lm[i_pr] - lm[i_la])


def refit(x_plant, u_now):
    """Fit the model to noisy readings of the plant now; returns theta and the fitted model."""
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

    fit = least_squares(residuals, theta, bounds=(-0.5, 0.5), diff_step=0.02, max_nfev=40)
    theta = fit.x
    model, beta_pl, _ = model_with(rc, theta)
    return model, beta_pl, u_model


with open(OUT, "a", newline="") as f:
    if f.tell() == 0:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
t0 = time.time()
fit_beta = np.nan
day_min = 1.0
for rec in loop.steps(HOURS):
    o = plant.outlets(rec.x, rec.u)
    cp = np.atleast_1d(o["c_prod"])
    pur = product_purity(cp, plant.p, rc.cut)
    day_min = min(day_min, pur)
    if abs(rec.t % REFIT_EVERY_H) < 1e-6:
        day = int(round(rec.t / REFIT_EVERY_H))
        u_now = loop.controller.apply(loop.controller.handles())
        if MODE == "refit":
            model, fit_beta, u_model = refit(rec.x, u_now)
            x_model = steady_state(model, u_model, x0=plant.observable(rec.x))
            loop.controller = new_controller(model, x_model, u_model)
        rec_frac = float(o["Q_prod"]) * cp[heavy].sum() / fed
        row = {"seed": SEED, "mode": MODE, "day": day, "true_beta_pr_lace": true_beta(rec.t),
               "fit_beta_pr_lace": fit_beta, "purity": pur, "recovery": rec_frac, "min_purity_day": day_min,
               "Q_org": float(u_now.Q_org), "Q_scrub": float(u_now.Q_scrub), "seconds": round(time.time() - t0)}
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"seed {SEED} {MODE:5s} day {day:2d}: true beta {row['true_beta_pr_lace']:.3f} fit {fit_beta:.3f}  "
              f"purity {100 * pur:6.2f} % (lowest today {100 * day_min:6.2f})  recovery {100 * rec_frac:6.2f} %  "
              f"organic {row['Q_org']:5.0f} scrub {row['Q_scrub']:5.0f}  ({row['seconds']} s)", flush=True)
        day_min = 1.0
