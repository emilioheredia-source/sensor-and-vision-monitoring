"""What should the controller's output bias do when a refitted model is swapped in?

The question (Emilio, Oct 22). The MPC carries an output bias: the filtered difference between what
the plant reports and what its model says (purity and recovery), its integral action. When the
model is refitted once a day and the controller rebuilt on it, notebooks 32 and 37 start the new
controller with no bias. Part of what the old bias carried is now explained by the new model, so
keeping it would count that twice; dropping it throws away what the refit didn't explain, and the
controller plans without correction for the first hours. Better: set the new bias so the new model
plus its bias gives exactly what the plant reports at the swap, as in a bumpless transfer.

The method. Notebook 32's main case: the drifting plant (the slow waves plus La+Ce's constant rising
6 % a week, so the Pr/(La+Ce) factor slides about 10 % in two weeks), seeds 11 and 4, 14 days, the
MPC over level loops, full state. Every 24 h the twelve readings (notebook 35's instruments, noise
from seed + 7000) are fitted with the corrected set (both separation factors and the acid exponent,
MAP; day 1 from four starts, then from the previous day's answer) and the controller is rebuilt on
the fitted model at the current handles, its state the fitted model's rest point there (decision 21:
the model state stays on the model's own trajectory). Four ways:
  fixed    never refitted (the reference)
  reset    the new controller starts with no bias (notebooks 32 and 37)
  keep     the new controller takes the old one's bias
  match    the new bias = what the plant reports now - what the new model gives now
Scored on purity and recovery along the run, the 12 h after each swap, and the size of the bias
just before and just after each swap (a refit that explains the plant better should leave a
smaller one).

Run:  pixi run python notebooks/46_model_swap.py
Writes runs/model_swap_ndpr_main.csv (one row per run) and runs/model_swap_ndpr_main.npz (traces).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.config import build_circuit                                    # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.registry import mpc_over_level_loops                            # noqa: E402
from src.runs import RunSession, RunSpec                                 # noqa: E402
from study_tools import (SPREAD, extended_model, notebook_35, run_jobs, separation_factors, settle,   # noqa: E402
                         workers, write_rows)

CONFIG = "ndpr_main"
HOURS, REFIT_EVERY_H, TREND = 336.0, 24.0, 0.06
SEEDS = (11, 4)
MODES = ("fixed", "reset", "keep", "match")
OUT = HERE / "runs" / f"model_swap_{CONFIG}.csv"
OUT_NPZ = HERE / "runs" / f"model_swap_{CONFIG}.npz"
SPEC = 0.99
FIT_SET = ("beta_pl", "beta_np", "dn_acid")     # the corrected fitted set (notebook 43)
STARTS = (0.0, 0.15, 0.3, -0.15)                # day 1's starts on Pr/(La+Ce); later days start from the last fit
BIAS_LIMITS = (0.2, 0.5)                        # purity, recovery: the clip the MPC puts on its own bias


class DailyRefit:
    """The once-a-day fit of the corrected set to the twelve readings of a drifting plant.

    Keeps the last answer (the next day's starting point) and the instruments' random stream (seed +
    7000, as notebook 32), so each day's fit follows on from the one before.
    """

    def __init__(self, plant, rc, m, seed: int):
        """plant: the drifting plant; rc: its run file; m: notebook 31 (readings and noise); seed: the plant's."""
        self.plant, self.rc, self.m = plant, rc, m
        self.names = list(plant.p.names)
        self.rng = np.random.default_rng(seed + 7000)
        self.prior = np.array([SPREAD[c] for c in FIT_SET])
        self.theta = None

    def __call__(self, x_plant, u_now):
        """Fit to noisy readings of the plant now; returns (fitted model, its inputs, its rest state there)."""
        from scipy.optimize import least_squares
        y_true = self.m.readings(self.plant, x_plant, u_now, self.names)
        sd = self.m.noise_sd(y_true)
        y = y_true + self.rng.normal(0.0, sd)
        u_model = u_now.replace(k_factor=np.ones(self.plant.N))
        residuals = self._residuals(y, sd, u_model, x_plant)
        starts = [self.theta] if self.theta is not None else [np.array([s0, 0.0, 0.0]) for s0 in STARTS]
        best = None
        for x0 in starts:
            f = least_squares(residuals, x0, bounds=([-0.5, -0.5, -1.0], [0.5, 0.5, 1.0]), diff_step=0.02, max_nfev=60)
            best = f if best is None or f.cost < best.cost else best
        self.theta = best.x
        circ, kf = extended_model(self.rc, dict(zip(FIT_SET, self.theta)))
        u_model = u_model.replace(k_factor=kf)
        return circ, u_model, settle(circ, u_model, x0=self.plant.observable(x_plant))

    def _residuals(self, y, sd, u_model, x_plant):
        """The MAP residual function for today's readings (settled solves, warm-started)."""
        state = {"x": self.plant.observable(x_plant)}

        def residuals(theta):
            """Noise-scaled reading errors and theta over its prior spread."""
            circ, kf = extended_model(self.rc, dict(zip(FIT_SET, theta)))
            u = u_model.replace(k_factor=kf)
            x = settle(circ, u, x0=state["x"])
            state["x"] = x
            return np.concatenate([(self.m.readings(circ, x, u, self.names) - y) / sd, np.asarray(theta) / self.prior])
        return residuals


def swapped_controller(loop, rc, refit_result, mode: str):
    """The controller rebuilt on the refitted model, its bias set by `mode`, and the bias sizes.

    reset: no bias (notebooks 32 and 37). keep: the old controller's. match: what the plant reports
    now minus what the new model gives now, so the new model plus its bias reproduces the plant at the
    swap (clipped as the MPC clips its own). Returns (controller, |old bias|, |new bias|), purity and
    recovery parts summed.
    """
    old = loop.controller
    circ, u_model, x_model = refit_result
    new = mpc_over_level_loops(rc, circ, x_model, u_model, level_loops=old.level_loops)
    if mode == "keep":
        new.bias = old.bias.copy()
    elif mode == "match":
        u_full = old._u_full(old._d_last)
        b = old._reported_outputs(loop.obs, u_full) - np.array(new._y_fun(new.x_m, u_full)).ravel()
        b[~np.isfinite(b)] = 0.0
        b[0] = float(np.clip(b[0], -BIAS_LIMITS[0], BIAS_LIMITS[0]))
        b[1] = float(np.clip(b[1], -BIAS_LIMITS[1], BIAS_LIMITS[1]))
        new.bias = b
    return new, np.abs(old.bias[:2]).sum(), np.abs(new.bias[:2]).sum()


def run(job):
    """One plant and one bias rule, 14 days with a daily refit (none for "fixed").

    Returns (the result row, the traces, the list of swaps with the fit and the true factor each day)."""
    seed, mode = job
    warnings.filterwarnings("ignore")
    s = RunSession(RunSpec(config=CONFIG, plant="drifting", controller="mpc_level_loops", upset="none", hours=HOURS,
                           run_file={"plant": {"seed": seed, "generator": "ratios", "drift_trend": TREND}},
                           controller_settings={"acid_floor": 1e-3}))
    loop, plant, rc = s.loop, s.op.plant, s.op.rc
    refit = DailyRefit(plant, rc, notebook_35(CONFIG).nb31(), seed)
    fed = float(s.op.u_model.Q_feed) * np.asarray(s.op.u_model.c_feed, float)[np.array(heavy_mask(plant.p, rc.cut))].sum()
    beta_model = separation_factors(build_circuit(rc))[0]
    series, swaps, t0 = {"t": [], "purity": [], "recovery": [], "bias": []}, [], time.time()
    for rec in loop.steps(HOURS):
        record_step(series, (plant, rc.cut, fed), rec, loop.controller)
        if mode == "fixed" or rec.t <= 0 or abs(rec.t % REFIT_EVERY_H) >= 1e-6:
            continue
        u_now = loop.controller.apply(loop.controller.handles())
        loop.controller, before, after = swapped_controller(loop, rc, refit(rec.x, u_now), mode)
        swaps.append({"t": rec.t, "bias_before": before, "bias_after": after, "dn_acid": float(refit.theta[2]),
                      "fit_beta": float(beta_model * np.exp(refit.theta[0])), "true_beta": true_beta(plant, rec.t)})
    traces = {f"{mode}_{seed}_{k}": np.array(v) for k, v in series.items()}
    return {"seed": seed, "mode": mode, **run_measures(series, swaps), "seconds": round(time.time() - t0)}, traces, swaps


def record_step(series: dict, plant_cut_fed: tuple, rec, controller) -> None:
    """Append one interval's time, purity, recovery and the controller's bias to the series.
    plant_cut_fed: (the plant, the cut, the NdPr fed at the design inputs, mol/h)."""
    plant, cut, fed = plant_cut_fed
    o = plant.outlets(rec.x, rec.u)
    cp = np.atleast_1d(o["c_prod"])
    series["t"].append(rec.t)
    series["purity"].append(product_purity(cp, plant.p, cut))
    series["recovery"].append(float(o["Q_prod"]) * cp[np.array(heavy_mask(plant.p, cut))].sum() / fed)
    series["bias"].append(np.asarray(controller.bias[:2], float).copy())


def true_beta(plant, t: float) -> float:
    """The drifting plant's Pr/(La+Ce) at time t."""
    names = list(plant.p.names)
    lm = plant.log_multipliers(t) - plant.log_multipliers(0.0)
    return float(separation_factors(plant)[0] * np.exp(lm[names.index("Pr")] - lm[names.index("LaCe")]))


def run_measures(series: dict, swaps: list) -> dict:
    """End, minimum and hours below spec; the 12 h after each swap; the mean bias before and after."""
    t, pur, rec = np.array(series["t"]), np.array(series["purity"]), np.array(series["recovery"])
    dt = np.diff(np.concatenate([[0.0], t]))
    after = np.zeros(len(t), bool)
    for sw in swaps:
        after |= (t > sw["t"]) & (t <= sw["t"] + 12.0)
    return {"purity_end": pur[-1], "recovery_end": rec[-1], "purity_min": pur.min(),
            "hours_below_spec": float(dt[pur < SPEC].sum()),
            "purity_min_12h_after_swaps": float(pur[after].min()) if after.any() else np.nan,
            "recovery_mean_12h_after_swaps": float(rec[after].mean()) if after.any() else np.nan,
            "bias_before_mean": float(np.mean([s["bias_before"] for s in swaps])) if swaps else np.nan,
            "bias_after_mean": float(np.mean([s["bias_after"] for s in swaps])) if swaps else np.nan,
            "refits": len(swaps)}


def main():
    """Every plant and bias rule; rows and traces written as each run finishes."""
    rows, traces = [], {}
    for _, (row, tr, swaps) in run_jobs(run, [(s, md) for s in SEEDS for md in MODES], workers()):
        rows.append(row)
        traces.update(tr)
        print(f"seed {row['seed']:2d} {row['mode']:6s}: purity end {100 * row['purity_end']:.2f} % (min {100 * row['purity_min']:.2f}, "
              f"{row['hours_below_spec']:.1f} h below spec), recovery end {100 * row['recovery_end']:.2f} %, "
              f"bias before/after swap {row['bias_before_mean']:.4f}/{row['bias_after_mean']:.4f}", flush=True)
        if swaps:
            print("    fit/true Pr/(La+Ce) by day: " + " ".join(f"{s['fit_beta']:.2f}/{s['true_beta']:.2f}" for s in swaps), flush=True)
        write_rows(OUT, rows, first=("seed", "mode"))
        np.savez(OUT_NPZ, **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
