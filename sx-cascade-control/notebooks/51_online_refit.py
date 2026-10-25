"""Can the controller learn the plant's numbers while it runs? (v1.3, first test)

The question (Oct 24). On plant 10 of the fair test, with the ridge's estimate (Pr/(La+Ce) about 3 %
optimistic), the MPC raised the saponification and cut the scrub acid until the acid ran out in the
extraction and scrub stages. The plant ran 2-5 points below what the model expected the whole time,
and the controller carried that gap as an output bias, which says nothing about why. Where the
uncertain numbers are known, the practice in the literature is to estimate them on the running plant
and update the model (Huang, Patwardhan and Biegler, DYCOPS 2010), and a correction where the error
enters is more robust than one on the outputs (Pannocchia and Rawlings, 2001). Here the uncertain
numbers are known: both separation factors and the acid exponent (notebook 43, Test 2).

The method. Plant 10 of notebook 45 (perturbed physics: its stage rates and more differ from the
model too), at rest on the design handles, the MPC over level loops, full state, 48 h. Four
controllers:
  ridge        the ridge's estimate, as in notebook 45 (the run that over-saponified)
  true         the plant's own model, the reference
  fit_at_rest  the three numbers fitted once at rest, before the controller starts (notebook 43's
               multistart fit to the twelve readings)
  ridge_factors_fit_acid, fit_factors_nominal_acid
               the two swaps: the ridge's factors with the fitted acid exponent, and the fitted
               factors with the nominal one (which of the two the ridge run was missing)
  online       starts on the ridge's estimate; every REFIT_H the three numbers are fitted to the
               readings over the last WINDOW_H (the model run from the plant's state at the window's
               start, under the inputs the plant got), and the controller is rebuilt on the new
               model: its state the new model's own at the window's end, its bias what the plant
               reports minus what the new model gives there (notebook 46's "match")
Every step records purity, recovery, the four handles and the lowest acid in the extraction and the
scrub stages; the online run also records each fit.

Run:  pixi run python notebooks/51_online_refit.py [variant ...]
(named variants only, their rows replacing the same variants' in the files; default all)
Writes runs/online_refit_ndpr_main.csv (one row per controller) and .npz (the traces).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import warnings                                                         # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from src.circuit import Inputs                                          # noqa: E402
from src.loop import ClosedLoop                                         # noqa: E402
from src.mpc import BIAS_LIMIT_PURITY, BIAS_LIMIT_RECOVERY              # noqa: E402
from src.registry import mpc_over_level_loops                           # noqa: E402
from src.simulate import implicit_euler                                 # noqa: E402
from study_tools import (SPEC, extended_model, load_notebook, notebook_35, purity_recovery,   # noqa: E402
                         run_jobs, settle, workers, write_rows)

CONFIG = "ndpr_main"
SEED = 10
HOURS = 48.0
REFIT_H = 2.0                                   # h between refits (online)
WINDOW_H = 2.0                                  # h of readings each refit uses
STEP_H = 0.5                                    # h, the control interval
FIT_SET = ("beta_pl", "beta_np", "dn_acid")     # the corrected fitted set (notebook 43)
VARIANTS = ("ridge", "true", "fit_at_rest", "online", "ridge_factors_fit_acid", "fit_factors_nominal_acid")
HANDLES = ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")
OUT = HERE / "runs" / f"online_refit_{CONFIG}.csv"


def notebooks():
    """Notebooks 43 and 45 (the fair test's fit and its plants)."""
    nb43 = load_notebook("nb43", "43_fair_fit.py", ("--config", CONFIG))
    return nb43, load_notebook("nb45", "45_net_size.py")


def fit_at_rest(case, nb43, seed: int) -> np.ndarray:
    """Notebook 43's multistart fit of the three numbers to the plant's readings at rest."""
    m = notebook_35(CONFIG).nb31()
    residuals, _ = nb43.set_residuals(m, case, FIT_SET, nb43.noisy_readings(m, case, seed))
    return nb43.multistart_fit(residuals, FIT_SET).x


def starting_model(variant: str, case, ridge_theta, fit):
    """The model the controller starts on and its stage rates: (circuit, k_factor, theta or None).
    fit: the fit at rest (None when the variant doesn't use it). The two swaps take one part from
    the fit at rest and the other from the ridge: the factors (theta[:2]) or the acid exponent (theta[2])."""
    if variant == "true":
        return case.plant, case.plant_k, None
    theta = np.asarray(ridge_theta, float)
    if fit is not None:
        theta = {"fit_at_rest": fit, "ridge_factors_fit_acid": np.r_[theta[:2], fit[2]],
                 "fit_factors_nominal_acid": np.r_[fit[:2], 0.0]}[variant]
    circ, kf = extended_model(case.rc, dict(zip(FIT_SET, theta)))
    return circ, kf, np.asarray(theta, float)


class WindowRefit:
    """The three numbers fitted to the readings of the last WINDOW_H, the model run from the plant's
    state at the window's start under the inputs the plant got (a short moving-horizon fit).

    Keeps the history of the run (plant states, inputs, noisy readings), the last answer (the next
    fit's start) and the instruments' random stream (seed + 7000, as notebook 46).
    """

    def __init__(self, case, theta0, seed: int):
        """case: the plant at rest (PlantCase); theta0: the numbers the controller starts on; seed:
        the plant's (the readings' noise is drawn from seed + 7000)."""
        self.case, self.theta = case, np.asarray(theta0, float)
        self.m = notebook_35(CONFIG).nb31()
        self.names = list(case.plant.p.names)
        self.rng = np.random.default_rng(seed + 7000)
        self.states, self.inputs, self.readings, self.sds = [case.plant.observable(case.x_rest)], [], [], []

    def record(self, x_plant, u_plant):
        """One interval's end state, the inputs over it and the noisy readings at its end."""
        plant = self.case.plant
        y_true = self.m.readings(plant, x_plant, Inputs.from_vector(u_plant, plant.n, plant.N), self.names)
        sd = self.m.noise_sd(y_true)
        self.states.append(plant.observable(x_plant))
        self.inputs.append(np.asarray(u_plant, float))
        self.readings.append(y_true + self.rng.normal(0.0, sd))
        self.sds.append(sd)

    def fit(self, bounds):
        """Fit the window from the last answer; returns (circuit, k_factor, model state at the window's end)."""
        from scipy.optimize import least_squares
        n = int(round(WINDOW_H / STEP_H))
        f = least_squares(self._residuals(n), self.theta, bounds=bounds, diff_step=0.02, max_nfev=60)
        self.theta = f.x
        circ, kf = extended_model(self.case.rc, dict(zip(FIT_SET, self.theta)))
        return circ, kf, self._run_window(circ, kf, n)[0]

    def _run_window(self, circ, kf, n: int):
        """The model over the last n intervals from the plant's state at their start: (end state, its
        noise-scaled reading errors)."""
        x, errors = self.states[-n - 1].copy(), []
        for u_plant, y, sd in zip(self.inputs[-n:], self.readings[-n:], self.sds[-n:]):
            u = Inputs.from_vector(u_plant, circ.n, circ.N).replace(k_factor=kf)
            x = implicit_euler(circ, x, u.numpy(), STEP_H)
            errors.append((self.m.readings(circ, x, u, self.names) - y) / sd)
        return x, np.concatenate(errors)

    def _residuals(self, n: int):
        """The residual function over the last n intervals."""
        def residuals(theta):
            """Noise-scaled reading errors of the model with these numbers."""
            circ, kf = extended_model(self.case.rc, dict(zip(FIT_SET, theta)))
            return self._run_window(circ, kf, n)[1]
        return residuals


def rebuilt_controller(loop, rc, refit_result):
    """The controller rebuilt on the refitted model at the handles held now: its state the model's
    own at the window's end, its bias what the plant reports minus what the new model gives there."""
    old = loop.controller
    circ, kf, x_end = refit_result
    u_model = old.apply(old.handles()).replace(k_factor=kf)
    new = mpc_over_level_loops(rc, circ, settle(circ, u_model, x0=x_end), u_model, level_loops=old.level_loops)
    new.x_m = np.asarray(x_end, float).copy()
    u_full = old._u_full(old._d_last)
    b = old._reported_outputs(loop.obs, u_full) - np.array(new._y_fun(new.x_m, u_full)).ravel()
    b[~np.isfinite(b)] = 0.0
    b[0] = float(np.clip(b[0], -BIAS_LIMIT_PURITY, BIAS_LIMIT_PURITY))
    b[1] = float(np.clip(b[1], -BIAS_LIMIT_RECOVERY, BIAS_LIMIT_RECOVERY))
    new.bias = b
    return new


def lowest_acid(plant, x) -> tuple:
    """The lowest acid in the extraction stages and in the scrub stages, mmol/L."""
    h = np.asarray(plant.unpack(plant.observable(x))[2], float)
    sections = np.array([plant.cmap.section(j) for j in range(plant.N)])
    return 1000 * h[sections == "extraction"].min(), 1000 * h[sections == "scrub"].min()


def record_step(series: dict, case, rec) -> None:
    """Append one interval's time, purity, recovery, handles and lowest acids to the series."""
    plant = case.plant
    lay = Inputs.layout(plant.n, plant.N)
    pur, recov = purity_recovery(plant, rec.x, rec.u, case.rc.cut)
    acid_ext, acid_scrub = lowest_acid(plant, rec.x)
    for k, v in (("t", rec.t), ("purity", pur), ("recovery", recov), ("acid_extraction", acid_ext),
                 ("acid_scrub", acid_scrub), *((h, float(np.asarray(rec.u)[lay[h].start])) for h in HANDLES)):
        series.setdefault(k, []).append(v)


def run(variant: str, after_step=None, label: str | None = None, seed: int = SEED):
    """One controller on one of the fair test's plants (default plant 10) for HOURS; returns (the
    result row, the traces). after_step(loop, rec, series): called after every interval (and any
    refit), for studies that add one change to these runs (notebooks 52, 53); label: the row's name
    (default the variant)."""
    warnings.filterwarnings("ignore")
    nb43, nb45 = notebooks()
    case, y, h, truth = nb45.plant_case(seed)
    case.x_rest = settle(case.plant, case.u_design.replace(k_factor=case.plant_k), x0=case.x_rest)
    ridge = notebook_35(CONFIG).load_estimators()[1](y, h)
    fit = fit_at_rest(case, nb43, seed) if variant in ("fit_at_rest", "ridge_factors_fit_acid",
                                                        "fit_factors_nominal_acid") else None
    circ, kf, theta = starting_model(variant, case, (float(ridge[0]), float(ridge[1]), 0.0), fit)
    u_model = case.u_design.replace(k_factor=kf)
    ctl = mpc_over_level_loops(case.rc, circ, settle(circ, u_model, x0=case.plant.observable(case.x_rest)), u_model)
    loop = ClosedLoop(case.plant, case.x_rest, ctl, None, None, None, np.asarray(case.plant_k) / np.asarray(kf))
    refit = WindowRefit(case, theta, seed) if variant == "online" else None
    series, fits, t0 = {}, [], time.time()
    stalled = drive(loop, (case, refit, nb43.fit_bounds(FIT_SET), after_step), series, fits)
    label = label or variant
    row = summary(label, series, (stalled, theta, fits, truth))
    traces = {f"{label}_{k}": np.array(v) for k, v in series.items()}
    traces[f"{label}_fits"] = np.array(fits) if fits else np.zeros((0, 4))
    return {**row, "seed": seed, "seconds": round(time.time() - t0)}, traces


def drive(loop, setup: tuple, series: dict, fits: list):
    """Run the loop, refitting when due; returns the hour the plant simulation stalled, or None.
    setup: (case, the refit or None, the fit's bounds, after_step or None)."""
    try:
        run_steps(loop, setup, series, fits)
    except RuntimeError:
        return series["t"][-1] if series.get("t") else 0.0
    return None


def run_steps(loop, setup: tuple, series: dict, fits: list) -> None:
    """Every control interval to HOURS: record it, refit when due (online only), then after_step."""
    case, refit, _, after_step = setup
    for rec in loop.steps(HOURS):
        record_step(series, case, rec)
        if refit is not None:
            refit_if_due(loop, rec, setup, fits)
        if after_step is not None:
            after_step(loop, rec, series)


def refit_if_due(loop, rec, setup: tuple, fits: list) -> None:
    """Record the interval for the refit; every REFIT_H (once a full window is in) refit and rebuild
    the controller, and append [t, numbers] to fits. setup: (case, the refit, the fit's bounds, _)."""
    case, refit, bounds, _ = setup
    refit.record(rec.x, rec.u)
    if rec.t < WINDOW_H or abs(rec.t % REFIT_H) >= 1e-6:
        return
    loop.controller = rebuilt_controller(loop, case.rc, refit.fit(bounds))
    fits.append([rec.t, *refit.theta])


def summary(variant: str, series: dict, outcome: tuple) -> dict:
    """The row for one controller. outcome: (hour stalled or None, starting numbers, fits, true log factors)."""
    stalled, theta0, fits, truth = outcome
    t, pur = np.array(series["t"]), np.array(series["purity"])
    dt = np.diff(np.concatenate([[0.0], t]))
    last = fits[-1][1:] if fits else theta0
    return {"variant": variant, "stalled_at_h": "" if stalled is None else stalled, "hours_run": t[-1],
            "purity_min": pur.min(), "purity_end": pur[-1], "recovery_end": series["recovery"][-1],
            "hours_below_spec": float(dt[pur < SPEC].sum()), "sapon_max": max(series["saponified_fraction"]),
            "scrub_acid_min": min(series["h_scrub"]), "acid_extraction_min_mM": min(series["acid_extraction"]),
            "acid_scrub_min_mM": min(series["acid_scrub"]), "refits": len(fits),
            "beta_pl_log_err_start": "" if theta0 is None else float(theta0[0] - truth[0]),
            "beta_pl_log_err_end": "" if last is None else float(last[0] - truth[0]),
            "dn_acid_end": "" if last is None else float(last[2])}


def earlier_results(variants: tuple) -> tuple:
    """The rows and traces already written for the other variants: (rows, traces)."""
    import csv
    if not OUT.exists():
        return [], {}
    rows = [r for r in csv.DictReader(open(OUT)) if r["variant"] not in variants]
    with np.load(OUT.with_suffix(".npz")) as d:
        traces = {k: d[k] for k in d.files if not any(k.startswith(v + "_") for v in variants)}
    return rows, traces


def main():
    """The named controllers (default all) in parallel; the rows and traces as each finishes."""
    variants = tuple(a for a in sys.argv[1:] if a in VARIANTS) or VARIANTS
    rows, traces = earlier_results(variants)
    for _, result in run_jobs(run, variants, min(workers(), len(variants))):
        if result is None:
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        write_rows(OUT, rows, first=("variant",))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
