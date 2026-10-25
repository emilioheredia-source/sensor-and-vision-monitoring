"""Is the net better than a least-squares fit, or was the fit handicapped?

The question (review session, Oct 18). Notebook 35 on the first plant: mean absolute error in the
Pr/(La+Ce) separation factor over notebook 31's ten plants, net 0.011, ridge 0.026, and the
least-squares fit started from the net's answer 0.107. The fit starts near the truth and walks
away from it. Four suspects: a third unknown (the common stage-rate factor k) that trades off
against the separation factor; the 60-evaluation cap; the finite-difference step (0.02) through a
steady-state solve; the solve's own tolerance.

The method. The same ten plants, the same noisy readings (rng seed + 5000, as notebooks 31 and 35),
the same start (the net's answer, k at 1), the fit re-run several ways:
  as_nb35          three unknowns, 60 evaluations, step 0.02 (notebook 35, a check)
  a_k_fixed        k held at 1: two unknowns, the net's job
  b_more_evals     as a, 300 evaluations
  c_small_step     as b, finite-difference step 0.005
  d_tight_solve    as c, the steady-state solve to 1e-11 instead of 1e-8
  e_three_strong   three unknowns again, with b, c and d's search (is it k, or the search?)
For each fit: the cost (sum of squared weighted residuals) where it stopped, and the cost at the
true separation factors with k at 1. If the fit stops at a lower cost than the truth, the search
did its job and the model, fitted to these readings, simply prefers a wrong answer: the plant
differs from the model in ways two or three numbers can't absorb.

Run:  pixi run python notebooks/42_polish_variants.py [--workers 4]
Writes runs/polish_variants_ndpr_plant.csv (one row per plant and variant).
"""
import csv
import sys
import time
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import notebook_35, separation_factors                  # noqa: E402

SEEDS = (2, 3, 6, 10, 14, 16, 20, 1, 7, 12)          # notebook 31's ten plants (seven hard, three easy)
HARD = (2, 3, 6, 10, 14, 16, 20)
VARIANTS = {   # name: (unknowns, max evaluations, finite-difference step, steady-state tolerance)
    "as_nb35": (3, 60, 0.02, 1e-8),
    "a_k_fixed": (2, 60, 0.02, 1e-8),
    "b_more_evals": (2, 300, 0.02, 1e-8),
    "c_small_step": (2, 300, 0.005, 1e-8),
    "d_tight_solve": (2, 300, 0.005, 1e-11),
    "e_three_strong": (3, 300, 0.005, 1e-11),
}
OUT = HERE / "runs" / "polish_variants_ndpr_plant.csv"
FIELDS = ["seed", "group", "variant", "beta_pr_lace_true", "beta_pr_lace_net", "beta_pr_lace_fit",
          "beta_nd_pr_true", "beta_nd_pr_fit", "k_fit", "cost_fit", "cost_at_truth", "evals", "status", "seconds"]


@dataclass
class Readings:
    """One test plant's noisy readings and what the fits need: the run file, the plant at rest, the
    model's inputs (k = 1), the readings and their noise, the net's estimate, the true log factors, the
    model's own factors and notebook 31 (its model builder and readings)."""
    rc: object
    plant: object
    x_rest: np.ndarray
    u_model: object
    y: np.ndarray
    sd: np.ndarray
    est_net: np.ndarray
    truth: np.ndarray
    base: np.ndarray
    m: object


def plant_readings(seed: int) -> Readings:
    """The first plant's test plant `seed` at rest on the design handles, read with notebook 31's noise."""
    from src.commission import commission
    from src.config import build_circuit, nominal_inputs
    n35 = notebook_35("ndpr_plant")
    m = n35.nb31()
    rc, plant, plant_k = n35.plant_for(seed)
    u0 = nominal_inputs(rc, plant).replace(k_factor=plant_k)
    x_rest, u_design = commission(plant, u0)
    y_true = m.readings(plant, x_rest, u_design, list(plant.p.names))
    sd = m.noise_sd(y_true)
    y = y_true + np.random.default_rng(seed + 5000).normal(0.0, sd)
    model = build_circuit(rc)
    return Readings(rc, plant, x_rest, u_design.replace(k_factor=np.ones(plant.N)), y, sd,
                    n35.load_estimators()[0](y, n35.handles_of(u_design, u0)), n35.true_log_betas(plant, model),
                    separation_factors(model), m)


def residuals_for(r: Readings, tol: float):
    """Notebook 31's residual function on these readings, the steady state solved to `tol` and
    warm-started from the last solve. Returns (residuals, state counting evaluations)."""
    from src.simulate import steady_state
    state = {"x": r.plant.observable(r.x_rest), "evals": 0}

    def residuals(theta3):
        """Noise-scaled reading errors of notebook 31's model with these three log numbers."""
        mod, _, _ = r.m.model_with(r.rc, theta3)
        x = steady_state(mod, r.u_model, x0=state["x"], tol=tol)
        state["x"], state["evals"] = x, state["evals"] + 1
        return (r.m.readings(mod, x, r.u_model, list(r.plant.p.names)) - r.y) / r.sd
    return residuals, state


def fit_variant(r: Readings, name: str):
    """One variant's fit from the net's answer: (log numbers with k, the fit, evaluations, cost at the truth)."""
    from scipy.optimize import least_squares
    n_unknowns, nfev, step, tol = VARIANTS[name]
    residuals, state = residuals_for(r, tol)
    cost_truth = 0.5 * float(np.sum(residuals([r.truth[0], r.truth[1], 0.0]) ** 2))
    state["evals"] = 0
    if n_unknowns == 3:
        fit = least_squares(residuals, np.clip([r.est_net[0], r.est_net[1], 0.0], -0.49, 0.49),
                            bounds=(-0.5, 0.5), diff_step=step, max_nfev=nfev)
        return fit.x, fit, state["evals"], cost_truth
    fit = least_squares(lambda th: residuals([th[0], th[1], 0.0]), np.clip(r.est_net[:2], -0.49, 0.49),
                        bounds=(-0.5, 0.5), diff_step=step, max_nfev=nfev)
    return np.array([fit.x[0], fit.x[1], 0.0]), fit, state["evals"], cost_truth


def one_plant(seed: int) -> list:
    """Every variant on one plant; a list of rows."""
    warnings.filterwarnings("ignore")
    r = plant_readings(seed)
    rows = []
    for name in VARIANTS:
        t0 = time.time()
        theta, fit, evals, cost_truth = fit_variant(r, name)
        rows.append({"seed": seed, "group": "hard" if seed in HARD else "easy", "variant": name,
                     "beta_pr_lace_true": r.base[0] * np.exp(r.truth[0]), "beta_pr_lace_net": r.base[0] * np.exp(r.est_net[0]),
                     "beta_pr_lace_fit": r.base[0] * np.exp(theta[0]), "beta_nd_pr_true": r.base[1] * np.exp(r.truth[1]),
                     "beta_nd_pr_fit": r.base[1] * np.exp(theta[1]), "k_fit": float(np.exp(theta[2])),
                     "cost_fit": float(fit.cost), "cost_at_truth": cost_truth, "evals": evals,
                     "status": int(fit.status), "seconds": round(time.time() - t0)})
        row = rows[-1]
        print(f"seed {seed:2d} {name:15s} Pr/(La+Ce) true {row['beta_pr_lace_true']:.3f} net {row['beta_pr_lace_net']:.3f} "
              f"fit {row['beta_pr_lace_fit']:.3f}  k {row['k_fit']:.2f}  cost fit {row['cost_fit']:.1f} at truth "
              f"{row['cost_at_truth']:.1f}  ({row['evals']} evals, status {row['status']}, {row['seconds']} s)", flush=True)
    return rows


def summary(rows: list) -> None:
    """Mean absolute error in Pr/(La+Ce) per variant, and on how many plants the fit beat the truth's cost."""
    print("\nmean absolute error in Pr/(La+Ce), ten plants (net 0.011 in notebook 35):")
    net = np.mean([abs(r["beta_pr_lace_net"] - r["beta_pr_lace_true"]) for r in rows if r["variant"] == "as_nb35"])
    print(f"  {'net':15s} {net:.3f}")
    for name in VARIANTS:
        sel = [r for r in rows if r["variant"] == name]
        err = np.mean([abs(r["beta_pr_lace_fit"] - r["beta_pr_lace_true"]) for r in sel])
        below = sum(r["cost_fit"] < r["cost_at_truth"] for r in sel)
        print(f"  {name:15s} {err:.3f}   fit cost below the truth's on {below} of {len(sel)}")


if __name__ == "__main__":
    n_workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    with Pool(n_workers) as pool:
        results = pool.map(one_plant, SEEDS)
    all_rows = [r for plant_rows in results for r in plant_rows]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    summary(all_rows)
    print("wrote", OUT.name)
