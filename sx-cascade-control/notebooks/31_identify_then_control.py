"""Measure the plant, fit the model's numbers to it, then control with the fitted model.

The question. On the first plant (30 L mixers), the MPC with the nominal model moved the organic
the wrong way on the seven plants that separate worse than the model, and with the plant's true
model it brought all seven to spec (notebook 21, with the acidity floor). A real plant never hands
over its true model. Can a few realistic measurements, fitted to the physics model, get close
enough to the true model for the controller to work?

The method, per plant:
  1. The plant at rest on the design flows (commissioned, nobody touching it).
  2. What instruments would read there, with noise:
       XRF on the product and on the raffinate      every element, 2 % (relative)
       UV-Vis on stages 2, 4 and 6 (extraction)     Pr and Nd only, 1 % (relative)
  3. Fit three numbers of the model by weighted least squares (scipy least_squares), each trial
     a model rebuilt with those numbers and solved at steady state on the same flows:
       the Pr/(La+Ce) separation factor, the Nd/Pr separation factor, a common factor on the
       stage transfer rate (how close stages get to equilibrium)
     Everything else stays at the model's values: the plant's other differences (acid exponent
     3.3, densities, hold-up, each stage's own efficiency) are not fitted, as they wouldn't be.
  4. The MPC (acidity floor on, as in notebook 21) built on the fitted model runs the plant from
     the design flows for 120 h.
Compared with notebook 21's nominal-model and true-model runs on the same plants.

Instrument sets (--sensors):
  xrf_uv        XRF on product and raffinate, UV-Vis on three stages (the default)
  xrf_only      XRF on product and raffinate: roughly what plants already run
  uv_relative   only the UV-Vis on three stages, used as ratios: each stage's Pr and Nd against stage
                6's, and Nd/Pr within each spectrum. The readings carry an unknown 15 % gain error,
                which the ratios cancel: relative numbers only.

Run:  pixi run python notebooks/31_identify_then_control.py [--sensors SET] [seed ...]
Writes runs/identify_then_control_ndpr_plant.csv (xrf_uv) or ..._<SET>_ndpr_plant.csv (row by row)
"""
import csv
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, Inputs                                  # noqa: E402
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.loop import run_closed_loop                                    # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.mpc import CircuitMPC                                          # noqa: E402
from src.operating_point import controller_settings                     # noqa: E402
from src.registry import PLANTS, mpc_over_level_loops                  # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
MAIN = CONFIG == "ndpr_main"            # the main case: plants 1 to 10, the MPC over level loops
HOURS = 120.0
HARD, EASY = (2, 3, 6, 10, 14, 16, 20), (1, 7, 12)
SENSORS = sys.argv[sys.argv.index("--sensors") + 1] if "--sensors" in sys.argv else "xrf_uv"
SEEDS = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (tuple(range(1, 11)) if MAIN else HARD + EASY)
UV_GAIN_ERROR = 1.15                 # chosen: the UV-Vis reads 15 % high, unknown to the fit
UV_STAGES = (2, 4, 6)
XRF_SD, UV_SD = 0.02, 0.01
OUT = HERE / "runs" / (f"identify_then_control_{CONFIG}.csv" if SENSORS == "xrf_uv"
                       else f"identify_then_control_{SENSORS}_{CONFIG}.csv")
FIELDS = ["seed", "group", "beta_pr_lace_true", "beta_pr_lace_fit", "beta_nd_pr_true", "beta_nd_pr_fit",
          "k_factor_fit", "fit_cost", "fit_evals", "purity_end", "recovery_end", "rest_purity", "rest_recovery",
          "Q_org_end", "solver_failures", "seconds"]


def readings(circ, x, u, names):
    """The instruments' readings at state x (model layout), as one vector: product, raffinate (all
    elements present), then Pr and Nd at the UV-Vis stages."""
    o = circ.outlets(x, u.numpy())
    c_aq = circ.unpack(circ.observable(x))[0]
    i_pr, i_nd = names.index("Pr"), names.index("Nd")
    uv = [float(np.asarray(c_aq[j], float).ravel()[i]) for j in UV_STAGES for i in (i_pr, i_nd)]
    return np.concatenate([np.atleast_1d(o["c_prod"])[:3], np.atleast_1d(o["c_raff"])[:3], uv])


def noise_sd(y):
    """Each reading's noise standard deviation: a share of the reading (XRF, then UV-Vis), with a floor."""
    n_xrf = 6
    return np.concatenate([np.maximum(XRF_SD * np.abs(y[:n_xrf]), 1e-4), np.maximum(UV_SD * np.abs(y[n_xrf:]), 1e-5)])


def observe(y_full, y_noisy=None):
    """What the chosen instrument set gives, from the full reading vector (product 3, raffinate 3,
    then Pr, Nd at each UV stage). For uv_relative: stage ratios against the last UV stage and the
    Nd/Pr ratio per spectrum (a common gain cancels in every one)."""
    y = y_full if y_noisy is None else y_noisy
    if SENSORS == "xrf_only":
        return y[:6]
    if SENSORS == "uv_relative":
        uv = y[6:].reshape(len(UV_STAGES), 2)            # rows: stages; columns: Pr, Nd
        ref = uv[-1]
        stage_ratios = (uv[:-1] / ref).ravel()
        band_ratios = uv[:, 1] / uv[:, 0]
        return np.concatenate([stage_ratios, band_ratios])
    return y


def model_with(rc, theta):
    """The nominal model with the fitted numbers: theta = log factors on Pr/(La+Ce), Nd/Pr and k."""
    base = build_circuit(rc)
    p = base.p
    names = list(p.names)
    i_la, i_pr, i_nd = names.index("LaCe"), names.index("Pr"), names.index("Nd")
    K = np.array(p.K, float).copy()
    beta_pl = K[i_pr] / K[i_la] * np.exp(theta[0])
    beta_np = K[i_nd] / K[i_pr] * np.exp(theta[1])
    K[i_la] = K[i_pr] / beta_pl
    K[i_nd] = K[i_pr] * beta_np
    p2 = dataclasses.replace(p, K=K, k_transfer=p.k_transfer * np.exp(theta[2]))
    return Circuit(base.cmap, p2, model=base.model, scrub_reflux=base.scrub_reflux), beta_pl, beta_np


def plant_at_rest(seed):
    """The perturbed plant for one seed, commissioned at rest on the design flows:
    (run file, plant, its stage-rate factors, state at rest, design inputs)."""
    rc = load_config(CONFIG, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    plant, plant_k = PLANTS["perturbed"].build(rc)
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=plant_k))
    return rc, plant, plant_k, x_rest, u_design


def measured(plant, x_rest, u_design, seed):
    """What the chosen instruments report at rest, with this plant's noise draw: (the readings the fit
    sees, their noise standard deviations)."""
    rng = np.random.default_rng(seed + 5000)
    y_true = readings(plant, x_rest, u_design, list(plant.p.names))
    sd_full = noise_sd(y_true)
    y_read = y_true + rng.normal(0.0, sd_full)
    y_read[6:] *= UV_GAIN_ERROR if SENSORS == "uv_relative" else 1.0
    y_meas = observe(y_read)
    # the noise of what the fit sees: each reading's own, and for ratios both readings' (in quadrature)
    if SENSORS == "uv_relative":
        return y_meas, np.abs(y_meas) * UV_SD * np.sqrt(2)
    return y_meas, observe(sd_full)


def fit_numbers(rc, plant, x_rest, data):
    """The three numbers fitted by least squares to the readings. data: (model inputs, readings, their
    noise). Returns (the fit, the number of steady-state solves)."""
    u_model, y_meas, sd = data
    names = list(plant.p.names)
    state = {"x": plant.observable(x_rest), "evals": 0}

    def residuals(theta):
        """Noise-scaled reading errors of the model with these numbers (warm-started steady state)."""
        model, _, _ = model_with(rc, theta)
        x = steady_state(model, u_model, x0=state["x"])
        state["x"], state["evals"] = x, state["evals"] + 1
        return (observe(readings(model, x, u_model, names)) - y_meas) / sd

    fit = least_squares(residuals, np.zeros(3), bounds=(-0.5, 0.5), diff_step=0.02, max_nfev=60)
    return fit, state["evals"]


def fitted_controller(rc, model, x_model, u_model):
    """The MPC on the fitted model: over level loops on the main case, with the acidity floor otherwise."""
    if MAIN:
        return mpc_over_level_loops(rc, model, x_model, u_model)
    settings = controller_settings(rc, acid_floor=1e-3)
    return CircuitMPC(model, x_model, u_model, rc.cut, rc.targets()[0], rc.design_purity, rc.targets()[1], settings)


def purity_recovery_of(plant, rc, u_design):
    """A function (state, inputs) -> the plant's product purity and recovery (fractions)."""
    heavy = np.array(heavy_mask(plant.p, rc.cut))
    fed = float(u_design.Q_feed) * np.asarray(u_design.c_feed, float)[heavy].sum()

    def pur_rec(x, uv):
        """The plant's product purity and recovery (fractions) at state x, inputs uv."""
        o = plant.outlets(x, uv)
        cp = np.atleast_1d(o["c_prod"])
        return product_purity(cp, plant.p, rc.cut), float(o["Q_prod"]) * cp[heavy].sum() / fed
    return pur_rec


def result_row(seed, plant, fitted, outcome):
    """The plant's row. fitted: (fit, solves, Pr/(La+Ce) fitted, Nd/Pr fitted); outcome: (trajectory,
    log, end purity and recovery, at-rest purity and recovery)."""
    fit, evals, beta_pl, beta_np = fitted
    traj, log, end, rest = outcome
    names = list(plant.p.names)
    i_la, i_pr, i_nd = names.index("LaCe"), names.index("Pr"), names.index("Nd")
    lay = Inputs.layout(plant.n, plant.N)
    return {"seed": seed, "group": "main" if MAIN else ("hard" if seed in HARD else "easy"),
            "beta_pr_lace_true": plant.p.K[i_pr] / plant.p.K[i_la], "beta_pr_lace_fit": beta_pl,
            "beta_nd_pr_true": plant.p.K[i_nd] / plant.p.K[i_pr], "beta_nd_pr_fit": beta_np,
            "k_factor_fit": float(np.exp(fit.x[2])), "fit_cost": float(fit.cost), "fit_evals": evals,
            "purity_end": end[0], "recovery_end": end[1], "rest_purity": rest[0], "rest_recovery": rest[1],
            "Q_org_end": float(traj.u[-1][lay["Q_org"].start]),
            "solver_failures": int(sum(1 for r in log if r.get("qp_ok") is False))}


def run(seed):
    """One plant: fit the model to its noisy readings at rest, run the MPC on the fitted model, and return
    the fit, the end and at-rest purity and recovery, and the organic flow at the end."""
    t0 = time.time()
    rc, plant, plant_k, x_rest, u_design = plant_at_rest(seed)
    y_meas, sd = measured(plant, x_rest, u_design, seed)
    u_model = u_design.replace(k_factor=np.ones(plant.N))
    fit, evals = fit_numbers(rc, plant, x_rest, (u_model, y_meas, sd))
    model, beta_pl, beta_np = model_with(rc, fit.x)
    x_model = steady_state(model, u_model, x0=plant.observable(x_rest))
    traj, log = run_closed_loop(plant, x_rest, fitted_controller(rc, model, x_model, u_model), None, HOURS,
                                plant_k_factor=plant_k)
    pur_rec = purity_recovery_of(plant, rc, u_design)
    end = pur_rec(traj.x[-1], traj.u[-1])
    rest = pur_rec(steady_state(plant, traj.u[-1], x0=traj.x[-1]), traj.u[-1])
    row = result_row(seed, plant, (fit, evals, beta_pl, beta_np), (traj, log, end, rest))
    return {**row, "seconds": round(time.time() - t0)}


done = set()
if OUT.exists():
    done = {int(r["seed"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for seed in SEEDS:
    if seed in done:
        continue
    row = run(seed)
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    print(f"seed {seed:2d} ({row['group']}): Pr/LaCe true {row['beta_pr_lace_true']:.2f} fit {row['beta_pr_lace_fit']:.2f}, "
          f"Nd/Pr true {row['beta_nd_pr_true']:.2f} fit {row['beta_nd_pr_fit']:.2f}, k x{row['k_factor_fit']:.2f} "
          f"({row['fit_evals']} evals) | end {100 * row['purity_end']:.2f}/{100 * row['recovery_end']:.2f} %, at rest "
          f"{100 * row['rest_purity']:.2f}/{100 * row['rest_recovery']:.2f} %, fails {row['solver_failures']} "
          f"({row['seconds']} s)", flush=True)
