"""Why is the NdPr split hard to control? The plant's gains, their direction, and the model's error.

The question. High-purity distillation is the textbook hard plant (Skogestad, Morari and Doyle
1988): its two product compositions answer strongly to a change in how the feed is split between
the products, and weakly to anything else. The gain matrix is ill-conditioned, and a controller
that inverts a model of it can be sent the wrong way by a small model error in the strong
direction. The SX cascade has the same structure: the organic flow and the scrub flow together
decide how much metal goes to the product. Is it ill-conditioned, does the model get the
directions right on realistic plants, and do ratio handles (as in distillation, L/D and V/B;
feed-ratio control in SX) help?

The method, at steady state (no controller):
  Outputs, the distillation convention (logs of the small numbers):
     y1 = log10(1 - purity)      impurity in the product
     y2 = log10(1 - recovery)    NdPr lost to the raffinate
  Inputs, relative moves (d ln u), so a 1 % move of either flow counts the same:
     flows   u1 = ln Q_scrub (strip tied: strip = net product + scrub), u2 = ln Q_org
     ratios  v1 = ln(Q_scrub / Q_org) with Q_org held; v2 = ln Q_org with the ratio held
             (both flows by the same factor). G_ratio = G_flows T, T = [[1, 1], [0, 1]].
  G by central differences of the steady state, +-0.5 %.
  Measures:
     condition number   sigma_max / sigma_min of G (large: ill-conditioned)
     relative gain      lambda_11 (pairing u1-y1); negative or large is trouble for single loops
     model error        E = (G_plant - G_model) G_model^-1 at the same handles; sigma_max(E) > 1
                        means a controller that inverts the model can move the plant the wrong way
                        in some direction (the standard robustness bound for inverse-based control)
  Where: at the design handles (where every plant starts) and at each plant's own design point
  (the design optimizer told the plant's true numbers: least reagent at 99.5 % purity and 95 %
  recovery, the same rule that made the model's design), with the model's gains taken at the same
  handles, as a re-linearizing controller would. (A first version used each plant's best recovery
  instead: that sends every plant to the organic pump's limit at ~99 % recovery, far from any
  real operating point, and its gains were distorted by that limit.)
  Plants: the twenty realistic perturbed plants (generator "ratios").

Run:  pixi run python notebooks/24_conditioning.py [run_file ...] [--settled]
      (default ndpr_plant_sized ndpr_plant)
Writes runs/conditioning_<run_file>.csv, or runs/conditioning_<run_file>_settled.csv with --settled.

--settled (added Oct 24): every steady state by study_tools.settle (Newton, then 800 h of
integration) instead of the default ladder. The default ladder doesn't settle the main case (FINDINGS
section 4, dated note Oct 22: the gain matrix's condition number went from 43 to 204), so the
per-plant numbers of the first run are rerun this way and kept side by side with the old file.
"""
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.design import optimize_operating_point, reflux_link            # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from src.registry import PLANTS                                         # noqa: E402
from src.simulate import steady_state as ladder_steady_state            # noqa: E402
from study_tools import settle                                          # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SETTLED = "--settled" in sys.argv
CONFIGS = tuple(a for a in sys.argv[1:] if not a.startswith("--")) or ("ndpr_plant_sized", "ndpr_plant")
steady_state = settle if SETTLED else ladder_steady_state                # every solve below goes through this
STEP = 0.005                                       # relative move for the differences
T_RATIO = np.array([[1.0, 1.0], [0.0, 1.0]])      # ratio moves -> flow moves (in ln units)



@dataclass
class Split:
    """What every gain needs to know about the split: which elements are product (heavy), the
    product metal fed (mol/h) and the net strip flow (L/h; strip = net + scrub)."""
    heavy: np.ndarray
    fed: float
    net: float


def outputs(circ, x, u, split: Split) -> np.ndarray:
    """(log10 impurity, log10 loss) at state x, inputs u."""
    o = circ.outlets(x, u.numpy())
    cp = np.atleast_1d(o["c_prod"])
    purity = cp[split.heavy].sum() / cp.sum()
    recovery = float(o["Q_prod"]) * cp[split.heavy].sum() / split.fed
    return np.log10(np.maximum([1 - purity, 1 - recovery], 1e-9))


def moved(u0, name: str, sign: int, split: Split):
    """The inputs with one flow moved by +-STEP in ln (the strip flow follows the scrub flow)."""
    v = float(getattr(u0, name)) * np.exp(sign * STEP)
    u = u0.replace(**{name: v})
    return u.replace(Q_strip=split.net + v) if name == "Q_scrub" else u


def gains(circ, u0, x0, split: Split) -> tuple:
    """Steady-state gain matrix d y / d ln(Q_scrub, Q_org) by central differences, y at u0, and the
    base state. Every step is solved from the base state."""
    x_base = steady_state(circ, u0, x0=x0)
    y0 = outputs(circ, x_base, u0, split)
    G = np.zeros((2, 2))
    for col, name in enumerate(("Q_scrub", "Q_org")):
        steps = [moved(u0, name, s, split) for s in (+1, -1)]
        ys = [outputs(circ, steady_state(circ, u, x0=x_base), u, split) for u in steps]
        G[:, col] = (ys[0] - ys[1]) / (2 * STEP)
    return G, y0, x_base


def measures(G: np.ndarray) -> dict:
    """Condition number and relative gain lambda_11 of a 2x2 gain matrix."""
    s = np.linalg.svd(G, compute_uv=False)
    det = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
    return {"cond": s[0] / max(s[1], 1e-12), "rga11": G[0, 0] * G[1, 1] / det if det != 0 else np.nan}


def model_error(Gp: np.ndarray, Gm: np.ndarray) -> float:
    """sigma_max of (Gp - Gm) Gm^-1."""
    try:
        return float(np.linalg.svd((Gp - Gm) @ np.linalg.inv(Gm), compute_uv=False)[0])
    except np.linalg.LinAlgError:
        return np.inf


def both_handles(G: np.ndarray) -> dict:
    """measures() for the flow handles and for the ratio handles, as flows_* and ratio_*."""
    return {**{f"flows_{k}": v for k, v in measures(G).items()},
            **{f"ratio_{k}": v for k, v in measures(G @ T_RATIO).items()}}


def model_at_design(config: str):
    """The model at its design point: (model, design state, its gains there, the split, its row)."""
    rc0 = load_config(config)
    model = build_circuit(rc0)
    x_d, u_d = commission(model, nominal_inputs(rc0, model))
    heavy = np.array(heavy_mask(model.p, rc0.cut))
    split = Split(heavy, float(u_d.Q_feed) * np.asarray(u_d.c_feed, float)[heavy].sum(),
                  float(u_d.Q_strip) - float(u_d.Q_scrub))
    Gm_d, _, _ = gains(model, u_d, x_d, split)
    row = {"case": "model at design", **both_handles(Gm_d), "G": np.round(Gm_d, 3).tolist()}
    print(f"\n{config}: model at design  G (rows: log impurity, log loss; cols: ln scrub, ln organic)\n"
          f"  {np.round(Gm_d, 2).tolist()}  cond {row['flows_cond']:.1f}  lambda11 {row['flows_rga11']:+.2f}"
          f"   ratio handles: cond {row['ratio_cond']:.1f}  lambda11 {row['ratio_rga11']:+.2f}", flush=True)
    return model, x_d, Gm_d, split, row


def free_handles(rc) -> dict:
    """The handles the design optimizer may move, with their ranges."""
    design = rc.design
    return {"Q_org": (design["organic_min_L_per_h"], design["pump_headroom_fraction"] * design["pump_max_L_per_h"]),
            "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}


def spec_columns(Gp_s: np.ndarray, Gm_s: np.ndarray, r) -> dict:
    """The spec_* columns from the plant's and the model's gains at the plant's own design point r."""
    return {"spec_flows_cond": measures(Gp_s)["cond"], "spec_flows_rga11": measures(Gp_s)["rga11"],
            "spec_ratio_cond": measures(Gp_s @ T_RATIO)["cond"], "spec_ratio_rga11": measures(Gp_s @ T_RATIO)["rga11"],
            "spec_model_error_flows": model_error(Gp_s, Gm_s),
            "spec_model_error_ratio": model_error(Gp_s @ T_RATIO, Gm_s @ T_RATIO),
            "spec_recovery": r.recovery, "spec_Q_org": float(r.u.Q_org), "spec_Q_scrub": float(r.u.Q_scrub),
            "G_spec": np.round(Gp_s, 3).tolist(), "G_model_spec": np.round(Gm_s, 3).tolist()}


def at_own_design_point(plant, u_pd, model_case: tuple, rc) -> dict:
    """The plant's own design point (least reagent at the targets) and the model's gains at the same
    handles: the spec_* columns, or {} if the design failed. model_case: (model, x_d, split)."""
    model, x_d, split = model_case
    r = optimize_operating_point(plant, u_pd, rc.cut, rc.design_purity, free_handles(rc), link=reflux_link(split.net),
                                 objective="reagent", min_recovery=rc.targets()[1])
    if not r.success:
        return {}
    Gp_s, _, _ = gains(plant, r.u, r.x, split)
    u_ms = r.u.replace(k_factor=np.ones(model.N))
    Gm_s, _, _ = gains(model, u_ms, steady_state(model, u_ms, x0=x_d), split)
    return spec_columns(Gp_s, Gm_s, r)


def plant_row(config: str, seed: int, model_case: tuple, Gm_d: np.ndarray) -> dict:
    """One perturbed plant: its gains and the model's error at the design handles, then at its own
    design point. model_case: (model, x_d, split)."""
    rc = load_config(config, overrides={"plant": {"mismatch": "perturbed", "seed": seed, "generator": "ratios"}})
    plant, k = PLANTS["perturbed"].build(rc)
    u_pd = nominal_inputs(rc, plant).replace(k_factor=k)
    Gp_d, _, _ = gains(plant, u_pd, steady_state(plant, u_pd), model_case[2])
    row = {"case": f"seed {seed}", "where": "design handles", **both_handles(Gp_d),
           "model_error_flows": model_error(Gp_d, Gm_d), "model_error_ratio": model_error(Gp_d @ T_RATIO, Gm_d @ T_RATIO),
           "G": np.round(Gp_d, 3).tolist()}
    return {**row, **at_own_design_point(plant, u_pd, model_case, rc)}


def print_row(seed: int, row: dict, seconds: float) -> None:
    """One plant's line on the console."""
    print(f"  seed {seed:2d}: at design cond {row['flows_cond']:7.1f} (ratio {row['ratio_cond']:6.1f})  "
          f"model error {row['model_error_flows']:6.2f}   at own spec point: cond {row.get('spec_flows_cond', np.nan):7.1f} "
          f"model error {row.get('spec_model_error_flows', np.nan):6.2f} "
          f"(ratio handles {row.get('spec_model_error_ratio', np.nan):6.2f})  ({seconds:.0f} s)", flush=True)


def write_table(config: str, rows: list) -> None:
    """The rows to runs/conditioning_<config>[_settled].csv, "case" first and the rest sorted."""
    out = HERE / "runs" / f"conditioning_{config}{'_settled' if SETTLED else ''}.csv"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


def main():
    """Every run file: the model at design, then the twenty plants."""
    for config in CONFIGS:
        model, x_d, Gm_d, split, row = model_at_design(config)
        rows = [row]
        for seed in range(1, 21):
            t0 = time.time()
            rows.append(plant_row(config, seed, (model, x_d, split), Gm_d))
            print_row(seed, rows[-1], time.time() - t0)
        write_table(config, rows)


if __name__ == "__main__":
    main()
