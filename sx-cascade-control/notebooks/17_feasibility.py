"""Which plants can make spec at all? A steady-state answer, with no controller in it.

The question. The seed study (notebook 14) found five perturbed plants on which the controller
never reached 99.5 % purity at 95 % recovery. Is that the controller, or can those plants not do
it whatever anyone does? This script answers with the design optimizer (src/design.py): told a
plant's true numbers, it finds the best recovery at 99.5 % purity over every setting of the
handles. No dynamics, no controller, no model error: if the optimizer can't reach 95 %, nothing
can, at that feed rate.

It asks four things:
  1. each of the 20 perturbed plants' Pr/(La+Ce) separation factor, from its random draw (the
     model assumes 2.0; Yin et al. 2018 quote 1.8-2.2 for industrial circuits)
  2. each plant's best recovery at spec, twice: under the design rules (organic at most 85 % of
     the pump, organic loaded at most 85 % of capacity) and with those rules relaxed to physical
     limits (the full pump, the organic loaded to 98 %)
  3. for a plant that can't reach 95 % even relaxed: how far the feed has to be cut before it can
  4. the same questions for the model with only the separation factor changed, across the
     published range and 10 % beyond it on each side (1.62 to 2.42): the design's margin

Run:  pixi run python notebooks/17_feasibility.py
Writes runs/feasibility_ndpr_plant.csv
"""
import csv
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit                                     # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs   # noqa: E402
from src.design import optimize_operating_point, reflux_link        # noqa: E402
from src.registry import PLANTS                                     # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_plant"
SEEDS = range(1, 21)
FEED_FRACTIONS = (0.9, 0.8, 0.7, 0.6, 0.5)              # feed rates tried, as a fraction of 600 L/h
BETAS = (1.62, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.42)  # Pr/(La+Ce) separation factors for the sweep
TARGET = 0.95                                            # the recovery target at spec

rc0 = load_config(CONFIG)
design = rc0.design
PUMP_MAX = design["pump_max_L_per_h"]
RULES = {"design rules": dict(q_org_max=design["pump_headroom_fraction"] * PUMP_MAX, max_loading=None),
         "relaxed": dict(q_org_max=PUMP_MAX, max_loading=0.98)}


# ---- the optimizer, asked for the best recovery at spec ----
def best_recovery(circuit, k_factor, rc, feed_fraction=1.0, q_org_max=None, max_loading=None) -> float:
    """The best recovery the circuit can reach at the design purity (99.5 %), over every setting of
    the four handles within their bounds, at feed_fraction of the run file's feed rate.

    Two starting points (the run file's handles, and more organic and scrub), best kept: IPOPT
    finds a local optimum, and two starts guard against a poor one. nan if neither converges."""
    u0 = nominal_inputs(rc, circuit).replace(k_factor=k_factor,
                                             Q_feed=float(nominal_inputs(rc, circuit).Q_feed) * feed_fraction)
    free = {"Q_org": (design["organic_min_L_per_h"], q_org_max), "Q_scrub": (50.0, 2000.0),
            "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}
    link = reflux_link(design["net_product_L_per_h"])
    best = float("nan")
    for start in (u0, u0.replace(Q_org=min(3000.0, q_org_max), Q_scrub=500.0)):
        r = optimize_operating_point(circuit, start, rc.cut, rc.design_purity, free, link=link,
                                     max_loading=max_loading)
        if r.success and r.purity >= rc.design_purity - 1e-4:
            best = r.recovery if np.isnan(best) else max(best, r.recovery)
    return best


def feed_needed(circuit, k_factor, rc) -> float:
    """The highest feed fraction (of FEED_FRACTIONS) at which 95 % recovery at spec is reachable
    under the design rules; nan if none of them is enough."""
    for f in FEED_FRACTIONS:
        if best_recovery(circuit, k_factor, rc, f, **RULES["design rules"]) >= TARGET:
            return f
    return float("nan")


def separation_factor(params) -> float:
    """K_Pr / K_LaCe: how much more Pr than La+Ce the extractant takes."""
    return float(params.K[params.index("Pr")] / params.K[params.index("LaCe")])


rows = []


def record(kind: str, label, beta: float, circuit, k_factor, rc) -> None:
    """Score one plant on the questions above and keep the row."""
    best = {name: best_recovery(circuit, k_factor, rc, **opts) for name, opts in RULES.items()}
    needs_cut = not (best["relaxed"] >= TARGET)
    feed = feed_needed(circuit, k_factor, rc) if needs_cut else 1.0
    rows.append({"kind": kind, "plant": label, "separation_factor": round(beta, 3),
                 "best_recovery_design_rules": best["design rules"], "best_recovery_relaxed": best["relaxed"],
                 "feed_fraction_for_95": feed})
    print(f"{kind:5s} {str(label):5s}  beta {beta:5.2f}   best recovery at spec: design rules "
          f"{best['design rules']:.3f}, relaxed {best['relaxed']:.3f}   feed for 95 %: "
          f"{'full' if feed == 1.0 else (f'{feed:.0%}' if feed == feed else 'below 50 %')}", flush=True)


# ---- 1-3. the 20 perturbed plants ----
for seed in SEEDS:
    rc = load_config(CONFIG, overrides={"plant": {"seed": seed}})
    plant, k_factor = PLANTS["perturbed"].build(rc)
    record("seed", seed, separation_factor(plant.p), plant, k_factor, rc)

# ---- 4. the model, with only the separation factor changed ----
model = build_circuit(rc0)
K = np.array(model.p.K, float)
for beta in BETAS:
    K_beta = K.copy()
    K_beta[model.p.index("LaCe")] = K[model.p.index("Pr")] / beta
    circuit = Circuit(model.cmap, replace(model.p, K=K_beta), model=model.model, scrub_reflux=model.scrub_reflux)
    record("beta", beta, beta, circuit, np.ones(model.N), rc0)

out = HERE / "runs" / f"feasibility_{CONFIG}.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("wrote", out)
