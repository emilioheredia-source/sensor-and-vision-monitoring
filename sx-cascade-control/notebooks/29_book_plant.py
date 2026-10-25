"""The plant built closer to the book: which of its mechanisms matter, and how many stages it needs.

The question. plant_models.BookCircuit adds four things from Qi (2018) that the controller's model
doesn't have: a La+Ce/Pr separation factor weaker where one side is a trace, a weaker one in the
scrub, a weaker acid dependence (slope 2.5 instead of 3), and transfer slowing down in acid. On the
main case's design (8 + 20 stages), the first look gave a plant that can't make 95 % recovery at
99.5 % purity with any flows. Which mechanism is responsible, and how many stages would a plant
like this need?

The method, at steady state, on the model's own numbers (no perturbation, no drift), so only the
book's mechanisms differ:
  1. Each mechanism alone, then all four: the best recovery at 99.5 % purity (design optimizer told
     the plant's true equations; organic pump to its maximum), on 8 + 20 stages.
  2. All four, over stage layouts: the best recovery at 99.5 %, and the least reagent for 95 %
     recovery where that's reachable. The same for the model, for comparison.

Run:  pixi run python notebooks/29_book_plant.py
Writes runs/book_plant_ndpr_main.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                               # noqa: E402
from src.config import load_config, nominal_inputs                       # noqa: E402
from src.design import optimize_operating_point, reflux_link             # noqa: E402
from src.plant_models import BookCircuit                                 # noqa: E402
from src.simulate import steady_state                                    # noqa: E402

HERE = Path(__file__).resolve().parent.parent
rc = load_config("ndpr_main")
NET = rc.design["net_product_L_per_h"]
FREE = {"Q_org": (rc.design["organic_min_L_per_h"], rc.design["pump_max_L_per_h"]),
        "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}
# each mechanism switched off by its neutral value
NEUTRAL = dict(trace_drop=0.0, beta_mid=2.0, scrub_beta_factor=1.0, n_plant=None, h_half=1e9)
MECHANISMS = {
    "none (the model)": {},
    "trace ends weaker": dict(trace_drop=0.20, beta_mid=2.1),
    "scrub weaker": dict(scrub_beta_factor=0.92),
    "acid slope 2.5": dict(n_plant=2.5),
    "slower in acid": dict(h_half=0.2),
    "all four": dict(trace_drop=0.20, beta_mid=2.1, scrub_beta_factor=0.92, n_plant=2.5, h_half=0.2),
}


def plant_for(n_ext, n_scrub, mech: dict):
    """The circuit for a layout, with the book's mechanisms in `mech` (none: the model itself)."""
    cmap = build_map(n_ext, n_scrub, 4)
    if not mech:
        return Circuit(cmap, rc.params, scrub_reflux=True)
    kw = {**NEUTRAL, **mech}
    if kw["n_plant"] is None:
        kw["n_plant"] = float(rc.params.n_acid)
    return BookCircuit(cmap, rc.params, scrub_reflux=True, common_amp=0.0, split_amp=0.0, **kw)


def best(plant, objective="recovery"):
    """The best recovery at 99.5 % purity, and the least reagent and organic flow at 95 % recovery."""
    u0 = nominal_inputs(rc, plant).replace(Q_strip=NET + float(nominal_inputs(rc, plant).Q_scrub))
    x0 = steady_state(plant, u0)
    r1 = optimize_operating_point(plant, u0, rc.cut, 0.995, FREE, link=reflux_link(NET), x_guess=x0)
    out = {"best_recovery": r1.recovery if r1.success else np.nan, "status1": r1.status}
    if r1.success and r1.recovery >= 0.95:
        r2 = optimize_operating_point(plant, r1.u, rc.cut, 0.995, FREE, link=reflux_link(NET), x_guess=r1.x,
                                      objective="reagent", min_recovery=0.95)
        out.update(reagent_at_95=r2.reagent_mol_per_h if r2.success else np.nan,
                   Q_org_at_95=float(r2.u.Q_org) if r2.success else np.nan)
    return out


rows = []
OUT = HERE / "runs" / "book_plant_ndpr_main.csv"
KEYS = ["part", "case", "layout", "best_recovery", "status1", "reagent_at_95", "Q_org_at_95"]
with open(OUT, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=KEYS).writeheader()


def keep(row):
    """Write one row straight away, so a stopped run keeps what it had."""
    rows.append(row)
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=KEYS).writerow({k: row.get(k, "") for k in KEYS})


print("Each mechanism on 8 + 20 stages: best recovery at 99.5 % purity")
for name, mech in MECHANISMS.items():
    r = best(plant_for(8, 20, mech))
    keep({"part": "mechanism", "case": name, "layout": "8+20", **r})
    print(f"  {name:20s} {100 * r['best_recovery']:6.2f} %  ({r['status1']})", flush=True)

print("Stage layouts, the model against the book plant")
for n_ext, n_scrub in ((8, 20), (10, 26), (12, 32), (14, 38), (16, 44)):
    for name in ("none (the model)", "all four"):
        r = best(plant_for(n_ext, n_scrub, MECHANISMS[name]))
        keep({"part": "layout", "case": name, "layout": f"{n_ext}+{n_scrub}", **r})
        print(f"  {n_ext:2d}+{n_scrub:2d} {name:18s} best recovery {100 * r['best_recovery']:6.2f} %   reagent for 95 % "
              f"{r.get('reagent_at_95', float('nan')):6.0f} mol/h, organic {r.get('Q_org_at_95', float('nan')):5.0f} L/h",
              flush=True)

print("wrote", OUT)
