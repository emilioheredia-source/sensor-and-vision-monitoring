"""The controller, with its ordinary model, on the plant built closer to the book.

The question. Notebook 29: a plant with the book's mechanisms (plant "book": separation weaker
toward the cascade ends and in the scrub, acid slope 2.5, slower transfer in acid, slow chemistry
drift) can't make 95 % recovery at 99.5 % purity on 8 + 20 stages with any flows, and can on
10 + 26. What happens under control on each, when the controller knows none of this?

The method: each design run the main case's way (margin targets, acidity floor, level loops,
120 h trim with the controller), on the book plant for several seeds, no upset and the feed
switch, 120 h, uncontrolled and under the MPC over level loops. Scored on NdPr lost or made off
spec, the lowest purity and recovery, and solver failures.

Run:  pixi run python notebooks/30_book_plant_control.py [config ...]   (default: ndpr_main ndpr_main_10x26)
Writes runs/book_plant_control.csv (row by row)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runs import RunSession, RunSpec                                # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIGS = tuple(a for a in sys.argv[1:] if not a.isdigit()) or ("ndpr_main", "ndpr_main_10x26")
SEEDS = tuple(int(a) for a in sys.argv[1:] if a.isdigit()) or (1, 2, 3)
HOURS = 120.0
OUT = HERE / "runs" / "book_plant_control.csv"
FIELDS = ["config", "seed", "upset", "controller", "off_spec_mol", "lost_excess_mol", "start_purity",
          "start_recovery", "min_purity", "min_recovery", "end_purity", "end_recovery", "solver_failures", "seconds"]

done = set()
if OUT.exists():
    done = {(r["config"], int(r["seed"]), r["upset"], r["controller"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for config in CONFIGS:
    for seed in SEEDS:
        for upset in ("none", "feed_switch"):
            for ctl in ("open_loop_level_loops", "mpc_level_loops"):
                if (config, seed, upset, ctl) in done:
                    continue
                t0 = time.time()
                s = RunSession(RunSpec(config=config, plant="book", controller=ctl, upset=upset, hours=HOURS,
                                       run_file={"plant": {"seed": seed, "generator": "ratios"}}))
                for _ in s.steps():
                    pass
                r = s.result()
                pur, rec = np.asarray(r.series["purity"]), np.asarray(r.series["recovery"])
                row = {"config": config, "seed": seed, "upset": upset, "controller": ctl,
                       "off_spec_mol": r.scorecard["off_spec_mol"], "lost_excess_mol": r.scorecard["lost_excess_mol"],
                       "start_purity": pur[0], "start_recovery": rec[0], "min_purity": pur.min(),
                       "min_recovery": rec.min(), "end_purity": pur[-1], "end_recovery": rec[-1],
                       "solver_failures": r.scorecard.get("solver_failures", 0), "seconds": round(time.time() - t0)}
                with open(OUT, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                print(f"{config:16s} seed {seed} {upset:11s} {ctl:22s} start {100 * pur[0]:6.2f}/{100 * rec[0]:6.2f} %  "
                      f"end {100 * pur[-1]:6.2f}/{100 * rec[-1]:6.2f} %  off spec {row['off_spec_mol']:6.0f}  lost "
                      f"{row['lost_excess_mol']:6.0f}  fails {row['solver_failures']}  ({row['seconds']} s)", flush=True)
