"""The feed-loss interlock on twenty plants: does it fix the pump stop?

The question. The feed pump stopping for two hours is the upset the controller has handled worst
all along: on the main case's twenty plants it was worse than doing nothing on about half (small
amounts, but consistently). src/interlock.py adds what a plant does: freeze the handles when the
feed stops, re-sync the controller and hand back two hours after the feed returns; optionally stop
saponifying while the feed is off.

The method: the main case (configs/ndpr_main.toml), the twenty realistic plants of the seed study,
the pump stop, 80 h; the MPC over level loops without and with the interlock (both forms).
Open loop over the level loops is run the same way, 80 h, for a like-for-like comparison.

Run:  pixi run python notebooks/34_interlock.py [seed ...]
Writes runs/interlock_ndpr_main.csv (row by row)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runs import RunSpec, execute                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
SEEDS = tuple(int(a) for a in sys.argv[1:]) or tuple(range(1, 21))
CONTROLLERS = ("mpc_level_loops", "mpc_interlock", "mpc_interlock_base", "open_loop_level_loops")
OUT = HERE / "runs" / "interlock_ndpr_main.csv"
FIELDS = ["seed", "controller", "off_spec_mol", "lost_excess_mol", "min_purity", "solver_failures", "seconds"]

done = set()
if OUT.exists():
    done = {(int(r["seed"]), r["controller"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for seed in SEEDS:
    for ctl in CONTROLLERS:
        if (seed, ctl) in done:
            continue
        t0 = time.time()
        r = execute(RunSpec(config="ndpr_main", plant="perturbed", controller=ctl, upset="feed_interrupt", hours=80.0,
                            run_file={"plant": {"seed": seed, "generator": "ratios"}}))
        row = {"seed": seed, "controller": ctl, "off_spec_mol": r.scorecard["off_spec_mol"],
               "lost_excess_mol": r.scorecard["lost_excess_mol"], "min_purity": float(np.min(r.series["purity"])),
               "solver_failures": r.scorecard.get("solver_failures", 0), "seconds": round(time.time() - t0)}
        with open(OUT, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"seed {seed:2d} {ctl:20s} off spec {row['off_spec_mol']:6.0f}  lost {row['lost_excess_mol']:6.0f}  "
              f"min purity {100 * row['min_purity']:6.2f} %", flush=True)
