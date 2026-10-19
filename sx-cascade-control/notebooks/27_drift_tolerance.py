"""How much chemistry drift does the main case tolerate, with and without the controller?

The question. On the drifting plant (plant_models.DriftingCircuit) every extraction constant
moves together by up to ~10 % and La+Ce against the rest by `drift_split`, as slow waves over one
to seven days. At the 3 % first chosen, the main case (stage-sized, operated with margin) stayed on
spec for a week even with nobody touching it. How big a drift in the separation factor can it take,
open loop and under the MPC?

The method: realistic plants (generator "ratios"; seeds that separate worse than the model, and
ones that separate better), drift_split 3 %, 6 %, 10 %, one week, no other upset; open loop with
level loops, and the MPC over level loops. Scored on product made off spec and NdPr lost.

Run:  pixi run python notebooks/27_drift_tolerance.py [seed ...]
Writes runs/drift_tolerance_ndpr_main.csv (row by row)
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runs import RunSession, RunSpec                                # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_main"
HOURS = 168.0
SEEDS = tuple(int(a) for a in sys.argv[1:]) or (3, 10, 16, 1, 7)
SPLITS = (0.03, 0.06, 0.10)
CONTROLLERS = ("open_loop_level_loops", "mpc_level_loops")
OUT = HERE / "runs" / f"drift_tolerance_{CONFIG}.csv"
FIELDS = ["seed", "drift_split", "controller", "off_spec_mol", "lost_excess_mol", "min_purity", "min_recovery",
          "solver_failures", "seconds"]

done = set()
if OUT.exists():
    done = {(int(r["seed"]), float(r["drift_split"]), r["controller"]) for r in csv.DictReader(open(OUT))}
else:
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
for seed in SEEDS:
    for split in SPLITS:
        for ctl in CONTROLLERS:
            if (seed, split, ctl) in done:
                continue
            t0 = time.time()
            s = RunSession(RunSpec(config=CONFIG, plant="drifting", controller=ctl, upset="none", hours=HOURS,
                                   run_file={"plant": {"seed": seed, "generator": "ratios", "drift_split": split}}))
            for _ in s.steps():
                pass
            r = s.result()
            row = {"seed": seed, "drift_split": split, "controller": ctl,
                   "off_spec_mol": r.scorecard["off_spec_mol"], "lost_excess_mol": r.scorecard["lost_excess_mol"],
                   "min_purity": float(np.min(r.series["purity"])), "min_recovery": float(np.min(r.series["recovery"])),
                   "solver_failures": r.scorecard.get("solver_failures", 0), "seconds": round(time.time() - t0)}
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"seed {seed:2d} drift {100 * split:4.0f} % {ctl:22s} off spec {row['off_spec_mol']:7.0f} mol, lost "
                  f"{row['lost_excess_mol']:7.0f} mol, min purity {100 * row['min_purity']:6.2f} %, min recovery "
                  f"{100 * row['min_recovery']:6.2f} % ({row['seconds']} s)", flush=True)
