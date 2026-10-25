"""The two fixes for the reagent runaway on all ten fair-test plants (v1.3, third test)

The question (Oct 24). On plant 10, the controller learning its three numbers while it runs
(notebook 51) and the saponification tied to the acid balance (notebook 52) each stopped the runaway
alone. Does that hold on the other plants, and what does each cost?

The method. Notebook 51's run (48 h from rest on the design handles, MPC over level loops, full
state) on plants 1 to 10 of notebook 45, six controllers each:
  ridge, online, true                 the starting model as in notebook 51 (ridge's estimate; ridge
                                      then refitted every 2 h; the plant's own model)
  ridge+balance, online+balance, true+balance
                                      the same with notebook 52's cap on the saponification
So each change reads on its own: the refit (ridge against online), the balance (each against its
+balance), and both (ridge against online+balance), with the true model as the reference.

Run:  pixi run python notebooks/53_ten_plants.py
Writes runs/ten_plants_v13_ndpr_main.csv (one row per plant and controller) and .npz (the traces,
keys "<seed>_<controller>_<series>").
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                              # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import load_notebook, run_jobs, workers, write_rows    # noqa: E402

CONFIG = "ndpr_main"
SEEDS = tuple(range(1, 11))
CONTROLLERS = (("online", False), ("online", True), ("ridge", False), ("ridge", True), ("true", False), ("true", True))
OUT = HERE / "runs" / f"ten_plants_v13_{CONFIG}.csv"


def run(job):
    """One plant and one controller: (the result row, the traces). job: (seed, base, balanced)."""
    seed, base, balanced = job
    nb51 = load_notebook("nb51", "51_online_refit.py")
    hook = load_notebook("nb52", "52_saponification_balance.py").cap_saponification if balanced else None
    label = base + ("+balance" if balanced else "")
    row, traces = nb51.run(base, hook, label, seed)
    return row, {f"{seed}_{k}": v for k, v in traces.items()}


def summary(rows: list) -> None:
    """Per controller: plants stalled, lowest extraction acid, hours below spec and end purity."""
    for base, balanced in CONTROLLERS:
        label = base + ("+balance" if balanced else "")
        r = [x for x in rows if x["variant"] == label]
        if not r:
            continue
        stalled = sum(1 for x in r if x["stalled_at_h"] != "")
        print(f"{label:16s} plants {len(r):2d}  stalled {stalled}  lowest extraction acid "
              f"{min(x['acid_extraction_min_mM'] for x in r):6.2f} mmol/L  hours below spec "
              f"{np.mean([x['hours_below_spec'] for x in r]):5.1f} (mean)  end purity "
              f"{100 * min(x['purity_end'] for x in r):6.2f} % (worst)", flush=True)


def main():
    """Every plant and controller in parallel (the slow online runs first); rows as each finishes."""
    jobs = [(s, b, bal) for b, bal in CONTROLLERS for s in SEEDS]
    rows, traces = [], {}
    for job, result in run_jobs(run, jobs, workers()):
        if result is None:
            print("no result for", job, flush=True)
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print(f"plant {row['seed']:2d} {row['variant']:16s} stalled {row['stalled_at_h'] or 'no':>5}  "
              f"min extraction acid {row['acid_extraction_min_mM']:6.2f} mmol/L  max sapon. {row['sapon_max']:.3f}  "
              f"h below spec {row['hours_below_spec']:4.1f}", flush=True)
        write_rows(OUT, sorted(rows, key=lambda r: (r["seed"], r["variant"])), first=("seed", "variant"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    summary(rows)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
