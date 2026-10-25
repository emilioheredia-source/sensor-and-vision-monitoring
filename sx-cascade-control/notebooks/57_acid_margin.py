"""A margin above the acid floor in the saponification balance (v1.3, seventh test)

The question (Oct 25). With all three fixes over 600 h (notebook 56) the balance cap holds the
extraction acid at the floor itself on plants 3, 9 and 10 (1.00 to 1.04 mmol/L), never under it: the
controller uses all the base the balance allows. Keeping the planned steady state away from a limit
is the standard advice (Kuntz and Rawlings 2024, constraint back-offs in the steady-state target).
One change: the balance aims the raffinate MARGIN above the floor instead of at it. What does the
margin hold, and what does it cost?

The method. Notebook 56's combined run (ridge's model refitted every 2 h, the balance cap, the
rest-point check every 12 h; 600 h, no upset, notebook 54's logging) on the three plants that reached
the floor, with notebook 52's RAFFINATE_MARGIN set. Compared with notebook 56's rows for those plants.

Run:  pixi run python notebooks/57_acid_margin.py
Writes runs/acid_margin_ndpr_main.csv (one row per plant) and .npz (the traces).
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
SEEDS = (3, 9, 10)                             # the plants where the cap held the acid at the floor
MARGIN = 3e-3                                  # mol/L above the 1 mmol/L floor
LABEL = "online+balance+rest+margin"
OUT = HERE / "runs" / f"acid_margin_{CONFIG}.csv"


def run(seed: int):
    """Notebook 56's combined run on one plant with the margin: (row, traces)."""
    nb56 = load_notebook("nb56", "56_all_fixes.py")
    nb56.study("nb52").RAFFINATE_MARGIN = MARGIN               # the one change
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.HOURS = nb56.study("nb54").LONG_HOURS
    row, traces = nb51.run("online", nb56.all_fixes, LABEL, seed)
    series = {k[len(LABEL) + 1:]: v for k, v in traces.items()}
    keep = ("stalled_at_h", "sapon_max", "acid_extraction_min_mM", "hours_below_spec", "refits", "seconds")
    return {"variant": LABEL, **nb56.study("nb54").drift_measures(seed, series), **{k: row[k] for k in keep}}, \
        {f"{seed}_{k}": v for k, v in traces.items()}


def main():
    """The three plants in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, SEEDS, min(workers(), len(SEEDS)), timeout_s=10 * 3600):
        if result is None:
            print("no result for plant", job, flush=True)
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        write_rows(OUT, sorted(rows, key=lambda r: r["seed"]), first=("seed", "variant"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
