"""The rest-point check every step instead of every 12 h (v1.3, eighth test)

The question (Emilio, Oct 25). Notebook 55 checks where the model would settle every 24 steps (12 h).
Why not every step? The drift it catches builds over days, so 12 h was a guess at "often enough".
One change: REST_CHECK_STEPS = 1. Same ten long runs (true model, no upset, 600 h from rest, notebook
54's logging). Compared plant by plant with notebook 55 (every 12 h) and notebook 54 (no check):
drift caught, hours below spec, and computing time.

Run:  pixi run python notebooks/58_rest_check_every_step.py
Writes runs/rest_check_every_step_ndpr_main.csv (one row per plant) and .npz (the traces).
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
OUT = HERE / "runs" / f"rest_check_every_step_{CONFIG}.csv"


def run(seed: int):
    """Notebook 55's run on one plant with a check every step: (row, traces)."""
    nb55 = load_notebook("nb55", "55_rest_point_check.py")
    nb55.REST_CHECK_STEPS = 1                                    # the one change
    return nb55.run(seed)


def main():
    """Every plant in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, SEEDS, workers(), timeout_s=10 * 3600):
        if result is None:
            print("no result for plant", job, flush=True)
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        write_rows(OUT, sorted(rows, key=lambda r: r["seed"]), first=("seed",))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
