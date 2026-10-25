"""The controller's purity output and floors in log impurity (v1.3, eleventh test)

The question (Emilio, Oct 25). The MPC holds purity as a fraction. Near 100 % that barely moves when a
handle moves, so its linear view goes flat, which is part of why plants 7 and 8 drifted (notebook 54).
In -log10(1 - purity), the "number of nines", 99.9 against 99.99 % is a full unit; logarithms were the
biggest single help against this nonlinearity in distillation (Mejdell and Skogestad 1991). One change:
MPCSettings.log_impurity = True (new in src/mpc.py, off by default). The floors are moved into nines
and each floor's slack scaled by the slope there, so a thousandth of purity below a floor costs what it
costs now; only the shape away from the floors changes.

The method. Notebook 60's run with this setting instead of its own: notebook 54's ten long runs (true
model, no upset, 600 h from rest, the same logging). Compared plant by plant with notebook 54 (purity
as a fraction, no check) and notebook 55 (the rest-point check).

Run:  pixi run python notebooks/61_log_impurity.py
Writes runs/log_impurity_ndpr_main.csv (one row per plant) and .npz (the traces).
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

LABEL = "true+log_impurity"
OUT = HERE / "runs" / "log_impurity_ndpr_main.csv"


def run(seed: int):
    """Notebook 60's run on one plant with log_impurity instead of its setting: (row, traces)."""
    nb60 = load_notebook("nb60", "60_loss_recovery.py")
    nb60.SETTINGS, nb60.LABEL = {"log_impurity": True}, LABEL       # the one change
    return nb60.run(seed)


def main():
    """Every plant in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, tuple(range(1, 11)), workers(), timeout_s=10 * 3600):
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
