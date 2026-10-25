"""The controller holding recovery counted by losses (v1.3, tenth test)

The question (Emilio, Oct 25). The MPC holds recovery as NdPr in the product now over NdPr fed now.
While the circuit's stock drains that reads above 100 % (notebook 54: up to 104 % on the two plants
heading off spec), so the controller sees good news. Recovery counted by what's lost, 1 - NdPr in the
raffinate / NdPr fed, can't read above 100 % and doesn't jump when the stock drains. Would it be the
better thing to control? One change: MPCSettings.recovery_from_losses = True (new in src/mpc.py, off
by default). It's part of the controller's output function, which is built when the controller is,
so the run machinery's controller builder gets the setting (and so does any rebuilt controller).

The method. Notebook 54's run exactly (true model, no upset, 600 h from rest, plants 1 to 10, the
same logging), compared plant by plant with notebook 54 (product-based recovery, no check) and
notebook 55 (product-based, rest-point check): drift, recovery by losses, hours below spec, reagent.

Run:  pixi run python notebooks/60_loss_recovery.py
Writes runs/loss_recovery_ndpr_main.csv (one row per plant) and .npz (the traces).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                              # noqa: E402
from functools import lru_cache, partial                                # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import load_notebook, run_jobs, workers, write_rows    # noqa: E402

CONFIG = "ndpr_main"
SETTINGS = {"recovery_from_losses": True}         # the one change (notebook 61 swaps in its own)
LABEL = "true+loss_recovery"
OUT = HERE / "runs" / f"loss_recovery_{CONFIG}.csv"


@lru_cache(maxsize=None)
def nb54():
    """Notebook 54 (its logging, measures and settings), loaded once per process."""
    return load_notebook("nb54", "54_slow_drift.py")


def run(seed: int):
    """Notebook 54's run on one plant with the controller built with SETTINGS: (row, traces)."""
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.HOURS = nb54().LONG_HOURS                                     # the long run, as notebook 54
    nb51.mpc_over_level_loops = partial(nb51.mpc_over_level_loops, **SETTINGS)   # the one change
    row, traces = nb51.run("true", nb54().log_drift, LABEL, seed)
    series = {k[len(LABEL) + 1:]: v for k, v in traces.items()}
    return {**nb54().drift_measures(seed, series), "stalled_at_h": row["stalled_at_h"], "seconds": row["seconds"]}, \
        {f"{seed}_{k}": v for k, v in traces.items()}


def main():
    """Every plant in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, nb54().SEEDS, workers(), timeout_s=10 * 3600):
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
