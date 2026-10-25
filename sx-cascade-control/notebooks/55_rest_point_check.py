"""Does checking where the model would settle stop the slow drift near 100 %? (v1.3, fifth test)

The question (Oct 24). Notebook 54 measures the drift with the true model: purity reads 100 % while
the plant, on the handles held, would settle below spec. The controller's steady-state floors use a
long-run view linearized where the model is now, which near 100 % purity is flat. The literature's
answer for a slow, nonlinear plant is to look at where it's heading with the full model (Chew, Jones
and Wilson 2006 recompute the reachable steady state every step; Kuntz and Rawlings 2024 keep the
steady-state target honest). One change: the MPC's rest-point check (MPCSettings.rest_check_every,
new in src/mpc.py): every REST_CHECK_STEPS steps the model is solved to rest on the handles held
now, linearized there, and the steady-state floors are anchored on that rest point.

The method. Notebook 54's run exactly (true model, no upset, LONG_HOURS from rest, plants 1 to 10,
the same logging), with only the check switched on. The setting is switched on after the first
interval (the run machinery builds the controller), so the first check comes at about 12 h.
Compared plant by plant with runs/slow_drift_ndpr_main.csv.

Run:  pixi run python notebooks/55_rest_point_check.py
Writes runs/rest_point_check_ndpr_main.csv (one row per plant) and .npz (the traces).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                              # noqa: E402
from functools import lru_cache                                         # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import load_notebook, run_jobs, workers, write_rows    # noqa: E402

CONFIG = "ndpr_main"
REST_CHECK_STEPS = 24                          # one check every 12 h of 0.5 h steps
OUT = HERE / "runs" / f"rest_point_check_{CONFIG}.csv"


@lru_cache(maxsize=None)
def nb54():
    """Notebook 54 (its logging, measures and settings), loaded once per process."""
    return load_notebook("nb54", "54_slow_drift.py")


def checked_and_logged(loop, rec, series: dict) -> None:
    """Switch the rest-point check on (idempotent), then notebook 54's logging."""
    loop.controller.s.rest_check_every = REST_CHECK_STEPS
    nb54().log_drift(loop, rec, series)


def run(seed: int):
    """One plant with the true model and the rest-point check for notebook 54's hours: (row, traces)."""
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.HOURS = nb54().LONG_HOURS                           # the long run; everything else as notebook 51
    row, traces = nb51.run("true", checked_and_logged, "true+rest", seed)
    series = {k[len("true+rest_"):]: v for k, v in traces.items()}
    return {**nb54().drift_measures(seed, series), "stalled_at_h": row["stalled_at_h"], "seconds": row["seconds"]}, \
        {f"{seed}_{k}": v for k, v in traces.items()}


def main():
    """Every plant in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, nb54().SEEDS, workers(), timeout_s=6 * 3600):
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
