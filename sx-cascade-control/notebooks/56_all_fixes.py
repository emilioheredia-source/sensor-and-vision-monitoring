"""All three fixes together, from a wrong model, over the long runs (v1.3, sixth test)

The question (Oct 24). Each fix was tested alone on the ten plants: the online refit (notebook 53),
the saponification balance (notebooks 52, 53) and the rest-point check (notebook 55). Together, from
the ridge's wrong model, over notebook 54's 600 h, does anything fight?

The method. Notebook 54's long run (no upset, 600 h from rest, plants 1 to 10, the same logging),
two controllers:
  ridge                 the ridge's model, no fix (the reference)
  online+balance+rest   the ridge's model refitted every 2 h (notebook 51), the saponification capped
                        by the balance (notebook 52), and the rest-point check every 12 h (notebook 55)
Every rebuilt controller gets the cap and the check again after each interval (the hook runs after
the refit), so a new controller checks its rest point on its first step.

Run:  pixi run python notebooks/56_all_fixes.py
Writes runs/all_fixes_ndpr_main.csv (one row per plant and controller) and .npz (the traces).
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
CONTROLLERS = ("online+balance+rest", "ridge")       # the slow one first
OUT = HERE / "runs" / f"all_fixes_{CONFIG}.csv"


@lru_cache(maxsize=None)
def study(name: str):
    """Notebooks 52, 54 and 55, loaded once per process."""
    files = {"nb52": "52_saponification_balance.py", "nb54": "54_slow_drift.py", "nb55": "55_rest_point_check.py"}
    return load_notebook(name, files[name])


def all_fixes(loop, rec, series: dict) -> None:
    """After each interval (and any refit): the rest-point check on, the balance cap, the drift log."""
    loop.controller.s.rest_check_every = study("nb55").REST_CHECK_STEPS
    study("nb52").cap_saponification(loop, rec, series)
    study("nb54").log_drift(loop, rec, series)


def run(job):
    """One plant and controller for notebook 54's hours: (row, traces). job: (seed, controller)."""
    seed, label = job
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.HOURS = study("nb54").LONG_HOURS                    # the long run; everything else as notebook 51
    fixed = label != "ridge"
    row, traces = nb51.run("online" if fixed else "ridge", all_fixes if fixed else study("nb54").log_drift, label, seed)
    series = {k[len(label) + 1:]: v for k, v in traces.items()}
    keep = ("stalled_at_h", "sapon_max", "acid_extraction_min_mM", "hours_below_spec", "refits", "seconds")
    return {"variant": label, **study("nb54").drift_measures(seed, series), **{k: row[k] for k in keep}}, \
        {f"{seed}_{k}": v for k, v in traces.items()}


def finished_runs() -> tuple:
    """What an earlier, interrupted run of this notebook already wrote: (rows, traces); empty if none.
    Why: the first run (Oct 24) stopped on a summary bug after 11 of 20 runs, most of them the slow ones."""
    import csv
    if not OUT.exists():
        return [], {}
    rows = list(csv.DictReader(open(OUT)))
    with np.load(OUT.with_suffix(".npz")) as d:
        return rows, {k: d[k] for k in d.files}


def main():
    """Every plant and controller not already written, in parallel; rows and traces as each finishes."""
    rows, traces = finished_runs()
    done = {(int(r["seed"]), r["variant"]) for r in rows}
    jobs = [(s, c) for c in CONTROLLERS for s in study("nb54").SEEDS if (s, c) not in done]
    for job, result in run_jobs(run, jobs, workers(), timeout_s=10 * 3600):
        if result is None:
            print("no result for", job, flush=True)
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        write_rows(OUT, sorted(rows, key=lambda r: (int(r["seed"]), r["variant"])), first=("seed", "variant"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
