"""A penalty that starts before the acid floor and grows faster below it (v1.3, twelfth test)

The question (Emilio, Oct 25). The acid floor is a soft constraint with a squared penalty: nothing
until the acid reaches the floor, then little for a small shortfall. With the purity floor's weight
far above it, the controller walks the acid down to the floor: with the true model it went under on
5 of 10 plants in 48 h (notebook 53). Would a penalty that starts a little before the floor and gets
steeper below it do better? One change: MPCSettings.acid_band = BAND (new in src/mpc.py, off by
default), a second soft floor BAND above the first with the same weight, so below floor + BAND the
cost starts, and below the floor it grows twice as fast.

The method. Notebook 53's 48 h runs on plants 1 to 10 with the ridge's model and with the true model,
the band added and nothing else (no refit, no balance cap). Compared plant by plant with notebook 53's
ridge, true, ridge+balance and true+balance rows.

Run:  pixi run python notebooks/62_acid_band.py
Writes runs/acid_band_ndpr_main.csv (one row per plant and model) and .npz (the traces).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys                                                              # noqa: E402
from functools import partial                                           # noqa: E402
from pathlib import Path                                                # noqa: E402

import numpy as np                                                      # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import load_notebook, run_jobs, workers, write_rows    # noqa: E402

BAND = 3e-3                                    # mol/L above the 1 mmol/L floor where the cost starts
BASES = ("ridge", "true")
OUT = HERE / "runs" / "acid_band_ndpr_main.csv"


def run(job):
    """Notebook 51's 48 h run on one plant and model with the acid band: (row, traces). job: (seed, base)."""
    seed, base = job
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.mpc_over_level_loops = partial(nb51.mpc_over_level_loops, acid_band=BAND)   # the one change
    row, traces = nb51.run(base, None, f"{base}+band", seed)
    return row, {f"{seed}_{k}": v for k, v in traces.items()}


def main():
    """Every plant and model in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, [(s, b) for b in BASES for s in range(1, 11)], workers()):
        if result is None:
            print("no result for", job, flush=True)
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print(f"plant {row['seed']:2d} {row['variant']:11s} stalled {row['stalled_at_h'] or 'no':>5}  min extraction acid "
              f"{row['acid_extraction_min_mM']:6.2f} mmol/L  h below spec {row['hours_below_spec']:4.1f}", flush=True)
        write_rows(OUT, sorted(rows, key=lambda r: (r["seed"], r["variant"])), first=("seed", "variant"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
