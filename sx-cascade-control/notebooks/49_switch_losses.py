"""How much NdPr really leaves in the raffinate through the feed switch? (Part E1 of the Oct 23 check)

The question. Notebook 41's recovery is the product rate over the NdPr fed at that moment. Through
the switch to the rich feed the NdPr fed doubles within an hour, and the product can't follow until
that metal has worked through the cascade, so recovery reads about 53 % for 4 to 5 hours with any
controller, the true model included. That's metal in transit, not metal lost. Recovery keeps its
definition (the plant runs near steady state almost all the time); for the switch, this adds what
leaves in the raffinate, against what was fed, over windows long enough to cover the settling.

The method. Notebook 41's runs, longer: the main case, the MPC over level loops, full state, the feed
switch at 5 h; the five plants that dipped (seeds 3, 10, 14, 16, 17) and three weak plants that
didn't (2, 6, 9); the controller on the nominal model (as run) and on the plant's true model; 240 h.
For each run: the NdPr lost to the raffinate over the NdPr fed, cumulative from the switch over
24, 48, 96, 168 and 235 h, and before the switch; recovery as notebook 41 reads it; purity.

Two math threads per worker, the project's setting (pixi.toml), as notebook 41 ran: these runs start
from a trimmed operating point, and the trim depends on the thread count (FINDINGS phase 9).

Run:  pixi run python notebooks/49_switch_losses.py
Writes runs/switch_losses_ndpr_main.csv (one row per run) and .npz (lost, fed, purity, recovery over t).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"

import importlib.util                                                     # noqa: E402
import sys                                                                # noqa: E402
import warnings                                                           # noqa: E402
from pathlib import Path                                                  # noqa: E402

import numpy as np                                                        # noqa: E402

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import run_jobs, workers, write_rows                    # noqa: E402

HOURS, T_SWITCH = 240.0, 5.0
WINDOWS = (24.0, 48.0, 96.0, 168.0, 235.0)
SEEDS = (3, 10, 14, 16, 17, 2, 6, 9)
VARIANTS = ("as_run", "true_model")
OUT = HERE / "runs" / "switch_losses_ndpr_main.csv"


def nb41():
    """Notebook 41's definitions as a module (its study is top-level code, so only the part before it
    runs), with the run length set to HOURS."""
    spec = importlib.util.spec_from_file_location("nb41", HERE / "notebooks" / "41_feed_switch_replay.py")
    module = importlib.util.module_from_spec(spec)
    source = spec.loader.get_source("nb41").split("\nrows, traces = [], {}")[0]
    module.__dict__["__file__"] = str(spec.origin)
    old, sys.argv = sys.argv, [sys.argv[0]]
    try:
        exec(compile(source, str(spec.origin), "exec"), module.__dict__)
    finally:
        sys.argv = old
    module.HOURS = HOURS
    return module


def run(job):
    """One plant and controller through the switch: (row, traces). The NdPr fed is recovered from the
    loss and the loss as a share of feed, both of which every run records."""
    seed, variant = job
    warnings.filterwarnings("ignore")
    r, _ = nb41().run(seed, variant)
    t = np.asarray(r.t)
    lost = np.asarray(r.series["lost_rate"], float)                          # mol/h of NdPr in the raffinate
    pct = np.asarray(r.series["lost_pct_of_feed"], float)
    fed = np.where(pct > 0, lost / np.maximum(pct, 1e-12) * 100.0, np.nan)    # mol/h of NdPr fed
    pur, rec = np.asarray(r.series["purity"], float), np.asarray(r.series["recovery"], float)
    dt = np.diff(t)
    row = {"seed": seed, "variant": variant, "purity_min": pur[1:].min(), "recovery_min": rec[1:].min(),
           "hours_recovery_below_95": float(dt[rec[1:] < 0.95].sum()), **loss_shares(t, lost, fed)}
    return row, {"t": t, "lost": lost, "fed": fed, "purity": pur, "recovery": rec}


def loss_shares(t, lost, fed) -> dict:
    """NdPr lost as a share of NdPr fed, %: before the switch (hours 1 to 5, the plant still being
    pulled from its design handles), and cumulative from the switch over each window."""
    dt = np.diff(t)
    before = (t > 1) & (t < T_SWITCH)
    out = {"lost_pct_before_switch": float(100 * lost[before].mean() / fed[before].mean())}
    for w in WINDOWS:
        sel = (t[1:] > T_SWITCH) & (t[1:] <= T_SWITCH + w)
        out[f"lost_pct_{int(w)}h"] = float(100 * np.sum(lost[1:][sel] * dt[sel]) / np.sum(fed[1:][sel] * dt[sel]))
    return out


def main():
    """Every plant and controller; rows and traces written as each run finishes."""
    rows, traces = [], {}
    for _, (row, tr) in run_jobs(run, [(s, v) for s in SEEDS for v in VARIANTS], workers(threads_each=2)):
        rows.append(row)
        traces.update({f"{row['variant']}_{row['seed']}_{k}": v for k, v in tr.items()})
        print(f"seed {row['seed']:2d} {row['variant']:10s}: lost {row['lost_pct_before_switch']:.2f} % before; after the switch "
              + " ".join(f"{int(w)} h {row[f'lost_pct_{int(w)}h']:.2f} %" for w in WINDOWS)
              + f" | recovery min {100 * row['recovery_min']:.1f} %, purity min {100 * row['purity_min']:.2f} %", flush=True)
        write_rows(OUT, rows, first=("seed", "variant"))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
