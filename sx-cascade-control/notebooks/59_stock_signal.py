"""The circuit's NdPr stock as a drift signal (v1.3, ninth test; analysis of existing runs)

The question (Emilio, Oct 25). Could the controller watch the circuit's metal stock directly? The
stock change is a running sum of (fed - product - raffinate), all measured flows and assays, so it
needs no model. Notebook 54 logged it: over 600 h the two plants that drifted lost about 900 mol of
NdPr from the stock, the other eight 17 to 138. Every plant also drains at start-up, while it settles
from the design handles.

The rule tested: alarm when the stock has drained faster than THRESHOLD mol/h over the last WINDOW_H,
for HOLD_H in a row, after the first START_H (start-up). No run is repeated; the rule is applied to
the traces already written:
  notebook 54   true model, no fix: should flag the plants that drift (7, 8) and no other
  notebook 55   true model, rest-point check: the drift is fixed, so it should stay quiet
  notebook 56   the ridge's wrong model, no fix: should flag the plants that drift there
Each alarm is compared with when the plant's settled purity first fell below spec (the rest-point
log) and when the recovery reading first went above 100 %.

Run:  pixi run python notebooks/59_stock_signal.py
Writes runs/stock_signal_ndpr_main.csv (one row per run and plant).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "notebooks"))
from study_tools import SPEC, write_rows                                # noqa: E402

THRESHOLD = 0.5            # mol/h of NdPr drained from the stock (chosen: start-up aside, the quiet plants stay under 0.35)
WINDOW_H = 48.0            # h over which the drain rate is measured
HOLD_H = 24.0              # h the rate has to stay above the threshold
START_H = 96.0             # h of start-up ignored (every plant drains while it settles)
RUNS = (("54 true model", "slow_drift_ndpr_main.npz", "true"),
        ("55 rest check", "rest_point_check_ndpr_main.npz", "true+rest"),
        ("56 ridge, no fix", "all_fixes_ndpr_main.npz", "ridge"))
OUT = HERE / "runs" / "stock_signal_ndpr_main.csv"


def drain_rate(t: np.ndarray, stock: np.ndarray) -> np.ndarray:
    """mol/h drained from the stock over the last WINDOW_H at each time (NaN before a full window)."""
    earlier = np.interp(t - WINDOW_H, t, stock, left=np.nan)
    return (earlier - stock) / WINDOW_H


def first_alarm(t: np.ndarray, rate: np.ndarray):
    """The first time the drain rate has stayed above THRESHOLD for HOLD_H after START_H, or None."""
    above = (rate > THRESHOLD) & (t >= START_H)
    since = None
    for ti, a in zip(t, above):
        since = (since if since is not None else ti) if a else None
        if since is not None and ti - since >= HOLD_H:
            return float(ti)
    return None


def first_time(t: np.ndarray, hit: np.ndarray):
    """The first time a condition holds, or None."""
    return float(t[hit][0]) if hit.any() else None


def plant_row(label: str, traces, key: str, seed: int) -> dict:
    """One plant of one run: the alarm time against the settled purity and the recovery reading."""
    t, stock = traces[f"{key}_t"], traces[f"{key}_stock_change"]
    rest_t, rest_p = traces.get(f"{key}_rest_t"), traces.get(f"{key}_rest_purity")
    rest_off = first_time(rest_t, (rest_p < SPEC) & (rest_t >= START_H)) if rest_t is not None and len(rest_t) else None
    return {"run": label, "seed": seed, "alarm_h": first_alarm(t, drain_rate(t, stock)) or "",
            "settled_below_spec_after_start_h": rest_off or "",
            "recovery_read_over_100_h": first_time(t, traces[f"{key}_recovery"] > 1.0) or "",
            "stock_change_end_mol": float(stock[-1]), "hours_run": float(t[-1])}


def run_rows(label: str, name: str, variant: str) -> list:
    """The rule on every plant of one run's traces."""
    with np.load(HERE / "runs" / name) as traces:
        seeds = sorted({int(k.split("_")[0]) for k in traces.files if k.endswith(f"_{variant}_stock_change")})
        return [plant_row(label, traces, f"{seed}_{variant}", seed) for seed in seeds]


def main():
    """The rule on every plant of the three runs; a summary per run."""
    rows = [row for run in RUNS for row in run_rows(*run)]
    write_rows(OUT, rows, first=("run", "seed"))
    for label, _, _ in RUNS:
        r = [x for x in rows if x["run"] == label]
        flagged = [x["seed"] for x in r if x["alarm_h"] != ""]
        drifted = [x["seed"] for x in r if x["settled_below_spec_after_start_h"] != ""]
        print(f"{label:18s} alarm on {flagged}; settled below spec after start-up on {drifted}")
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
