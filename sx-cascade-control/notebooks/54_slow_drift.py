"""How often, and how far, does a plant drift off spec while purity reads 100 %? (v1.3, fourth test)

The question (Oct 24). With the true model, plants 7 and 8 read 100 % purity and 102-103 % recovery
after 400 h, handles settled, while at rest on those handles their purity would be 95.6-96.3 %
(FINDINGS phase 9): near 100 % purity barely answers a move, so the controller's linear view sees
nothing, while the slow metal stock carries the plant off. Before changing the controller, measure
it: on how many of the ten plants, how far, and how fast; and which signal would have shown it.

The method. Notebook 51's run with the true model (the drift isn't a model error), no upset,
LONG_HOURS from rest on the design handles, plants 1 to 10. Logged every interval besides
notebook 51's series:
  recovery_loss   1 - NdPr in the raffinate / NdPr fed, now (recovery counted by what's lost)
  stock_change    the NdPr the circuit has gained since the start, mol: the running integral of
                  fed - product - raffinate (negative: the stock is draining)
and every REST_EVERY_H the purity and recovery the plant would settle at if the handles were held
where they are now (study_tools.settle from the plant's state).

Run:  pixi run python notebooks/54_slow_drift.py
Writes runs/slow_drift_ndpr_main.csv (one row per plant) and .npz (the traces).
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
from src.circuit import Inputs                                          # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from study_tools import SPEC, load_notebook, purity_recovery, run_jobs, settle, workers, write_rows   # noqa: E402

CONFIG = "ndpr_main"
SEEDS = tuple(range(1, 11))
LONG_HOURS = 600.0
REST_EVERY_H = 24.0
CUT = "LaCe"                                    # NdPr as one product (the main case)
OUT = HERE / "runs" / f"slow_drift_{CONFIG}.csv"


def ndpr_flows(plant, x, u_vec) -> tuple:
    """NdPr in (feed), out with the product and out with the raffinate, mol/h."""
    heavy = np.array(heavy_mask(plant.p, CUT))
    u = Inputs.from_vector(np.asarray(u_vec, float), plant.n, plant.N)
    o = plant.outlets(x, u.numpy())
    fed = float(u.Q_feed) * np.asarray(u.c_feed, float)[heavy].sum()
    return (fed, float(o["Q_prod"]) * np.atleast_1d(o["c_prod"])[heavy].sum(),
            float(o["Q_raff"]) * np.atleast_1d(o["c_raff"])[heavy].sum())


def log_drift(loop, rec, series: dict) -> None:
    """After each interval: recovery by losses, the stock change, and every REST_EVERY_H where the
    plant would settle on the handles held now."""
    plant = loop.plant
    fed, prod, raff = ndpr_flows(plant, rec.x, rec.u)
    t_prev = series["t"][-2] if len(series["t"]) > 1 else 0.0
    stock = (series["stock_change"][-1] if series.get("stock_change") else 0.0) + (fed - prod - raff) * (rec.t - t_prev)
    series.setdefault("recovery_loss", []).append(1.0 - raff / fed)
    series.setdefault("stock_change", []).append(stock)
    if abs(rec.t % REST_EVERY_H) < 1e-6:
        rest = purity_recovery(plant, settle(plant, rec.u, x0=rec.x), rec.u, CUT)
        for k, v in (("rest_t", rec.t), ("rest_purity", rest[0]), ("rest_recovery", rest[1])):
            series.setdefault(k, []).append(v)


def drift_measures(seed: int, series: dict) -> dict:
    """The plant's row: what the reading says against where the plant is heading. A run that stopped
    before 24 h (a stalled simulation) has no rest-point checks: those columns are left empty."""
    t, pur, rec = (np.array(series[k]) for k in ("t", "purity", "recovery"))
    rest_t, rest_p = np.array(series.get("rest_t", [])), np.array(series.get("rest_purity", []))
    off, late = rest_t[rest_p < SPEC], t >= 24
    return {"seed": seed, "purity_end": pur[-1], "purity_min_after_24h": pur[late].min() if late.any() else "",
            "recovery_end": rec[-1], "recovery_max_after_24h": rec[late].max() if late.any() else "",
            "recovery_loss_end": series["recovery_loss"][-1], "stock_change_end_mol": series["stock_change"][-1],
            "rest_purity_end": rest_p[-1] if rest_p.size else "", "rest_purity_min": rest_p.min() if rest_p.size else "",
            "first_rest_below_spec_h": float(off[0]) if off.size else ""}


def run(seed: int):
    """One plant with the true model for LONG_HOURS: (the row, the traces)."""
    nb51 = load_notebook("nb51", "51_online_refit.py")
    nb51.HOURS = LONG_HOURS                                  # the long run; everything else as notebook 51
    row, traces = nb51.run("true", log_drift, "true", seed)
    series = {k[len("true_"):]: v for k, v in traces.items()}
    return {**drift_measures(seed, series), "stalled_at_h": row["stalled_at_h"], "seconds": row["seconds"]}, \
        {f"{seed}_{k}": v for k, v in traces.items()}


def main():
    """Every plant in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for job, result in run_jobs(run, SEEDS, workers(), timeout_s=4 * 3600):
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
