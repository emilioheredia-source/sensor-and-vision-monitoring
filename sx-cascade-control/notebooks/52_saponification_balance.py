"""Does tying the saponification to the acid balance stop the reagent runaway? (v1.3, second test)

The question (Oct 24). Saponifying the extractant pre-neutralizes it with base, so that loading a
rare-earth ion gives back sodium instead of three H+. How much base a plant can use is set by
counting: the metal the organic carries out, and the acid coming in. Base beyond that has nothing
to do but eat the acid in the stages, which is what plant 10 showed (notebook 51). Plants set the
saponification as a ratio to the metal load, and the balance tying the scrub acid, the organic flow
times its saponification and the feed metal is written down by Yang, Xu, Lu and Ding (2015). The
balance is counting, not the fitted numbers, so it stays right when the model is wrong.

The balance (extraction and scrub sections, at rest; it closes exactly on plant 10's rest point):
    base fed = s * L * Q_org = 3 * M_org + h_feed * Q_feed + h_scrub * Q_scrub - h_raff * Q_raff
  s the saponified fraction, L the extractant monomer (mol/L), M_org the metal the loaded organic
  carries to the strip (mol/h), h the acids (mol/L), Q the flows (L/h). With the raffinate held at
  the controller's acid floor, the most saponification the plant can use now is
    s_max = (3 * M_org + h_feed * Q_feed + h_scrub * Q_scrub - floor * Q_raff) / (L * Q_org)
  Every interval s_max is computed from what the plant shows (the loaded organic, as a loaded-organic
  assay would read it; the acids and flows as metered) and set as the controller's upper bound on
  the saponification, never asking for a faster cut than the handle's rate limit.

The method. Notebook 51's runs on plant 10 (48 h, same plant, same settings), with only this cap
added: the ridge's model, the true model, and the online refit. Compared row for row with notebook
51's ridge, true and online rows.

Run:  pixi run python notebooks/52_saponification_balance.py
Writes runs/saponification_balance_ndpr_main.csv (one row per controller) and .npz (the traces).
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
from study_tools import load_notebook, run_jobs, workers, write_rows    # noqa: E402

CONFIG = "ndpr_main"
BASES = ("ridge", "true", "online")            # notebook 51's variants this adds the cap to
DEFAULT_FLOOR = 1e-3                           # mol/L, if the controller has no acid floor (pH 3)
RAFFINATE_MARGIN = 0.0                         # mol/L the balance aims above the floor (notebook 57 sets it)
OUT = HERE / "runs" / f"saponification_balance_{CONFIG}.csv"


def saponification_limit(plant, x, u_vec, floor: float) -> float:
    """The most saponification the plant can use now with the raffinate at `floor` (the balance above)."""
    u = Inputs.from_vector(np.asarray(u_vec, float), plant.n, plant.N)
    c_org = plant.unpack(plant.observable(x))[1]
    last_scrub = max(j for j in range(plant.N) if plant.cmap.section(j) == "scrub")
    metal_out = float(np.sum(np.asarray(c_org[last_scrub], float))) * float(u.Q_org)
    acid_in = float(u.h_feed) * float(u.Q_feed) + float(u.h_scrub) * float(u.Q_scrub)
    q_raff = float(plant.outlets(x, u.numpy())["Q_raff"])
    extractant = plant.p.raw["extractant"]["concentration_monomer"]
    return (plant.p.ligands_per_metal * metal_out + acid_in - floor * q_raff) / (extractant * float(u.Q_org))


def handle_entry(ctl, name: str) -> int:
    """Where a one-entry handle sits in the controller's handle vector."""
    i = ctl._u_names.index(name)
    return int(sum(ctl._u_sizes[:i]))


def cap_saponification(loop, rec, series: dict) -> None:
    """After each interval: the balance's limit from what the plant shows, set as the controller's
    upper bound on the saponification (never below what one rate-limited move can reach)."""
    ctl = loop.controller
    floor = ctl.s.acid_floor if ctl.s.acid_floor is not None else DEFAULT_FLOOR
    limit = saponification_limit(loop.plant, rec.x, rec.u, floor + RAFFINATE_MARGIN)
    i = handle_entry(ctl, "saponified_fraction")
    reachable = float(ctl.u_prev[i]) - float(ctl._rate[i])
    ctl._ub_u[i] = max(min(ctl.s.bounds["saponified_fraction"][1], limit), reachable)
    series.setdefault("sapon_limit", []).append(limit)


def run(base: str):
    """Notebook 51's run for `base` with the saponification capped by the balance."""
    nb51 = load_notebook("nb51", "51_online_refit.py")
    row, traces = nb51.run(base, after_step=cap_saponification, label=f"{base}+balance")
    limits = traces[f"{base}+balance_sapon_limit"]
    return {**row, "sapon_limit_min": float(limits.min()), "sapon_limit_max": float(limits.max())}, traces


def main():
    """The three controllers with the cap, in parallel; rows and traces as each finishes."""
    rows, traces = [], {}
    for _, result in run_jobs(run, BASES, min(workers(), len(BASES))):
        if result is None:
            continue
        row, tr = result
        rows.append(row)
        traces.update(tr)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        write_rows(OUT, rows, first=("variant",))
        np.savez(OUT.with_suffix(".npz"), **traces)
    print("wrote", OUT.name)


if __name__ == "__main__":
    main()
