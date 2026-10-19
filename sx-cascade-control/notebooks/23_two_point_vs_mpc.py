"""Does holding two compositions inside the cascade do what the model-based MPC can't?

The question. On realistic plants that separate a little worse than the model thinks (Pr/La+Ce
below the model's 2.0), the MPC planning on the nominal model can't find the plant's operating
point: it pushes the scrub when the plant needs organic (notebooks 14, 21). Notebook 22 found
which measurements, held at their design values by the scrub and the organic flows, take the
plants closest to spec at steady state. Here those loops run on the plants, against the MPC.

The method.
  1. Pairs: the best pairs from notebook 22 (runs/what_to_hold_ndpr_plant.csv), each checked for
     its relative gain at the design point (lambda = g11 g22 / (g11 g22 - g12 g21)). A pairing
     with lambda < 0 makes two integral loops fight, so it's dropped; nearest to 1 is best.
  2. Plants: the seed study's realistic plants (generator "ratios") that start off spec (the hard
     ones), and three that start on spec, from the same trimmed operating point every study uses.
  3. Controllers, full information, no upset, 150 h:
        mpc         the MPC as built
        two_point   scrub holds A, organic holds B (src/two_point.py), level loops underneath
  4. Scored: purity and recovery at the end, where the plant would rest on the final handles,
     the handles, and hours until both targets hold to the end.

Run:  pixi run python notebooks/23_two_point_vs_mpc.py [pairs | run SEED ...]
Writes runs/two_point_pairs_ndpr_plant.csv and runs/two_point_vs_mpc_ndpr_plant.csv
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Inputs                                          # noqa: E402
from src.metrics import heavy_mask, product_purity                      # noqa: E402
from src.runs import RunSession, RunSpec, operating_point                # noqa: E402
from src.simulate import steady_state                                   # noqa: E402
from src.two_point import TwoPointController                            # noqa: E402

HERE = Path(__file__).resolve().parent.parent
CONFIG = "ndpr_plant"
HOURS = 150.0
HARD, EASY = (2, 3, 6, 9, 10, 14, 16, 17, 20), (1, 7, 12)
PAIRS_CSV = HERE / "runs" / f"two_point_pairs_{CONFIG}.csv"
OUT = HERE / "runs" / f"two_point_vs_mpc_{CONFIG}.csv"
RECOVERY_OK = 0.9495


def pick_pairs(n_top: int = 12) -> list:
    """The top pairs of notebook 22, with their relative gain at the design point."""
    rows = [r for r in csv.DictReader(open(HERE / "runs" / f"what_to_hold_{CONFIG}.csv"))
            if r["strategy"].startswith("two handles: scrub holds")]
    rows.sort(key=lambda r: (-int(r["cases_on_spec"]),
                             -min(float(r["worst_purity"] or 0) / 0.99, float(r["worst_recovery"] or 0) / 0.95)))
    op = operating_point(RunSpec(config=CONFIG, plant="matched"))
    out = []
    for r in rows[:n_top]:
        words = r["strategy"].replace(",", "").split()
        a, b = (words[4], int(words[6])), (words[9], int(words[11]))
        ctl = TwoPointController(op.model, op.x_model, op.u_model, op.rc.cut, a, b, 0.5,
                                 {"Q_scrub": (50, 2000), "Q_org": (600, 4700)}, {"Q_scrub": 10, "Q_org": 50}, 300.0)
        out.append({"a_kind": a[0], "a_stage": a[1], "b_kind": b[0], "b_stage": b[1],
                    "cases_on_spec": r["cases_on_spec"], "worst_purity": r["worst_purity"],
                    "worst_recovery": r["worst_recovery"], "relative_gain": ctl.relative_gain,
                    "g11": ctl.G[0, 0], "g12": ctl.G[0, 1], "g21": ctl.G[1, 0], "g22": ctl.G[1, 1]})
        print(f"  A {a}  B {b}  on spec {r['cases_on_spec']}/26  relative gain {ctl.relative_gain:+.2f}", flush=True)
    with open(PAIRS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    return out


def best_pair() -> tuple:
    """The best-ranked pair with a workable pairing: (scrub-A, organic-B) if its relative gain
    lambda is between 0.5 and 2, else the swap (organic-A, scrub-B) if 1 - lambda is.
    Returns (A, B, swap)."""
    for r in csv.DictReader(open(PAIRS_CSV)):
        lam = float(r["relative_gain"])
        a, b = (r["a_kind"], int(r["a_stage"])), (r["b_kind"], int(r["b_stage"]))
        if 0.5 <= lam <= 2.0:
            return a, b, False
        if 0.5 <= 1.0 - lam <= 2.0:
            return a, b, True
    raise RuntimeError("no pair with a workable pairing")


def run(seed: int, controller: str, settings: dict) -> dict:
    spec = RunSpec(config=CONFIG, plant="perturbed", controller=controller, upset="none", hours=HOURS,
                   run_file={"plant": {"seed": seed, "generator": "ratios"}}, controller_settings=settings)
    s = RunSession(spec)
    for _ in s.steps():
        pass
    r = s.result()
    plant, rc = s.op.plant, s.op.rc
    pur, rec = np.asarray(r.series["purity"], float), np.asarray(r.series["recovery"], float)
    on = (pur >= rc.targets()[0]) & (rec >= RECOVERY_OK)
    off = np.where(~on)[0]
    to_spec = 0.0 if len(off) == 0 else (np.nan if off[-1] == len(on) - 1 else float(r.t[off[-1] + 1]))
    u_end = np.asarray(r.u[-1], float)
    x_rest = steady_state(plant, u_end, x0=np.asarray(r.x[-1], float))
    o = plant.outlets(x_rest, u_end)
    heavy = np.array(heavy_mask(plant.p, rc.cut))
    c_feed = np.asarray(s.op.u_model.c_feed, float)
    cp = np.atleast_1d(o["c_prod"])
    lay = Inputs.layout(plant.n, plant.N)
    return {"purity_end": pur[-1], "recovery_end": rec[-1], "rest_purity": product_purity(cp, plant.p, rc.cut),
            "rest_recovery": float(o["Q_prod"]) * cp[heavy].sum() / (float(s.op.u_model.Q_feed) * c_feed[heavy].sum()),
            "Q_scrub_end": u_end[lay["Q_scrub"].start], "Q_org_end": u_end[lay["Q_org"].start],
            "hours_to_spec": to_spec, "solver_failures": r.scorecard.get("solver_failures", 0)}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pairs":
        pick_pairs()
        sys.exit(0)
    seeds = [int(a) for a in sys.argv[2:]] if len(sys.argv) > 2 and sys.argv[1] == "run" else list(HARD + EASY)
    a, b, swap = best_pair()
    print(f"two-point pair: {'organic' if swap else 'scrub'} holds {a}, {'scrub' if swap else 'organic'} holds {b}",
          flush=True)
    fields = ["seed", "group", "controller", "purity_end", "recovery_end", "rest_purity", "rest_recovery",
              "Q_scrub_end", "Q_org_end", "hours_to_spec", "solver_failures", "seconds"]
    done = set()
    if OUT.exists():
        done = {(int(r["seed"]), r["controller"]) for r in csv.DictReader(open(OUT))}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    for seed in seeds:
        for controller, settings in (("mpc", {}), ("two_point", {"hold_a": list(a), "hold_b": list(b), "swap": swap})):
            if (seed, controller) in done:
                continue
            t0 = time.time()
            row = run(seed, controller, settings)
            row.update(seed=seed, group="hard" if seed in HARD else "easy", controller=controller,
                       seconds=round(time.time() - t0))
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            print(f"seed {seed:2d} {controller:9s} end {100 * row['purity_end']:6.2f} / {100 * row['recovery_end']:6.2f} %"
                  f"   at rest {100 * row['rest_purity']:6.2f} / {100 * row['rest_recovery']:6.2f} %   scrub "
                  f"{row['Q_scrub_end']:5.0f} organic {row['Q_org_end']:5.0f}   to spec {row['hours_to_spec']} h"
                  f"  ({row['seconds']} s)", flush=True)
