"""Deploying the controller in steps: shadow, advisory, closed loop, with limits and a watchdog.

The question. What does each step of a careful deployment cost or buy, and do the guards get in the
way? src/deployment.py puts a guard between the controller and the plant: in shadow mode nothing
reaches the plant, in advisory mode an operator applies the advice every 2 h, in closed loop the
controller acts itself; in every mode a handle moves at most 2 % of its commissioned value per
interval and 20 % in all, and a watchdog holds the handles if the controller fails.

Three parts:
  1. Twenty plants (the main case's realistic plants), the feed switch and the stronger feed, 80 h:
     open loop over level loops, advisory, closed loop with the guard, closed loop without it
     (mpc_interlock). Shadow mode is open loop by construction; part 2 checks that.
  2. One plant, one run, the handover in steps: shadow for the first 24 h (through the feed
     switch at 5 h), advisory to 48 h, closed loop after. The journal is saved, every interval.
  3. The watchdog: the same plant in closed loop, with the controller made to crash from 20 h to
     23 h (a stand-in for a solver or data fault). With the guard, the handles are held and the
     controller re-armed at 30 h by hand; the level loops run throughout.

  4. (sized) The guarded and advisory modes again on the 20 plants, with the ceiling sized from the
     upset instead of 20 %: the unguarded controller took the organic flow up about 38 % on the feed
     switch, so the ceiling is set at 50 %.

Run:  pixi run python notebooks/36_deployment.py [fleet|handover|watchdog|sized]   (the first three if none given)
Writes runs/deployment_ndpr_main.csv, runs/deployment_handover_journal.csv,
runs/deployment_watchdog_journal.csv.
"""
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from src.runs import RunSpec, start                                     # noqa: E402

SEEDS = tuple(range(1, 21))
UPSETS = ("feed_switch", "feed_step")
CONTROLLERS = ("open_loop_level_loops", "mpc_advisory", "mpc_guarded", "mpc_interlock")
HOURS = 80.0
OUT = HERE / "runs" / "deployment_ndpr_main.csv"
FIELDS = ["seed", "upset", "controller", "off_spec_mol", "lost_excess_mol", "reagent_mol", "min_purity",
          "clipped_intervals", "alarms", "seconds"]


def spec(seed, upset, controller, hours=HOURS):
    return RunSpec(config="ndpr_main", plant="perturbed", controller=controller, upset=upset, hours=hours,
                   run_file={"plant": {"seed": seed, "generator": "ratios"}})


def one(args):
    seed, upset, controller = args
    t0 = time.time()
    session = start(spec(seed, upset, controller))
    for _ in session.steps():
        pass
    r = session.result()
    journal = getattr(session.controller, "journal", [])
    return {"seed": seed, "upset": upset, "controller": controller,
            "off_spec_mol": r.scorecard["off_spec_mol"], "lost_excess_mol": r.scorecard["lost_excess_mol"],
            "reagent_mol": r.scorecard["reagent_mol"], "min_purity": float(np.min(r.series["purity"])),
            "clipped_intervals": sum(1 for e in journal if e.get("clipped")),
            "alarms": sum(1 for e in journal if e.get("alarm")), "seconds": round(time.time() - t0)}


def fleet():
    done = set()
    if OUT.exists():
        done = {(int(r["seed"]), r["upset"], r["controller"]) for r in csv.DictReader(open(OUT))}
    else:
        with open(OUT, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    todo = [(s, u, c) for u in UPSETS for s in SEEDS for c in CONTROLLERS if (s, u, c) not in done]
    with Pool(8) as pool:
        for row in pool.imap_unordered(one, todo):
            with open(OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"seed {row['seed']:2d} {row['upset']:12s} {row['controller']:22s} lost {row['lost_excess_mol']:7.1f} "
                  f"off spec {row['off_spec_mol']:6.1f} clipped {row['clipped_intervals']}", flush=True)


SIZED_OUT = HERE / "runs" / "deployment_sized_ndpr_main.csv"
SIZED_LIMITS = {"guard_max_offset": 0.50}


def one_sized(args):
    seed, upset, controller = args
    t0 = time.time()
    s = spec(seed, upset, controller)
    s = s.__class__(**{**s.__dict__, "controller_settings": dict(SIZED_LIMITS)})
    session = start(s)
    for _ in session.steps():
        pass
    r = session.result()
    journal = session.controller.journal
    return {"seed": seed, "upset": upset, "controller": controller,
            "off_spec_mol": r.scorecard["off_spec_mol"], "lost_excess_mol": r.scorecard["lost_excess_mol"],
            "reagent_mol": r.scorecard["reagent_mol"], "min_purity": float(np.min(r.series["purity"])),
            "clipped_intervals": sum(1 for e in journal if e.get("clipped")),
            "alarms": sum(1 for e in journal if e.get("alarm")), "seconds": round(time.time() - t0)}


def sized():
    with open(SIZED_OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    todo = [(s, u, c) for u in UPSETS for s in SEEDS for c in ("mpc_advisory", "mpc_guarded")]
    with Pool(8) as pool:
        for row in pool.imap_unordered(one_sized, todo):
            with open(SIZED_OUT, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print(f"seed {row['seed']:2d} {row['upset']:12s} {row['controller']:14s} off spec {row['off_spec_mol']:7.1f} "
                  f"lost {row['lost_excess_mol']:7.1f} clipped {row['clipped_intervals']}", flush=True)


def write_journal(journal, path):
    """The guard's journal as a table: one row per interval (or event), handles flattened."""
    rows = []
    for e in journal:
        row = {"t": e["t"], "mode": e.get("mode", ""), "event": e.get("event", ""), "alarm": e.get("alarm", ""),
               "clipped": e.get("clipped", ""), "review": e.get("review", "")}
        for kind in ("proposed", "applied"):
            for name, value in (e.get(kind) or {}).items():
                row[f"{kind}_{name}"] = value
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def handover(seed=1):
    """Shadow to 24 h, advisory to 48 h, closed loop after; the journal saved."""
    session = start(spec(seed, "feed_switch", "mpc_shadow", hours=72.0))
    guard = session.controller
    for rec in session.steps():
        if rec.t >= 24.0 - 1e-9 and guard.mode == "shadow":
            guard.set_mode("advisory")
        if rec.t >= 48.0 - 1e-9 and guard.mode == "advisory":
            guard.set_mode("closed")
    r = session.result()
    write_journal(guard.journal, HERE / "runs" / "deployment_handover_journal.csv")
    p = r.series["purity"]
    t = r.t
    for a, b in ((0, 24), (24, 48), (48, 72)):
        m = (t >= a) & (t <= b)
        print(f"{a:2d}-{b:2d} h  purity {100 * p[m].min():.2f} to {100 * p[m].max():.2f} %")
    print("handover:", r.scorecard)


class Crashing:
    """Makes the wrapped controller raise between two times: a stand-in for a solver or data fault."""

    def __init__(self, guard, t0, t1):
        self.guard, self.t0, self.t1 = guard, t0, t1
        inner = guard.inner
        real_step = inner.step

        def step(obs):
            if self.t0 <= guard.t < self.t1:
                raise RuntimeError("injected fault")
            return real_step(obs)
        inner.step = step


def watchdog(seed=1):
    session = start(spec(seed, "feed_switch", "mpc_guarded", hours=60.0))
    guard = session.controller
    Crashing(guard, 20.0, 23.0)
    for rec in session.steps():
        if rec.t >= 30.0 - 1e-9 and guard.tripped:
            guard.rearm()
    r = session.result()
    write_journal(guard.journal, HERE / "runs" / "deployment_watchdog_journal.csv")
    alarms = [(e["t"], e["alarm"]) for e in guard.journal if e.get("alarm")]
    print("watchdog alarms:", alarms[:3], "...", len(alarms))
    print("watchdog run:", r.scorecard)


if __name__ == "__main__":
    parts = sys.argv[1:] or ["fleet", "handover", "watchdog"]
    for part in parts:
        {"fleet": fleet, "handover": handover, "watchdog": watchdog, "sized": sized}[part]()
