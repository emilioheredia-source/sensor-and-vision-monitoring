"""Does what the report says hold on other perturbed plants, or only on the one it used?

The question. Every result in the report is on one perturbed plant, drawn from seed 1
(src/mismatch.py): one set of extraction constants, acid exponent, stage transfer rates,
densities and hold-up, all a little off the model. A different draw is a different plant. If a
conclusion only holds on seed 1, it isn't a conclusion.

The method. Draw many perturbed plants (seeds 1 to N_SEEDS) with their separation factors inside
the published ranges (generator "ratios", decision 31), commission and trim each one the
same way (operating_point.prepare, through runs.py), and run every upset of the catalogue
(src/scenarios.py) on each, uncontrolled and under the controller, 40 h, full state. These are new plants: seed 1 here is not the
report's plant (that one uses the independent generator and sits inside every published range).

Results are written row by row, so a stopped run picks up where it left off.

Run:  pixi run python notebooks/14_seed_spread.py [n_seeds]      (default 20)
Writes runs/seed_spread_ndpr_plant.csv, runs/seed_oracle_ndpr_plant.csv (each plant's best
recovery at spec) and figures/14_seed_spread_ndpr_plant.png
"""
import csv
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, style          # noqa: E402
from src.runs import RunSpec, execute                                    # noqa: E402

HERE = Path(__file__).resolve().parent.parent
# ---- 0. what to run: the command line ----
#   n_seeds              how many plants (default 20)
#   --config NAME        the run file (default ndpr_plant; ndpr_plant_sized for stages sized to the flows)
#   --hours H            length of each run (default 40; the sized plant is ~3.5x slower)
#   --upsets a,b,c       a subset of the catalogue (default: all eight); "none" runs the no-upset
#                        baseline, which the summary takes off every upset when it's there
#   --seeds A-B          run only these seeds, into their own table, and stop (to run batches in
#                        parallel); a final call without --seeds gathers every batch and summarizes
ARGS = sys.argv[1:]


def option(name: str, default):
    """The value after --name on the command line, or the default."""
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


N_SEEDS = int(ARGS[0]) if ARGS and not ARGS[0].startswith("--") else 20
CONFIG = option("--config", "ndpr_plant")
# --plant NAME: a registry plant other than "perturbed" (e.g. "book"); its tables get the plant in the name
PLANT = option("--plant", "perturbed")
TAG = CONFIG if PLANT == "perturbed" else f"{CONFIG}_{PLANT}"
HOURS = float(option("--hours", 40.0))
BATCH = option("--seeds", None)
SEED_RANGE = range(int(BATCH.split("-")[0]), int(BATCH.split("-")[1]) + 1) if BATCH else range(1, N_SEEDS + 1)
# The plants are drawn with separation factors inside their published ranges, widened 10 % each side
# (src/mismatch.py, generator "ratios", decision 31). The first version of this study used the
# independent generator, which drew five plants no published circuit has (notebook 17).
GENERATOR = "ratios"
UPSETS = tuple(option("--upsets", "feed_switch,feed_step,feed_acid,feed_interrupt,flowmeter_bias,"
                                  "interface_drift,reflux_acid,efficiency_loss").split(","))
# --controllers A,B: the uncontrolled reference first, the controller second (default open_loop,mpc;
# the main case runs open_loop_level_loops,mpc_level_loops)
CONTROLLERS = tuple(option("--controllers", "open_loop,mpc").split(","))
OPEN, CTRL = CONTROLLERS
KEYS = ("off_spec_mol", "lost_excess_mol", "interface_excursion_m_h", "reagent_mol", "time_to_settle_h",
        "solver_failures")
OUT = HERE / "runs" / (f"seed_spread_{TAG}.csv" if not BATCH else f"seed_spread_{TAG}_s{BATCH}.csv")
FIELDS = ["seed", "upset", "controller", *KEYS, "seconds"]


def _batch_tables() -> list:
    """This run file's batch tables, seed_spread_<config>_s<a>-<b>.csv, and nothing else (a plain
    "_s*" pattern also matched seed_spread_<config>_sized.csv, another run file's table)."""
    pattern = re.compile(rf"seed_spread_{re.escape(TAG)}_s\d+-\d+\.csv")
    return sorted(p for p in (HERE / "runs").glob(f"seed_spread_{TAG}_s*.csv") if pattern.fullmatch(p.name))


# ---- 1. what is already done (the table is written row by row) ----
# A run counts as done if it's in this table or in any batch table for the same run file, so a
# final call without --seeds only summarizes what the batches already ran.
ALL_TABLES = [HERE / "runs" / f"seed_spread_{TAG}.csv"] + _batch_tables()
done = {(int(r["seed"]), r["upset"], r["controller"]) for t in ALL_TABLES if t.exists() for r in csv.DictReader(open(t))}
if not OUT.exists():
    with open(OUT, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def record(row: dict) -> None:
    """Append one finished run to the table straight away, so nothing is lost if the script stops."""
    with open(OUT, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


# ---- 2. the runs: every seed, every upset, with and without the controller ----
# The seed goes in as an edit to the run file's [plant] section; everything else is ndpr_plant.
for seed in SEED_RANGE:
    for upset in UPSETS:
        for controller in CONTROLLERS:
            if (seed, upset, controller) in done:
                continue
            t0 = time.time()
            spec = RunSpec(config=CONFIG, plant=PLANT, controller=controller, upset=upset,
                           hours=HOURS, run_file={"plant": {"seed": seed, "generator": GENERATOR}})
            sc = execute(spec).scorecard
            row = {"seed": seed, "upset": upset, "controller": controller,
                   **{k: sc.get(k, 0) for k in KEYS}, "seconds": round(time.time() - t0, 1)}
            record(row)
            print(f"seed {seed:2d} {upset:16s} {controller:9s} off spec {row['off_spec_mol']:7.0f} mol, "
                  f"lost excess {row['lost_excess_mol']:7.0f} mol, interface {row['interface_excursion_m_h']:.2f} m·h "
                  f"({row['seconds']:.0f} s)", flush=True)


# ---- 3. can each plant make spec at all? ----
# The design optimizer, told the plant's true numbers, gives the best recovery that plant can reach
# at the design purity (99.5 %) on this circuit. A plant whose best is under the 95 % recovery
# target can't meet both targets with any handles; the controller is then asked the impossible.
if BATCH:
    print(f"batch {BATCH} done: {OUT}")
    sys.exit(0)
ORACLE = HERE / "runs" / f"seed_oracle_{TAG}.csv"


def best_recovery_at_spec(seed: int) -> float:
    """The best recovery the seed's perturbed plant can reach at 99.5 % purity (design optimizer)."""
    from src.config import load_config, nominal_inputs
    from src.design import optimize_operating_point, reflux_link
    from src.registry import PLANTS
    rc = load_config(CONFIG, overrides={"plant": {"seed": seed, "generator": GENERATOR}})
    plant, k_factor = PLANTS[PLANT].build(rc)
    design = rc.design
    free = {"Q_org": (design["organic_min_L_per_h"], design["pump_headroom_fraction"] * design["pump_max_L_per_h"]),
            "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0), "saponified_fraction": (0.0, 0.6)}
    u0 = nominal_inputs(rc, plant).replace(k_factor=k_factor)
    r = optimize_operating_point(plant, u0, rc.cut, rc.design_purity, free,
                                 link=reflux_link(design["net_product_L_per_h"]))
    return float(r.recovery) if r.success else float("nan")


oracle = {}
if ORACLE.exists():
    oracle = {int(r["seed"]): float(r["best_recovery_at_spec"]) for r in csv.DictReader(open(ORACLE))}
for seed in range(1, N_SEEDS + 1):
    if seed not in oracle:
        oracle[seed] = best_recovery_at_spec(seed)
        print(f"seed {seed:2d}: best recovery at spec {oracle[seed]:.3f}", flush=True)
with open(ORACLE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "best_recovery_at_spec"])
    writer.writerows(sorted(oracle.items()))
CAN = {s for s, r in oracle.items() if r >= 0.95}          # plants that can meet both targets


# ---- 4. the summary: median and range over seeds, for each upset ----
# every table for this run file: the main one and any batches (seed_spread_<config>_s<a>-<b>.csv)
TABLES = [OUT] + _batch_tables()
rows, seen = [], set()
for t in TABLES:
    for r in (csv.DictReader(open(t)) if t.exists() else []):
        key = (int(r["seed"]), r["upset"], r["controller"])
        if key in seen or int(r["seed"]) > N_SEEDS or r["upset"] not in UPSETS:
            continue                                   # one row per run, even if two tables have it
        seen.add(key)
        rows.append(r)


def values(upset: str, controller: str, key: str, seeds=None) -> np.ndarray:
    """One number per seed for this upset and controller (only the given seeds, if any)."""
    return np.array([float(r[key]) for r in rows if r["upset"] == upset and r["controller"] == controller
                     and (seeds is None or int(r["seed"]) in seeds)])


def cost(upset: str, controller: str, seeds=None) -> np.ndarray:
    """NdPr lost beyond nominal (a gain counts as zero) plus product made off spec, mol, per seed."""
    return (np.maximum(values(upset, controller, "lost_excess_mol", seeds), 0)
            + values(upset, controller, "off_spec_mol", seeds))


# The no-upset baseline. A plant whose trim (30 h) ends before it has settled keeps drifting through
# the run, toward or away from spec, whatever the upset: on seeds like that, every upset's cost is
# mostly that drift. The same 40 h with no upset, for each plant and controller, measures the drift;
# taking it off leaves what the upset itself cost.
BASE = {}
for t in TABLES:
    for r in (csv.DictReader(open(t)) if t.exists() else []):
        if r["upset"] == "none" and int(r["seed"]) <= N_SEEDS:
            BASE[(int(r["seed"]), r["controller"])] = (max(float(r["lost_excess_mol"]), 0) + float(r["off_spec_mol"]))
HAS_BASE = all((s, c) in BASE for s in range(1, N_SEEDS + 1) for c in CONTROLLERS)


def upset_cost(upset: str, controller: str, seeds) -> np.ndarray:
    """cost() less the plant's no-upset baseline for the same controller (what the upset itself cost)."""
    seeds = sorted(seeds)
    return cost(upset, controller, set(seeds)) - np.array([BASE[(s, controller)] for s in seeds])


CANNOT = sorted(set(range(1, N_SEEDS + 1)) - CAN)
TIE_MOL = 10.0                                     # chosen: closer than this, call it the same
print(f"\n{len(rows)} runs, seeds 1-{N_SEEDS}. {len(CAN)} plants can meet both targets; "
      f"{len(CANNOT)} can't (best recovery at spec under 95 %): seeds {CANNOT}")
for label, seeds in ((f"the {len(CAN)} plants that can meet both targets", CAN),
                     (f"the {len(CANNOT)} that can't", set(CANNOT))):
    if not seeds:
        continue
    print(f"On {label}. Lost beyond nominal + off spec, mol: median [min, max]")
    for upset in UPSETS:
        o, m = cost(upset, OPEN, seeds), cost(upset, CTRL, seeds)
        better = int(np.sum(m < o - 1.0))
        print(f"  {upset:16s} open loop {np.median(o):7.0f} [{o.min():.0f}, {o.max():.0f}]   controller "
              f"{np.median(m):7.0f} [{m.min():.0f}, {m.max():.0f}]   controller better on {better}/{len(o)}")
    if not HAS_BASE:
        continue
    ob = np.array([BASE[(s, OPEN)] for s in sorted(seeds)])
    mb = np.array([BASE[(s, CTRL)] for s in sorted(seeds)])
    print(f"  {'no upset':16s} open loop {np.median(ob):7.0f} [{ob.min():.0f}, {ob.max():.0f}]   controller "
          f"{np.median(mb):7.0f} [{mb.min():.0f}, {mb.max():.0f}]   (the drift after the trim)")
    print(f"  What the upset itself cost (run less the no-upset baseline), mol: median [min, max]; "
          f"better/same/worse = controller against open loop, 'same' within {TIE_MOL:.0f} mol")
    for upset in UPSETS:
        o, m = upset_cost(upset, OPEN, seeds), upset_cost(upset, CTRL, seeds)
        better, worse = int(np.sum(m < o - TIE_MOL)), int(np.sum(m > o + TIE_MOL))
        print(f"  {upset:16s} open loop {np.median(o):7.0f} [{o.min():.0f}, {o.max():.0f}]   controller "
              f"{np.median(m):7.0f} [{m.min():.0f}, {m.max():.0f}]   {better}/{len(o) - better - worse}/{worse}")


# ---- 5. the figure: one dot per plant, uncontrolled and controlled, for each upset ----
# Plants that can't meet both targets are drawn hollow: on them the controller is asked the impossible.
style()
fig, ax = plt.subplots(figsize=(9.0, 4.2))
rng = np.random.default_rng(0)                       # horizontal jitter only, so dots don't overlap
for i, upset in enumerate(UPSETS):
    for j, controller in enumerate(CONTROLLERS):
        xc = i + (j - 0.5) * 0.36
        for seeds, filled in ((CAN, True), (set(CANNOT), False)):
            if not seeds:                            # no plant in this group: no dots, no legend entry
                continue
            v = cost(upset, controller, seeds)
            x = xc + rng.uniform(-0.07, 0.07, len(v))
            label = None
            if i == 0:
                label = f"{'uncontrolled' if j == 0 else 'controller'}" + ("" if filled else ", plant can't make spec")
            ax.plot(x, v + 1, "o", color=SLOT[j], alpha=0.6, markersize=3.8, label=label,
                    markerfacecolor=SLOT[j] if filled else "none")
        ax.plot([xc - 0.12, xc + 0.12], [np.median(cost(upset, controller, CAN)) + 1] * 2, color=INK, lw=1.4)
ax.set_yscale("log")
ax.set_xticks(range(len(UPSETS)), [u.replace("_", " ") for u in UPSETS], rotation=20, ha="right")
ax.set_ylabel("NdPr lost beyond nominal + off spec, mol (+1)")
ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
ax.set_title(f"{CONFIG}: {N_SEEDS} perturbed plants, {HOURS:.0f} h per upset "
             f"(bar = median of the {len(CAN)} that can make spec)", loc="left", color=INK)
ax.text(0.0, -0.34, "Each dot is one perturbed plant (src/mismatch.py, generator 'ratios', seeds 1-%d), separation "
        "factors inside published ranges +-10 %%. Hollow: can't reach 95 %% recovery at spec with any handles." % N_SEEDS,
        transform=ax.transAxes, color=MUTED, fontsize=7.5)
fig.savefig(FIGURES / f"14_seed_spread_{TAG}.png")
print("wrote", FIGURES / f"14_seed_spread_{TAG}.png")
