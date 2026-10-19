"""The whole disturbance catalogue, open loop and under MPC, scored on transition
cost. One table: what the controller buys on each upset.

The question: across all eight upsets, where does the controller help, where
does it do nothing, and where does it make things worse?

Method: commission the plant and trim it onto spec, once (ARCHITECTURE.md
decision 21). Record what the trim moved and what it costs in reagent. Then,
for each upset: run it open loop and under a fresh MPC from the same start,
score both on transition cost (src/metrics.py), and keep one row. The figure
sets the two side by side for the two costs that moved most: product metal
lost beyond nominal, and interface time outside the window.

Run:  pixi run python notebooks/05b_mpc_catalogue.py [config_name] [t_end] [--plot-only]
      config_name defaults to ndpr_plant, t_end to 40 h.
Writes runs/mpc_catalogue_<config>.csv, runs/commissioning_<config>.csv (the trim) and
figures/05b_mpc_catalogue_<config>.png. --plot-only redraws the figure from the CSV.
"""
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.commission import commission, trim_under_control               # noqa: E402
from src.config import build_circuit, build_plant, load_config, nominal_inputs   # noqa: E402
from src.disturbances import (EfficiencyLoss, FeedAcid, FeedStep, FeedSwitch, FlowStep,   # noqa: E402
                              FlowmeterBias, InterfaceDrift, RefluxAcid, Scenario)
from src.metrics import transition_cost                                 # noqa: E402
from src.mpc import CircuitMPC, MPCSettings, run_closed_loop            # noqa: E402
from src.plotting import FIGURES, INK, MUTED, SLOT, style         # noqa: E402
from src.simulate import simulate, steady_state                         # noqa: E402

# ---- 1. the command line and the run file ----
HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
PLOT_ONLY = "--plot-only" in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
name = args[0] if args else "ndpr_plant"
T_END = float(args[1]) if len(args) > 1 else 40.0      # run length, h
T0 = 5.0                                               # when each upset starts, h

rc = load_config(name)
p = rc.params
SPEC, REC = rc.targets()                               # purity spec and recovery target, fractions
PURITY_TARGET = 0.995                                  # the design point the controller aims for
# in reflux mode the controller keeps the strip flow tied to the scrub flow, 300 L/h of net product
settings = MPCSettings(strip_tied_to_scrub=300.0 if rc["scrub"]["metal"] == "reflux" else None)
circ = build_circuit(rc)                               # the controller's model
PRODUCT = "Nd" if rc.cut == "Pr" else "NdPr"


def reagent(uu):
    """Reagent fed per hour at inputs uu, mol/h: caustic for saponification plus strip and scrub acid."""
    return (uu.saponified_fraction * p.raw["extractant"]["concentration_monomer"] * uu.Q_org
            + uu.Q_strip * uu.h_strip + uu.Q_scrub * uu.h_scrub)


# ---- 2. commission and trim, or read the old results back ----
# With --plot-only nothing is simulated: the rows come from the CSV and the script skips to the figure.
# Otherwise: start-up on the model's design handles, then the controller trims the plant onto spec
# with full information. kf holds the plant's per-stage transfer-rate multipliers; the controller
# assumes ones.
if PLOT_ONLY:
    rows = []
    with open(RUNS / f"mpc_catalogue_{name}.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (v if k == "scenario" else float(v)) for k, v in r.items()})
else:
    t0 = time.time()
    plant, kf = build_plant(rc)
    ones = np.ones(circ.N)
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))   # start-up on the design handles
    x_ss = steady_state(circ, u_design.replace(k_factor=ones), x0=x_rest)               # the model's own rest point there
    trim = CircuitMPC(circ, x_ss, u_design.replace(k_factor=ones), rc.cut, SPEC, PURITY_TARGET, REC, settings)
    x0, u_ctl = trim_under_control(plant, x_rest, kf, trim)  # the controller puts the real plant on spec
    u = u_ctl.replace(k_factor=kf)                           # the plant's view of the trimmed inputs
    x_ss = steady_state(circ, u_ctl, x0=x0)                  # the model's rest point under the trimmed handles
    print(f"commissioned and trimmed in {time.time()-t0:.0f}s", flush=True)
    # what the trim moved and what it costs: the model's design point against the plant's operating point.
    # The difference in reagent is what the model's error costs (README, "The operating point").
    with open(RUNS / f"commissioning_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["handle", "design", "trimmed"])
        for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction"):
            w.writerow([h, round(float(getattr(u_design, h)), 4), round(float(getattr(u_ctl, h)), 4)])
        w.writerow(["reagent_mol_per_h", round(reagent(u_design)), round(reagent(u_ctl))])
    print(f"trim: reagent {reagent(u_design):.0f} -> {reagent(u_ctl):.0f} mol/h; "
          + ", ".join(f"{h} {getattr(u_design, h):.3g} -> {getattr(u_ctl, h):.3g}"
                      for h in ("Q_org", "Q_scrub", "h_scrub", "saponified_fraction")),
          flush=True)
n_ext, mid = rc["circuit"]["n_ext"], circ.N // 2

# ---- 3. the eight upsets ----
# The same catalogue as notebooks 03 and 05; src/scenarios.py explains each. All start at T0.
CATALOGUE = {} if PLOT_ONLY else {
    "feed_switch": FeedSwitch(t=T0, ramp_h=1.0, to="yangibana", params=p,
                              removed_upstream=tuple(rc["feed"]["removed_upstream"])),
    "feed_step": FeedStep(t=T0, factor=1.2),
    "feed_acid": FeedAcid(t=T0, value=0.3),
    "feed_interrupt": FlowStep(t=T0, stream="feed", factor=0.0, duration_h=2.0),
    "flowmeter_bias": FlowmeterBias(t=T0, stream="organic", bias=0.05),
    "interface_drift": InterfaceDrift(t=T0, ramp_h=5.0, stages=(mid,), delta_m=0.10),
    "reflux_acid": RefluxAcid(t=T0, value=min(1.5 * u.h_scrub, 4.0)),
    "efficiency_loss": EfficiencyLoss(t=T0, stages=(n_ext - 1, n_ext), factor=0.3),
}

# ---- 4. each upset, open loop and under MPC ----
# KEYS are the transition costs kept for each run: mol of product off spec, mol of product metal lost
# beyond nominal, hours to settle, metre-hours of interface outside the window, mol of reagent.
# Each row holds every cost twice, with _open and _mpc on the end, plus the lowest purity of each run.
KEYS = ("off_spec_mol", "lost_excess_mol", "time_to_settle_h", "interface_excursion_m_h", "reagent_mol")
if not PLOT_ONLY:
    rows = []
for key, dist in ([] if PLOT_ONLY else CATALOGUE.items()):
    sc = Scenario([dist])
    t0 = time.time()
    ol = transition_cost(plant, simulate(plant, x0, sc.u_of_t(u), T_END, dt_out=0.25, breakpoints=sc.breakpoints),
                         rc.cut, SPEC)
    ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, SPEC, PURITY_TARGET, REC, settings)   # fresh controller state
    cl = transition_cost(plant, run_closed_loop(plant, x0, ctl, sc, T_END, plant_k_factor=kf)[0], rc.cut, SPEC)
    row = {"scenario": key, "purity_min_open": float(ol["purity_series"].min()),
           "purity_min_mpc": float(cl["purity_series"].min())}
    for k in KEYS:
        row[k + "_open"], row[k + "_mpc"] = ol[k], cl[k]
    rows.append(row)
    print(f"{key:16s} {time.time()-t0:5.0f}s | purity min {row['purity_min_open']:.4f} -> {row['purity_min_mpc']:.4f} | "
          f"lost excess {ol['lost_excess_mol']:7.0f} -> {cl['lost_excess_mol']:7.0f} mol | "
          f"interface {ol['interface_excursion_m_h']:.3f} -> {cl['interface_excursion_m_h']:.3f} m·h | "
          f"reagent {ol['reagent_mol']:.0f} -> {cl['reagent_mol']:.0f}", flush=True)

# ---- 5. save the table ----
if not PLOT_ONLY:
    with open(RUNS / f"mpc_catalogue_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ---- 6. the figure: paired bars, open loop vs MPC, for the two costs that moved most ----
# One horizontal bar pair per upset: grey for open loop, colour for MPC.
style()
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"wspace": 0.35})
labels = [r["scenario"].replace("_", " ") for r in rows]
y = np.arange(len(rows))
for ax, k, title in ((axes[0], "lost_excess_mol", f"{PRODUCT} lost beyond nominal, mol"),
                     (axes[1], "interface_excursion_m_h", "interface outside window, m·h")):
    ax.barh(y - 0.18, [r[k + "_open"] for r in rows], height=0.34, color=MUTED, label="open loop")
    ax.barh(y + 0.18, [r[k + "_mpc"] for r in rows], height=0.34, color=SLOT[2], label="MPC")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", color=INK)
    ax.grid(axis="y", visible=False)
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False)
fig.suptitle(f"{name}: what the controller buys on each upset ({T_END:.0f} h runs)", x=0.01, ha="left", color=INK)
fig.savefig(FIGURES / f"05b_mpc_catalogue_{name}.png")
print("wrote", FIGURES / f"05b_mpc_catalogue_{name}.png")
if not PLOT_ONLY:
    print("wrote", RUNS / f"mpc_catalogue_{name}.csv")
