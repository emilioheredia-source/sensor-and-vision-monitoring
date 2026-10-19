"""Disturbance catalogue, open loop: what each realistic upset does to product
purity and to the settler interfaces when nobody touches anything.

The question: before any controller, how bad is each upset? These are the 60 h
numbers in REPORT.md section 4. The controller scripts (05, 05b) run their own
open-loop cases over 40 h, so their numbers are smaller.

Method: commission the plant the run file names (the perturbed plant for the
ndpr_* files), trim it onto spec with the controller the way a start-up crew
would, then leave the handles alone. Apply one upset at t = 5 h, simulate
60 h, and score the run on transition cost (src/metrics.py): product made off
spec, product metal lost to the raffinate beyond what was lost before, time to
settle, how far the interfaces left their window, reagent used.

Sources per scenario: Rollat et al. 2024 (feed switch, flowmeter bias, interface drift);
Lyon 2016 (feed strength and acidity, flow upsets, reflux acid, stage efficiency).
src/scenarios.py has the same eight upsets, each explained.

Run:  pixi run python notebooks/03_disturbances.py [config_name]
Writes figures/03_<config>_<scenario>.png, one per upset, and runs/disturbances_<config>.csv
with the transition-cost numbers.
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
from src.mpc import CircuitMPC, MPCSettings                             # noqa: E402
from src.plotting import COLOR, FIGURES, GRID, INK, INK2, MUTED, SLOT, style   # noqa: E402
from src.simulate import simulate, steady_state                         # noqa: E402

# ---- 1. settings ----
HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
RUNS.mkdir(exist_ok=True)

name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_plant"
rc = load_config(name)
p = rc.params
SPEC, REC = rc.targets()                       # purity spec and recovery target, fractions
geo = p.raw["stage_geometry"]
H = geo["settler_height_m"]                    # settler height, m
Z_LO, Z_HI = (f * H for f in geo["interface_window_fraction"])   # the interface's allowed band, m
T_END, DT, T0 = 60.0, 0.25, 5.0                # run length, output step, upset time: all in hours

# ---- 2. commission the plant and trim it onto spec ----
# Two circuits: model is what the controller believes, circ is the plant it drives (they differ
# when the run file's [plant] says "perturbed"). kf holds the plant's per-stage transfer-rate
# multipliers; the model's are all ones.
# The plant starts on the model's design handles (u_design). On a plant that isn't the model, that
# lands off the design point, so the controller runs once with full information until the product
# is on spec (trim_under_control). Where its handles end up is the plant's operating point, and
# every upset below starts from there. ARCHITECTURE.md decision 21 has why.
model = build_circuit(rc)                                    # the controller's model, for the commissioning trim
circ, kf = build_plant(rc)                                   # the plant the run file's [plant] mismatch names
ones = np.ones(circ.N)
x_rest, u_design = commission(circ, nominal_inputs(rc, circ).replace(k_factor=kf))   # start-up on the design handles
x_ss = steady_state(model, u_design.replace(k_factor=ones), x0=x_rest)
trim = CircuitMPC(model, x_ss, u_design.replace(k_factor=ones), rc.cut, SPEC, 0.995, REC,
                  MPCSettings(strip_tied_to_scrub=300.0 if rc["scrub"]["metal"] == "reflux" else None))
x0, u_ctl = trim_under_control(circ, x_rest, kf, trim)       # commissioned: the real plant on spec, then left alone
u = u_ctl.replace(k_factor=kf)                               # the trimmed handles, with the plant's own rates
n_ext = rc["circuit"]["n_ext"]
mid = circ.N // 2                                            # the settler the interface-drift upset pushes

# ---- 3. the eight upsets ----
# Each entry: the figure title, and the upset itself (src/disturbances.py). All start at T0.
CATALOGUE = {
    "feed_switch":      ("Feed switch Eneabba → Yangibana over 1 h (Carester)",
                         FeedSwitch(t=T0, ramp_h=1.0, to="yangibana", params=p, removed_upstream=tuple(rc["feed"]["removed_upstream"]))),
    "feed_step":        ("Feed strength +20 % (Idaho: day-to-day variation)",
                         FeedStep(t=T0, factor=1.2)),
    "feed_acid":        ("Feed acidity 0.05 → 0.3 mol/L (Idaho: pH sensitivity)",
                         FeedAcid(t=T0, value=0.3)),
    "feed_interrupt":   ("Feed pump stops for 2 h (Idaho: flow upsets)",
                         FlowStep(t=T0, stream="feed", factor=0.0, duration_h=2.0)),
    "flowmeter_bias":   ("Organic flowmeter reads 5 % low: plant gets +5 % (Carester)",
                         FlowmeterBias(t=T0, stream="organic", bias=0.05)),
    "interface_drift":  (f"Hydraulic drift on settler {mid}: +10 cm over 5 h (Carester)",
                         InterfaceDrift(t=T0, ramp_h=5.0, stages=(mid,), delta_m=0.10)),
    "reflux_acid":      ("Scrub liquor acid ×1.5: scrub starts stripping (Idaho 4.1.3)",
                         RefluxAcid(t=T0, value=min(1.5 * u.h_scrub, 4.0))),
    "efficiency_loss":  (f"Mixers at stages {n_ext-1}, {n_ext} run at 30 % (Idaho: stage efficiency)",
                         EfficiencyLoss(t=T0, stages=(n_ext - 1, n_ext), factor=0.3)),
}


# ---- 4. a helper for the lower panel of each figure ----
def interface_figure(ax, t, z, drift_stage=None):
    """Interface positions: band of all stages, plus the one that was pushed if any.

    ax           the matplotlib axis to draw on.
    t            times, h.
    z            interface heights, m: one row per time, one column per stage.
    drift_stage  the stage the drift upset pushes, drawn on its own; None for other upsets.

    Draws the band from the lowest to the highest interface, their mean, and the
    allowed window as dotted lines.
    """
    ax.fill_between(t, z.min(axis=1), z.max(axis=1), color=GRID, lw=0, label="all stages")
    ax.plot(t, z.mean(axis=1), color=INK2, lw=1.0)
    if drift_stage is not None:
        ax.plot(t, z[:, drift_stage], color=SLOT[1])
        ax.annotate(f"stage {drift_stage}", (t[-1], z[-1, drift_stage]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=8)
    for y, lab in ((Z_LO, "window"), (Z_HI, "")):
        ax.axhline(y, color=MUTED, lw=0.8, ls=":")
    ax.text(t[0], Z_HI, " interface window (chosen)", va="bottom", color=MUTED, fontsize=8)
    ax.set_ylabel("interface, m")
    ax.set_ylim(0, H)


# ---- 5. run every upset open loop, score it, draw it ----
# For each upset: simulate the plant on the fixed handles with the upset applied, score the run,
# pull every stage's interface height out of the state (unpack()[3] is the interfaces), and keep a
# row for the CSV. Then one figure: purity on top, interfaces below, a dashed line at the upset.
rows = []
style()
for key, (title, dist) in CATALOGUE.items():
    sc = Scenario([dist])
    t0 = time.time()
    traj = simulate(circ, x0, sc.u_of_t(u), T_END, dt_out=DT, breakpoints=sc.breakpoints)
    cost = transition_cost(circ, traj, rc.cut, SPEC)
    purity = cost.pop("purity_series")                       # product purity over time, mol fraction
    z = np.array([np.asarray(circ.unpack(xk)[3], float) for xk in traj.x])
    rows.append(dict(scenario=key, **{k: round(v, 4) for k, v in cost.items()},
                     purity_min=round(float(purity.min()), 5), purity_final=round(float(purity[-1]), 5),
                     interface_min_m=round(float(z.min()), 4), interface_max_m=round(float(z.max()), 4)))
    print(f"{key:16s} {time.time()-t0:5.1f}s  purity min {purity.min():.4f} final {purity[-1]:.4f}  "
          f"off-spec {cost['off_spec_mol']:.0f} mol  settle {cost['time_to_settle_h']:.1f} h  "
          f"interface {z.min():.3f}-{z.max():.3f} m", flush=True)

    # the figure: purity against the spec line on top, interfaces below
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"hspace": 0.3})
    axes[0].plot(traj.t, purity, color=COLOR["Nd"])
    axes[0].axhline(SPEC, color=MUTED, lw=0.8, ls=":")
    axes[0].text(traj.t[0], SPEC, f" spec {SPEC:.0%}", va="top", color=MUTED, fontsize=8)
    axes[0].set_ylabel(f"{'Nd' if rc.cut == 'Pr' else 'NdPr'} in product, mol fraction")
    lo = min(float(purity.min()), SPEC) - 0.01
    axes[0].set_ylim(max(0.0, lo), 1.0005)
    axes[0].set_title(title, loc="left", color=INK)
    interface_figure(axes[1], traj.t, z, drift_stage=mid if key == "interface_drift" else None)
    for ax in axes:
        ax.axvline(T0, color=MUTED, lw=0.8, ls="--")
    axes[1].set_xlabel("time, h")
    fig.savefig(FIGURES / f"03_{name}_{key}.png")
    plt.close(fig)

# ---- 6. the table ----
with open(RUNS / f"disturbances_{name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("wrote", RUNS / f"disturbances_{name}.csv", f"and figures/03_{name}_*.png")
