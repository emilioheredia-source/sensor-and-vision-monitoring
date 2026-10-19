"""One disturbance, run open loop and under MPC, scored on transition cost.

The question: on one upset, what does the controller do that leaving the
handles alone doesn't, and what does that cost in reagent?

Method: commission the plant the run file names and trim it onto spec (the
same start as every study, ARCHITECTURE.md decision 21). Apply one upset at
t = 5 h. Run it twice from that same start: once with the handles left alone,
once with the MPC moving them every half hour. Score both on transition cost
(src/metrics.py) and draw them on one figure.

Run:  pixi run python notebooks/05_mpc.py [config_name] [scenario] [t_end]
      scenario is one of the upset keys of src/scenarios.py (default feed_switch).
      config_name defaults to ndpr_small, t_end to 40 h. The report's figures use ndpr_plant.
Writes figures/05_mpc_<config>_<scenario>.png and prints both scores.
"""
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
from src.plotting import COLOR, FIGURES, INK, INK2, MUTED, SLOT, style   # noqa: E402
from src.simulate import simulate, steady_state                         # noqa: E402

# ---- 1. the command line and the run file ----
name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_small"
key = sys.argv[2] if len(sys.argv) > 2 else "feed_switch"
T_END = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0     # run length, h
T0 = 5.0                                                       # when the upset starts, h

rc = load_config(name)
p = rc.params
SPEC, REC = rc.targets()                       # purity spec and recovery target, fractions
geo = p.raw["stage_geometry"]
H = geo["settler_height_m"]                    # settler height, m
Z_LO, Z_HI = (f * H for f in geo["interface_window_fraction"])   # the interface's allowed band, m

# ---- 2. the model, the plant, and the start-up ----
# circ is the controller's model, plant is what it drives, kf the plant's per-stage transfer-rate
# multipliers (the controller assumes ones). In reflux mode the controller keeps the strip flow tied
# to the scrub flow, 300 L/h of net product.
# Start-up: settle the plant on the model's design handles, find the model's own rest point there,
# then let a controller trim the plant onto spec with full information. x0 and u are where every
# run below starts.
PURITY_TARGET = 0.995                                        # the design point; spec is SPEC
settings = MPCSettings(strip_tied_to_scrub=300.0 if rc["scrub"]["metal"] == "reflux" else None)
circ = build_circuit(rc)                                     # the controller's model
plant, kf = build_plant(rc)                                  # what it drives (the run file's [plant] mismatch)
ones = np.ones(circ.N)                                       # the controller does not know the plant's efficiencies
x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))   # start-up on the design handles
x_ss = steady_state(circ, u_design.replace(k_factor=ones), x0=x_rest)               # the model's own rest point there
trim = CircuitMPC(circ, x_ss, u_design.replace(k_factor=ones), rc.cut, SPEC, PURITY_TARGET, REC, settings)
x0, u_ctl = trim_under_control(plant, x_rest, kf, trim)      # the controller puts the real plant on spec; its handles
u = u_ctl.replace(k_factor=kf)                               # are the operating point; u is the plant's view of it
x_ss = steady_state(circ, u_ctl, x0=x0)                      # the model's rest point under the trimmed handles
n_ext, mid = rc["circuit"]["n_ext"], circ.N // 2

# ---- 3. the upset ----
# The same eight upsets as notebook 03 (src/scenarios.py explains each). Only the one named on the
# command line is run.
CATALOGUE = {
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
sc = Scenario([CATALOGUE[key]])

# ---- 4. open loop: the handles left where the trim put them ----
t0 = time.time()
open_loop = simulate(plant, x0, sc.u_of_t(u), T_END, dt_out=0.25, breakpoints=sc.breakpoints)
cost_ol = transition_cost(plant, open_loop, rc.cut, SPEC)
print(f"open loop: {time.time()-t0:.1f}s", {k: round(v, 3) for k, v in cost_ol.items() if k != "purity_series"})

# ---- 5. closed loop: the MPC moves the handles every half hour ----
# A fresh controller, started from the model's rest point under the trimmed handles.
# run_closed_loop returns the plant's trajectory and a log of the handles at each step.
t0 = time.time()
ctl = CircuitMPC(circ, x_ss, u_ctl, rc.cut, SPEC, PURITY_TARGET, REC, settings)
print(f"controller built in {time.time()-t0:.1f}s; horizon {settings.horizon} x {settings.t_step} h, "
      f"{settings.control_horizon} moves")
t0 = time.time()
closed, handles = run_closed_loop(plant, x0, ctl, sc, T_END, plant_k_factor=kf)
cost_cl = transition_cost(plant, closed, rc.cut, SPEC)
print(f"MPC: {time.time()-t0:.0f}s", {k: round(v, 3) for k, v in cost_cl.items() if k != "purity_series"})

# ---- 6. the figure: purity, interfaces, handles ----
style()
fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True,
                         gridspec_kw={"hspace": 0.32, "height_ratios": [1, 0.8, 1]})

# top panel: product purity, open loop in grey and MPC in colour, against the spec line
ax = axes[0]
PRODUCT = "Nd" if rc.cut == "Pr" else "NdPr"
ax.plot(open_loop.t, cost_ol["purity_series"], color=MUTED)
ax.plot(closed.t, cost_cl["purity_series"], color=COLOR["Nd"])
# end labels staggered so they stay apart when the two lines meet
y_ol, y_cl = cost_ol["purity_series"][-1], cost_cl["purity_series"][-1]
up, down = ((("open loop", MUTED, y_ol), ("MPC", COLOR["Nd"], y_cl)) if y_ol >= y_cl
            else (("MPC", COLOR["Nd"], y_cl), ("open loop", MUTED, y_ol)))
for (lab, col, yv), dy in ((up, 6), (down, -6)):
    ax.plot([closed.t[-1]], [yv], "o", color=col, markersize=4)
    ax.annotate(lab, (closed.t[-1], yv), xytext=(5, dy), textcoords="offset points", va="center", ha="left",
                color=INK2, fontsize=8)
ax.axhline(SPEC, color=MUTED, lw=0.8, ls=":")
ax.text(0, SPEC, f" spec {SPEC:.0%}", va="top", color=MUTED, fontsize=8)
lo = min(float(cost_ol["purity_series"].min()), float(cost_cl["purity_series"].min()), SPEC) - 0.01
ax.set_ylim(max(0, lo), 1.0005)
ax.set_ylabel(f"{PRODUCT} in product")
ax.set_title(f"{name}, {key}: open loop vs MPC", loc="left", color=INK)

# middle panel: every stage's interface, as a band from lowest to highest with the mean as a line
ax = axes[1]
for traj, color, lab in ((open_loop, MUTED, "open loop"), (closed, COLOR["Nd"], "MPC")):
    z = np.array([np.asarray(plant.unpack(xk)[3], float) for xk in traj.x])
    ax.fill_between(traj.t, z.min(axis=1), z.max(axis=1), color=color, alpha=0.25, lw=0)
    ax.plot(traj.t, z.mean(axis=1), color=color, lw=1.0)
for y in (Z_LO, Z_HI):
    ax.axhline(y, color=MUTED, lw=0.8, ls=":")
ax.text(0, Z_HI, " interface window", va="bottom", color=MUTED, fontsize=8)
ax.set_ylabel("interfaces, m (all stages)")
ax.set_ylim(0, H)

# bottom panel: three of the controller's handles, each divided by its trimmed value (1 = unchanged)
ax = axes[2]
th = np.array([h["t"] for h in handles])
series = []
for k, (hname, lab) in enumerate((("Q_org", "organic flow"), ("Q_scrub", "scrub flow"),
                                  ("saponified_fraction", "saponification"))):
    y = np.array([h[hname] for h in handles]) / getattr(u, hname)
    ax.step(th, y, where="post", color=SLOT[k])
    series.append((y[-1], lab, SLOT[k]))
# end labels in order of their final value, pushed apart when two end close together
series.sort(key=lambda t: t[0])
ymin, ymax = ax.get_ylim()
gap = 0.045 * (ymax - ymin)
placed = []
for yv, lab, col in series:
    yl = yv if not placed else max(yv, placed[-1] + gap)
    placed.append(yl)
    ax.plot([th[-1]], [yv], "o", color=col, markersize=4)
    if yl == yv:
        ax.annotate(lab, (th[-1], yv), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    color=INK2, fontsize=8, xycoords="data")
    else:
        ax.annotate(lab, (th[-1], yl), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                    color=INK2, fontsize=8)
ax.set_ylabel("handle / nominal")
ax.set_xlabel("time, h")
for a in axes:
    a.axvline(T0, color=MUTED, lw=0.8, ls="--")
fig.savefig(FIGURES / f"05_mpc_{name}_{key}.png")
print("wrote", FIGURES / f"05_mpc_{name}_{key}.png")
