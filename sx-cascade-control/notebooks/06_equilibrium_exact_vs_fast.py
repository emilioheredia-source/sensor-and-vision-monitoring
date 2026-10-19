"""Exact equilibrium stages against the fast (rate) model.

The question: the model's stages move metal toward equilibrium at a rate and
never quite get there. How different would the answers be with stages that
reach equilibrium exactly, and what would that cost in solve time?

In the fast model each stage gets a fraction k*tau/(1+k*tau) of the way to
equilibrium (tau = mixer residence time). The nominal k was set to give 95 %
at the Idaho lab's "three minutes to equilibrium"; the run files put more flow
through the mixer than that, so their stages get less of the way (the NdPr run files about 81 to
89 %, ARCHITECTURE.md, the stage equations). An exact equilibrium stage is
the k -> infinity limit of the same equations: c_org = D c_aq holds exactly.
So the exact model is reached here by raising k, and the question becomes
how far the nominal circuit sits from it, and what it costs in solve time.

Measured on the run file's circuit (default ndpr_small) at its designed operating point:
  - steady-state profiles and product purity / recovery at k x 1, 10, 100, 1000
  - time to solve the steady state at each
  - the operating point the optimizer picks at k x 1 and k x 1000

Run:  pixi run python notebooks/06_equilibrium_exact_vs_fast.py [config_name]
Writes figures/06_exact_vs_fast_<config>.png
"""
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Circuit, build_map                       # noqa: E402
from src.config import load_config, nominal_inputs               # noqa: E402
from src.design import optimize_operating_point, reflux_link     # noqa: E402
from src.metrics import summarize_steady                         # noqa: E402
from src.plotting import COLOR, FIGURES, INK, INK2, section_bands, style   # noqa: E402
from src.simulate import steady_state                            # noqa: E402

# ---- 1. the run file ----
# p0 is the nominal parameter set; FACTORS are the multipliers on the transfer rate k to try.
name = sys.argv[1] if len(sys.argv) > 1 else "ndpr_small"
rc = load_config(name)
p0 = rc.params
c = rc["circuit"]
FACTORS = [1, 10, 100, 1000]
SPEC, REC = rc.targets()

# ---- 2. the same circuit at four transfer rates ----
# For each factor: a copy of the parameters with k raised, the circuit built on it, and its steady
# state on the run file's inputs, timed. results[f] keeps the state, circuit, inputs, scores, solve
# time and a rough stage efficiency.
# The stage efficiency is k*tau/(1+k*tau) per stage, with tau the mixer volume over the total flow
# through that stage's mixer (aqueous plus organic), in hours; it is kept as the range over stages.
results = {}
for f in FACTORS:
    p = replace(p0, k_transfer=p0.k_transfer * f)
    circ = Circuit(build_map(c["n_ext"], c["n_scrub"], c["n_strip"]), p,
                   scrub_reflux=(rc["scrub"]["metal"] == "reflux"))
    u = nominal_inputs(rc, circ)
    t0 = time.time()
    x = steady_state(circ, u)
    dt = time.time() - t0
    s = summarize_steady(circ, x, u, rc.cut)
    q_total = np.asarray(circ.aqueous_flows(u), float).ravel() + float(u.Q_org)   # L/h through each mixer
    tau = p0.raw["stage_geometry"]["mixer_volume_L"] / q_total                    # h, per stage
    eff_stages = p.k_transfer * tau / (1 + p.k_transfer * tau)
    eff = (float(eff_stages.min()), float(eff_stages.max()))                      # lowest and highest stage
    results[f] = dict(x=x, circ=circ, u=u, summary=s, solve_s=dt, efficiency=eff)
    print(f"k x {f:5d}: stage efficiency {eff[0]:.4f}-{eff[1]:.4f}  purity {s['product_purity']:.4f}  "
          f"recovery {s['recovery']:.4f}  steady state in {dt:.2f}s", flush=True)

# ---- 3. the operating point the optimizer picks, fast vs exact ----
# The most recovery at 99.5 % purity, over the same four handles as the design step (notebook 02),
# started from each circuit's own steady state. If the two answers are close, the rate model is
# good enough for design.
FREE = {"Q_org": (600.0, 4000.0), "Q_scrub": (50.0, 2000.0), "h_scrub": (0.05, 2.0),
        "saponified_fraction": (0.0, 0.6)}
for f in (1, 1000):
    r = results[f]
    t0 = time.time()
    link = reflux_link(300.0) if rc["scrub"]["metal"] == "reflux" else None   # strip = 300 L/h net + scrub
    d = optimize_operating_point(r["circ"], r["u"], rc.cut, 0.995, FREE, link=link, x_guess=r["x"])
    print(f"design at k x {f}: max recovery {d.recovery:.4f} at purity {d.purity:.4f} | "
          f"O/A {d.u.Q_org / float(r['u'].Q_feed):.2f} "
          f"Q_scrub {d.u.Q_scrub:.0f} h_scrub {d.u.h_scrub:.2f} sapon {d.u.saponified_fraction:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

# ---- 4. the figure ----
# Organic concentration of every component the feed carries, stage by stage: solid lines for the
# nominal rate, dashed for k x 1000. Each component is labelled once, at its peak on the solid line.
style()
fig, ax = plt.subplots(figsize=(7.2, 3.8))
stages = np.arange(results[1]["circ"].N)
for f, ls in ((1, "-"), (1000, "--")):
    circ, x = results[f]["circ"], results[f]["x"]
    _, c_org, _, _ = circ.unpack(x)
    c_org = np.array([np.asarray(v, float).ravel() for v in c_org])
    present = [c for c in p0.names if float(results[1]["u"].c_feed[p0.index(c)]) > 0]
    for comp in present:
        y = c_org[:, p0.index(comp)]
        ax.plot(stages, y, ls, color=COLOR[comp])
        if f == 1:
            k = int(np.argmax(y))
            ax.annotate(comp, (stages[k], y[k]), xytext=(0, 5), textcoords="offset points", ha="center",
                        color=INK2, fontsize=8)
ax.plot([], [], "-", color=INK2, label=f"rate model, nominal k (stage efficiency {results[1]['efficiency'][0]:.0%}"
                                f"–{results[1]['efficiency'][1]:.0%})")
ax.plot([], [], "--", color=INK2, label="exact equilibrium stages (k x 1000)")
ax.legend(loc="upper left")
section_bands(ax, results[1]["circ"].cmap)
ax.set_xlabel("stage (0 = raffinate end)")
ax.set_ylabel("organic, mol/L")
ax.set_ylim(bottom=0)
ax.set_title(f"{name}: the rate model against exact equilibrium stages", loc="left", color=INK, pad=16)
fig.savefig(FIGURES / f"06_exact_vs_fast_{name}.png")
print("wrote", FIGURES / f"06_exact_vs_fast_{name}.png")
