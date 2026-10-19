"""Where should the analyser go? Before any controller runs: for one on-stream
composition analyser (every component, in the aqueous) at each possible stage, how well
does the estimator pin down product purity? Then the best pair, greedily.

Measured on the model: the steady-state estimation covariance of the
linearized circuit with that sensor, projected onto product purity. Every
sensor set is judged on the same estimator model (two disturbance states),
so a second sensor can only help. The metric counts precision and sampling,
not delay: a late reading is scored as if it were on time, which flatters
the lab assay; the closed-loop study (08) is where delay costs show.
Compared with the ECJTU patent's choice: one point in extraction, one in scrub.

Then the same question on the perturbed plant, where the model is wrong: for
each single-analyser position, run the estimator for 40 h through a feed
switch, open loop, and measure how far its product purity sits from the
truth. REPORT.md section 6 and ARCHITECTURE.md decision 23 discuss the results.

Run:  pixi run python notebooks/07_sensor_placement.py [config_name] [--plot-only]
Writes runs/sensor_placement_<config>.npz (the scores, which --plot-only redraws from) and
figures/07_sensor_placement_<config>.png
"""
import sys
import time
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.circuit import Inputs                                          # noqa: E402
from src.commission import commission                                   # noqa: E402
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.estimator import Estimator                                     # noqa: E402
from src.metrics import heavy_mask                                      # noqa: E402
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, section_bands, style   # noqa: E402
from src.sensors import Sensor, SensorSet                               # noqa: E402

# ---- 1. settings ----
# CACHE holds the scores, so --plot-only can redraw without the hours of solving.
# N is the number of stages, n the number of components. ecjtu is the patent's layout: one
# analyser in the middle of the extraction section, one in the middle of the scrub.
args = [a for a in sys.argv[1:] if not a.startswith("--")]
name = args[0] if args else "ndpr_plant"
PLOT_ONLY = "--plot-only" in sys.argv
HERE = Path(__file__).resolve().parent.parent
CACHE = HERE / "runs" / f"sensor_placement_{name}.npz"
rc = load_config(name)
p = rc.params
circ = build_circuit(rc)
N, n = circ.N, circ.n
T_STEP = 0.5                            # the estimator's step, h
ND = 2                                  # disturbance states for every set: what one analyser can support
n_ext, n_scrub = rc["circuit"]["n_ext"], rc["circuit"]["n_scrub"]
ecjtu = [n_ext // 2, n_ext + n_scrub // 2]

# ---- 2. on the model: the estimation covariance for each analyser position ----
# sd_* are standard deviations of the product purity estimate (fractions; the figure shows them
# in thousandths). sd_single[j]: one analyser at stage j. sd_pair[j]: a second at stage j, with
# the first fixed at the best single stage. sd_ecjtu and sd_lab: the patent's pair, and a lab
# assay of the product only.
if PLOT_ONLY and CACHE.exists():
    d = np.load(CACHE)
    sd_single, sd_pair, sd_ecjtu, sd_lab = d["sd_single"], d["sd_pair"], float(d["sd_ecjtu"]), float(d["sd_lab"])
    best1, best2 = int(d["best1"]), int(d["best2"])
else:
    t0 = time.time()
    x0, u = commission(circ, nominal_inputs(rc, circ))
    print(f"commissioned in {time.time()-t0:.0f}s", flush=True)

    # d purity / d x at the operating point: how product purity moves with each state.
    # Built symbolically in CasADi, from the circuit's own outlet equations, then evaluated at x0.
    x = ca.SX.sym("x", circ.nx)
    uv = ca.SX.sym("u", Inputs.size(n, N))
    out = circ._out(x=x, u=uv)
    heavy = [i for i, m in enumerate(heavy_mask(p, rc.cut)) if m]          # the product-side components
    purity = ca.sum1(out["c_prod"][heavy]) / ca.sum1(out["c_prod"])
    g = np.array(ca.Function("g", [x, uv], [ca.jacobian(purity, x)])(x0, u.numpy())).ravel()

    # one analyser per candidate stage
    sd_single = np.full(N, np.nan)
    t0 = time.time()
    for j in range(N):
        ss = SensorSet([Sensor("composition_aq", stage=j)], circ)
        sd_single[j] = Estimator(circ, ss, x0, u, T_STEP, max_disturbances=ND).purity_uncertainty(g)
        print(f"  stage {j:2d}: purity sd {sd_single[j]:.2e}", flush=True)
    print(f"{N} single-sensor covariances in {time.time()-t0:.0f}s", flush=True)
    best1 = int(np.nanargmin(sd_single))
    np.savez(CACHE, sd_single=sd_single, best1=best1)           # partial cache, in case the rest dies

    # best second sensor given the first
    sd_pair = np.full(N, np.nan)
    for j in range(N):
        if j == best1:
            continue
        ss = SensorSet([Sensor("composition_aq", stage=best1), Sensor("composition_aq", stage=j)], circ)
        sd_pair[j] = Estimator(circ, ss, x0, u, T_STEP, max_disturbances=ND).purity_uncertainty(g)
    best2 = int(np.nanargmin(sd_pair))

    # the patent's layout, and a lab assay only, for comparison
    sd_ecjtu = Estimator(circ, SensorSet([Sensor("composition_aq", stage=j) for j in ecjtu], circ), x0, u, T_STEP,
                         max_disturbances=ND).purity_uncertainty(g)
    sd_lab = Estimator(circ, SensorSet([Sensor("product_assay")], circ), x0, u, T_STEP,
                       max_disturbances=ND).purity_uncertainty(g)
    np.savez(CACHE, sd_single=sd_single, sd_pair=sd_pair, sd_ecjtu=sd_ecjtu, sd_lab=sd_lab, best1=best1, best2=best2)

print(f"best single analyser: stage {best1} ({circ.cmap.section(best1)}), purity sd {sd_single[best1]:.2e}")
print(f"best pair: stages {best1} + {best2} ({circ.cmap.section(best2)}), purity sd {sd_pair[best2]:.2e}")
print(f"ECJTU-style pair {ecjtu}: purity sd {sd_ecjtu:.2e}")
print(f"lab assay of the product only (2 h / 4 h delay, 0.5 %): purity sd {sd_lab:.2e}")

# ---- 3. the same question on a plant that is not the model ----
# The covariance above takes the model as right. On the perturbed plant (mismatch.py) an analyser
# mid-cascade pins purity only as well as the model's inference from there holds. So, for every
# single-analyser position: how far does the estimator's product purity sit from the truth over
# 40 h with a feed switch at 5 h, open loop? RMS error in purity, thousandths.
from src.config import build_plant                                      # noqa: E402
from src.disturbances import FeedSwitch, Scenario                       # noqa: E402
from src.metrics import product_purity                                  # noqa: E402
from src.simulate import simulate, steady_state                         # noqa: E402


def purity_error(sensors, t_end=40.0):
    """RMS error of the estimator's product purity against the plant's true purity, as a fraction.

    sensors  the list of Sensor objects to give the estimator.
    t_end    length of the run, h.

    Runs the perturbed plant open loop on the design handles through the feed switch,
    one estimator step (T_STEP) at a time. The estimator runs on the nominal model and
    is told the feed flow but not the feed composition. Uses the module-level plant,
    sc, u_plant, u_m, x_rest and x_ss_m set up below.
    """
    ss = SensorSet(sensors, plant, seed=1)
    est = Estimator(circ, ss, x_ss_m, u_m, T_STEP, max_disturbances=ND)
    u_of_t = sc.u_of_t(u_plant)
    # hist keeps (time, state, inputs) for the sensors, which need the past to give delayed readings
    x, t, hist, err = x_rest.copy(), 0.0, [(0.0, x_rest.copy(), u_plant.numpy())], []
    while t < t_end - 1e-9:
        uv = u_of_t(t).numpy()
        x = simulate(plant, x, lambda _t: uv, T_STEP, dt_out=T_STEP).x[-1]
        t += T_STEP
        hist = (hist + [(t, x.copy(), uv)])[-200:]
        x_hat = est.step(t, u_m.replace(Q_feed=u_of_t(t).Q_feed), ss.readings(t, hist))
        true = product_purity(plant.outlets(x, uv)["c_prod"], p, rc.cut)
        seen = product_purity(circ.outlets(x_hat, u_m.numpy())["c_prod"], p, rc.cut)
        err.append(seen - true)
    return float(np.sqrt(np.mean(np.square(err))))


# Run it for every single-analyser position, for the patent's pair and for the lab assay, unless
# --plot-only finds them in the cache. u_plant is the design handles with the plant's own transfer
# rates; u_m the same handles as the model sees them; x_ss_m the model's rest point there.
if PLOT_ONLY and CACHE.exists() and "err_single" in np.load(CACHE):
    d = np.load(CACHE)
    err_single, err_ecjtu, err_lab = d["err_single"], float(d["err_ecjtu"]), float(d["err_lab"])
else:
    plant, kf = build_plant(rc)
    x_rest, u_design = commission(plant, nominal_inputs(rc, plant).replace(k_factor=kf))
    u_plant, u_m = u_design, u_design.replace(k_factor=np.ones(N))
    x_ss_m = steady_state(circ, u_m, x0=x_rest)
    sc = Scenario([FeedSwitch(t=5.0, ramp_h=1.0, to="yangibana", params=p,
                              removed_upstream=tuple(rc["feed"]["removed_upstream"]))])
    t0 = time.time()
    err_single = np.array([purity_error([Sensor("composition_aq", stage=j)]) for j in range(N)])
    err_ecjtu = purity_error([Sensor("composition_aq", stage=j) for j in ecjtu])
    err_lab = purity_error([Sensor("product_assay")])
    print(f"{N + 2} estimator runs on the perturbed plant in {time.time()-t0:.0f}s", flush=True)
    d = dict(np.load(CACHE))
    d.update(err_single=err_single, err_ecjtu=err_ecjtu, err_lab=err_lab)
    np.savez(CACHE, **d)
j_err = int(np.nanargmin(err_single))
print(f"on the perturbed plant, one analyser: best at stage {j_err} ({circ.cmap.section(j_err)}), purity error "
      f"{1e3 * err_single[j_err]:.2f} thousandths rms; at stage {best1} (best on the model) "
      f"{1e3 * err_single[best1]:.2f}; ECJTU pair {1e3 * err_ecjtu:.2f}; lab assay {1e3 * err_lab:.2f}")

# ---- 4. the figure ----
# Top panel: on the model, purity uncertainty against the stage the analyser sits on, for one
# analyser and for a second one; the patent's pair and the lab assay as horizontal lines.
# Bottom panel: the same against the perturbed plant, as rms error. Both on a log scale, in thousandths.
style()
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 7.4), sharex=True, gridspec_kw={"hspace": 0.28})
stages = np.arange(N)
ax.plot(stages, 1e3 * sd_single, color=SLOT[0])
ax.annotate("one analyser, at this stage", (stages[N // 3], 1e3 * sd_single[N // 3]), xytext=(0, 8),
            textcoords="offset points", ha="center", color=INK2, fontsize=8)
ax.plot(stages, 1e3 * sd_pair, color=SLOT[1])
ax.annotate(f"a second analyser, the first fixed at stage {best1}", (stages[N // 3], 1e3 * sd_pair[N // 3]),
            xytext=(0, -6), textcoords="offset points", ha="center", va="top", color=INK2, fontsize=8)
ax.plot([best1], [1e3 * sd_single[best1]], "o", color=INK, markersize=4)
ax.annotate(f"best single: stage {best1}", (best1, 1e3 * sd_single[best1]), xytext=(-6, 10),
            textcoords="offset points", ha="right", color=INK2, fontsize=8)
ax.plot([best2], [1e3 * sd_pair[best2]], "o", color=INK, markersize=4)
ax.annotate(f"best second: stage {best2}", (best2, 1e3 * sd_pair[best2]), xytext=(6, -12),
            textcoords="offset points", ha="right", color=INK2, fontsize=8)
ax.axhline(1e3 * sd_ecjtu, color=MUTED, lw=0.8, ls=":")
ax.text(1, 1e3 * sd_ecjtu, f"ECJTU-style pair (stages {ecjtu[0]}, {ecjtu[1]})", ha="left", va="bottom",
        color=MUTED, fontsize=8)
ax.axhline(1e3 * sd_lab, color=MUTED, lw=0.8, ls="--")
ax.text(1, 1e3 * sd_lab, "lab assay of the product only (delay not counted)", ha="left", va="top",
        color=MUTED, fontsize=8)
section_bands(ax, circ.cmap)
ax.set_ylabel("product purity uncertainty, thousandths")
ax.set_yscale("log")
ax.set_ylim(bottom=0.6e3 * min(np.nanmin(sd_single), np.nanmin(sd_pair)))
ax.set_title(f"{name}: where one on-stream analyser pins product purity down best", loc="left", color=INK, pad=26)
ax.text(0, 1.08, "on the model (estimation covariance)", transform=ax.transAxes, ha="left", va="bottom",
        color=INK2, fontsize=8)

# the same on the plant that is not the model
ax2.plot(stages, 1e3 * err_single, color=SLOT[0])
ax2.plot([j_err], [1e3 * err_single[j_err]], "o", color=INK, markersize=4)
ax2.annotate(f"best on the real plant: stage {j_err}", (j_err, 1e3 * err_single[j_err]), xytext=(8, -4),
             textcoords="offset points", ha="left", va="top", color=INK2, fontsize=8)
ax2.plot([best1], [1e3 * err_single[best1]], "o", color=SLOT[1], markersize=4)
ax2.annotate(f"the model's choice, stage {best1}", (best1, 1e3 * err_single[best1]), xytext=(-8, 8),
             textcoords="offset points", ha="right", color=INK2, fontsize=8)
ax2.axhline(1e3 * err_ecjtu, color=MUTED, lw=0.8, ls=":")
ax2.text(1, 1e3 * err_ecjtu, "ECJTU-style pair", ha="left", va="bottom", color=MUTED, fontsize=8)
ax2.axhline(1e3 * err_lab, color=MUTED, lw=0.8, ls="--")
ax2.text(1, 1e3 * err_lab, "lab assay of the product only", ha="left", va="top", color=MUTED, fontsize=8)
section_bands(ax2, circ.cmap)
ax2.set_xlabel("stage the analyser sits on (0 = raffinate end)")
ax2.set_ylabel("purity estimate error, thousandths rms")
ax2.set_yscale("log")
ax2.text(0, 1.08, "on the perturbed plant, 40 h with a feed switch, open loop", transform=ax2.transAxes,
         ha="left", va="bottom", color=INK2, fontsize=8)
fig.savefig(FIGURES / f"07_sensor_placement_{name}.png")
print("wrote", FIGURES / f"07_sensor_placement_{name}.png")
