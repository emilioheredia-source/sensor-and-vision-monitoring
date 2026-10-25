"""Where along the cascade do the instruments see what the fit needs? A sensitivity sweep.

The question. Notebook 31 put three optical (UV-Vis) points on extraction stages 2, 4 and 6 and
found that their ratios identify the separation factor best. Notebook 33 then found a fault in the
scrub that one cascade-wide stage-rate factor couldn't describe: it moved the wrong way and sat on
its bound. So: where does each unknown number move the profile most, and which instrument layout
pins each one, including a separate rate factor for each section, which would say where a fault is?

The method, on the model of notebook 31's plant (8 extraction, 20 scrub, 4 strip stages, stages
counted from the raffinate end) at its design point, no simulation in time:
  1. Six unknowns: the log Pr/(La+Ce) and Nd/Pr separation factors, a log rate factor for each
     section (extraction, scrub, strip), and the log gain of the optical instrument (unknown
     calibration, common to all its readings; it's what makes optical readings relative).
  2. For each, the change in every stage's log Pr and log Nd concentration, aqueous and organic,
     by finite differences on the steady state (step 0.02 in the log).
  3. For candidate layouts, the Fisher information of the readings (XRF on product and raffinate
     at 2 %, optical Pr and Nd at 1 %), and from it the smallest standard error each unknown can
     be estimated with: a lower bound, the precision a fit could reach with these readings.

Run:  pixi run python notebooks/39_sensor_placement_sensitivity.py
Writes runs/sensor_sensitivity_ndpr_plant.csv (per stage), runs/sensor_layouts_ndpr_plant.csv (per
layout) and figures/39_sensor_sensitivity.png.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from src.config import build_circuit, load_config, nominal_inputs       # noqa: E402
from src.simulate import steady_state                                   # noqa: E402

CONFIG = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "ndpr_plant"
STEP = 0.02
XRF_SD, UV_SD = 0.02, 0.01
UNKNOWNS = ("beta Pr/(La+Ce)", "beta Nd/Pr", "rate, extraction", "rate, scrub", "rate, strip", "optical gain")
LAYOUTS = {
    "XRF ends only": (),
    "optical 2, 4, 6 (notebook 31)": (2, 4, 6),
    "optical 2, 6 + scrub 12": (2, 6, 12),
    "optical 4 + scrub 10, 16": (4, 10, 16),
    "optical 2, 6 + scrub 12, 20": (2, 6, 12, 20),
    "optical 2, 6 + scrub 12 + strip 30": (2, 6, 12, 30),
}

_spec = importlib.util.spec_from_file_location("nb31", HERE / "notebooks" / "31_identify_then_control.py")
_src = _spec.loader.get_source("nb31").split("\ndone = set()")[0]
NB31 = {"__file__": str(_spec.origin)}
argv, sys.argv = sys.argv, [sys.argv[0], "0"]
exec(compile(_src, str(_spec.origin), "exec"), NB31)
sys.argv = argv
model_with = NB31["model_with"]

rc = load_config(CONFIG)
base = build_circuit(rc)
cm = base.cmap
N = base.N
sections = np.array([base.cmap.section(j) for j in range(N)])
names = list(base.p.names)
i_pr, i_nd = names.index("Pr"), names.index("Nd")
u0 = nominal_inputs(rc, base)


def profile(theta):
    """log of Pr and Nd in every stage, aqueous and organic, and the XRF readings (log, product and
    raffinate, La+Ce, Pr, Nd), for the unknowns theta (the first five; the gain doesn't move the plant)."""
    model, _, _ = model_with(rc, [theta[0], theta[1], 0.0])
    k = np.ones(N)
    for s, name in zip(theta[2:5], ("extraction", "scrub", "strip")):
        k[sections == name] *= np.exp(s)
    u = u0.replace(k_factor=k)
    x = steady_state(model, u, x0=profile.x)
    profile.x = x
    c_aq, c_org = model.unpack(model.observable(x))[:2]
    c_aq, c_org = np.asarray(c_aq, float), np.asarray(c_org, float)
    o = model.outlets(x, u.numpy())
    xrf = np.log(np.maximum(np.concatenate([np.atleast_1d(o["c_prod"])[:3], np.atleast_1d(o["c_raff"])[:3]]), 1e-12))
    return (np.log(c_aq[:, [i_pr, i_nd]]), np.log(c_org[:, [i_pr, i_nd]]), xrf)


profile.x = None
theta0 = np.zeros(5)
aq0, org0, xrf0 = profile(theta0)
d_aq, d_org, d_xrf = [], [], []
for p in range(5):
    th = theta0.copy()
    th[p] += STEP
    aq, org, xrf = profile(th)
    d_aq.append((aq - aq0) / STEP)
    d_org.append((org - org0) / STEP)
    d_xrf.append((xrf - xrf0) / STEP)
d_aq, d_org, d_xrf = np.array(d_aq), np.array(d_org), np.array(d_xrf)   # (unknown, stage, Pr/Nd) and (unknown, 6)

with open(HERE / "runs" / f"sensor_sensitivity_{CONFIG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stage", "section", "phase", "element"] + list(UNKNOWNS[:5]))
    for j in range(N):
        for phase, d in (("aqueous", d_aq), ("organic", d_org)):
            for e, el in enumerate(("Pr", "Nd")):
                w.writerow([j, sections[j], phase, el] + [f"{d[p, j, e]:.4f}" for p in range(5)])


def standard_errors(stages, phase="aqueous"):
    """Lower bound on each unknown's standard error (log units) from XRF at the ends plus optical Pr
    and Nd at `stages` with an unknown common gain. Infinite where the readings can't tell."""
    d = d_aq if phase == "aqueous" else d_org
    rows, sds = [], []
    for r in range(6):                                   # XRF: no gain unknown
        rows.append(list(d_xrf[:, r]) + [0.0])
        sds.append(XRF_SD)
    for j in stages:
        for e in range(2):
            rows.append(list(d[:, j, e]) + [1.0])         # optical: the gain adds to every log reading
            sds.append(UV_SD)
    J = np.array(rows) / np.array(sds)[:, None]
    F = J.T @ J
    if not stages:                                       # no optical readings: the gain isn't a question
        F = F[:5, :5]
    try:
        cov = np.linalg.inv(F)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(F.shape[0], np.inf)
    cond = np.linalg.cond(F)
    if cond > 1e12:
        se = np.full(F.shape[0], np.inf)
    return list(se[:5]), cond


with open(HERE / "runs" / f"sensor_layouts_{CONFIG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layout", "phase"] + [f"se {u}" for u in UNKNOWNS[:5]] + ["condition number"])
    for name, stages in LAYOUTS.items():
        for phase in ("aqueous", "organic") if stages else ("aqueous",):
            se, cond = standard_errors(stages, phase)
            w.writerow([name, phase] + [f"{s:.4f}" if np.isfinite(s) else "inf" for s in se] + [f"{cond:.3g}"])
            print(f"{name:38s} {phase:8s} " + "  ".join(f"{s:7.4f}" if np.isfinite(s) else "    inf" for s in se)
                  + f"   cond {cond:.2g}")


def shade_scrub(ax, stages) -> None:
    """Shade the scrub stages on one panel and draw its zero line."""
    from src.plotting import GRID, INK2
    js = stages[sections == "scrub"]
    ax.axvspan(js[0] - 0.5, js[-1] + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.axhline(0, color=INK2, lw=0.6)


def figure():
    """How much each unknown moves the aqueous Pr and Nd at every stage, and the Nd/Pr ratio, as a figure."""
    import matplotlib.pyplot as plt
    from src.plotting import FIGURES, INK2, SLOT, style
    style()
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.6), sharex=True)
    stages = np.arange(N)
    groups = [("separation factors", [0, 1]), ("rate factors, per section", [2, 3, 4])]
    colors = [SLOT[0], SLOT[1], SLOT[2], SLOT[3], SLOT[4]]
    for ax, (title, ps) in zip(axes[:2], groups):
        for p in ps:
            ax.plot(stages, d_aq[p, :, 0], color=colors[p], lw=1.8, label=f"{UNKNOWNS[p]}, Pr")
            ax.plot(stages, d_aq[p, :, 1], color=colors[p], lw=1.2, ls="--", label=f"{UNKNOWNS[p]}, Nd")
        ax.set_ylabel("change in log\nconcentration\nper unit log")
        ax.set_title(f"Aqueous Pr and Nd, each stage: how much the {title} move them", fontsize=9.5)
        ax.legend(fontsize=7, ncol=2, frameon=False)
    # the Nd/Pr ratio inside one spectrum (the gain cancels): its sensitivity to the two betas
    ratio = d_aq[:, :, 1] - d_aq[:, :, 0]
    for p in range(5):
        axes[2].plot(stages, ratio[p], color=colors[p], lw=1.6, label=UNKNOWNS[p])
    axes[2].set_ylabel("change in log Nd/Pr\nper unit log")
    axes[2].set_title("The Nd/Pr ratio within one spectrum (calibration cancels)", fontsize=9.5)
    axes[2].legend(fontsize=7, ncol=3, frameon=False)
    axes[2].set_xlabel("stage, from the raffinate end")
    for ax in axes:
        shade_scrub(ax, stages)
    for name in ("extraction", "scrub", "strip"):
        js = stages[sections == name]
        axes[0].text((js[0] + js[-1]) / 2, axes[0].get_ylim()[1], name, ha="center", va="bottom", fontsize=8, color=INK2)
    fig.tight_layout()
    name = "39_sensor_sensitivity.png" if CONFIG == "ndpr_plant" else f"39_sensor_sensitivity_{CONFIG}.png"
    fig.savefig(FIGURES / name, dpi=150)
    print("wrote", FIGURES / name)


figure()
