"""Diagrams for the report: things the result figures do not show.

  diag_circuit.png        the three-section circuit, who feeds whom
  diag_stage.png          one mixer-settler, with the weir and the jackleg
  diag_control.png        the two layers of control, the estimator loop, where upsets enter
  diag_flowsheet.png      the circuit in general terms: every stream, the stage it enters and leaves
  diag_mpc_loop.png/.svg  the two-layer controller as a classical feedback loop, with the output bias
  diag_model.png          the model: one stage, its states, and the equations every stage carries

Drawn with matplotlib so they are reproducible and share the figures' look.
Every diagram is drawn on a plain axis with its frame turned off, in made-up
coordinates: boxes, arrows and text placed by hand. The numbers in the calls
below are positions on that axis, not process values. Nothing is simulated.

It also refreshes the copy of diag_flowsheet.png that ARCHITECTURE.md carries
inside the file (see section 4).

Run:  pixi run python notebooks/11_diagrams.py
Writes figures/diag_*.png (and diag_mpc_loop.svg), and the flowsheet image inside ARCHITECTURE.md.
"""
import base64
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotting import FIGURES, GRID, INK, INK2, MUTED, SLOT, STATUS_CRITICAL, SURFACE, style   # noqa: E402

style()
ORG, AQ = SLOT[3], SLOT[0]          # organic in yellow, aqueous in blue, throughout


def arrow(ax, p, q, color, lw=1.6):
    """Draw a straight arrow on ax from point p to point q, both (x, y), in the given colour and line width.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, ha="center", va="bottom", size=8, color=INK2):
    """Write text on ax at (x, y); ha and va set which side of the point the text sits on, size is in points.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color)


# ---------------------------------------------------------------------------
# 1. the circuit
# ---------------------------------------------------------------------------
# The ndpr_plant circuit: three boxes for the sections, the organic loop on top, the two aqueous
# paths below. The thin lines inside a box mark stages, but only as a sketch: extraction and scrub
# are both drawn with eight cells and the strip with four, whatever the real counts.
fig, ax = plt.subplots(figsize=(9.6, 4.3))
ax.set_xlim(-0.6, 13.0)
ax.set_ylim(-2.4, 2.7)
ax.axis("off")
sections = [("extraction", 0.0, 4.2, "8 stages"), ("scrub", 4.5, 9.3, "20 stages"), ("strip", 9.6, 11.6, "4 stages")]
for name, x0, x1, n in sections:
    ax.add_patch(Rectangle((x0, -0.55), x1 - x0, 1.1, fc=GRID if name == "scrub" else SURFACE, ec=INK2, lw=0.9))
    label(ax, (x0 + x1) / 2, 0.72, name, size=10, color=INK)
    label(ax, (x0 + x1) / 2, -0.78, n, va="top", color=MUTED)
    k = 8 if name != "strip" else 4
    for i in range(1, k):
        xx = x0 + (x1 - x0) * i / k
        ax.plot([xx, xx], [-0.55, 0.55], color=INK2, lw=0.4, alpha=0.5)
label(ax, 0.08, 0.0, "stage 0", ha="left", va="center", size=7, color=MUTED)
label(ax, 11.52, 0.0, "stage 31", ha="right", va="center", size=7, color=MUTED)

# organic: enters at stage 0, runs left to right through everything, recycled
arrow(ax, (-0.5, 0.3), (0.0, 0.3), ORG)
arrow(ax, (4.2, 0.3), (4.5, 0.3), ORG)
arrow(ax, (9.3, 0.3), (9.6, 0.3), ORG)
arrow(ax, (11.6, 0.3), (12.3, 0.3), ORG)
ax.plot([12.3, 12.3, -0.5, -0.5], [0.3, 2.0, 2.0, 0.3], color=ORG, lw=1.6)
label(ax, 6.0, 2.06, "stripped organic goes round again")
label(ax, -0.45, 1.2, "organic in", ha="left", va="center")
label(ax, 2.1, 0.33, "organic  →")

# aqueous path A: scrub liquor in at the last scrub stage, flows right to left, feed joins at the
# last extraction stage, raffinate out at stage 0
arrow(ax, (8.6, -1.25), (8.6, -0.55), AQ)
label(ax, 8.5, -1.32, "scrub liquor in\n(strip product sent back,\nat a set acidity)", ha="right", va="top")
arrow(ax, (4.5, -0.3), (4.2, -0.3), AQ)
label(ax, 6.9, -0.33, "←  aqueous", va="top")
arrow(ax, (3.9, -1.25), (3.9, -0.55), AQ)
label(ax, 3.9, -1.32, "feed in\n(monazite liquor,\nSm and heavier\nremoved upstream)", va="top")
label(ax, 2.1, -0.33, "←  aqueous", va="top")
arrow(ax, (0.0, -0.3), (-0.5, -0.3), AQ)
label(ax, -0.55, -0.6, "raffinate out\n(La and Ce)", ha="left", va="top")

# aqueous path B: strip acid in at the last stage, product out of the first strip stage
arrow(ax, (11.3, -1.25), (11.3, -0.55), AQ)
label(ax, 11.35, -1.32, "strip acid\nin (2 M)", ha="left", va="top")
label(ax, 10.6, -0.33, "←  aqueous", va="top")
arrow(ax, (9.9, -0.55), (9.9, -1.25), AQ)
label(ax, 9.85, -1.32, "NdPr product\nout (Pr and\nNd together)", ha="right", va="top")
label(ax, 6.0, 2.5, "The NdPr circuit (ndpr_plant): 32 mixer-settlers, two liquids moving opposite ways", size=10, color=INK)
fig.savefig(FIGURES / "diag_circuit.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. one stage
# ---------------------------------------------------------------------------
# One mixer-settler: the mixer on the left, the settler with the organic over the aqueous, the
# organic leaving over a weir and the aqueous through the jackleg. ARCHITECTURE.md, "The interface",
# has the balance the caption describes.
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.set_xlim(-0.4, 12.0)
ax.set_ylim(-1.3, 3.5)
ax.axis("off")
# mixer
ax.add_patch(Rectangle((0.0, 0.0), 2.2, 2.4, fc=SURFACE, ec=INK2, lw=1.0))
label(ax, 1.1, 2.5, "mixer", size=9, color=INK)
ax.plot([1.1, 1.1], [1.25, 2.4], color=INK2, lw=1.0)
ax.plot([0.7, 1.5], [1.25, 1.25], color=INK2, lw=2.0)
ax.add_patch(Rectangle((0.0, 0.0), 2.2, 2.4, fc="none", ec=INK2, lw=1.0))
label(ax, 1.1, 1.05, "both liquids stirred\ntogether: metal moves\nbetween them,\nacid is released", va="top", size=7)
# settler
ax.add_patch(Rectangle((2.9, 0.0), 5.4, 2.4, fc=SURFACE, ec=INK2, lw=1.0))
ax.add_patch(Rectangle((2.9, 0.0), 5.4, 1.1, fc=AQ, alpha=0.18, lw=0))
ax.add_patch(Rectangle((2.9, 1.1), 5.4, 1.1, fc=ORG, alpha=0.25, lw=0))
ax.plot([2.9, 8.3], [1.1, 1.1], color=INK, lw=1.4)
label(ax, 5.6, 1.17, "interface  (must stay inside its window)", color=INK)
label(ax, 5.6, 1.75, "organic (lighter, on top)", va="center")
label(ax, 5.6, 0.5, "aqueous (heavier, below)", va="center")
label(ax, 5.6, 2.5, "settler", size=9, color=INK)
# emulsion in
arrow(ax, (2.2, 1.6), (2.9, 1.6), INK2)
label(ax, 2.55, 1.72, "mixed\nliquids", size=7)
# organic weir out
ax.plot([8.3, 8.3], [2.2, 2.4], color=INK2, lw=2.5)
arrow(ax, (8.3, 2.25), (9.4, 2.25), ORG)
label(ax, 8.35, 2.45, "organic out, over the weir", ha="left")
# jackleg: pipe from the bottom, up to an adjustable height, then out
ax.plot([8.3, 8.9, 8.9, 9.4], [0.25, 0.25, 1.75, 1.75], color=AQ, lw=2.2)
arrow(ax, (9.4, 1.75), (10.1, 1.75), AQ)
ax.annotate("", xy=(9.15, 1.75), xytext=(9.15, 0.25), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=0.8))
label(ax, 9.5, 1.55, "jackleg: aqueous out\nover an adjustable height,\nmoved by the settler's\nlevel loop", ha="left", va="top", size=7.5)
label(ax, 5.6, -0.45, "At rest the interface sits where the two liquid columns balance: raise the jackleg and it rises;\n"
      "load the organic with metal (heavier) and it drops. In the model it relaxes to that height in minutes.", va="top")
label(ax, 5.2, 3.15, "One mixer-settler stage", size=10, color=INK)
fig.savefig(FIGURES / "diag_stage.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. the control structure
# ---------------------------------------------------------------------------
# Five boxes: optimizer, MPC and plant along the top, estimator and sensors below, with what passes
# between them on the arrows.
fig, ax = plt.subplots(figsize=(9.6, 5.4))
ax.set_xlim(-0.2, 12.4)
ax.set_ylim(-0.6, 6.6)
ax.axis("off")


def box(x, y, w, h, title, body, fc=SURFACE):
    """Draw a rounded box with a bold title at the top and smaller body text at the bottom.
    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.

    x, y   the lower-left corner; w, h the width and height, all in the diagram's coordinates.
    fc     the fill colour.
    Draws on the module-level ax, so it lands on whichever diagram is being drawn
    (used again in section 5).
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=INK2, lw=1.0))
    ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=9.5, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + 0.16, body, ha="center", va="bottom", fontsize=7.4, color=INK2)


box(0.0, 4.2, 3.2, 1.7, "Steady-state optimizer",
    "nonlinear model, IPOPT\nleast reagent at purity ≥ 99.5 %,\nrecovery ≥ 95 % (the run files);\nthen a trim on the plant itself")
box(4.5, 4.2, 3.2, 1.7, "MPC (every 30 min)",
    "its own model state, re-linearized,\nplus a bias from what is reported;\nQP: floors on purity and recovery,\ninterfaces in window, little reagent")
box(9.0, 4.2, 3.2, 1.7, "Plant (the simulator)",
    "32 stages, nonlinear\nupsets enter here: feed,\nflowmeter bias, hydraulic drift,\nslow mixers", fc=GRID)
box(9.0, 1.1, 3.2, 1.5, "Sensors (readings only)",
    "analysers, pH, density, interface\nlevel, lab assay — each with\nnoise, delay, sampling")
box(4.5, 1.1, 3.2, 1.5, "Estimator",
    "predicts with the model,\ncorrects from readings; estimates\nfeed offset and flowmeter bias too")
arrow(ax, (3.2, 5.05), (4.5, 5.05), INK2)
label(ax, 3.85, 5.12, "operating\npoint", size=7)
arrow(ax, (7.7, 5.05), (9.0, 5.05), INK2)
label(ax, 8.35, 5.12, "handles:\nflows, acid,\nsaponification,\njacklegs", size=7)
arrow(ax, (10.6, 4.2), (10.6, 2.6), INK2)
label(ax, 10.7, 3.4, "true state", ha="left", va="center", size=7.6)
arrow(ax, (9.0, 1.85), (7.7, 1.85), INK2)
label(ax, 8.35, 1.95, "readings", size=7.6)
arrow(ax, (6.1, 2.6), (6.1, 4.2), INK2)
label(ax, 6.2, 3.4, "state estimate,\nfeed estimate,\nflow offset", ha="left", va="center", size=7.6)
ax.plot([10.6, 10.6, 6.1, 6.1], [5.9, 6.15, 6.15, 5.9], color=MUTED, lw=0.9, ls="--")
label(ax, 8.35, 6.18, "feed analyser and feed flowmeter: feedforward", size=7.6, color=MUTED)
label(ax, 6.0, 0.15, "Two layers, the way plant MPC is built. With a sensor set the controller sees only readings; the full-state reference sees the true state.")
label(ax, 6.0, -0.35, "The control structure", size=10, color=INK)
fig.savefig(FIGURES / "diag_control.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. the flowsheet in general terms: every stream, where it enters and leaves
# ---------------------------------------------------------------------------
FEED_C, SCRUB_C, STRIP_C = SLOT[0], SLOT[2], SLOT[1]     # feed -> raffinate blue, scrub green, strip orange
fig, ax = plt.subplots(figsize=(10.4, 6.2))
ax.set_xlim(-2.6, 14.0)
ax.set_ylim(-3.3, 4.3)
ax.axis("off")
# Each section is a box split into k cells. The labels above the cells name the first, second and
# last stage of the section the way the code numbers them: n_e extraction stages, n_s scrub stages,
# N stages in all, counted from 0 at the raffinate end. The strip has no second label.
Y_ORG, Y_AQ = 1.05, 0.4                       # the organic runs in the top half of each box, the aqueous below
blocks = [("extraction", 0.0, 3.6, 6, ("0", "1", "$n_e-1$")),
          ("scrub", 4.4, 8.0, 6, ("$n_e$", "$n_e+1$", "$n_e+n_s-1$")),
          ("strip", 8.8, 11.0, 3, ("$n_e+n_s$", None, "$N-1$"))]
for name, x0, x1, k, (first, second, last) in blocks:
    ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.4, fc=GRID if name == "scrub" else SURFACE, ec=INK2, lw=0.9))
    for i in range(1, k):
        xx = x0 + (x1 - x0) * i / k
        ax.plot([xx, xx], [0.0, 1.4], color=INK2, lw=0.4, alpha=0.5)
    w = (x1 - x0) / k
    label(ax, (x0 + x1) / 2, 2.4, name, size=10, color=INK)
    label(ax, (x0 + x1) / 2, 2.0, f"stages {first} … {last}", size=8, color=INK2)
    # every stage is one cell; the first ones and the last one are named, as in the code
    label(ax, x0 + w / 2, 1.5, first, size=7.5, color=INK)
    if second is not None:
        label(ax, x0 + 1.5 * w, 1.5, second, size=7.5, color=INK)
    label(ax, (x0 + x1) / 2 + (w / 2 if second is not None else 0), 1.5, "…", size=7.5, color=INK)
    label(ax, x1 - w / 2, 1.5, last, size=7.5, color=INK)

# organic: in at stage 0, left to right through every section, out of N-1, round again
for a, b in [(-1.2, 0.0), (3.6, 4.4), (8.0, 8.8), (11.0, 12.2)]:
    arrow(ax, (a, Y_ORG), (b, Y_ORG), ORG)
for x0, x1 in [(0.0, 3.6), (4.4, 8.0), (8.8, 11.0)]:
    ax.plot([x0, x1], [Y_ORG, Y_ORG], color=ORG, lw=1.6, alpha=0.6)
ax.plot([12.2, 12.2, -1.2, -1.2], [Y_ORG, 3.15, 3.15, Y_ORG], color=ORG, lw=1.6)
arrow(ax, (5.6, 3.15), (5.4, 3.15), ORG)
label(ax, 5.5, 3.25, "ORGANIC, a closed loop: stripped of its metal in the strip section and sent round again", color=INK)
label(ax, -1.35, Y_ORG, "ORGANIC\ninto stage 0", ha="right", va="center", color=INK)
label(ax, 12.35, Y_ORG, "ORGANIC out\nof stage $N-1$", ha="left", va="center", color=INK)

# aqueous path A: scrub liquor in at the last scrub stage, flows right to left through the scrub,
# picks up the feed at the last extraction stage, leaves as raffinate from stage 0
arrow(ax, (7.75, -1.15), (7.75, 0.0), SCRUB_C)
ax.plot([7.75, 4.4], [Y_AQ, Y_AQ], color=SCRUB_C, lw=1.6, alpha=0.6)
arrow(ax, (4.4, Y_AQ), (3.6, Y_AQ), SCRUB_C)
label(ax, 7.85, -1.25, "SCRUB LIQUOR in\nat stage $n_e+n_s-1$\n(last scrub stage)", ha="right", va="top", color=INK)
arrow(ax, (3.35, -1.15), (3.35, 0.0), FEED_C)
ax.plot([3.35, 0.0], [Y_AQ, Y_AQ], color=FEED_C, lw=1.6, alpha=0.6)
arrow(ax, (0.0, Y_AQ), (-1.2, Y_AQ), FEED_C)
label(ax, 3.45, -1.25, "FEED in\nat stage $n_e-1$\n(last extraction stage)", ha="right", va="top", color=INK)
label(ax, -1.25, 0.25, "RAFFINATE\nout of stage 0\n(La and Ce)", ha="right", va="top", color=INK)

# aqueous path B: strip acid in at N-1, right to left through the strip only, product out of its first stage
arrow(ax, (10.75, -1.15), (10.75, 0.0), STRIP_C)
ax.plot([10.75, 9.05], [Y_AQ, Y_AQ], color=STRIP_C, lw=1.6, alpha=0.6)
arrow(ax, (9.05, 0.0), (9.05, -2.25), STRIP_C)
label(ax, 10.85, -1.25, "STRIP ACID in\nat stage $N-1$", ha="left", va="top", color=INK)
label(ax, 9.15, -2.35, "PRODUCT out of stage $n_e+n_s$\n(first strip stage; NdPr)", ha="left", va="top", color=INK)
label(ax, 8.4, -0.2, "no aqueous\nlink here", va="top", size=7, color=MUTED)

# directions, and the reflux note
for x0, x1 in [(0.0, 3.6), (4.4, 8.0), (8.8, 11.0)]:
    label(ax, (x0 + x1) / 2, 0.68, "organic →", size=7, color=INK2)
    label(ax, (x0 + x1) / 2, 0.08, "← aqueous", size=7, color=INK2)
label(ax, 5.5, -3.2, "Two aqueous paths that never meet, one organic loop through both. With a reflux scrub "
      "(the NdPr run files) the scrub liquor is part of the product, sent back.", va="bottom", size=7.8)
label(ax, 5.5, 3.85, "The circuit in general terms: $n_e$ extraction, $n_s$ scrub, $N$ stages in all, "
      "numbered from 0 at the raffinate end", size=10, color=INK)
fig.savefig(FIGURES / "diag_flowsheet.png", bbox_inches="tight", pad_inches=0.4)
plt.close(fig)

# ARCHITECTURE.md carries this figure inside the file (base64), between two marker comments, so it
# shows where the document is read even without the figures folder; refreshed on every redraw.
# If either marker is missing, the file is left alone.
doc = FIGURES.parent / "ARCHITECTURE.md"
start, end = "<!-- flowsheet figure: written by notebooks/11_diagrams.py -->", "<!-- end of flowsheet figure -->"
text = doc.read_text(encoding="utf-8")
if start in text and end in text:
    data = base64.b64encode((FIGURES / "diag_flowsheet.png").read_bytes()).decode("ascii")
    block = f"{start}\n![the circuit's streams, where each enters and leaves](data:image/png;base64,{data})\n{end}"
    head, rest = text.split(start, 1)
    doc.write_text(head + block + rest.split(end, 1)[1], encoding="utf-8", newline="\n")
print("wrote", FIGURES / "diag_circuit.png", FIGURES / "diag_stage.png", FIGURES / "diag_control.png",
      FIGURES / "diag_flowsheet.png")

# ---------------------------------------------------------------------------
# 5. the two-layer controller as a classical feedback loop
# ---------------------------------------------------------------------------
# The same controller as section 3, drawn the way a control engineer would: the blocks, the
# feedforward path from the feed analyser, the feedback path through sensors and estimator, and the
# output bias (the model's outputs subtracted from the estimate's, filtered, added to the MPC's
# prediction). ARCHITECTURE.md decision 21 has why the bias is there.
FF, NOT_BUILT = SLOT[2], MUTED                 # feedforward in green, what isn't built in grey
fig, ax = plt.subplots(figsize=(11.0, 7.4))
ax.set_xlim(-0.3, 14.6)
ax.set_ylim(0.7, 10.6)
ax.axis("off")

box(0.2, 6.2, 3.6, 1.9, "Steady-state optimizer",
    "nonlinear steady-state optimizer (IPOPT)\nmost recovery, then least reagent (mol),\n"
    "run offline: writes the design point")
box(5.0, 6.0, 3.9, 2.3, "MPC (every 30 min)",
    "linear MPC — re-linearize every step,\nsolve QP (OSQP)\n\non its own model state $x_m$;\n"
    "plans on  $\\hat y = y(x_m) + b$;\nkeeps extraction acid above a floor")
box(10.4, 6.2, 3.4, 1.9, "Plant",
    "nonlinear, 32 stages\n(perturbed, drifting, book,\nsettler or entrainment)", fc=GRID)
# the lower layer: a level loop on each settler, under the MPC
ax.add_patch(FancyBboxPatch((10.2, 4.65), 1.8, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=SURFACE, ec=INK2, lw=0.9))
label(ax, 11.1, 4.82, "level loop per settler\nPI: interface\n→ jackleg", size=6.2, color=INK)
arrow(ax, (11.1, 5.6), (11.1, 6.2), INK2)
box(10.4, 2.3, 3.4, 1.7, "Sensors",
    "analysers, pH, interface level,\nlab assay — each with noise,\ndelay and sample interval")
box(5.0, 2.3, 3.9, 1.7, "Estimator",
    "estimator — nonlinear predict\n+ Kalman correct, disturbance states:\nfeed-composition and organic-flow offsets")

# targets into the optimizer, operating point to the MPC, handles to the plant
arrow(ax, (2.0, 9.2), (2.0, 8.1), INK2)
label(ax, 2.0, 9.28, "targets: purity 99.8 %, recovery 97 %\n(spec 99 % / 95 %, run with margin)", size=7.6)
arrow(ax, (3.8, 7.15), (5.0, 7.15), INK2)
label(ax, 4.4, 7.25, "operating\npoint, floors", size=7.2)
arrow(ax, (8.9, 7.15), (10.4, 7.15), INK2)
label(ax, 9.65, 7.25, "four handles", size=7.6)
label(ax, 9.65, 7.0, "organic flow,\nscrub flow,\nscrub acidity,\nsaponification",
      va="top", size=6.8)

# upsets enter the plant after the handles are written
arrow(ax, (13.0, 9.55), (13.0, 8.1), STATUS_CRITICAL)
label(ax, 12.75, 9.65, "upsets, applied after the handles:\nflowmeter bias × commanded organic,\n"
      "hydraulic drift + jackleg set by its level loop", size=7.2)

# feed analyser: feedforward, separate from the feedback loop
arrow(ax, (10.8, 10.3), (10.8, 8.1), INK2)
label(ax, 10.9, 10.0, "feed", ha="left", size=7.6)
ax.add_patch(FancyBboxPatch((8.8, 9.35), 1.5, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=SURFACE, ec=FF, lw=1.0))
label(ax, 9.55, 9.47, "feed analyser", size=7.4, color=INK)
ax.plot([10.8, 10.3], [9.6, 9.6], color=FF, lw=1.2, ls="--")
ax.plot([8.8, 7.6, 7.6], [9.6, 9.6, 8.6], color=FF, lw=1.4, ls="--")
arrow(ax, (7.6, 8.6), (7.6, 8.3), FF)
label(ax, 7.45, 9.0, "feedforward:\nfeed composition", ha="right", va="center", size=7.4, color=INK)

# feedback: plant -> sensors -> estimator
arrow(ax, (12.1, 6.2), (12.1, 4.0), INK2)
label(ax, 12.2, 5.1, "true state", ha="left", va="center", size=7.6)
arrow(ax, (10.4, 3.15), (8.9, 3.15), INK2)
label(ax, 9.65, 3.25, "readings", size=7.6)

# the output bias: reported outputs minus the model's own, filtered, added to the prediction
SX, SY, R = 6.0, 4.95, 0.22
ax.add_patch(plt.Circle((SX, SY), R, fc=SURFACE, ec=INK, lw=1.1))
label(ax, SX, SY, "Σ", va="center", size=9, color=INK)
arrow(ax, (SX, 4.0), (SX, SY - R), INK2)
label(ax, SX - 0.1, 4.35, "+  $y(\\hat x)$", ha="right", va="center", size=7.6, color=INK)
ax.plot([5.35, 5.35], [6.0, SY], color=INK2, lw=1.6)
arrow(ax, (5.35, SY), (SX - R, SY), INK2)
label(ax, 5.25, 5.45, "$y(x_m)$", ha="right", va="center", size=7.6, color=INK)
label(ax, SX - R - 0.05, SY + 0.12, "−", ha="right", va="bottom", size=9, color=INK)
ax.add_patch(FancyBboxPatch((6.55, 4.7), 1.0, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=SURFACE, ec=INK2, lw=0.9))
label(ax, 7.05, 4.83, "filter α = 0.5", size=6.8, color=INK)
arrow(ax, (SX + R, SY), (6.55, SY), INK2)
arrow(ax, (7.05, 5.2), (7.05, 6.0), INK2)
label(ax, 7.15, 5.6, "output bias $b$\n(integral action)", ha="left", va="center", size=7.2, color=INK)

# what else the estimator hands the MPC: offsets it can act on
ax.plot([8.6, 8.6], [4.0, 5.4], color=INK2, lw=1.6)
arrow(ax, (8.6, 5.4), (8.6, 6.0), INK2)
label(ax, 8.7, 4.55, "organic-flow offset;\nfeed estimate when\nthere's no analyser", ha="left", va="center", size=6.8)

# the supervised model update: the model's numbers refitted to the readings, reviewed before use
# (notebooks 31-33). Shown as supervised on purpose: an update that went live unattended could take
# a bad reading or a fault into the model.
SUP = SLOT[4]
box(0.2, 2.2, 3.7, 1.95, "Model update (daily, supervised)",
    "fit the separation factors and the acid\nexponent to the readings; an engineer\nreviews each update before it goes live")
ax.plot([5.0, 3.9], [3.15, 3.15], color=SUP, lw=1.4, ls="-.")
arrow(ax, (4.0, 3.15), (3.9, 3.15), SUP)
label(ax, 4.45, 3.25, "readings,\nestimate", size=6.8)
ax.plot([2.05, 2.05], [4.15, 6.05], color=SUP, lw=1.4, ls="-.")
arrow(ax, (2.05, 6.0), (2.05, 6.2), SUP)
label(ax, 2.15, 5.1, "reviewed model\nto the optimizer\nand the MPC", ha="left", va="center", size=6.8)

# v1.3: what keeps the controller out of trouble when the model is wrong or the plant is slow, dotted
# because none of it is in the default controller: the rest-point check is a setting, off by default
# (ARCHITECTURE decision 42); the acid-balance cap and the 2-hour refit are built in the study
# notebooks (52, 51) around the controller.
OPT = SLOT[1]
ax.add_patch(FancyBboxPatch((4.0, 8.45), 1.6, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=SURFACE, ec=OPT, lw=1.1, ls=":"))
label(ax, 4.8, 8.52, "rest-point check:\nmodel solved to rest\nevery 12 h; long-run\nfloors set there", size=6.2,
      color=INK)
arrow(ax, (5.5, 8.45), (5.5, 8.3), OPT)
ax.add_patch(FancyBboxPatch((9.0, 7.55), 1.3, 0.95, boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc=SURFACE, ec=OPT, lw=1.1, ls=":"))
label(ax, 9.65, 7.62, "acid-balance cap\non saponification\n(loaded organic,\nacids, flows)", size=6.0, color=INK)
ax.plot([9.04, 9.04], [7.55, 7.32], color=OPT, lw=1.2, ls=":")
arrow(ax, (9.04, 7.32), (9.04, 7.17), OPT)
label(ax, 2.05, 2.08, "v1.3 studies: the same three numbers refitted\nevery 2 h on the last 2 h of readings, inside bounds",
      va="top", size=6.6, color=INK)

# key
ax.plot([0.3, 0.9], [1.0, 1.0], color=OPT, lw=1.4, ls=":")
label(ax, 1.0, 1.0, "dotted: an option in the controller, or built in the v1.3 study notebooks around it",
      ha="left", va="center", size=7.4)
ax.plot([0.3, 0.9], [1.45, 1.45], color=INK2, lw=1.6)
label(ax, 1.0, 1.45, "feedback", ha="left", va="center", size=7.4)
ax.plot([2.4, 3.0], [1.45, 1.45], color=FF, lw=1.4, ls="--")
label(ax, 3.1, 1.45, "feedforward", ha="left", va="center", size=7.4)
ax.plot([4.8, 5.4], [1.45, 1.45], color=SUP, lw=1.4, ls="-.")
label(ax, 5.5, 1.45, "supervised update", ha="left", va="center", size=7.4)
label(ax, 7.6, 1.45, "With the full-state reference sensor set, the true state stands in for $\\hat x$.",
      ha="left", va="center", size=7.4, color=MUTED)
label(ax, 7.2, 10.35, "The control layers as a feedback loop", size=10.5, color=INK)
fig.savefig(FIGURES / "diag_mpc_loop.png", bbox_inches="tight", pad_inches=0.3)
fig.savefig(FIGURES / "diag_mpc_loop.svg", bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("wrote", FIGURES / "diag_mpc_loop.png", "and .svg")

# ---------------------------------------------------------------------------
# 6. the model: one stage and the equations
# ---------------------------------------------------------------------------
# What the CasADi model is, on one page: one stage with its states and the streams it trades with
# its neighbours, and the four sets of equations every stage carries (APPENDIX_A_model.md A.4 to
# A.9, src/chemistry.py and src/circuit.py). The numbers quoted are the main case's.
fig, ax = plt.subplots(figsize=(12.4, 7.0))
ax.set_xlim(0.0, 12.4)
ax.set_ylim(0.0, 7.0)
ax.axis("off")

# One stage: organic on top (yellow), aqueous below (blue), the interface between them.
sx0, sx1, sy0, sy1, zy = 1.3, 3.9, 2.55, 5.05, 3.75
ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, zy - sy0, fc=AQ, alpha=0.18, lw=0))
ax.add_patch(Rectangle((sx0, zy), sx1 - sx0, sy1 - zy, fc=ORG, alpha=0.25, lw=0))
ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, fc="none", ec=INK2, lw=1.0))
ax.plot([sx0, sx1], [zy, zy], color=INK, lw=1.4)
label(ax, (sx0 + sx1) / 2, sy1 + 0.1, "stage $j$", size=10, color=INK)
label(ax, (sx0 + sx1) / 2, 4.4, "organic\n$c^o_{ij}$  (4 metals)", va="center", size=8.5, color=INK)
label(ax, (sx0 + sx1) / 2, 3.15, "aqueous\n$c^a_{ij}$  (4 metals),  acid $h_j$", va="center", size=8.5, color=INK)
label(ax, sx1 - 0.05, zy + 0.04, "interface $z_j$", ha="right", size=7.5, color=INK)
# streams: organic runs left to right, aqueous right to left
arrow(ax, (0.25, 4.7), (sx0, 4.7), ORG)
label(ax, 0.25, 4.78, "organic from\nstage $j-1$", ha="left", size=7.2)
arrow(ax, (sx1, 4.7), (5.0, 4.7), ORG)
label(ax, 4.95, 4.78, "to $j+1$", ha="right", size=7.2)
arrow(ax, (5.0, 2.9), (sx1, 2.9), AQ)
label(ax, 4.95, 2.45, "aqueous from $j+1$\n(+ feed or scrub\nwhere they enter)", ha="right", va="top", size=7.2)
arrow(ax, (sx0, 2.9), (0.25, 2.9), AQ)
label(ax, 0.25, 2.8, "to $j-1$", ha="left", va="top", size=7.2)
# what the stage is made of, and what can be moved
label(ax, 2.6, 1.72, "Per stage: 4 + 4 + 1 + 1 = 10 states. 32 stages in the main case: 320 states.\n"
      "Metals: La+Ce, Pr, Nd, Sm and heavier.", va="top", size=7.8, color=INK2)
label(ax, 0.25, 1.08, "One set of equations, used three ways:\n"
      "    integrated in time, it is the simulated plant;\n"
      "    solved for steady state (IPOPT), it designs the operating point;\n"
      "    linearized as it runs, it is the MPC's prediction model.\n"
      "The harder test plants change terms in it; the controller keeps this one.",
      ha="left", va="top", size=7.8, color=INK)
label(ax, 2.6, 6.55, "Handles: organic flow $Q^o$, scrub flow $Q^s$ and acid $h^s$,\n"
      "saponification $s$, jacklegs $H_j$. The feed comes from upstream.", va="top", size=7.8, color=INK2)

# The four sets of equations, each in a box.
EQ = 10.5


def eq_box(x, y, w, h, n, title, lines, note):
    """A numbered box: title on top, equations (mathtext) in the middle, a source note at the bottom.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10", fc=SURFACE,
                                ec=INK2, lw=1.0))
    ax.text(x + 0.15, y + h - 0.15, f"{n}  {title}", ha="left", va="top", fontsize=9.5, color=INK, fontweight="bold")
    for k, (text, size) in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.72 - 0.58 * k, text, ha="center", va="center", fontsize=size, color=INK)
    ax.text(x + 0.15, y + 0.12, note, ha="left", va="bottom", fontsize=7.2, color=INK2)


eq_box(5.6, 3.75, 3.3, 2.75, 1, "Equilibrium",
       [(r"$\mathrm{RE^{3+}} + 3\,\mathrm{H_2L_2} \rightleftharpoons \mathrm{RE(HL_2)_3} + 3\,\mathrm{H^+}$", 8.5),
        (r"$D_{ij} = \frac{K_i\,L_{\mathrm{free},j}^{\,3}}{h_j^{\,3} + h_{\min}^{\,3}}$", EQ + 1),
        (r"$L_{\mathrm{free},j} = L_{\mathrm{tot}} - 3\sum_i c^o_{ij}$", EQ - 1)],
       "Mass action on the extraction reaction.\n$K_i$ fitted to Lyon's batch data (2016);\n"
       "the separation factor is a ratio of $K$s.")
eq_box(9.05, 3.75, 3.3, 2.75, 2, "Transfer, at a finite rate",
       [(r"$r_{ij} = k\,\kappa_j\,V_{\mathrm{mix}}(1-\phi_j)\,(D_{ij}\,c^a_{ij} - c^o_{ij})$", 8.8),
        (r"$\phi_j = Q^a_j / (Q^a_j + Q^o)$", EQ - 1),
        (r"stage efficiency  $\eta_j = \frac{k\tau_j}{1 + k\tau_j}$,   $\tau_j = \frac{V_{\mathrm{mix}}}{Q^a_j + Q^o}$",
         EQ - 1.5)],
       "Metal moves toward equilibrium.\n$k$ = 380 /h: 95 % of the way in the\n"
       r"lab's 3 minutes. $\kappa_j$ = 1 unless fouled.")
eq_box(5.6, 0.55, 3.3, 2.95, 3, "Balances, metal and acid",
       [(r"$V^a_j\,\dot c^a_{ij} = \sum_{\mathrm{in}} Q\,c_i - Q^a_j\,c^a_{ij} - r_{ij}$", EQ - 0.5),
        (r"$V^o_j\,\dot c^o_{ij} = \sum_{\mathrm{in}} Q\,c_i - Q^o\,c^o_{ij} + r_{ij}$", EQ - 0.5),
        (r"$V^a_j\,\dot h_j = \sum_{\mathrm{in}} Q\,h - Q^a_j\,h_j + 3\sum_i r_{ij} - b_j$", EQ - 0.5)],
       "What comes in, minus what goes out,\nminus what transfers. Three H$^+$ per metal;\n"
       "$b_j$ is the saponification base spent there.")
eq_box(9.05, 0.55, 3.3, 2.95, 4, "Interface",
       [(r"$z^{\mathrm{rest}}_j = \frac{\rho^a_j\,H_j - \rho^o_j\,H_w}{\rho^a_j - \rho^o_j}$", EQ + 1),
        (r"$\tau_z\,\dot z_j = z^{\mathrm{rest}}_j - z_j$", EQ - 0.5),
        (r"$\rho^o_j = \rho_0 + \sum_i M_i\,c^o_{ij} / 1000$", EQ - 1.5)],
       "The two liquid columns balance over the\nweir $H_w$ and the jackleg $H_j$. Densities\n"
       "rise with the metal each phase carries.")

label(ax, 6.2, 6.62, "The model: one stage, and the equations CasADi solves for all of them", size=11, color=INK)
fig.savefig(FIGURES / "diag_model.png", bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print("wrote", FIGURES / "diag_model.png")
