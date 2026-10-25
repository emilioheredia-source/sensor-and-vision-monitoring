"""The chemistry from monazite to an NdPr product, as one picture to remember it by.

Steps 1-2 (crack and leach) are textbook monazite processing, not from the project's sources.
Steps 3-5 (load, scrub, strip) are the reaction the sx-cascade-control model uses
(APPENDIX_A_model.md, A.3: Lyon 2016, Rollat et al. 2024).

Run:  pixi run python notebooks/11b_chemistry_route.py
Writes figures/chemistry_route.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

# the project's figure look (projects/sx-cascade-control/src/plotting.py)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BASE, ACID, ORG, OUTC = "#2a78d6", "#eb6834", "#eda100", "#898781"   # caustic blue, acid orange, organic yellow
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "mathtext.default": "regular"})


def box(ax, x, y, w, h, title, sub, edge):
    """A rounded box at (x, y), w by h, with a bold title at the top and a smaller line underneath.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=SURFACE, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
    ax.text(x + w / 2, y + 0.18, sub, ha="center", va="bottom", fontsize=7.6, color=INK2)


def arrow(ax, p, q, color, lw=1.6, ls="-"):
    """An arrow from point p to point q.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, size=8, color=INK2, ha="left", va="center", weight="normal"):
    """Text at (x, y) in the figure's style.

    Its parameters are kept, more than the guidelines like: each is a coordinate or a style setting of
    one drawing call, and bundling them would make every call below harder to read.
    """
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight=weight)


fig, ax = plt.subplots(figsize=(11.5, 12.5))
ax.set_xlim(0, 15)
ax.set_ylim(-0.6, 16.4)
ax.axis("off")

# column headers
text(ax, 1.7, 15.7, "step", 9, INK, "center", weight="bold")
text(ax, 7.0, 15.7, "the reaction", 9, INK, "center", weight="bold")
text(ax, 12.9, 15.7, "what leaves here", 9, INK, "center", weight="bold")

# step rows: (y, title, conditions, edge colour, reaction lines, side stream)
X0, W, H = 0.2, 3.0, 1.75
rows = [
    (13.4, "1. Crack", "hot caustic\n(NaOH, ~140 °C)", BASE,
     [r"$REPO_4 + 3\,NaOH \rightarrow RE(OH)_3\downarrow + Na_3PO_4$",
      "the base takes the phosphate away: the metal stays as a\nsolid hydroxide cake; thorium stays in the cake too"],
     "sodium phosphate solution,\nwashed out (a saleable by-product)"),
    (10.9, "2. Leach", "hydrochloric acid", ACID,
     [r"$RE(OH)_3 + 3\,HCl \rightarrow RECl_3 + 3\,H_2O$",
      "the acid puts the metal into solution:\nthis chloride liquor is the circuit's feed"],
     "thorium (and uranium),\ntaken out before the circuit;\nSm and heavier too (in the model)"),
    (8.4, "3. Load", "extraction section:\nlow acid, saponified\norganic (PC88A)", ORG,
     [r"$RE^{3+} + 3\,H_2L_2(org) \rightleftharpoons RE(HL_2)_3(org) + 3\,H^+$",
      "metal moves into the organic and releases acid; the\nsaponified extractant (pre-neutralized with NaOH) soaks\nthat acid up. Pr and Nd load more than La and Ce"],
     "raffinate: La and Ce chlorides\n(out of stage 0)"),
    (5.9, "4. Scrub", "scrub section:\nNdPr liquor sent back", ORG,
     [r"$Pr^{3+} + Ce(HL_2)_3(org) \rightarrow Pr(HL_2)_3(org) + Ce^{3+}$",
      "the same reaction, as an exchange: product metal in the\nscrub liquor pushes the La and Ce the organic carried\nalong back into the aqueous (Nd does the same)"],
     "nothing leaves: La and Ce flow\nback toward the raffinate"),
    (3.4, "5. Strip", "strip section:\nstrong acid (2 M HCl)", ACID,
     [r"$RE(HL_2)_3(org) + 3\,H^+ \rightarrow RE^{3+} + 3\,H_2L_2(org)$",
      "the reaction run backwards: lots of acid pushes the\nmetal out of the organic, and the empty organic\ngoes round again"],
     "NdPr chloride: the product"),
]
for y, title, cond, edge, (eq, note), side in rows:
    box(ax, X0, y, W, H, title, cond, edge)
    text(ax, 3.7, y + 1.25, eq, 10.5, INK)
    text(ax, 3.7, y + 0.5, note, 7.8, INK2)
    arrow(ax, (10.9, y + 0.88), (11.6, y + 0.88), OUTC)
    text(ax, 11.75, y + 0.88, side, 7.8, INK)

# what passes from one step to the next
between = [(13.4, "monazite concentrate"), (10.9, "hydroxide cake"), (8.4, "chloride liquor (the feed)"),
           (5.9, "loaded organic"), (3.4, "clean loaded organic")]
arrow(ax, (X0 + W / 2, 15.45), (X0 + W / 2, 13.4 + H), INK2)
for (y, what), (y_next, _) in zip(between[1:], between[:-1]):
    arrow(ax, (X0 + W / 2, y_next), (X0 + W / 2, y + H), INK2)
    text(ax, X0 + W / 2 + 0.12, (y_next + y + H) / 2, what, 7.4, MUTED)
text(ax, X0 + W / 2 + 0.12, 15.35, "monazite concentrate: REPO$_4$ (+ Th, U)", 7.4, MUTED)

# the organic loop and the bracket for the circuit
ax.plot([3.45, 3.45], [3.5, 10.05], color=ORG, lw=2.2, alpha=0.6)
text(ax, 3.55, 10.25, "the sx-cascade-control circuit", 7.6, ORG, weight="bold")
ax.plot([3.45, 3.45], [11.0, 15.0], color=MUTED, lw=1.0, ls=":")
text(ax, 3.55, 15.15, "before the circuit (textbook, not in the project files)", 7.6, MUTED)

# memory hooks
hooks = [
    ("Base takes the phosphate away; acid puts the metal in solution.", BASE),
    ("Inside the circuit, acid sets the direction: little acid, metal goes into the organic;\n"
     "lots of acid, it comes back out. Every metal ion moved trades places with three H$^+$.", ACID),
    ("Pr and Nd stay together the whole way: they leave as one NdPr product.", ORG),
]
yy = 2.35
for s, c in hooks:
    ax.add_patch(FancyBboxPatch((0.25, yy - 0.28), 0.22, 0.22, boxstyle="round,pad=0.01", fc=c, ec=c))
    text(ax, 0.65, yy - 0.17, s, 8.8, INK)
    yy -= 0.85 if "\n" in s else 0.6
text(ax, 7.5, 16.15, "From monazite to an NdPr product: the chemistry, step by step", 12, INK, "center")
fig.savefig(OUT / "chemistry_route.png", dpi=160, bbox_inches="tight", pad_inches=0.3)
print("wrote", OUT / "chemistry_route.png")
