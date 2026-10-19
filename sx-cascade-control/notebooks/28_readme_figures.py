"""Three plain figures for the README, on the main case (configs/ndpr_main.toml).

  1. readme_margin.png    what running with margin buys and costs, for four stage counts (notebook 26):
                          the band of scrub flow that keeps the product on spec, and the reagent.
  2. readme_feed_switch.png   one upset over time: the feed changing between two published deposits,
                          on the report's plant, with nobody touching it and under the controller.
  3. readme_twenty_plants.png the twenty realistic plants of the main-case seed study: per upset, the
                          median NdPr lost or made off spec, uncontrolled and controlled, with the range.

Figures 1 and 3 are drawn from results already saved (runs/operating_margin_summary.csv from
notebook 26, runs/seed_spread_ndpr_main_s*-*.csv from notebook 14); figure 2 runs its two
simulations here.

Run:  pixi run python notebooks/28_readme_figures.py
Writes the three figures to figures/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotting import FIGURES, INK, INK2, MUTED, SLOT, style         # noqa: E402
from src.runs import RunSpec, execute                                   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
style()

# ---- 1. what margin buys, by stage count (notebook 26) ----
# Each of the twenty realistic plants retuned to its own cheapest flows that meet the policy's targets;
# the band of scrub flow that keeps it on spec measured there (median over the plants), and the reagent.
m = pd.read_csv(HERE / "runs" / "operating_margin_summary.csv")
layouts = ["8+20", "10+26", "12+32", "14+38"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.6))
x = np.arange(len(layouts))
for off, policy, color, label in [(-0.2, "edge", SLOT[3], "cheapest flows (99.5 % / 95 %)"),
                                  (0.2, "margin", SLOT[0], "with margin (99.8 % / 97 %)")]:
    rows = [m[m.case == f"{lay}_130L_{policy}"].iloc[0] for lay in layouts]
    ax1.bar(x + off, [r["median_window_pct"] for r in rows], width=0.38, color=color, label=label)
    ax2.bar(x + off, [r["median_reagent"] for r in rows], width=0.38, color=color, label=label)
for ax in (ax1, ax2):
    ax.set_xticks(x, [f"{lay} stages" for lay in layouts])
    ax.set_xlabel("extraction + scrub stages")
ax1.set_ylabel("on-spec band, % of scrub flow")
ax1.set_title("Room before the product goes off spec", loc="left", color=INK, fontsize=9)
ax2.set_ylabel("reagent, mol/h")
ax2.set_title("What it costs in reagent", loc="left", color=INK, fontsize=9)
ax2.set_ylim(0, None)
ax1.legend(frameon=False, fontsize=8, loc="upper left")
fig.text(0.01, -0.06, "Twenty plants drawn inside the published separation-factor ranges, each retuned to its own "
         "cheapest flows that meet the targets; medians (notebook 26).", fontsize=7.5, color=MUTED)
fig.savefig(FIGURES / "readme_margin.png", bbox_inches="tight")
plt.close(fig)
print("wrote readme_margin.png")

# ---- 2. one upset over time ----
fig, axes = plt.subplots(2, 1, figsize=(8.0, 4.6), sharex=True)
for ctl, color, label in [("open_loop_level_loops", SLOT[3], "nobody touching it"),
                          ("mpc_level_loops", SLOT[0], "under the controller")]:
    r = execute(RunSpec(config="ndpr_main", plant="perturbed", controller=ctl, upset="feed_switch", hours=80.0))
    axes[0].plot(r.t, 100 * np.asarray(r.series["purity"]), color=color, lw=1.8, label=label)
    axes[1].plot(r.t, 100 * np.asarray(r.series["recovery"]), color=color, lw=1.8)
axes[0].axhline(99.0, color=MUTED, lw=0.8, ls=":")
axes[1].axhline(95.0, color=MUTED, lw=0.8, ls=":")
axes[0].set_ylabel("purity, %")
axes[1].set_ylabel("recovery, %")
axes[1].set_xlabel("hours (the feed changes at 5 h, over one hour)")
axes[0].legend(frameon=False, fontsize=8)
axes[0].set_title("The feed switches from one published deposit to another (NdPr 21 % to 41 % of the metal)",
                  loc="left", color=INK, fontsize=9)
fig.savefig(FIGURES / "readme_feed_switch.png", bbox_inches="tight")
plt.close(fig)
print("wrote readme_feed_switch.png")

# ---- 3. twenty plants ----
files = sorted((HERE / "runs").glob("seed_spread_ndpr_main_s*-*.csv"))
d = pd.concat([pd.read_csv(f) for f in files])
d["cost"] = d["lost_excess_mol"].clip(lower=0) + d["off_spec_mol"]
names = {"feed_switch": "feed changes deposit", "feed_step": "stronger feed", "feed_acid": "more acid in the feed",
         "reflux_acid": "scrub acid too strong", "flowmeter_bias": "organic flowmeter 5 % low",
         "feed_interrupt": "feed pump stops 2 h", "interface_drift": "settler level drifts",
         "efficiency_loss": "two tired mixers"}
fig, ax = plt.subplots(figsize=(8.0, 4.2))
y = np.arange(len(names))
for k, (ctl, color, label, off) in enumerate([("open_loop_level_loops", SLOT[3], "nobody touching it", 0.18),
                                              ("mpc_level_loops", SLOT[0], "under the controller", -0.18)]):
    med = [d[(d.upset == u) & (d.controller == ctl)]["cost"].median() + 1 for u in names]
    lo = [d[(d.upset == u) & (d.controller == ctl)]["cost"].min() + 1 for u in names]
    hi = [d[(d.upset == u) & (d.controller == ctl)]["cost"].max() + 1 for u in names]
    ax.barh(y + off, med, height=0.34, color=color, label=label)
    ax.hlines(y + off, lo, hi, color=INK2, lw=0.8)
ax.set_xscale("log")
ax.set_yticks(y, list(names.values()))
ax.invert_yaxis()
ax.set_xlabel("NdPr lost or made off spec in 80 h, mol (+1, log scale): median bar, line from best to worst plant")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Twenty plants drawn inside the published separation-factor ranges, eight upsets each",
             loc="left", color=INK, fontsize=9)
fig.savefig(FIGURES / "readme_twenty_plants.png", bbox_inches="tight")
plt.close(fig)
print("wrote readme_twenty_plants.png")
