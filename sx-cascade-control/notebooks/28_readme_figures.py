"""Two figures for the README's results, on the main case (configs/ndpr_main.toml).

  1. readme_twenty_plants.png   the controller on the twenty plants drawn inside the published
                          separation-factor ranges, through the feed switch to the richer deposit: purity
                          and recovery over time, one line per plant (notebook 40's traces).
  2. readme_where_to_measure.png  how much a 1 % change in the Pr/(La+Ce) separation factor moves the
                          Pr reading at each stage (notebook 39).

Both are drawn from results already saved: runs/twenty_plants_traces_ndpr_main.npz (notebook 40) and
runs/sensor_sensitivity_ndpr_main.csv (notebook 39). The README's first picture is a recording of the
workbench (figures/workbench_feed_switch.gif), not made here.

Run:  pixi run python notebooks/28_readme_figures.py
Writes the two figures to figures/.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotting import FIGURES, GRID, INK, INK2, MUTED, SLOT, style   # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RUNS = HERE / "runs"
CONTROLLER = SLOT[0]
SECTIONS = [(0, 8, "extraction"), (8, 28, "scrub"), (28, 32, "strip")]      # the main case's 8 + 20 + 4
style()


def sections(ax, label=True):
    """Shade the scrub and name the three sections above the axes."""
    for k, (first, end, name) in enumerate(SECTIONS):
        if k == 1:
            ax.axvspan(first - 0.5, end - 0.5, color=GRID, alpha=0.5, lw=0)
        if label:
            ax.text((first + end - 1) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=MUTED, fontsize=8)


# ---- 1. the controller on twenty plants ----
traces = np.load(RUNS / "twenty_plants_traces_ndpr_main.npz")
seeds = sorted({int(k.split("_")[1]) for k in traces.files})
fig, (top, low) = plt.subplots(2, 1, figsize=(10.0, 5.4), sharex=True, gridspec_kw={"hspace": 0.18})
below_spec, back_by = 0, []
for seed in seeds:
    t = traces[f"t_{seed}"]
    purity, recovery = 100 * traces[f"purity_{seed}"], 100 * traces[f"recovery_{seed}"]
    top.plot(t, purity, color=CONTROLLER, lw=1.0, alpha=0.55)
    low.plot(t, recovery, color=CONTROLLER, lw=1.0, alpha=0.55)
    below_spec += int(purity.min() < 99.0)
    after = (t > 6.0) & (recovery >= 95.0)
    back_by.append(float(t[after][0]) if after.any() else np.nan)
top.axhline(99.0, color=INK2, lw=0.9, ls="--")
top.text(80.5, 99.0, "spec", va="center", fontsize=8, color=INK2)
low.axhline(97.0, color=INK2, lw=0.9, ls="--")
low.text(80.5, 97.0, "target", va="center", fontsize=8, color=INK2)
for ax in (top, low):
    ax.axvline(5.0, color=MUTED, lw=0.8, ls=":")
top.text(5.4, 1.0, "feed switches", transform=top.get_xaxis_transform(), fontsize=8, color=MUTED, va="top")
top.set_ylim(98.0, 100.1)
top.set_ylabel("purity, %")
low.set_ylim(40, 115)
low.set_ylabel("recovery, %")
low.set_xlabel("hours")
low.set_xlim(0, 80)
top.set_title("The same controller on twenty plants, each one different from the model it plans on",
              loc="left", color=INK, fontsize=9.5)
fig.text(0.01, -0.02,
         "The feed switches to a richer deposit at 5 h (NdPr from 21 % to 41 % of the metal). One line per plant: "
         "the main case's controller over level loops, seeing only the product, raffinate and feed\nanalysers "
         "(as in the workbench run above), on plants drawn inside the published separation-factor ranges (seeds "
         "1-20). Traces: notebook 40.\nRecovery is the NdPr going out in the product over the NdPr coming in, at "
         "that moment; it goes above 100 % while the cascade gives up metal it was holding.",
         fontsize=7.5, color=MUTED, va="top")
fig.savefig(FIGURES / "readme_twenty_plants.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote readme_twenty_plants.png ({below_spec} of {len(seeds)} dip below spec; recovery back above 95 % "
      f"by {np.nanmedian(back_by):.1f} h median, {np.nanmax(back_by):.1f} h at the latest)")


# ---- 2. where to measure ----
s = pd.read_csv(RUNS / "sensor_sensitivity_ndpr_main.csv")
pr = s[(s.phase == "aqueous") & (s.element == "Pr")].sort_values("stage")
fig, ax = plt.subplots(figsize=(10.0, 3.4))
sections(ax)
ax.bar(pr["stage"], pr["beta Pr/(La+Ce)"].abs(), width=0.75, color=CONTROLLER, zorder=3)
peak = pr.loc[pr["beta Pr/(La+Ce)"].abs().idxmax()]
extraction_max = pr[pr.section == "extraction"]["beta Pr/(La+Ce)"].abs().max()
ax.annotate(f"stage {int(peak.stage)}: {abs(peak['beta Pr/(La+Ce)']):.0f} %", (peak.stage, abs(peak["beta Pr/(La+Ce)"])),
            xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2, ha="center", va="bottom")
ax.annotate(f"extraction: {extraction_max:.0f} % or less", (3.5, extraction_max), xytext=(0, 14),
            textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
ax.set_xlim(-0.8, 31.8)
ax.set_ylim(0, 108)
ax.set_xlabel("stage (raffinate leaves at stage 0)")
ax.set_ylabel("change in the Pr reading, %")
ax.set_title("How much a 1 % change in the Pr/(La+Ce) separation factor moves the aqueous Pr reading, stage by "
             "stage", loc="left", color=INK, fontsize=9.5, pad=16)
fig.text(0.01, -0.04,
         "On the main case at its operating point (notebook 39). A reading where the profile is steep tells the "
         "most about the plant's chemistry; one at the extraction end tells almost nothing.",
         fontsize=7.5, color=MUTED, va="top")
fig.savefig(FIGURES / "readme_where_to_measure.png", bbox_inches="tight")
plt.close(fig)
print(f"wrote readme_where_to_measure.png (peak stage {int(peak.stage)}, {peak['beta Pr/(La+Ce)']:.1f}; "
      f"extraction max {extraction_max:.1f})")
