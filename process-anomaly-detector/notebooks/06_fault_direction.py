# %% [markdown]
# # Step 6 — what the first eigenvector physically is
#
# An eigenvector is a direction through the 14-dimensional cloud of sensor
# readings; its eigenvalue is how far the data spreads along it. Step 5 showed
# that over a full life one direction carries 62.5% of all the variance.
#
# That direction is not an abstraction. It is the combination of sensor
# movements that HPC wear produces: this much hotter, that much lower pressure,
# in fixed proportion. It is the fault signature, and it can be read off and
# checked against the physics.
#
# It also lets us test the hypothesis left open in step 4: PCA fitted on
# *healthy* data helped, even though healthy data has no correlation structure.
# The suggested reason was that the baseline window is not perfectly healthy -
# degradation has already begun by cycle 20 - so the fault direction picks up
# slightly more variance than the rest and floats into the leading components.
#
# If that is true, the leading healthy eigenvectors should point along the
# fault direction more than the trailing ones. That is a measurable claim.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cmapss
import health
import models
import plotting
from plotting import plt

df = cmapss.add_rul(cmapss.load("FD001", "train"))
live = [s for s in cmapss.SENSOR_COLS if s not in cmapss.constant_sensors(df)]
trend = pd.Series({s: df.groupby("unit").apply(
    lambda g, c=s: g[c].corr(g["rul"]), include_groups=False).median() for s in live})
SENSORS = trend[abs(trend) > 0.5].index.tolist()

z = models.z_frame(df, SENSORS, health.baseline_stats(df, SENSORS))
z_all = z.to_numpy()
z_healthy = z_all[(df.cycle <= health.BASELINE_CYCLES).to_numpy()]

# %% [markdown]
# ## Reading the fault signature
#
# The loadings are the eigenvector's components: how much each sensor
# contributes, and in which direction. Sign matters - it says which sensors
# rise together and which fall as the others rise.

# %%
ev_full, vec_full = np.linalg.eigh(np.cov(z_all, rowvar=False))
order = np.argsort(ev_full)[::-1]
ev_full, vec_full = ev_full[order], vec_full[:, order]
fault_dir = vec_full[:, 0]

# fix the arbitrary sign so the direction points the way degradation goes
if np.dot(fault_dir, np.sign([trend[s] for s in SENSORS])) > 0:
    fault_dir = -fault_dir

loadings = pd.DataFrame({
    "loading": fault_dir,
    "trend with RUL": [trend[s] for s in SENSORS],
    "sensor": [cmapss.SENSOR_NAMES[s] for s in SENSORS],
}, index=SENSORS).sort_values("loading", key=abs, ascending=False)

print(f"first component carries {ev_full[0] / ev_full.sum():.1%} of full-life variance\n")
print("FAULT SIGNATURE - how each sensor moves as the engine wears:\n")
print(loadings.round(3).to_string())
print("""
Positive loading = rises as the engine degrades. Negative = falls.
The 'trend with RUL' column is the independent check from step 1: RUL counts
DOWN to failure, so a sensor that rises with wear must correlate negatively
with RUL. Every sign should be opposite. If they are, two different methods
agree on the same physical picture.
""")
agree = (np.sign(loadings.loading) == -np.sign(loadings["trend with RUL"])).all()
print(f"all signs consistent: {agree}")

# %% [markdown]
# ## Testing the step 4 hypothesis
#
# Project each healthy eigenvector onto the fault direction. |cos| near 1 means
# that component points along the fault; near 0 means it is orthogonal to it,
# carrying no degradation information at all.

# %%
ev_h, vec_h = np.linalg.eigh(np.cov(z_healthy, rowvar=False))
order_h = np.argsort(ev_h)[::-1]
ev_h, vec_h = ev_h[order_h], vec_h[:, order_h]

align = np.abs(vec_h.T @ fault_dir)
tab = pd.DataFrame({
    "healthy eigenvalue": ev_h,
    "|cos| with fault direction": align,
}, index=[f"comp {i+1}" for i in range(len(SENSORS))])
print(tab.round(3).to_string())

top6, rest = align[:6], align[6:]
print(f"\nmean alignment, components 1-6:  {top6.mean():.3f}")
print(f"mean alignment, components 7-14: {rest.mean():.3f}")
print(f"random 14-dim vectors would average about {np.sqrt(2/(np.pi*len(SENSORS))):.3f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

ax = axes[0]
colors = ["firebrick" if v > 0 else "steelblue" for v in loadings.loading]
ax.barh(range(len(loadings)), loadings.loading, color=colors)
ax.set_yticks(range(len(loadings)))
ax.set_yticklabels([f"{i}  {n.split(' - ')[0]}" for i, n in
                    zip(loadings.index, loadings.sensor)], fontsize=8)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("loading on the fault direction")
ax.set_title("The fault signature: red rises with wear, blue falls")
ax.grid(alpha=0.3, axis="x")

ax = axes[1]
ax.bar(range(1, len(SENSORS) + 1), align, color=["seagreen"] * 6 + ["lightgrey"] * 8)
ax.axhline(np.sqrt(2 / (np.pi * len(SENSORS))), color="crimson", linestyle="--",
           linewidth=1.2, label="random direction would give this")
ax.set_xlabel("healthy component (ordered by eigenvalue)")
ax.set_ylabel("|cos| with the fault direction")
ax.set_title("Do the leading healthy components point along the fault?")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8)

fig.tight_layout()
plotting.save(fig, "06_fault_direction.png")

# %%
plotting.show()
