# %% [markdown]
# # C-MAPSS FD001 — look at the data first
#
# Before any detection code: what is actually in here, and which sensors carry
# a degradation signal at all? Skipping this step is how people end up training
# a model on a sensor that never moves.
#
# FD001: 100 engines, one operating condition, one fault mode (HPC degradation).
# Every training engine runs to failure, so the last cycle of each unit is its
# failure point.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cmapss
import health
import plotting
from plotting import plt

train = cmapss.add_rul(cmapss.load("FD001", "train"))
print(f"rows: {len(train):,}   engines: {train['unit'].nunique()}")
print(f"cycles per engine: min {train.groupby('unit')['cycle'].max().min()}, "
      f"median {int(train.groupby('unit')['cycle'].max().median())}, "
      f"max {train.groupby('unit')['cycle'].max().max()}")

# %% [markdown]
# ## Which sensors are dead?
#
# A sensor with zero variance tells us nothing. In a plant these are the
# channels that are stuck, unscaled, or not commissioned — worth finding early
# and saying so, rather than feeding them to a model.

# %%
dead = cmapss.constant_sensors(train)
print(f"constant sensors ({len(dead)}): {dead}")

spread = train[cmapss.SENSOR_COLS].std().sort_values()
print("\nstandard deviation, lowest first:")
print(spread.to_string())

live = [s for s in cmapss.SENSOR_COLS if s not in dead]
print(f"\n{len(live)} sensors carry any variation at all")

# %% [markdown]
# ## Which of the living sensors actually trend toward failure?
#
# Variation is not the same as signal. A noisy sensor varies but says nothing
# about wear. The question is whether a sensor drifts *consistently* as an
# engine approaches failure.
#
# Simple, honest test: correlate each sensor against RUL, within each engine,
# then look at the median correlation across all 100 engines. Doing it per
# engine matters — it avoids being fooled by differences between units.

# %%
def per_unit_corr(df: pd.DataFrame, sensor: str) -> float:
    """Median within-engine correlation of a sensor against RUL."""
    corrs = df.groupby("unit").apply(
        lambda g: g[sensor].corr(g["rul"]), include_groups=False
    )
    return corrs.median()

trend = pd.Series(
    {s: per_unit_corr(train, s) for s in live}
).sort_values(key=abs, ascending=False)

print("median within-engine correlation with RUL (|r| descending):")
print(trend.round(3).to_string())

strong = trend[abs(trend) > 0.5]
print(f"\n{len(strong)} sensors with |r| > 0.5: these carry the degradation signal")

# %% [markdown]
# ## Look at them
#
# Numbers are not enough. Plot the strongest sensors across a handful of
# engines, on a common axis of cycles-until-failure, and see whether the drift
# is real and whether it is gradual or late-breaking. That distinction decides
# how much warning any detector can possibly give.

# %%
units = [1, 2, 3, 5, 8]
top = strong.index[:6]

fig, axes = plt.subplots(len(top), 1, figsize=(9, 2.0 * len(top)), sharex=True)
for ax, sensor in zip(axes, top):
    for u in units:
        g = train[train.unit == u]
        ax.plot(-g["rul"], g[sensor], linewidth=0.9, alpha=0.85, label=f"unit {u}")
    ax.set_ylabel(sensor)
    ax.set_title(cmapss.SENSOR_NAMES[sensor], fontsize=8, loc="left", pad=2)
    ax.grid(alpha=0.25)
axes[0].legend(ncol=len(units), fontsize=8, loc="upper left")
axes[-1].set_xlabel("cycles until failure  (0 = failure)")
fig.suptitle("FD001 - strongest-trending sensors, five engines", y=0.998)
fig.tight_layout()
plotting.save(fig, "01_sensor_traces.png")

# %% [markdown]
# ## How much does an engine's healthy baseline vary between units?
#
# This decides the whole approach. If every engine starts in the same place, a
# single global model works. If each engine has its own healthy baseline, the
# detector has to be referenced to *that engine's* early life — which is
# exactly how condition monitoring works in a plant, where no two pumps are
# identical.

# %%
first20 = train[train.cycle <= health.BASELINE_CYCLES].groupby("unit")[list(top)].mean()
summary = pd.DataFrame({
    f"between-unit std (first {health.BASELINE_CYCLES})": first20.std(),
    "within-life std (all cycles)": train[list(top)].std(),
})
summary["ratio"] = (summary.iloc[:, 0] / summary.iloc[:, 1]).round(2)
print(summary.round(4).to_string())
print("""
Observed ratios are about 0.5-0.65: engine-to-engine differences at healthy
baseline are substantial, roughly half the range the sensor covers over a full
life, but smaller than the degradation itself.

So a global threshold is not hopeless, but it wastes signal. Referencing each
engine to its own early-life baseline removes a known nuisance source before
any detection is attempted. That is also how condition monitoring is done in a
plant: no two pumps are identical, so you trend each one against itself.
""")

# %%
plotting.show()
