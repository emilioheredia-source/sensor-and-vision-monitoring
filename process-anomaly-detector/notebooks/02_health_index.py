# %% [markdown]
# # Step 2 — a baseline that has to be beaten
#
# Before any model: build the simplest thing that could work, measure it
# honestly, and make that the number PCA and Isolation Forest have to beat.
#
# The health index is one number per cycle: how far this engine is running from
# **its own** healthy baseline, averaged over the sensors that actually trend.
# No training, no labels. An operator can be told what it means in one sentence.

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

train_all = cmapss.add_rul(cmapss.load("FD001", "train"))

# The 14 sensors that trended with RUL in 01_explore, recomputed here so this
# script stands alone.
live = [s for s in cmapss.SENSOR_COLS if s not in cmapss.constant_sensors(train_all)]
trend = pd.Series({
    s: train_all.groupby("unit").apply(
        lambda g, c=s: g[c].corr(g["rul"]), include_groups=False).median()
    for s in live
})
SENSORS = trend[abs(trend) > 0.5].index.tolist()
print(f"using {len(SENSORS)} trending sensors: {SENSORS}")

# %% [markdown]
# ## Split by engine, never by row
#
# This is the step that decides whether the numbers mean anything. Cycles from
# one engine are not independent of each other — they are the same machine,
# minutes apart. Shuffling rows would put cycle 100 of engine 7 in training and
# cycle 101 in test, and the model would score beautifully by memorising that
# engine rather than by learning degradation.
#
# So: whole engines go to train or to test, never both.

# %%
rng = np.random.default_rng(42)
units = np.sort(train_all.unit.unique())
test_units = rng.choice(units, size=30, replace=False)
train_units = np.setdiff1d(units, test_units)

train = train_all[train_all.unit.isin(train_units)].copy()
test = train_all[train_all.unit.isin(test_units)].copy()
print(f"train: {len(train_units)} engines, {len(train):,} cycles")
print(f"test:  {len(test_units)} engines, {len(test):,} cycles")

# %%
train_hi = health.health_index(train, SENSORS)
test_hi = health.health_index(test, SENSORS)

healthy = train_hi[train_hi.cycle <= health.BASELINE_CYCLES]["hi_smooth"]
print(f"\nHI during the baseline window: median {healthy.median():.2f}, "
      f"95th pct {healthy.quantile(0.95):.2f}")
print("(HI is in units of that engine's own healthy scatter, so ~1 when well)")

# %% [markdown]
# ## Choose the threshold on training engines only
#
# Sweeping the threshold gives a trade-off, not a single answer: alarm early
# and you get more warning but more false alarms on healthy machines; alarm
# late and you are reliable but useless. Which point to pick is an operations
# decision, not a modelling one — so report the curve, not one number.

# %%
# Start below the healthy level (HI ~0.8 when well) so the sweep actually finds
# where false alarms begin. A grid that starts above it reports zero false
# alarms everywhere and hides the trade-off entirely.
grid = np.round(np.arange(0.85, 4.01, 0.05), 2)
sweep = pd.DataFrame([health.evaluate(train_hi, t) for t in grid])
print(sweep[sweep.threshold <= 2.0].to_string(index=False))
print("...")
print(sweep[sweep.threshold > 2.0].iloc[::4].to_string(index=False))

# %% [markdown]
# ## Pick a point, then score it on engines never seen
#
# The alarm level is measured in units of the healthy scatter itself:
# **threshold = healthy mean + k standard deviations**, with the smallest k that
# raises no false alarm on the training engines.
#
# It has to be scale-free. A multiplicative margin ("25% above the highest
# healthy value") is not, and it is unusable across detectors whose scores have
# different ranges: Isolation Forest's score is bounded and can only reach about
# 1.4x its healthy level, while mean z^2 is unbounded and reaches 157x.


# %%
chosen, k_sd = health.choose_threshold(train_hi)
healthy = train_hi[train_hi.cycle <= health.HEALTHY_UNTIL]["hi_smooth"]
print(f"healthy window: mean {healthy.mean():.2f}, sd {healthy.std():.2f}, "
      f"max {healthy.max():.2f}")
print(f"threshold = mean + {k_sd:g} sd = {chosen:.2f}")

result = health.evaluate(test_hi, chosen)
print("\n--- held-out engines ---")
for k, v in result.items():
    print(f"{k:>14}: {v:.2f}")

alarms = health.first_alarm(test_hi, chosen)
print(f"\nlead time across {int(alarms.alarmed.sum())} alarming test engines "
      f"(cycles of warning before failure):")
print(alarms.lead_time.describe().round(1).to_string())

# %% [markdown]
# ## The trade-off curve — this is the deliverable
#
# Not accuracy. Accuracy is meaningless here: most cycles are healthy, so
# calling everything healthy scores well and warns nobody. What a plant asks is
# "how much notice do I get, and how often are you wrong when nothing is
# happening."

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(sweep.early_rate * 100, sweep.lead_time, "o-", linewidth=1.4,
        markersize=3.5, color="steelblue", alpha=0.8,
        label="threshold sweep (training engines)")

# annotate sparsely: the pile-up near zero is unreadable otherwise
for t in [0.85, 0.95, 1.05, 1.2, 1.6, 2.4, 3.5]:
    r = sweep[np.isclose(sweep.threshold, t)]
    if len(r):
        r = r.iloc[0]
        ax.annotate(f"{t:g}", (r.early_rate * 100, r.lead_time),
                    textcoords="offset points", xytext=(7, 3), fontsize=8.5)

ax.scatter([result["early_rate"] * 100], [result["lead_time"]], s=150,
           marker="*", zorder=5, color="crimson",
           label=f"chosen {chosen:.2f} (mean + {k_sd:g} sd), held-out engines")
ax.set_xlabel("engines alarming during the healthy window  (%)")
ax.set_ylabel("median warning before failure  (cycles)")
ax.set_title("Warning time vs false alarms - baseline health index")
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plotting.save(fig, "02_tradeoff.png")

# %% [markdown]
# ## What one engine looks like
#
# The trade-off curve is the result; this is the sanity check. If the index
# does not visibly rise before failure on a single engine, the summary numbers
# are hiding something.

# %%
fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
for ax, u in zip(axes, alarms[alarms.alarmed].unit.head(3)):
    g = test_hi[test_hi.unit == u]
    a = alarms[alarms.unit == u].iloc[0]
    ax.plot(g.cycle, g.hi, linewidth=0.7, alpha=0.4, label="health index")
    ax.plot(g.cycle, g.hi_smooth, linewidth=1.6, label=f"smoothed ({health.SMOOTH_WINDOW})")
    ax.axhline(chosen, color="grey", linestyle="--", linewidth=1, label="threshold")
    ax.axvline(a.alarm_cycle, color="crimson", linewidth=1.2,
               label=f"alarm ({int(a.lead_time)} cycles warning)")
    ax.set_ylabel(f"unit {u}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel("cycle")
fig.suptitle("Health index over life, three held-out engines", y=0.995)
fig.tight_layout()
plotting.save(fig, "02_engine_traces.png")

# %%
plotting.show()
