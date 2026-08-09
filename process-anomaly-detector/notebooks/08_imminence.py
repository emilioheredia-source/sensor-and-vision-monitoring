# %% [markdown]
# # Step 8 — telling an operator that failure is close
#
# The alarm says something is drifting, roughly 100 flights before failure. That
# is enough to plan work, but it does not answer the question a plant actually
# asks next, which is whether this machine can wait until the next shutdown.
#
# Here the question is put directly to the data. Given the health index right
# now, how much life is left, and is there a level above which failure is
# reliably close?

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

rng = np.random.default_rng(42)
units = np.sort(df.unit.unique())
test_u = rng.choice(units, size=30, replace=False)
tr = df[df.unit.isin(np.setdiff1d(units, test_u))].copy()
te = df[df.unit.isin(test_u)].copy()

z_tr = models.z_frame(tr, SENSORS, health.baseline_stats(tr, SENSORS)).to_numpy()
z_te = models.z_frame(te, SENSORS, health.baseline_stats(te, SENSORS)).to_numpy()

tr_scored = health.health_index(tr, SENSORS)
te = health.health_index(te, SENSORS)
alarm, k = health.choose_threshold(tr_scored)
print(f"alarm level, set on the training engines: {alarm:.2f} (mean + {k:g} sd)")

# %% [markdown]
# ## What the score says about the life left
#
# Only readings past the alarm are considered, since before that the operator
# has no reason to be asking.

# %%
past = te[te.hi_smooth > alarm]
edges = past.hi_smooth.quantile([0, .2, .4, .6, .8, .9, .95, 1.0]).values

rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    band = past[(past.hi_smooth >= lo) & (past.hi_smooth < hi)]
    if len(band) < 30:
        continue
    rows.append({"from": lo, "to": hi, "readings": len(band),
                 "median RUL": band.rul.median(),
                 "10th": band.rul.quantile(.1), "90th": band.rul.quantile(.9),
                 "fails within 25": 100 * (band.rul < 25).mean()})
bands = pd.DataFrame(rows).round(1)
print("\nhealth index band -> flights of life left:\n")
print(bands.to_string(index=False))

# %% [markdown]
# ## As a decision rule
#
# For each candidate level: of all readings above it, what fraction belong to an
# engine that fails within 10, 25 or 50 flights, and how many engines ever reach
# that level before failing. The second number is the cost. A very high level is
# almost always right and most engines never get there.

# %%
rule = []
for t in [1.5, 2, 3, 4, 5, 6, 8, 10]:
    sub = te[te.hi_smooth > t]
    if len(sub) < 20:
        continue
    rule.append({"level": t, "readings": len(sub),
                 "fails <10": 100 * (sub.rul < 10).mean(),
                 "fails <25": 100 * (sub.rul < 25).mean(),
                 "fails <50": 100 * (sub.rul < 50).mean(),
                 "engines reaching it": sub.unit.nunique()})
rule = pd.DataFrame(rule)
rule[rule.columns.drop("level")] = rule[rule.columns.drop("level")].round(0)
print(f"\ndecision rule, out of {te.unit.nunique()} held-out engines:\n")
print(rule.to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

ax = axes[0]
ax.scatter(te.hi_smooth, te.rul, s=3, alpha=0.12, color="steelblue")
mid = (bands["from"] + bands["to"]) / 2
ax.plot(mid, bands["median RUL"], "o-", color="black", linewidth=2, label="median")
ax.fill_between(mid, bands["10th"], bands["90th"], color="black", alpha=0.12,
                label="10th to 90th percentile")
ax.axvline(alarm, color="crimson", linestyle="--", linewidth=1.2,
           label=f"alarm ({alarm:.2f})")
ax.axhline(25, color="seagreen", linestyle=":", linewidth=1.4, label="25 flights left")
ax.set_xscale("log")
ax.set_xlabel("health index now (log scale)")
ax.set_ylabel("flights of life actually left")
ax.set_ylim(0, 200)
ax.set_title("The score knows roughly how close failure is")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(rule.level, rule["fails <25"], "o-", linewidth=2, color="darkorange",
        label="of readings above this level,\n% that fail within 25 flights")
ax.plot(rule.level, 100 * rule["engines reaching it"] / te.unit.nunique(), "s-",
        linewidth=2, color="steelblue",
        label="% of engines that ever reach it")
ax.axvline(5, color="seagreen", linestyle=":", linewidth=1.6,
           label="a workable second alarm")
ax.set_xlabel("second-alarm level (health index)")
ax.set_ylabel("per cent")
ax.set_ylim(0, 105)
ax.set_title("Certainty against coverage")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, loc="center left")

fig.tight_layout()
plotting.save(fig, "08_imminence.png")

# %%
five = te[te.hi_smooth > 5]
print(f"""
A second alarm at 5 is the usable compromise. Of the readings above it,
{100*(five.rul<25).mean():.0f}% belong to an engine that fails within 25 flights, and
{five.unit.nunique()} of {te.unit.nunique()} engines reach it before failing. Raising it to 6 makes the
first number 99% and drops the second to 18, so the cost of being more certain
is engines that fail without ever giving the second warning.

What this does not give is a number of flights. In the band just above 5 the
median is around 12 flights but the range runs from 3 to 22, hence "act now" is
reliable while "you have twelve flights" is not. Detection and imminence are
answerable here, remaining-life prediction is not.
""")

# %%
plotting.show()
