# %% [markdown]
# # Step 7 — what the scores do after the alarm
#
# Warning time cannot separate these detectors, so what distinguishes them is
# what their score does once it has crossed the alarm. This matters for ranking
# machines by severity, and for anything that needs to know how bad things are
# rather than only that something is wrong.
#
# First, one dot per flight: how isolated the forest thinks each reading is,
# against how much life the engine actually had left when it was taken.
#
# The score is built like this: about 300 trees are grown from the healthy
# training data using random cuts. To score a reading, drop it down each tree
# and count the cuts needed before it sits on its own, then average over the
# trees. A reading in a sparse neighbourhood gets isolated in few cuts. That is
# what the score measures - sparseness of the neighbourhood, not distance from
# a centre. The two usually agree but they are not the same quantity.

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
units = np.sort(df.unit.unique())

rng = np.random.default_rng(42)
test_u = rng.choice(units, size=30, replace=False)
train_u = np.setdiff1d(units, test_u)
tr = df[df.unit.isin(train_u)].copy()
te = df[df.unit.isin(test_u)].copy()

z_tr = models.z_frame(tr, SENSORS, health.baseline_stats(tr, SENSORS)).to_numpy()
z_te = models.z_frame(te, SENSORS, health.baseline_stats(te, SENSORS)).to_numpy()
healthy = z_tr[(tr.cycle <= health.BASELINE_CYCLES).to_numpy()]

HEAVY = 21   # display only; detection still uses health.SMOOTH_WINDOW = 5

te = te.copy()
te["iforest"] = models.isolation_forest_scores(healthy, z_te)
te["iforest_smooth"] = te.groupby("unit")["iforest"].transform(
    lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
# heavier, centred average to look for a change of slope rather than to alarm on
te["iforest_heavy"] = te.groupby("unit")["iforest"].transform(
    lambda x: x.rolling(HEAVY, min_periods=5, center=True).mean())
# slope in score per flight, over the heavily smoothed curve
te["slope"] = te.groupby("unit")["iforest_heavy"].transform(
    lambda x: x.diff().rolling(HEAVY, min_periods=5, center=True).mean())

tr_scored = tr.copy()
tr_scored["hi_smooth"] = pd.Series(
    models.isolation_forest_scores(healthy, z_tr), index=tr.index
).groupby(tr.unit).transform(lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
threshold, k = health.choose_threshold(tr_scored)
print(f"alarm level = healthy mean + {k:g} sd = {threshold:.3f}")

# %% [markdown]
# ## Five engines, every flight
#
# Life left runs right to left, so failure is at the left edge of each panel and
# time flows leftwards. Faint dots are single flights, the line is the same
# thing smoothed over five.

# %%
show = te.unit.unique()[:5]
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

ax = axes[0]
for u in show:
    g = te[te.unit == u].sort_values("rul", ascending=False)
    p = ax.plot(g.rul, g.iforest, ".", markersize=2.5, alpha=0.3)
    ax.plot(g.rul, g.iforest_heavy, linewidth=2.0, color=p[0].get_color(),
            label=f"unit {u}")
ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.3,
           label="alarm level")
ax.invert_xaxis()
ax.set_xlabel("flights of life left  (failure at 0, on the left)")
ax.set_ylabel("isolation score  (higher = more isolated)")
ax.set_title(f"Five held-out engines, {HEAVY}-flight average")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# %% [markdown]
# ## All 30 held-out engines at once
#
# Every reading from every engine, so the shape of the relationship is visible
# rather than five particular histories.

# %%
ax = axes[1]
ax.scatter(te.rul, te.iforest, s=3, alpha=0.12, color="mediumpurple")
band = te.groupby(pd.cut(te.rul, bins=np.arange(0, 260, 10), labels=False))["iforest"]
centres = np.arange(5, 255, 10)[: len(band.median())]
ax.plot(centres, band.median(), color="black", linewidth=2, label="median")
ax.fill_between(centres, band.quantile(0.1), band.quantile(0.9),
                color="black", alpha=0.12, label="10th to 90th percentile")
ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.3,
           label="alarm level")
ax.invert_xaxis()
ax.set_xlabel("flights of life left  (failure at 0, on the left)")
ax.set_ylabel("isolation score")
ax.set_title(f"All {te.unit.nunique()} held-out engines, {len(te):,} readings")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

# %% [markdown]
# ## Is there a change of slope before failure?
#
# A curve bending is hard to judge by eye, so the third panel plots the slope
# itself: how much the score moves per flight.
#
# The Isolation Forest score is bounded, since path length in a tree cannot
# shrink below a single split, so any flattening near failure could be the
# detector running out of range rather than the engine settling down. The
# squared index is unbounded and reaches 157 times its healthy level, hence
# plotting both slopes together separates the two explanations. Each is scaled
# by its own peak so the shapes can be compared.

# %%
te["sq"] = (z_te ** 2).mean(axis=1)
te["lin"] = np.abs(z_te).mean(axis=1)

# Differentiate the binned medians rather than smoothing each engine and then
# differencing. A centred rolling window cannot see past the last flight, so it
# averages only the earlier half near failure and pushes the slope down exactly
# where the interesting behaviour is. With that method the squared index appeared
# to peak 15 flights out and then fall; it does not, it is still climbing at
# failure. Each bin already holds about 300 readings, so the median is smooth
# enough to differentiate directly.
bins = np.arange(0, 260, 10)
grp = te.groupby(pd.cut(te.rul, bins=bins, labels=False))
centres = np.arange(5, 255, 10)[: len(grp["iforest"].median())]


def slope_of(col):
    """d(score)/d(flights towards failure), from the binned medians."""
    med = grp[col].median().values
    return pd.Series(-np.gradient(med, centres), index=centres)


curves = {
    "Isolation Forest (bounded)": (slope_of("iforest"), "mediumpurple", "s-"),
    "health index, mean |z| (linear)": (slope_of("lin"), "steelblue", "^-"),
    "squared index, mean z^2": (slope_of("sq"), "darkorange", "o-"),
}

ax = axes[2]
for name, (series, colour, style) in curves.items():
    ax.plot(centres, series / series.max(), style, markersize=4, linewidth=2,
            color=colour, label=name)
ax.axhline(0, color="grey", linewidth=1)
ax.invert_xaxis()
ax.set_xlabel("flights of life left  (failure at 0, on the left)")
ax.set_ylabel("slope, scaled to its own peak")
ax.set_title("Rate of change: bounded, linear and squared")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

print("\nwhere each score's slope peaks:")
for name, (series, _, _) in curves.items():
    print(f"  {name:>34}: {centres[np.argmax(series)]:3d} flights before failure")

print("""
The forest slope peaks about 65 flights before failure and then falls away,
which looks like the degradation easing off but is not: the score is bounded,
path length cannot shrink below one split, and it has simply run out of range.

The two unbounded scores show what the engines are actually doing. |z| grows
with the deviation itself, so its slope is the degradation rate, and that rate
climbs through mid-life and then holds roughly flat over the last 15 to 20
flights. z^2 grows with the square, so its slope carries an extra factor of the
deviation which is still increasing, hence it keeps climbing to the end.

So the engines do not decelerate, but they do not keep accelerating either. The
rate rises and then holds, and everything past that in the forest curve belongs
to the detector.
""")

fig.tight_layout()
plotting.save(fig, "07_score_behaviour.png")

# %%
prof = te.groupby(pd.cut(te.rul, [0, 25, 50, 75, 100, 150, 200, 300]),
                  observed=True).agg(
    score=("iforest_heavy", "median"), slope=("slope", "median")).round(4)
prof["slope per 100 flights"] = (100 * prof.slope).round(2)
print("\nscore and its slope by remaining life:")
print(prof[["score", "slope per 100 flights"]].to_string())

# %%
q = te.groupby(pd.cut(te.rul, [0, 25, 50, 100, 150, 300]), observed=True)["iforest"]
print("\nisolation score by remaining life:")
print(pd.DataFrame({"median": q.median().round(3),
                    "10th pct": q.quantile(0.1).round(3),
                    "90th pct": q.quantile(0.9).round(3),
                    "readings": q.size()}).to_string())
print(f"\nalarm level {threshold:.3f}")

# %% [markdown]
# ## The three signals themselves
#
# The slopes above say how each score changes, but not what it looks like. Here
# is each score in its own units against remaining life, with no normalising and
# no log axis, since both of those hide exactly the difference worth seeing.
# Median over all 30 held-out engines, with the 10th to 90th percentile band,
# and each detector's own alarm level marked.

# %%
signals = [
    ("Isolation Forest", "iforest", "mediumpurple", "path length, bounded"),
    ("health index, mean |z|", "lin", "steelblue", "wobbles from own normal"),
    ("squared index, mean z^2", "sq", "darkorange", "wobbles squared"),
]

# alarm level for each, set on the training engines exactly as in step 3
z_tr_arr = z_tr
train_scores = {
    "iforest": models.isolation_forest_scores(healthy, z_tr_arr),
    "lin": np.abs(z_tr_arr).mean(axis=1),
    "sq": (z_tr_arr ** 2).mean(axis=1),
}

fig2, axes2 = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
bins = np.arange(0, 260, 10)
centres = np.arange(5, 255, 10)

for ax, (label, col, colour, units) in zip(axes2, signals):
    tr_scored = tr.copy()
    tr_scored["hi_smooth"] = pd.Series(train_scores[col], index=tr.index).groupby(
        tr.unit).transform(lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
    thr, _ = health.choose_threshold(tr_scored)

    band = te.groupby(pd.cut(te.rul, bins=bins, labels=False))[col]
    c = centres[: len(band.median())]
    ax.scatter(te.rul, te[col], s=2, alpha=0.06, color=colour)
    ax.plot(c, band.median(), color=colour, linewidth=2.5, label="median")
    ax.fill_between(c, band.quantile(0.1), band.quantile(0.9),
                    color=colour, alpha=0.18, label="10th to 90th percentile")
    ax.axhline(thr, color="crimson", linestyle="--", linewidth=1.2,
               label=f"alarm level ({thr:.2f})")
    ax.set_ylabel(f"{label}\n({units})", fontsize=9)
    ax.set_ylim(bottom=0 if col != "iforest" else None)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")

    top = band.median().iloc[0]
    base = band.median().iloc[-1]
    ax.set_title(f"healthy {base:.2f}  ->  at failure {top:.2f}   "
                 f"({top / base:.0f}x)", fontsize=9, loc="right")

axes2[0].invert_xaxis()
axes2[-1].set_xlabel("flights of life left  (failure at 0, on the right)")
fig2.suptitle("The three scores in their own units, 30 held-out engines", y=0.997)
fig2.tight_layout()
plotting.save(fig2, "07b_three_signals.png")

# %%
plotting.show()
