# %% [markdown]
# # Step 3 — three approaches, compared
#
# All three are fitted on healthy flights only, which is the situation in a
# plant: plenty of running-fine data, almost no labelled failures. All three get
# the same inputs, the same smoothing and the same scoring code, and engines are
# split whole into training and held-out sets, never by row.
#
# **Health index** — average the 14 z-scores. One number: how far this engine is
# running from its own normal. No fitting.
#
# **PCA monitoring** — learn the directions healthy data varies in, then flag
# readings that travel too far along them (T2) or in directions healthy data
# never used (Q).
#
# **Isolation Forest** — split the space with random cuts and see how quickly a
# reading gets isolated. Sparse neighbourhoods are unusual.
#
# The alarm level is set the same way for each: healthy mean plus k standard
# deviations of that detector's own score, smallest k with no false alarm on the
# training engines. Each score is on a different scale, so the rule has to be in
# units of that score's own healthy scatter.
#
# **One split is not a result.** Thirty held-out engines gives a standard
# deviation of 7 to 9 flights between splits, so a single run cannot separate
# detectors that are 3 flights apart. Everything below is averaged over ten
# random splits.

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
SEEDS = [0, 1, 7, 42, 123, 2024, 77, 555, 9001, 31337]
N_COMP = 6

print(f"{len(SENSORS)} trending sensors, baseline window "
      f"{health.BASELINE_CYCLES} flights, {len(SEEDS)} splits of 30 held-out engines")


def smoothed(frame, values):
    out = frame.copy()
    out["s"] = values
    out["hi_smooth"] = out.groupby("unit")["s"].transform(
        lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
    return out


def build(tr, te):
    """The five scores, on one split."""
    z_tr = models.z_frame(tr, SENSORS, health.baseline_stats(tr, SENSORS)).to_numpy()
    z_te = models.z_frame(te, SENSORS, health.baseline_stats(te, SENSORS)).to_numpy()
    healthy = z_tr[(tr.cycle <= health.BASELINE_CYCLES).to_numpy()]
    pca = models.PCAMonitor(n_components=N_COMP).fit(healthy)
    return {
        "health index (mean |z|)":  (np.abs(z_tr).mean(1), np.abs(z_te).mean(1)),
        "squared index (mean z^2)": ((z_tr ** 2).mean(1), (z_te ** 2).mean(1)),
        f"PCA T2 ({N_COMP} comp)":  (pca.scores(z_tr)[0], pca.scores(z_te)[0]),
        f"PCA Q ({N_COMP} comp)":   (pca.scores(z_tr)[1], pca.scores(z_te)[1]),
        "Isolation Forest":         (models.isolation_forest_scores(healthy, z_tr),
                                     models.isolation_forest_scores(healthy, z_te)),
    }


# %%
records = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    test_u = rng.choice(units, size=30, replace=False)
    tr = df[df.unit.isin(np.setdiff1d(units, test_u))].copy()
    te = df[df.unit.isin(test_u)].copy()

    for name, (a, b) in build(tr, te).items():
        train_scored, test_scored = smoothed(tr, a), smoothed(te, b)
        threshold, k = health.choose_threshold(train_scored)
        records.append({"seed": seed, "detector": name, "k": k,
                        **health.evaluate(test_scored, threshold)})

runs = pd.DataFrame(records)

# %%
summary = runs.groupby("detector", sort=False).agg(
    warning=("lead_time", "mean"),
    spread=("lead_time", "std"),
    worst_10pct=("lead_time_p10", "mean"),
    found=("coverage", "mean"),
    false_alarms=("early_rate", "mean"),
).round(1)
summary["found"] = (100 * summary["found"]).round().astype(int).astype(str) + "%"
summary["false_alarms"] = (100 * summary["false_alarms"]).round().astype(int).astype(str) + "%"

print("\nflights of warning before failure, averaged over splits:\n")
print(summary.to_string())

typical_spread = summary.spread.mean()
close = summary[summary.warning > summary.warning.max() - typical_spread]
print(f"""
{len(close)} of {len(summary)} approaches land within one split-to-split standard
deviation ({typical_spread:.0f} flights) of the best. Differences that size
cannot be separated with 30 engines per split, so warning time on its own does
not choose between them.
""")

# %%
fig, ax = plt.subplots(figsize=(8.5, 4.6))
x = np.arange(len(summary))
ax.bar(x - 0.2, summary.warning, width=0.4, yerr=summary.spread, capsize=4,
       color="steelblue", label="warning (error bar = spread across splits)")
ax.bar(x + 0.2, summary.worst_10pct, width=0.4, color="lightsteelblue",
       label="worst 10% of engines")
ax.set_xticks(x)
ax.set_xticklabels([n.split(" (")[0] for n in summary.index], rotation=15, ha="right")
ax.set_ylabel("flights of warning before failure")
ax.set_title(f"Three approaches, {len(SEEDS)} random splits of the engines")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9)
fig.tight_layout()
plotting.save(fig, "03_comparison.png")

# %% [markdown]
# ## One engine, all five scores
#
# Each divided by its own alarm level, so 1.0 means "alarm" for all of them and
# the shapes can be compared on one axis.

# %%
rng = np.random.default_rng(42)
test_u = rng.choice(units, size=30, replace=False)
tr = df[df.unit.isin(np.setdiff1d(units, test_u))].copy()
te = df[df.unit.isin(test_u)].copy()

u = int(test_u[0])
fig, ax = plt.subplots(figsize=(9, 4.6))
for name, (a, b) in build(tr, te).items():
    train_scored, test_scored = smoothed(tr, a), smoothed(te, b)
    threshold, _ = health.choose_threshold(train_scored)
    g = test_scored[test_scored.unit == u]
    ax.plot(g.cycle, g.hi_smooth / threshold, linewidth=1.4, label=name.split(" (")[0])
ax.axhline(1.0, color="grey", linestyle="--", linewidth=1, label="alarm level")
ax.set_yscale("log")
ax.set_xlabel("flight")
ax.set_ylabel("score / that detector's alarm level")
ax.set_title(f"Unit {u}: every score against its own alarm level")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.tight_layout()
plotting.save(fig, "03_detectors_one_engine.png")

# %%
plotting.show()
