# %% [markdown]
# # Step 5 — show the evidence, and measure what "better" left out
#
# Two loose ends from step 3 and 4.
#
# **The eigenvalue claim was asserted, not shown.** I said the healthy
# covariance is "essentially the identity, so PCA has nothing to model". That
# is testable and it deserves a picture.
#
# **"Beat" only ever meant earlier detection.** Nothing about whether the score
# is steady, or whether an alarm stays up once raised. A detector that fires
# early and then flickers on and off is worse in a control room than one that
# fires later and stays put, because operators learn to ignore a chattering
# alarm. That was never measured.

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

all_df = cmapss.add_rul(cmapss.load("FD001", "train"))
live = [s for s in cmapss.SENSOR_COLS if s not in cmapss.constant_sensors(all_df)]
trend = pd.Series({s: all_df.groupby("unit").apply(
    lambda g, c=s: g[c].corr(g["rul"]), include_groups=False).median() for s in live})
SENSORS = trend[abs(trend) > 0.5].index.tolist()
P = len(SENSORS)

rng = np.random.default_rng(42)
units = np.sort(all_df.unit.unique())
test_u = rng.choice(units, size=30, replace=False)
train_u = np.setdiff1d(units, test_u)
train = all_df[all_df.unit.isin(train_u)].copy()
test = all_df[all_df.unit.isin(test_u)].copy()

stats_tr = health.baseline_stats(train, SENSORS)
stats_te = health.baseline_stats(test, SENSORS)
z_tr = models.z_frame(train, SENSORS, stats_tr)
z_te = models.z_frame(test, SENSORS, stats_te)

healthy_mask = (train.cycle <= health.BASELINE_CYCLES).to_numpy()
z_healthy = z_tr.to_numpy()[healthy_mask]
z_full = z_tr.to_numpy()

# %% [markdown]
# ## Is the healthy covariance really just noise?
#
# There is a known answer for what pure noise looks like. For n samples of p
# uncorrelated variables, the sample covariance eigenvalues do not all come out
# at exactly 1 - sampling noise spreads them, and the Marchenko-Pastur law says
# how far: between (1 - sqrt(p/n))^2 and (1 + sqrt(p/n))^2.
#
# If the observed eigenvalues sit inside that band, the spread is what random
# chance produces from data with no structure at all.
#
# The comparison that makes it concrete: run the same calculation on full-life
# data instead of healthy-only. Degradation drives all these sensors together,
# so if correlation appears with wear, full-life data should look nothing like
# noise.

# %%
n_healthy = len(z_healthy)
gamma = P / n_healthy
mp_lo, mp_hi = (1 - np.sqrt(gamma)) ** 2, (1 + np.sqrt(gamma)) ** 2

ev_healthy = np.linalg.eigvalsh(np.cov(z_healthy, rowvar=False))[::-1]
ev_full = np.linalg.eigvalsh(np.cov(z_full, rowvar=False))[::-1]

print(f"healthy rows: {n_healthy:,}   sensors: {P}   p/n = {gamma:.4f}")
print(f"Marchenko-Pastur band for pure noise: [{mp_lo:.3f}, {mp_hi:.3f}]")
print(f"healthy eigenvalues:   {np.round(ev_healthy, 3)}")
print(f"  -> range [{ev_healthy.min():.3f}, {ev_healthy.max():.3f}], "
      f"all inside band: {ev_healthy.min() >= mp_lo and ev_healthy.max() <= mp_hi}")
print(f"\nfull-life eigenvalues: {np.round(ev_full, 2)}")
print(f"  -> largest is {ev_full[0]:.1f}, "
      f"{ev_full[0] / ev_healthy[0]:.0f}x the healthy largest")
print(f"  -> first component explains {ev_full[0] / ev_full.sum():.1%} of full-life variance")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

ax = axes[0]
ax.axhspan(mp_lo, mp_hi, color="lightcoral", alpha=0.25,
           label=f"noise band, Marchenko-Pastur\n[{mp_lo:.2f}, {mp_hi:.2f}]")
ax.plot(range(1, P + 1), ev_healthy, "o-", color="steelblue", label="healthy data")
ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
ax.set_xlabel("component")
ax.set_ylabel("eigenvalue")
ax.set_title("Healthy data: every eigenvalue sits in the noise band")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(range(1, P + 1), ev_full, "s-", color="darkorange", label="full life")
ax.plot(range(1, P + 1), ev_healthy, "o-", color="steelblue", label="healthy only")
ax.axhspan(mp_lo, mp_hi, color="lightcoral", alpha=0.25, label="noise band")
ax.set_yscale("log")
ax.set_xlabel("component")
ax.set_ylabel("eigenvalue (log)")
ax.set_title("Same sensors over full life: one direction dominates")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
plotting.save(fig, "05_eigenvalues.png")

# %% [markdown]
# ## What "beat" left out: is the score steady?
#
# Two things an operator cares about that lead time does not capture.
#
# **Chatter.** Once the alarm goes up, does it stay up? Count how many times a
# detector crosses its threshold over an engine's life. One crossing is a clean
# detection. Five is an alarm nobody will trust.
#
# **Ripple.** How much does the score bounce around while the engine is healthy?
# Measured as the cycle-to-cycle change relative to the distance to threshold -
# a score that wanders halfway to the alarm line on its own is living dangerously.

# %%
def attach(df, vals):
    out = df.copy()
    out["raw"] = vals
    out["hi_smooth"] = out.groupby("unit")["raw"].transform(
        lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
    return out

monitor = models.PCAMonitor(n_components=6).fit(z_healthy)
detectors = {
    "baseline (mean |z|)": (health.health_index(train, SENSORS),
                            health.health_index(test, SENSORS)),
    "control (mean z^2)":  (attach(train, (z_tr.to_numpy() ** 2).mean(1)),
                            attach(test, (z_te.to_numpy() ** 2).mean(1))),
    "PCA T2 (k=6)":        (attach(train, monitor.scores(z_tr.to_numpy())[0]),
                            attach(test, monitor.scores(z_te.to_numpy())[0])),
    "IsolationForest":     (attach(train, models.isolation_forest_scores(z_healthy, z_tr.to_numpy())),
                            attach(test, models.isolation_forest_scores(z_healthy, z_te.to_numpy()))),
}

rows = []
for name, (tr, te) in detectors.items():
    thr, _ = health.choose_threshold(tr)
    ev = health.evaluate(te, thr)

    crossings, ripple = [], []
    for _, g in te.groupby("unit"):
        g = g.sort_values("cycle")
        above = (g["hi_smooth"] > thr).to_numpy()
        crossings.append(int(np.sum(above[1:] & ~above[:-1])))
        early = g[g.cycle <= health.HEALTHY_UNTIL]["hi_smooth"]
        if len(early) > 1:
            ripple.append(early.diff().abs().mean() / thr)

    rows.append({
        "detector": name,
        "lead_time": ev["lead_time"],
        "worst_10pct": ev["lead_time_p10"],
        "threshold_crossings": np.mean(crossings),
        "healthy_ripple_%": 100 * np.mean(ripple),
    })

summary = pd.DataFrame(rows)
print(summary.round(2).to_string(index=False))
print("""
threshold_crossings  average times per engine the score crosses up through the
                     alarm level. 1.0 is a clean single detection; higher means
                     the alarm drops out and comes back.
healthy_ripple_%     average cycle-to-cycle wobble during the healthy window,
                     as a percentage of the distance to the alarm level.
""")

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
metrics = [("lead_time", "median warning (cycles)", "higher is better"),
           ("threshold_crossings", "threshold crossings per engine", "1.0 is ideal"),
           ("healthy_ripple_%", "healthy ripple (% of threshold)", "lower is better")]
colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]

for ax, (col, label, note) in zip(axes, metrics):
    ax.bar(range(len(summary)), summary[col], color=colors)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([d.split(" (")[0] for d in summary.detector],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(note, fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(summary[col]):
        ax.text(i, v, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=8)

fig.suptitle("Detectors judged on three things, not one", y=1.0)
fig.tight_layout()
plotting.save(fig, "05_three_metrics.png")

# %%
plotting.show()
