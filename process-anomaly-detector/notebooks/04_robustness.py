# %% [markdown]
# # Step 4 — does the PCA advantage survive a different split?
#
# Step 3 ended with PCA T2 ahead of the control by about 7 cycles on one split
# of the engines, with k chosen on training data. Two reasons not to believe it
# yet:
#
# 1. Adjacent values of k moved the answer by a few cycles for no principled
#    reason, so the noise floor is a few cycles and the claim is not far above it.
# 2. The eigenvalues of the healthy covariance run 1.15 down to 0.80, a ratio of
#    1.44. For 1400 samples in 14 dimensions, pure sampling noise on an identity
#    covariance predicts a spread of roughly that size. So the directions PCA
#    calls "largest" may be nothing but noise, and which ones land on top could
#    change completely with a different set of engines.
#
# One test settles it: repeat the whole procedure on several random splits. A
# real effect survives. An artefact of one split does not.

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
units = np.sort(all_df.unit.unique())


def smoothed(df, values):
    out = df.copy()
    out["s"] = values
    out["hi_smooth"] = out.groupby("unit")["s"].transform(
        lambda x: x.rolling(health.SMOOTH_WINDOW, min_periods=1).mean())
    return out


def lead_times(tr_scored, te_scored):
    """Threshold set on training engines; report warning on both."""
    thr, _ = health.choose_threshold(tr_scored)
    return (health.evaluate(tr_scored, thr)["lead_time"],
            health.evaluate(te_scored, thr)["lead_time"])


# %% [markdown]
# For each split: build the control, then choose k for PCA T2 **using training
# engines only**, then score that k once on the held-out engines. Choosing k by
# looking at the test result would be the whole point of the split, thrown away.

# %%
rows = []
for seed in [0, 1, 7, 42, 123, 2024]:
    rng = np.random.default_rng(seed)
    test_u = rng.choice(units, size=30, replace=False)
    train_u = np.setdiff1d(units, test_u)
    tr = all_df[all_df.unit.isin(train_u)].copy()
    te = all_df[all_df.unit.isin(test_u)].copy()

    z_tr = models.z_frame(tr, SENSORS, health.baseline_stats(tr, SENSORS))
    z_te = models.z_frame(te, SENSORS, health.baseline_stats(te, SENSORS))
    healthy = z_tr.to_numpy()[(tr.cycle <= health.BASELINE_CYCLES).to_numpy()]

    _, control = lead_times(smoothed(tr, (z_tr.to_numpy() ** 2).mean(1)),
                            smoothed(te, (z_te.to_numpy() ** 2).mean(1)))

    best_k, best_train = None, -np.inf
    for k in range(1, len(SENSORS)):
        m = models.PCAMonitor(n_components=k).fit(healthy)
        tr_lead, _ = lead_times(smoothed(tr, m.scores(z_tr.to_numpy())[0]),
                                smoothed(te, m.scores(z_te.to_numpy())[0]))
        if tr_lead > best_train:
            best_train, best_k = tr_lead, k

    m = models.PCAMonitor(n_components=best_k).fit(healthy)
    _, t2 = lead_times(smoothed(tr, m.scores(z_tr.to_numpy())[0]),
                       smoothed(te, m.scores(z_te.to_numpy())[0]))

    rows.append({"seed": seed, "k": best_k, "control": control,
                 "pca_t2": t2, "gain": t2 - control})

res = pd.DataFrame(rows)
print(res.to_string(index=False))
print(f"\nk chosen across splits: {sorted(res.k)}")
print(f"PCA T2 minus control: mean {res.gain.mean():+.1f} cycles, "
      f"range {res.gain.min():+.0f} to {res.gain.max():+.0f}")
print(f"PCA T2 ahead in {(res.gain > 0).sum()} of {len(res)} splits")

# %% [markdown]
# ## Result
#
# The advantage holds on every split, by more than the noise floor. So the
# earlier conclusion - that nothing modelling correlation structure beats a
# plain squared-deviation statistic - was wrong, and this is the correction.
#
# **Why it works is worth being careful about.** The healthy covariance really
# is close to the identity, so PCA is not finding strong correlations. The most
# likely explanation is that the baseline window is not perfectly healthy:
# degradation is gradual and has already begun during cycles 1-20, which gives
# the degradation direction slightly more variance than the rest. PCA then
# ranks that direction near the top, and keeping a handful of components keeps
# the signal while discarding mostly-noise dimensions.
#
# That is a hypothesis consistent with the evidence, not something demonstrated
# here. Testing it would mean checking whether the leading eigenvectors align
# with the direction the sensors drift as engines age.

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.4))
x = np.arange(len(res))
ax.bar(x - 0.2, res.control, width=0.4, label="control (mean z^2)", color="darkorange")
ax.bar(x + 0.2, res.pca_t2, width=0.4, label="PCA T2 (k chosen on train)", color="steelblue")
for i, r in res.iterrows():
    ax.text(i + 0.2, r.pca_t2 + 1, f"k={r.k}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in res.seed])
ax.set_ylabel("median warning before failure (cycles)")
ax.set_title(f"PCA T2 ahead on all {len(res)} splits (mean {res.gain.mean():+.1f} cycles)")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=9)
plotting.save(fig, "04_robustness.png")

# %%
plotting.show()
