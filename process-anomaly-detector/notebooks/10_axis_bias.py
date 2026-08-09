# %% [markdown]
# # Step 10 — the score bands along the axes are an artifact
#
# Step 9 found that a reading far out on one sensor scores lower than a reading
# mildly out on two, and the score map shows four bands of low score running
# along the axes. This step tests where those bands come from.
#
# There are two competing explanations. Either the bands follow the data,
# meaning they are telling us something about the engines, or they follow the
# coordinate axes, meaning they are a property of a method that only ever cuts
# one sensor at a time.
#
# The test is to rotate the data by 45 degrees, fit a second forest on the
# rotated copy, and rotate its score map back for display. Nothing about the
# cloud changes, only the frame it is expressed in. If the bands follow the
# data they will land in the same place, and if they follow the axes they will
# turn with the frame.

# %%
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

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

A, B = "s_11", "s_4"
X = z[[A, B]].to_numpy()
healthy_mask = (df.cycle <= health.BASELINE_CYCLES).to_numpy()
rng = np.random.default_rng(0)
train = X[healthy_mask][rng.choice(healthy_mask.sum(), 256, replace=False)]

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

plain = IsolationForest(n_estimators=300, max_samples=256, random_state=0).fit(train)
turned = IsolationForest(n_estimators=300, max_samples=256, random_state=0).fit(train @ R.T)

score_plain = lambda P: -plain.score_samples(P)
score_turned = lambda P: -turned.score_samples(P @ R.T)   # same points, rotated frame

# %% [markdown]
# ## A reading beyond the training range cannot be cut away from the cloud
#
# The claim to test is that one lucky split on the extreme feature isolates such
# a reading quickly. Scikit-learn draws each split threshold uniformly between
# the smallest and largest value **in that node**, and the node only ever holds
# training points, hence no threshold can ever be placed between the healthy
# cloud and a reading sitting outside its range. Every split on that feature
# sends the reading the same way, and the reading inherits the path length of
# whichever training region it falls into.
#
# If that is right, the score stops changing as soon as the reading passes the
# edge of the training data, and going further out buys nothing.

# %%
print(f"training range on {A}: {train[:,0].min():.2f} to {train[:,0].max():.2f}\n")
print(f"{'position on the ' + A + ' axis':>32}  {'score':>6}")
for t in (1, 2, 2.8, 3, 4, 6, 12, 50, 1000):
    print(f"{f'({t:g}, 0)':>32}  {score_plain(np.array([[t, 0.0]]))[0]:6.3f}")

# %% [markdown]
# ## Along an axis against along the diagonal, at matched distance
#
# Both rays start at the middle of the healthy cloud and travel the same
# Euclidean distance, one along a sensor axis and one at 45 degrees between the
# two sensors.

# %%
dist = np.linspace(0, 12, 40)
along_axis = np.c_[dist, np.zeros_like(dist)]
along_diag = np.c_[dist / np.sqrt(2), dist / np.sqrt(2)]

print(f"\n{'distance':>9}  {'along axis':>10}  {'along diagonal':>14}  {'gap':>6}")
for d, a, g in zip(dist, score_plain(along_axis), score_plain(along_diag)):
    if d % 2 < 0.35:
        print(f"{d:9.1f}  {a:10.3f}  {g:14.3f}  {g-a:6.3f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
LIM = (-12, 12)
gx, gy = np.meshgrid(np.linspace(*LIM, 240), np.linspace(*LIM, 240))
grid = np.c_[gx.ravel(), gy.ravel()]

for ax, scorer, title in [
    (axes[0], score_plain, "Fitted in the sensor frame"),
    (axes[1], score_turned, "Fitted in a frame turned 45 degrees\n(same data, same points scored)"),
]:
    S = scorer(grid).reshape(gx.shape)
    im = ax.contourf(gx, gy, S, levels=25, cmap="magma")
    ax.scatter(train[:, 0], train[:, 1], s=4, color="deepskyblue", alpha=0.5)
    ax.set_xlim(LIM), ax.set_ylim(LIM)
    ax.set_xlabel(f"{A} (z)"), ax.set_ylabel(f"{B} (z)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="isolation score")

ax = axes[2]
ax.plot(dist, score_plain(along_axis), linewidth=2, color="steelblue",
        label="along a sensor axis")
ax.plot(dist, score_plain(along_diag), linewidth=2, color="crimson",
        label="along the diagonal")
ax.axvline(train[:, 0].max(), color="grey", linestyle=":", linewidth=1.2,
           label="edge of the training data")
ax.set_xlabel("Euclidean distance from the middle of the healthy cloud")
ax.set_ylabel("isolation score")
ax.set_title("Same distance, different score, depending only\non direction", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

fig.tight_layout()
plotting.save(fig, "10_axis_bias.png")

# %% [markdown]
# ## What the rotation settles
#
# The bands turn with the frame. The cloud is identical in both panels and the
# same points are being scored, so a difference between the two panels cannot
# be a property of the engines, and the bands are a property of the axes.
#
# This is the known limitation that Extended Isolation Forest was built to fix,
# by cutting with randomly oriented hyperplanes instead of one feature at a
# time, which removes the bands and makes the score close to rotationally
# invariant.
#
# It also settles what step 9 should have said. The forest is not "built to
# catch sensors moving together", since nothing in it was designed with that in
# mind. What is true is narrower: with axis-aligned cuts, a fault that moves
# several sensors at once lands in a corner of the healthy cloud where cuts on
# every axis help isolate it, while a fault that moves one sensor lands in a
# slab where only one axis helps. The C-MAPSS fault happens to be the first
# kind, hence the forest keeps up with the simpler detectors here, and that is
# a fact about this fault rather than a virtue of the method.

# %%
same_point = np.array([[6.0, 7.0]])
print(f"""
the same reading at (6, 7):
   scored by the forest fitted in the sensor frame : {score_plain(same_point)[0]:.3f}
   scored by the forest fitted in the turned frame : {score_turned(same_point)[0]:.3f}

A rotationally invariant detector would return the same number twice. The gap is
the size of the artifact.""")

# %%
plotting.show()
