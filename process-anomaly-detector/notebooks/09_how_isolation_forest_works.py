# %% [markdown]
# # Step 9 — how Isolation Forest actually works
#
# The comparison used Isolation Forest as a black box, so this opens it up in
# two dimensions using two of the real sensors. It also shows where the ceiling
# on its score comes from, which is what stops it distinguishing a nearly-dead
# engine from a moderately sick one in step 7.
#
# **Training happens once.** Take the healthy data, build 300 trees, and give
# each tree a random subsample of 256 points. Each tree splits recursively: pick
# a feature at random, pick a threshold at random between that feature's minimum
# and maximum in the current node, cut, repeat. Stop when a point sits alone or
# the depth cap is reached. The trees are then finished and are never touched
# again.
#
# **Scoring does not rebuild anything.** A new reading is dropped down each of
# the 300 existing trees, follows the splits that were already decided, and the
# depth it lands at is recorded. Average those 300 depths, and a short average
# means the point was easy to isolate. Readings do not join the training data
# and nothing accumulates, hence scoring reading 5000 first and reading 1 last
# gives identical answers.

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

# two sensors so the geometry can be drawn
A, B = "s_11", "s_4"
X = z[[A, B]].to_numpy()
healthy_mask = (df.cycle <= health.BASELINE_CYCLES).to_numpy()

rng = np.random.default_rng(0)
train = X[healthy_mask][rng.choice(healthy_mask.sum(), 256, replace=False)]
forest = IsolationForest(n_estimators=300, max_samples=256, random_state=0).fit(train)

LIM = (-4, 12)


def draw_cuts(ax, tree, node=0, xlim=LIM, ylim=LIM, depth=0, max_depth=7):
    """Draw one tree's axis-aligned cuts as lines, shallowest darkest."""
    t = tree.tree_
    if t.children_left[node] == -1 or depth >= max_depth:
        return
    f, thr = t.feature[node], t.threshold[node]
    shade = str(max(0.85 - 0.1 * depth, 0.15))
    width = max(1.6 - 0.2 * depth, 0.4)
    if f == 0:
        ax.plot([thr, thr], ylim, color=shade, linewidth=width)
        draw_cuts(ax, tree, t.children_left[node], (xlim[0], thr), ylim, depth + 1, max_depth)
        draw_cuts(ax, tree, t.children_right[node], (thr, xlim[1]), ylim, depth + 1, max_depth)
    else:
        ax.plot(xlim, [thr, thr], color=shade, linewidth=width)
        draw_cuts(ax, tree, t.children_left[node], xlim, (ylim[0], thr), depth + 1, max_depth)
        draw_cuts(ax, tree, t.children_right[node], xlim, (thr, ylim[1]), depth + 1, max_depth)


def path_length(tree, x):
    """How many cuts before this reading lands in a leaf of an existing tree."""
    t = tree.tree_
    node, d = 0, 0
    while t.children_left[node] != -1:
        node = (t.children_left[node] if x[t.feature[node]] <= t.threshold[node]
                else t.children_right[node])
        d += 1
    return d


normal_pt = np.array([0.3, 0.2])     # inside the healthy cloud
worn_pt = np.array([6.0, 7.0])       # a worn engine

# The alarm goes where the rest of the project puts it: healthy mean plus the
# smallest number of standard deviations that leaves no healthy reading above.
score = -forest.score_samples(X)
sh = score[healthy_mask]
near_failure = (~healthy_mask) & (df.rul.to_numpy() < 60)
k = next(k for k in (2, 3, 4, 5, 6, 8, 10, 12) if (sh > sh.mean() + k * sh.std()).sum() == 0)
alarm = sh.mean() + k * sh.std()

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

ax = axes[0]
ax.scatter(train[:, 0], train[:, 1], s=14, color="steelblue", alpha=0.7,
           label="healthy training points (256)", zorder=3)
draw_cuts(ax, forest.estimators_[0])
for pt, colour, lab in [(normal_pt, "seagreen", "a normal reading"),
                        (worn_pt, "crimson", "a worn reading")]:
    ax.scatter(*pt, s=190, marker="*", color=colour, edgecolor="black",
               linewidth=0.8, zorder=5, label=lab)
ax.set_xlim(LIM), ax.set_ylim(LIM)
ax.set_xlabel(f"{A} (z)"), ax.set_ylabel(f"{B} (z)")
ax.set_title("One tree: random cuts, one feature at a time\n"
             "(first 7 levels only)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
d_norm = [path_length(t, normal_pt) for t in forest.estimators_]
d_worn = [path_length(t, worn_pt) for t in forest.estimators_]
bins = np.arange(0, max(max(d_norm), max(d_worn)) + 2) - 0.5
ax.hist(d_norm, bins=bins, alpha=0.75, color="seagreen",
        label=f"normal reading, mean {np.mean(d_norm):.1f} cuts")
ax.hist(d_worn, bins=bins, alpha=0.75, color="crimson",
        label=f"worn reading, mean {np.mean(d_worn):.1f} cuts")
ax.set_xlabel("cuts needed to isolate this reading")
ax.set_ylabel("number of trees (out of 300)")
ax.set_title("The same two readings through all 300 trees", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
gx, gy = np.meshgrid(np.linspace(*LIM, 220), np.linspace(*LIM, 220))
grid = -forest.score_samples(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
im = ax.contourf(gx, gy, grid, levels=25, cmap="magma")
ax.contour(gx, gy, grid, levels=[alarm], colors="white", linewidths=2)
ax.scatter(X[healthy_mask][:, 0], X[healthy_mask][:, 1], s=3,
           color="deepskyblue", alpha=0.35, label="healthy")
ax.scatter(X[near_failure][:, 0], X[near_failure][:, 1], s=3,
           color="lime", alpha=0.3, label="within 60 flights of failure")
ax.set_xlim(LIM), ax.set_ylim(LIM)
ax.set_xlabel(f"{A} (z)"), ax.set_ylabel(f"{B} (z)")
ax.set_title(f"The finished forest: score everywhere in the plane\n"
             f"(white line = alarm, healthy mean + {k} sd)", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.colorbar(im, ax=ax, label="isolation score")

fig.tight_layout()
plotting.save(fig, "09_isolation_forest_explained.png")

# %%
cap = int(np.ceil(np.log2(256)))
print(f"normal reading at {normal_pt}: {np.mean(d_norm):4.1f} cuts, "
      f"score {-forest.score_samples([normal_pt])[0]:.3f}")
print(f"worn   reading at {worn_pt}: {np.mean(d_worn):4.1f} cuts, "
      f"score {-forest.score_samples([worn_pt])[0]:.3f}")
print(f"""
The trees were built once from {len(train)} healthy points and never touched again.

Where the ceiling comes from: with 256 samples per tree the depth cap is about
log2(256) = {cap}, and the green histogram is a single spike sitting on it. A normal
reading cannot score deeper than that, hence path length lives between 1 and {cap}
and the score between roughly 0.38 and 0.80.

That bound is why the score saturates near failure in step 7 and stops telling a
nearly-dead engine from a moderately sick one. It is a property of the method,
not of the engines.
""")

# %% [markdown]
# ## Plenty of near-failure readings sit inside the alarm line
#
# The third panel is a two-sensor slice of a fourteen-sensor detector, and it is
# scoring single readings with no smoothing, hence it is a much weaker test than
# the one the results table reports.

# %%
print(f"""alarm in this two-sensor slice: {alarm:.3f} = healthy mean + {k} sd

  healthy readings above it:                 {100 * (sh > alarm).mean():.2f}%
  readings within 60 flights above it:       {100 * (score[near_failure] > alarm).mean():.1f}%""")
for lo, hi in [(45, 60), (30, 45), (15, 30), (0, 15)]:
    m = (~healthy_mask) & (df.rul.to_numpy() >= lo) & (df.rul.to_numpy() < hi)
    print(f"    of those, {lo:2d} to {hi:2d} flights out:{'':11}{100 * (score[m] > alarm).mean():5.1f}%")
print(f"""
So the green cloud straddles the line rather than sitting outside it, and about
a third of the readings within 60 flights of failure are on the healthy side.
Two sensors out of fourteen and no smoothing is why. The full detector averages
all fourteen, smooths over five flights and needs three flights in a row above
the line, and it catches every engine.

The alarm at {alarm:.2f} also sits close to the {0.80:.2f} ceiling, which leaves the
detector very little room to work in, and is the same bound seen in the middle
panel arriving as a practical limit rather than as a curiosity.""")

# %% [markdown]
# ## One sensor far out scores lower than two sensors mildly out
#
# The score is probed at chosen points rather than read off the picture. The
# points are chosen, the numbers are measured.

# %%
def path_length_mean(x):
    return np.mean([path_length(t, x) for t in forest.estimators_])


def survivors_after_first_cut(x):
    """Training points still sharing this reading's node after one cut."""
    out = []
    for t in forest.estimators_:
        T = t.tree_
        child = (T.children_right[0] if x[T.feature[0]] > T.threshold[0]
                 else T.children_left[0])
        out.append(T.n_node_samples[child])
    return np.mean(out)


print(f"training data spans {A} {train[:,0].min():.1f} to {train[:,0].max():.1f}, "
      f"{B} {train[:,1].min():.1f} to {train[:,1].max():.1f}")
print(f"alarm {alarm:.3f}\n")
print(f"{'point':>12}  {'cuts':>5}  {'left after cut 1':>16}  {'score':>6}")
for p in [(0, 0), (3, 3), (4, 4), (6, 7), (10, 0), (12, 2), (2, 12), (12, 12)]:
    x = np.array(p, dtype=float)
    s = -forest.score_samples([x])[0]
    print(f"{str(p):>12}  {path_length_mean(x):5.2f}  {survivors_after_first_cut(x):16.1f}  "
          f"{s:6.3f}  {'ALARM' if s > alarm else ''}")

# %% [markdown]
# Two readings 3 standard deviations out on both sensors score higher than one
# reading 12 standard deviations out on a single sensor, and the lone extreme
# reading takes more cuts to isolate rather than fewer.
#
# **The explanation, which is reasoning and not measurement:** each cut picks an
# axis at random, hence for the diagonal reading both axes carve away
# neighbours, while for the lone extreme reading every cut landing on the
# normal-looking axis leaves it with company. The survivor column is the
# evidence for that, since it counts how many training points still share the
# reading's node after a single cut.
#
# **Where this claim is fragile:** the ordering of the scores is solid, but
# whether the lone extreme reading alarms depends on where the alarm sits, since
# it scores 0.69 against an alarm of 0.74. One step down in the margin rule, at
# mean + 4 sd, would put the alarm at 0.686 and the lone extreme reading would
# then alarm. The cell below shows that directly.

# %%
for kk in (3, 4, 5, 6):
    thr = sh.mean() + kk * sh.std()
    lone = -forest.score_samples([[12.0, 2.0]])[0]
    both = -forest.score_samples([[3.0, 3.0]])[0]
    print(f"alarm at mean + {kk} sd = {thr:.3f}   healthy above {100*(sh>thr).mean():5.2f}%   "
          f"lone 12 sd {'alarms' if lone > thr else 'silent'}, "
          f"both 3 sd {'alarms' if both > thr else 'silent'}")

# %% [markdown]
# ## The same test on the detector as actually built
#
# Two sensors is a toy. This repeats it on all 14, which is the detector the
# results table reports, and asks what happens when a single sensor goes bad on
# its own while the other 13 read normal. That is an everyday plant fault,
# namely one instrument drifting or failing.

# %%
X14 = z[SENSORS].to_numpy()
r14 = np.random.default_rng(0)
f14 = IsolationForest(n_estimators=300, max_samples=256, random_state=0).fit(
    X14[healthy_mask][r14.choice(healthy_mask.sum(), 256, replace=False)])
sh14 = -f14.score_samples(X14[healthy_mask])
alarm14 = sh14.mean() + 5 * sh14.std()
hi_alarm = 1.098   # the health-index alarm, from health.choose_threshold

print(f"14 sensors. healthy score mean {sh14.mean():.3f}, sd {sh14.std():.3f}, "
      f"alarm {alarm14:.3f}")
print(f"the health index alarms at {hi_alarm:.2f}\n")
print("one sensor at z, the other 13 reading normal:\n")
print(f"{'z':>6}  {'forest':>7}  {'':8}  {'health index':>12}")
for v in (3, 6, 9, 12, 20, 40, 100):
    p = np.zeros(len(SENSORS))
    p[0] = v
    s = -f14.score_samples([p])[0]
    print(f"{v:6d}  {s:7.3f}  {'ALARM' if s > alarm14 else 'silent':8}  "
          f"{v/len(SENSORS):12.2f}  {'ALARM' if v/len(SENSORS) > hi_alarm else 'silent'}")

# %% [markdown]
# The forest score is flat at 0.411 whether the sensor reads 3 standard
# deviations out or 100, hence it never alarms at any magnitude, and 0.411 sits
# below the healthy mean of 0.443, so the reading looks more ordinary than an
# average healthy one.
#
# **The explanation, and the flatness is the evidence for it:** path length
# depends only on which side of each cut a reading falls, and once the reading
# is past every training point on that axis, every cut on that axis sends it the
# same way whether it is 12 out or 100 out, hence distance beyond the training
# range buys nothing. Meanwhile the other 13 coordinates sit at dead centre,
# which is the densest place they can be, and those are the axes doing the
# isolating.
#
# The health index is not immune either, since averaging 14 sensors divides the
# lone excursion by 14 and it needs 15.4 standard deviations on one sensor
# before the mean clears 1.098. It does get there, however, and the forest does
# not get there at all.
#
# Neither of these is a single-sensor fault detector, and neither is meant to
# be, since a plain per-sensor limit check is what catches that case and it
# should run alongside.

# %%
plotting.show()
