"""A health index built from each engine's own healthy baseline.

The idea, in one sentence an operator can be told:

    "This engine is now running further from its own normal than it ever did
     when it was healthy."

No training, no labels, no model. Each engine is referenced against its own
early life, because engines differ from each other at healthy baseline by
roughly half the range they cover over a full life (measured in 01_explore).

That is also how condition monitoring is done in a plant: no two pumps are
identical, so you trend each one against itself rather than against a fleet
average.
"""

import numpy as np
import pandas as pd

BASELINE_CYCLES = 40      # how much early life defines "healthy" for an engine
SMOOTH_WINDOW = 5         # rolling mean, to stop single noisy readings alarming
PERSISTENCE = 3           # consecutive cycles above threshold before alarming
HEALTHY_UNTIL = BASELINE_CYCLES + 10   # cycles treated as definitely healthy


def baseline_stats(df: pd.DataFrame, sensors: list[str],
                   n_cycles: int = BASELINE_CYCLES) -> pd.DataFrame:
    """Mean and std of each sensor over each engine's first n_cycles."""
    early = df[df.cycle <= n_cycles]
    stats = early.groupby("unit")[sensors].agg(["mean", "std"])
    return stats


def health_index(df: pd.DataFrame, sensors: list[str],
                 stats: pd.DataFrame | None = None,
                 smooth: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """Add a health index column: mean |z| across sensors, per engine.

    z is measured against that engine's own baseline, and it grows as the engine
    drifts away from the condition it started in.

    A healthy engine reads about 0.80 rather than 0, because |z| folds the
    scatter onto one side and the mean absolute value of a standard normal is
    sqrt(2/pi) = 0.798. The measured healthy mean here is 0.792, so that is the
    floor of the scale rather than a coincidence.
    """
    if stats is None:
        stats = baseline_stats(df, sensors)

    out = df.copy()
    z_total = np.zeros(len(out))

    for s in sensors:
        mu = out["unit"].map(stats[(s, "mean")])
        sd = out["unit"].map(stats[(s, "std")])
        # A baseline std of zero would divide by zero; such a sensor carries no
        # usable scale for this engine, so leave it out by contributing nothing.
        sd = sd.replace(0.0, np.nan)
        z_total += (out[s] - mu).div(sd).abs().fillna(0.0).to_numpy()

    out["hi"] = z_total / len(sensors)
    out["hi_smooth"] = (
        out.groupby("unit")["hi"]
        .transform(lambda x: x.rolling(smooth, min_periods=1).mean())
    )
    return out


def choose_threshold(train_scored: pd.DataFrame, healthy_until: int = HEALTHY_UNTIL,
                     ks=(2, 3, 4, 5, 6, 8, 10, 12),
                     score: str = "hi_smooth",
                     persistence: int = PERSISTENCE) -> tuple[float, float]:
    """Smallest threshold, in units of healthy scatter, that never false-alarms.

        threshold = healthy mean + k * healthy standard deviation

    Returns (threshold, k).

    The rule has to be measured in units of each score's own healthy scatter,
    the same reasoning as the z-scores underneath. A multiplicative margin
    ("25% above the healthy maximum") is not scale-free and is not usable here:
    Isolation Forest's score comes from average path length in a tree and can
    only reach about 1.4x its healthy level, while mean z^2 is unbounded and
    reaches 157x. The same percentage means completely different things to the
    two of them.
    """
    healthy = train_scored[train_scored.cycle <= healthy_until][score]
    mu, sd = healthy.mean(), healthy.std()
    for k in ks:
        t = mu + k * sd
        if evaluate(train_scored, t, healthy_until, persistence, score)["early_rate"] == 0:
            return float(t), float(k)
    t = mu + ks[-1] * sd
    return float(t), float(ks[-1])


def first_alarm(df: pd.DataFrame, threshold: float,
                persistence: int = PERSISTENCE,
                score: str = "hi_smooth") -> pd.DataFrame:
    """First cycle where `score` stays above threshold for `persistence` cycles.

    Returns one row per engine: the alarm cycle, the RUL at that moment (the
    warning you would have had), and whether it alarmed at all.

    `score` is a column name so that every detector - the baseline health
    index, PCA, isolation forest - is scored by exactly the same code. A fair
    comparison needs the evaluation to be identical, not merely similar.
    """
    rows = []
    for unit, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        over = (g[score] > threshold).to_numpy()
        # run length of consecutive Trues ending at each position
        run = np.zeros(len(over), dtype=int)
        for i, flag in enumerate(over):
            run[i] = run[i - 1] + 1 if flag and i > 0 else int(flag)
        hit = np.argmax(run >= persistence) if (run >= persistence).any() else None

        if hit is None:
            rows.append({"unit": unit, "alarmed": False,
                         "alarm_cycle": np.nan, "lead_time": np.nan})
        else:
            row = g.iloc[hit]
            rows.append({"unit": unit, "alarmed": True,
                         "alarm_cycle": int(row["cycle"]),
                         "lead_time": int(row["rul"])})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, threshold: float,
             healthy_until: int = HEALTHY_UNTIL,
             persistence: int = PERSISTENCE,
             score: str = "hi_smooth") -> dict:
    """Score a threshold the way a plant would care about it.

    lead_time      cycles of warning before failure (median over engines)
    coverage       fraction of engines that alarmed at all before failing
    early_rate     fraction that alarmed during the healthy window

    `healthy_until` is a convention, not ground truth: C-MAPSS does not label
    fault onset. It sits just past the baseline window and well before any drift
    is visible (the shortest engine life is 128 cycles), so an alarm there is
    not a real detection.
    """
    alarms = first_alarm(df, threshold, persistence, score=score)
    alarmed = alarms[alarms.alarmed]
    early = alarmed[alarmed.alarm_cycle <= healthy_until]

    return {
        "threshold": threshold,
        "lead_time": alarmed["lead_time"].median() if len(alarmed) else np.nan,
        "lead_time_p10": alarmed["lead_time"].quantile(0.10) if len(alarmed) else np.nan,
        "coverage": len(alarmed) / len(alarms),
        "early_rate": len(early) / len(alarms),
    }
