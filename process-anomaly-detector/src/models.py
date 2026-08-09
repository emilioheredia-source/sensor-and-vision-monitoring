"""Two standard detectors, for comparison against the baseline health index.

Both are fitted on healthy data only - the baseline window of the training
engines - because in a plant you have plenty of "running fine" data and almost
no labelled failures. That is the realistic setup, and it is why unsupervised
methods get used at all.

Both consume the same per-engine z-scores the baseline uses. Giving them the
same preprocessing is what makes the comparison about the *method* rather than
about who got the better inputs.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def z_frame(df: pd.DataFrame, sensors: list[str], stats: pd.DataFrame) -> pd.DataFrame:
    """Per-engine z-scores: (reading - that engine's normal) / its normal wobble."""
    z = pd.DataFrame(index=df.index)
    for s in sensors:
        mu = df["unit"].map(stats[(s, "mean")])
        sd = df["unit"].map(stats[(s, "std")]).replace(0.0, np.nan)
        z[s] = (df[s] - mu).div(sd).fillna(0.0)
    return z


class PCAMonitor:
    """Classical multivariate process monitoring: Hotelling's T-squared and Q.

    Fit a PCA on healthy data. Two things can then go wrong with a new reading:

    T2  it moves too far along the directions the healthy data normally varies
        in - the right pattern, but too much of it.
    Q   it moves in a direction healthy data never varied in at all, so the
        model cannot reproduce it. Also called SPE, squared prediction error.

    Q is usually the one that catches a genuinely new fault, because a new
    failure mode breaks the correlations the healthy model was built on.
    """

    def __init__(self, n_components: float | int = 0.95):
        self.pca = PCA(n_components=n_components)

    def fit(self, z_healthy: np.ndarray) -> "PCAMonitor":
        self.pca.fit(z_healthy)
        self.var_ = self.pca.explained_variance_
        return self

    def scores(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = self.pca.transform(z)
        t2 = np.sum(t**2 / self.var_, axis=1)
        recon = self.pca.inverse_transform(t)
        q = np.sum((z - recon) ** 2, axis=1)
        return t2, q


def isolation_forest_scores(z_healthy: np.ndarray, z_all: np.ndarray,
                            seed: int = 42) -> np.ndarray:
    """Isolation Forest anomaly score, higher = more anomalous.

    Trained on healthy rows only. sklearn's score_samples returns higher values
    for *more normal* points, so it is negated here to match the convention of
    every other detector in this project: larger means worse.
    """
    forest = IsolationForest(
        n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1
    )
    forest.fit(z_healthy)
    return -forest.score_samples(z_all)
