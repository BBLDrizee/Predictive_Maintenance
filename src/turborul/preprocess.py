"""Sensor selection, operating-condition clustering and per-condition normalisation (FR-2).

Everything is fitted on training engines only and then applied unchanged to
validation and test engines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from turborul.data import OP_SETTINGS, SENSORS


def trend_strength(df: pd.DataFrame, sensors: list[str]) -> pd.Series:
    """Mean over engines of |Spearman rho| between each sensor and cycle within the engine.

    Unlike a global correlation with RUL, this does not cancel out sensors that
    degrade in opposite directions under different fault modes. Constant sensors
    score NaN.
    """
    ranks = df.groupby("unit")[[*sensors, "cycle"]].rank()
    ranks["unit"] = df["unit"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        per_engine = ranks.groupby("unit")[[*sensors, "cycle"]].apply(lambda e: e[sensors].corrwith(e["cycle"]))
    return per_engine.abs().mean()


class Preprocessor:
    """Cluster operating conditions, z-score sensors within each condition, and keep informative sensors."""

    def __init__(self, n_conditions: int = 1, trend_threshold: float = 0.2, seed: int = 0) -> None:
        self.n_conditions = n_conditions
        self.trend_threshold = trend_threshold
        self.seed = seed
        self.kmeans: KMeans | None = None
        self.means: pd.DataFrame | None = None  # condition x sensor
        self.stds: pd.DataFrame | None = None
        self.strength: pd.Series | None = None
        self.sensors: list[str] = []

    def _conditions(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans is None:
            return np.zeros(len(df), dtype=int)
        return self.kmeans.predict(df[OP_SETTINGS].to_numpy())

    def _normalise(self, df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
        cond = self._conditions(df)
        mu = self.means.loc[cond, sensors].to_numpy()
        sd = self.stds.loc[cond, sensors].to_numpy()
        out = df[["unit", "cycle"] + (["rul"] if "rul" in df else [])].copy()
        out["condition"] = cond
        out[sensors] = (df[sensors].to_numpy() - mu) / sd
        return out

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        if self.n_conditions > 1:
            self.kmeans = KMeans(self.n_conditions, n_init=10, random_state=self.seed).fit(train[OP_SETTINGS].to_numpy())
        cond = self._conditions(train)
        grouped = train[SENSORS].groupby(cond)
        self.means = grouped.mean()
        # Constant-within-condition sensors get std 1 so they normalise to 0; selection drops them.
        self.stds = grouped.std().fillna(0).replace(0, 1.0)
        self.strength = trend_strength(self._normalise(train, SENSORS), SENSORS)
        self.sensors = [s for s in SENSORS if self.strength[s] >= self.trend_threshold]
        if not self.sensors:
            raise ValueError("no sensor passes the trend-strength threshold")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return unit, cycle, (rul,) condition and the selected sensors, normalised."""
        if self.means is None:
            raise RuntimeError("Preprocessor is not fitted")
        return self._normalise(df, self.sensors)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)
