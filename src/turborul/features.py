"""Engineered window features for classical models: mean, std, slope, last, min and max per sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd

STATS: tuple[str, ...] = ("mean", "std", "slope", "last", "min", "max")


def window_features(x: np.ndarray, sensors: list[str]) -> pd.DataFrame:
    """Turn windows of shape (n, window_len, n_sensors) into one row of features per window.

    The slope is the least-squares linear trend per cycle across the window.
    """
    n, length, _ = x.shape
    t = np.arange(length, dtype=np.float64) - (length - 1) / 2
    centred = x - x.mean(axis=1, keepdims=True)
    stats = {
        "mean": x.mean(axis=1),
        "std": x.std(axis=1),
        "slope": np.einsum("t,ntf->nf", t, centred) / (t @ t),
        "last": x[:, -1, :],
        "min": x.min(axis=1),
        "max": x.max(axis=1),
    }
    cols = {f"{s}_{stat}": stats[stat][:, i] for stat in STATS for i, s in enumerate(sensors)}
    return pd.DataFrame(cols, index=pd.RangeIndex(n))
