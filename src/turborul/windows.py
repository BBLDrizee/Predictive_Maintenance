"""Sliding windows over engine histories.

Windows never cross engines. Each window's label is the RUL at its last cycle.
Engines shorter than the window are front-padded by repeating their first cycle,
so the most recent readings always sit at the end of the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _pad_front(values: np.ndarray, window_len: int) -> np.ndarray:
    missing = window_len - len(values)
    if missing <= 0:
        return values
    return np.concatenate([np.repeat(values[:1], missing, axis=0), values])


def make_windows(df: pd.DataFrame, features: list[str], window_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All windows of every engine: X (n, window_len, n_features), y (n,), unit IDs (n,)."""
    xs, ys, units = [], [], []
    for unit, engine in df.groupby("unit", sort=True):
        values = _pad_front(engine[features].to_numpy(np.float32), window_len)
        rul = _pad_front(engine["rul"].to_numpy(np.float32), window_len)
        x = sliding_window_view(values, window_len, axis=0).transpose(0, 2, 1)  # (n, L, F)
        xs.append(x)
        ys.append(rul[window_len - 1:])
        units.append(np.full(len(x), unit))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(units)


def last_windows(df: pd.DataFrame, features: list[str], window_len: int) -> tuple[np.ndarray, np.ndarray]:
    """The final window of every engine (the standard test protocol): X (n_engines, L, F), unit IDs."""
    xs, units = [], []
    for unit, engine in df.groupby("unit", sort=True):
        values = engine[features].to_numpy(np.float32)[-window_len:]
        xs.append(_pad_front(values, window_len))
        units.append(unit)
    return np.stack(xs), np.asarray(units)
