"""Engine-level train/validation split (FR-3) and test-like validation truncation.

Rows from one engine are strongly correlated, so splitting by row would leak an
engine's future into training. Every split here is by engine ID.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_engines(units: np.ndarray | list[int], val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle unique engine IDs and hold out ``val_fraction`` of them (at least one) for validation."""
    units = np.unique(np.asarray(units))
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    n_val = max(1, int(round(len(units) * val_fraction)))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def kfold_engines(units: np.ndarray | list[int], n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Engine-level k-fold split: every engine is in exactly one validation fold."""
    units = np.unique(np.asarray(units))
    if not 2 <= n_folds <= len(units):
        raise ValueError("n_folds must be between 2 and the number of engines")
    shuffled = np.random.default_rng(seed).permutation(units)
    folds = np.array_split(shuffled, n_folds)
    return [(np.sort(np.concatenate(folds[:i] + folds[i + 1:])), np.sort(folds[i])) for i in range(n_folds)]


def split_frame(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a frame of engine histories into train and validation frames by engine."""
    train_units, val_units = split_engines(df["unit"].unique(), val_fraction, seed)
    return (
        df[df["unit"].isin(train_units)].reset_index(drop=True),
        df[df["unit"].isin(val_units)].reset_index(drop=True),
    )


def truncate_engines(df: pd.DataFrame, cuts_per_engine: int, min_cycles: int, max_rul: int | None,
                     seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Mimic the official test set from run-to-failure histories.

    Each engine is cut at ``cuts_per_engine`` random cycles, keeping at least
    ``min_cycles`` of history, at least one cycle before failure and, if given,
    a true RUL of at most ``max_rul``. Every cut becomes a pseudo engine with a
    new ``unit`` ID. Returns the truncated histories and the true, uncapped RUL
    at each cut, so validation is scored exactly like the test set.
    """
    rng = np.random.default_rng(seed)
    pieces, truth = [], {}
    new_id = 1
    for _, engine in df.groupby("unit", sort=True):
        life = int(engine["cycle"].max())
        low = min(min_cycles, life - 1)
        if max_rul is not None:
            low = max(low, life - max_rul)
        cuts = rng.integers(low, life, size=cuts_per_engine)  # cut cycle in [low, life - 1]
        for cut in cuts:
            piece = engine[engine["cycle"] <= cut].copy()
            piece["unit"] = new_id
            pieces.append(piece)
            truth[new_id] = life - int(cut)
            new_id += 1
    out = pd.concat(pieces, ignore_index=True)
    return out, pd.Series(truth, name="rul").rename_axis("unit")
