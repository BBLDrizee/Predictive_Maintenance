"""Tests for sliding windows, last-window extraction, padding and window features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turborul.data import add_train_rul
from turborul.features import STATS, window_features
from turborul.windows import last_windows, make_windows

L = 5


def _engines(lengths: dict[int, int]) -> pd.DataFrame:
    """Sensor 'a' encodes unit*1000 + cycle so every value identifies its row."""
    rows = [(u, c, u * 1000 + c, -c) for u, n in lengths.items() for c in range(1, n + 1)]
    return add_train_rul(pd.DataFrame(rows, columns=["unit", "cycle", "a", "b"]), cap=None)


def test_window_count_shape_and_labels() -> None:
    df = _engines({1: 8, 2: 6})
    x, y, units = make_windows(df, ["a", "b"], L)
    assert x.shape == ((8 - L + 1) + (6 - L + 1), L, 2)
    # First window of engine 1 covers cycles 1..5; label is RUL at cycle 5 = 8 - 5.
    np.testing.assert_array_equal(x[0, :, 0], [1001, 1002, 1003, 1004, 1005])
    assert y[0] == 3
    # Last window of each engine ends at failure.
    assert y[units == 1][-1] == 0 and y[units == 2][-1] == 0


def test_windows_never_cross_engines() -> None:
    x, _, units = make_windows(_engines({1: 8, 2: 6, 3: 7}), ["a"], L)
    owners = (x[:, :, 0] // 1000).astype(int)
    assert (owners == units[:, None]).all()


def test_windows_are_consecutive_cycles() -> None:
    x, _, _ = make_windows(_engines({1: 9}), ["a"], L)
    assert (np.diff(x[:, :, 0], axis=1) == 1).all()


def test_short_engine_is_front_padded() -> None:
    x, y, _ = make_windows(_engines({1: 3}), ["a"], L)
    assert x.shape == (1, L, 1)
    np.testing.assert_array_equal(x[0, :, 0], [1001, 1001, 1001, 1002, 1003])
    assert y[0] == 0


def test_last_windows_take_final_cycles_and_pad() -> None:
    x, units = last_windows(_engines({2: 7, 1: 3}), ["a"], L)
    assert units.tolist() == [1, 2]
    np.testing.assert_array_equal(x[0, :, 0], [1001, 1001, 1001, 1002, 1003])
    np.testing.assert_array_equal(x[1, :, 0], [2003, 2004, 2005, 2006, 2007])


def test_window_features_values() -> None:
    x = np.zeros((1, 4, 2), dtype=np.float32)
    x[0, :, 0] = [1, 2, 3, 4]       # slope 1
    x[0, :, 1] = [5, 5, 5, 5]       # flat
    f = window_features(x, ["s1", "s2"])
    assert list(f.columns) == [f"{s}_{stat}" for stat in STATS for s in ["s1", "s2"]]
    row = f.iloc[0]
    assert row["s1_mean"] == pytest.approx(2.5)
    assert row["s1_slope"] == pytest.approx(1.0)
    assert row["s1_last"] == 4 and row["s1_min"] == 1 and row["s1_max"] == 4
    assert row["s2_std"] == 0 and row["s2_slope"] == pytest.approx(0.0)
