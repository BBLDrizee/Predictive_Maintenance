"""Tests for the engine-level split and validation truncation (FR-3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turborul.split import split_engines, split_frame, truncate_engines


def _history(lengths: dict[int, int]) -> pd.DataFrame:
    rows = [(u, c) for u, n in lengths.items() for c in range(1, n + 1)]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


def test_split_engines_disjoint_and_complete() -> None:
    train, val = split_engines(np.arange(1, 101), 0.2, seed=0)
    assert len(val) == 20 and len(train) == 80
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(range(1, 101))


def test_split_engines_deterministic_per_seed() -> None:
    a = split_engines(np.arange(1, 101), 0.2, seed=1)
    b = split_engines(np.arange(1, 101), 0.2, seed=1)
    c = split_engines(np.arange(1, 101), 0.2, seed=2)
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_engines_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        split_engines([1, 2, 3], 1.0, seed=0)


def test_split_frame_never_shares_an_engine() -> None:
    df = _history({u: 50 + u for u in range(1, 31)})
    train, val = split_frame(df, 0.2, seed=0)
    assert set(train["unit"]).isdisjoint(val["unit"])
    assert len(train) + len(val) == len(df)
    # Whole engines move together: every engine keeps all its rows.
    for part in (train, val):
        sizes = part.groupby("unit").size()
        assert (sizes == 50 + sizes.index).all()


def test_truncate_engines_truth_and_bounds() -> None:
    df = _history({1: 200, 2: 100})
    cut, truth = truncate_engines(df, cuts_per_engine=50, min_cycles=1, max_rul=150, seed=0)
    assert truth.index.tolist() == list(range(1, 101))
    lengths = cut.groupby("unit")["cycle"].max()
    life = np.repeat([200, 100], 50)
    np.testing.assert_array_equal(truth.to_numpy(), life - lengths.to_numpy())
    assert truth.between(1, 150).all()
    assert (cut.groupby("unit")["cycle"].min() == 1).all()


def test_truncate_engines_without_max_rul_can_cut_early() -> None:
    _, truth = truncate_engines(_history({1: 300}), cuts_per_engine=200, min_cycles=1, max_rul=None, seed=0)
    assert truth.max() > 150
