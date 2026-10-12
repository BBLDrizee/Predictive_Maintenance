"""Tests for sensor selection and per-condition normalisation (FR-2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turborul.config import load_config
from turborul.data import OP_SETTINGS, SENSORS, SUBSETS, load_subset
from turborul.preprocess import Preprocessor
from turborul.split import split_frame

RAW_DIR = load_config()["paths"]["raw_dir"]
needs_data = pytest.mark.skipif(not all((RAW_DIR / f"train_{s}.txt").exists() for s in SUBSETS),
                                reason="C-MAPSS files not in data/raw")


def _synthetic(n_units: int = 20, life: int = 60, conditions: int = 2, seed: int = 0) -> pd.DataFrame:
    """s1 trends with cycle, s2 is pure noise, s3 is constant; condition shifts every sensor level."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, life + 1):
            k = rng.integers(conditions)
            rows.append({"unit": u, "cycle": c, "op_setting_1": 10.0 * k, "op_setting_2": 0.1 * k, "op_setting_3": 100.0,
                         **{s: 0.0 for s in SENSORS}, "s1": 100 * k + 0.05 * c + rng.normal(0, 0.1),
                         "s2": 100 * k + rng.normal(0, 1), "s3": 7.0})
    return pd.DataFrame(rows)


def test_selects_trending_sensor_only() -> None:
    prep = Preprocessor(n_conditions=2, trend_threshold=0.2, seed=0).fit(_synthetic())
    assert prep.sensors == ["s1"]
    assert np.isnan(prep.strength["s3"])


def test_normalisation_removes_condition_offsets() -> None:
    df = _synthetic()
    out = Preprocessor(n_conditions=2, trend_threshold=0.2, seed=0).fit_transform(df)
    by_cond = out.groupby("condition")["s1"].agg(["mean", "std"])
    np.testing.assert_allclose(by_cond["mean"], 0, atol=1e-9)
    np.testing.assert_allclose(by_cond["std"], 1, atol=1e-9)
    assert list(out.columns[:3]) == ["unit", "cycle", "condition"]


def test_transform_uses_training_statistics_only() -> None:
    train, other = _synthetic(seed=0), _synthetic(seed=1)
    prep = Preprocessor(n_conditions=2, seed=0).fit(train)
    means_before = prep.means.copy()
    shifted = other.assign(s1=other["s1"] + 50)
    out = prep.transform(shifted)
    pd.testing.assert_frame_equal(prep.means, means_before)
    assert out["s1"].mean() > 10  # a shift unseen in training is not normalised away


def test_transform_requires_fit() -> None:
    with pytest.raises(RuntimeError):
        Preprocessor().transform(_synthetic(n_units=2))


@needs_data
def test_fd001_selection_matches_eda() -> None:
    train, _ = split_frame(load_subset("FD001", RAW_DIR).train, 0.2, seed=42)
    prep = Preprocessor(n_conditions=1, trend_threshold=0.2).fit(train)
    dropped = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
    assert set(prep.sensors) == set(SENSORS) - dropped


@needs_data
def test_fd002_conditions_recovered() -> None:
    train = load_subset("FD002", RAW_DIR).train
    prep = Preprocessor(n_conditions=6, seed=0).fit(train)
    cond = prep.transform(train)["condition"]
    # Each k-means cluster should be one tight operating point.
    spread = train.groupby(cond.to_numpy())[OP_SETTINGS].std().max().max()
    assert spread < 0.01
