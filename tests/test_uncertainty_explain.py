"""Tests for k-fold splitting, calibrated ranges (FR-8), tiers (FR-9) and explanations (FR-10)."""

from __future__ import annotations

import joblib
import numpy as np
import pytest

from turborul.config import load_config
from turborul.data import SUBSETS
from turborul.pipeline import RULModel, prepare
from turborul.split import kfold_engines
from turborul.tiers import assign_tier, recommended_action

CFG = load_config()
needs_data = pytest.mark.skipif(not all((CFG["paths"]["raw_dir"] / f"train_{s}.txt").exists() for s in SUBSETS),
                                reason="C-MAPSS files not in data/raw")
SMALL = {
    "linear": {},
    "xgb": {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 3, "early_stopping_rounds": 10},
    "lstm": {"hidden": 16, "layers": 1, "dropout": 0.2, "lr": 3e-3, "batch_size": 512},
    "cnn": {"channels": 8, "n_conv": 2, "kernel": 3, "dropout": 0.2, "lr": 3e-3, "batch_size": 512},
}


@pytest.fixture(scope="module")
def data():
    return prepare(CFG, "FD001", 20, 125, include_test=True)


def _fit(name: str, data) -> RULModel:
    return RULModel(name, SMALL[name], data.sensors, data.window_len, data.rul_cap, seed=0,
                    training={"max_epochs": 2, "patience": 2}).fit(data).fit_interval(data, 0.1)


def test_kfold_engines_partition() -> None:
    folds = kfold_engines(np.arange(1, 101), 5, seed=0)
    vals = np.concatenate([v for _, v in folds])
    assert sorted(vals) == list(range(1, 101))
    for train, val in folds:
        assert set(train).isdisjoint(val) and len(train) + len(val) == 100


def test_tiers_and_actions() -> None:
    assert assign_tier([0, 30, 31, 60, 61], 30, 60).tolist() == ["Critical", "Critical", "Watch", "Watch", "Healthy"]
    assert "within 12 cycles" in recommended_action("Critical", 20, 12.7)
    assert "No action" in recommended_action("Healthy", 100, 80)


@needs_data
@pytest.mark.parametrize("name", ["linear", "xgb", "lstm", "cnn"])
def test_ranges_contain_point_and_are_calibrated(name: str, data) -> None:
    m = _fit(name, data)
    point, low, high = m.predict_interval(data.x_valcut)
    assert (low <= point).all() and (point <= high).all() and (low >= 0).all()
    # Conformal calibration guarantees the target coverage on the calibration set itself (capped truth).
    capped = np.minimum(data.y_valcut, data.rul_cap)
    assert np.mean((capped >= low) & (capped <= high)) >= 0.9


@needs_data
@pytest.mark.parametrize("name,tol", [("linear", 1e-3), ("xgb", 1e-2), ("lstm", 1.0), ("cnn", 3.0)])
def test_contributions_add_up_to_prediction(name: str, tol: float, data) -> None:
    m = _fit(name, data)
    x = data.x_test[:5]
    contrib, base = m.explain(x)
    raw = m.model.predict(m._inputs(x))
    assert list(contrib.columns) == data.sensors
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy() + base, raw, atol=tol)


@needs_data
@pytest.mark.parametrize("name", ["xgb", "cnn"])
def test_model_survives_save_and_load(name: str, data, tmp_path) -> None:
    m = _fit(name, data)
    joblib.dump(m, tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    for a, b in zip(m.predict_interval(data.x_test), loaded.predict_interval(data.x_test)):
        np.testing.assert_allclose(a, b, rtol=1e-5)
