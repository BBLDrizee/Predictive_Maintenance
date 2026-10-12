"""Tests for the prediction function (FR-7) using a small bundle built in a temp directory."""

from __future__ import annotations

import json
import time

import joblib
import pytest

from turborul.config import load_config
from turborul.data import SUBSETS, load_subset
from turborul.pipeline import RULModel, prepare
from turborul.predict import load_bundle, predict, predict_many, predict_trajectory

CFG = load_config()
pytestmark = pytest.mark.skipif(not all((CFG["paths"]["raw_dir"] / f"train_{s}.txt").exists() for s in SUBSETS),
                                reason="C-MAPSS files not in data/raw")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    data = prepare(CFG, "FD001", 20, 125, include_test=True)
    m = RULModel("xgb", {"n_estimators": 60, "learning_rate": 0.1, "max_depth": 3, "early_stopping_rounds": 10},
                 data.sensors, 20, 125, seed=0).fit(data).fit_interval(data, 0.1)
    (root / "FD001").mkdir()
    joblib.dump(m, root / "FD001" / "model.joblib")
    joblib.dump(data.preprocessor, root / "FD001" / "preprocessor.joblib")
    meta = {"model": "xgb", "alpha": 0.1, "tiers": CFG["tiers"], "window_len": 20, "rul_cap": 125}
    (root / "FD001" / "meta.json").write_text(json.dumps(meta))
    return load_bundle("FD001", str(root))


@pytest.fixture(scope="module")
def test_engines():
    return load_subset("FD001", CFG["paths"]["raw_dir"], rul_cap=None).test


def test_predict_one_engine(bundle, test_engines) -> None:
    history = test_engines[test_engines.unit == 1]
    start = time.perf_counter()
    out = predict(history, bundle)
    assert time.perf_counter() - start < 2  # PRD: a single prediction in < 2 s
    assert set(out) >= {"rul", "low", "high", "tier", "action", "top_sensors", "cycle"}
    assert out["low"] <= out["rul"] <= out["high"]
    assert out["cycle"] == int(history.cycle.max())
    assert out["tier"] in ("Critical", "Watch", "Healthy")
    assert len(out["top_sensors"]) == 5 and all(t["sensor"] in bundle.model.sensors for t in out["top_sensors"])


def test_predict_without_unit_column_and_unsorted(bundle, test_engines) -> None:
    history = test_engines[test_engines.unit == 2]
    a = predict(history, bundle)
    b = predict(history.drop(columns="unit").sample(frac=1, random_state=0), bundle)
    assert a["rul"] == b["rul"]


def test_predict_many_matches_single(bundle, test_engines) -> None:
    many = predict_many(test_engines[test_engines.unit.isin([3, 4])], bundle)
    single = predict(test_engines[test_engines.unit == 4], bundle)
    assert many.set_index("unit").loc[4, "rul"] == pytest.approx(single["rul"], abs=0.05)


def test_trajectory_has_one_row_per_cycle_and_ends_at_latest(bundle, test_engines) -> None:
    history = test_engines[test_engines.unit == 5]
    traj = predict_trajectory(history, bundle)
    assert len(traj) == len(history)
    assert traj.rul.iloc[-1] == pytest.approx(predict(history, bundle)["rul"], abs=0.05)


def test_bad_inputs(bundle, test_engines) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        predict(test_engines[test_engines.unit == 1].drop(columns="s4"), bundle)
    with pytest.raises(ValueError, match="more than one engine"):
        predict(test_engines[test_engines.unit.isin([1, 2])], bundle)
