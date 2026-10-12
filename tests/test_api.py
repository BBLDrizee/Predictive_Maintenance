"""Tests for the REST API (FR-7) against the exported models in models/ (skipped until they exist)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.main import app
from turborul.config import load_config
from turborul.data import load_subset

CFG = load_config()
pytestmark = pytest.mark.skipif(not (CFG["paths"]["models_dir"] / "FD001" / "meta.json").exists(),
                                reason="no exported models; run python -m turborul.export")
client = TestClient(app)


def _history(unit: int) -> list[dict]:
    test = load_subset("FD001", CFG["paths"]["raw_dir"]).test
    return test[test.unit == unit].drop(columns=["unit", "rul"]).to_dict("records")


def test_health_lists_models() -> None:
    r = client.get("/health")
    assert r.status_code == 200 and "FD001" in r.json()["models"]


def test_fleet_sorted_by_predicted_rul_with_tiers() -> None:
    rows = client.get("/fleet/FD001").json()
    assert len(rows) == 100
    ruls = [r["rul"] for r in rows]
    assert ruls == sorted(ruls)
    critical = client.get("/fleet/FD001", params={"tier": "Critical"}).json()
    assert critical and all(r["tier"] == "Critical" and r["rul"] <= 30 for r in critical)


def test_predict_endpoint_matches_fleet_and_is_fast() -> None:
    start = time.perf_counter()
    r = client.post("/predict", json={"subset": "FD001", "history": _history(7)})
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["low"] <= out["rul"] <= out["high"] and out["top_sensors"]
    fleet = {row["unit"]: row for row in client.get("/fleet/FD001").json()}
    assert out["rul"] == pytest.approx(fleet[7]["rul"], abs=0.1)
    assert elapsed < 2  # PRD: a single prediction in < 2 s


def test_engine_detail() -> None:
    e = client.get("/engines/FD001/7").json()
    assert {"trajectory", "sensor_trends", "top_sensors", "true_rul", "action"} <= set(e)
    assert len(e["trajectory"]) == e["cycle"]


def test_errors() -> None:
    assert client.get("/fleet/FD009").status_code == 404
    assert client.get("/engines/FD001/9999").status_code == 404
    bad = _history(7)
    del bad[0]["s3"]
    assert client.post("/predict", json={"subset": "FD001", "history": bad}).status_code == 422
    assert client.post("/predict", json={"subset": "FD001", "history": []}).status_code == 422


def test_results_endpoint() -> None:
    res = client.get("/results").json()
    assert res["summary"] and res["buckets"]
    assert client.get("/results/FD001/xgb/predictions").status_code == 200
