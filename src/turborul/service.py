"""JSON-ready queries behind the REST API and the dashboard.

The "fleet" of a subset is its official test engines: in-service histories cut
off before failure. Their true RUL is known, so the dashboard can show errors
honestly next to each prediction.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from turborul.config import load_config
from turborul.data import SENSOR_INFO, load_subset
from turborul.predict import available_subsets, load_bundle, predict, predict_many, predict_trajectory
from turborul.reporting import MODEL_LABEL, bucket_errors, latest_runs, predictions, summarise
from turborul.tiers import assign_tier


@lru_cache(maxsize=4)
def _test_data(subset: str) -> tuple[pd.DataFrame, pd.Series]:
    ds = load_subset(subset, load_config()["paths"]["raw_dir"], rul_cap=None)
    return ds.test, ds.test_rul


def subsets() -> list[dict[str, Any]]:
    out = []
    for s in available_subsets():
        meta = load_bundle(s).meta
        out.append({"subset": s, "model": meta["model"], "model_label": MODEL_LABEL[meta["model"]],
                    "params_source": meta["params_source"], "window_len": meta["window_len"], "rul_cap": meta["rul_cap"],
                    "sensors": meta["sensors"], "test_metrics": meta["test_metrics"], "created": meta["created"]})
    return out


def fleet_frame(subset: str, use_cache: bool = True) -> pd.DataFrame:
    """Scores, ranges, tiers and explanations for every test engine.

    Read from models/<subset>/fleet.json (written by turborul.export) when present
    and newer than the bundle; otherwise computed from the model.
    """
    d = load_config()["paths"]["models_dir"] / subset
    cached = d / "fleet.json"
    if use_cache and cached.exists() and cached.stat().st_mtime >= (d / "model.joblib").stat().st_mtime:
        return pd.read_json(cached, orient="records")
    bundle = load_bundle(subset)
    test, truth = _test_data(subset)
    df = predict_many(test, bundle, explain=True)
    df["true_rul"] = truth.loc[df.unit].to_numpy()
    df["true_tier"] = assign_tier(df.true_rul, bundle.critical, bundle.watch)
    return df.sort_values(["rul", "unit"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _fleet(subset: str) -> pd.DataFrame:
    return fleet_frame(subset)


def fleet(subset: str, tier: str | None = None) -> list[dict[str, Any]]:
    """Every in-service engine, lowest predicted RUL first."""
    df = _fleet(subset).drop(columns=["contributions", "base_value"])
    if tier:
        df = df[df.tier == tier]
    return df.round({"rul": 1, "low": 1, "high": 1}).to_dict("records")


def global_importance(subset: str) -> list[dict[str, Any]]:
    """Mean absolute contribution (cycles) of each sensor across the fleet."""
    contrib = pd.DataFrame(list(_fleet(subset).contributions))
    imp = contrib.abs().mean().sort_values(ascending=False)
    return [{"sensor": s, "symbol": SENSOR_INFO[s][0], "name": SENSOR_INFO[s][1], "mean_abs_contribution": round(float(v), 2)}
            for s, v in imp.items()]


def engine(subset: str, unit: int, n_sensors: int = 6) -> dict[str, Any]:
    """Prediction, explanation, RUL trajectory and sensor trends for one in-service engine."""
    bundle = load_bundle(subset)
    test, truth = _test_data(subset)
    history = test[test.unit == unit]
    if history.empty:
        raise KeyError(f"no engine {unit} in {subset}")
    result = predict(history, bundle, top_k=8)
    traj = predict_trajectory(history, bundle)
    traj["true_rul"] = history["rul"].to_numpy()  # uncapped truth, counted back from the official value
    top = [t["sensor"] for t in result["top_sensors"][:n_sensors]]
    norm = bundle.preprocessor.transform(history)
    trends = {s: {"symbol": SENSOR_INFO[s][0], "name": SENSOR_INFO[s][1],
                  "value": norm[s].round(3).tolist(), "smoothed": norm[s].ewm(span=10).mean().round(3).tolist()}
              for s in top}
    return {
        **result, "unit": int(unit), "true_rul": float(truth.loc[unit]),
        "true_tier": str(assign_tier(truth.loc[unit], bundle.critical, bundle.watch)),
        "trajectory": traj.round(2).to_dict("records"),
        "sensor_trends": {"cycle": norm["cycle"].tolist(), "sensors": trends,
                          "units": "z-score within operating condition (0 = training average)"},
    }


def predict_rows(subset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return predict(pd.DataFrame(rows), load_bundle(subset))


def model_comparison() -> dict[str, Any]:
    """Final configuration per (subset, model) plus per-bucket errors, for the comparison screen."""
    cfg = load_config()
    s = summarise(latest_runs(cfg["paths"]["results_csv"]))
    finals = s[s.final].copy()
    finals["model_label"] = finals.model.map(MODEL_LABEL)
    buckets = []
    for (subset, model, source), _ in finals.groupby(["subset", "model", "params_source"]):
        preds = predictions(cfg["paths"]["predictions_dir"], subset, model, source, cfg["seeds"])
        buckets.append(bucket_errors({model: preds}).assign(subset=subset))
    return {"summary": finals.replace({np.nan: None}).to_dict("records"),
            "buckets": pd.concat(buckets).to_dict("records") if buckets else []}


def test_predictions(subset: str, model: str) -> list[dict[str, Any]]:
    """Seed-0 test predictions of a model's final configuration (for the predicted-vs-true scatter)."""
    cfg = load_config()
    s = summarise(latest_runs(cfg["paths"]["results_csv"], subset))
    row = s[s.final & (s.model == model)]
    if row.empty:
        raise KeyError(f"no results for {model} on {subset}")
    preds = predictions(cfg["paths"]["predictions_dir"], subset, model, row.iloc[0].params_source, cfg["seeds"][:1])
    return preds.round(2).to_dict("records")
