"""Predict RUL for any engine from its raw sensor history (FR-7, FR-8, FR-9, FR-10).

    from turborul.predict import load_bundle, predict
    result = predict(history_df, load_bundle("FD001"))
    # {"rul": 42.1, "low": 30.5, "high": 55.0, "tier": "Watch", "action": "...", "top_sensors": [...], ...}

``history`` holds one row per cycle with columns cycle, op_setting_1..3 and s1..s21
(raw units, as in the C-MAPSS files). Only the latest window is scored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from turborul.config import load_config
from turborul.data import OP_SETTINGS, SENSOR_INFO, SENSORS
from turborul.pipeline import RULModel
from turborul.preprocess import Preprocessor
from turborul.tiers import assign_tier, recommended_action
from turborul.windows import last_windows

REQUIRED = ["cycle", *OP_SETTINGS, *SENSORS]


@dataclass
class Bundle:
    subset: str
    meta: dict[str, Any]
    model: RULModel
    preprocessor: Preprocessor

    @property
    def critical(self) -> float:
        return self.meta["tiers"]["critical"]

    @property
    def watch(self) -> float:
        return self.meta["tiers"]["watch"]


@lru_cache(maxsize=8)
def load_bundle(subset: str, models_dir: str | None = None) -> Bundle:
    root = Path(models_dir) if models_dir else load_config()["paths"]["models_dir"]
    d = root / subset
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"no exported model for {subset} in {root}; run python -m turborul.export")
    return Bundle(subset, json.loads((d / "meta.json").read_text()), joblib.load(d / "model.joblib"),
                  joblib.load(d / "preprocessor.joblib"))


def available_subsets(models_dir: str | None = None) -> list[str]:
    root = Path(models_dir) if models_dir else load_config()["paths"]["models_dir"]
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists()) if root.exists() else []


def _normalise(history: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if history.empty:
        raise ValueError("history is empty")
    h = history.sort_values("cycle").assign(unit=history["unit"] if "unit" in history else 0)
    return bundle.preprocessor.transform(h)


def predict_many(histories: pd.DataFrame, bundle: Bundle, explain: bool = False) -> pd.DataFrame:
    """Latest-cycle prediction for every unit in ``histories`` (one row per unit)."""
    norm = _normalise(histories, bundle)
    x, units = last_windows(norm, bundle.model.sensors, bundle.model.window_len)
    rul, low, high = bundle.model.predict_interval(x)
    out = pd.DataFrame({"unit": units, "cycle": norm.groupby("unit")["cycle"].max().loc[units].to_numpy(),
                        "rul": rul.astype(float), "low": low.astype(float), "high": high.astype(float)})
    out["tier"] = assign_tier(out["rul"], bundle.critical, bundle.watch)
    out["action"] = [recommended_action(t, r, lo) for t, r, lo in zip(out.tier, out.rul, out.low)]
    if explain:
        contrib, base = bundle.model.explain(x)
        out["contributions"] = [row.to_dict() for _, row in contrib.iterrows()]
        out["base_value"] = base
    return out


def _top_sensors(contrib: dict[str, float], k: int) -> list[dict[str, Any]]:
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [{"sensor": s, "symbol": SENSOR_INFO[s][0], "name": SENSOR_INFO[s][1], "contribution": round(float(v), 2)}
            for s, v in ranked]


def predict(history: pd.DataFrame, bundle: Bundle, top_k: int = 5) -> dict[str, Any]:
    """Prediction for one engine at its latest cycle, with range, tier, action and top contributing sensors."""
    if "unit" in history and history["unit"].nunique() > 1:
        raise ValueError("history contains more than one engine; use predict_many")
    row = predict_many(history, bundle, explain=True).iloc[0]
    return {
        "subset": bundle.subset, "model": bundle.meta["model"], "cycle": int(row.cycle),
        "rul": round(float(row.rul), 1), "low": round(float(row.low), 1), "high": round(float(row.high), 1),
        "confidence": 1 - bundle.meta["alpha"], "tier": str(row.tier), "action": row.action,
        "base_value": round(float(row.base_value), 1), "top_sensors": _top_sensors(row.contributions, top_k),
    }


def predict_trajectory(history: pd.DataFrame, bundle: Bundle) -> pd.DataFrame:
    """Predicted RUL and range at every cycle of one engine's history, as it would have been seen live."""
    norm = _normalise(history, bundle)
    values = norm[bundle.model.sensors].to_numpy(np.float32)
    L = bundle.model.window_len
    padded = np.concatenate([np.repeat(values[:1], L - 1, axis=0), values])
    x = sliding_window_view(padded, L, axis=0).transpose(0, 2, 1)
    rul, low, high = bundle.model.predict_interval(np.ascontiguousarray(x))
    return pd.DataFrame({"cycle": norm["cycle"].to_numpy(), "rul": rul.astype(float), "low": low.astype(float),
                         "high": high.astype(float)})
