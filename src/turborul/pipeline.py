"""Shared data preparation and the RUL model wrapper used by training, tuning, export and serving.

Protocol (PRD rules 1-3):
- Training engines are split by engine into train and validation (a fixed
  80/20 split, or one fold of an engine-level k-fold for tuning); the
  preprocessor is fitted on the train engines only.
- Early stopping uses every window of the validation engines (capped labels).
- Tuning, calibration and validation scores use a test-like set: validation
  engines cut at random points, scored on their last window against the true
  RUL. The official test set is only loaded when ``include_test`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from turborul import explain
from turborul.data import add_train_rul, load_subset
from turborul.features import window_features
from turborul.models.baselines import LinearModel, XGBModel
from turborul.models.deep import DeepModel
from turborul.preprocess import Preprocessor
from turborul.split import kfold_engines, split_frame, truncate_engines
from turborul.windows import last_windows, make_windows

CLASSICAL = ("linear", "xgb")
DEEP = ("lstm", "cnn")
MODELS = CLASSICAL + DEEP


@dataclass
class Prepared:
    subset: str
    window_len: int
    rul_cap: int | None
    sensors: list[str]
    preprocessor: Preprocessor
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray            # all validation windows, capped labels (early stopping)
    y_val: np.ndarray
    x_valcut: np.ndarray         # last windows of truncated validation engines (tuning, calibration)
    y_valcut: np.ndarray         # true, uncapped RUL
    x_test: np.ndarray | None = None
    y_test: np.ndarray | None = None  # official RUL_FDxxx truth, uncapped
    test_units: np.ndarray | None = None


@lru_cache(maxsize=32)
def _normalised(subset: str, raw_dir: str, n_conditions: int, trend_threshold: float, val_fraction: float,
                cuts: int, max_rul: int | None, seed: int, include_test: bool, fold: int | None, n_folds: int) -> tuple:
    """Load, split and normalise once per subset and fold; window length and RUL cap are applied later."""
    ds = load_subset(subset, raw_dir, rul_cap=None)
    if fold is None:
        train, val = split_frame(ds.train, val_fraction, seed)
    else:
        train_units, val_units = kfold_engines(ds.train["unit"].unique(), n_folds, seed)[fold]
        train = ds.train[ds.train["unit"].isin(train_units)].reset_index(drop=True)
        val = ds.train[ds.train["unit"].isin(val_units)].reset_index(drop=True)
    prep = Preprocessor(n_conditions, trend_threshold, seed).fit(train)
    # Cut at >= 1 cycle so short histories (padded) are represented, as in the official test set.
    valcut, valcut_truth = truncate_engines(val, cuts, min_cycles=1, max_rul=max_rul, seed=seed)
    test = prep.transform(ds.test) if include_test else None
    return prep, prep.transform(train), prep.transform(val), prep.transform(valcut), valcut_truth, test, ds.test_rul


def prepare(cfg: dict[str, Any], subset: str, window_len: int, rul_cap: int | None, include_test: bool = False,
            fold: int | None = None) -> Prepared:
    prep, train, val, valcut, valcut_truth, test, test_rul = _normalised(
        subset, str(cfg["paths"]["raw_dir"]), cfg["preprocess"]["n_conditions"][subset], cfg["preprocess"]["trend_threshold"],
        cfg["split"]["val_fraction"], cfg["split"]["val_cuts_per_engine"], cfg["split"]["val_max_rul"], cfg["seed"],
        include_test, fold, cfg["tuning"]["folds"],
    )
    sensors = prep.sensors
    x_tr, y_tr, _ = make_windows(add_train_rul(train, rul_cap), sensors, window_len)
    x_va, y_va, _ = make_windows(add_train_rul(val, rul_cap), sensors, window_len)
    x_vc, vc_units = last_windows(valcut, sensors, window_len)
    out = Prepared(subset, window_len, rul_cap, sensors, prep, x_tr, y_tr, x_va, y_va,
                   x_vc, valcut_truth.loc[vc_units].to_numpy(np.float32))
    if include_test:
        out.x_test, out.test_units = last_windows(test, sensors, window_len)
        out.y_test = test_rul.loc[out.test_units].to_numpy(np.float32)
    return out


class RULModel:
    """Uniform fit / predict / interval / explain over all model families.

    Point predictions are clipped to [0, rul_cap]. Intervals (FR-8) come from the
    model's own spread (MC dropout, XGBoost quantiles, or a constant width for the
    linear model), scaled by a split-conformal factor fitted on validation engines
    so that about (1 - alpha) of validation truths fall inside.
    """

    def __init__(self, name: str, params: dict[str, Any], sensors: list[str], window_len: int, rul_cap: int | None,
                 seed: int = 0, training: dict[str, Any] | None = None) -> None:
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
        self.name, self.params, self.sensors, self.window_len, self.rul_cap = name, params, list(sensors), window_len, rul_cap
        self.upper = rul_cap if rul_cap is not None else np.inf
        self.calibration: float | None = None
        self.alpha: float | None = None
        if name == "linear":
            self.model = LinearModel(seed=seed, **params)
        elif name == "xgb":
            self.model = XGBModel(seed=seed, **params)
        else:
            training = training or {"max_epochs": 80, "patience": 10}
            self.model = DeepModel(name, len(sensors), window_len, rul_cap or 125, seed=seed,
                                   max_epochs=training["max_epochs"], patience=training["patience"], **params)

    def _inputs(self, x: np.ndarray) -> np.ndarray | pd.DataFrame:
        return window_features(x, self.sensors) if self.name in CLASSICAL else x

    def fit(self, data: Prepared, **kwargs: Any) -> RULModel:
        self.model.fit(self._inputs(data.x_train), data.y_train, self._inputs(data.x_val), data.y_val, **kwargs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.clip(self.model.predict(self._inputs(x)), 0, self.upper)

    # --- uncertainty (FR-8) -------------------------------------------------

    def _raw_interval(self, x: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uncalibrated range, at least 1 cycle either side so the conformal scale stays finite."""
        if self.name == "xgb":
            q = np.clip(self.model.predict_quantiles(self._inputs(x)), 0, None)
            lo, hi = q[:, 0], q[:, 1]
        elif self.name in DEEP:
            samples = np.clip(self.model.predict_mc(x), 0, self.upper)
            lo, hi = np.quantile(samples, [self.alpha / 2, 1 - self.alpha / 2], axis=0)
        else:
            lo, hi = point, point  # linear: constant width, set entirely by calibration
        return np.minimum(lo, point - 1.0), np.maximum(hi, point + 1.0)

    def fit_interval(self, data: Prepared, alpha: float = 0.1) -> RULModel:
        """Fit quantiles if needed, then the conformal scale on the test-like validation set.

        Validation truth is clipped at the RUL cap: a capped model can only say "at
        least cap cycles", so long-life engines are not allowed to inflate every range.
        """
        self.alpha = alpha
        if self.name == "xgb":
            self.model.fit_quantiles(self._inputs(data.x_train), data.y_train, self._inputs(data.x_val), data.y_val, alpha)
        point = self.predict(data.x_valcut)
        lo, hi = self._raw_interval(data.x_valcut, point)
        y = np.minimum(data.y_valcut, self.upper)
        eps = 1e-6
        scores = np.maximum((point - y) / np.maximum(point - lo, eps), (y - point) / np.maximum(hi - point, eps))
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.calibration = float(np.quantile(scores, level, method="higher"))
        return self

    def predict_interval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(point, low, high) with the calibrated (1 - alpha) range."""
        if self.calibration is None:
            raise RuntimeError("call fit_interval first")
        point = self.predict(x)
        lo, hi = self._raw_interval(x, point)
        c = self.calibration
        return point, np.clip(point - c * (point - lo), 0, None), point + c * (hi - point)

    # --- explanations (FR-10) -----------------------------------------------

    def explain(self, x: np.ndarray) -> tuple[pd.DataFrame, float]:
        """Per-sensor contributions in cycles (n, n_sensors) and the base value they are measured from."""
        if self.name == "linear":
            feats = self._inputs(x)
            m = self.model
            contrib = explain.linear_contributions(m.model.coef_, feats, m.feature_means, self.sensors)
            base = float(m.model.intercept_ + m.model.coef_ @ m.feature_means)
            return contrib, base
        if self.name == "xgb":
            return explain.tree_shap(self.model.booster(), self._inputs(x), self.sensors)
        attr, base = explain.integrated_gradients(self.model.net, x, self.model.scale)
        return pd.DataFrame(attr.sum(axis=1), columns=self.sensors), base

    def size(self) -> dict[str, int]:
        if self.name == "xgb":
            return {"n_trees": self.model.n_trees}
        if self.name in DEEP:
            return {"n_params": self.model.n_params, "best_epoch": self.model.best_epoch + 1}
        return {"n_features": len(self.sensors) * 6}
