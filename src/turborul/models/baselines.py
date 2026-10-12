"""Linear regression and XGBoost on engineered window features (FR-4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


class LinearModel:
    """Ordinary least squares baseline."""

    def __init__(self, seed: int = 0, **_: Any) -> None:
        self.model = LinearRegression()
        self.feature_means: np.ndarray | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None) -> LinearModel:
        self.model.fit(x, y)
        self.feature_means = x.mean().to_numpy()
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)


class XGBModel:
    """Gradient-boosted trees; the number of trees is chosen by early stopping on validation engines.

    ``fit_quantiles`` adds a second model for the lower and upper quantiles (FR-8).
    """

    def __init__(self, seed: int = 0, early_stopping_rounds: int = 100, **params: Any) -> None:
        self.params = dict(objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=-1,
                           early_stopping_rounds=early_stopping_rounds, **params)
        self.model = XGBRegressor(**self.params)
        self.quantile_model: XGBRegressor | None = None

    def fit(self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None) -> XGBModel:
        if x_val is None or y_val is None:
            raise ValueError("XGBModel needs validation data for early stopping")
        self.model.fit(x, y, eval_set=[(x_val, y_val)], verbose=False)
        return self

    def fit_quantiles(self, x: pd.DataFrame, y: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray, alpha: float) -> None:
        params = {**self.params, "objective": "reg:quantileerror", "quantile_alpha": np.array([alpha / 2, 1 - alpha / 2])}
        self.quantile_model = XGBRegressor(**params).fit(x, y, eval_set=[(x_val, y_val)], verbose=False)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)

    def predict_quantiles(self, x: pd.DataFrame) -> np.ndarray:
        """(n, 2) lower and upper quantiles."""
        if self.quantile_model is None:
            raise RuntimeError("call fit_quantiles first")
        return self.quantile_model.predict(x)

    def booster(self):
        """The booster truncated at the early-stopping iteration, matching ``predict``."""
        return self.model.get_booster()[: self.n_trees]

    @property
    def n_trees(self) -> int:
        return int(self.model.best_iteration) + 1
