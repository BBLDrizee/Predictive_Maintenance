"""Per-prediction sensor contributions (FR-10), in cycles of RUL.

- Linear: exact contributions, coefficient x (feature - training mean).
- XGBoost: TreeSHAP on the engineered features.
- LSTM / 1D-CNN: integrated gradients from an all-zeros window, which after
  per-condition normalisation is the training-average reading for every sensor.

Feature-level attributions are summed per sensor (over statistics or time steps).
Positive values push the predicted RUL up (healthier); negative values push it
down (closer to failure). Each method's contributions plus its base value add up
to the model's raw (unclipped) prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def per_sensor(feature_attr: np.ndarray, feature_names: list[str], sensors: list[str]) -> pd.DataFrame:
    """Sum attributions of features named '<sensor>_<stat>' per sensor."""
    owner = [name.rsplit("_", 1)[0] for name in feature_names]
    frame = pd.DataFrame(feature_attr, columns=owner)
    return frame.T.groupby(level=0).sum().T[sensors]


def linear_contributions(coef: np.ndarray, x_feat: pd.DataFrame, mean_feat: np.ndarray, sensors: list[str]) -> pd.DataFrame:
    return per_sensor((x_feat.to_numpy() - mean_feat) * coef, list(x_feat.columns), sensors)


def tree_shap(booster, x_feat: pd.DataFrame, sensors: list[str]) -> tuple[pd.DataFrame, float]:
    import shap

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(x_feat)
    return per_sensor(values, list(x_feat.columns), sensors), float(np.ravel(explainer.expected_value)[0])


def integrated_gradients(net: nn.Module, x: np.ndarray, scale: float, steps: int = 32) -> tuple[np.ndarray, float]:
    """Attributions (n, window_len, n_features) in cycles, and the prediction at the baseline."""
    net.eval()
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    base = torch.zeros_like(xt)
    total = torch.zeros_like(xt)
    for k in range(steps):
        alpha = (k + 0.5) / steps  # midpoint Riemann sum
        inp = (base + alpha * (xt - base)).requires_grad_(True)
        net(inp).sum().backward()
        total += inp.grad
    with torch.no_grad():
        base_value = float(net(base[:1]).item()) * scale
    return ((xt - base) * total / steps).numpy() * scale, base_value
