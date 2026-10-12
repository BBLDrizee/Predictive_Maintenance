"""Evaluation metrics (FR-6). Lower is better for RMSE and the NASA score."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sqrt(np.mean(d**2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PHM08 asymmetric score: late predictions (d > 0) are penalised more than early ones."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def critical_recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 30) -> float:
    """Share of engines with true RUL <= threshold that are predicted <= threshold (flagged Critical)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    critical = y_true <= threshold
    if not critical.any():
        return float("nan")
    return float(np.mean(y_pred[critical] <= threshold))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, critical_threshold: float = 30) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
        "critical_recall": critical_recall(y_true, y_pred, critical_threshold),
    }
