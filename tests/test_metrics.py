"""Tests for RMSE, the NASA score and critical recall (FR-6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from turborul.metrics import critical_recall, nasa_score, rmse


def test_rmse() -> None:
    assert rmse([0, 0], [3, 4]) == pytest.approx(math.sqrt(12.5))
    assert rmse([5, 5], [5, 5]) == 0


def test_nasa_score_penalises_late_more_than_early() -> None:
    early = nasa_score([50], [40])   # d = -10
    late = nasa_score([50], [60])    # d = +10
    assert early == pytest.approx(math.exp(10 / 13) - 1)
    assert late == pytest.approx(math.e - 1)
    assert late > early
    assert nasa_score([10, 20], [10, 20]) == 0


def test_critical_recall() -> None:
    y_true = np.array([10, 25, 30, 31, 100])
    y_pred = np.array([12, 40, 30, 5, 100])
    assert critical_recall(y_true, y_pred, 30) == pytest.approx(2 / 3)
    assert math.isnan(critical_recall(np.array([100]), np.array([5]), 30))
