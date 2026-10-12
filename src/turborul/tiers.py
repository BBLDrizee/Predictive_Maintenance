"""Risk tiers (FR-9) and plain-language maintenance actions."""

from __future__ import annotations

import numpy as np

TIERS: tuple[str, ...] = ("Critical", "Watch", "Healthy")


def assign_tier(rul: np.ndarray | float, critical: float = 30, watch: float = 60) -> np.ndarray:
    """Critical if RUL <= critical, Watch if <= watch, else Healthy."""
    rul = np.asarray(rul, dtype=float)
    return np.where(rul <= critical, "Critical", np.where(rul <= watch, "Watch", "Healthy"))


def recommended_action(tier: str, rul: float, low: float) -> str:
    """One sentence a planner can act on. Uses the lower bound of the range to stay on the safe side."""
    if tier == "Critical":
        if low < 1:
            return "Schedule a shop visit immediately (the lower end of the predicted range is 0 cycles)."
        return f"Schedule a shop visit within {int(low)} cycles (lower end of the predicted range)."
    if tier == "Watch":
        return f"Plan a shop visit at the next maintenance slot; the engine may have as few as {max(0, int(low))} cycles left."
    return f"No action needed. Continue routine monitoring (about {int(rul)} cycles left)."
