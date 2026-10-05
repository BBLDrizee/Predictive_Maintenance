"""Load the YAML config and resolve its paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read a config file; entries under ``paths`` become absolute ``Path`` objects."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg.get("paths", {}).items()}
    return cfg
