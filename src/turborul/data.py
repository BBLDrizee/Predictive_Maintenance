"""Load C-MAPSS subsets and compute RUL labels (FR-1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
OP_SETTINGS: list[str] = [f"op_setting_{i}" for i in range(1, 4)]
SENSORS: list[str] = [f"s{i}" for i in range(1, 22)]
COLUMNS: list[str] = ["unit", "cycle", *OP_SETTINGS, *SENSORS]

# Symbol and description of each sensor (Saxena et al., PHM08, Table 2).
SENSOR_INFO: dict[str, tuple[str, str]] = {
    "s1": ("T2", "Fan inlet temperature"),
    "s2": ("T24", "LPC outlet temperature"),
    "s3": ("T30", "HPC outlet temperature"),
    "s4": ("T50", "LPT outlet temperature"),
    "s5": ("P2", "Fan inlet pressure"),
    "s6": ("P15", "Bypass-duct pressure"),
    "s7": ("P30", "HPC outlet pressure"),
    "s8": ("Nf", "Physical fan speed"),
    "s9": ("Nc", "Physical core speed"),
    "s10": ("epr", "Engine pressure ratio"),
    "s11": ("Ps30", "HPC outlet static pressure"),
    "s12": ("phi", "Fuel flow to Ps30 ratio"),
    "s13": ("NRf", "Corrected fan speed"),
    "s14": ("NRc", "Corrected core speed"),
    "s15": ("BPR", "Bypass ratio"),
    "s16": ("farB", "Burner fuel-air ratio"),
    "s17": ("htBleed", "Bleed enthalpy"),
    "s18": ("Nf_dmd", "Demanded fan speed"),
    "s19": ("PCNfR_dmd", "Demanded corrected fan speed"),
    "s20": ("W31", "HPT coolant bleed"),
    "s21": ("W32", "LPT coolant bleed"),
}


def sensor_label(sensor: str) -> str:
    """'s11' -> 's11 · Ps30 (HPC outlet static pressure)'."""
    symbol, desc = SENSOR_INFO[sensor]
    return f"{sensor} · {symbol} ({desc})"


@dataclass
class CMAPSSSubset:
    """One C-MAPSS subset: training histories, truncated test histories and test RUL."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # true RUL at each test engine's last cycle, indexed by unit


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read a train/test file into a frame with named columns.

    The raw files are space-separated with no header and trailing spaces, which
    the whitespace separator absorbs; any empty trailing columns are dropped.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    return df.sort_values(["unit", "cycle"]).reset_index(drop=True)


def read_rul_file(path: str | Path) -> pd.Series:
    """Read ``RUL_FDxxx.txt``: one true RUL per test engine, in engine order (units 1..n)."""
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(int)
    return pd.Series(values.to_numpy(), index=pd.RangeIndex(1, len(values) + 1, name="unit"), name="rul")


def add_train_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Label run-to-failure histories: ``RUL = max_cycle - cycle``, clipped at ``cap``."""
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def add_test_rul(df: pd.DataFrame, rul_at_last: pd.Series, cap: int | None = 125) -> pd.DataFrame:
    """Label every cycle of truncated test histories from the true RUL at their last cycle.

    Only the last cycle is used for scoring; the earlier labels let the dashboard
    overlay true RUL across an engine's observed life.
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    missing = set(out["unit"].unique()) - set(rul_at_last.index)
    if missing:
        raise ValueError(f"no true RUL for test units {sorted(missing)[:5]}")
    out["rul"] = out["unit"].map(rul_at_last) + (last_cycle - out["cycle"])
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out


def load_subset(name: str, raw_dir: str | Path, rul_cap: int | None = 125) -> CMAPSSSubset:
    """Load one subset with RUL labels on both train and test rows."""
    if name not in SUBSETS:
        raise ValueError(f"unknown subset {name!r}; expected one of {SUBSETS}")
    raw_dir = Path(raw_dir)
    train = read_cmapss_file(raw_dir / f"train_{name}.txt")
    test = read_cmapss_file(raw_dir / f"test_{name}.txt")
    test_rul = read_rul_file(raw_dir / f"RUL_{name}.txt")
    if test["unit"].nunique() != len(test_rul):
        raise ValueError(f"{name}: {test['unit'].nunique()} test engines but {len(test_rul)} RUL values")
    return CMAPSSSubset(
        name=name,
        train=add_train_rul(train, rul_cap),
        test=add_test_rul(test, test_rul, rul_cap),
        test_rul=test_rul,
    )


def load_all(raw_dir: str | Path, rul_cap: int | None = 125) -> dict[str, CMAPSSSubset]:
    """Load all four subsets, keyed by name."""
    return {name: load_subset(name, raw_dir, rul_cap) for name in SUBSETS}
