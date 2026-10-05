"""Tests for loading and RUL labelling (FR-1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turborul.config import load_config
from turborul.data import (
    COLUMNS,
    SUBSETS,
    add_test_rul,
    add_train_rul,
    load_subset,
    read_cmapss_file,
    read_rul_file,
)

RAW_DIR = load_config()["paths"]["raw_dir"]
HAS_DATA = all((RAW_DIR / f"train_{s}.txt").exists() for s in SUBSETS)
needs_data = pytest.mark.skipif(not HAS_DATA, reason="C-MAPSS files not in data/raw")


def _history(lengths: dict[int, int]) -> pd.DataFrame:
    """Minimal frame with one row per (unit, cycle)."""
    rows = [(u, c) for u, n in lengths.items() for c in range(1, n + 1)]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


# --- train labels -----------------------------------------------------------

def test_train_rul_counts_down_to_zero_without_cap() -> None:
    df = add_train_rul(_history({1: 5, 2: 3}), cap=None)
    assert df.loc[df.unit == 1, "rul"].tolist() == [4, 3, 2, 1, 0]
    assert df.loc[df.unit == 2, "rul"].tolist() == [2, 1, 0]


def test_train_rul_cap_gives_piecewise_linear_target() -> None:
    df = add_train_rul(_history({1: 200}), cap=125)
    rul = df["rul"].to_numpy()
    assert rul.max() == 125
    assert (rul[:75] == 125).all()                       # flat early life
    assert rul[75:].tolist() == list(range(124, -1, -1))  # then linear to 0


def test_train_rul_respects_custom_cap() -> None:
    assert add_train_rul(_history({1: 300}), cap=130)["rul"].max() == 130


def test_train_rul_does_not_mutate_input() -> None:
    df = _history({1: 3})
    add_train_rul(df)
    assert "rul" not in df.columns


# --- test labels ------------------------------------------------------------

def test_test_rul_last_cycle_matches_truth_and_counts_up_backwards() -> None:
    truth = pd.Series({1: 10, 2: 0})
    df = add_test_rul(_history({1: 4, 2: 2}), truth, cap=None)
    assert df.loc[df.unit == 1, "rul"].tolist() == [13, 12, 11, 10]
    assert df.loc[df.unit == 2, "rul"].tolist() == [1, 0]


def test_test_rul_cap() -> None:
    df = add_test_rul(_history({1: 50}), pd.Series({1: 100}), cap=125)
    assert df["rul"].max() == 125
    assert df["rul"].iloc[-1] == 100


def test_test_rul_missing_unit_raises() -> None:
    with pytest.raises(ValueError, match="no true RUL"):
        add_test_rul(_history({1: 3, 2: 3}), pd.Series({1: 5}))


# --- file parsing -----------------------------------------------------------

def test_read_file_handles_trailing_spaces_and_sorts(tmp_path: Path) -> None:
    sensors = " ".join(["1.0"] * 21)
    lines = [f"2 1 0.0 0.0 100.0 {sensors}  ", f"1 2 0.0 0.0 100.0 {sensors} ", f"1 1 0.0 0.0 100.0 {sensors}"]
    path = tmp_path / "train_X.txt"
    path.write_text("\n".join(lines) + "\n")
    df = read_cmapss_file(path)
    assert list(df.columns) == COLUMNS
    assert df[["unit", "cycle"]].values.tolist() == [[1, 1], [1, 2], [2, 1]]
    assert df["unit"].dtype.kind == "i"


def test_read_file_rejects_wrong_column_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("1 1 0.0\n")
    with pytest.raises(ValueError, match="expected 26 columns"):
        read_cmapss_file(path)


def test_read_rul_file_indexes_by_unit(tmp_path: Path) -> None:
    path = tmp_path / "RUL_X.txt"
    path.write_text("112 \n98 \n69 \n")
    rul = read_rul_file(path)
    assert rul.index.tolist() == [1, 2, 3]
    assert rul.tolist() == [112, 98, 69]


# --- real data (skipped when files are absent) --------------------------------

# (train, test) engines as found in the files. The C-MAPSS readme (and PRD §4) swap FD004's
# counts as 248/249; the data has 249 training engines and 248 test engines with 248 RUL values.
EXPECTED_ENGINES = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}


@needs_data
@pytest.mark.parametrize("name", SUBSETS)
def test_real_subset_shapes_and_labels(name: str) -> None:
    ds = load_subset(name, RAW_DIR, rul_cap=125)
    n_train, n_test = EXPECTED_ENGINES[name]
    assert ds.train["unit"].nunique() == n_train
    assert ds.test["unit"].nunique() == n_test
    assert len(ds.test_rul) == n_test
    assert not ds.train[COLUMNS].isna().any().any()

    # Every training engine runs to failure: last cycle has RUL 0.
    assert (ds.train.groupby("unit")["rul"].last() == 0).all()
    assert ds.train["rul"].between(0, 125).all()

    # Test labels at the last cycle equal the official truth (clipped at the cap).
    last = ds.test.groupby("unit")["rul"].last()
    np.testing.assert_array_equal(last.to_numpy(), ds.test_rul.clip(upper=125).to_numpy())

    # Cycles are contiguous from 1 within each engine.
    for df in (ds.train, ds.test):
        g = df.groupby("unit")["cycle"]
        assert (g.min() == 1).all() and (g.max() == g.count()).all()
