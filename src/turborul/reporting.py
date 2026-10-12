"""Summaries of results/ shared by the dashboard, the API and the progress report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = ["linear", "xgb", "lstm", "cnn"]
MODEL_LABEL = {"linear": "Linear regression", "xgb": "XGBoost", "lstm": "LSTM", "cnn": "1D-CNN"}
SOURCE_ORDER = ["default", "tuned", "tuned-cv", "transfer"]
BUCKETS = [(0, 30), (31, 60), (61, 90), (91, 125), (126, 1000)]
BUCKET_LABEL = ["0–30", "31–60", "61–90", "91–125", "> 125"]


def latest_runs(results_csv: Path, subset: str | None = None) -> pd.DataFrame:
    """The most recent run of each (subset, model, params_source, seed)."""
    df = pd.read_csv(results_csv)
    if subset is not None:
        df = df[df["subset"] == subset]
    df = df.sort_values("timestamp")
    return df.drop_duplicates(["subset", "model", "params_source", "seed"], keep="last").reset_index(drop=True)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean/std per (subset, model, params_source), with the final configuration of each family flagged.

    Final = the cross-validation-tuned configuration (or, for the LSTM on FD002-FD004,
    the FD001-tuned one transferred) where one exists - the protocol fixed in Phase 4;
    otherwise the configuration with the lowest validation RMSE.
    """
    g = runs.groupby(["subset", "model", "params_source"])
    agg = {
        "seeds": ("seed", "nunique"), "window_len": ("window_len", "first"), "rul_cap": ("rul_cap", "first"),
        "val_rmse": ("val_rmse", "mean"), "test_rmse": ("test_rmse", "mean"), "test_rmse_std": ("test_rmse", "std"),
        "nasa": ("test_nasa_score", "mean"), "nasa_std": ("test_nasa_score", "std"),
        "recall": ("test_critical_recall", "mean"), "recall_std": ("test_critical_recall", "std"),
        "train_seconds": ("train_seconds", "mean"),
    }
    for col, name in [("test_coverage", "coverage"), ("test_coverage_capped", "coverage_capped"), ("test_width", "width"),
                      ("test_critical_recall_low", "recall_low"), ("n_params", "n_params"), ("n_trees", "n_trees")]:
        if col in runs:
            agg[name] = (col, "mean")
    s = g.agg(**agg).reset_index()
    s["final"] = False
    for (_, _), grp in s.groupby(["subset", "model"]):
        cv = grp[grp.params_source.isin(["tuned-cv", "transfer"])]
        s.loc[cv.index[0] if len(cv) else grp.val_rmse.idxmin(), "final"] = True
    s["_m"] = s.model.map(MODEL_ORDER.index)
    s["_s"] = s.params_source.map(SOURCE_ORDER.index)
    return s.sort_values(["subset", "_m", "_s"]).drop(columns=["_m", "_s"]).reset_index(drop=True)


def predictions(pred_dir: Path, subset: str, model: str, source: str, seeds: list[int]) -> pd.DataFrame:
    frames = [pd.read_csv(pred_dir / f"{subset}_{model}_{source}_seed{s}.csv").assign(seed=s) for s in seeds]
    return pd.concat(frames, ignore_index=True)


def bucket_errors(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RMSE and mean signed error by true-RUL bucket, averaged over seeds."""
    rows = []
    for model, df in preds.items():
        for (lo, hi), label in zip(BUCKETS, BUCKET_LABEL):
            sel = df[(df.true_rul >= lo) & (df.true_rul <= hi)]
            if sel.empty:
                continue
            per_seed = sel.groupby("seed").apply(
                lambda d: pd.Series({"rmse": np.sqrt(np.mean((d.pred_rul - d.true_rul) ** 2)), "bias": np.mean(d.pred_rul - d.true_rul)}),
                include_groups=False)
            rows.append({"model": model, "bucket": label, "engines": sel.unit.nunique(),
                         "rmse": per_seed.rmse.mean(), "bias": per_seed.bias.mean()})
    return pd.DataFrame(rows)
