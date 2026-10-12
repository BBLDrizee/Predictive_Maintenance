"""Train and evaluate a model on one subset, logging to MLflow and results/results.csv.

    python -m turborul.train --subset FD001 --model xgb
    python -m turborul.train --subset FD001 --model lstm --seeds 0 1 2 3 4

Hyperparameters come from results/best_params/<subset>_<model>.json when it
exists (written by turborul.tune), otherwise from configs/default.yaml. Every run
also fits calibrated 90% prediction ranges and records their test coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turborul.config import REPO_ROOT, load_config
from turborul.metrics import critical_recall, evaluate
from turborul.pipeline import MODELS, RULModel, prepare


def resolve_params(cfg: dict[str, Any], subset: str, model: str, use_tuned: bool = True) -> tuple[dict[str, Any], str]:
    """Return (params incl. window_len and rul_cap, source)."""
    path = cfg["paths"]["params_dir"] / f"{subset}_{model}.json"
    if use_tuned and path.exists():
        tuned = json.loads(path.read_text())
        if tuned.get("transferred_from"):
            return tuned["params"], "transfer"  # tuned on another subset, reused here
        return tuned["params"], "tuned-cv" if tuned.get("folds", 1) > 1 else "tuned"
    params = dict(cfg["models"][model] or {})
    params.update(window_len=cfg["windows"]["window_len"], rul_cap=cfg["labels"]["rul_cap"])
    return params, "default"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def interval_metrics(y_true: np.ndarray, low: np.ndarray, high: np.ndarray, rul_cap: float | None, critical: float) -> dict[str, float]:
    capped = np.minimum(y_true, rul_cap) if rul_cap is not None else y_true
    return {
        "coverage": float(np.mean((y_true >= low) & (y_true <= high))),
        "coverage_capped": float(np.mean((capped >= low) & (capped <= high))),
        "width": float(np.mean(high - low)),
        "critical_recall_low": critical_recall(y_true, low, critical),  # flag Critical on the lower bound
    }


def run(subset: str, model: str, seeds: list[int], use_tuned: bool = True, config: str | None = None) -> pd.DataFrame:
    import mlflow

    cfg = load_config(config)
    params, source = resolve_params(cfg, subset, model, use_tuned)
    model_params = {k: v for k, v in params.items() if k not in ("window_len", "rul_cap")}
    data = prepare(cfg, subset, params["window_len"], params["rul_cap"], include_test=True)
    critical = cfg["tiers"]["critical"]
    alpha = cfg["uncertainty"]["alpha"]

    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"].as_uri())
    mlflow.set_experiment(f"turborul-{subset}")
    pred_dir: Path = cfg["paths"]["predictions_dir"]
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        start = time.perf_counter()
        m = RULModel(model, model_params, data.sensors, data.window_len, data.rul_cap, seed, cfg["training"]).fit(data)
        train_seconds = time.perf_counter() - start
        m.fit_interval(data, alpha)
        val = evaluate(data.y_valcut, m.predict(data.x_valcut), critical)
        y_pred, low, high = m.predict_interval(data.x_test)
        test = evaluate(data.y_test, y_pred, critical)
        test |= interval_metrics(data.y_test, low, high, data.rul_cap, critical)

        pd.DataFrame({"unit": data.test_units, "true_rul": data.y_test, "pred_rul": y_pred, "low": low, "high": high}).to_csv(
            pred_dir / f"{subset}_{model}_{source}_seed{seed}.csv", index=False)

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "subset": subset, "model": model, "seed": seed, "params_source": source,
            "window_len": data.window_len, "rul_cap": data.rul_cap, "n_sensors": len(data.sensors),
            **{f"val_{k}": v for k, v in val.items()},
            **{f"test_{k}": v for k, v in test.items()},
            "calibration": m.calibration, "train_seconds": round(train_seconds, 1), **m.size(),
            "params": json.dumps(model_params, sort_keys=True),
        }
        rows.append(row)

        with mlflow.start_run(run_name=f"{model}-seed{seed}"):
            mlflow.log_params({"subset": subset, "model": model, "seed": seed, "params_source": source,
                               "window_len": data.window_len, "rul_cap": data.rul_cap,
                               "sensors": ",".join(data.sensors), **model_params})
            mlflow.log_metrics({k: v for k, v in row.items()
                                if (k.startswith(("val_", "test_")) or k in ("train_seconds", "calibration")) and not pd.isna(v)})
        print(f"{subset} {model} seed={seed}: val RMSE {val['rmse']:.2f} | test RMSE {test['rmse']:.2f}, "
              f"NASA {test['nasa_score']:.0f}, critical recall {test['critical_recall']:.2f}, "
              f"90% range coverage {test['coverage']:.2f} (width {test['width']:.0f}) ({train_seconds:.0f}s)")

    append_results(cfg["paths"]["results_csv"], pd.DataFrame(rows))
    return pd.DataFrame(rows)


def append_results(csv: Path, rows: pd.DataFrame, timeout: float = 120) -> None:
    """Append rows to results.csv under a lock file, so parallel training runs cannot overwrite each other."""
    csv.parent.mkdir(parents=True, exist_ok=True)
    lock = csv.with_suffix(".lock")
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"{lock} held for over {timeout}s; delete it if no run is active") from None
            time.sleep(0.2)
    try:
        out = pd.concat([pd.read_csv(csv), rows], ignore_index=True) if csv.exists() else rows
        out.to_csv(csv, index=False)
    finally:
        os.close(fd)
        os.remove(lock)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", help="defaults to the seeds in the config")
    parser.add_argument("--default-params", action="store_true", help="ignore tuned parameters")
    parser.add_argument("--config", help="path to a config YAML")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else load_config(args.config)["seeds"]
    out = run(args.subset, args.model, seeds, use_tuned=not args.default_params, config=args.config)
    if len(out) > 1:
        print(f"mean +/- std over {len(out)} seeds: test RMSE {out.test_rmse.mean():.2f} +/- {out.test_rmse.std():.2f}, "
              f"NASA {out.test_nasa_score.mean():.0f} +/- {out.test_nasa_score.std():.0f}")


if __name__ == "__main__":
    main()
