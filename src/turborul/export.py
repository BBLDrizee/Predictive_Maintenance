"""Export one deployable model bundle per subset for the API and dashboard.

    python -m turborul.export                # all subsets
    python -m turborul.export --subset FD001

The deployed family is the final (cross-validation-tuned) configuration with the
lowest mean validation RMSE across its five seeds, so the test set plays no part
in the choice. It is retrained with the first configured seed, its 90% ranges are
calibrated on validation engines, and the bundle is written to models/<subset>/:
model.joblib, preprocessor.joblib and meta.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import joblib

from turborul.config import load_config
from turborul.data import SUBSETS
from turborul.metrics import evaluate
from turborul.pipeline import RULModel, prepare
from turborul.reporting import latest_runs, summarise
from turborul.train import _git_commit, interval_metrics, resolve_params


def choose_family(cfg: dict[str, Any], subset: str) -> tuple[str, list[dict[str, Any]]]:
    summary = summarise(latest_runs(cfg["paths"]["results_csv"], subset))
    finals = summary[summary.final & (summary.model != "linear")].sort_values("val_rmse")
    if finals.empty:
        raise RuntimeError(f"no trained models for {subset}; run turborul.train first")
    table = finals[["model", "params_source", "val_rmse", "test_rmse"]].round(3).to_dict("records")
    return str(finals.iloc[0].model), table


def export(subset: str, config: str | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    family, candidates = choose_family(cfg, subset)
    params, source = resolve_params(cfg, subset, family)
    model_params = {k: v for k, v in params.items() if k not in ("window_len", "rul_cap")}
    data = prepare(cfg, subset, params["window_len"], params["rul_cap"], include_test=True)
    seed = cfg["seeds"][0]
    m = RULModel(family, model_params, data.sensors, data.window_len, data.rul_cap, seed, cfg["training"]).fit(data)
    m.fit_interval(data, cfg["uncertainty"]["alpha"])
    point, low, high = m.predict_interval(data.x_test)
    test = evaluate(data.y_test, point, cfg["tiers"]["critical"]) | interval_metrics(
        data.y_test, low, high, data.rul_cap, cfg["tiers"]["critical"])

    out_dir = cfg["paths"]["models_dir"] / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out_dir / "model.joblib")
    joblib.dump(data.preprocessor, out_dir / "preprocessor.joblib")
    meta = {
        "subset": subset, "model": family, "params_source": source, "params": model_params, "seed": seed,
        "window_len": data.window_len, "rul_cap": data.rul_cap, "sensors": data.sensors,
        "alpha": m.alpha, "calibration": m.calibration, "tiers": cfg["tiers"],
        "test_metrics": test, "candidates": candidates,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": _git_commit(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    # Precompute the test-fleet scores and explanations so the dashboard's first view is fast.
    from turborul import predict, service

    predict.load_bundle.cache_clear()
    service._fleet.cache_clear()
    service.fleet_frame(subset, use_cache=False).to_json(out_dir / "fleet.json", orient="records")
    size_kb = sum(f.stat().st_size for f in out_dir.iterdir()) / 1024
    print(f"{subset}: exported {family} ({source}) — test RMSE {test['rmse']:.2f}, coverage {test['coverage']:.2f}, "
          f"{size_kb:.0f} KB")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", choices=SUBSETS, nargs="*")
    parser.add_argument("--config")
    args = parser.parse_args()
    for subset in args.subset or SUBSETS:
        export(subset, args.config)


if __name__ == "__main__":
    main()
