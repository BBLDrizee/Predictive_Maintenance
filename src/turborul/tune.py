"""Tune hyperparameters with Optuna using engine-level k-fold cross-validation (the test set is never loaded).

    python -m turborul.tune --subset FD001 --model lstm --trials 20

Each trial is scored by the mean, over folds, of RMSE on the test-like validation
set (truncated validation engines scored on their last window against the true,
uncapped RUL), so window length and RUL cap can be tuned alongside model
parameters. Trials can be pruned after any fold. The best parameters are written
to results/best_params/<subset>_<model>.json for turborul.train.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import optuna

from turborul.config import load_config
from turborul.metrics import rmse
from turborul.pipeline import DEEP, MODELS, RULModel, prepare


def search_space(trial: optuna.Trial, model: str) -> dict[str, Any]:
    p: dict[str, Any] = {
        "window_len": trial.suggest_categorical("window_len", [20, 30, 40, 50]),
        "rul_cap": trial.suggest_int("rul_cap", 110, 140, step=5),
    }
    if model == "xgb":
        p |= {
            "n_estimators": 2000, "early_stopping_rounds": 100,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30, log=True),
        }
    elif model == "lstm":
        p |= {
            "hidden": trial.suggest_categorical("hidden", [32, 64, 128]),
            "layers": trial.suggest_int("layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    elif model == "cnn":
        p |= {
            "channels": trial.suggest_categorical("channels", [16, 32, 64]),
            "n_conv": trial.suggest_int("n_conv", 2, 4),
            "kernel": trial.suggest_categorical("kernel", [3, 5, 7]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.4),
            "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        }
    else:
        raise ValueError(f"no search space for {model!r}")
    return p


def tune(subset: str, model: str, n_trials: int | None = None, config: str | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    tcfg = cfg["tuning"]
    n_trials = n_trials or tcfg["n_trials"][model]
    training = {"max_epochs": tcfg["max_epochs"], "patience": tcfg["patience"]}
    stride = tcfg["train_stride"] if model in DEEP else 1

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial, model)
        window_len, rul_cap = params.pop("window_len"), params.pop("rul_cap")
        scores = []
        for fold in range(tcfg["folds"]):
            data = prepare(cfg, subset, window_len, rul_cap, include_test=False, fold=fold)
            data.x_train, data.y_train = data.x_train[::stride], data.y_train[::stride]
            m = RULModel(model, params, data.sensors, window_len, rul_cap, cfg["seed"], training).fit(data)
            scores.append(rmse(data.y_valcut, m.predict(data.x_valcut)))
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_rmse", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize", study_name=f"{subset}-{model}",
        sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - start

    best = search_space(optuna.trial.FixedTrial(study.best_params), model)
    out = {"subset": subset, "model": model, "params": best, "cv_rmse": study.best_value,
           "cv_fold_rmse": study.best_trial.user_attrs["fold_rmse"], "folds": tcfg["folds"],
           "n_trials": len(study.trials), "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
           "tuning_seconds": round(elapsed)}
    params_dir = cfg["paths"]["params_dir"]
    params_dir.mkdir(parents=True, exist_ok=True)
    (params_dir / f"{subset}_{model}.json").write_text(json.dumps(out, indent=2))
    study.trials_dataframe().to_csv(params_dir / f"{subset}_{model}_trials.csv", index=False)
    print(f"{subset} {model}: best CV RMSE {study.best_value:.2f} after {len(study.trials)} trials "
          f"({out['n_pruned']} pruned, {elapsed / 60:.1f} min)\n{json.dumps(best, indent=2)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="FD001")
    parser.add_argument("--model", choices=[m for m in MODELS if m != "linear"], required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--config")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune(args.subset, args.model, args.trials, args.config)


if __name__ == "__main__":
    main()
