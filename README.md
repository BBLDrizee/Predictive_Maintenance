# TurboRUL

Remaining Useful Life (RUL) prediction for turbofan engines on NASA's C-MAPSS dataset, with uncertainty ranges,
risk tiers, sensor-level explanations, a REST API and a maintenance-planning dashboard. See [PRD.md](PRD.md) for
scope and plan.

> Work in progress: Phases 1–5 are done (data, baselines, deep models, all four subsets, uncertainty, explanations,
> API, dashboard). Progress reports: [reports/](reports/). Code walkthrough and demo script:
> [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python -m pip install -e .
```

## Data

Download `CMAPSSData.zip` (12.4 MB, MD5 `79a22f36e80606c69d0e9e4da5bb2b7a`) from the
[Zenodo archive](https://zenodo.org/records/15346912) and unzip it into `data/raw/`.
The raw files are git-ignored.

Note: the dataset's readme lists FD004 as 248 train / 249 test engines, but the files contain 249 train / 248 test.

## Run the demo

```bash
python -m turborul.export                       # build models/<subset>/ bundles from the trained results
uvicorn api.main:app --port 8000                # REST API, docs at http://localhost:8000/docs
streamlit run app/dashboard.py                  # dashboard (in-process)
TURBORUL_API_URL=http://localhost:8000 streamlit run app/dashboard.py   # dashboard against the API
```

API endpoints: `POST /predict`, `GET /fleet/{subset}`, `GET /fleet/{subset}/importance`, `GET /engines/{subset}/{unit}`,
`GET /results`, `GET /results/{subset}/{model}/predictions`, `GET /subsets`, `GET /health`.

## Train, tune and evaluate

```bash
# Tune with engine-level 5-fold cross-validation (never sees the test set)
python -m turborul.tune --subset FD001 --model xgb      # also: lstm, cnn

# Train with 5 seeds; score the official test set; log to MLflow and results/results.csv
python -m turborul.train --subset FD001 --model linear  # also: xgb, lstm, cnn

mlflow ui --backend-store-uri mlruns                    # browse runs
python reports/build_report.py                          # rebuild the progress report
```

Evaluation follows the standard protocol: each test engine is scored on its last window against `RUL_FDxxx.txt`.
Tuning, interval calibration and model selection only use validation engines, which are cut at random points so
they are scored the same way as the test set.

## Tests

```bash
python -m pytest
```

## Layout

- `configs/default.yaml`: every setting (caps, windows, seeds, split, sensor threshold, model defaults, tuning, tiers, paths)
- `src/turborul/`
  - `data.py`: loading, RUL labels, sensor names (FR-1)
  - `preprocess.py`: sensor selection and per-condition normalisation (FR-2)
  - `split.py`: engine-level split and k-fold (FR-3), test-like validation truncation
  - `windows.py`, `features.py`: sliding windows and window features
  - `models/`: linear and XGBoost (FR-4), LSTM and 1D-CNN (FR-5)
  - `metrics.py`: RMSE, NASA score, critical recall (FR-6)
  - `pipeline.py`: shared preparation; `RULModel` with calibrated ranges (FR-8) and explanations
  - `explain.py`: TreeSHAP, integrated gradients, linear contributions (FR-10)
  - `tiers.py`: risk tiers and recommended actions (FR-9)
  - `train.py`, `tune.py`, `export.py`: training CLI, Optuna cross-validated tuning, model bundle export
  - `predict.py`, `service.py`: prediction function (FR-7) and the queries behind the API and dashboard
- `api/main.py`: FastAPI app
- `app/dashboard.py`: Streamlit dashboard, with fleet overview, engine detail and model comparison screens (FR-11)
- `models/`: exported bundles (git-ignored)
- `results/`: `results.csv`, per-engine test predictions, tuned parameters
- `reports/`: progress reports (PDF), their build scripts, and dashboard screenshots
- `tests/`: unit tests for labelling, splitting, windowing, preprocessing, metrics, uncertainty, explanations, prediction,
  API and dashboard
