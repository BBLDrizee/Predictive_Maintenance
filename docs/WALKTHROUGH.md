# TurboRUL: how the code works, and how to demo it

This guide has two parts.

- **Part 1** explains how the code turns raw engine sensor logs into a maintenance decision, following the data from file to dashboard.
- **Part 2** is a step-by-step script for a live demonstration. It covers setup, starting the API and dashboard, and what to click and say.

For the project's scope see [PRD.md](../PRD.md). For results see the reports in [reports/](../reports/).

---

## Part 1: How the code works

### 1.1 The problem in one paragraph

NASA's C-MAPSS dataset simulates fleets of turbofan engines.

- **Training files** follow each engine from healthy to failure, one row per flight cycle. Each row holds 3 operating settings and 21 sensor readings.
- **Test files** stop each engine at some point before failure. A separate file (`RUL_FDxxx.txt`) gives the true number of cycles that engine had left.

The task is to predict that number, the **Remaining Useful Life (RUL)**. TurboRUL also gives:

- a 90% range around each prediction;
- a risk tier (Critical / Watch / Healthy) and a recommended action;
- the sensors that drove the prediction, measured in cycles.

There are four subsets:

| Subset | Operating conditions | Fault modes | Train / test engines |
|---|---|---|---|
| FD001 | 1 | HPC degradation | 100 / 100 |
| FD002 | 6 | HPC degradation | 260 / 259 |
| FD003 | 1 | HPC + fan degradation | 100 / 100 |
| FD004 | 6 | HPC + fan degradation | 249 / 248 |

### 1.2 The pipeline at a glance

```
raw .txt files
   │  data.py         load, name columns, attach RUL labels (capped)
   ▼
engine-level split    split.py        train / validation engines (never split by row)
   │
   ▼
preprocess.py         cluster operating conditions → z-score each sensor within its condition
                      keep only sensors that trend with age
   │
   ▼
windows.py            last N cycles of each engine → 3-D array (engines × window × sensors)
features.py           (classical models only) summarise each window: mean, std, slope, last, min, max
   │
   ▼
models/               Linear · XGBoost · LSTM · 1D-CNN          ← tune.py (Optuna, 5-fold CV)
   │                                                            ← train.py (5 seeds, MLflow, results.csv)
   ▼
pipeline.RULModel     prediction + calibrated 90% range + per-sensor explanation
   │
   ▼
export.py             pick the best family on validation → models/<subset>/ bundle
   │
   ▼
predict.py            load a bundle, score any engine history → RUL, range, tier, action, top sensors
service.py            JSON-ready queries (fleet, engine detail, results)
   │
   ├── api/main.py         FastAPI REST API
   └── app/dashboard.py    Streamlit dashboard (in-process, or calling the API)
```

Every setting lives in [configs/default.yaml](../configs/default.yaml). This includes caps, window lengths, seeds, split sizes, the sensor threshold, model defaults, tuning budgets, tier thresholds and paths. No code file hard-codes them.

### 1.3 Module by module

All library code is in [src/turborul/](../src/turborul/).

#### Data and labels: [data.py](../src/turborul/data.py)

- `load_subset(subset, raw_dir, rul_cap)` reads the three files for a subset and names the columns `unit`, `cycle`, `op_setting_1..3` and `s1..s21`.
- **Training RUL** is `last cycle − current cycle` for each engine.
- **Test RUL** is counted back from the official value in `RUL_FDxxx.txt`.
- **The cap.** Labels use a *piecewise-linear cap* (default 125, tuned per model between 110 and 140). An engine early in life shows no measurable wear, so the label says "at least the cap" rather than an unknowable exact number. This is the standard choice in the C-MAPSS literature.
- `SENSOR_INFO` maps `s1..s21` to their physical names (e.g. `s4` = T50, LPT outlet temperature). Every chart and explanation uses these names.

#### Preprocessing: [preprocess.py](../src/turborul/preprocess.py)

`Preprocessor` is fitted on training engines only and then applied to validation, test and live data.

1. **Operating conditions.** FD002 and FD004 run under 6 flight conditions (altitude, Mach, throttle). Raw sensor values jump between conditions far more than they drift with wear. k-means on the three operating settings finds the conditions. Each sensor is then z-scored *within its condition*, which leaves only the wear signal. FD001 and FD003 have a single condition, so this reduces to an ordinary z-score.
2. **Sensor selection.** For each sensor, it computes the Spearman correlation with cycle inside each engine and averages the absolute value across engines (the "trend strength"). Sensors below 0.2 are dropped. These are the flat or pure-noise ones, such as s1, s5, s10, s16, s18 and s19 on FD001.

#### Splitting and validation: [split.py](../src/turborul/split.py)

- **Split by engine.** Engines, never rows, go to train or validation (`split_engines`, `kfold_engines`). A row-level split would leak an engine's future into training.
- **Validation is made to look like the test set.** `truncate_engines` cuts each validation engine at 10 random points (true RUL ≤ 150). Only the last window before each cut is scored. Validation numbers are therefore measured the same way as test numbers.

#### Windows and features: [windows.py](../src/turborul/windows.py), [features.py](../src/turborul/features.py)

- `make_windows` slides a window of N cycles (30–50, tuned) over each training engine. The label is the RUL at the window's last cycle.
- `last_windows` takes only each engine's most recent window. This is used for scoring.
- Engines shorter than the window are front-padded with their first reading.
- **Deep models** consume the 3-D windows directly.
- **Linear and XGBoost** get `window_features`: six statistics per sensor (mean, std, slope, last, min, max).

#### Models: [models/baselines.py](../src/turborul/models/baselines.py), [models/deep.py](../src/turborul/models/deep.py)

| Model | What it is | Range method | Explanation method |
|---|---|---|---|
| Linear | least-squares linear regression on window features | constant width (validation residuals) | exact: coefficient × (value − mean) |
| XGBoost | gradient-boosted trees, early stopping | quantile models (5% / 95%) | TreeSHAP |
| LSTM | recurrent net over the window (PyTorch) | MC dropout, 50 passes | integrated gradients |
| 1D-CNN | temporal convolutions over the window | MC dropout, 50 passes | integrated gradients |

The deep models:

- train with Adam on MSE;
- divide targets by the cap so they lie in about [0, 1];
- stop early on validation loss.

#### One model object: [pipeline.py](../src/turborul/pipeline.py)

`prepare()` does load → split → preprocess → window in one call and caches the result. `RULModel` wraps any of the four families behind one interface:

- `fit(data)` trains it.
- `predict(x)` returns point predictions in cycles.
- `fit_interval` / `predict_interval` handle ranges. A raw range from the method in the table above is scaled by a **split-conformal factor**. That factor is fitted on validation engines so that about 90% of them fall inside.
- `explain(x)` returns per-sensor contributions in cycles that sum to (prediction − baseline).

#### Metrics: [metrics.py](../src/turborul/metrics.py)

- **RMSE** in cycles.
- **NASA score**, an asymmetric exponential penalty. Predicting failure *late* costs more than predicting it early.
- **Critical recall**: of the engines truly within 30 cycles of failure, the share we flagged.

#### Tiers and actions: [tiers.py](../src/turborul/tiers.py)

- Critical ≤ 30 cycles, Watch ≤ 60, otherwise Healthy.
- The recommended action uses the **lower end of the 90% range**, so an uncertain engine is treated cautiously. Example: "Schedule a shop visit within 2 cycles (lower end of the predicted range)."

#### Experiments: [tune.py](../src/turborul/tune.py), [train.py](../src/turborul/train.py)

**`tune.py`**

- Runs Optuna (TPE sampler, median pruner) with **engine-level 5-fold cross-validation**. It never touches the test set.
- Searches the model's hyperparameters plus window length and RUL cap.
- Writes the best set to `results/best_params/<subset>_<model>.json`.

**`train.py`**

- Loads tuned parameters if present, otherwise the defaults.
- Trains with 5 seeds and scores each on validation and the official test set.
- Fits and evaluates the 90% ranges.
- Logs to MLflow (`mlruns/`) and appends a row to `results/results.csv`.
- Saves per-engine test predictions to `results/predictions/`.

#### Deployment bundle: [export.py](../src/turborul/export.py)

For each subset, `export` picks the non-linear family with the **lowest mean validation RMSE**. Test scores are never used to choose. It retrains that family with seed 0, calibrates its ranges and writes `models/<subset>/` with these files:

| File | Contents |
|---|---|
| `model.joblib` | the fitted `RULModel` |
| `preprocessor.joblib` | the fitted `Preprocessor` |
| `meta.json` | model, parameters, window, cap, sensors, calibration factor, test metrics, git commit |
| `fleet.json` | precomputed predictions and explanations for every test engine, so the dashboard loads fast |

The LSTM won on all four subsets.

#### Serving: [predict.py](../src/turborul/predict.py), [service.py](../src/turborul/service.py), [api/main.py](../api/main.py)

- **`predict.predict(history, bundle)`** is the core function. It takes one engine's raw history (one row per cycle, raw units), preprocesses it with the bundle's own preprocessor and scores the latest window. It returns:

  ```json
  {"subset": "FD001", "model": "lstm", "cycle": 213, "rul": 9.6, "low": 1.7, "high": 19.1,
   "confidence": 0.9, "tier": "Critical", "base_value": 125.7,
   "action": "Schedule a shop visit within 1 cycles (lower end of the predicted range).",
   "top_sensors": [{"sensor": "s2", "symbol": "T24", "name": "LPC outlet temperature", "contribution": -25.85}, ...]}
  ```

- **`predict_trajectory`** re-scores the engine at every past cycle. This shows how the prediction would have evolved live.
- **`service.py`** shapes results into JSON for both front ends:
  - the fleet table and global sensor importance;
  - engine detail (prediction, trajectory, sensor trends);
  - model comparison.
- **`api/main.py`** exposes these as REST endpoints and validates inputs with pydantic.

| Endpoint | Returns |
|---|---|
| `GET /health` | status and which subsets have a model |
| `GET /subsets` | deployed model per subset with its test metrics |
| `POST /predict` | RUL, range, tier, action and explanation for a posted history |
| `GET /fleet/{subset}?tier=Critical` | every test engine, most urgent first |
| `GET /fleet/{subset}/importance` | mean absolute contribution of each sensor across the fleet |
| `GET /engines/{subset}/{unit}` | full detail for one engine |
| `GET /results` | final metrics for every model and subset |
| `GET /results/{subset}/{model}/predictions` | per-engine test predictions |

#### Dashboard: [app/dashboard.py](../app/dashboard.py), [app/client.py](../app/client.py)

The dashboard is a Streamlit app with four pages.

- **Fleet overview**: tier counts, then a table of engines sorted by predicted RUL with their range, tier and action. It can filter by tier, toggle true RUL, and open any engine.
- **Engine detail**: predicted RUL, range, true RUL and the action. It shows three charts:
  - the RUL trajectory over the engine's life;
  - *Why this prediction*: signed per-sensor contributions;
  - trends of the most influential sensors.
- **Model comparison**: test RMSE, NASA score and coverage for all models and subsets, a predicted-vs-true scatter, and error by RUL bucket.
- **About**: what the data is and what the numbers mean.

`client.py` decides where the data comes from:

- **`LocalClient`** calls `service.py` in-process. This is the default.
- **`HttpClient`** calls the API when the environment variable `TURBORUL_API_URL` is set. The same dashboard then works as a thin front end.

#### Reports: [reports/](../reports/)

- `build_report.py` rebuilds the PDF progress report from `results/` and the bundles, using charts from `figures.py`.
- `screenshots.py` captures the dashboard pages with Playwright, using the installed Edge browser.

### 1.4 Rules that keep the results honest

- **No test peeking.** Tuning, interval calibration and model selection use only training and validation engines. The test set is scored once, for reporting.
- **Engine-level splits everywhere**, including inside cross-validation.
- **Five seeds per configuration.** Tables report mean ± std over the seeds.
- **Reproducibility.** Each results row records the git commit, the parameters and their source (`default`, `tuned-cv` or `transfer`).

### 1.5 Headline results

These are five-seed test RMSEs, in cycles, for the deployed LSTM.

| Subset | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Test RMSE | 15.20 | 24.47 | 14.88 | 24.64 |

- **FD002/FD004 errors** are driven by test engines whose true RUL is far above the cap. Against capped truth, these subsets are in line with validation.
- **Tiers across all test engines:**
  - 92% of truly critical engines are flagged on the point prediction, and 100% using the lower bound.
  - No critical engine is labelled Healthy.
- **Main known weakness:** 90% ranges cover 82–92% (capped truth) rather than 90%, because calibration rests on about 20 validation engines.

The full analysis is in [TurboRUL_Report_Phase5.pdf](../reports/TurboRUL_Report_Phase5.pdf).

---

## Part 2: Running a demonstration

Commands are shown for **Windows PowerShell** from the repo root (`C:\Users\asher\Predictive_Maintenance`). On macOS or Linux, replace `.venv\Scripts\` with `.venv/bin/`.

### 2.1 One-time setup (skip if already done)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
```

Data:

1. Download `CMAPSSData.zip` from the [Zenodo archive](https://zenodo.org/records/15346912).
2. Unzip it into `data\raw\`. You should see `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` and the same for FD002–FD004.

### 2.2 Make sure the model bundles exist

The dashboard and API serve the bundles in `models\FD00x\`. This folder is git-ignored, so on a fresh clone it must be rebuilt.

```powershell
dir models\FD001          # expect: fleet.json  meta.json  model.joblib  preprocessor.joblib
```

If it's missing, rebuild from the stored results:

```powershell
.venv\Scripts\python -m turborul.export          # all four subsets; a few minutes on CPU
```

`export` needs `results\results.csv` and `results\best_params\`. Both are committed.

### 2.3 Pre-demo check (1 minute)

```powershell
.venv\Scripts\python -m pytest -q                # expect: 64 passed
```

### 2.4 Demo option A: dashboard only (simplest)

```powershell
.venv\Scripts\streamlit run app\dashboard.py
```

The browser opens at **http://localhost:8501**. The first load after starting takes a few seconds while PyTorch is imported. Later loads take under a second.

### 2.5 Demo option B: API + dashboard (shows the service architecture)

Use two terminals.

```powershell
# Terminal 1: REST API
.venv\Scripts\uvicorn api.main:app --port 8000
```

```powershell
# Terminal 2: dashboard reading from the API
$env:TURBORUL_API_URL = "http://localhost:8000"
.venv\Scripts\streamlit run app\dashboard.py
```

Interactive API docs are at **http://localhost:8000/docs**. There you can call any endpoint with "Try it out".

### 2.6 Suggested demo script (about 8 minutes)

**1. Set the scene (30 s), on the About page.**
"These are simulated turbofan engines from NASA. We see each engine's sensor history up to today, and we want to know how many flights it has left and what to do about it."

**2. Fleet overview (2 min).** Choose subset **FD001** in the sidebar.

- Point at the tier cards: 23 Critical, 15 Watch, 62 Healthy.
- The table is sorted most urgent first. Each row has a prediction, a 90% range and a plain-language action.
- Turn on **Show true RUL**. For the top engines the predictions are close: engine 34 is predicted 7 against a true 7, engine 81 is 10 against 8.
- Explain that the action uses the *lower* end of the range, so uncertainty makes the advice more cautious, never less.

**3. Engine detail (3 min).** Open **engine 81** (Critical) with *Open engine detail →*.

- **Metrics row:** predicted RUL, the 90% range, the current cycle, and the true RUL with the error.
- **Trajectory chart:**
  - Early in life the prediction sits flat at the cap. The engine looks healthy and the model doesn't pretend to know more.
  - Then it tracks the dashed true line down into the Critical band.
  - The shaded range narrows near failure.
- **Why this prediction:** each bar is how many cycles that sensor added to or removed from the average-engine baseline of about 126 cycles. Magenta bars pull towards failure. For engine 81 the top drivers are rising gas-path temperatures (s2 T24, s4 T50, s3 T30) and HPC outlet static pressure (s11 Ps30), which are classic signs of compressor degradation.
- **Sensor trends:** the same sensors over time, normalised so that 0 is the training average.
- For contrast, open a **Watch** engine such as **engine 90**: predicted 31, true 28. The advice changes to "Plan a shop visit at the next maintenance slot; the engine may have as few as 24 cycles left".
- If asked why the range on this page differs slightly from the fleet table (e.g. 2–18 vs 2–19 for engine 81): the deep models' ranges come from 50 random dropout passes, so each fresh computation varies by a cycle or so. The fleet table is precomputed at export.

**4. Harder data (1 min).** Switch the subset to **FD004**: six operating conditions and two fault modes.

- The same workflow works.
- The ranges are wider, an honest reflection of noisier data.

**5. Model comparison (1.5 min).**

- The tables show all four models on all four subsets, as mean ± std over five seeds.
- In the scatter, points near the diagonal are good. The shaded band at the bottom is the critical zone, where accuracy matters most.
- The bucket chart shows that errors are smallest near failure, which is where decisions are made.
- The deployed model was chosen on validation data only, never on these test numbers.

**6. API (1 min, option B only).** Open http://localhost:8000/docs.

- Run `GET /fleet/FD001?tier=Critical`.
- Run `GET /engines/FD001/81` and show that it returns the same numbers as the dashboard.

### 2.7 Calling the API from code

Here `POST /predict` scores any engine history in raw units. This example sends test engine 81's history:

```powershell
.venv\Scripts\python -c @'
import requests
from turborul.config import load_config
from turborul.data import load_subset
test = load_subset("FD001", load_config()["paths"]["raw_dir"]).test
history = test[test.unit == 81].drop(columns=["unit", "rul"], errors="ignore").to_dict("records")
r = requests.post("http://localhost:8000/predict", json={"subset": "FD001", "history": history})
out = r.json()
print(out["rul"], out["low"], out["high"], out["tier"])
print(out["action"])
for s in out["top_sensors"][:3]:
    print(s["sensor"], s["symbol"], round(s["contribution"], 1))
'@
```

Without the API running, call the library directly:

```python
from turborul.predict import load_bundle, predict
result = predict(history_df, load_bundle("FD001"))
```

### 2.8 Optional: show the experiment pipeline

These commands are not needed for the demo, but they show where the numbers come from. Typical times per seed on a CPU:

- linear and XGBoost: seconds;
- LSTM and CNN: 10 s to about 2 minutes, depending on the subset.

```powershell
.venv\Scripts\python -m turborul.train --subset FD001 --model xgb --seeds 0    # train + score one seed
.venv\Scripts\python -m turborul.tune  --subset FD001 --model xgb --trials 5   # short tuning run
.venv\Scripts\mlflow ui --backend-store-uri mlruns                             # browse every run at http://localhost:5000
.venv\Scripts\python reports\build_report.py                                   # rebuild the PDF report
```

> **Note:** `train` appends rows to `results\results.csv`, and `tune` overwrites `results\best_params\<subset>_<model>.json`. For a demo, either run them on a throwaway branch or restore afterwards with `git checkout -- results`.

### 2.9 Troubleshooting

| Symptom | Fix |
|---|---|
| `no exported model for FD00x` | Run `python -m turborul.export`. |
| `FileNotFoundError` for `train_FD00x.txt` | Unzip the dataset into `data\raw\`. |
| Port 8501 or 8000 already in use | Add `--server.port 8502` (Streamlit) or `--port 8001` (uvicorn), and update `TURBORUL_API_URL` to match. |
| Dashboard shows connection errors in option B | Check that the API terminal is running, and that `TURBORUL_API_URL` was set in the *same* terminal as Streamlit. |
| First page load is slow | Expected once per server start, while PyTorch and scikit-learn are imported. |
| `streamlit` / `uvicorn` not found | Use the `.venv\Scripts\` prefix, or activate the venv with `.venv\Scripts\Activate.ps1`. |
