# PRD: TurboRUL — AI Predictive Maintenance for Aircraft Engines

> **For Claude Code:** This is the product requirements document for this repository. Read it fully before starting work. Build in the phase order in [Implementation plan](#implementation-plan), respect the [Rules](#rules-for-implementation), and treat requirement IDs (FR-x) as the source of truth for scope. Priorities: **P0** = must ship, **P1** = portfolio-ready, **P2** = stretch.

---

## 1. Overview

TurboRUL predicts the **Remaining Useful Life (RUL)** of turbofan engines from sensor data. It turns those predictions into a maintenance planning dashboard. It is a one-semester project for an Artificial Intelligence college course, built on NASA's public **C-MAPSS** engine degradation dataset.

**Problem.** Airlines maintain engines either on fixed schedules or after something breaks.
- Fixed schedules replace parts that still have life left.
- Reactive repairs ground aircraft unexpectedly (AOG, aircraft-on-ground events), cancel flights and strand passengers.

Predicting how many cycles an engine has left lets planners schedule work just in time, during planned downtime.

**Why airlines care.** Engines are among the most expensive components on an aircraft, and unplanned removals ripple through the schedule. Airlines, engine makers and MROs (maintenance, repair and overhaul providers) all invest in predictive maintenance.

**One-line pitch:** Given an engine's sensor history, TurboRUL predicts how many flight cycles it has left, explains why, and tells the planner which engines to schedule for maintenance first.

---

## 2. Goals, non-goals and success metrics

The project succeeds if:
- a trained model beats simple baselines by a clear margin on held-out engines, and
- a live demo lets anyone pick an engine and see its predicted RUL with an explanation.

### Goals
1. Train and compare at least three RUL models, from a simple baseline to a deep sequence model.
2. Report results on the official test split with standard metrics, so they are comparable to published work.
3. Convert predictions into a maintenance decision: a risk tier and a ranked work list.
4. Explain each prediction by showing which sensors drove it.
5. Ship a public demo, a GitHub repo and a short write-up.

### Non-goals
- Certified or production-grade software. This is a research prototype on simulated data.
- Real airline data, or integration with real maintenance systems.
- Diagnosing the specific failed part. Fault classification is a stretch goal (FR-13), not a core goal.

### Success metrics

| Metric | Target | Why it matters |
| --- | --- | --- |
| RMSE on FD001 test set (cycles) | ≤ 15; stretch ≤ 13 | Standard accuracy measure. Strong published deep models reach roughly 11–13 (approximate; verify against recent papers). |
| NASA scoring function on FD001 | Report it, and beat own baselines | Penalises late predictions more than early ones, matching real safety costs. |
| Improvement over linear baseline | ≥ 25% lower RMSE | Shows the AI adds real value. |
| Results on all four subsets | RMSE reported for FD001–FD004 | Shows the model handles multiple operating conditions and faults. |
| Critical-engine recall | ≥ 90% of engines with true RUL ≤ 30 cycles flagged Critical | The metric a planner actually cares about. |
| Demo latency | Prediction shown in < 2 s | Keeps the demo usable live. |

---

## 3. Users and user stories

| Persona | Who they are | What they need |
| --- | --- | --- |
| Maintenance planner | Schedules engine shop visits across a fleet | A ranked list of engines to act on, with a confidence range |
| Reliability engineer | Investigates degradation trends | Sensor trends per engine, and the features behind each prediction |
| Course evaluator | Grades the AI methods | Clear methodology, fair baselines, honest metrics |
| Airline or MRO data scientist | Reviews the portfolio as a potential hire | Clean code, reproducible results, a working demo, business framing |

**User stories**
- As a planner, I want every engine sorted by predicted RUL, so I know which to schedule first.
- As a planner, I want each engine tagged Healthy, Watch or Critical, so I can act without reading numbers.
- As a planner, I want a prediction range, not just one number, so I know how much to trust it.
- As a reliability engineer, I want to see the sensor trends behind a prediction, so I can sanity-check it.
- As an evaluator, I want to rerun training from one command and get the reported numbers.
- As a recruiter, I want to understand the project's value in under two minutes from the README or demo.

---

## 4. Dataset

**NASA Turbofan Engine Degradation Simulation (C-MAPSS).** This is about 12 MB of run-to-failure sensor data for simulated engines.

**Download:**
- NASA PCoE repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Kaggle mirror: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Zenodo archive: https://zenodo.org/records/15346912

### Files and format
For each subset `FDxxx` there are three files:
- `train_FDxxx.txt` — full run-to-failure histories.
- `test_FDxxx.txt` — histories cut off at some point before failure.
- `RUL_FDxxx.txt` — the true RUL at the last test cycle, one line per test engine, in engine order.

The files are space-separated with no header, and may have trailing spaces or empty trailing columns, so strip them. Each row is one engine at one operating cycle, with 26 columns:

| Column(s) | Name |
| --- | --- |
| 1 | `unit` (engine ID) |
| 2 | `cycle` |
| 3–5 | `op_setting_1..3` (altitude, Mach number, throttle resolver angle) |
| 6–26 | `s1..s21` (sensor readings: temperatures, pressures, fan and core speeds, etc.) |

### Subsets

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
| --- | --- | --- | --- | --- |
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC degradation) |
| FD002 | 260 | 259 | 6 | 1 (HPC degradation) |
| FD003 | 100 | 100 | 1 (sea level) | 2 (HPC + fan degradation) |
| FD004 | 248 | 249 | 6 | 2 (HPC + fan degradation) |

HPC stands for high-pressure compressor. **FD001 is the primary benchmark** because it is the simplest and most widely reported. FD004 is the hardest.

### Labelling
- **Training engines:** `RUL = max_cycle_of_engine − current_cycle`.
- **Cap:** clip RUL at **125 cycles**, giving a piecewise-linear target. Make the cap configurable. Early-life engines show no measurable degradation, so the model can't be expected to tell 300 cycles left from 200.
- **Test engines:** the true RUL at the last cycle comes from `RUL_FDxxx.txt`.

### Known limitations
State these in the README and report:
- The data is simulated, not from a real fleet.
- Failure is defined by a threshold, not by real removal decisions.
- There is no cost, schedule or parts data.

N-CMAPSS, which uses realistic flight profiles, is a stretch option.

---

## 5. Functional requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | Load all four C-MAPSS subsets and compute RUL labels with a configurable cap (default 125). | P0 |
| FR-2 | Remove constant or uninformative sensors, and normalise features. For FD002 and FD004, normalise per operating condition. | P0 |
| FR-3 | Split training engines into train and validation sets **by engine ID, never by row**, to avoid leakage. | P0 |
| FR-4 | Train a baseline model (linear regression) and a tree model (XGBoost or LightGBM) on windowed features. | P0 |
| FR-5 | Train at least one deep sequence model (LSTM, 1D-CNN or GRU) on sliding windows of 30 cycles. | P0 |
| FR-6 | Evaluate every model on the official test split with RMSE and the NASA score, and save results to a comparison table. | P0 |
| FR-7 | Predict RUL for any engine at its latest cycle, through a Python function and a REST endpoint. | P1 |
| FR-8 | Give each prediction an uncertainty range, for example with Monte Carlo dropout or quantile regression. | P1 |
| FR-9 | Assign a risk tier: Critical (RUL ≤ 30), Watch (31–60), Healthy (> 60). Thresholds are configurable. | P1 |
| FR-10 | Explain each prediction with its top contributing sensors: SHAP for tree models; gradient-based attribution or SHAP for deep models. | P1 |
| FR-11 | Dashboard with three screens: fleet list, engine detail and model comparison (see §8). | P1 |
| FR-12 | Simulate a live engine: replay a test engine cycle by cycle and watch the prediction update. | P2 |
| FR-13 | Classify the likely fault mode (HPC vs fan) on FD003 and FD004. | P2 |

---

## 6. Technical approach

The pipeline moves from simple to complex models, so each added layer of AI has to earn its place against the one before it.

1. **Exploratory analysis.** Plot each sensor over each engine's life. Drop sensors that stay flat; several do in FD001. Check correlations between sensors.
2. **Preprocessing.**
   - Use min-max or z-score scaling, **fitted on training data only**.
   - For FD002 and FD004, cluster the 3 operational settings into the 6 operating conditions with k-means, then normalise each sensor within its condition. This matters more than the model choice on those subsets.
   - Optionally smooth the sensors with an exponential moving average to reduce noise.
3. **Windowing.** Slice each engine's history into overlapping windows of 30 cycles, with the window length configurable. Each window's label is the capped RUL at its last cycle. Pad test engines that have fewer than 30 cycles.
4. **Feature engineering for classical models.** For each window and sensor, compute the mean, standard deviation, slope (linear trend), last value, min and max.
5. **Models.**
   - Baseline: linear regression on engineered features.
   - Tree ensemble: XGBoost or LightGBM on engineered features.
   - Deep sequence model: a 2-layer LSTM with 64 units, or a 1D-CNN, trained on raw windows with early stopping.
   - Optional, if time allows: a small Transformer encoder.
6. **Hyperparameter tuning.** Tune window length, RUL cap, learning rate and layer sizes with Optuna, on the **validation engines only**.
7. **Evaluation.** Score each test engine on its **last window only**, which is the standard protocol. Repeat deep-model training with 5 random seeds and report the mean ± standard deviation.

### Metrics
- **RMSE** in cycles.
- **NASA scoring function**, which punishes late predictions (overestimating remaining life) more than early ones. With `d_i = predicted_RUL_i − true_RUL_i`:

```
S = Σ_i  { exp(−d_i / 13) − 1   if d_i < 0
         { exp( d_i / 10) − 1   if d_i ≥ 0
```

```python
import numpy as np

def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
```

For both metrics, lower is better. Also report **critical-engine recall**: the share of engines with true RUL ≤ 30 that are flagged Critical.

---

## 7. System architecture and tech stack

The system has an offline training pipeline and a lightweight online demo, so the hosted app never retrains. That keeps it fast and free to run.

```
OFFLINE: train and evaluate
  [C-MAPSS data]  ──►  [Data pipeline]  ──►  [Model training]
   4 subsets,           RUL labels (cap 125)    Linear, XGBoost, LSTM
   21 sensors           per-condition norm      tuned with Optuna
                        30-cycle windows        runs logged in MLflow
                                                      │
                                                      │ export best model
                                                      ▼
ONLINE: serve the demo
  [Streamlit dashboard]  ◄──  [Prediction API (FastAPI)]  ◄──  [Saved best model]
   fleet list, engine          RUL + uncertainty range           model file + scaler
   detail, model comparison    risk tier (30 / 60 cycles)        results table
                               top sensors via SHAP              versioned in MLflow
```

| Layer | Tool | Why |
| --- | --- | --- |
| Language | Python 3.11 | Standard for ML |
| Data | pandas, NumPy | Loading and windowing |
| Classical ML | scikit-learn, XGBoost | Baselines and tree ensemble |
| Deep learning | PyTorch (or TensorFlow/Keras) | LSTM and 1D-CNN models |
| Tuning | Optuna | Efficient hyperparameter search |
| Explainability | SHAP | Per-prediction sensor contributions |
| Experiment tracking | MLflow | Logs configs, metrics and models |
| API | FastAPI | Simple REST prediction endpoint |
| Dashboard | Streamlit + Plotly | Fast to build, interactive |
| Hosting | Streamlit Community Cloud or Hugging Face Spaces | Free public demo link |
| Testing | pytest | Guards the labelling and windowing logic |

### Suggested repository layout

```
turborul/
├── PRD.md                  # this file
├── README.md
├── requirements.txt        # pinned versions
├── configs/
│   └── default.yaml        # rul_cap, window_len, seeds, tier thresholds, paths
├── data/
│   ├── raw/                # C-MAPSS .txt files (git-ignored)
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/turborul/
│   ├── __init__.py
│   ├── data.py             # loading, column names, RUL labelling (FR-1)
│   ├── preprocess.py       # sensor selection, scaling, per-condition norm (FR-2)
│   ├── split.py            # engine-level train/val split (FR-3)
│   ├── windows.py          # sliding windows + padding
│   ├── features.py         # engineered window features for classical models
│   ├── models/
│   │   ├── baselines.py    # linear, XGBoost (FR-4)
│   │   └── deep.py         # LSTM / 1D-CNN (FR-5)
│   ├── metrics.py          # RMSE, NASA score, critical recall (FR-6)
│   ├── train.py            # CLI: train + evaluate + log to MLflow
│   ├── predict.py          # predict(engine_history) -> RUL, range, tier (FR-7..9)
│   └── explain.py          # SHAP / attributions (FR-10)
├── api/
│   └── main.py             # FastAPI app (FR-7)
├── app/
│   └── dashboard.py        # Streamlit app (FR-11)
├── models/                 # exported best model + scaler
├── results/
│   └── results.csv         # model × subset comparison table
└── tests/
    ├── test_data.py        # RUL labels, cap
    ├── test_split.py       # no engine overlap between train and val
    └── test_windows.py     # window shapes, labels, padding
```

---

## 8. Dashboard UX

The first screen must answer the planner's question, "which engines need attention?", without any clicks.

**1. Fleet overview (home)**
- Summary cards showing the number of engines in Critical, Watch and Healthy.
- A table of all engines sorted by predicted RUL, lowest first, with engine ID, current cycle, predicted RUL with range, and a risk tier badge.
- Filters by subset (FD001–FD004) and by tier.

**2. Engine detail**
- Predicted RUL with its uncertainty range, and the risk tier.
- A chart of predicted RUL over the engine's life. For test engines, overlay the true RUL to show the errors honestly.
- Trend charts for the 4–6 most informative sensors.
- A "Why this prediction" bar chart of the top contributing sensors.
- A recommended action in plain words, for example "Schedule shop visit within 20 cycles".
- A replay button (P2, FR-12) to step through cycles and watch the prediction update.

**3. Model comparison**
- A table of RMSE and NASA score for each model and subset.
- A scatter plot of predicted vs true RUL on the test set.
- Error by true-RUL bucket, to show the model is most accurate near failure, where it matters most.

**Design rules**
- Use red, amber and green only for risk tiers.
- Label every chart with units (cycles).
- Include an "About this project" panel stating that the data is simulated.

---

## 9. Non-functional requirements

| Area | Requirement |
| --- | --- |
| Reproducibility | Fixed random seeds, pinned package versions, and one command to train and evaluate |
| Compute | Each model trains in < 30 min on a free Colab GPU or a laptop CPU |
| Performance | Dashboard loads in < 3 s; a single prediction takes < 2 s |
| Code quality | Modular Python package, type hints, and unit tests for labelling, splitting and windowing (the leak-prone steps) |
| Experiment tracking | Every run logged with its config and metrics (MLflow) |
| Honesty | The test set is scored once per final model and never used for tuning; the limitations of simulated data are stated |
| Cost | Zero: free dataset, open-source tools, free hosting tier |
| Documentation | README with the problem, results table, a demo GIF, how to run it, and its limitations |

---

## 10. Implementation plan

This is a 12-week semester plan. **Gate: the P0 pipeline must work end to end by the end of week 6.**

| Weeks | Phase | Exit criteria |
| --- | --- | --- |
| 1–2 | Setup and EDA: repo, data, sensor plots, RUL labels | Data loads for all four subsets; RUL labels are unit-tested; EDA notebook shows which sensors to drop |
| 3–4 | Baselines: windows, features, linear and XGBoost | Linear and XGBoost RMSE on the FD001 test set recorded in `results/results.csv` |
| 5–6 | Deep models: LSTM and 1D-CNN, tuning on FD001 | The deep model beats XGBoost, or the gap is explained; results averaged over 5 seeds |
| **End wk 6** | **Gate: P0 pipeline works end to end** | FR-1 to FR-6 done |
| 7–8 | Robustness and explainability: FD002–FD004, uncertainty, SHAP, tiers | RMSE reported on all four subsets; uncertainty ranges and SHAP explanations working |
| 9–10 | API and dashboard: FastAPI, three Streamlit screens | All three screens working locally against the API |
| 11–12 | Polish and showcase: report, demo video, deploy, write-up | Demo deployed; report and video done; write-up published |

### Task checklist (in build order)

**Phase 1: Setup and EDA**
- [ ] Create the repo layout above, `requirements.txt` and `configs/default.yaml`
- [ ] `data.py`: load train, test and RUL files for FD001–FD004 with named columns (FR-1)
- [ ] Add RUL labelling with a configurable cap, plus tests in `tests/test_data.py`
- [ ] `notebooks/01_eda.ipynb`: sensor trends, flat-sensor detection, correlations

**Phase 2: Baselines**
- [ ] `preprocess.py`: drop flat sensors; scaler fitted on training data only (FR-2)
- [ ] `split.py`: engine-level train/val split, plus a test that train and val share no engines (FR-3)
- [ ] `windows.py`: 30-cycle sliding windows, last-window extraction for test engines, padding, plus tests
- [ ] `features.py`: per-window mean, std, slope, last, min, max
- [ ] `models/baselines.py`: linear regression and XGBoost (FR-4)
- [ ] `metrics.py`: RMSE, NASA score, critical recall (FR-6)
- [ ] `train.py` CLI, e.g. `python -m turborul.train --subset FD001 --model xgb`, which logs to MLflow and appends to `results/results.csv`

**Phase 3: Deep models**
- [ ] `models/deep.py`: LSTM and 1D-CNN in PyTorch with early stopping (FR-5)
- [ ] Optuna tuning on validation engines
- [ ] 5-seed runs, reporting mean ± std

**Phase 4: Robustness and explainability**
- [ ] k-means operating-condition clustering and per-condition normalisation for FD002 and FD004
- [ ] Results for all four subsets
- [ ] Uncertainty ranges via MC dropout or quantile regression (FR-8)
- [ ] Risk tiers from config thresholds (FR-9)
- [ ] `explain.py`: top contributing sensors per prediction (FR-10)

**Phase 5: API and dashboard**
- [ ] `predict.py`: `predict(engine_history) -> {rul, low, high, tier, top_sensors}` (FR-7)
- [ ] `api/main.py`: a FastAPI `POST /predict` endpoint and a `GET /fleet/{subset}` endpoint
- [ ] `app/dashboard.py`: fleet overview, engine detail and model comparison screens (FR-11)

**Phase 6: Polish and showcase**
- [ ] README with problem, results table, demo GIF, how to run, and limitations
- [ ] Deploy to Streamlit Community Cloud or Hugging Face Spaces
- [ ] Course report and a 3-minute demo video
- [ ] Optional P2 items: replay mode (FR-12), fault-mode classification (FR-13)

---

## Rules for implementation

1. **Never leak across engines.** Split by engine ID, and fit scalers and condition clusters on training engines only.
2. **Never tune on the test set.** Use validation engines for every choice. Score the test set only for final models.
3. **Use the standard evaluation.** Score each test engine on its last window only, against `RUL_FDxxx.txt`.
4. **Configure, don't hard-code.** RUL cap, window length, seeds, tier thresholds and paths all live in `configs/default.yaml`.
5. **Test the leak-prone steps** (labelling, splitting, windowing) before building models on them.
6. **Keep it reproducible.** Seed everything (NumPy, PyTorch, Python `random`), and pin versions.
7. **Keep it small.** Everything should run on a laptop CPU or a free Colab GPU. Exported models should be small enough for free hosting.
8. **Build simple first.** Get a P0 pipeline working end to end before adding P1 or P2 features.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Data leakage (rows from one engine in both train and validation) | Inflated scores that collapse under scrutiny | Split by engine ID; unit-test the split |
| Tuning on the test set | Results not credible | Tune on validation engines only; score the test set once per final model |
| Deep model underperforms XGBoost | Weaker AI story | Still a valid finding; report it honestly and explain why |
| Poor results on FD002 and FD004 | The model looks fragile | Per-condition normalisation; present FD001 as primary and the others as robustness checks |
| Running out of time | Unfinished demo | Get P0 working end to end by week 6; add dashboard features in priority order |
| "It's only simulated data" criticism | Reduced credibility with airlines | State it upfront; frame the project as a method demonstration; mention N-CMAPSS as the next step |
| Free hosting limits | Demo offline when someone visits | Keep model files small; record a backup demo video |

---

## 12. Deliverables, stretch goals and showcase plan

### Deliverables
- [ ] GitHub repository with code, tests, results table, and a README with a demo GIF
- [ ] Live dashboard on Streamlit Community Cloud or Hugging Face Spaces
- [ ] Course report: problem, method, experiments, results, limitations
- [ ] 3-minute demo video
- [ ] Write-up on LinkedIn or a blog: problem, approach, key result, link to the demo

### Stretch goals
- Train and evaluate on N-CMAPSS, which uses realistic flight profiles.
- Fault-mode classification on FD003 and FD004 (FR-13).
- A cost-aware maintenance policy: compare "schedule when RUL is under X" against fixed-interval maintenance, and estimate the unplanned failures avoided.
- Transfer learning: train on FD001, then adapt to FD003 using only a few engines.

### Showcase plan
- Lead with the business result, not the model. For example: "flags 9 in 10 engines within 30 cycles of failure, with a typical error of about 13 cycles". Replace these with your real numbers.
- Include a short "how an airline would use this" section and a limitations section.
- Share the write-up and tag airline engineering and digital teams, MROs and engine makers.
- Add the project to your resume as one line: the method, the metric, and the demo link.

---

## Sources
- NASA PCoE Data Set Repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- NASA Turbofan Jet Engine Data Set (Kaggle): https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- PCoE Turbofan Engine Degradation Simulation (Zenodo): https://zenodo.org/records/15346912

The benchmark figures above (RMSE of roughly 11–13 on FD001) are approximate. Verify them against recent papers before quoting them in the report.
