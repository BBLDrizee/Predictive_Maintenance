# TurboRUL

Remaining Useful Life (RUL) prediction for turbofan engines on NASA's C-MAPSS dataset. See [PRD.md](PRD.md) for scope and plan.

> Work in progress: Phase 1 (setup and EDA) is done. Results, demo and limitations will be added in later phases.

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

## Tests

```bash
.venv/Scripts/python -m pytest
```

## Layout

- `configs/default.yaml`: RUL cap, window length, seeds, tier thresholds, paths
- `src/turborul/data.py`: loading and RUL labelling (FR-1)
- `notebooks/01_eda.ipynb`: exploratory analysis and the sensor-selection decision
- `tests/`: unit tests for the leak-prone steps
