# animal-movement-fm
Benchmarking deep learning architectures for animal movement prediction across species and ecological domains.

ANU honours thesis in machine learning (2026), Jack Jenkins.

## What this is

A benchmarking suite for animal movement prediction that scales models from trivial baselines (persistence, averaging) through classical statistics (ARIMA), gradient boosting (XGBoost), and recurrent networks (LSTM) up to transformer-based foundation models. Models are trained on a large multi-species GPS tracking corpus and evaluated for cross-species and cross-domain transfer.

## Repository layout

```
animal-movement-fm/
├── src/animal_movement_fm/    # library code (importable)
│   ├── data/                  # loaders, splits, segment building, resampling, synthetic
│   ├── features/              # feature engineering (time of day, ...)
│   ├── models/                # one file per model family
│   ├── eval/                  # metrics + evaluation harness
│   ├── analysis/              # analysis helpers (randomness, ...)
│   └── train/                 # training loops
├── configs/                   # YAML experiment configs
├── experiments/               # runnable scripts, grouped by purpose
│   ├── eda/                   # exploratory + regime/randomness analysis
│   ├── baselines/             # ladder runners, head to heads, ablations
│   ├── cadence/               # cadence sweeps + inter-fix analysis
│   ├── training/              # Gadi training entry points + PBS
│   ├── smoke/                 # GPU smoke tests
│   └── reporting/             # prediction dumps + writeup figures
├── notebooks/                 # exploratory analysis only — not for results
├── tests/                     # pytest unit tests
├── results/                   # output tables, plots (gitignored)
├── methods.md                 # decisions log
├── pyproject.toml             # dependencies + tool config
└── README.md
```

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/jackjenkins414/animal-movement-fm.git
cd animal-movement-fm
uv sync --extra dev
```

This creates a `.venv/` and installs all dependencies including dev tools (pytest, ruff, jupyter).

## Common commands

```bash
# Run tests
uv run pytest

# Lint + format
uv run ruff check .
uv run ruff format .

# Run an experiment
uv run python experiments/baselines/run_baselines_real.py

# Open jupyter
uv run jupyter lab
```

## Status

Full baseline ladder (persistence -> averaging -> ARIMA -> XGBoost -> LSTM -> Transformer) complete on real 6h white-stork data; 6h cadence resolved as near-optimal. Transformer convergence resolved on NCI Gadi. Two headline results: a raw sequence Transformer vs LSTM head to head (a tie), and a time of day 2×2 showing attention exploits the covariate while recurrence does not. Now focused on modelling/feature work (added covariates, multi-seed spread, val-curve monitoring) toward the foundation model, infrastructure is in place.
