# animal-movement-fm
Benchmarking deep learning architectures for animal movement prediction across species and ecological domains.

ANU honours thesis in machine learning (2026), Jack Jenkins.

## What this is

A benchmarking suite for animal movement prediction that scales models from trivial baselines (persistence, averaging) through classical statistics (ARIMA), gradient boosting (XGBoost), and recurrent networks (LSTM) up to transformer-based foundation models. Models are trained on a large multi-species GPS tracking corpus and evaluated for cross-species and cross-domain transfer.

## Repository layout

```
animal-movement-fm/
├── src/animal_movement_fm/    # library code (importable)
│   ├── data/                  # synthetic generators, datasets, splits
│   ├── models/                # one file per model family
│   ├── eval/                  # metrics + evaluation harness
│   └── train/                 # training loops
├── configs/                   # YAML experiment configs
├── experiments/               # runnable scripts (one per experiment)
├── notebooks/                 # exploratory analysis only — not for results
├── tests/                     # pytest unit tests
├── results/                   # output tables, plots (gitignored)
├── scripts/                   # ad-hoc utilities (NCI submission etc.)
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
uv run python experiments/01_synthetic_baselines.py

# Open jupyter
uv run jupyter lab
```

## Status

Currently: scaffolding + synthetic data validation.
