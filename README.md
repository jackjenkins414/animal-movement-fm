# animal-movement-fm
Benchmarking deep learning architectures for animal movement prediction across species and ecological domains.

ANU honours thesis in machine learning (2026), Jack Jenkins.

## What this is

A benchmarking suite for animal movement prediction that scales models from trivial baselines (persistence, averaging) through classical statistics (ARIMA), gradient boosting (XGBoost), and recurrent networks (LSTM) up to transformer based foundation models. Models are trained on a large multi-species GPS tracking corpus and evaluated for **zero-shot transfer to species never seen in training**.

We currently run a two phase preregistration (Hofman et al., 2023). We first document the arms, metrics, success bands and predictions, and ensure those are committed before any model is trained; held-out test partitions are opened once, by a sealed evaluator that refuses to run until the matching freeze is committed to git. Exploratory work lives in a separate tier (Tier E) that is validation only and may never be reported as a registered result.

**Tiers.** Tier 1: the baseline ladder on synthetic and single species data. Tier 2: three source species, three held-out species, one per domain. Tier 3: source-pool scaling, the same three held-out species, nine sources. Tier 4: all remaining eligible species, exploratory.

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
├── registered/                # write once artefacts: selection, sampling weights, boundaries
├── splits/                    # frozen per species split manifests (seed 42)
├── results/                   # output tables, plots (gitignored)
├── preregistration_tier2*.md  # Tier 2 registration, tuning amendments, pre test freeze
├── preregistration_tier3*.md  # Tier 3 registration and pre test freeze
├── tierE_exploratory_ablations.md  # exploratory arms, each registered before it runs
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

**Tier 1** complete: the full ladder (persistence, averaging, ARIMA, XGBoost, LSTM, transformer) on real 6 hour white stork
data, with 6 hours resolved as the near-optimal cadence.

**Tier 2** sealed and reported (26-08-2026): **WEAK/NULL**. The foundation model does not beat persistence zero-shot on
any of the three held-out species, the same trunk with the FM package stripped matches or beats it, and a single species
specialist reaches +0.20 skill on northern elephant seal. The failure is concentrated in near stationary windows.

**Tier E** (exploratory, validation only) locates the failure inside the architecture: within a single species, with
ample data, the full FM package never beats a plain transformer trunk, and the context conditioning path is negative on
every species tested. The masked reconstruction objective is the one component that helps, and only when learned across
species.

**Tier 3** sealed and reported (23-09-2026): **WEAK/NULL**. Changing one variable — three source species to nine, with
configuration, budget, representation, metrics, held-out species and protocol fixed — moves zero-shot travel p90 skill by
at most +0.018 against Tier 2, inside the ±0.05 noise scale. A three source same domain pool beats the nine source pool
on all three held-out species by point estimate, but the gap is resolved only on northern elephant seal (+0.11), where a
single same domain source reaches +0.18 against the nine-source +0.004 — and that source is the congeneric southern
elephant seal, so domain and genus are confounded. Fine-tuning the nine source model loses to a specialist trained from
scratch on the identical 1% and 10% subsets on 17 of 18 draws. In the descriptive baseline ladder the single-species LSTM
is the strongest rung on all three species, and single-species XGBoost (which carries no validation privilege) beats
every pooled model on two of three.

**26 source pool** sealed and reported (27-09-2026), exploratory and never a tiered result. Pooling every eligible
source moves zero-shot travel p90 skill by at most +0.004 against the nine-source pool, and the FM package sits within
0.007 of the plain trunk on all three held-out species, inside the $\pm$ 0.05 noise scale. On northern elephant seal, +0.16
for a seven species marine pool against +0.009 for all 26 sources; removing southern elephant seal costs −0.047 with
the package and −0.011 without, both inside noise, so the marine gain reads as domain rather than genus. White stork's
share of the training loss fell from 64% to 18% with no change in skill, so the stork imbalance was not the cause.
Fine-tuning still loses to a from-scratch specialist on 13 of 18 draws, and the single species LSTM is still the
strongest rung on all three species.

**Bird only study** sealed and reported (28-09-2026), preregistered: pretrained on 22 flying birds, tested on the
lesser black-backed gull, the greater white-fronted goose and the European turtle dove. Zero-shot, +0.067 (FM) and
+0.128 (plain trunk) on the dove are the only transformer results that clear persistence by more than 0.05 from that
pool; adding five congener pool birds lifts the gull to +0.081. On the dove, the pooled LSTM and XGBoost are level with
their own specialists (+0.011, +0.026); it has the fewest training birds (88). This is the first case in the project
where models that never saw a species predict it as well as models trained on it. Earlier zero-shot transfer (NES)
stayed below the specialists. It comes from pooled LSTM and XGBoost rather than the foundation model, and from one
species, but it is first evidence that zero-shot prediction of an unseen species at specialist level is possible. On
the goose every transformer arm is within $\pm$ 0.03 of persistence. The package costs on every bird by point estimate
(−0.025, −0.024, −0.060), beyond noise only on the dove. Three of the five registered predictions held on at least
one bird.

**Next:** a cross-species randomness analysis on training individuals, a pooling study on the 26 source pool
(domain-balanced sampling and MoE routing), environmental covariate for birds, and a quick descriptive comparison of 
fix intervals. A probabilistic head and a full covariate study are probably future work at this point. 
