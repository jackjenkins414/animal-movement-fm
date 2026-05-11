# Methods log

A running log of every methodological decision taken in this project, with the date and the reasoning. This file is the single source of truth for the methods chapter of the thesis. Here we append only and never edit past entries; so if a decision is reversed, we add a new entry explaining the reversal.

Format: every entry has a date, a short title, and 1–4 sentences of rationale. 

---

## 2026-05-09: Project scaffolded

Initialised repository with `src/` layout, `uv` for environment management, `ruff` for linting, `pytest` for testing. Chose `src/` layout over flat layout to make the package installable (`pip install -e .`) so it imports cleanly from anywhere, including future NCI job scripts. Configs live in `configs/` as YAML; experiments live in `experiments/` as runnable scripts; `src/` contains library code only.

## 2026-05-09: Synthetic-data-first validation strategy

Baselines (persistence, averaging, ARIMA, XGBoost, LSTM, Transformer) will be validated on synthetic data generated from known state-space models *before* being applied to real Movebank trajectories. Three synthetic datasets of escalating difficulty: pure Brownian motion (persistence is Bayes-optimal), correlated random walk with von Mises turning angles (autoregressive models should beat persistence), and state-switching CRW with hidden Markov regimes (only models with memory should detect regime switches). Our rationale for this is that ground truth lets us verify each model behaves as theory predicts before trusting any rankings on real data.

## 2026-05-09: Synthetic generators implemented

Three generators in `src/animal_movement_fm/data/synthetic.py`: Brownian motion (BM), correlated random walk (CRW with Gamma step lengths and von Mises turning angles), and state-switching CRW (two-state hidden Markov chain over a CRW). Output schema made to match Movebank; [individual_id, timestamp, lon, lat], with optional ground truth state column for the switching variant. Validated by 12 pytest tests generated using GenAI (referenced) covering schema, reproducibility, and statistical properties (increment distribution for BM; circular variance of consecutive headings for CRW; state conditional step length and run length distribution for switching CRW). All 12 tests passing as of this commit.

## 2026-05-09: CRW step_scale tuned for visual interpretability

Increased the default `step_scale` in `generate_crw` from 0.0005 to 0.005. With the smaller value, per-step displacement was so small that the directional persistence (kappa=4) was not visually apparent in 500-step trajectories; they looked indistinguishable from Brownian motion despite having genuine autocorrelation. The new value produces trajectories with visibly persistent runs, making the BM => CRW => switching CRW difficulty progression visually clear. No statistical tests changed; tests check relative properties (circular variance) not absolute step magnitudes.

## 2026-05-11: Eval harness implemented

Single shared evaluation pipeline in `src/animal_movement_fm/eval/`. Every model implements a `Predictor` protocol with one method (`predict(history) -> next_position`); the harness runs sliding-window single-step prediction over each individual's trajectory and returns a per prediction error DataFrame stratifiable by any column. Headline metric is haversine great circle distance, which we measure in metres (not Euclidean degrees, which mis-states distance at non-equatorial latitudes). Validated against textbook geographic constants (1° lat = 111.195 km, antipodal = πR) rather than another haversine implementation, to avoid circular validation. Single-step prediction follows the MoveFormer (Cifka et al., 2023) and deepSSF (Forrest et al., 2026) convention; `Predictor` protocol design follows sklearn API conventions (Buitinck et al., 2013).

## 2026-05-11: Persistence baseline reference numbers (synthetic)

First evaluation run complete. Persistence baseline (predict next = current) evaluated on 1760 windows across all three synthetic datasets (10 individuals x 200 steps, context_length=24):
- Brownian motion: median 1320 m, mean 1404 m, p99 717 m
- CRW: median 930 m, mean 1115 m, p99 790 m
- Switching CRW: median 348 m, mean 1006 m, p99 1340 m
The switching-CRW mean/median gap (~3x) reflects the bimodal step-length distribution from the latent state structure: persistence does well on foraging-state steps but poorly on transit-state steps. These numbers are the reference benchmarks against which all other models will be compared.

