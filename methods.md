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

## 2026-05-12: First two baseline models implemented (persistence, displacement averaging)

Two zero-parameter baselines committed under `src/animal_movement_fm/models/`. Persistence (`next = current`) is the floor every other model must beat; it's Bayes-optimal on Brownian motion, so any model that substantially beats persistence there has a bug. Displacement averaging (`next = current + mean of last k displacement vectors`) is parameterised by `k` rather than split into separate classes (`k=1` is the constant velocity model, a strong baseline in the pedestrian-trajectory literature); this makes the k-ablation a single loop instead of multiple files. Displacements are taken in raw (lon, lat) degree space rather than on the sphere; a small-step approximation acceptable at per-step scales of ~1 km, and any bias surfaces in the haversine metric (no hidden cheat). Both models satisfy the `Predictor` protocol and have no `fit` method; 26 new tests passing, full suite at 58/58.


## 2026-05-15: ARIMA baseline implemented

ARIMA(p, d, q) baseline committed under `src/animal_movement_fm/models/arima.py` as the last classical step in the baseline ladder (persistence -> averaging -> ARIMA -> XGBoost -> LSTM -> transformer). The default order is (2, 0, 2) on the displacement series equivalent to ARIMA(2, 1, 2) on raw positions; the displacements are computed explicitly via `np.diff` rather than via the `d` parameter so the stationarity assumption is visible in the code. The two coordinate channels (delt lon, delt lat) are fit as independent univariate ARIMA models, which loses cross-coupling between axes (left it for now); this matches the simplicity of a reference baseline rather than aiming for the best classical model. The model fits at predict time on each sliding window rather than fit-then-store, so it sees exactly the same context length the deep learning baselines will see. Obviously this is rather wasteful but keeps the comparison honest. Fitting uses `statsmodels.tsa.arima.model.ARIMA` (state-space representation + Kalman-filter likelihood + numerical MLE); convergence failures, singular Hessians, and near-constant series fall back to persistence per channel with a logged warning. We also made 24 new tests (GenAI generated, referenced in the test file) covering construction, prediction, fallback behaviour, properties (translation invariance, determinism), and Predictor-protocol contract; full suite now at 82/82.

## 2026-05-17: XGBoost baseline complete; full ladder benchmarked on synthetic data

Implemented `src/animal_movement_fm/models/xgboost.py`. Approach follows Wijeyakulasuriya et al. (2020), "Machine learning for modeling animal movement", and used GenAI to assist in building some features of this mode. Following the paper, we used per-axis regressors (one XGBRegressor for Δlon, one for Δlat), hand-engineered tabular features (lag positions, lag displacements, speeds, turning angles), and global training across all trajectories in the train split. Differences from the reference are that we omit their behavioural-state-classification stage (would give XGBoost an information advantage relative to the LSTM rung), add explicit speed and turning-angle features, and a time-of-day sin/cos encoding, disabled in synthetic runs.

### Configuration

- `n_lags=5`, `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`
- `objective="reg:squarederror"`, `random_state=0`
- Trained on per-individual 80/20 split; 16 individuals × 500 steps = 8000 rows for training, 4 individuals × 500 steps = 2000 rows for eval
- Hyperparameters are reference-baseline defaults, not tuned. 

### Results: full baseline ladder on synthetic data

20 individuals × 500 steps per dataset, 80/20 per-individual split, context_length=24 (~1 day of hourly data), n_test_windows=1904 per model.

**Brownian motion (sanity check,l persistence is Bayes-optimal):**

| Model | Median (m) | Mean (m) | p99 (m) |
|---|---:|---:|---:|
| persistence    | **1316.3** | **1390.8** | **3221.8** |
| averaging_k10  | 1370.1 | 1450.5 | 3303.7 |
| xgboost_lag5   | 1350.5 | 1435.0 | 3478.0 |
| arima_202      | 1503.3 | 1581.8 | 3683.8 |
| averaging_k1   | 1820.8 | 1947.0 | 4527.2 |

Persistence wins as theory requires. XGBoost feature importances on BM are near-uniform across all features (max 0.046, min 0.042 in top 10),  the correct footprint of a model failing to find structure that isn't there. Sanity check passes; the harness and predictors are behaving correctly.

**Correlated random walk (linear-with-autoregression test):**

| Model | Median (m) | Mean (m) | p99 (m) |
|---|---:|---:|---:|
| xgboost_lag5   | **707.3** | **837.6** | **3071.9** |
| arima_202      | 846.0 | 1005.7 | 3266.0 |
| averaging_k1   | 838.8 | 1008.7 | 3414.8 |
| averaging_k10  | 872.9 | 1027.9 | 3299.5 |
| persistence    | 941.2 | 1122.1 | 3781.0 |

XGBoost wins uniformly across all percentiles (25% improvement on median, 19% on p99 vs persistence). Feature importances are dominated by`dlon_lag_1` and `dlat_lag_1` (~0.40 each, ~10× higher than other features), confirming the model has correctly identified the autoregressive structure that defines CRW. Error CDFs (see `results/plots/cdf_crw.png`) show XGBoost dominating uniformly, not just on average.

**Switching CRW (regime-detection test):**

| Model | Median (m) | Mean (m) | p99 (m) |
|---|---:|---:|---:|
| persistence    | **457.1** | 1232.8 | 6448.1 |
| averaging_k1   | 551.5 | 1129.2 | 5634.1 |
| xgboost_lag5   | 595.5 | **976.5** | **5031.8** |
| arima_202      | 809.4 | 1233.8 | 5719.9 |
| averaging_k10  | 908.7 | 1194.9 | 5627.1 |

The bimodal-error signature (mean/median ≈ 2.7 for persistence, p99/median ≈ 14) shows up as predicted: foraging steps dominate the typical step (persistence wins median), but transit steps blow up the tail (persistence loses mean and p99 badly).

XGBoost trims the long tail by 22% (p99: 5032 vs 6448) and improves mean by 21%. Feature importances on switching-CRW show the displacement
features still dominant but with notable mass on `speed_lag_*`, `turn_lag_*`, and higher-order `dlat_lag_*`,  the *correct* footprint of a model trying to detect regime through indirect signals. The CDF (`results/plots/cdf_switching_crw.png`) shows the crossover explicitly: persistence wins below ~600m error (foraging regime), XGBoost wins above (transit regime). This is exactly the headroom the LSTM rung is expected to close: explicit state representation should compress the long tail substantially below 5032m.


