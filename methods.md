# Methods log

A running log of every methodological decision taken in this project, with the date and the reasoning. This file is the single source of truth for the methods chapter of the thesis. Here we append only and never edit past entries; so if a decision is reversed, we add a new entry explaining the reversal.

Format: every entry has a date, a short title, and 1–4 sentences of rationale. 

---

## 09-05-2026: Project scaffolded

Initialised repository with `src/` layout, `uv` for environment management, `ruff` for linting, `pytest` for testing. Chose `src/` layout over flat layout to make the package installable (`pip install -e .`) so it imports cleanly from anywhere, including future NCI job scripts. Configs live in `configs/` as YAML; experiments live in `experiments/` as runnable scripts; `src/` contains library code only.

## 09-05-2026: Synthetic-data-first validation strategy

Baselines (persistence, averaging, ARIMA, XGBoost, LSTM, Transformer) will be validated on synthetic data generated from known state-space models *before* being applied to real Movebank trajectories. Three synthetic datasets of escalating difficulty: pure Brownian motion (persistence is Bayes-optimal), correlated random walk with von Mises turning angles (autoregressive models should beat persistence), and state-switching CRW with hidden Markov regimes (only models with memory should detect regime switches). Our rationale for this is that ground truth lets us verify each model behaves as theory predicts before trusting any rankings on real data.

## 09-05-2026: Synthetic generators implemented

Three generators in `src/animal_movement_fm/data/synthetic.py`: Brownian motion (BM), correlated random walk (CRW with Gamma step lengths and von Mises turning angles), and state-switching CRW (two-state hidden Markov chain over a CRW). Output schema made to match Movebank; [individual_id, timestamp, lon, lat], with optional ground truth state column for the switching variant. Validated by 12 pytest tests generated using GenAI (referenced) covering schema, reproducibility, and statistical properties (increment distribution for BM; circular variance of consecutive headings for CRW; state conditional step length and run length distribution for switching CRW). All 12 tests passing as of this commit.

## 09-05-2026: CRW step_scale tuned for visual interpretability

Increased the default `step_scale` in `generate_crw` from 0.0005 to 0.005. With the smaller value, per-step displacement was so small that the directional persistence (kappa=4) was not visually apparent in 500-step trajectories; they looked indistinguishable from Brownian motion despite having genuine autocorrelation. The new value produces trajectories with visibly persistent runs, making the BM => CRW => switching CRW difficulty progression visually clear. No statistical tests changed; tests check relative properties (circular variance) not absolute step magnitudes.

## 11-05-2026: Eval harness implemented

Single shared evaluation pipeline in `src/animal_movement_fm/eval/`. Every model implements a `Predictor` protocol with one method (`predict(history) -> next_position`); the harness runs sliding-window single-step prediction over each individual's trajectory and returns a per prediction error DataFrame stratifiable by any column. Headline metric is haversine great circle distance, which we measure in metres (not Euclidean degrees, which mis-states distance at non-equatorial latitudes). Validated against textbook geographic constants (1° lat = 111.195 km, antipodal = πR) rather than another haversine implementation, to avoid circular validation. Single-step prediction follows the MoveFormer (Cifka et al., 2023) and deepSSF (Forrest et al., 2026) convention; `Predictor` protocol design follows sklearn API conventions (Buitinck et al., 2013).

## 11-05-2026: Persistence baseline reference numbers (synthetic)

First evaluation run complete. Persistence baseline (predict next = current) evaluated on 1760 windows across all three synthetic datasets (10 individuals x 200 steps, context_length=24):
- Brownian motion: median 1320 m, mean 1404 m, p99 717 m
- CRW: median 930 m, mean 1115 m, p99 790 m
- Switching CRW: median 348 m, mean 1006 m, p99 1340 m
The switching-CRW mean/median gap (~3x) reflects the bimodal step-length distribution from the latent state structure: persistence does well on foraging-state steps but poorly on transit-state steps. These numbers are the reference benchmarks against which all other models will be compared.

## 12-05-2026: First two baseline models implemented (persistence, displacement averaging)

Two zero-parameter baselines committed under `src/animal_movement_fm/models/`. Persistence (`next = current`) is the floor every other model must beat; it's Bayes-optimal on Brownian motion, so any model that substantially beats persistence there has a bug. Displacement averaging (`next = current + mean of last k displacement vectors`) is parameterised by `k` rather than split into separate classes (`k=1` is the constant velocity model, a strong baseline in the pedestrian-trajectory literature); this makes the k-ablation a single loop instead of multiple files. Displacements are taken in raw (lon, lat) degree space rather than on the sphere; a small-step approximation acceptable at per-step scales of ~1 km, and any bias surfaces in the haversine metric (no hidden cheat). Both models satisfy the `Predictor` protocol and have no `fit` method; 26 new tests passing, full suite at 58/58.


## 15-05-2026: ARIMA baseline implemented

ARIMA(p, d, q) baseline committed under `src/animal_movement_fm/models/arima.py` as the last classical step in the baseline ladder (persistence -> averaging -> ARIMA -> XGBoost -> LSTM -> transformer). The default order is (2, 0, 2) on the displacement series equivalent to ARIMA(2, 1, 2) on raw positions; the displacements are computed explicitly via `np.diff` rather than via the `d` parameter so the stationarity assumption is visible in the code. The two coordinate channels (delt lon, delt lat) are fit as independent univariate ARIMA models, which loses cross-coupling between axes (left it for now); this matches the simplicity of a reference baseline rather than aiming for the best classical model. The model fits at predict time on each sliding window rather than fit-then-store, so it sees exactly the same context length the deep learning baselines will see. Obviously this is rather wasteful but keeps the comparison honest. Fitting uses `statsmodels.tsa.arima.model.ARIMA` (state-space representation + Kalman-filter likelihood + numerical MLE); convergence failures, singular Hessians, and near-constant series fall back to persistence per channel with a logged warning. We also made 24 new tests (GenAI generated, referenced in the test file) covering construction, prediction, fallback behaviour, properties (translation invariance, determinism), and Predictor-protocol contract; full suite now at 82/82.

## 17-05-2026: XGBoost baseline complete; full ladder benchmarked on synthetic data

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

## 02-06-2026: LSTM baseline complete; fifth rung benchmarked on synthetic data

Implemented `src/animal_movement_fm/models/lstm.py`, the first model in the ladder that learns its own representation rather than reading off hand-engineered features, and the first trained-once-then-reloaded rather than fit inline. Architecture is a standard gated LSTM (`torch.nn.LSTM`) feeding a linear head that emits a 2 element displacement (Delta lon, Delta lat) added to the last position; predicting displacements keeps the target stationary, matching ARIMA and XGBoost. Window definition is delegated to the harness's `iter_windows` so the net trains on exactly the windows it is evaluated on, and inputs are standardised using train-split statistics stored on the model and reapplied identically at predict time. Like XGBoost it trains globally across all train trajectories; unlike XGBoost it is fit-then-store *with checkpoint persistence* (`save`/`from_checkpoint` round-trip weights + scaler stats + architecture), which is where the train on NCI / evaluate elsewhere separation the foundation model relies on first enters the ladder. 18 new tests (used GenAI to generate them, (referenced)). 

### Configuration

- `hidden_dim=64`, `num_layers=1`, `context_length=24` (matched to harness), `epochs=30`, `batch_size=64`, `lr=1e-3`, Adam, MSE loss
- CPU, `torch.manual_seed(0)`; same 80/20 per-individual split as the rest of the ladder (n_test_windows=1904)
- Reference-baseline defaults, not tuned

### Results: LSTM vs the ladder

| Dataset | Model | Median (m) | Mean (m) | p99 (m) |
|---|---|---:|---:|---:|
| brownian_motion | persistence | **1316.3** | **1390.8** | 3221.8 |
| brownian_motion | lstm_h64_l1 | 1333.4 | 1421.8 | 3311.3 |
| crw | lstm_h64_l1 | **642.5** | **820.2** | 3179.0 |
| crw | xgboost_lag5 | 707.3 | 837.6 | **3071.9** |
| switching_crw | persistence | **457.1** | 1232.8 | 6448.1 |
| switching_crw | lstm_h64_l1 | 543.0 | **960.1** | 5061.2 |

- **Brownian:** tied with persistence — correct, since independent increments leave nothing to learn; training loss bottomed out by epoch 2 and flatlined.
- **CRW:** best of all six models, beating XGBoost and well clear of persistence; the learned representation extracts more than hand-engineered features on genuinely autoregressive data.
- **Switching CRW:** best median and mean overall. Caveat: raw persistence keeps the lowest *median* (457 m), being near-unbeatable on slow foraging steps, but pays with the worst mean and p99; the LSTM trades a little median to compress the tail (p99 5061 vs 6448). Whether the transformer's attention closes the tail further is the next rung's test.

### Note: OpenMP runtime conflict (macOS)

torch + XGBoost in one process segfaults on Apple Silicon (`xgboost/data.py::_meta_from_numpy`); separate OpenMP runtimes. `KMP_DUPLICATE_LIB_OK=TRUE` does not help (governs Intel `libiomp`, not the `libomp` here); `OMP_NUM_THREADS=1` does, set at the top of the experiment script. Cleaner fix (shared OpenMP, or separate processes) deferred.

## 28-06-2026: Transformer baseline complete; sixth rung benchmarked on synthetic data

Implemented `src/animal_movement_fm/models/transformer.py`, the final rung before the foundation model and structurally the same architecture the FM will be built from. Where the LSTM compresses history into a single hidden state passed step to step, the transformer reads the whole window at once via bidirectional self-attention (`nn.TransformerEncoder`, no causal mask), so every position attends directly to every other; a regime switch early in the window reaches the prediction without being squeezed through a recurrent bottleneck. Input displacements are projected from 2D to the model width via a linear layer, then stamped with fixed sinusoidal positional encoding (Vaswani et al., 2017) before the encoder stack. The readout mirrors the LSTM exactly: `out[:, -1, :]` feeds a linear head emitting a 2-element displacement. Everything except the inner network is held identical to the LSTM rung (same differenced-displacement inputs, same train-statistic standardisation, same single-displacement target, same checkpoint persistence), so any difference in results is attributable to attention vs recurrence and nothing else. 17 new tests (GenAI generated and referenced/validated).

### Training recipe

Initial runs (30 epochs, no warmup, lr=1e-3 Adam) showed the transformer ~20% worse than the LSTM on plain CRW median; suspicious, since CRW is trivially autoregressive. A convergence check confirmed train_mse had flatlined by epoch 20, ruling out undertraining; the gap was an optimisation artefact from withholding warmup, the standard transformer recipe. Locked recipe: 60 epochs for both neural rungs; linear warmup + cosine decay (`warmup_frac=0.1`, lr is the peak) on the transformer only, constant lr on the LSTM. Each architecture gets its conventional setup. With warmup the CRW gap closed (transformer 657 vs LSTM 709 median across 10 seeds), confirming the earlier underperformance was recipe, not architecture.

### Configuration

- `d_model=64`, `nhead=4`, `num_layers=2`, `dim_feedforward=128`, `dropout=0.1`
- `context_length=24`, `epochs=60`, `batch_size=64`, `lr=1e-3`, `warmup_frac=0.1`, Adam, MSE loss
- LSTM comparison run at `epochs=60` (all else unchanged from LSTM entry) so neither rung is undertrained
- Reference-baseline defaults, not tuned

### Results: 10-rep multi-seed benchmark (total variance)

Each of 10 master seeds (0–9) drives both data generation and torch init, so reported variance includes data sampling + training stochasticity. 20 individuals x 500 steps, 80/20 per individual split, context_length=24, n_test_windows=1904 per model per seed.

**Brownian motion (sanity check):**

| Model | Median (m) | Median range | p99 (m) | p99 range |
|---|---:|---|---:|---|
| persistence | 1,299 | [1,277 – 1,331] | 3,350 | [3,221 – 3,569] |
| transformer_d64_l2 | 1,301 | [1,282 – 1,348] | 3,341 | [3,248 – 3,517] |
| lstm_h64_l1 | 1,314 | [1,285 – 1,353] | 3,368 | [3,312 – 3,582] |
| xgboost_lag5 | 1,357 | [1,327 – 1,418] | 3,516 | [3,346 – 3,796] |

All models collapse to persistence; ranges fully overlap. Sanity check passes.

**CRW:**

| Model | Median (m) | Median range | p99 (m) | p99 range |
|---|---:|---|---:|---|
| transformer_d64_l2 | 657 | [613 – 728] | 3,106 | [2,940 – 3,520] |
| xgboost_lag5 | 705 | [679 – 725] | 3,137 | [2,754 – 3,366] |
| lstm_h64_l1 | 709 | [664 – 759] | 3,123 | [2,879 – 3,398] |
| persistence | 936 | [882 – 960] | 3,750 | [3,387 – 4,087] |

Transformer leads on CRW median (657 vs LSTM 709, XGBoost 705), though all three learner ranges overlap. All well clear of persistence.

**Switching-CRW:**

| Model | Median (m) | Median range | p99 (m) | p99 range |
|---|---:|---|---:|---|
| lstm_h64_l1 | 598 | [511 – 783] | 4,948 | [4,421 – 5,476] |
| transformer_d64_l2 | 600 | [531 – 732] | 5,114 | [4,693 – 5,389] |
| xgboost_lag5 | 611 | [536 – 792] | 4,844 | [4,537 – 5,199] |
| persistence | 482 | [387 – 805] | 6,360 | [5,972 – 6,775] |

LSTM and transformer p99 ranges fully overlap (LSTM [4,421 – 5,476], transformer [4,693 – 5,389]) -> statistically indistinguishable. All three learners overlap on every metric. Persistence retains the best median (foraging body) but pays with the worst tail.

### Interpretation

Attention matches but does not beat recurrence on small single-distribution synthetic data, consistent with the known data-hunger of transformers: high capacity with low inductive bias needs large, diverse data to pay off. On ~7,600 training windows of one distribution, the LSTM's stronger sequence bias is equally effective. This result is itself the motivation for pretrain-then-finetune; the foundation model is where the transformer's capacity meets the data scale to exploit it.

## 29/28-06-2026: Real data phase: stork resampling pipeline 

First real-data work after the synthetic ladder. Built the full pipeline from raw Movebank fixes to a benchmarked baseline ladder on real white stork, the chosen build species (just had the largest individual pool, and the switching migratory/foraging regime also mirrors that of the synthetic switching-CRW). So for now all decisions below are on stork unless noted.

### Gap characterisation and the splitting decision

Before building a resampler we characterised the gap structure of stork trajectories via sampled DuckDB queries over `locations_gps.parquet` (filter-to-sample-first, aggregate-only; never materialising the ~160M stork rows into pandas (old bug mistake this was)). We has 40 survivor trajectories; holes are near universal (37 of 40 carry a >6h gap), native cadence is heterogeneous within stork (rates from ~1s to 240 min; a fleet of distinct tag programs (recall these storks are spread across like 23 or so movebank studies)), and maximum single gaps reach ~290 days (tags dark for most of a year). So trajectory splitting must be a core resampler feature, not a rare guard.

Swept the split threshold (>1d, >2d, >7d, >14d); usable 6h points varied <2% across the range, so the choice is data insensitive. Chose **>2 days (`MAX_GAP_BINS = 8`)** on scientific grounds (same logic as the cadence choice); a multi-day gap is a behavioural discontinuity (animal stopped transmitting, reappeared elsewhere), not a hole to bridge; interpolating across it is fabrication regardless of how few points it costs. >2d is comfortably above the ~4.2h duty-cycle dropouts, so it doesn't fragment healthy tracks.

### Resampler (`src/animal_movement_fm/data/resample.py`)

One irregular trajectory in -> one or more regular-cadence segments out. Design: 6h bins (`BIN_SECONDS`); each bin represented by the **real observed fix nearest the bin centre** (never interpolated); split at gaps exceeding `MAX_GAP_BINS`; drop segments under `MIN_SEGMENT_BINS = 120`. Nearest-to-centre was chosen over mean-of-bin or first-fix because the models predict displacement between consecutive bins, so a constant ~6h spacing between representatives is the property that matters; the mean drifts mid-leg during transit and biases the switching behaviour; the first fix anchors to a floating instant and fakes speed variation; nearest-to-centre keeps each representative as close as the data allows to a fixed phase of the 6h grid. Internal sub (`MAX_GAP_BINS` gaps are *preserved, not filled*); each bin carries its absolute integer `bin_index` so downstream windowing can decide whether to cross a small gap, and a `centre_offset_s` field records how far each chosen fix sat from its bin centre (a free diagnostic for later bin-quality filtering). Validated on one real 1.5M-fix bird (7,096 bins, ~99.7% of possible bins occupied, median centre-offset 0.13 min). 14 boundary tests in `tests/test_resample.py` (nearest-to-centre selection, bin-edge floor convention, gap == vs > threshold, length floor inclusive, empty input), all passing. Deliberately a working baseline; a cadence-invariant model using $\Delta$ t is something I plan to investigate in the future (for now I want this working). 

### Segment build (`src/animal_movement_fm/data/build_segments.py`)

Two-pass build over all stork survivors, never materialising raw points into pandas; pass one `COPY`s survivor fixes to a parquet sorted by `(individual_id, deployment_id, timestamp)` so pass-two per-trajectory reads prune row groups; pass two resamples each trajectory and tags segments with a composite `segment_id`. Result: **160,318,302 raw fixes from 855 survivor trajectories -> 1,215 segments, 1,207,611 6h bins** (`stork_segments.parquet`, 29.6 MB). 8 trajectories yielded no segments ("summary optimism" — average cadence cleared the filter but real binning fragmented them below the 120-bin floor; <1%, reported honestly). Introduced DuckDB as a project dependency here (`pyproject.toml`/`uv.lock`).

### Train/val/test split (`src/animal_movement_fm/data/split.py`)

Individual-level split (all of one bird's windows go entirely to one split) because windows from the same animal are heavily autocorrelated; a window level split would leak near duplicates and measure memorisation, not generalisation to unseen animals. Threeway, ~75/12.5/12.5, fixed `seed=42`, stratified by a key (taxon — trivial within single-species stork, load-bearing at Tier 2 when whole species are held out). Per-individual window counts are steeply skewed (median 444, max 14,713); dropping individuals with `<100` windows removed 6 of 846 birds and 0.05% of windows, so set `min_windows = 100`. Final split is **train 630 birds (895,463 windows, 76.0%), val 105 (160,744, 13.6%), test 105 (121,656, 10.3%)**. Window % wobble is the expected effect of skewed per-bird counts; no leakage by construction.

### Benchmark adapter (`src/animal_movement_fm/data/benchmark.py`)

Our earlier implementation has our models and harness predicting absolute (lon, lat) positions in degrees (models difference internally but their interface is positions), and the harness windows and scores positions via haversine; the position-based system is kept to preserve comparability with the synthetic results. The adapter (`load_split_frame`, `evaluate_on_split`, `run_ladder`) filters segments to a split's individuals via DuckDB `SEMI JOIN` and feeds the existing harness, **grouping windows by `segment_id`** so a window can never cross a transmission gap (the >2d split is respected structurally, for free); the real `individual_id` is carried as an extra column for per-bird summaries. Neural models (`lstm.py`, `transformer.py`) gained a `group_col` parameter (default `individual_id`, preserving the synthetic scripts) so their internal `fit`-time windowing also groups by `segment_id` on real data.

### 29-07-2026 correction: modelling continuity and eligibility

BUG CATCH: The benchmark adapter we previously had only the coarse segment boundary created by gaps longer than two days. However, our resampler also preserves shorter gaps within each segment. Because windows were grouped by `segment_id` alone, some histories and targets crossed missing 6 hour bins. On the frozen test split, 3,997 of 121,656 targets (3.29%) were more than one bin ahead, and 14,922 windows (12.27%) contained a gap somewhere in the history or target. All real data results based on these windows are therefore provisional (RIP). The synthetic benchmark is unaffected because its trajectories contain no internal gaps.

Window continuity is now enforced directly. Within each coarse segment, `contiguous_run_slices` begins a new modelling run whenever `diff(bin_index) != 1`, and the shared window iterator creates context target windows only within those runs. This iterator is now used consistently by the evaluation harness and by XGBoost, LSTM and Transformer training. State windows follow the same run boundaries and are ordered by `bin_index`. Production regime labelling now requires `bin_index`, with labels and movement bouts restarting at every run boundary. Experiment scripts that calculate state composition or launch related quantities also load this field, preventing them from silently crossing gaps. Runs that are too short to produce a modelling window are discarded before labels are generated.

We left the frozen split manifest unchanged rather than regenerating or toping it up with replacement individuals, and instead eligibility is recalculated using the corrected contiguous window counts. Training individuals need at least one valid window, while validation and test individuals retain the existing minimum of 100 windows to keep per individual evaluation stable. After this correction, the usable data consists of 625 of 630 training individuals with 774,627 windows, 100 of 105 validation individuals with 139,714 windows, and 91 of 105 test individuals with 105,862 windows. The six corpus individuals that were not included in the original frozen manifest remain excluded.

This correction also changes the state label distribution because a movement bout now ends whenever a contiguous run ends. On our corrected frozen test set, the proportions are 0.517 moving, 0.169 settle, 0.147 local launch, 0.143 foraging and 0.023 migratory launch. Before the correction, these proportions were 0.541, 0.159, 0.137, 0.140 and 0.022 respectively. The increase in transition labels and reduction in mid bout `moving` labels are expected because gaps now truncate bouts rather than being crossed. Our classifier prevalence and PR-AUC must be recomputed and should not be compared directly with the earlier values (more RIP).

As a sensitivity check, the existing prediction dumps were filtered to retain only contiguous windows. This was used as an audit only and does not replace retraining on the corrected data. Median / p90 / p99 errors changed by −3.7% / −6.6% / −0.6% for persistence, −2.5% / −1.4% / −0.6% for one-step averaging, −3.2% / +0.4% / −0.4% for three step averaging, −4.7% / −7.6% / −4.3% for the LSTM, −2.9% / −6.6% / −3.8% for the Transformer, −8.1% / −5.9% / −1.7% for the LSTM with time features, and −5.2% / −6.8% / −1.3% for the Transformer with time features. So our main qualitative conclusions remained unchanged, although the settling tail was noticeably sensitive to the filtering.

## 05-07-2026: Diagnostic phase — characterising why persistence is hard to beat

We spent the bulk of this portion of the project on interpreting the above results. We paused forward model development to answer "my model isn't working because ..." before building further. We set out to characterise the problem and establish whether it is learnable, rather than reaching for a solution. Six model-independent (or persistence-based) analyses on the current uncleaned stork segments (105 held-out test birds, 121,656 predictions), each a runnable script in `experiments/`. A shared enabler (`dump_predictions.py`) re-runs the harness on all five models and writes per-prediction parquets (target/pred positions, error, and the true step length as a regime tag); coherence-checked against the headline runs (persistence median 0.599 km, LSTM median 1.399 km, both exact). All error reporting switched to km with spread, not point estimates alone.

### Finding: the target is bimodal, and ~a third is near-stationary (`step_displacement_dist.py`)

The 6h step-length distribution is sharply bimodal on a log axis; a near stationary foraging peak (~8 m) and a directed movement peak (~1.6 km); with an empirical trough at ~60 m that we adopt as the foraging/moving regime cut (which is data derived, not hand picked). Approx 33% of steps fall under 100 m; median step 0.63 km vs mean 7.2 km (an 11x gap that is the two-regime structure in one statistic). The distribution is not normal (a two-peak mixture, confirmed by negative log-kurtosis); the maximum step (1,405 km / 6h ≈ 234 km/h) is biologically impossible and flags the presence of a small number of data artefacts.

### Finding: 6h resampling is lossy, asymmetrically (`within_bin_variance.py`, `within_bin_drilldown.py`)

Going back to the raw fixes (median 46 fixes per surviving bin), the 6h resampling discards approximately 99.2% of raw observations, and within-bin spread rises with step length; the representation is lossiest exactly where movement is fast. A striking sub-case: some bins labelled near-stationary by their 6h step actually contain 30–100 km of within-window movement (out and back loops the grid cannot see). A drill-down classifier split these into approximately 56% genuine foraging loops versus approximately 44% bad GPS fixes, so the effect is both a real representation limitation (argument for finer cadence) and a data quality signal (a second, independent artefact detector alongside a speed cap). Threshold precision here is approximately $\pm$ 5%; treated as indicative.

### Finding: the p99 tail is real migration, not artefact (`step3_error_dist.py`)

The persistence error percentile ladder shows a 14x jump from p90 (11.1 km) to p99 (156.8 km); the bottom 90% is tightly bounded and the last decile explodes. This justifies that perhaps we should report p90 as the honest headline tail metric,since p99 is dominated by migration onset timing; an intrinsically unpredictable event, not by model quality. A plausibility check exploits the fact that persistence error equals the true 6h step, so each error converts to an implied ground speed; against a literature-backed 100 km/h stork ceiling (add ref later; 90/108 km/h sensitivity, applied over each step's real duration), only 2 of 121,656 steps are biologically impossible. The tail is therefore genuine migration the model cannot time, not data noise; cleaning is worthwhile hygiene but will not move the headline (error mass above ceiling 0.3%).

### Finding: foraging is predictable, transit is not; plus a strong diurnal rhythm (`step4_randomness.py`, `step4_hourofday.py`)

Displacement (not position) autocorrelation, split by regime, inverts the naive expectation: foraging step magnitude is predictable (lag-1 autocorr +0.35, consistent small steps) while transit is near-random (+0.02, ~47% of windows anti-persistent, erratic magnitudes); this is the mechanism behind persistence winning the foraging median. A per-window predictability score correlates with persistence error (Spearman −0.21): randomness is a real but partial driver of error, the other half being launch-onset timing which is orthogonal to within-window structure. Separately, step magnitude shows a strong 24h diurnal rhythm (autocorrelation spikes at lag 4 and 8), which would coincide with a thermal-soaring cycle; near-zero movement at night (98% stationary), peaking mid-afternoon; which is learnable, time-locked structure motivating a (local-solar) time-of-day feature.

### Finding: performance is dominated by the individual (`step5_overlays.py`)

Ranking all test segments by persistence error gives a ~900x spread (best segment median 6 m, worst 5.4 km), so the pooled aggregate median describes no real bird and is uninformative without stratification. Predicted vs actual overlays (persistence vs transformer, 9 representative segments) show the two regimes geographically; trivial foraging blobs vs full Europe -> Africa migrations; and colouring tracks by per-step error shows error concentrates on movement generally (both migration and erratic foraging), not on a single regime: "stillness easy, movement hard."

### Finding: launch onset is the hard bucket, settling is trivial (`step6_regime.py`)

Decomposing error into foraging / moving / launch / settle (foraging/moving via the 60 m cut with a 40–100 m hysteresis band to suppress threshold jitter; transitions split by direction) makes the central claims precise. Persistence wins foraging by construction (0.007 km median; learned models 30–50x worse because they predict spurious motion); learned models win the moving tail (~33% p99 reduction, 193 -> 128 km). We set out to find if "transitions are hopeless" and this was largely correct but only for launches: split by direction, launch (forage -> move) is near-unpredictable for every model (baseline p99 identical to the decimal at 81 km) while settle (move -> forage) is trivial (persistence p99 0.04 km); lumping them had masked the asymmetry. A recurring theme across every low-movement bucket: the learned models cannot predict stillness as well as persistence, which caps their achievable aggregate median and is the key limitation to carry into the FM.

### Synthesis and consequences

The answer to "my model isn't working because ...": 6h stork movement is two regimes with opposite demands. A near-stationary third of the data has "predict no movement" as its optimum, so persistence wins by construction and any learned model that predicts motion adds error; the directed remainder has a learnable part (diurnal rhythm, directed transit, seasonal-mean direction) the models do capture and an unpredictable part (launch-onset timing) that no model recovers from 6-day history and that dominates the tail. The error is the structure of the problem at this cadence, not an undertrained model. Consequences carried forward: (1) report per-regime, never the pooled median alone; (2) the aggregate-median ceiling is set by the stillness regimes, so perhaps our 'win condition' is "establish when and why the model helps" (the moving/launch tail), not "beat every baseline"; (3) motivates the cadence sweep (finer/coarser), time-of-day and seasonal features, and (for the launch residual the calendar cannot reach) environmental covariates. A raw-fix speed-cleaning pass (100 km/h, 90/108 sensitivity) is bookmarked for the next segment rebuild, verified negligible on current results.

## 07-07-2026: Cadence sweep — native inter-fix cadence characterisation

First step of the cadence sweep: before resampling at any new interval, we characterise the native inter-fix cadence to see how fine we can honestly resample. `experiments/interfix_cadence.py` measures the time between consecutive raw fixes within each trajectory (`individual_id, deployment_id`; never across a deployment boundary) over the full staged `stork_fixes.parquet`, reporting the pooled interval distribution, the per-trajectory native cadence (its spread is the fleet of tag-programs mixture), and a candidate cadence support table (fraction of fixes and of trajectories natively at least as fine as each C).

Result (full corpus, 855 trajectories, 160.3M consecutive-fix pairs): native sampling is far finer than 6h; pooled median interval 1 s, per-trajectory median 5 min, coarsest per-trajectory median 6 h. The candidate grid is supported down to 30 min for 91.6% of trajectories (99.4% of fixes); everything at 6 h and coarser is at 100% trajectory coverage. The `frac_fixes_le` / `frac_traj_le` divergence at the fine end (99.4% vs 91.6% at 30 min) is the mixture stated precisely: a ~8% minority of coarser tags contribute almost no fixes, so by data volume the corpus looks uniformly high-frequency but by animal ~8% would bin-starve a 30 min grid. For a per-bird held-out evaluation the trajectory-weighted 91.6% is the number to carry, not the fix-weighted 99.4%.

Conclusion: native cadence does not constrain the modelling interval anywhere in the 30 min–24 h range, so the interval is a modelling choice the sweep resolves, not something the gap structure derives. The finer end (30 min / 1 h / 3 h) is data-honest for ~92%+ of birds and is the interesting direction, being what attacks the 6 h representation loss the diagnostics found (~99.2% raw-obs discard; stationary-labelled bins hiding within-window loops). Caveat: this is the optimistic median-based proxy; it cannot see within trajectory duty cycling (a fine median tag can still leave fine cadence bins empty between bursts), so true fillability at any C comes only from real bin occupancy in the resample-and-benchmark step.

## 08-07-2026: Cadence sweep — resample and benchmark; 6h is near-optimal

Having established that native cadence does not constrain the modelling interval anywhere in 30 min–24 h, we ran the actual sweep: resample the stork corpus at each of 24 h / 12 h / 6 h / 3 h / 1 h / 30 min, and benchmark. `resample.py` was parameterised for this (cadence, gap threshold and segment floor became keyword arguments defaulting to the committed 6 h constants, so existing callers and the 14 resample tests are untouched). Three design decisions, all recorded in the sweep scripts: (1) the >2 d gap threshold is held in *time*, converted to bins per cadence (`round(2 days / C)`, which is 8 at 6h;  a built in coherence check); (2) the segment floor (120 bins) and context length (24 bins) are held in *bins*, so the model's input shape is identical across cadences and only the sampling rate varies; the architecture fair choice, at the cost of coarse cadences data starving (120 bins is 120 days at 24 h); (3) cross-cadence comparison uses **skill relative to same-cadence persistence**, not absolute km, because a stork moves less in 30 min than in 6 h so persistence error shrinks at fine cadence for trivial step scaling reasons; the *improvement over persistence* factors that scaling out.

The sweep ran in three passes (`experiments/cadence_sweep.py`, `_averaging.py`, `_xgboost.py`), each re-matching the canonical 6h build exactly (149 test segments, persistence median 0.599 km, p99 156.8 km) as its coherence guard.

**Coarse cadences are not headline-comparable.** At 24 h only 63 segments from 41 of 105 test birds survive the 120-bin floor (120 days of clean track), at 12 h 71 birds; the survivors are the long-track migrators, so those rows sit on a different, migration-biased subset. Full 105 bird coverage holds from 6h down. So the honest sweep is 6 h / 3 h / 1 h / 30 min; 24 h / 12 h aren't worth it. 

**Result.** Averaging (training free, CVM and k=3/10) suggested rising tail tractability toward fine cadence (k=1 p99-skill −10 % at 6 h climbing to +29 % at 30 min), but XGBoost, a model that can actually exploit structure, disconfirmed it: XGBoost p99-skill is **flat at ~36 %** across every cadence (37.9 / 33.7 / 35.6 / 36.5 % at 6 h / 3 h / 1 h / 30 min). So finer sampling yields no additional migration-tail tractability; the averaging "rise" was a weak model artefact of flat velocity extrapolation. Meanwhile the *median* penalty worsens sharply toward fine cadence (XGBoost median-skill −68 % at 6 h → −171 % at 1 h): the same "learned models inject motion into a growing stationary fraction" limitation from the diagnostic, now sharpening rather than softening as cadence gets finer.

**Conclusion.** 6 h is near-optimal on this evidence, it gives up nothing on the recoverable tail (~36 % everywhere) and carries the smallest stationary-regime median penalty. The scientifically-motivated 6 h choice is now sweep-backed rather than asserted. 

## 09-07-2026: Added PredictionContext - models consume a context object, not a bare history

We have migrated the `Predictor` contract from `predict(history)` to `predict(context)`, where `PredictionContext` (a frozen dataclass in `eval/harness.py`) bundles the `(T, 2)` history window with the target step `timestamp` and an open `extra` dict for future side information. `evaluate` now constructs one context per sliding window, carrying the **target** timestamp (the time of the step being predicted, which is known in advance and so is a proper input rather than leakage) and hands it to `predict`. All six models were migrated (each unpacks `context.history` as its first line; only XGBoost currently also reads `context.timestamp` (for now...)), the six model test suites and the harness dummy predictor updated, and a `conftest.py` added to set `OMP_NUM_THREADS=1` so the torch+XGBoost suite does not hit my macOS OpenMP segfault. Full suite green at 188.

### Why this change, and why a context object

XGBoost's sin/cos time of day feature was built and trained whenever the fit DataFrame carried a `timestamp` column, but at evaluation time the harness `predict(history)` contract had no channel to pass the target timestamp through, so `_build_features` always took its `timestamp=None` midnight fallback. The model therefore *trained* on real hours but *predicted* every window at "midnight"; a silent train/predict mismatch that degraded the diurnal feature to noise really. Wiring the timestamp through was the fix; the design question was how.

The obious fix is a second `predict(history, timestamp=None)` argument, in favour of the context object deliberately. The current numbers are exploratory scaffolding, not final (no time feature in the neural rungs right now, no covariates, no converged transformer yet too), so the model interface is being changed *now*, while nothing downstream depends on it, precisely to avoid changing it again later. A bare timestamp argument would have solved my past two day's need but forces another signature break the next time a model needs a new input, such as environmental covariates for the launch/onset residual, or $\Delta$ t for a cadence invariant model (both bookmarked for future exploration). Under `PredictionContext` those arrive as new fields (or entries in `extra`) with no change to any model's signature. The cost is a one time breaking change across the whole models layer, taken while it is cheap. The context is `frozen` so a model cannot mutate it and leak state across windows, consistent with the stateless predictor assumption the harness already relies on. History is bundled *inside* the context (true single argument `predict(context)`) rather than left as a separate positional argument, so the contract is uniform and the container is the sole input surface.

### Why only XGBoost reads the timestamp today

The timestamp now reaches every model, but only XGBoost consumes it as of now, and I'm writing this because it is a deliberate sequencing choice rather than an omission purely to stay consistent with our incremental, one change at a time approach. From our baseline ladder, the time of day is being proven on XGBoost first, a model that already trains cleanly; so we can measure its effect in isolation before it is added anywhere else. The neural rungs are deferred for two reasons: adding a feature to the LSTM/Transformer now would contaminate the raw-sequence-only comparison they exist to make, and more importantly right now, the Transformer does not yet converge on the real 6h data, so a feature added there could not be attributed cleanly against an unstable training background. So our intended order is therefore: (1) measure time of day on XGBoost; (2) resolve neural convergence at 6h, changing nothing else; (3) if the XGBoost result shows time of day earns its place, propagate it to the now converging neural models as its own isolated increment. Because the timestamp already flows in the context, step (3) is a local change (read `context.timestamp`, concatenate onto the input) with no further interface break.

Neural model internals (`_make_windows`, `fit`) and the saved checkpoints are unaffected; the change is purely at the eval-time `predisct` entry point, and the LSTM/Transformer checkpoint round-trip tests confirm existing `.pt` files still predict identically under the new contract.

## 09-07-2026: Time of day feature on 6h XGBoost; UTC A/B, then local solar correction

Based off our diurnal finding, and now that PredictionContext delivers the target timestamp end-to-end, we measure the time of day feature on XGBoost in isolation before propagating it anywhere. `experiments/xgboost_timefeature_6h.py` fits XGBoost twice on the same 6 h split; `use_time_features` off, then on, and evaluates both on the same 105 test birds, so the only thing that changes is the feature. The off run re-matches the sweep (median 1.007 km, p99 97.4 km) as its coherence guard.

**UTC first (a lower bound).** With raw UTC timestamps the feature gave a small consistent lift: median +2.8 %, p90 +4.0 %, p99 +2.5 %. This is a floor, because storks span ~3 h of longitude, so a UTC hour smears the solar rhythm across birds.

**Local-solar correction.** We then corrected the feature inside `_build_features` by shifting the hour by `longitude / 15` (15 degrees = 1 h of Earth's rotation, using the last history fix's longitude) before the sin/cos, so every bird's solar afternoon peak maps to the same corrected hour. Mean solar time only, the equation of time wobble ($\leq$  ~15 min) is negligible at 6 h cadence, so we omit it. Re-running the A/B, the correction lifted the **body** further (median +2.8 -> +4.4 %, p90 +4.0 -> +5.1 %) while leaving the **p99 tail unchanged** (+2.5 -> +2.4 %).

**Interpretation.** The correction did the physically sensible thing and confirmed our hypothesis that aligning solar phase across birds extracted more signal. But the benefit appears to land on the day to day movement body, not the migration tail, which makes sense, because the p99 tail is launch onset timing (a seasonal/calendar event) that time of day cannot reach, while the body carries the daily thermal soaring on/off rhythm the feature captures. The effect is real but modest, on untuned XGBoost treating sin/cos as two columns among thirty. Verdict for the plan is therefore that time of day earns its place and the solar correction is worth keeping, so it is carried forward to the neural models at step 3 (once they converge first (we have to fix that next)), with the expectation that the benefit is a body tightening one concentrated off the tail.s

## 12-07-2026: Transformer convergence resolved on real 6h stork data (Gadi V100)

Before adding the time of day feature to the neural models, we first isolated the existing Transformer training problem at 6h cadence and changed no model inputs or architectural components. Training was moved to NCI Gadi using a V100 32 GB GPU on the `gpuvolta` queue. Three enabling changes were made: PyTorch was pinned to `2.5.1+cu121`, because newer CUDA 13 wheels no longer supported the V100's compute capability 7.0 and produced `CUBLAS_STATUS_ARCH_MISMATCH`; optional gradient clipping was added to the Transformer, with a default of `None` so existing behaviour and tests remain unchanged; and the approximately 170 MB training tensor set was preloaded onto the GPU once inside `Transformer.fit()`, with batches selected using an on-device `randperm` rather than copied from host memory through a `DataLoader` each iteration. The training entry point is `experiments/train_transformer_gadi.py`, submitted through `experiments/train_transformer.pbs`.

The optimisation diagnostic was per-epoch training MSE. The previously converged LSTM training MSE of `0.0179` is our reference, although the two architectures need not share the same attainable training loss and the final comparison remains based on held-out movement error.

### Two learning-rate runs

Both Transformer runs used the same architecture and training configuration:

* `d_model=64`
* `nhead=4`
* `num_layers=2`
* `dim_feedforward=128`
* `dropout=0.1`
* `context_length=24`
* `batch_size=256`
* `epochs=80`
* `warmup_frac=0.1`
* linear warmup followed by cosine decay to approximately zero
* `grad_clip=1.0`
* Adam optimiser
* MSE loss

The only experimental difference was the peak learning rate, which we discuss below. 

* **iter-1 (`lr=3e-4`).** This run applied the full Transformer stabilisation recipe at 6h for the first time: a lower peak learning rate, linear warmup, cosine decay, and gradient clipping. Training was smooth and approximately monotonic, with `train_mse` decreasing from `0.0482` to `0.0214` and flattening over roughly the final 15 epochs. On the 121,656 held out test predictions, the model achieved a median error of 1.25 km, mean error of 6.97 km, p90 of 14.3 km, and p99 of 106 km. Training took approximately 72 minutes, with observed GPU utilisation around 15%.

* **iter-2 (`lr=1e-3`).** The second run tested whether iter-1 had become under-driven after the batch size was increased from 64 to 256 while the peak learning rate was reduced. Under a simple linear-scaling intuition, the larger batch might have supported a larger learning rate and potentially closed some of the training loss gap to the LSTM. The run remained stable and continued improving throughout training, with `train_mse` decreasing from `0.0510` to `0.0227`, but it finished above iter-1 and had not reached an equally clear plateau by epoch 80.

| run                | peak lr | final train_mse | test median |    test p90 |   test p99 |
| ------------------ | ------: | --------------: | ----------: | ----------: | ---------: |
| LSTM reference     |       — |          0.0179 |           — |           — |          — |
| Transformer iter-1 |  `3e-4` |      **0.0214** |     1.25 km | **14.3 km** | **106 km** |
| Transformer iter-2 |  `1e-3` |          0.0227 | **1.07 km** |     15.1 km |     109 km |

### The larger learning rate does not support the under-driven hypothesis

The iter-2 curve does not show conventional optimisation instability: the loss does not diverge, oscillate strongly, or produce numerical failures. Its small epoch to epoch increases are consistent with ordinary minibatch and dropout variation. We initially hypothesised the results as “learning rate too high” signature, but after investigation we cannot conclude this.

The comparison provides useful evidence against the specific hypothesis that iter-1 was simply under-driven. Increasing the peak learning rate from `3e-4` to `1e-3`, while holding the architecture, batch size, scheduler, clipping, epoch budget, and dataset fixed, did not reduce the final training MSE. The higher-LR run instead finished at `0.0227`, compared with `0.0214` for iter-1. The loss also improved only slowly during the early and middle portions of training before declining more strongly later in the schedule, suggesting that the larger peak learning rate did not improve optimisation efficiency under this configuration.

This result does not establish a universal upper learning rate limit, nor does it identify the exact optimisation mechanism. In particular, the two run comparison is insufficient to conclude that rare migration displacements created a sharp or ill conditioned loss surface (which we also initially suspected), even though the heavy tailed target distribution makes that a plausible hypothesis. So our conclusion is narrower; `1e-3` was less effective than `3e-4` under the tested recipe, and there is no evidence from this experiment that the lower-LR run was limited by updates that were too small.

### Median–tail trade-off

The held-out results reveal a trade-off that is not visible from training MSE alone. Iter-2 improved the median test error from 1.25 km to 1.07 km, while worsening p90 from 14.3 km to 15.1 km and p99 from 106 km to 109 km. The higher-LR checkpoint therefore performed better on the typical prediction but slightly worse in the upper tail.

This pattern is consistent with the model fitting the common, low displacement portion of the distribution differently from the rarer high displacement portion, but the aggregate metrics do not establish which behavioural regimes caused the change. In particular, they are not sufficient to claim that the higher learning rate fitted stationary steps at the expense of moving or launch steps. That mechanism must be checked directly using the existing foraging, moving, launch, and settle decomposition.

Also recall that the result also reinforces that pooled training MSE is not the final optimisation target for our project. Since squared error weights large displacement errors heavily, it is sensitive to the rare movement tail, while the geographic median is dominated by the much more common low-movement body. So our Transformer vs LSTM comparison must therefore use held out, regime decomposed errors, with particular attention to the moving and launch tails.

### Verdict and GPU-preload result

We retain **iter-1 (`lr=3e-4`, `transformer_stork_convergence_iter1.pt`) as the best tested Transformer checkpoint at 6h cadence**. It achieved the lowest training MSE and the stronger p90 and p99 results, while iter-2's only clear advantage was the aggregate median. The approximately `0.0214` training MSE is treated as a practical convergence result for this architecture and training recipe, not as an architecture level lower bound. A denser learning-rate sweep and repeated seeds could refine the estimate, but the current evidence is sufficient to proceed without attributing the earlier underperformance to insufficient training duration.

Transformer convergence at 6h is therefore considered operationally resolved: with warmup, cosine decay, clipping, and a suitable peak learning rate, the model trains stably and reaches a clear plateau. we can now complete the uncontaminated raw-sequence Transformer vs LSTM comparison and then propagate the local solar time of day feature to both neural models as a separate experimental increment.

The GPU preload produced only a modest runtime improvement, from approximately 72 minutes to 58 minutes, while GPU utilisation remained low at around 11% and reported device memory use was approximately 656 MB, including the preloaded training tensors. This indicates that host to device transfer was not the main bottleneck. The model and current batch are small enough that Python loop and kernel launch overhead likely dominate device computation. A substantially larger batch could improve utilisation now that the complete training set resides on the GPU, but batch size also changes the optimisation regime and is therefore deferred rather than altered solely for throughput.

## 13-07-2026: Raw sequence Transformer vs LSTM head to head (regime decomposed)

We now have both neural rungs converged at 6h under a matched budget, we ran the comparison the Transformer rung exists to make: does attention beat recurrence on this data? Because aggregate error is floored by the stationary body, the comparison is decomposed by regime, with attention on the moving and launch tails. Note this version is the raw sequence one, so no time of day feature on either model (yet; that is next), and kept deliberately clean so the architecture difference is not confounded with a feature. 

### Matched LSTM checkpoint

Our prev on disk LSTM (`lstm_stork_h64_l1_e60.pt`) was trained pre-Gadi at 60 epochs / batch 64, so it was not budget matched to the 80 epoch / batch-256 Transformer. We retrained it on Gadi under the Transformer's same budget (80 epochs, batch 256, same seed=42 split, with the GPU preload mirrored into `lstm.py`'s `fit()`), keeping the LSTM's conventional constant learning rate (no warmup/cosine, no clipping; each architecture keeps its own recipe). `train_mse` descended smoothly to 0.0165; below the old checkpoint's 0.0179, and was somewhat near plateau by epoch 80 (the last ~15 epochs drift ~0.13%/epoch with noise), so the ~3x fewer optimiser steps at batch 256 did not under converge it. Saved as `lstm_stork_convergence_matched.pt`.

### Method

`dump_predictions.py` was pointed at the two converged checkpoints (`lstm_stork_convergence_matched.pt`, `transformer_stork_convergence_iter1.pt`) and re-run to refresh the per prediction dumps; the existing `step6_regime.py` decomposition then ran unchanged (60 m foraging/moving cut with a 40–100 m hysteresis band, transitions split by direction into launch and settle). Coherence guards held: persistence median 0.599 km (the fixed split guard between regimes), and the LSTM/Transformer test medians (1.29 / 1.25 km) match their Gadi self evaluations. Both neural dumps were produced on mac M3 Pro CPU, so the two models are compared on identical footing. Regime composition of the 121,656 test predictions: foraging 14.2%, moving 53.7%, launch 16.0%, settle 16.0%, unknown 0.1%.

### Result: no evidence attention beats recurrence at this scale

The two tail regimes that carry the architecture question (p99, km):

| regime | persistence | lstm | transformer |
| --- | ---: | ---: | ---: |
| moving p99 | 193.2 | 127.4 | 129.0 |
| launch p99 | 81.3 | 60.4 | 60.9 |

The LSTM and Transformer are indistinguishable on both: moving p99 127.4 vs 129.0 and launch p99 60.4 vs 60.9, each ~1% apart. The remaining regimes only shuffle tiny, alternating edges; the Transformer is marginally cleaner on the stationary regimes (foraging p99 5.4 vs 7.7, settle p99 16.7 vs 17.1) and on moving median (2.07 vs 2.12 km), the LSTM marginally better on moving p99 and launch median. There is no consistent winner; every gap is somewhat 1–2 km on quantities of 60–130 km. On real 6h stork data, with matcheing converged models, attention does not beat recurrence; the same conclusion our previous 06-28 synthetic benchmark reached, now on real data. Because this is a single training seed and the gaps sit within plausible seed noise, the we don't overstate this and so can conclude that, as of now, there is "no evidence attention beats recurrence."

### What the learners buy, and what they pay

The comparison against persistence sharpens the 07-05 picture. The learned models' value is concentrated at the extreme moving tail: moving p99 falls 193 -> ~128 km (~34%). But at moving p90 they are slightly worse than persistence (27.6 vs 25.8 km) and at moving median clearly worse (2.1 vs 1.56 km); they do not predict the typical moving step better, they only suppress the worst case blow ups. On the stationary regimes they pay the stillness tax documented throughout: persistence sits at ~0.007 km median on both foraging and settle, the learned models at 0.2–0.7 km, with settle p99 increasing to ~17 km where the models inject motion into a bird that has actually stopped.

### Launch tail: tractable, not hopeless

Both learners cut launch p99 from the ~81 km baseline to ~60 km (~26%). This qualifies our 07-05 characterisation of launch as "near-unpredictable for every model": the converged learners clearly beat the baselines on the launch tail, so onset timing is partially recoverable, not hopeless, and could imporve with later additions. The moving p99 (~128 km) is unchanged from the 07-05 run, a useful coherence check. Whether convergence is what unlocked the launch improvement, or the earlier learned models already did this and the entry was worded around the baseline figure, is not recoverable; the pre-convergence per-regime numbers were not archived (results are gitignored).

### Caveat and next step

This is one training seed, and every LSTM vs Transformer gap is 1–2%, which is inside single seed variance. To state a genuine tie rather than a slight edge, both models should be retrained across a small set of training seeds (split seed 42 held fixed, training seed swept) and the moving/launch p99 reported with spread. The next increment is the local solar time of day feature on both neural rungs; and form our 07-09 XGBoost evidence that lift lands on the median/body and leaves the tail essentially unchanged, so it is expected to move the stationary and moving median numbers here rather than the moving/launch p99 the architecture question turns on.

## 14-07-2026: Time of day 2x2 table; attention exploits time, recurrence does not

We now ropagate the local solar time of day feature to the neural rungs as its own isolated increment, giving a 2x2 (raw vs time, LSTM vs Transformer). Each time model is its raw counterpart's recipe with the feature turned on as the single change (Transformer: iter-1's lr=3e-4 / warmup / clip; LSTM: matched constant lr=1e-3), so time vs raw is a one variable ablation per architecture. Our 07-09 XGBoost A/B  test predicted the lift would land on the median/body and leave the p99 tail flat; that prediction held for the LSTM but was overturned for the Transformer.

### The feature and its wiring (Design B)

The new feature lives in `features/time.py` (`local_solar_sincos`, and comes from XGBoost so the definition is shared and identical). We chose per-position injection over a single target time value: each displacement is tagged with the local solar (sin, cos) of the fix it lands on (destination tagging), so the input is 4 channels `[dlon, dlat, sin, cos]` and the sequence model sees time at every step; for the Transformer this means attention can weight fixes by time of day, which is the whole point of asking whether attention buys anything. Note, only the two displacement channels are standardised; the sin/cos are already bounded. `PredictionContext` gained a `history_timestamps` field so `evaluate` threads each window's per-fix clocks through to predict time, matching the fit time feature exactly. The whole thing is a default-off toggle (`use_time_features`), so the raw models, their checkpoints, and the tests are untouched (suite now 214 with the time coverage). Trained on Gadi to `transformer_stork_time.pt` and `lstm_stork_time.pt`; both converged cleanly (train_mse 0.0182 and 0.0146, each below its raw counterpart's 0.0214 / 0.0165; a lower training loss with the extra informative feature, no instability). Same `dump_predictions.py` / `step6_regime.py`, `MODELS` extended to the four neural cells. Coherence held with persistence median 0.599 km; time model medians (transformer 0.95, lstm 1.37 km) match their Gadi self evaluations.

### Result: a stark architecture x feature interaction

p99 error by regime and model (km):

| regime | persistence | lstm | transformer | lstm_time | transformer_time |
| --- | ---: | ---: | ---: | ---: | ---: |
| foraging | 0.09 | 7.66 | 5.43 | 7.80 | **4.62** |
| moving | 193.2 | 127.4 | 129.0 | 122.4 | **120.8** |
| launch | 81.3 | 60.4 | 60.9 | 61.5 | **57.5** |
| settle | 0.04 | 17.08 | 16.69 | 14.75 | **6.22** |

median error by regime and model (km):

| regime | persistence | lstm | transformer | lstm_time | transformer_time |
| --- | ---: | ---: | ---: | ---: | ---: |
| foraging | 0.007 | 0.202 | 0.402 | 0.493 | 0.258 |
| moving | 1.558 | 2.121 | 2.071 | 2.247 | **1.833** |
| launch | 1.619 | 1.697 | 1.807 | 1.756 | 1.660 |
| settle | 0.007 | 0.709 | 0.661 | 0.584 | 0.388 |

We interpret this as evidence of an architecture by feature interaction under the tested recipes. The Transformer improves relative to its raw counterpart in every reported regime on both median and p99: foraging p99 falls from 5.43 to 4.62 km, moving p99 from 129.0 to 120.8 km and moving median from 2.07 to 1.83 km, launch p99 from 60.9 to 57.5 km, and settle p99 from 16.7 to 6.22 km. The LSTM response is mixed: moving median worsens from 2.12 to 2.25 km, foraging median rises from 0.20 to 0.49 km, and launch p99 changes little, while moving and settle p99 improve modestly. Under this feature wiring and these training recipes, the Transformer therefore makes broader use of the temporal covariate than the LSTM. `transformer_time` is the best learned model in every p99 regime and in the moving, launch, and settle medians; the raw LSTM retains the best learned model foraging median.

### Time reached the tail, with the largest gain on settling transitions

The Transformer result overturns the earlier expectation that time would affect only the distribution body. Relative to the raw Transformer, moving p99 falls from 129.0 to 120.8 km and launch p99 from 60.9 to 57.5 km, both reductions of approximately 6%. These effects are modest and require multi-seed confirmation, particularly for launch.

The clearest effect occurs on steps classified retrospectively as settling: Transformer p99 falls from 16.7 to 6.22 km, a reduction of approximately 63%, while median error falls from 0.661 to 0.388 km. This is consistent with local solar time helping the model avoid extrapolating recent movement into periods when continued movement is less likely. It does not mean that the model directly identifies the settling regime, and persistence remains structurally dominant once the realised target step is near-stationary.

### Synthesis

Taken together, the raw and time augmented runs show that the two neural architectures are difficult to distinguish when trained only on displacement history, but respond differently when local solar time is attached to every position. Under the tested recipes, the Transformer converts the additional temporal information into improvements across all four regimes and across both the distribution body and tail. The LSTM achieves a lower training MSE but shows mixed held out effects, improving some tail and settling errors while worsening several regime medians.

This is evidence that the Transformer is the more effective covariate-conditioned architecture in the present experiment, rather than evidence that recurrence is generally unable to use temporal covariates. It supports carrying a Transformer conditioned on side information forward into the foundation-model design, while leaving the strength and reproducibility of the architecture by feature interaction to be established through repeated training seeds and additional covariates.

### Caveats and next

Single training seed. The Transformer's large effects (settle p99 -63%, aggregate median 1.25 -> 0.95 km) are big enough to be almost certainly real; the ~6% tail moves and the LSTM's small negatives sit within plausible seed noise and will need multi-seed confirmation (split seed 42 fixed, training seed swept, p99 reported with spread) before they are leaned on. It is also a single covariate; the launch residual the calendar cannot reach still motivates environmental covariates later. Immediate follow-ups: the multi-seed spread, and refactoring `xgboost.py` to call the shared `local_solar_sincos` (its inline copy is identical but not yet the single source of truth).

## 20-07-2026: Movement state labelling, the migratory launch definition, and a scrapped onset probe

Following John's advice to make the regime state a prediction target, not just a diagnostic lens, we will try to predict the state, then forecast movement, so we built the labelling infrastructure the state work needs. Four commits are in; one exploratory script is being deleted and redone. This entry records both.

### What went into the codebase

`analysis/regime.py` is now the canonical movement state labeller. The hysteresis rule and the launch/settle direction split were lifted verbatim from `experiments/eda/step6_regime.py` so the new state work and the existing regime decomposition share one definition, exactly as `features/time.py` did for time of day. A fix's state is set by the step that arrived at it, which matches `true_step_km` in the prediction dumps. Validated against the Step 6 labels on the test split: `step_km` reproduces `true_step_km` to 2.3e-13 km, and 121,453 of 121,656 regimes match. All 203 differences are window edge artefacts and the raw labeller is the better informed one in every case: 149 are each segment's first prediction target, which the preds frame cannot label (its shifted `true_step_km` is NaN), and the other 54 are a grey band cascade seeded by that same boundary, where the preds sequence has no prior clear state to inherit and falls back to the 0.06 km cut while the raw labeller inherits the bird's true prior regime.

The composition run (`experiments/state/state_composition.py`) then produced a result that changed our plans. Over 1.2M fixes the four states are moving 54.0%, launch 15.8%, settle 15.8%, foraging 14.3%, with launch and settle balanced at ~157 per segment and 98.9% of segments containing at least one. Base foraging (foraging + settle) is ~30%, which recovers the "~1/3 near stationary" diagnostic exactly. But **launch as defined is not the migration onset event**: it is any foraging to moving flip, i.e. the bird's ordinary daily start of movement, and it is common rather than rare. The rare tail event we care about is buried inside it.

So we refined it. `moving_bouts` decomposes each track into the moving bouts a launch opens, measuring net displacement, path length, straightness and duration; `label_migratory_launches` then splits every launch into `migratory_launch` / `local_launch` on the bout's net displacement. The cut is `MIGRATORY_NET_KM = 37.0`, taken from a C. ciconia paper we read (will add ref later or in thesis) rather than from our own trough. The paper classifed a white stork day as migratory versus stopover at 37 km moved between consecutive night roosts. This sits above the documented resident foraging range (<=28 km breeding, <=48 km to landfills) and is bracketed by C. ciconia's own realised migration rates. An earlier 30 km candidate was rejected because it came from the congener C. boyciana. Note the cross scale mapping; the literature threshold is per 24 h roost to roost, our bout is the moving portion and often sub daily, so a bout reaching 37 km net inside a day meets or exceeds a migratory day's rate. Under this definition 26,758 of 190,421 launches are migratory (14.05% of launches, 2.22% of all fixes) with median straightness 0.95 and median length 3 fixes.

We validated three ways, in a throwaway script not kept in the repo; the numbers here are the record. 
1. Geography: the migratory bouts trace both known white stork flyways, pinching at the Strait of Gibraltar and the Bosphorus/Levant and avoiding the open central Mediterranean, exactly as a soaring migrant must, and reaching the Sahel and southern Africa. 
2. Seasonality: launches peak in August-September and February-April with a trough in the May-July breeding months. 
3. Direction: autumn bouts are 90.7% southward (mean dlat -1.43 deg), spring bouts ~89% northward (+1.52 deg), endpoints spanning -29 to +56 N. 23 unit tests cover the labeller and the bout/migratory split.

### The onset probe, and why it is being deleted

We then ran a short script to check whether an imminent migratory launch can be forecast; standing on a currently foraging fix, does the next 6 h step begin a migratory launch. We used a gradient boosted classifier over nine hand engineered features (day of year, position, time of day, and movement summaries over an 8 fix window), conditioned on currently foraging so the negatives were not padded with trivial cases where a launch is impossible by definition. It worked! Test ROC-AUC 0.956 and PR-AUC 0.710 against a 7.2% prevalence, roughly 10x chance, with 80% precision at 50% recall. The informative part was the permutation importance, where recent movement dominated (`recent_step_sum` 0.30, `recent_step_max` 0.23), time of day came third (0.13), and season and latitude were marginal (<=0.03). Reading: the sharpest sign a foraging stork is about to launch is that it has moved a lot in the last two days, i.e. it is at a stopover mid migration, and migration begets migration. Season adds little on top because large recent movements only happen in migration season anyway, so the movement features already carry it.

But I am not commiting this script to the repository. It answered the question with a bespoke 8 fix feature window and its own pipeline rather than our standard 24 fix context window and `evaluate` harness, so its numbers are not comparable to any ladder rung and, more importantly, it cannot become a second head on the LSTM or Transformer, which is what John's advice suggested (predict state, then movement) and mixture of experts ideas require. But it was a nice short experiment to run. The replacement is a `window -> next state` classifier on the standard harness and split, reporting per class precision/recall with `migratory_launch` as the headline class and the currently foraging slice reported separately as the hard subset, which recovers the probe's framing as a slice of a general model rather than a separate pipeline. 

Two findings carry forward. The dominant signal, recent movement magnitude over ~2 days, sits comfortably inside a 24 fix (6 day) window, so a window based state head should have it natively without feature engineering. And the place season should matter is precisely where movement history is uninformative: the cold first departure from the breeding grounds, not the mid migration relaunch.

### Next

Wire a day of year sin/cos covariate before building the classifier, so the movement models and the state head share one input representation. One encoding only (day of year, not month and season and day of year, which are the same signal at three resolutions and would just add redundant channels), cyclic only, no absolute year. This takes the neural input from 4 to 6 channels, standardising the displacement channels only, as before. Persistence and averaging are training free and ignore covariates; ARIMA is a trained rung but fits ARIMA(2,0,2) per channel on the displacement series inside the context window with no exogenous term, and day of year is effectively constant across a 6 day window, so a seasonal exog regressor would be a degenerate constant column. And so our rerun is therefore XGBoost, LSTM and Transformer. Prediction to record before running: unlike time of day, season barely varies within a window, so it acts as a conditioning signal rather than something attention can weight positions by, and we would not expect it to reproduce the architecture by feature interaction; and given the probe, expect a modest aggregate effect with the launch tail the place to look. These runs will be multi seed (split seed 42 fixed, training seed swept) so the result arrives with spread rather than the single seed caveat carried by the time of day 2x2.

## 22-07-2026: Multi-seed covariate sweep (5 seeds x 2 architectures x 3 feature cells) on Gadi H200

Every neural result up to here rested on a single training seed, so our time of day 2x2's small effects could not be separated from seed noise. We ran the full grid as one PBS array job (`experiments/training/train_multiseed.pbs` + `train_multiseed_gadi.py`): 3 feature cells (raw / +time / +time+season) x 2 architectures (LSTM, Transformer) x 5 training seeds = 30 runs, split seed fixed at 42 so every run sees identical windows and the only within-cell variation is the training seed. Ran on the `gpuhopper` queue (NVIDIA H200); ~18 min/run, ~950 SU total. Recipes are unchanged from the single seed runs (LSTM lr 1e-3 constant; Transformer lr 3e-4 with warmup+cosine+grad-clip), so the numbers are comparable to the earlier ones. Inference was written over both val and test as one predictions parquet per run; the analysis is a separate, decoupled step (`experiments/reporting/aggregate_multiseed.py`) that re-tags each prediction's regime with the canonical labeller and reports mean +/- sd across the 5 seeds. A per-run coherence check (recomputed aggregate median vs the run's own summary JSON) passed for all 30.

### Result: moving-tail p99 (km, test, mean +/- sd over 5 seeds)

| cell        | lstm          | transformer   | transformer - lstm |
| ----------- | ------------: | ------------: | -----------------: |
| raw         | 131.7 +/- 0.5 | 130.1 +/- 0.5 |               -1.6 |
| time        | 126.7 +/- 0.8 | 120.5 +/- 0.6 |               -6.2 |
| time+season | 123.8 +/- 1.1 | 117.8 +/- 0.6 |               -6.0 |

Launch p99 (km): raw 61.4 +/- 1.0 (lstm) / 60.9 +/- 0.5 (transformer); time 59.7 +/- 1.3 / 57.8 +/- 0.5; time+season 59.6 +/- 1.1 / 58.1 +/- 0.6. Aggregate median (km): transformer flat at ~1.0 across all three cells; lstm 1.4 -> 1.6 -> 1.7 (worsens as covariates are added). Seed sd is small throughout (<=1.3 km on p99, <=0.1 km on median), so the effects below are resolved rather than noise.

### Three findings, now with error bars

1. **Raw sequences: attention approximately equals recurrence.** Moving p99 130.1 vs 131.7, a ~1% gap. It is technically beyond 1 sd, but the sign is the opposite of the single seed head to head (which had the Transformer 1.6 km *worse*), so the honest reading is a near-tie whose direction is not even stable across setups. Without a covariate, attention buys essentially nothing over recurrence at this scale. This is exactly the case the single seed run could not settle, and the seed sweep settles it.

2. **Time of day: the architecture x feature interaction is real.** Adding time widens the moving p99 gap to ~6 km: the Transformer improves by -9.6 km, the LSTM only -5.0 km, and the LSTM's aggregate median *worsens* (1.4 -> 1.6). Our previous single seed 2x2 claim ("attention exploits the covariate, recurrence squanders it") replicates across 5 seeds and is thus no longer a single-seed caveat.

3. **Season behaves as pre-registered (both predictions in the prior methods entry confirmed).** We recorded two predictions before running: season would NOT reproduce the interaction (it is near constant within a window, a conditioning signal rather than something attention weights positions by), and it would not reach the launch tail.
   - Interaction: adding season on top of time helps the moving tail *architecture neutrally*, -2.9 km (lstm) vs -2.7 km (transformer); roughly equal, unlike time of day. Confirmed.
   - Launch tail: time+season vs time on launch p99 is -0.1 km (lstm) and +0.3 km (transformer), both inside seed noise. Season conditions the diurnal body but cannot call the seasonal *onset* event. Confirmed.

### Interpretation and consequences

This sweep supersedes the single seed time of day 2x2 as the headline DL result and upgrades its central claim from suggestive to "robust with error bars"; the Transformer is the more effective covariate conditioned architecture, the LSTM is destabilised on the body by the same covariates (its median degrades), and neither the clock nor the calendar reaches the launch onset tail. Two consequences. For the foundation model: a covariate conditioned Transformer is the right shape. For the launch residual: it needs *environmental* covariates (wind, thermals, land cover), since both within day (time) and within-year (season) signals are now shown, with spread, to leave it essentially untouched. Outputs: `results/multiseed/regime_summary.csv`, `regime_by_run.csv`, `multiseed_regime_p99.png` (all gitignored). Caveat: still a single data split (seed 42) and a single site/species; the seed sweep addresses training init variance, not split or population variance. 


## 25-07-2026: Part B state work - next state classifier, the two factor state head, and a pre-registered null

Two meetings ago, John suggested making movement regime an explicit prediction target rather than using it only as a diagnostic lens. The idea was to predict the state first, then forecast movement conditioned on that state (two factor, with a mixture of experts planned for later). This entry covers that full piece of work, which ultimately ended in a disconfirmed prediction.

### The labelled window dataset (`state/dataset.py`, tested)

`build_state_windows()` turns a segment frame into supervised examples using the same input representation as the neural rungs: 24 fixes differenced into 23 displacements, with up to 6 channels, and each covariate attached to the fix on which its displacement lands. This means anything trained on these windows can later become a head on the same trunk without needing to rebuild the windows. The label is the refined state (`analysis.regime`) of the window's target fix, which is the same fix whose position is predicted by the movement head. The target fix is not included in the input window, so this is a genuine one step forecast rather than a nowcast. A nowcast would use the state of the final inwwwindow fix, which is already made obvious by its incoming step. `meta.can_launch` marks windows whose last fix is foraging. Since a launch is defined as a foraging->moving transition, only these windows can come immediately before a launch, so they are also reported separately as the harder subset. This behaviour is covered by `tests/test_state_dataset.py`, which checks the window counts, label and target alignment, displacements against `np.diff`, covariate channels against `features/time.py`, and agreement with the canonical labeller.

### Is state predictable at all? (`experiments/state/train_classifier.py`)

We used one lean XGBoost multiclass model; a flattened 24x6 window, plus momentum summaries calculated from the same window, balanced class weights, and a 5-way softmax over the refined states. Overall accuracy is not very informative when migratory launch prevalence is only ~2%, so the main metrics are per-class precision, recall and one vs rest PR-AUC, compared against the prevalence floor.

| class | precision | recall | F1 | support | prevalence | PR-AUC | lift |
| ---------------- | --------: | -----: | ----: | ------: | ---------: | -----: | ---: |
| foraging | 0.846 | 0.855 | 0.850 | 17,061 | 0.140 | 0.935 | 6.7x |
| moving | 0.962 | 0.902 | 0.931 | 65,770 | 0.541 | 0.981 | 1.8x |
| local_launch | 0.812 | 0.739 | 0.774 | 16,709 | 0.137 | 0.866 | 6.3x |
| migratory_launch | 0.522 | 0.805 | 0.633 | 2,729 | 0.022 | 0.744 | 33.2x |
| settle | 0.730 | 0.878 | 0.798 | 19,387 | 0.159 | 0.852 | 5.3x |

The main result is that migratory launch onset is forecastable one step ahead, with PR-AUC 0.744 against a 2.24% base rate, or a 33x lift. Importantly, this result also holds on the harder subset. When restricted to the 36,499 windows ending in foraging, where prevalence is 7.48%, PR-AUC is 0.746 and therefore essentially unchanged. This suggests the result is not caused by filling the negative class with cases where a launch is mechanically impossible. Most of the gain comes from recent movement momentum (`mag_last`, `mag_last3`) and the time of day of the most recent fixes. No day of year feature appears in the top 15, which is consistent with season only mattering for the initial cold season departure. Precision is 0.52, with most of the limitation coming from ambiguity between local and migratory launch. Some of this is unavoidable because the distinction depends on how far the bout travels, which has not yet been observed at prediction time.

### The two factor state head (Rung 2), on both architectures

Given that state was predictable, we added a second head to both the LSTM and Transformer trunks behind `use_state_head`. It is off by default, so all previous checkpoints, results and tests remain unchanged. The state head reads the same window summary as the position head and produces 5 state logits. Its softmax output is then concatenated back onto the summary before it is passed to the position head, making the displacement prediction explicitly conditioned on regime. The model is trained jointly as `MSE(position) + lambda * CE(state)`. Per-window state labels are produced by running the same window loop agaim, and tests assert that they match `build_state_windows` exactly. This ensures the state head is trained on the same targets as the standalone classifier.

A CPU smoke run then exposed an important scale issue. The standardised position MSE is around ~0.02, while the state cross-entropy is around ~0.7. With the initially chosen lambda=0.5, roughly 95% of the gradient came from the state task, and the treatment lost to the baseline on nearly every cell. The approximate balance point is ~0.03. Rather than choosing one value immediately, the Gadi job was changed into a lambda sweep.

### Result: 2 architectures x lambda in {0.01, 0.03, 0.10}, test split, seed 0

We ran one PBS array (`experiments/state/eval_state_head.pbs`) with 6 subjobs. Each subjob trains a baseline model using time+season with no state head, and a treatment model with the state head at the selected lambda. Both use the same recipe and seed and are evaluated through the standard harness. The full error distribution is reported both overall and by true next state. Since the lambda=0 baseline is retrained inside each subjob, it appears three times per architecture. These repeated baselines were byte identical across all three subjobs: LSTM 7.085 / 1.706 / 100.065 km mean/median/p99, and Transformer 6.305 / 1.008 / 95.642. This also provides a useful determinism check for the seeding and split. Runtime was ~47 min per LSTM cell and ~2 h 05 per Transformer cell. The harness predicts one window at a time, so Transformer inference across 121,656 test windows accounts for most of the difference.

| arch | lambda | median km | p90 km | p99 km | moving p99 | state acc |
| ----------- | -----: | --------: | -----: | -----: | ---------: | --------: |
| lstm | base | 1.706 | 14.271 | 100.065 | 124.206 | — |
| lstm | 0.01 | 1.751 | 14.579 | 100.473 | 124.053 | 0.819 |
| lstm | 0.03 | 1.866 | 14.593 | 99.913 | 122.642 | 0.828 |
| lstm | 0.10 | 1.819 | 14.001 | 99.832 | 122.296 | 0.843 |
| transformer | base | 1.008 | 13.169 | 95.642 | 116.927 | — |
| transformer | 0.01 | 1.048 | 13.311 | 96.181 | 117.112 | 0.790 |
| transformer | 0.03 | 1.013 | 13.030 | 96.139 | 118.135 | 0.844 |
| transformer | 0.10 | 1.081 | 13.362 | 99.717 | 122.748 | 0.872 |

### The pre-registered prediction is disconfirmed

Before running the experiment, we guessed that moving/launch p99 would improve while the median would barely change. Neither prediction held. The overall median gets worse in all six cells: LSTM +2.6 / +9.4 / +6.6%, and Transformer +3.9 / +0.5 / +7.2%. The tail also does not improve consistently. Overall p99 changes by between -0.2% and +0.4% for the LSTM, which is effectively no change, and by +0.5 to +4.3% for the Transformer, which is worse. Moving p99 contains the only result that could arguably be a real effect: a monotonic LSTM improvement with lambda of -0.1 / -1.3 / -1.5%. However, the largest improvement is only ~1.5%, which is about 2 sd of the seed spread measured in the 22-07-2026 sweep. The Transformer also moves in the opposite direction at +0.2 / +1.0 / +5.0%.

Migratory launch p99 varies without a clear pattern: LSTM -0.4 / +7.3 / -10.3%, and Transformer -3.7 / -0.2 / +7.6%. This is not surprising. With n=2,729, the p99 is approximately the 27th largest error, so this column is badly under powered and no conclusion should be drawn from it. The state head itself is not what failed. State accuracy increases monotonically with lambda for both architectures: LSTM 0.819 -> 0.843 and Transformer 0.790 -> 0.872. This confirms that the loss weight is doing what it is meant to do. The best results are also comparable to the standalone XGBoost classifier's accuracy of 0.866. The state task is being learned; it simply does not improve the position decoder.

### Why, and what it does and does not rule out

There is a structural reason this design was unlikely to help, which we under weighted when choosing it. The conditioning vector is a deterministic function of the trunk summary. The state head computes `softmax(W . last)`, and we then pass `[last, softmax(W . last)]` to the position head, even though the position head already had access to `last`. From an information perspective, nothing new is being added. The position head could already have learned state dependent behaviour directly from the same vector. What the state head really adds is gradient pressure from the CE term and a 5 dimensional compressed view of information already stored in the trunk. For this task, neither appears to help. At the same time, the CE term creates a measurable cost for the near-stationary body of the distribution, which is where the median lies.

This is a null result for cheap soft conditioning, NOT for the wider state programme. A mixture of experts changes the functional form rather than simply changing the input to a shared head. Separate expert parameter sets, selected by a gate, would allow a stationary expert to specialise in "predict approximately zero" while a transit expert specialises in directed displacement. A single shared linear head cannot represent these behaviours separately in the same way. Rung 3 therefore remains open on its own merits and was not tested by this experiment. The main caveat is that this used a single training seed (seed 0) and a single split (42). Most of the changes are smaller than the seed sd measured in the multi seed sweep. We therefore describe the result as a null because the effects are both very small and inconsistent in direction across architectures, which is more consistent with noise than with a real effect being suppressed. Outputs: `results/state_head/eval_state_head_{arch}_s0_lam{lambda}.{json,csv}` (gitignored).


## 25-07-2026: A split reproducibility bug, and why the split is now a committed manifest

**Noticed.** While comparing the state-head sweep with the classifier, the per-class test supports did not match. The classifier run on the Mac reported foraging 17,335 / moving 65,305 / migratory_launch 2,781, while the sweep run on Gadi reported 17,061 / 65,770 / 2,729. Both runs contained exactly 121,656 windows, so the problem was not the total number of windows. The difference was in which windows received each label.

**Diagnosed.** The parquet file was byte identical on both machines, with md5 `459456b9d4a090275b6345c6c23f5009`, and the numpy (2.4.4) and pandas (3.0.2) versions were also the same. The two independent labelling paths, the positional approach used in `build_state_windows` and the `(segment_id, timestamp)` merge used during evaluation, agreed exactly with each other on each machine, which ruled out the labelling code. We also checked the floating point explanation directly rather than assuming it was impossible. The nearest step to a hysteresis threshold is 5.5e-06 km away, and no step is within 1e-06 km of a cut. ARM vs x86 float64 differences are around ~1e-16 relative, which is twelve orders of magnitude too small to change any decision. After this check, I turned to GenAI and explained what was going on and began pasting a particular files that could be the root cause of this bug, and sure enough, GenAI helped me spot the issue (referenced in the code, and note, it suggested the fix, and I implemented it myself). 

Hashing the split revealed the actual issue: the two machines shared only 76 of 105 test individuals. `make_split` seeds its shuffle, but `rng.permutation` is applied to whatever order it receives. That input order came from `window_counts_per_individual`, which ends with an unstable `sort_values("n_windows")`. Of the 840 eligible individuals, 393 share a window count with at least one other bird. These tied birds could therefore appear in a different order on each platform, after which the seeded permutation selected different individuals. This was confirmed directly. With the data and `seed=42` held fixed, simply shuffling the rows of `counts` reduced the overlap to 13/105 shared test individuals. The seed makes the shuffle reproducible, but it cannot correct a non-canonical input order. I also realised that `data/split.py` had no test file, (a huge oversight on my behalf and certainly a big learning curve) which is why this remained unnoticed after the split was first written.

**Fixed.** `make_split` now sorts IDs canonically before applying the seeded permutation, making the split a pure function of (individual_ids, seed). On the real corpus, the assignment is now stable at 105/105 test individuals under arbitrary input reordering. More importantly, experiment scripts no longer recompute the split. The assignment is now frozen in a committed ID manifest, `splits/stork_split_seed42.json`, containing 630/105/105 individuals and loaded through `data.split.stork_split()`. A manifest for any dataset can be generated with `experiments/data/make_split_manifest.py`, and `tests/test_split.py` contains 11 tests covering order invariance, disjointness, the min_windows filter and the manifest round trip.

We deliberately froze the GADI assignment rather than replacing it with the newly canonical split. No split is more correct than another as long as it is performed by individual, fixed and reproducible, and the Gadi assignment is also the one used for all GPU results, including the 30 run multi-seed sweep and the six cell state head sweep. A split regenerated using the corrected code shares only 17/105 test birds with the Gadi split, so freezing the existing assignment was the only way to retain those results without recomputing all of them. Every within machine comparison was still valid, including the multi-seed findings and the state head null, because each baseline and treatment was trained in the same subjob using the same individuals. The invalid comparison was placing Mac and Gadi results side by side, which is what exposed the bug. Freezing a separate split for each dataset will also matter as the corpus grows. A recomputed split depends on the full ID set, so adding species at Tier 2 would reshuffle existing individuals and silently invalidate Tier 1 results.

## 26-07-2026: Rung 3, the mixture of experts. The mechanism is confirmed, but only in the stationary regime

Rung 2, which conditioned the position head on the predicted state distribution, returned a null. The reason (we think at least) to be structural rather than empirical. Its head is LINEAR over `[last, p_state]`, so the state enters ADDITIVELY and can only add a regime specific offset; it cannot change how the head reads the trunk.

A mixture of experts is different:

    pos = sum_k g_k * (W_k . last + b_k),   g = softmax(gate . last)

So the effective mapping `sum_k g_k W_k` changes from one window to the next. This is a obviously a different mechanism, not simply a stronger version of Rung 2, and that is why our results from Rung 2 null did not rule it out; and so to ensure we stay rigourous, we thought we should implement and test this approach. We used K=5 experts, one for each refined state, with the gate supervised using the canonical labels so that expert k aligned with state k. Routing was soft, and the experts were kept linear, and so the comparison is exactly one linear head versus five linear heads plus a gate.

We ran a PBS array (`experiments/state/eval_moe.pbs`): 2 architectures x lambda in {0.01, 0.03, 0.10}, 80 epochs, frozen split, seed 0. Each subjob also retrained its own lambda=0 baseline as an internal control. Runtime was ~47 min per LSTM cell and ~2 h 05 per Transformer cell.

### The routing worked; this is not a degenerate result

The baseline was byte identical across all three lambda cells within each architecture; LSTM median 1.706404 km and Transformer median 1.008140 km. This confirms that the runs were deterministic. Gate usage also tracked the true state prevalence closely, with foraging 0.14 vs 14.0% actual, moving 0.53 vs 54.1%, and `migratory_launch` 0.022 vs 2.2%. So the gate did not appear to collapse onto a single expert.

Diagonal routing mass increased steadily with lambda. For LSTM foraging it rose from 0.631 -> 0.728 -> 0.754, while for Transformer foraging it rose from 0.589 -> 0.774 -> 0.780. At lambda >= 0.03, every state routed mainly to its matching expert. But at lambda = 0.01, `migratory_launch` was the only exception bc it routed mainly to `local_launch` on the LSTM and to `moving` on the Transformer. Gate accuracy reached 0.85-0.87, which is comparable to the standalone XGBoost classifier.

### Result: the pre-registered prediction is disconfirmed, but the mechanism is confirmed

We guessed that moving/launch p99 would improve while the median stayed roughly unchanged. Almost the opposite happened. Moving p99 was essentially flat across all six cells, ranging from -1.2% to +1.5%. Migratory launch p99 was inconsistent: LSTM -11.5 / -14.1 / -0.7%, and Transformer +5.9 / +3.1 / +2.8%. With n = 2,729, that p99 is based on roughly the 27th largest error, so it is under powered and no conclusion should depend on it.

The clear result is in the STATIONARY regime, which improves substantially in every cell and on both architectures!

| median error change | foraging | settle | local_launch | moving | migratory_launch |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM lam 0.01 / 0.03 / 0.10 | -33 / -24 / -49% | -34 / -9 / -18% | -9 / -8 / -10% | -1.5 / +12.6 / +15.6% | -0.5 / -7.3 / -4.5% |
| Transformer lam 0.01 / 0.03 / 0.10 | -39 / -63 / -60% | -20 / +46 / -7% | -2 / -12 / -7% | -2.0 / -6.1 / -4.3% | +0.7 / +0.8 / +3.6% |

Foraging p99 also improves in all six cells, by between -9% and -52%. This matches the mechanism we had in mind; a stationary expert can specialise in predicting approximately no movement, while a transit expert handles directed steps. One shared linear head cannot specialise in both behaviours at the same time.

The pooled metrics barely change because `moving` makes up 54% of the windows and dominates the pooled tail, yet it does not improve. The best cell is LSTM lambda = 0.01, with overall median -13.8%, p90 -1.9%, mean -2.6%, and p99 +1.4%. This is the only lambda where the stationary gains are not clearly paid for elsewhere. Obvioously this is still a single seed result, roughly 2-3 sd of the multi-seed spread, and the Transformer at the same lambda moves in the opposite direction on median (+6.1%). The overall effect is therefore not consistent across architectures.

### Lambda trades one regime against the other

On the LSTM, increasing lambda makes the gate specialise more strongly and continues to improve foraging, but moving median gets steadily worse:

    -1.5% -> +12.6% -> +15.6%

So clearly we face a trade of with lambda here, where it doesn't have 'one best value', but rather it controls how much travel accuracy is traded for stationary specialisation, which is a useful result in its own right.

### The number that reframes the whole state programme

Persistence, which predicts exactly zero displacement, gets a median error of 0.008 km on foraging fixes. The baseline gets 0.961 km and the MoE gets 0.639 km. So our MoE therefore closes about a third of the gap to persistence on the stationary states; 34% foraging, 34% settle, and 38% local_launch. However, it is still roughly 80x worse than simply predicting no movement.

The learned models are still badly overpredicting movement when the animal is stationary. A soft mixture of LINEAR heads recovers only part of the available gain because it can move towards zero, but it cannot commit to zero as a discrete outcome.

### Consequences

The three experiments now form a fairly clear argument. State is highly predictable from the standard window, with PR-AUC 0.754 and a 33x lift. Conditioning a shared linear head on that state additively does nothing, for a structural reason. Gating separate experts multiplicatively confirms that regime specific specialisation is real, but the benefit appears almost entirely in the stationary regime and only closes about a third of the gap to persistence.

The main conclusion is that the regime split probably belongs in the OUTPUT DISTRIBUTION rather than in the architecture. A more suitable design perhaps would be(we will explore this later) a Bernoulli "will it move" gate multiplied by a heavy tailed displacement density, such as a bivariate Student-t. This would allow the model to emit no movement as a discrete decision rather than only approaching zero continuously.

That design would also be label free and therefore more species general. By contrast, our supervised 5 expert gate depends on `analysis/regime.py`, whose 60 m hysteresis cut and 37 km migratory threshold are white stork constants and do not transfer across species. For that reason, the MoE architecture will be reported as a Tier 1 result but will NOT be carried into the multi species tiers. The finding will be carried forward instead.

## 10-08-2026: Context length 24 vs 12, and why a 1% tail cost buys the marine leg

Our preregistration amendment A6 shortens the context from 24 history bins to 12. Here records the ablation behind it.

**Why the question arose.** Windows per individual are sum over maximal contiguous runs of $max(0, L − C)$, so the context length is subtracted **once per contiguous run** rather than once per individual. That distinction is invisible on white stork, whose tracks are long and near continuous at 6h. But on our fragmented marine tracks, this was a big issue. Halving C multiplies usable windows by 1.04x for stork but 1.70x for SOES, 1.57x for NES, 1.80x for Adelie penguin and 2.62x for Weddell seal. It is the largest data lever available without changing the 6h cadence or moving to some form of $Delta$ taware model; both of which are beyond our timeframe as of now. 

That framing is also exactly why the ablation needed a rule fixed in advance. "Shorten the context until the marine species become eligible" is a convenience driven choice dressed as a methods decision, and it is indistinguishable from the honest version after the fact. So the decision metric, the tolerance and the action under each outcome were written into `experiments/reporting/context_ablation.py` and committed at `491320d` before the C=12 runs were read. The script records the rule as `rule_set_before_reading` and writes the verdict, the per architecture pass flags and the consequent action to `results/context_ablation/decision.json` mechanically, so the verdict is not something we decide after results are viewed. 

**Design.** The C=12 cells were run as indices 30–39 of the existing multiseed PBS array, on the `time_season` covariate cell, both architectures, 5 seeds each, against the frozen manifest `splits/stork_split_seed42.json`. Every C=24 target is also a C=12 target, because shortening the context can only add eligible windows at the start of a run and never remove one. The comparison is therefore **paired**; `c12_paired` scores the C=12 model on the C=24 target set and inherits the C=24 regime labels, and that arm is the decision number. Inheriting the labels matters because `tag_regime` is hysteretic, so relabelling on the C=12 frame would let the regime assignment itself shift and confound the comparison. `c12_native` is reported alongside as the arm on its own larger eligible set, and the script hard fails if any C=24 target is absent from the C=12 run, which is the coherence guard that the two arms really did read the same manifest.

**Registered rule.** Preserved if and only if `|p99(c12_paired) − p99(c24)| $\leq$ 2 x pooled seed sd` on the moving regime, for **both** architectures.

**Result.** Moving regime, test split, mean across 5 seeds with seed sd in parentheses:

| arch | metric | C=24 | C=12 paired | change |
| --- | --- | ---: | ---: | ---: |
| LSTM | mean | 10.303 (0.098) km | 10.245 (0.048) km | −0.56% |
| | median | 2.465 (0.116) km | 2.246 (0.041) km | −8.89% |
| | p90 | 26.705 (0.344) km | 26.795 (0.318) km | +0.34% |
| | p99 | 121.839 (0.791) km | 123.299 (0.569) km | +1.20% |
| Transformer | mean | 9.495 (0.013) km | 9.483 (0.029) km | −0.13% |
| | median | 1.924 (0.012) km | 1.922 (0.014) km | −0.09% |
| | p90 | 25.658 (0.201) km | 25.612 (0.161) km | −0.18% |
| | p99 | 114.875 (0.667) km | 115.770 (0.781) km | +0.78% |

LSTM p99 rose 1.46 km against a 1.9 km tolerance; Transformer p99 rose 0.90 km against 2.1 km. Both pass, so our verdict was PRESERVED and the registered action fires; register C=12 as an amendment, and rebuild Tier 2 eligibility at C=12.

**What the numbers actually say, which is less than "C=12 is better".** Both p99 deltas are positive. The registered rule asks only that the tail be preserved within tolerance, and it is, so the correct phrasing is then *statistically indistinguishable*, not *improved*. The cost is roughly a 1% increase confined to the extreme tail; the LSTM median improves 8.89%, which is a larger proportional move but lands in a different part of the distribution, and the two do not net against one another. What we bought is not skill, it is windows...

**The assumption this leaves open.** The ablation is white stork only. The species that stand to gain most from a shorter context, like SOES, NES, Adelie penguin, Weddell seal, are precisely the ones not in it, and their tracks differ from stork in the exact property the context length interacts with. A model with 12 bins of history on a fragmented run has both less history and, relatively, more of its windows near a run boundary. That C=12 preserves tail skill on stork is a result; that it does so on fragmented marine tracks is an assumption, and A6 declares it as one rather than reporting it as established.


## 16-08-2026: Tier 2 pretraining composition and the sampling rule (A8)

In section 2 we locked in our modelling window and split policy, but we never said how we sampled windows from the source pool. Below are the window counts from our frozen manifes:

| species | domain | role | train ind. | train | val | test |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| *Ciconia ciconia* | flying | source | 630 | 806,595 | 146,603 | 110,897 |
| *Larus fuscus* | flying | held-out | 332 | 597,407 | 10,400 | 29,817 |
| *Cervus elaphus* | terrestrial | source | 306 | 556,579 | 18,189 | 19,106 |
| *Canis lupus* | terrestrial | held-out | 99 | 47,136 | 7,258 | 7,619 |
| *Mirounga leonina* | marine | source | 417 | 82,731 | 2,370 | 2,332 |
| *M. angustirostris* | marine | held-out | 353 | 53,442 | 3,150 | 1,408 |

**Our issue.** If we do a uniform draw over pooled windows (every dataloaders default currently), this would mean our marine section of the corpus is **5.7%** of the loss. So our Z model would be evaluated on NES without having barely seen a seal, and thus this likely weak marine result would be unattributable between source scarcity and taxonomic distance.

**Best of a bad situation...** We draw species proportional to n^0.5, so then a window uniformly within the species we have: *C. ciconia* 46.491%, *C. elaphus* 38.619%, *M. leonina* 14.889%.

## 16-08-2026: physical plausibility QC applied to the marine pair (A9)

We dropped the four physical plausibility flags at extraction instead of only reporting them, which cost us 29.4% of our SOES fixes and 13.5% of NES but pulls the worst 6h step down from 12,959 km to 3,822 km. Windows take a big loss, going down by 36% and 16% against 5.7% and 3.9%, since emptying a single bin splits a contiguous run and costs up to twelve windows. We rederived Gate 2 on the refrozen standardisation and nothing important moved; A4 still returns `0.1 x p50`, the marine pair goes 1.42 -> 1.46, so no role changed, and our A8 draw probabilities become 47.911 / 39.799 / 12.290; though SOES rho flipped sign (-0.165 -> +0.041) once (what appears) the speed outliers stopped manufacturing fake reversals.


## 26-08-2026: Tier 2 freeze and sealed evaluation — WEAK/NULL

Froze the configuration (grid 11: `d_model` 256, 4 layers, LR 1e-3, 20,000 updates at batch 256), the seed set 0-4, S and F's checkpoint-selection rule and F's budget, then opened the three held-out test partitions once. Verdict **WEAK/NULL**: Z does not beat persistence on any held-out species, B (package stripped) matches or beats Z, and S reaches +0.2047 on NES. The failure is concentrated in the low displacement stratum, where Z appears to over move by up to 28x persistence's p90.

## 27-08 to 20-09-2026: Tier E, exploratory, validation only

Four blocks, each registered in `tierE_exploratory_ablations.md` before running, none reportable as a Tier result:

- **Architecture 2x2 pooled.** Reconstruction alone (`Brecon`) reaches +0.2279 on NES, the largest pooled skill anywhere
  in the study; the conditioning path costs on every species.
- **Species conditioning (E0-E2).** A learned per species row helps only NES, and identity can be supplied for an unseen
  species without its data, but it is not what was missing.
- **F budget (Flong, Spre).** At the registered 2,000 steps fine-tuning barely moves the model; this motivated Tier 3's
  20,000 step F.
- **Single species 2x2 (65 runs).** Within one species the full package never beats the plain trunk, on any of four
  species including white stork where data is not the limit; the conditioning path is negative in all four.

## 16-09-2026: Tier 3 registered — source pool scaling, three sources to nine

One variable changes: the pool. Configuration, budget, representation, metrics, held-out species and protocol are Tier 2's. The nine sources were chosen mechanically — keep Tier 2's three, add two per domain, maximise mean pairwise descriptor distance over training partitions only, with a 2,000 window floor. Three further species are reserved from Tier 3 and Tier 4 entirely so a later study has unread test partitions. 158 registered runs: Z9, B9, Za9, Zd3, Zd1, B9dom, F9, F3c, Sk and the Tier 1 baseline ladder rebuilt on the registered method.

## 20-09-2026: A8 counts corrected before any Tier 3 run

The window counter matched manifest ids to data ids as plain strings where the training pipeline matches them canonically, so four candidate species were undercounted and one was wrongly excluded by the eligibility floor. Re run mechanically on corrected counts the rule selects the same nine; only the draw probabilities move. We recoreded this as a dated correction in `preregistration_tier3.md` sec 1, with the superseded values kept inside the regenerated artefact.

## 21-09-2026: Tier 3 trained and frozen

All 158 runs complete, 1,876.8 SU, every run at its full budget with nothing withheld. Two things recorded in the freeze before the seal: doubling the budget to 40,000 steps does not lower the training loss, so the budget is not the binding constraint; and at the registered A8 weights white stork takes about 64% of the training loss from 33% of the draws while the three marine sources take about 3% between them, so the nine source pool is in gradient terms close to a flying species model. The pre seal prediction is recorded in sec 7.
