# ERA5 covariates for the 1 h pooled birds (exploratory)

**GenAI reference:** I gave GenAI the env study's design, the dry run, calibration and coverage outputs, the driver and evaluator, and our 1 h freeze, and it wrote the tables and numbers from them in that freeze's layout. Wording is mine. Tables = GenAI, paragraphs surrounding them = mine. 

To adhere to our prereg, this is exploratory and never reported as a tiered result. Written on 16-10-2026, before any env run of the plan; the only env training so far is the fold 6 calibration in sec 1 (training windows only, no run record, no validation or test). Committed before any run, then committed again with sec 7 filled before the sealed evaluation. 

## 1. Status and what has been opened

| | |
|---|---|
| status | exploratory; no number here is a Tier 2 or Tier 3 result |
| env runs trained | none |
| ERA5 extraction | 25,894,524 segment rows, the 30 birds' every 1 h fix (training, validation and test individuals), from NCI's `rt52` ERA5 single-level reanalysis; coverage 1.0 for every channel and bird, no missing month (`results/birds_1h_env_build/coverage.csv`) |
| dry run (15-10) | all 10 folds, training individuals only: the env pool's movement inputs bitwise equal to the main study's (first 12 channels, masks, targets, scale, k_h); env_stats; the share beyond ±10 SD (sec 5). `results/birds_1h_env_build/dry_run_f*.json` |
| calibration (15-10) | fold 6 on Gadi: BbE and LSTMBbE at 1,000 updates, one XGBBbE 24 h fit. Training windows only; no run record, no validation or test (`results/birds_1h_env_calibration/`) |
| test partitions | the env parquets hold ERA5 at test fixes too, but no test ids, windows or targets were read. All 30 birds' test partitions were opened before this study (1 h freeze sec 1) |
| 1 h study | sealed separately (`birds_1h_exploratory_freeze.md`); its ablation evaluation runs under that freeze. This study edits none of its files |
| 1 h ablation | the sealed evaluation finished on 16-10 and was read before this freeze was committed. Longer context raised LSTM-Bb and XGB-Bb skill at 6 h; Bb's predictions collapsed toward persistence at C ≥ 48 (median predicted 6 h displacement 5 m at C = 96, against 583 m for LSTM-Bb) (post-hoc check of the saved 1 h predictions, seed 0: median over birds of the median distance between each arm's 6 h prediction and persistence's). Sec 6 was written on 16-10, before those results were available, and is unchanged |
| reserved | *O. dalli* and *P. adeliae* are not in this set and stay unread |

## 2. Question

Given ERA5 weather at each context fix, does the pooled transformer (Bb) gain more than LSTM-Bb and XGB-Bb? All three get identical inputs: the 1 h main study's 12 movement channels plus the same 15 env channels (sec 5), on the same folds, pools, windows and recipes.

## 3. Design

Folds, pools, exclusions, recipes, encoding, k_h, heads and seeds are our 1 h main study's at C = 12 (`birds_1h_exploratory_freeze.md` secs 2 and 3), unchanged. An env arm adds its 15 channels and nothing else (`TrainConfig.env_channels` = 15; every other config field is its base arm's).

| arm | env | base (main study) | per | seeds / horizons | plan indices | runs | queue | compared with |
|---|---|---|---|---|---|---:|---|---|
| BbE | yes | Bb | fold (0–9) | seeds 0–2 | 0–29 | 30 | gpuhopper | BbR |
| LSTMBbE | yes | LSTM-Bb | fold | seeds 0–2 | 30–59 | 30 | gpuhopper | LSTMBbR |
| BbR | no | Bb | fold | seeds 0–2 | 60–89 | 30 | gpuhopper | Bb (main, gpuvolta) |
| LSTMBbR | no | LSTM-Bb | fold | seeds 0–2 | 90–119 | 30 | gpuhopper | LSTM-Bb (main, gpuvolta) |
| XGBBbE | yes | XGB-Bb | fold | 1, 6, 24 h | 120–149 | 30 | normal | XGB-Bb (main) |
| **total** | | | | | | **150** (120 neural, 30 XGBoost) | | |

- **Reruns.** BbR and LSTMBbR are our main study's Bb and LSTM-Bb retrained without env on gpuhopper, with the same seeds and configs. Every neural run here is full FP32 (TF32 off in cuDNN and matmuls, recorded per run), so BbE − BbR and LSTMBbE − LSTMBbR compare env against no env on the same hardware and precision. BbR − Bb and LSTMBbR − LSTM-Bb are reported as a hardware check (sec 4).
- **XGBBbE** fits on the main study's XGB-Bb draws: the same 1,000,000 A8 draws per horizon from windows with a target at h, seed 0, with the 15 env channels appended to each token. It is compared with the main study's XGB-Bb directly.
- **Final checkpoint** for every pooled arm, as the main study.

| | |
|---|---|
| driver | `experiments/training/bird_1h_env_train.py`, `bird_1h_env_train.pbs` (gpuhopper) and `bird_1h_env_train_cpu.pbs` (normal); records tagged `birds_1h_env`, `exploratory: true`, `opened_test_partition: false`, with `env_stats`, `env_clip_sd`, the env files' sha256 and the TF32 flags. The driver refuses test partitions, the wrong queue, TF32 on, and any step budget but 20,000 |
| calibration | fold 6 (the largest pool, 13,811,209 windows), indices 18 / 48 / 140, NVIDIA H200: BbE 0.0252 s/update, 286 s build, 0.22 h extrapolated to 20,000 updates, 17.3 GB GPU allocated; LSTMBbE 0.0073 s/update, 280 s build, 0.12 h, 17.3 GB; XGBBbE 24 h 252 s build, 20.5 s fit, 6.7 GB RSS. The pool is 16.4 GB on device at 27 channels (7.3 GB at 12) |
| resources | neural: walltime 01:00:00, 48 GB, 12 CPUs, 1 GPU (gpuhopper); XGBoost: walltime 01:00:00, 32 GB, 16 CPUs (normal) |
| compute | about 1.9 KSU for the 150 runs (below); evaluation uncalibrated |

**Compute estimate**, from the calibration JSONs' extrapolated hours (build plus 20,000 updates; XGBoost build plus fit), at 90.0 SU per walltime hour on gpuhopper (our own calibration job 181332811: 10.70 SU in 7m08s, 12 CPUs, 1 GPU, 96 GB; the pooling study's index 5, 13m59s for 20.98 SU, agrees) and 32.0 on normal at 16 CPUs (1 h freeze sec 3). Fold 6 stands in for every fold, and the reruns take their env arm's figure, so both are upper bounds; per-job startup is not in the JSONs and is not included.

| arm | runs | h per run | SU per run | SU |
|---|---:|---:|---:|---:|
| BbE | 30 | 0.22 | 19.7 | 592 |
| LSTMBbE | 30 | 0.12 | 10.6 | 319 |
| BbR | 30 | ≤ 0.22 | ≤ 19.7 | ≤ 592 |
| LSTMBbR | 30 | ≤ 0.12 | ≤ 10.6 | ≤ 319 |
| XGBBbE | 30 | 0.08 | 2.4 | 73 |
| **total** | **150** | | | **≤ 1,895** |

## 4. Statistic and scoring

As our 1 h freeze sec 4, unchanged, at C = 12: travel p90 skill against P per bird and horizon, individual weighted, median across seeds (XGBoost single seed); paired bootstrap of 2,000 resamples at seed 42; $\pm$ 0.05 read as no difference. The same floor (150 windows per test individual per horizon), scored rule ($\geq$ 5 test_scored individuals at the floor), common set, travel boundary per bird and horizon, latitude rule and scoring rules, so the scored birds are the 1 h main study's: 28 at 1 h, 27 at 6 h, 28 at 24 h; common set 27. Every arm is scored on the 1 h main eval's test_scored windows. Evaluator: `experiments/reporting/bird_1h_env_sealed_eval.py`, which imports the 1 h evaluator for all of the above.

| reading | definition | per | reported |
|---|---|---|---|
| skill against P | BbE, LSTMBbE, XGBBbE, BbR, LSTMBbR, Bb, LSTM-Bb, XGB-Bb | bird × horizon | skill; medians over scored birds and over the common set |
| env effect | BbE − BbR; LSTMBbE − LSTMBbR; XGBBbE − XGB-Bb | bird × horizon | skill difference with metre and skill intervals; medians, counts above +0.05 / within / below −0.05, and intervals excluding 0, over scored birds and the common set |
| contrasts | (BbE − BbR) − (LSTMBbE − LSTMBbR); (BbE − BbR) − (XGBBbE − XGB-Bb) | bird × horizon | difference with a skill interval from the same resamples as the env effects (each arm's headline seed, the same draws and cached p90s); medians, band counts and intervals excluding 0 |
| hardware check | BbR − Bb; LSTMBbR − LSTM-Bb | bird × horizon | difference with intervals; medians only. No bands, no pass/fail |
| tie-back | re-predicted Bb, LSTM-Bb, XGB-Bb skill − the 1 h sealed report's (`results/birds_1h_sealed/birds_1h_results.json`) | bird × horizon | max \|diff\| and median per arm; no pass/fail |

So no threshold or p value decides any of them.

**Evaluator guards.** It refuses before this freeze is committed with secs 6 and 7 filled; unless all 150 env runs and the 90 main-study runs it reloads (Bb, LSTM-Bb, XGB-Bb at C = 12) are recorded and clean; on any env run whose `env_clip_sd` is not 10, or whose checkpoint disagrees with its record on env_stats, the clip, k_h or C; on any env parquet whose sha256 differs from the records; and without the 1 h sealed report. Env arms standardise our test windows with their own record's env_stats and the clip. The env test windows are checked bitwise against the 1 h main eval's (fixes, keys, anchors, targets, persistence). Inference is FP32 (TF32 off, recorded in the output). `--diagnostic` reads run records and env files only and opens no test partition. Only `--predict` reads test.

## 5. Covariates

Each window's 11 tokens carry ERA5 at the fix each token steps to (t − 10 … t), appended after the 12 movement channels. The first context fix (t − 11) and every target fix are never read.

| channels | what | units |
|---|---|---|
| u10, v10 | 10 m wind, east and north | m s⁻¹ |
| u100, v100 | 100 m wind, east and north | m s⁻¹ |
| t2m | 2 m temperature | K |
| blh | boundary-layer height | m |
| tp | total precipitation, hourly accumulation | m |
| msl | mean sea-level pressure | Pa |
| lsm | land–sea mask (static) | fraction, 0–1 |
| elevation | surface geopotential / 9.80665 (static) | m |
| tail10, cross10, tail100, cross100 | wind along and across the token's heading θ: tail = u cos θ + v sin θ, cross = v cos θ − u sin θ (positive to the left); 0 where the heading is 0 | m s⁻¹ |
| env_observed | 1 when all ten raw values are finite; else the token's 14 channels are 0 and the flag 0 | 0 / 1 |

Units are ERA5's own conventions; the probe did not record unit attributes. The training pool means are consistent with them (fold 6: t2m 288 K, msl 101,486 Pa, blh 617 m, elevation 208 m).

| | |
|---|---|
| time | each fix uses the ERA5 hour at or before its timestamp (floored to the hour) |
| space | bilinear interpolation on the 0.25° grid (1,440 × 721), packed values decoded at the four corners before interpolating; longitude wraps, latitude is clamped to the outer rows |
| coverage | 1.0 for every channel and bird (sec 1), so `env_observed` is 1 on every token in this dataset; it is kept for shape |
| statistics | per run, from that run's training pool only: mean and SD of the 14 standardised channels over observed tokens, each token weighted by its window's p_s / n_s. Recorded in the run record and reported checkpoint; validation and test windows use the record's values |
| clip | every standardised channel clipped at ±10 SD after standardisation (`era5.CLIP_SD`); env_stats stay those of the raw values; the flag is untouched. Decided 16-10 from the dry run, before any env training |

Dry run shares beyond $\pm$ 10 SD before the clip, over the 10 folds (observed training tokens, unweighted, 73.5M–151.9M per fold). After the clip, no value lies beyond $\pm$ 10 by construction.

| channel | share beyond ±10 SD | max \|z\| |
|---|---:|---:|
| tp | 0.105–0.132% | 81.6–89.6 |
| elevation | ≤ 1.75e-4 (≤ 26,426 tokens, fold 5) | 14.1 |
| blh | ≤ 27 tokens | 10.6 |
| the other 11 | 0 | ≤ 8.5 |

## 6. Expectations

Written before any env training. Again, $\pm$ 0.05 is no difference. Note, I gave dot point expectations to GenAI to write into the table when making this file. These are below. 

| item | expectation | reason |
|---|---|---|
| Env effect, 1 h | within ±0.05 for all three families | the next hour is dominated by the recent track |
| Env effect, 6 h and 24 h | median ≥ 0 for at least two of three families | wind and boundary-layer height shape travel speed and direction over hours |
| Contrasts (BbE − BbR) − (LSTMBbE − LSTMBbR) and (BbE − BbR) − (XGBBbE − XGB-Bb) | within ±0.05 at all horizons | Bb's near-zero skill at C = 12 looks like a pooling or optimisation limit that more input channels will not fix |
| Hardware check | medians of BbR − Bb and LSTMBbR − LSTM-Bb within ±0.02 | FP32 on both, same seeds and configs |

## 7. As trained

Will write from `bird_1h_env_sealed_eval.py --diagnostic` once all 150 runs are complete, before the sealed evaluation.

## 8. Limitations

- ERA5 orography is about 31 km: elevation and the near-surface fields are grid-cell values, not what a bird meets at its fix.
- Reanalysis is not forecast-available: the covariates at context fixes are known after the fact. This is an operational caveat, not target leakage; no target fix is read.
- Hourly ERA5 is floored to the hour, so a fix late in an hour reads conditions up to 59 minutes old.
- Covariates are read at context fixes only, never along the path to the target.

## 9. Note

No env run of the plan existed when this file was first written; the only env training was the fold 6 calibration (sec 1), on training windows only, with no record, validation or test. Only `bird_1h_env_sealed_eval.py` scores test partitions. 

A transformer diagnosis (training loss against zero and last-step reference predictors, and prediction behaviour on validation windows) runs in parallel from existing 1 h checkpoints. It reads no test partition and changes nothing in this study; any repaired transformer would be a separately preregistered arm.
