# 1 h rebuild of the rotating held-out birds (exploratory)

**GenAI reference:** I pasted terminal outputs and multiple files to GenAI to simply draft the tables, some of the numbers, species names, and then added the text. 

To adhere to our prerego, this is exploratory and never reported as a tiered result. Written on 11-10-2026, after our rotation study's seal, before any 1 h trained run of the plan; the only 1 h training so far is listed in sec 1 (which was training windows only, no run record, no validation or test). Committed before any run, then committed again with sec 7 filled before the sealed evaluation. 

## 1. Question and status

The rotation scored one 6 h step from 72 h of context. Every bird's median fix interval is $\leq$ 1 h, so the same 30 birds and folds are rebuilt on a 1 h grid: 12 h of context, direct forecasts at 1, 6 and 24 h. Does pooling across birds hold up against each bird's own specialist at each horizon, and does a longer, gappier history (C = 24–96 h) help?

| | |
|---|---|
| birds | the rotation's 30, folds 0–9 as our `registered/birds_rotation_folds.json`; fold 0 is trained fresh here, not carried |
| grid | 1 h bins, current edge binning; segments split at 48 bins (2 days), so no target up to 24 h crosses a split |
| main study | C = 12 contiguous bins ending at the anchor t; targets at t + 1, t + 6 and t + 24 h, direct, the bins between not required |
| ablation | C = 24, 48, 72, 96 on the main study's anchors and targets; the history before the 12 bin core is taken as observed and masked where empty |
| 1 h training before this freeze | Mac toy runs from build steps 3.5 / 3.6: 200 updates, *A. platyrhynchos* and *T. chrysostoma*, C = 12 and 24. The fold 6, C = 96 calibration on Gadi: Bb and LSTM-Bb at 1,000 updates, one XGB-Bb 24 h fit. Training windows only; no run record, no validation or test |
| test partitions | none opened for the first time: all 30 were opened by sec 22 or earlier. The cadence survey read test_scored window counts only |
| reserved | *O. dalli* and *P. adeliae* are not in this set and stay unread |

## 2. Folds and pools

Folds, pools, genus rule and training exclusions are our rotation's (`birds_rotation_exploratory_freeze.md` sec 2), unchanged. Fold 0s pool is the bird study's 22 bird main pool; its A8 probabilities were never registered (fold 0 was carried), and so they are derived by the rotation rule (`SPW.rotation_probabilities`) on its registered pool and post-exclusion counts, pinned in the driver and checked against `results/birds_1h_build/fold0_probabilities.json` on every load. A8 weights stay on the registered 6 h counts (sec 8).

| fold | held out | pool birds | pool training windows at 1 h |
|---:|---|---:|---:|
| 0 | *L. fuscus*, *A. albifrons*, *S. turtur* | 22 | 6,677,638 |
| 1 | *A. anser*, *A. brachyrhynchus*, *A. caerulescens*, *A. fabalis* | 25 | 12,300,258 |
| 2 | *C. ciconia*, *M. bassanus* | 28 | 12,397,942 |
| 3 | *A. alba*, *L. argentatus*, *T. antarctica* | 26 | 8,695,063 |
| 4 | *B. canadensis*, *B. leucopsis*, *P. apivorus* | 27 | 13,507,785 |
| 5 | *C. aura*, *G. himalayensis*, *M. milvus* | 27 | 13,745,003 |
| 6 | *A. platyrhynchos*, *M. americana*, *P. occidentalis* | 27 | 13,811,209 |
| 7 | *H. ostralegus*, *P. roseus*, *T. chrysostoma* | 27 | 13,163,077 |
| 8 | *C. corone*, *D. exulans*, *P. leucorodia* | 27 | 13,255,200 |
| 9 | *F. naumanni*, *F. tinnunculus*, *H. caspia* | 27 | 13,529,820 |

Windows are anchors with at least one target, training individuals after exclusions; per bird and horizon they are pinned in our driver and checked against `results/cadence/horizon_counts.csv`. Excluded individuals (as the rotation): *A. albifrons* 6, *C. ciconia* 44, *C. aura* 18, *G. himalayensis* 10, *M. americana* 1.

## 3. Arms and runs

Recipes by reference to the bird study's (`preregistration_birds.md` sec 4, through `BT.build_config`): grid 11, 20,000 applied updates at batch 256. Each neural arm is one model with three heads (1, 6, 24 h), loss masked where a target is missing; each head's target is divided by the window scale x k_h, k_h the A8 weighted median of r_h over the run's training windows (window weight p_s / n_s), written to the run record. Heads are averaged equally (masked mean per head, then the mean over heads present; a head with no target in a batch is dropped from that batch's mean). XGBoost fits one regressor per horizon in the same target space. The pooled arms report their final checkpoint. LSTM-S validates every 1,000 updates and keeps the lowest mean, over horizons with $\geq$ 1 validation individual at the floor, of its validation travel p90 / P's; if no horizon qualifies, it reports its final checkpoint (recorded). XGB-Bb fits on 1,000,000 A8 draws per horizon from windows with a target at h; XGB-S on all of a bird's training windows with a target at h. Seeds: neural 0–2, XGBoost 0.

**Encoding.** The registered 12 channel ENU tokens, `bin_seconds` 3,600; per window scale from the 12 bin core only, floored at 30 m; `d_east_scaled` and `d_north_scaled` clipped at $\pm$ 1,000 on every token, core and extended, in both studies. In our ablation, an extended token steps from the previous observed fix; a token spanning k > 1 bins divides its displacement by k (`dt_bins` and `log_dt` keep their timestamps); missing bins are masked (Bb: attention mask; LSTM-Bb: observed-flag channel, zero fill; XGB-Bb: NaN). The C = 12 core is bitwise the main study's (core tokens, scale, targets, k_h and anchor sets; `core_matches` in the dry run), so every C is scored on the same windows and persistence is identical across C.

| study | arm | per | seeds / horizons | plan indices | runs | queue |
|---|---|---|---|---|---:|---|
| main | Bb | fold (0–9) | seeds 0–2 | 0–29 | 30 | gpuvolta |
| main | LSTM-Bb | fold | seeds 0–2 | 30–59 | 30 | gpuvolta |
| main | LSTM-S | held-out bird (30) | seeds 0–2 | 60–149 | 90 | gpuvolta |
| main | XGB-Bb | fold | 1, 6, 24 h | 150–179 | 30 | normal |
| main | XGB-S | held-out bird | 1, 6, 24 h | 180–269 | 90 | normal |
| ablation | Bb | C × fold × seed (4 × 10 × 3) | seeds 0–2 | 270–389 | 120 | gpuvolta |
| ablation | LSTM-Bb | C × fold × seed | seeds 0–2 | 390–509 | 120 | gpuvolta |
| ablation | XGB-Bb | C × fold × horizon (4 × 10 × 3) | 1, 6, 24 h | 510–629 | 120 | normal |
| **total** | | | | | **630** (390 neural, 240 XGBoost) | |

P and the specialists run at C = 12 only. The plan is append only and contiguous by queue; the main study runs first, the ablation after, under this freeze.

| | |
|---|---|
| driver | `experiments/training/bird_1h_train.py`, `bird_1h_train.pbs` (gpuvolta, 12 CPUs) and `bird_1h_train_cpu.pbs` (normal, 16 CPUs); records tagged `birds_1h`, `exploratory: true`, `opened_test_partition: false`. The driver refuses test partitions, the wrong queue and any step budget but 20,000 |
| calibration | fold 6, C = 96, 1,000 updates (10-10): Bb 0.079 s/update (0.025 encode, 0.054 GPU), 147 s build, 0.48 h extrapolated to 20,000, GPU 4.5 GB; LSTM-Bb 0.026 s/update (0.023 / 0.003), 148 s build, 0.18 h, GPU 2.8 GB; XGB-Bb 24 h one fit in 11.9 min walltime, 14.75 GB. C = 12 runs use the on-device pool and no CPU encoding |
| resources | walltime 01:00:00, mem 32GB per subjob, both queues; no prefetch thread |
| compute | about 3.6 KSU for the 630 runs, of 98 KSU available (below); evaluation uncalibrated |
| persistence | P = the last core fix, per held-out bird and horizon; no other training-free rung |

**Compute estimate**, from the fold 6, C = 96 calibration jobs' measured walltimes:

| job | walltime | SU | SU per walltime hour |
|---|---:|---:|---:|
| Bb, gpuvolta | 00:06:02 | 3.62 | 36.0 |
| LSTM-Bb, gpuvolta | 00:04:39 | 2.79 | 36.0 |
| XGB-Bb 24 h, normal | 00:11:51 | 6.32 | 32.0 |

- Each neural job's walltime, less its build and 1,000 updates, is fixed startup per job: 136 s for Bb, 105 s for LSTM-Bb.
- A neural run is startup plus build plus 20,000 updates, with the time per update scaled by the token count (C − 1) / 95; at C = 12 it is the GPU part only, since the pool is on the device.
- An XGBoost run is a fixed 283 s (the Bb job's startup and build) plus the calibration's remaining 428 s, scaled by rows / 1,000,000 and (C − 1) / 95.
- The longest run is about 0.52 h, inside the 01:00:00 walltime.
- Cross-check: the C = 12 LSTM runs at 2.6 SU each are in line with the rotation's 493 SU over 198 runs.

| study | arm | C | runs | h per run | SU per run | SU |
|---|---|---:|---:|---:|---:|---:|
| main | Bb | 12 | 30 | 0.11 | 4.1 | 122 |
| main | LSTM-Bb | 12 | 30 | 0.07 | 2.6 | 78 |
| main | LSTM-S | 12 | 90 | 0.07 | 2.6 | 234 |
| main | XGB-Bb | 12 | 30 | 0.09 | 3.0 | 89 |
| main | XGB-S | 12 | 90 | 0.08 | 2.7 | 244 |
| ablation | Bb | 24 / 48 / 72 / 96 | 30 each | 0.19 / 0.30 / 0.41 / 0.52 | 6.7 / 10.6 / 14.6 / 18.6 | 1,517 |
| ablation | LSTM-Bb | 24 / 48 / 72 / 96 | 30 each | 0.11 / 0.14 / 0.18 / 0.21 | 3.8 / 5.1 / 6.4 / 7.7 | 691 |
| ablation | XGB-Bb | 24 / 48 / 72 / 96 | 30 each | 0.11 / 0.14 / 0.17 / 0.20 | 3.4 / 4.4 / 5.4 / 6.3 | 585 |
| **main** | | | **270** | | | **767** |
| **ablation** | | | **360** | | | **2,794** |
| **total** | | | **630** | | | **3,561** |

## 4. Statistic and scoring

Per bird and horizon: travel p90 skill against persistence, individual weighted, median across seeds (XGBoost single seed), paired bootstrap of 2,000 resamples at seed 42, $\pm$ 0.05 read as no difference. Each of our comparisons prints a metre interval and a skill unit interval from the same resamples, as sec 22. Evaluator: `experiments/reporting/bird_1h_sealed_eval.py`.

| | |
|---|---|
| floor | 150 windows per test individual per horizon (150 h, the F = 25 of the 6 h grid held at equal time) |
| scored | a bird is scored at h with ≥ 5 test_scored individuals at the floor; below that it is reported unscored at h, with n |
| common set | birds scored at all three horizons; every per-horizon summary carries a common-set column beside the scored one |
| travel boundary | per bird and horizon, max(A4, 30 m): A4 = 0.1 × p50 of displacement between occupied bins h apart within a segment, training individuals after exclusions; 30 m is `SCALE_FLOOR_M`. Pinned in the driver from `results/birds_1h_build/boundaries.csv` (90 values); the floor binds in 33 of the 90 cells. The share of test windows in travel is reported per bird and horizon |
| latitude rule | a prediction (east, north) from the anchor inverts at the predicted latitude: lat = anchor lat + north / R, then Δlon = east / (R cos lat). This is the exact inverse of the target encoding, which measures east at the target's latitude (`enu_to_lonlat_exact`); used by LSTM-S selection and the sealed evaluator alike |
| ablation strata | the share of extended-history bins observed at that C: gappy < 0.9 (0 included), full ≥ 0.9. Individuals are held at the floor on the whole horizon, not within a stratum |

**Scoring rules** are the rotation's sec 4, unchanged: a test bird is dropped if it shares ≥ 100 exact fixes with any individual of another species (identity) or with a validation bird of its own species (val–test). Known drops: *A. alba* 8867928 (identity); stork 5, *G. himalayensis* 2, *C. aura* 1 (val–test). The evaluator recomputes them, checks the scored counts against the rotation's sec 4, then checks each bird's test_scored counts per horizon (individuals with a window, individuals at the floor, windows) against `horizon_counts.csv`.

From those counts (test_scored, at the floor):

| h | birds scored | unscored (individuals at the floor, of 10) | test_scored windows |
|---:|---:|---|---:|
| 1 | 28 | *M. americana* 4, *P. apivorus* 3 | 1,483,827 |
| 6 | 27 | *M. milvus* 1, *M. americana* 0, *P. apivorus* 1 | 1,308,984 |
| 24 | 28 | *M. americana* 4, *P. apivorus* 4 | 1,498,595 |

Common set: 27 birds (all but *M. milvus*, *M. americana*, *P. apivorus*).

## 5. Readings (descriptive, no pass/fail)

| reading | per | what is reported |
|---|---|---|
| LSTM-Bb − LSTM-S, XGB-Bb − XGB-S | bird × horizon, scored cells | the gap, with metre and skill intervals |
| gap above +0.05 / within / below −0.05 | horizon, family | counts over the scored birds and over the common set |
| Bb, LSTM-Bb, XGB-Bb, LSTM-S, XGB-S against P | bird × horizon | skill, with intervals for the pooled arms; medians over scored birds and over the common set |
| share of test windows in travel | bird × horizon | the share, at the floor |
| ablation: Bb, LSTM-Bb, XGB-Bb against C | arm × horizon × C × stratum | skill at C, skill at C = 12 on the same windows, and the difference; per bird, and medians over scored birds and over the common set |
| Spearman: skill against Rbar_corr | horizon × arm, scored birds | ρ for LSTM-S, XGB-S, LSTM-Bb and XGB-Bb at each horizon, as the rotation (sec 22.3); Rbar_corr from `results/randomness/species_randomness.csv` (sec 21.4, 6 h training birds; stork as `white_stork`), joined after the seal by `bird_1h_sealed_eval.py --join` |
| persistence across C | bird × C | a check, identical bit for bit, not a reading |

No threshold or p value decides any of them.

## 6. Pre-run predictions

Again we have $\pm$ 0.05 as no difference; "most" means more than half the scored birds at that horizon. This table below is mine. 

| reading | prediction | basis |
|---|---|---|
| LSTM-Bb and XGB-Bb vs P, 1 h and 6 h | Above +0.05 for most birds | sec 22: 21 and 23 of 30 birds were above +0.05 at 6 h using 72 h of context |
| Bb vs P, 1 h and 6 h | Within $\pm$ 0.05 for most birds | sec 22: 18 of 30 birds were within this range |
| Pooled models vs P, 24 h | Lower median skill at 24 h than 6 h for Bb, LSTM-Bb and XGB-Bb | Many birds have lower error at 24 h than 6 h for persistence, possibly because they return to the same roost |
| LSTM-S and XGB-S vs P | Above +0.05 for most birds at 1 h and 6 h | 23 of 30 birds for LSTM-S and 20 for XGB-S at 6 h |
| Pooled vs specialist models | Within $pm$ 0.05 for most birds at each horizon, with more negative than positive gaps | sec 22: 20 birds were within $\pm$ 0.05; 8 / 7 below −0.05 and 2 / 3 above +0.05 |
| Skill vs Rbar_corr | Positive correlation for all 4 models at 1 h and 6 h, but weaker at 24 h | scorrelations of +0.31 to +0.35 at 6 h |
| Context ablation, 1 h | Median C − C12 stays within $\pm$ 0.05 across models and context lengths | The last few steps should be enough to predict the next hour |
| Context ablation, 24 h, full | C − C12 above 0 for at least two of three models at C = 48, 72 and 96; within $\pm$ 0.05 at C = 24 | Longer histories include the bird's position a day earlier, which may help capture daily returns |

## 7. As trained

To fill from `bird_1h_sealed_eval.py --diagnostic` after the runs.

## 8. Limitations

- The 6 h horizon here is a six step forecast from 12 h of context, unlike sec 22's one step forecast from 72 h. The results are not directly comparable.
- Day only tracking reduces the available 6 h windows (*M. americana*: 5,226 training windows at 6 h versus 119,864 at 1 h). *M. milvus*, *M. americana* and *P. apivorus* are unscored at some horizons (sec 4).
- The 30 m scale floor has a greater effect at 1 h, affecting 25 of 30 birds, compared with 3 at 6 h and 5 at 24 h.
- At 24 h, many birds have smaller boundary displacements than at 6 h (*C. corone*: 20 -> 9 m; *H. ostralegus*: 140 -> 38 m; *H. caspia*: 445 -> 106 m, before the floor). This could be bc daily returns to roosts or colonies making persistence harder to beat.
- Pooled models use one `k_h` per pool rather than a bird specific value. At 24 h, median `r` varies widely, from 1.05 (*M. milvus*) to 119.5 (*G. himalayensis*).
- Scaling uses the 12 h core, so earlier movements can be much larger when the core is relatively stationary. For albatrosses, typical extended tokens reach 37–60× the core's p99. The ±1,000 clip affects 0.024% of core tokens and 0.11% of observed extended tokens at C = 96.
- LSTM-S selects its best validation checkpoint, while pooled models use their final checkpoint. This may favour the specialist by an unknown amount. Neither XGBoost model uses checkpoint selection.
- Some specialists have limited training data. At 1 h, five birds have fewer than 18,000 training windows; at 6 h, *M. americana* has 5,226 and *P. occidentalis* has 5,379.


## 9. Note

No 1 h run of the plan existed when this file was first written (and I guess committed); the only 1 h training was the Mac toy runs and the fold 6 calibration (sec 1), on training windows only, with no record, validation or test. Only `bird_1h_sealed_eval.py` scores test partitions. 
