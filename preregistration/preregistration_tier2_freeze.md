# Preregistration: Tier 2 pre-test freeze

**Date: 26-08-2026.** Phase (B) in the sense of Hofman et al. (2023), required by our `preregistration_tier2.md` we outlined in sec 2. We record here the finalised model and fixes the three things sec 2 and  3 deferred to it. Not the analysis; 

**Status as of now.** The 36 run grid is complete, the configuration is selected by the registered rule, and Z, B and S are trained. **No held-out species test partition has been opened, on any species, by any process.**

## 1. Selected configuration

| | |
|---|---|
| configurations evaluated | 12 of a declared budget of 12; none failed to train |
| runs | 12 x 3 pseudo-target folds x 1 seed = 36, all complete |
| selected | **grid index 11** — `d_model` 256, 4 layers, LR 1e-3 |
| fold skills | stork −0.0054, *C. elaphus* −0.0003, *M. leonina* −0.0206 |
| mean fold skill | **−0.0088** (minimum −0.0206) |
| decided by | `tier2_inner_loop.py --select`, recorded in `results/tier2_inner_loop/tier2_selection.json` |
| optimiser steps | 20,000, fixed by amendment T2 before the grid |
| parameters | Z 2,245,636 · S 2,245,636 · B 2,112,258 |

**Every configuration scored below zero**, from −0.0088 to −0.2533. None beat persistence on any fold. Our current registered rule selects the best available configuration, which it did; it does not require that the winner be good.

**The selection is a boundary selection at the corner of the grid, and it is fragile.** Grid 11 is the maximum of every axis searched, so "best available" and "best" are not distinguishable. There is no capacity trend to extrapolate; marginal means by width are −0.1635 / −0.1687 / −0.1413 at 64 / 128 / 256, and excluding the winning cell `d_model` 256 is the *worst* width in the grid at −0.1855. Raising the LR hurt us in four of the six architecture pairs and only helped in one, by +0.1618; two
thirds of the entire spread of the table, at one cell, on one seed. Tier 1 found 1e-3 at batch 256 converges worse than 3e-4. This is consistent with a favourable draw rather than a resolved optimum. This rule was registered before the numbers existed and is applied unchanged; the five seed Z runs give the first seed variance estimate at this cell, retrospectively.

Selection applied `preregistration_tier2_tuning.md` sec 4 mechanically; highest mean skill across folds, then highest minimum, then fewer parameters, then lowest grid index.

**Source pool as trained.** *C. ciconia* 806,595 windows / *C. elaphus* 556,579 / *M. leonina* 53,076, drawn at 47.911% / 39.799% / 12.290% (A8, revised by A9). All three counts are reproduced from our current pipeline.

## 2. Seeds and the checkpoint-selection rule

Training seeds **0, 1, 2, 3, 4** for Z, B, S and Za. The data split seed is 42 and is not a training
seed.

**This fixes what `preregistration_tier2.md` sec 3 deferred.** sec 3 grants S, and F on the same terms, the privilege of checkpoint selection on the held-out species' validation individuals:

- **S and F.** Train the full budget. Every **1,000** optimiser steps, score on that species' **validation** individuals using individual weighted travel-regime p90 at floor F = 25 — the same statistic our sec 4 judges on, and report the argmin checkpoint.
- **Z, B and Za.** The final checkpoint at 20,000 steps. No validation privilege, which is what makes S a somewhat oracle style benchmark rather than a like for like model.

**No patience constant.** The budget is fixed either way, so halting early would buy compute at the price of an unprincipled constant. The full selection history is written into each S result file, so the privilege is auditable rather than asserted.

## 3. The fine tuning budget for F

Also deferred here by sec 3.

| | |
|---|---|
| length | 2,000 optimiser steps, 10% of pretraining |
| schedule | linear warmup over the first 10%, then cosine decay — the registered shape |
| peak learning rate | 10% of the selected rate = **1e-4** |
| all other constants | unchanged from §1 |
| checkpoint selection | as S, above |

The same budget applies at every k, so the only thing varying along the F curve is how much target data was available. A lower peak rate is conventional and matters most at k = 1%, where the subset is five to six individuals. **A judgement call, made before any fine tuning run and any sealed number, not derived from a search**; F gets no budget search, for the same reason B gets no hyperparameter search.

## 4. Which p99

In A5 we moved p99 to species level but contrasted individual weighted against pooled **only for the median and p90**. Sec 4's STRONG band turns on travel-regime p99, so the ambiguity sits on a headline criterion.

**The §4 p99 criterion is read on the individual weighted species distribution**, for consistency with A5's weighting of the other two quantiles and for A5's own reason; a long deployment should not dominate a species' reported error. The pooled figure is reported alongside it (`p99_m_pooled`), so we make the choice visible rather than silent.

## 5. The comparator set, and what is deferred

| symbol | runs | status |
|---|---|---|
| P persistence · V constant velocity | 3 species each | run |
| Z zero-shot FM · B pooled-source baseline | 5 seeds each | run |
| S specialist | 3 species × 5 seeds | run |
| **Za domain ablation** | 15 | **deferred** |
| **F fine-tuned** | 21 | **deferred** |

Za and F are owed and are deferred past the first sealed evaluation because we just want results before meeting with John on 30-08-26, and because both are secondary in the preregistration's own terms; Za is secondary prediction 5, and sec 3 states F does not move the goalposts, which sec 4 defines on Z alone. This is a **reporting completeness gap, not a validity one**; no success band reads either arm. Both run after the meeting at the configuration and seeds frozen above. Nothing about the deferral is contingent on the sealed result; the arms were deferred before it existed.

## 6. What Z looks like before the seal, and what that predicts

Measured on source species **validation** individuals at the selected cell (`results/tier2_diagnostics/inner_loop_metrics_g11.md`; no held-out species, no test partition).

**Z has converged onto persistence.** Travel-regime p90 skill against P is −0.0054 / −0.0003 / −0.0206 on stork, *C. elaphus* and *M. leonina*; on red deer, one metre in 1,564. As a position on the line from persistence to constant velocity, Z sits 0.5% / 0.1% / 4.7% of the way from P to V, while V itself scores −1.10 / −0.49 / −0.43. The model is far better than naive extrapolation and marginally worse than predicting no movement, on every fold and every stratum.

**Where it loses is the low-displacement stratum**, at 2.3x / 3.2x / 5.0x persistence's p90. Stork's all windows skill (−0.0430) against its travel regime skill (−0.0054) is that effect, since 31.4% of its windows sit in that stratum. Our registered $\sigma$ = 30m floor bounds the divisor but not the ratio, and a near stationary floored window encodes as a large normalised displacement; that remains the candidate mechanism, though it is weaker than the pre-T4 measurement suggested and the penalty ordering no longer tracks floor exposure.

**Only one fold is well powered**: 105 validation individuals and 146,597 windows for stork, against 10 individuals each for *C. elaphus* and *M. leonina*, the latter with 17 low displacement windows.

**The prediction, recorded before the seal is opened.** Z is expected to land in WEAK/NULL on the held-out species. If it does, this is the anticipated mechanism, not a post hoc explanation. Candidate remedies; winsorising normalised inputs, raising $\sigma$ or flooring at a percentile, a pre-LN trunk; each change a registered constant and so are unavailable inside Tier 2; they are Tier E things we will explore...

## 7. Differences from `preregistration_tier2.md`

1. **S and F checkpoint selection** (sec 2) and **F's fine-tuning budget** (sec 3), deferred here by sec 3. New, not changes.
2. **The p99 form** (sec 4); a disambiguation of A5, not a revision. Both forms are reported.
3. **Za and F deferred past the first sealed evaluation** (sec 5); a change to *when* two registeredcomparators run, not to whether, and no success band reads either.
4. **Mean great circle error additionally reported**, all three strata, individual weighted and pooled, with a 95% bootstrap interval on the all windows mean. Sec 3 registers median, p90 and p99 and calls the mean uninformative under this skew; that judgement is unchanged and **no sec 4 band reads the mean**. Reported for comparability with the TFT movement forecasting literature. Decided before any sealed partition was opened.
6. **Single seed grid resolution.** Run to run variation at one seed is ≈ ±0.05 in skill, measured by re running the grid under T4. The spread from rank 2 to rank 12 is 0.165, so **the ranking below first place is unresolved and is not reported as a ranking**. Grid 11's margin to second place (0.08) is the only separation exceeding that scale.

Amendments in force: `preregistration_tier2_tuning.md` T1–T4; `preregistration_tier2.md` A1–A12 stand as written. T4 excludes the pre-amendment grid and final-training runs, retained under `results/*_T3_excluded/`.

## 8. Attestation

26-08-2026, before the first sealed evaluation was run:

After the registered ingest, QC, resampling, windowing and split freezing, I had not accessed any held-out test targets, predictions, plots, metrics or results for *Larus fuscus*, *Canis lupus* or *Mirounga angustirostris*. The configuration in sec 1 was selected using source species pseudo-target validation data only. S and F use held-out training and validation data only, while Z, B and Za have seen no held-out species data.

