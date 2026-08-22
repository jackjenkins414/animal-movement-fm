# Preregistration: Tier 2 configuration budget and selection rule

**Date: 20-08-2026.** We made this file required by `preregistration_tier2.md` 2, which fixes that this file is committed **before the first tuning run**. We records four things: the configuration budget, the selection metric, the rule for combining pseudo target results, and the tie break, plus the constants that are deliberately *not* searched. It is not the pre-test freeze; that is `preregistration_tier2_freeze.md`, written after selection and before any sealed evaluation.

**Status when made:** no Tier 2 model has been trained, no configuration has been evaluated, and no held-out species partition has been accessed.

## 1. The inner loop

Each source species is held out in turn as a **pseudo target**, giving three folds. A configuration is trained on the remaining two source species under the registered sampling rule, using the sub-pool draw probabilities fixed in A9, and scored on the pseudo target species' **validation** individuals. Test partitions are not read, on any species, at any point in this document.

No held-out species data (training, validation or test) enters the inner loop. The held-out species are absent from selection entirely, which is what makes Z zero-shot.

## 2. The budget

**Twelve configurations**, the full grid of

- `d_model` $\in$ {64, 128, 256}
- layers $\in$ {2, 4}
- learning rate $\in$ {3e-4, 1e-3}

**Grid order**, which is what tie break 4 in 4 indexes: `d_model` outermost ascending, then layers ascending, then learning rate ascending. Indices are **0 based**, matching the PBS array and `experiments/training/tier2_inner_loop.py`: index 0 is (64, 2, 3e-4) and index 11 is (256, 4, 1e-3).

**Why the budget spends on capacity rather than on learning rate.** Recall in our Tier 1 that we resolved convergence at 3e-4 under warmup, cosine decay and gradient clipping, which makes 1e-4 the least informative point on that axis. Capacity is where the budget is uninformed: the training partitions hold 1,798,786 distinct 6h observations, `d_model = 256 / L = 4` already sits at 1.25 parameters per observation, and Tier 1 tied the LSTM at `d_model = 64` while being optimisation limited rather than data starved. The grid therefore spans 29x in parameter count instead of 7.5×, at twelve configurations either way and at lower total SU. 

Each configuration is trained once per fold at **one seed**, giving 36 training runs. The selected configuration is then retrained with the **five seeds** of the Tier 1 multi-seed sweep (`methods.md`, 22-07-2026) and frozen for Z, Za, S and F. The grid is exhaustive, so there is no search order to report and no early stopping of the search itself.

If a configuration fails to train (divergence, OOM), that is recorded with its cause and the grid is **not** backfilled with a replacement. Twelve is the budget whether or not all twelve produce a number.

## 3. Selection metric

**Persistence skill on travel-regime p90**, computed per fold:

    skill = 1 − p90_travel(model) / p90_travel(persistence)

Travel regime is the stratum defined by the A4 boundary; p90 is individual weighted per A5. Skill rather than raw metres because p90 in metres is not comparable across species whose median steps differ by 35x, and persistence is a comparator already registered in 3. It is the same statistic the success bands are defined on, measured on a different population; source pseudo targets, never the held-out species.

## 4. Combining folds, and the tie break

The **primary criterion is the mean skill across the three folds**. A mean is used rather than the worst fold because at n = 3 the minimum is one number from one species and is dominated by fold noise.

Applied in order:

1. Highest mean persistence skill across the three folds.
2. **Highest minimum skill across folds.** This aligns the tie break with the STRONG band's requirement that *every* held-out species beat persistence, so where two configurations are equally good on average the more uniform one wins.
3. Fewer parameters.
4. Lowest index in the grid order given in §2, which is deterministic and fixes any remaining tie without a judgement call.

## 5. Constants excluded from the budget

These are fixed by design or by amendment and are **never searched**, except for optimiser steps under the separate five-budget procedure registered in amendment T2. That procedure fixes one common budget before any valid full-grid run; no grid configuration receives its own training-length choice.

| constant | value | fixed by |
| --- | --- | --- |
| cadence | 6h | `preregistration_tier2.md` 2 |
| context length C | 12 | A6 |
| species sampling exponent $\alpha$ | 0.5 | A8, weights revised by A9 |
| scale floor $\sigma$ | 30 m, applied in quadrature: `scale = sqrt(median_step² + σ²)` | this document 6 |
| loss space | normalised | this document, 6 |
| loss form | radial pseudo-Huber, $\delta$ = 1.0 in normalised units | this document, 6, and A12 |
| input representation | the 12 channels listed in 6 | this document, 6 |
| optimiser | Adam | this document, 6 |
| batch size | 256 | this document, 6 |
| learning-rate schedule | linear warmup over the first 10% of steps, then cosine decay to zero | this document, 6 |
| gradient clipping | global norm 1.0 | this document, 6 |
| dropout | 0.1 | this document, 6 |
| attention heads | 4 | this document, 6 |
| feedforward width | 2 x `d_model` | this document, 6 |
| masking ratio (auxiliary objective) | 0.15 | this document, 6 |
| reconstruction loss weight | 0.5 | this document, 6 |
| optimiser steps | selected once from {5k, 10k, 20k, 30k, 40k} by amendment T2, then common to every model | this document, 6, and T2 |

## 6. Representation and loss, with the measurements behind them

**The registered channel list.** Twelve channels per observed token, in this order:

| # | channel | what it is |
|---:|---|---|
| 1–2 | `d_east_scaled`, `d_north_scaled` | ENU displacement in a tangent plane anchored at the last context fix, divided by the window scale |
| 3–4 | `log_dt`, `dt_bins` | real elapsed time to the fix this step lands on, logged and in units of the 6h bin |
| 5 | `log_scale` | the removed window scale, handed back so magnitude is not destroyed |
| 6–7 | `heading_cos`, `heading_sin` | direction, gated to zero below 10 m where a heading is observation noise |
| 8 | `log_speed` | displacement over elapsed time |
| 9–10 | `solar_sin`, `solar_cos` | local solar time of day of the fix the step lands on |
| 11–12 | `doy_sin`, `doy_cos` | day of year, phase flipped below the equator by **the fix's own latitude** |

**Why these channels.** Time of day and season are in the list because Tier 1 measured them over five seeds; time of day widened the moving p99 gap by ~6 km, season helped the moving tail architecture neutrally at ≈−2.8 km (`methods.md`, 22-07-2026); not because the original literature derived list contained them. Neither obstructs the zero-shot claim: both are computable from a longitude and a timestamp, the same test the context derived conditioning passes.

**Hemisphere alignment is by the fix, never by the species**, because a white stork crosses the equator inside a single track. The flip is a rotation by π, a sign change on both day of year components, so hemisphere costs one bit where feeding latitude would reintroduce the absolute position this representation excludes.

**There is no gap channel.** Under our A2 a Tier 2 window cannot contain a missing bin, so the old `dt_bins > 1.5` flag was structurally zero; and once real elapsed times reached the encoder it would have fired on genuinely adjacent bins instead, a lying channel in place of a dead one. Dropped. 7's conditioning vector loses its gap fraction component with it, four dimensions to three.

**Scale floor.** The per window normaliser is the median context step magnitude, floored **in quadrature** so the transform is smooth. $\sigma$ = 30 m is a **numerical floor against division by near zero, not an error model**, consistent with the 11-08-2026 declared limitation that Argos observation error is not modelled. One universal $\sigma$ rather than a per class value; the floor is a GPS problem, *C. ciconia* has a p0.1 window scale of **0.0 m**, with 13.8% of windows at or below 30 m, while both *Mirounga* species sit above 590 m at p0.1.

**Loss space and form.** Inputs and target are scaled by the floored statistic and the loss is taken in that space; predictions are multiplied back before any metric, so 3s geodesic metres are unaffected. Plain MSE fails in both spaces — effective share of the loss, draw share × mean aggregate over training windows:

| | stork | *Cervus* | SOES |
| --- | ---: | ---: | ---: |
| draw share | 47.9% | 39.8% | 12.3% |
| metric-space MSE | 77.0% | **0.11%** | 22.9% |
| normalised-space MSE | **99.85%** | 0.12% | 0.03% |

Metric space gives the terrestrial source 0.11% of the gradient at 39.8% of draws; normalised space hands stork 99.85%. Neither is the space's fault; the mean of squares is a tail statistic and these tails differ by four orders of magnitude, and the robust form is what makes normalised space usable. Normalisation works where intended: median normalised error **0.797 / 0.982 / 0.996** across three taxa whose median steps span 35x, failing only in the tail (p99 137.2 / 9.1 / 5.9) where stork's figure is the migratory launch Tier 1 established is irreducible. 

**The training recipe is Tier 1's**, and that is the point: Adam, warmup then cosine, clip 1.0, batch 256, warmup 0.1, dropout 0.1, four heads, all as `experiments/training/train_multiseed_gadi.py`. 

**Training length is in optimiser steps, not epochs**, since draws are with replacement and there is no epoch to count. **One step budget applies to all three folds** even though their pools differ (609,655 / 859,671 / 1,363,174 windows); a smaller pool is resampled more often, not trained on for less time. Amendment T2 fixes that number on one configuration across all three pseudo target folds before the first valid full FM grid run. No grid configuration may be given more or fewer steps than another.

**The masked reconstruction objective.** A random **15%** of context tokens are masked, at least one per window, and the model reconstructs their scaled displacement at weight **0.5** alongside the forecast term. A masked token's projected embedding is replaced by a learned mask embedding (A12); elapsed time, window scale and the clock pairs stay visible, since those are known for a hidden step exactly as they are for the target step at inference. Both constants are conventional rather than tuned; 0.15 is the standard masking rate, 0.5 is the weight Tier 1 gave its auxiliary state head and **neither is in the configuration budget nor searched**. It trains **jointly with the forecast term**, so it is an auxiliary objective, not a pretraining stage; it is what §1 of `preregistration_tier2.md` means by the pretraining objective in the FM package, present in Z and absent in B. Two stage and causal next step variants are maybe something we should look at next (added in notes.md). 

*Recorded 20-08-2026, before the first tuning run.*

**T2; Probe divergence, 21-08-2026.** Our probe (grid index 6, seed 0) trained cleanly on the Cervus + SOES fold and returned NaN at every budget on both folds containing white stork. I assumed the loss was at fault but couldn't diagnose the bug. So I gave GenAI `train/loss.py`, `train/loop.py`, `models/fm.py` and `data/enu`.py with the probe logs. Its argument was that the loss is the one place the fault cannot be because the radial pseudo-Huber registered above is asymptotically linear, so its gradient with respect to the error is bounded by 1 per token. So however large that error grows, and pointed instead at the scale floor of this section, which bounds the divisor at $\sigma$ = 30m but not the ratio; 13.8% of stork windows sit at that floor, so a roosting window followed by a migratory launch encodes as a normalised displacement above 15,000 where Cervus and SOES stay below 610. It suggested gradient clipping to then convert a single overflowed element into a permanently NaN model, since an infinite total_norm drives the clip coefficient to zero. Added this just to reference GenAI, since this was a great catch from it. 

**T3; Nonfinite gradient steps are skipped, 22-08-2026.** Training on the two folds containing white stork produced NaN losses. As we said above, the cause is not the loss. A training only diagnostic with autograd anomaly detection found what it was; PyTorch's fused attention kernel returns NaN from its backward pass when a window's normalised displacement is extreme, which the $\sigma$ = 30m floor of this section gives us (where 13.8% of stork windows are on that floor), and one failing batch carried a value of 7,860. Once that NaN reaches the weights the run never recovers. From now on, when the gradient norm before clipping is not finite the optimiser step is not applied. The learning rate schedule will still advance so the registered cosine curve is unchanged, and no constant in 5 moves (clipping norm is still 1.0). Measured on fold 1 at 5,000 steps, this withholds 25 steps (0.50%) with the reconstruction head on and 4 (0.08%) without. The count of withheld steps is recorded for every run and reported in the freeze. Recorded before any scored run.
## 7. The pooled-source baseline B

Fixed here because 2 requires B's specification in this record. **B is the Tier 1 Transformer at C = 12**, trained supervised one step on the pooled source windows under the same sampling rule, representation, floor and loss as Z, with the same five seeds. It differs from Z in exactly the FM specific package; no auxiliary reconstruction objective, no conditioning path, no FM- pecific architecture. B gets no hyperparameter search of its own and inherits the configuration selected in 4, so a Z vs B gap is attributable to the package rather than to tuning effort.

## 8. What this document does not permit

No unregistered configuration enters the sealed evaluation. The inner loop may not alter the sampling rule, the split manifests, the regime boundary, the evaluation window floor, the context length, the cadence, or anything at all on the three held-out species. Any change to this document after the first tuning run is a dated amendment recorded here, with its status relative to test access, under the amendment policy of `preregistration_tier2.md`.

## 9. Amendments after the first probe on 21-08-2026

**T1; 21-08-2026 — invalid probe.** Gadi job `176913681` ran grid index 6, seed 0, fold 0 without the registered reconstruction head. Its 5k / 10k / 20k / 30k / 40k stork skills (-0.1748 / -0.2305 / -0.3353 / -0.3813 / -0.4035) are excluded. No source test partition or held-out species was scored, and no registered grid run had begun.

**Correction.** The driver now requires the complete FM, loads only the selected source partition, treats the five budgets as independent schedules, and keeps corrected artifacts separate from the invalid run. 

**T2; 21-08-2026 — corrected probe.** After seeing T1 but before any corrected score, fix training length using grid index 6 and seed 0 over all three pseudo target folds. In each fold independently train 5k, 10k, 20k, 30k and 40k schedules, opening no test partition or held-out-species file.

**Decision.** Choose the budget with highest mean travel-p90 persistence skill across folds; exact ties go to highest minimum skill, then fewer steps. The winner is fixed for all 36 grid runs and subsequent registered models; `--select-probe` applies and records this rule mechanically.
