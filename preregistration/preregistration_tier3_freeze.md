# Preregistration: Tier 3 pre test freeze

**Date: 21-09-2026.** This aims to serve as phase (B) in the sense of Hofman et al. (2023), the Tier 3 counterpart of`preregistration_tier2_freeze.md`. It records our models as trained and fixes what `preregistration_tier3.md` deferred to it. Not the analysis.

**Status when this was written.** All 158 registered Tier 3 runs are complete. Every run applied its full 20,000 updates with no withheld and no non-finite steps, so the T3/T4 clocks of `preregistration_tier2_freeze.md` are clean for this tier. **No held-out species test partition has been opened by any Tier 3 process.** The same three partitions were opened for Tier 2 on 26-08-2026; sec 8 states what that means for Tier 3.

## 1. Configuration and pool as trained

| | |
|---|---|
| configuration | **grid index 11** — `d_model` 256, 4 layers, LR 1e-3, we carry this fromTier 2 unchanged |
| retuning | none. `preregistration_tier3.md` sec 3: Tier 3 varies the source pool and nothing else |
| optimiser steps | 20,000 applied updates at batch 256, the Tier 2 budget |
| seeds | 0, 1, 2, 3, 4; split seed 42, which is not a training seed |
| registered runs | 158 (`tier3_train.py --list`) |
| parameters | Z9; Za9; Zd3; Zd1; Sk; F9; F3c 2,245,636 (Tier 2s Z exactly); B9 2,112,258; B9dom 2,248,242; LSTM rungs 34,242 |
| service units | 1,876.8 SU over the Tier 3 arrays, about 12.5 SU per neural run |

**The nine sources as trained** (`registered/tier3_source_pool.json`, which we write once by `source_pool_weights.py --tier3` and have verified on every later run):

| domain | source | training windows | A8 draw probability |
|---|---|---:|---:|
| flying | *C. ciconia* | 806,595 | 32.601% |
| flying | *B. canadensis* | 99,724 | 11.463% |
| flying | *T. antarctica* | 2,564 | 1.838% |
| terrestrial | *C. elaphus* | 556,579 | 27.081% |
| terrestrial | *B. bison* | 37,729 | 7.051% |
| terrestrial | *C. latrans* | 37,253 | 7.006% |
| marine | *M. leonina* | 53,076 | 8.363% |
| marine | *A. gazella* | 5,414 | 2.671% |
| marine | *E. chrysolophus* | 2,818 | 1.927% |

These are the corrected counts of 20-09-2026. `preregistration_tier3.md` sec 1 records the defect that produced the superseded ones, and the regenerated `registered/tier3_source_pool.json` carries those superseded values under `regenerated.supersedes`. The rule selected the same nine species on the corrected counts; only the probabilities
moved.

**Sub pools, renormalised under A8 over their members** (same file, same rule):

| sub-pool | members and probabilities |
|---|---|
| domain_flying (Zd3, gull) | stork 71.023%, *B. canadensis* 24.973%, *T. antarctica* 4.004% |
| domain_terrestrial (Zd3, wolf) | *C. elaphus* 65.830%, *B. bison* 17.139%, *C. latrans* 17.031% |
| domain_marine (Zd3, NES) | *M. leonina* 64.524%, *A. gazella* 20.608%, *E. chrysolophus* 14.868% |
| without_flying (Za9, gull) | *C. elaphus* 50.059%, *M. leonina* 15.458%, *B. bison* 13.033%, *C. latrans* 12.951%, *A. gazella* 4.937%, *E. chrysolophus* 3.562% |
| without_terrestrial (Za9, wolf) | stork 55.385%, *B. canadensis* 19.474%, *M. leonina* 14.207%, *A. gazella* 4.538%, *E. chrysolophus* 3.274%, *T. antarctica* 3.123% |
| without_marine (Za9, NES) | stork 37.455%, *C. elaphus* 31.113%, *B. canadensis* 13.170%, *B. bison* 8.101%, *C. latrans* 8.049%, *T. antarctica* 2.112% |

Zd1 draws one source with weight 1.0: stork for the gull, *C. elaphus* for the wolf, *M. leonina* for NES.

**The three reserved species** (*A. albifrons*, *O. dalli*, *P. adeliae*) entered no Tier 3 pool. `tier3_train.py` refuses a pool containing a held-out or reserved species, so this is enforced rather than asserted.

## 2. Seeds and checkpoint selection

Unchanged from `preregistration_tier2_freeze.md` sec 2 and extended to the arms Tier 3 adds:

- Validation selection, every 1,000 steps, on the held-out species' own validation individuals, argmin individual weighted travel-regime p90 at F = 25: S (carried), F9, F3c, Sk, LSTM-S.
- **Final checkpoint at 20,000 steps, no validation privilege:** Z9, B9, Za9, Zd3, Zd1, B9dom, LSTM-B3, LSTM-B9.
- **No checkpoint selection at all:** the XGBoost rungs, which fit once, and the training-free rungs.

The full selection history is written into each selecting run's result file.

## 3. The step budget, and the resampling rate it implies

20,000 applied updates for every neural arm, pooled and specialist, as §3 registers. Our correction note of 16-09 in `preregistration_tier3.md` sec 1 records what that means for the nine source pool: a 2,000-window source is revisited about 47 times in a run, and the three smallest selected sources 32–38 times, against about 2 for stork. Note that the floor is not changed and the budget is not changed.

**The 40,000 step check (Tier E, `Z9long`), run 21-09-2026.** We found that doubling the budget does not reduce the training loss. Averaging the last 10% of each run's logged loss, the median across five seeds is **6.009 at 40,000 steps against 5.648 at 20,000**, and the seed spread at each budget (4.36-7.18 and 5.09-8.23) is several times the difference
between them. As such there no evidence that the registered budget leaves the nine source pool underfitted, and the heavy resampling of the small sources noted in the 16-09 correction does not show up as a budget limitation. Note that this is exploratory, validation free and no Tier 3 number comes from it; it is recorded here because `preregistration_tier3.md` sec 1 carried the question to this freeze. The comparison is of training loss on a heterogeneous pool, which is noisy by construction, and is not a claim about skill.

## 4. The F budget change, and the two F objects

`preregistration_tier3.md` sec 3 registers F9, F3c and Sk at S's budget; 20,000 steps at peak LR 1e-3, selection every 1,000 steps. This replaces Tier 2's 2,000 steps at 1e-4, for Tier 3 only. **Tier 2's registered F is not re run, revised or replaced.**

The evidence behind that change was Tier E, validation only, and was seen before the change was registered; Flong and Spre, the F budget confound taken apart. F still initialises from seed 0 alone, so F type arms carry draw spread and no seed spread.

## 5. The run plan, and what is deferred or missing

| block | plan indices | runs | status |
|---|---|---:|---|
| Z9 | 0–4 | 5 | complete |
| B9 | 5–9 | 5 | complete |
| Za9 | 10–24 | 15 | complete |
| Zd3 | 25–39 | 15 | complete |
| Zd1 | 40–54 | 15 | complete |
| B9dom | 55–59 | 5 | complete |
| LSTM-S | 60–74 | 15 | complete |
| LSTM-B3 · LSTM-B9 | 75–84 | 10 | complete |
| Sk | 85–102 | 18 | complete |
| F3c | 103–123 | 21 | complete |
| F9 | 124–144 | 21 | complete |
| XGB-S | 145–147 | 3 | complete |
| XGB-B3 · XGB-B9 | 148–157 | 10 | complete |

Within a per species block the order is gull, wolf, NES; within an F or Sk species block, k = 1% draws 0–2, then k = 10% draws 0–2, then k = 100% where the arm has one.

No arm is incomplete, so this tier carries no reporting completeness gap.

**Two arms are not trained by Tier 3 and are scored on test as registered rungs**, both declared in `preregistration_tier3.md` sec 3 and 3.1 as having been chosen after their Tier E validation results were seen:

- **SB**, the single species plain trunk, trained under Tier E on each held-out species with S's recipe. Its configuration was fixed before any of its runs (`tierE_exploratory_ablations.md`, 17-09).
- **E1embed under the matching domain row**, Tier 2's three row label arm read as a domain table, with no retraining.

## 6. The analysis decisions this freeze fixes

1. **Predictions 3 and 4 are reported per draw.** Each k = 1% (and k = 10%) comparison pairs the same draw index on both arms and gets its own paired bootstrap; the mean and range across the three draws are descriptive only. The draws are subsets of the same training individuals, so pooling them into one interval would count the same test individuals three times. We locked this in on 18-09-2026, before any Tier 3 run and before any test partition was opened.
2. **Success bands.** `preregistration_tier2.md` sec 4 unchanged, applied to Z9 with B9 and the carried S and P. The evaluator presents Z9 as Z and B9 as B to Tier 2's own band implementation and records the mapping in the result file.
3. **Tier 2's numbers are read, not recomputed.** Every Tier 2 comparator enters Tier 3's tables from `results/tier2_sealed/predictions/`, so the paired bootstrap pairs identical test windows and no sealed Tier 2 number can move. Each file is checked row for row against persistence before use.
4. **The noise scale.** +/- 0.05 in skill, from `preregistration_tier2_freeze.md` sec 7.6. A gap smaller than that is reported as unresolved whatever the point estimate rule says.
5. **ARIMA's fallback rate** is estimated per axis from the prediction files, by comparing ARIMA's prediction to persistence's, because the model keeps no tally of failed fits.

## 7. What Tier E evidence was in hand before this freeze

All validation only, all exploratory, none of it a Tier 2 or Tier 3 result, and all of it seen before the seal:

- **§12** Flong and Spre: the F budget confound, which motivated our sec 4 above.
- **§16** the domain row read zero-shot: only NES gains (+0.0148 against B's +0.0032), inside the noise scale. This motivated B9dom and the domain models Zd3 and Zd1 (Ana, 17-09).
- **§17** the single species architecture ablations (John, 09-09; run 17–20-09): the full FM package does not beat the plain trunk within a species, on any of the four species tested, including stork where data is not the limit; the conditioning path is negative in all four and resolved on NES.

**The prediction, recorded before the seal is opened.** 

We now make a preliminary hypothesis, which is that we expect Tier 3 to return a second null, and specifically:

1. **Z9 will not clear Tier 2's Z.** We expect the two to sit within the +/- 0.05 noise scale of each other on all three held-out species. Prediction 1's point estimate rule may be met on two of three by a small margin, and if it is we will report it as met and unresolved, not as a scaling effect.
2. **Z9 will not beat B9.** If anything we expect B9 to be the better of the two, as B was in Tier 2 on gull and wolf.
3. **Zd3 will be close to Z9**, so prediction 5's four way table will show pool composition mattering less than pool membership of the held-out species' own domain.

**The mechanism I am betting on, so that being wrong is informative.** Three things point the same way. First, the single species ablations; the conditioning path is negative within a species on all four speciestested and the full package never beats the plain trunk, including on stork where data is not the limit. Adding source species does not address a component that does not pay for itself with ample data. Second, the loss shares in sec 5.1 below: at the registered A8 weights, white stork contributes about 64% of the training loss from 33% of the draws, and the three marine sources contribute about 3% between them, so the nine source pool is in gradient terms close to a stork model and the marine held-out species is the least well served. Third, Tier 2's failure was concentrated in the low-displacement stratum, where Z over moves by an order of magnitude; nothing about a larger source pool addresses that, and the registered representation and scale floor are unchanged.

**What would falsify it.** Z9 clearing Z by more than 0.05 on two or more species would mean source pool scaling works despite the gradient imbalance, and we would report the mechanism above as wrong. Z9 beating B9 anywhere by more than the noise scale would mean the package does pay at nine sources even though it does not at one, which is the result we would most want to understand.

### 7.1 Per-species loss shares at the registered weights

Below is the median across the five seeds of each arm, averaging the last half of each run's logged steps. B9 tracks Z9 closely, so this is a property of the data rather than of the FM package.

| source | A8 draw probability | share of Z9's loss | share of B9's loss |
|---|---:|---:|---:|
| *C. ciconia* | 32.60% | 64.05% | 67.27% |
| *B. canadensis* | 11.46% | 19.40% | 15.11% |
| *C. elaphus* | 27.08% | 8.15% | 8.30% |
| *C. latrans* | 7.01% | 2.59% | 3.14% |
| *B. bison* | 7.05% | 1.96% | 2.12% |
| *M. leonina* | 8.36% | 1.96% | 2.01% |
| *T. antarctica* | 1.84% | 1.19% | 1.07% |
| *A. gazella* | 2.67% | 0.77% | 0.87% |
| *E. chrysolophus* | 1.93% | 0.41% | 0.41% |

The two flying sources together take **83% of Z9's loss from 44% of its draws**; the three marine sources take **3.1%** from 12.9%, and *C. elaphus* takes 8.2% from 27.1%.

A8 equalises how often a species is drawn, not how much it moves the weights. Because the loss is radial pseudo-Huber on scale normalised displacements, a species whose steps vary a lot relative to its own window scale contributes more gradient than one whose steps are uniform, whatever its draw probability. This is recorded here as a property of the models as trained, before any test partition was opened. It changes nothing registered; our A8 is frozen and reweighting by loss contribution would be a different study or can be a possible ablation. 

## 8. Attestation

21-09-2026, before the first Tier 3 sealed evaluation was run:

**The Tier 3 held-out partitions are not newly blind.** *Larus fuscus*, *Canis lupus* and *Mirounga angustirostris* had their test partitions opened for the Tier 2 sealed evaluation on 26-08-2026, and Tier 3 reads the same partitions again. What follows is therefore a statement about Tier 3's own conduct, not a claim of a blind partition.

Since the Tier 2 seal, no Tier 3 model, arm, pool or configuration has been chosen, changed or discarded on the basis of any held-out test result. The nine sources were selected by our same mechanical rule in `preregistration_tier3.md` sec 1 from training partition descriptors alone, with no model performance of any kind entering the rule, and no held-out
species' descriptor row or distance entering it. Every arm, budget, prediction and reading rule above was written down before the corresponding runs. The arms named in sec 5 as chosen after Tier E validation results are declared as such. Tier 3's training scripts refuse to open a test partition, and the sealed evaluator refuses to run until this file is committed.

