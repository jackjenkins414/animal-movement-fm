# Rotating held-out birds (exploratory)

**GenAI reference:** GenAI drafted tables from my outline, the text is almost entirely mine. 

Exploratory and never reported as a tiered result by technicalities. Written on 29-09-2026, after the bird-only study's seal had been opened, but before any rotation run. Committed before any run, then committed again with sec 7 filled before the sealed evaluation, which refuses to run while sec 6 or sec 7 holds placeholder text to stay consistent with our prerego rules. 

## 1. Question and status

In the bird study the dove's pooled LSTM-Bb and XGB-Bb, trained only on other birds, were level with its own specialists (+0.011 and +0.026). Is that general? Every flying bird is held out once, so each gets a zero-shot estimate and a specialist comparison.

| | |
|---|---|
| birds | 30: the registered bird study's 22 main pool birds, 5 congeners and 3 held out |
| folds | 10; fold 0 is the bird study's gull, goose and dove, carried as reported |
| new test partitions | 26 opened for the first time; stork's was read in pre-registration work |
| 26-source pool | 14 of the 27 new held-out birds were sources there; their test partitions were never read |
| reserved | *O. dalli* and *P. adeliae* are not in this set and stay unread |

## 2. Folds and pools

**Rule.** Genus blocks among the 27 birds outside fold 0. A block larger than 3 birds is its own fold (*Anser*). The other blocks, largest A8 mass first ($\Sigma \sqrt{n}$ over the training windows we registered in `birds_source_pool.json`, genus name breaking ties), each go to the fold with the least held-out mass that still has room (at most 3 birds; ties to the lower fold). **Pool** = every other bird minus the fold's genus mates. A8 at alpha 0.5, renormalised per fold, on windows counted after the exclusions below. Written once to `registered/birds_rotation_folds.json` by `source_pool_weights.py --birds-rotation`; checked on every load.

| fold | held out | pool birds | pool training windows |
|---:|---|---:|---:|
| 0 | *L. fuscus*, *A. albifrons*, *S. turtur* (carried) | 22 | 2,009,726 as trained |
| 1 | *A. anser*, *A. brachyrhynchus*, *A. caerulescens*, *A. fabalis* | 25 (no *A. albifrons*) | 2,908,230 |
| 2 | *C. ciconia*, *M. bassanus* | 28 | 2,516,447 |
| 3 | *A. alba*, *L. argentatus*, *T. antarctica* | 26 (no *L. fuscus*) | 2,356,136 |
| 4 | *B. canadensis*, *B. leucopsis*, *P. apivorus* | 27 | 3,120,879 |
| 5 | *C. aura*, *G. himalayensis*, *M. milvus* | 27 | 3,077,771 |
| 6 | *A. platyrhynchos*, *M. americana*, *P. occidentalis* | 27 | 3,090,634 |
| 7 | *H. ostralegus*, *P. roseus*, *T. chrysostoma* | 27 | 3,094,647 |
| 8 | *C. corone*, *D. exulans*, *P. leucorodia* | 27 | 3,097,673 |
| 9 | *F. naumanni*, *F. tinnunculus*, *H. caspia* | 27 | 3,156,752 |

**Training exclusions**. Rule as `preregistration_birds.md` sec 3; a training bird with $\geq$ 100 exact (timestamp, lon, lat) fixes shared with one val or test bird. Every training set of the species, pooled and specialist, every fold.

| bird | excluded | training windows |
|---|---:|---|
| *A. albifrons* | 6 (the registered six) | 79,873 -> 77,550 |
| *C. ciconia* | 44 of 630 | 806,595 -> 779,723 |
| *C. aura* | 18 of 121 | 210,939 -> 188,267 |
| *G. himalayensis* | 10 of 24 | 9,621 -> 6,782 |
| *M. americana* | 1 of 109 (cross-species: 945 fixes with *A. alba* test bird 8867928) | 137,954 -> 135,968 |

- **Fold 0 was trained before these exclusions.** Its main pool held the 44 stork, 18 *C. aura*, 10 *G. himalayensis* and the *M. americana* bird: 2,009,726 windows, against 1,955,357 without them. Each duplicates birds of its own species, none of those species is scored in fold 0, and the cross-species check finds nothing against fold 0's three birds, so fold 0 stands as sealed.
- **Goose 1056306620.** Exact matching finds 5 of the registered six. 1056306620 shares 114 timestamps with test goose 1056306624 at positions a median 3.8 m apart (max 46 m), and no exact fix. It stays excluded as registered. A within 50m rule finds all six plus 2304560716, but cannot separate duplicates from flock mates, so the exact rule stands.

## 3. Arms and runs

Recipes by reference to `preregistration_birds.md` §4: grid 11, 20,000 applied updates at batch 256. The pooled arms report their final checkpoint; LSTM-S selects on validation birds. XGB-Bb fits on 1,000,000 A8 draws; XGB-S on all of a bird's training windows.

| arm | per | seeds | plan indices | runs |
|---|---|---|---|---:|
| Bb | fold (1–9) | 0–2 | 0–26 | 27 |
| LSTM-Bb | fold | 0–2 | 27–53 | 27 |
| XGB-Bb | fold | 0 | 54–62 | 9 |
| LSTM-S | held-out bird (27) | 0–2 | 63–143 | 81 |
| XGB-S | held-out bird | 0 | 144–170 | 27 |
| Zb | fold | 0–2 | 171–197 | 27 |
| **total** | | | | **198** (162 neural, 36 XGBoost) |

| | |
|---|---|
| driver | `experiments/training/bird_rotation_train.py`, `.pbs`; records tagged `birds_rotation`, `exploratory: true` |
| hardware | gpuvolta, fp32, as the bird study; walltime 01:00:00 per subjob (the bird study's longest run was 11m11s) |
| compute | about 955 SU (198 × 4.82, the bird study's 458.0 SU over 95 runs); evaluation about 50–100 SU, uncalibrated |
| fold 0 | carried at 5 seeds: pooled arms from the bird study's main pool; LSTM-S and XGB-S from Tier 3 (gull) and the bird study (goose, dove). Reported at 5 seeds as sealed, and at seeds 0–2 beside it, descriptively |
| persistence | P computed per held-out bird; no other training-free rung |

## 4. Statistic and scoring

As `preregistration_birds.md` Sec 5; travel p90 skill against persistence, individual weighted at F = 25, median across seeds (XGBoost single seed), paired bootstrap of 2,000 resamples at seed 42, $\pm$ 0.05 read as no difference. Each comparison prints both the metre interval (as sec 20) and a skill unit interval from the same resamples. Boundaries are the A4 values pinned in the driver (0.1 x p50 on training birds).

**Scoring rules** (fixed here, applied to every arm of the bird; the diagnostic prints ids and counts only):

| rule | drops a test bird that shares $\geq$ 100 exact fixes with | known now |
|---|---|---|
| identity | any individual of another species | *A. alba* 8867928 |
| val–test | any validation bird of the same species | stork 5, *G. himalayensis* 2, *C. aura* 1 |

| bird | test birds | scored |
|---|---:|---:|
| *C. ciconia* | 105 | 100 |
| *G. himalayensis* | 10 | 8 |
| *C. aura* | 10 | 9 |
| *A. alba* | 10 | 9 |
| the other 26, fold 0 included | 10 each | 10 each |

Neither rule drops a fold 0 bird, so fold 0 stands as sealed. Every test bird has $\geq$ 25 windows under its split rule; the floor is applied as registered and counts are reported.

## 5. Readings (descriptive, no pass/fail)

| reading | n | what is reported |
|---|---:|---|
| LSTM-Bb − LSTM-S, XGB-Bb − XGB-S | 30 each | the gap, with metre and skill intervals |
| gap above +0.05 / within / below −0.05 | 30 and 27 | counts per family |
| Bb, LSTM-Bb, XGB-Bb, Zb against P | 30 | skill, with intervals |
| Zb − Bb | 30 | the gap, with intervals |
| Spearman: gap against the specialist's training windows | 30 per family | ρ; counts after exclusions (registered file; fold 0 as trained) |
| Spearman: skill against Rbar_corr | 30 | ρ for LSTM-S, XGB-S, LSTM-Bb and XGB-Bb; Rbar_corr from `results/randomness/species_randomness.csv` (stork as `white_stork`) |

The Spearman rows ask whether pooling pays where specialists are data-limited, and whether skill tracks turning concentration across 30 birds rather than 5 (sec 21). No threshold or p-value decides them.

## 6. Pre-run predictions

Filled by me. 

| reading | prediction | basis |
|---|---|---|
| LSTM-Bb − LSTM-S, XGB-Bb − XGB-S | below −0.05 on most of the 27; within $\pm$ 0.05 mainly where the specialist is data limited | fold 0: dove +0.0106 / +0.0256, gull −0.0495 / −0.0489 (on the edge), goose −0.0964 / −0.1074 |
| gap against training windows | negative p, weak | the dove (34,737 windows) was level, but the gull (597,407) sat closer than the goose (77,550) |
| skill against Rbar_corr | positive p for LSTM-S and XGB-S, weaker than at n = 5 | sec 21: +0.90 / +1.00 over five species |
| Zb − Bb | negative on most birds, mostly within $\pm$ 0.05 | negative on all three birds (sec 20.1) and within a species (sec 17) |
| pooled arms against P | Bb within $\pm$ 0.05 on most birds; LSTM-Bb and XGB-Bb above +0.05 where Rbar_corr is high | the dove (Rbar_corr 0.295) was the one bird where pooled arms cleared P |

## 7. As trained

WILL FILL

## 8. Limitations

- LSTM-S selects its checkpoint on validation birds; LSTM-Bb reports its final one. The LSTM gap does lean toward the specialist by an unknown amount. Neither XGBoost arm selects.
- Tiny specialists: *M. bassanus* 1,884 windows (20,000 x 256 is about 2,700 passes); *G. himalayensis* 14 training birds, 6,782 windows; *T. antarctica* 2,564, *A. brachyrhynchus* 2,642, *T. chrysostoma* 2,645.
- 3 seeds here, 5 in fold 0.
- Pools of 22 to 28 birds (2.0M to 3.2M windows); genus, not family, is kept together (the albatrosses sit in folds 7 and 8; *M. milvus* and *G. himalayensis* share fold 5).
- Stork: test partition read before; Tier 2 manifest (min_windows 100 at C = 24, 105 val / 105 test), not the F = 25 policy.
- Overlaps below 100 shared fixes stay in. Training birds against val/test birds of their species: stork 22 (12 at 10–99), *M. americana* 52 (24 at 10–99), *C. aura* 13 (1 at 10–99), *B. leucopsis* 6, *H. caspia* 1 (10–99), *A. albifrons* 1. Val birds against test birds: stork 1 and *M. americana* 1, each at 1–9. No cross-species overlap below 100.
- Boundaries were computed before the new exclusions. Recomputed without them (sensitivity only; values used are unchanged):

| bird | used | without exclusions | change |
|---|---:|---:|---:|
| *C. ciconia* | 61.104 m | 61.530 m | +0.7% |
| *C. aura* | 74.886 m | 73.535 m | −1.8% |
| *G. himalayensis* | 304.605 m | 331.420 m | +8.8% |
| *M. americana* | 59.151 m | 59.375 m | +0.4% |

- Exact matching misses near duplicates (the goose 1056306620 is the known case).
- Chosen after the bird seal, knowing the dove result. No multiplicity correction across 30 birds.

## 9. Note 

No rotation run existed, and no test partition of the 27 had been read, when this file was first committed. Only `bird_rotation_sealed_eval.py` opens them, after this file is committed with sec 6 and sec 7 filled.
