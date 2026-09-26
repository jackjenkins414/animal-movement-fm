# Tier 4: the 26-source pool (exploratory)

Exploratory and never reported as a tiered result (`preregistration_tier3.md` sec 6). Written 24-09-2026, after the Tier 3 seal. We also decided to open the held-out test partitions for the third time. Committed before any run, then committed again with sec 7 filled before the sealed evaluation, which will not run until sec 7 is complete. Results we're again piling into `results.md` but marked exploratory.

## 1. Pool

All 26 eligible sources (eligible minus held-out and reserved). No window floor, corrected `id_key` counts, A8 proportional $\sqrtn$. Probabilities and sub pools we wrote once to `registered/tier4_source_pool.json`.

| domain | sources (training windows) | draws |
|---|---|---:|
| flying (14) | *C. ciconia* 806,595 · *C. aura* 210,939 · *A. caerulescens* 199,754 · *M. americana* 137,954 · *H. ostralegus* 127,354 · *C. corone* 111,895 · *B. canadensis* 99,724 · *F. naumanni* 31,995 · *M. milvus* 25,234 · *P. apivorus* 22,509 · *G. himalayensis* 9,621 · *D. exulans* 3,393 · *T. chrysostoma* 2,645 · *T. antarctica* 2,564 | 65.4% |
| terrestrial (5) | *C. elaphus* 556,579 · *C. taurinus* 51,417 · *B. bison* 37,729 · *C. latrans* 37,253 · *L. rufus* 20,206 | 25.0% |
| marine (7) | *M. leonina* 53,076 · *L. weddellii* 7,718 · *A. gazella* 5,414 · *E. chrysolophus* 2,818 · *O. rosmarus* 2,460 · *L. carcinophaga* 2,286 · *A. patagonicus* 1,170 | 9.6% |

Stork's draw share falls from 32.6% at Tier 3 to 14.9%.

## 2. Arms

Tier 3 configuration and budget: grid 11, 20,000 updates at batch 256, seeds 0–4, final checkpoint. Z is the full package and B is the plain trunk. Every Tier 3 arm that depends on the source pool is mirrored at 26 sources; the rest are carried. F26 selects on validation like F9 and needs Z26 seed 0 first.

| arm | Tier 3 counterpart | pool | scored on | runs |
|---|---|---|---|---:|
| Z26, B26 | Z9, B9 | all 26 | all three | 5 + 5 |
| Za26 | Za9 | 26 minus the held-out species' domain | all three | 15 |
| Zd-F | Zd3 | 14 flying | gull | 5 |
| Zd-T | Zd3 | 5 terrestrial | wolf | 5 |
| Zd-M, BdM | Zd3, new | 7 marine | NES | 5 + 5 |
| ZdM-noMir, BdM-noMir | new | 6 marine, no *M. leonina* | NES | 5 + 5 |
| B26dom | B9dom | all 26, three-row domain embedding | all three | 5 |
| F26 | F9 | Z26 seed 0 fine-tuned on the same k = 1 / 10 / 100% subsets and draws | all three | 21 |
| LSTM-B26, XGB-B26 | LSTM-B9, XGB-B9 | all 26 | all three | 5 + 5 |

**Carried from Tier 3's sealed prediction files, not retrained:** P, V, ARIMA, AVG3/10, S, SB, Sk, LSTM-S, XGB-S, Zd1, F9, F3c, and Tier 3's own pooled arms (Z9, B9, Za9, Zd3, B9dom, LSTM-B3/B9, XGB-B3/B9, E1embed_dom), which are the comparison points rather than pool-independent.

91 runs, about 1,075 SU at Tier 3's measured 12.5 SU per neural run (86 neural; the five XGB-B26 fits are not neural and cost less). Ten blocks of ten.

## 3. Reading rules

A gap within +/- 0.05 counts as no difference. The statistic is travel p90 skill at F = 25, median of five seeds, with a paired bootstrap of 2,000. Tier 2 and Tier 3 comparators are read from their sealed prediction files.

**NES: domain or genus?** $\Delta$_Z = ZdM-noMir − Zd-M and $\Delta$_B = BdM-noMir − BdM.

| $\Delta$_Z and $\Delta$_B | reading |
|---|---|
| both below −0.05 | genus |
| both within noise | domain |
| one below −0.05, the other within noise | the package and the congener interact |
| anything else | unresolved |

**Other comparisons.** Any pattern not listed is unresolved and reported per species. F-type rows pair the same draw index on both sides and get their own paired bootstrap (Tier 3 freeze sec 6.1); the mean across draws is descriptive.

| comparison | result | reading |
|---|---|---|
| B vs Z within each marine pool | B wins in both pools | the package, not the pool composition, is what matters |
| B26 vs Z26 | B26 wins on some species, Z26 on none | the package doesn't pay at 26 sources |
| | Z26 wins on some species, B26 on none | the package needs pool diversity |
| Z26 vs Z9 | Z26 wins on two or more | scaling works |
| | within noise on all three | going from 3 to 9 to 26 sources changes nothing |
| | Z26 loses on some, wins on none | more sources hurt |
| four-way pool: Z26 vs Za26 vs Zd vs Zd1 (carried), as Tier 3 prediction 5 | Z26 wins over both Za26 and Zd on two or more | both same-domain and other-domain sources help; transfer is not domain-bound |
| | Zd at or above Z26 on two or more | domain-bound at 26 sources too, repeating Tier 3 |
| | Za26 at or above Z26 on two or more | the other domains carry it and the same domain adds nothing |
| F26 vs Sk on the same draw at k = 1%, as Tier 3 prediction 3 | Sk wins on most draws | pretraining still loses to random initialisation, as at nine sources |
| F26 vs F9 on the same draw, as Tier 3 prediction 4 | F26 better on two or more species | pool size helps the initialisation, not the destination |
| ladder with LSTM-B26 and XGB-B26 rungs | — | descriptive, no pass/fail, as Tier 3 sec 3.1 |

## 4. Dilution diagnostic

Per species loss shares against draw shares (Tier 3 freeze sec 7.1), for Z26, B26 and the four NES arms. At Tier 3, stork took 64% of Z9's loss from 33% of draws. If stork's share falls and Z26 still doesn't improve, the imbalance wasn't the cause.

## 5. Limitations

- Chosen after the Tier 3 seal. Third opening of the same 30 test animals, with no multiplicity correction.
- No floor: *A. patagonicus* is passed about 25 times per run in Z26 and about 432 times in the no-*Mirounga* arms.
- Domain vs genus can be tested only on NES. The marine sources are mostly Southern Ocean; NES is North Pacific.
- 17 of the 26 sources have never been through a training run.

## 6. Pre-run prediction

- **NES 2 × 2:** both $\Delta$ below −0.05, the genus row. Tier 3's marine skill rose as the pool narrowed toward the congener (Zd1 +0.1798, Zd3 +0.1140, Z9 +0.0043), and the six no *Mirounga* sources hold 21,866 windows between them.
- **Z26 vs B26:** B26 wins on NES by more than 0.05, gull and wolf within noise. B9 beat Z9 by 0.1229 on NES, and `results.md` sec 17 found the conditioning path negative within a species as well.
- **Z26 vs Z9:** within noise on all three. Nothing in sec 18 moved with pool size, and a lower stork draw share does not touch the representation or the low displacement failure it sits on.
- **F26 vs F9 and Sk:** Sk beats F26 on most draws (17 of 18 at Tier 3), while F26 edges F9 on the gull only, because pool size showed up in the initialisation rather than the outcome.
- **Z26 vs the domain models:** Zd-M at or above Z26 on NES and resolved, gull and wolf flat, repeating prediction 5 — every source from outside the held-out species' domain subtracts.

## 7. As trained

**DRAFT (GenAI) — Jack to edit before commit.**

| | |
|---|---|
| runs | 91 of 91, all exit status 0, all on gpuhopper |
| neural (86) | 20,000 of 20,000 updates applied; none withheld, no non-finite steps |
| XGBoost (5) | fit once |
| compute | 1,261.4 SU, about 13.9 per run (sec 2 estimated about 1,075); longest run 21m12s |
| walltime | lowered with qalter while queued: 1 h for 20–59, 1.5 h for 60–90; no run came close |
| deferred or missing | none |

Dilution diagnostic (sec 4; pre-seal, training records only, median across seeds). Z26: stork 18.0% of loss from 14.9% of draws (Tier 3 Z9: 64.1%), turkey vulture 27.8% from 7.6%; by domain, rescaled to 100%, flying 93.7 / terrestrial 4.8 / marine 1.5% from draws of 65.4 / 25.0 / 9.6%. B26 shows the same pattern (turkey vulture 25.0%, stork 13.4%).
In the marine arms loss tracks draw share within about 6 points: *M. leonina* takes 34–35% of loss from 40.0% of draws, and without it *L. weddellii* takes 31–32% from 25.4%.

## 8. Note 

No Tier 4 test result exists. Every arm was chosen knowing the Tier 2 and Tier 3 results. The reserved species (*A. albifrons*, *O. dalli*, *P. adeliae*) are in no pool. Only the Tier 4 sealed evaluator opens test partitions.

## 9. Corrections

| date | what | effect |
|---|---|---|
| 27-09-2026 | Report step decoded carried `tier3:` tags with Tier 3's parse_tag, so every carried Tier 3 arm was ignored in our job 179924605's report (3_scaling not scored; Zd1/Zd3, F9, Sk and Tier 3 ladder rungs absent). Tier 4's parse_tag also placed subset tags ending in a seed (Sk, F9, F3c) under that seed rather than their draw; it now reads the trailing seed. summarise_all now takes Tier 4's decoder; build_report fails if a carried arm is missing. | Report-only rerun; no predictions regenerated, no test data re-opened for new arms, no arm list/metric/reading rule changed. Note as per the code, I used GenAI to diagnose this and help correct it. |
