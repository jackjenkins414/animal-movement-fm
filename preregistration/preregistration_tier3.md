# Preregistration: Tier 3 source pool scaling

**Date: 16-09-2026.** Sec 1 was written and committed before the source selection was run. Secs 1.1-6 were added the same day, after the selection and before any Tier 3 model was trained. The Tier 3 pre-test freeze (phase B) follows separately. This file will become our Tier 3 component of `preregistration_tier2.md`. Tier 3 differs from Tier 2 in one variable, the size and composition of the source pool (three sources growing to nine), and holds everything else fixed: the same three held-out species and frozen splits, metrics, success bands and protocol (`preregistration_tier2.md` sec 1). Sections beyond sec 1 are still to be written and will be completed before any Tier 3 model is trained.

**Status when sec 1 was written:** no Tier 3 model has been trained, no Tier 3 configuration evaluated, and the source selection below has **not yet been run**. Candidate eligibility (A7 gate 1 at C = 12) and the descriptor profiles were produced before this rule was written, as dataset property exploration in the sense of Hofman et al. (2023); no pairwise distance between Tier 3 candidates had been computed for the purpose of choosing them. The three held-out test partitions were opened for Tier 2 on 26-08-2026 and will be read again by Tier 3; that is declared here and will be restated in the Tier 3 freeze.

## 1. Selecting the nine source species

Tier 2 chose its sources by hand from printed distances (`tier2_species_roles.md`). For nine sources that is a large enough choice to be a degree of freedom, so the rule is fixed here and applied mechanically.

**The pool is nested.** The three Tier 2 sources, *Ciconia ciconia*, *Cervus elaphus* and *Mirounga leonina*, are retained. Two further sources are added per domain, giving three per domain as `preregistration_tier2.md` sec 1 anticipates. Tier 2's pool is therefore a subset of Tier 3's, so the 3 to 9 comparison changes the pool without replacing it.

**Candidates.** A species is a candidate if all of the following hold:

1. it clears A7 gate 1 at C = 12 (`results/species_eligibility.csv`, regenerated 16-09-2026: 32 species pass);
2. it holds no Tier 2 role;
3. it has a profile row in `results/regime_profiles.csv` with all eight descriptors present;
4. its **training partition holds at least 2,000 valid windows** at C = 12, counted from its frozen split manifest with the registered windowing rule (the $\n_s$ of A8). The floor exists because A8 draws a species in proportion to $\sqrt{n_s}$, so a very small source is resampled many times per run; below 2,000 windows it would be revisited roughly 20 times in a registered length run.

Domain is the `domain` column of the profile table as we corrected on 15-09-2026, so *D. exulans*, *T. chrysostoma* and *T. antarctica* count as flying and the penguins as marine. On the 16-09-2026 tables the floor removes *Thalassarche chrysostoma* (1,995) and *Aptenodytes patagonicus* (1,170), leaving **12 flying, 5 terrestrial and 6 marine candidates**, and 66 $\times$ 10 $\times$ 15 = **9,900** possible pools.

> **Correction, 16-09-2026, after the selection was run.** The counts in the previous sentence were miscounted when this section was drafted. The rule's four conditions actually leaves **13** flying candidates, not 12, so **78** x 10 × 15 = **11,700** pools were scored. The rule itself is unchanged, and the selection applied the conditions as written. Our rationale for condition 4 is also inaccurate: "roughly twenty times" came from the 29 source Tier 4 pool. In the selected 9 source pool, a 2,000 window source would be revisited about 47 times in a 20,000 step run, and the three smallest selected sources about 32–38 times. The floor is not changed; the consequence is carried to the step budget decision in the Tier 3 freeze.

> **Correction, 20-09-2026, after the selection was run and before any Tier 3 model was trained.** Our selection above ran on undercounted training windows (note GenAI found this, I added its suggested fixes and used its explanation to write this). `species_window_counts` matched frozen manifest ids to parquet ids as plain strings, while the training pipeline matches through `id_key`; `canonical_manifest_ids` writes an id through a float parse, and Python treats underscores inside digits as separators, so the RAATD id `146_2160` is spelled `1462160` in the manifest. Every individual whose id differs between the two spellings was silently counted as zero windows. Corrected counts: *A. gazella* 3,389 -> **5,414**, *D. exulans* 2,971 -> **3,393**, *T. chrysostoma* 1,995 -> **2,645**, *T. antarctica* 2,490 -> **2,564**. There are no `id_key` collisions, and Tier 2's three sources and all three held-out species reproduce their registered counts exactly, so **no sealed Tier 2 number moves**.
>
> The consequence for this section is that *Thalassarche chrysostoma* was removed by the 2,000 window floor on a wrong count. It is a candidate, and the sentence above naming it as excluded is superseded; *A. patagonicus* (1,170) remains excluded. There are therefore **14** flying candidates, not 13, and 91 x 10 x 15 = **13,650** pools were scored, not 11,700.
>
> **The rule and its outcome are unchanged.** Re run mechanically on the corrected counts, the objective selects the same nine species, with the same mean pairwise distance 5.0133 and minimum 2.2149, again one pool tied on the mean, and the same runner up (*A. caerulescens* for *B. canadensis*, 5.0050). *T. chrysostoma* entered the search and was not selected. What changes is now our A8 draw probabilities, which depend on the counts; the sec 1.1 table's training windows and probabilities are superseded by those in `registered/tier3_source_pool.json`, the largest single move being *A. gazella* 2.126% -> 2.671%. The registered sub pool probabilities for Za9 and Zd3 move with them; the largest is *A. gazella* within `domain_marine` (Zd3 marine), 17.04% -> 20.61%.
>
> `registered/tier3_source_pool.json` was regenerated on 20-09-2026 with `source_pool_weights.py --tier3 --regenerate`, and the superseded pool, counts, probabilities and sub pools are recorded inside the new file under `regenerated.supersedes`.

**Congeners of held-out species are allowed and declared.** *Canis latrans* shares a genus with the held-out *Canis lupus*. It remains a candidate, and if selected the link is declared as a confound for the terrestrial result, as the *Mirounga* pair already is in Tier 2. (*Larus argentatus* was excluded at ingest for Tier 3's flying candidates; that exclusion predates this rule and is not revisited.)

**Distance.** Euclidean distance over the eight gate 2 descriptors, each transformed and standardised with the **frozen Tier 2 standardisation** `registered/tier2_descriptor_standardisation.json` (A10 transforms, A16 freeze). One fixed metric is used across tiers rather than refitting over the Tier 3 table. That standardisation was fitted over the 21 species as they stood at the Tier 2 freeze, before A15's QC rebuilt the candidates, so its centres and scales reflect pre-QC marine tail ratios; that is a declared property of the metric, not a reason to refit it.

**Objective.** Among all admissible pools, choose the one with the **largest mean of the 36 pairwise distances** among its nine species. Ties (difference $\leq$ 1e-9) go, in order, to:

1. the larger minimum pairwise distance
2. the larger total training window count across the nine
3. the pool whose sorted species names come first alphabetically

Model performance is never used, and neither is any held-out species' descriptor row or distance.

### 1.1 The selected pool

**Result of our sec 1 rule, run once on 16-09-2026 and recorded in `registered/tier3_source_pool.json`.** 11,700 pools were scored and one was tied on the best mean. Mean pairwise distance was **5.0133**, minimum **2.2149** (Tier 2 pool: 5.03 / 2.94).

| domain | sources | training windows | A8 draw probability |
| --- | --- | ---: | ---: |
| flying | *C. ciconia* (Tier 2), *Branta canadensis*, *Thalassoica antarctica* | 806,595 / 99,724 / 2,490 | 32.792% / 11.530% / 1.822% |
| terrestrial | *C. elaphus* (Tier 2), *Bison bison*, *Canis latrans* | 556,579 / 37,729 / 37,253 | 27.240% / 7.092% / 7.047% |
| marine | *M. leonina* (Tier 2), *Arctocephalus gazella*, *Eudyptes chrysolophus* | 53,076 / 3,389 / 2,818 | 8.412% / 2.126% / 1.938% |

The runner up (with a mean of 5.0050) differs only in taking *Anser caerulescens* instead of *B. canadensis*. The draw probabilities follow A8 unchanged ($\alpha = 0.5$, window uniform within species). We also compute sub pool probabilities for Za9 and Zd3 (sec 3) with the same rule and add to the same file before the freeze.

## 2. Reserved future test species

Three eligible species are withheld from every Tier 3 and Tier 4 run so that a later study has held-out species whose test partitions nobody has read. Our rule, applied on 16-09-2026 to `results/species_eligibility.csv` and nothing else; in each domain, among eligible species that hold no Tier 2 role and are not Tier 3 sources, take the species with the most individuals clearing the evaluation floor F = 25; ties go to more individuals with at least one window, then we just resort to alphabetically first name rather than just our choice. 

| domain | reserved | individuals >= F | runner-up |
| --- | --- | ---: | --- |
| flying | *Anser albifrons* | 270 | *A. caerulescens* (140) |
| terrestrial | *Ovis dalli* | 72 | *Lynx rufus* (66) |
| marine | *Pygoscelis adeliae* | 112 | *Leptonychotes weddellii* (104) |

Note that no model has been trained on, or scored against, any of the three. Their splits and profiles exist from the candidate pipeline, and their descriptor rows entered the sec 1 distance computation as candidates (training individuals only). **Tier 4 is therefore every eligible species except the three held-out and the three reserved: 26 sources.** Reserving *A. albifrons* while Tier 4 trains on its congener *A. caerulescens* is noted for the later study. It will also take our later study from 3 held-out species to 6, which helps given how thin n = has been. 

## 3. Registered arms

Everything not stated here is as is located in `preregistration_tier2.md` and `preregistration_tier2_freeze.md`.

**Configuration.** Grid index 11 (`d_model` 256, 4 layers, LR 1e-3), unchanged and not re tuned, so there is no Tier 3 inner loop. The Tier 2 freeze called this selection fragile (a corner of the grid, only rank 1 resolved against +/- 0.05 single seed noise). We keep it anyway since Tier 3's purpose is a controlled 3 to 9 source pool comparison, and re tuning would change a second variable. Any 9 source re tuning is a later diagnostic and cannot alter this configuration.

**Budget.** 20,000 applied optimiser updates at batch 256, as in Tier 2, so compute is held fixed across pool sizes.

| arm | definition | runs |
| --- | --- | ---: |
| P, V | unchanged from Tier 2 | carried |
| S | unchanged from Tier 2: trained on the held-out species only, so it does not depend on the source pool | carried |
| **Z9** | the full FM package trained on the nine sources | 5 |
| **B9** | B trained on the nine sources | 5 |
| **Za9** | Z9 retrained without **all three** sources from the held-out species' domain (nine to six) | 15 |
| **F9** | Z9 seed 0 fine-tuned on k = 1%, 10%, 100% of the held-out species' training individuals (3 / 3 / 1 draws, the Tier 2 subsets) | 21 |
| **F3c** | *Tier 3 companion 3-source rerun*: Tier 2's Z seed 0 fine-tuned exactly as F9 | 21 |
| **Sk** | from scratch specialists trained only on the exact k = 1% and 10% F subsets | 18 |
| **Zd3** | *domain model*: the full FM package trained only on the **three Tier 3 sources from the held-out species' domain** (for example, stork, *B. canadensis* and *T. antarctica* for the gull) | 15 |
| **Zd1** | *Tier 3 companion one-source domain model*: the full FM package trained only on the **Tier 2 source from that domain** (stork for the gull, *C. elaphus* for the wolf, *M. leonina* for NES) | 15 |
| **B9dom** | B9's trunk plus a learned three row **domain** embedding driving the FiLM path (context statistics off, reconstruction off), trained on the nine sources. Each held-out species is scored under its own domain's row, which is known for an unseen species, so the arm stays zero-shot | 5 |

The three source comparison point for Z, B and Za is Tier 2's sealed result. For the domain label, the three source companion is Tier 2's exploratory E1embed (the same design, where each domain has one species), read under the matching domain's row. It is scored on the test partitions without retraining. Note that we chose that arm after its validation result which is declared here for clarity. 

**Domain models.** Zd3 and Zd1 use Z's configuration, budget and also final checkpoint rule, with no validation privilege. Zd3 draws its three sources under A8 renormalised over the three ($\sqrt{n_s}$). Zd1 has one source. Both of these were suggested by Ana Sequeira (17-09-2026) to test whether transfer is a within domain phenomenon. Together with Z9 (all nine) and Za9 (the six from other domains), each held-out species gets a same domain-only pool, an other domains only pool and the full pool at Tier 3. With Tier 2's Z, Za and Zd1, it gets the same three pools at one source per domain.

**F budget, and why there are now two F objects.** F9, F3c and Sk use S's budget; 20,000 steps at peak LR 1e-3, with checkpoint selection every 1,000 steps on the held-out species' validation individuals, on the same terms as S. This replaces Tier 2's 2,000 steps at 1e-4 for Tier 3 only. This change is motivated by our Tier E validation evidence; at the registered budget, fine-tuning barely moved the model. **Tier 2's registered F is not re run, revised or replaced and stays exactly as we first reported.** F3c is a new Tier 3 arm that pairs the 3 source pretraining with the new budget, so that F9 against F3c compares pool sizes under a single budget. It is not a corrected Tier 2 result. F still initialises from seed 0 only (A13), so F type arms carry draw spread and no seed spread. F9 against S at k = 100% now differs only in initialisation, which resolves A13(3) for Tier 3.

### 3.1 Baseline ladder

The Tier 1 baseline ladder is carried into Tier 3 so that S and Z are read against the model families that our project started from, in both the traditional single species setting and the pooled multi-species one. **Every rung is rebuilt on the registered Tier 2 method:**
- the same 12-channel encoded windows at C = 12;
- no absolute position;
- the same continuity rule, evaluation floor, strata, individual weighting and paired bootstrap.

**No rung is tuned.**
- Architectures and hyperparameters are Tier 1's settled values (written down in `methods.md`).
- Neural rungs train with Tier 2's recipe (radial pseudo Huber, 20,000 updates at batch 256, warmup plus cosine, clip 1.0).

| block | rung | definition | runs |
| --- | --- | --- | ---: |
| training free | ARIMA | ARIMA(2,0,2) fitted per window to the context displacements | 0 |
| training free | averaging | mean of the last k context displacements, k in {3, 10} (V is k = 1) | 0 |
| specialist | XGB-S | `WindowXGBoost` (200 trees, depth 6, rate 0.1) on each held-out species' own training windows | 3 |
| specialist | LSTM-S | LSTM trunk (hidden 64, 1 layer), plain, trained on each held-out species with S's recipe and validation selection | 15 |
| specialist | SB | the plain transformer trunk on each held-out species with S's recipe: Tier E arm `SB`, whose configuration was fixed before any of its runs, scored here on test | 0 |
| pooled | XGB-B3, XGB-B9 | `WindowXGBoost` on 1,000,000 windows drawn under A8 from the three Tier 2 sources or the nine Tier 3 sources | 10 |
| pooled | LSTM-B3, LSTM-B9 | the LSTM trunk trained on those pools like B | 10 |
| pooled | B (Tier 2), B9 | the plain transformer trunk trained on the three or nine sources; already registered in sec 3 (Tier 2's sealed B and the Tier 3 arm B9), listed here so the rung is complete | 0 |

**Seeds and determinism.**
- XGB-S is deterministic and run once per species, so it has no seed spread.
- XGB-B varies only through its seeded draw.
- Specialists do not depend on the source pool, so one set serves both tiers.

**Reporting.** The ladder is **descriptive**; travel p90 skill and the full error table for every rung, with no pass/fail and no success band. It exists to place S, Z and B, not to test a new claim.

**Declared differences:**
- XGBoost's pseudo-Huber is per axis, where the FM's is radial.
- The LSTM is not capacity matched to grid 11 (Tier 1's settled size is kept, rather than tuned).
- ARIMA on 11 displacements falls back to persistence where the fit fails, as in Tier 1, and the fallback rate is reported.

## 4. Success bands and predictions

**Success bands.** Tier 2 sec 4, unchanged, applied to Z9 with B9 and the carried S and P.

**Predictions**, with "beats" as in Tier 2 sec 3; a strictly lower headline travel-regime p90, the median across seeds for Z type arms and across the three draws for F type and Sk arms.

1. **Source-pool scaling at zero shot: the Tier 3 hypothesis.** More source species should in theory help zero-shot transfer; Z9 beats Tier 2's Z for at least two of three held-out species. Both are scored on the same test windows, so the registered paired bootstrap (2,000 resamples of test individuals, seed 42) is reported for Z9 minus Z. As everywhere in Tier 2, the interval does not create a second pass/fail rule, and gaps inside the ±0.05 single-seed noise scale are reported as unresolved even if the point-estimate rule is met.
2. **Domain ablation (Tier 2 prediction 5, carried).** Z9 beats Za9 for at least two of three held-out species. As in Tier 2, this cannot separate the matched domain from pool size (nine against six), and for *C. lupus* removing the terrestrial sources also removes its congener *C. latrans*.
3. **Few-shot, the main secondary prediction.** At k = 1%, F9 beats Sk trained on the same individuals for at least two of three held-out species. k = 10% is reported as the corresponding higher data point, with no pass/fail attached.
4. **Source-pool scaling under fine-tuning.** At k = 1%, F9 beats F3c on the same individuals for at least two of three held-out species. 10% and 100% are reported alongside.

5. **Is transfer domain-bound?** Z9 beats Zd3 for at least two of three held-out species, meaning sources from other domains add to same domain sources. The competing hypothesis (that Ana suggested to test), is that transfer is domain bound and other domains add noise, predicts Zd3 at or above Z9. Read together with prediction 2, per species:

   | Z9 vs Zd3 | Z9 vs Za9 | reading |
   | --- | --- | --- |
   | Z9 better | Z9 better | both kinds of source help; transfer is not domain bound |
   | Zd3 same or better | Z9 better | domain bound: the same domain sources carry the transfer |
   | Z9 better | Za9 same or better | other domains carry it; the same domain adds nothing |
   | Zd3 same or better | Za9 same or better | pooling across the nine adds nothing either way |

Gaps inside the +/- 0.05 noise scale we will read as the "same". Tier 2's Z against Zd1, and Zd1 against Zd3 (within domain scaling from one source to three), are reported alongside, with no pass/fail.

**Domain label, descriptive.** B9dom against B9, and E1embed read by domain row against Tier 2's B, are reported on the test partitions with no pass/fail. The only evidence so far is a validation gain on NES inside the noise scale (`results.md` sec 16).

**Trait conditioning is not a Tier 3 prediction.** Tier 3 runs Tier 2's FM package unchanged, so that the only difference from Tier 2 is the source pool. Trait conditioning is deferred to a later, separately registered ablation on the Tier 3 and Tier 4 pools. This departs from Tier 2 sec 5.1, which named Tier 3, and is recorded there as A17.

Tier 2 prediction 2 (transfer tracks descriptor distance) is not carried, because it is not testable with three held-out species at any tier. 

## 5. Descriptive diagnostics

**Regime mixture diagnostic.** Reported for every Tier 3 and Tier 4 species. The zero displacement fraction is reported separately; one, two and three component log normal mixtures are fitted to strictly positive one bin step lengths; the component count is chosen mechanically by BIC; and component weights, scales and fit diagnostics are reported, with the existing convergence and minimum component weight requirements kept so that a tiny component is not read as a movement regime. **Note that it is descriptive only.** It cannot change the frozen `0.1 x p50` boundary (A4, frozen under A16), species eligibility, source selection or any success criterion.

**Per species loss shares and resampling.** Reported for Z9, B9, Zd3 and Zd1, because the three smallest sources are revisited 32-38 times per 20,000 step run against about 2 for the largest.

## 6. Declared limitations

1. **The Tier 3 held out partitions are not newly blind.** They are the same fixed partitions that we opened for Tier 2 on 26-08-2026, and so tier 3 decisions are preregistered before Tier 3 training, but they are necessarily made with knowledge of the earlier Tier 2 results.
2. **The configuration is Tier 2's fragile selection**, we keep deliberately (sec 3).
3. **F has a new budget, and F3c is a new arm.** Tier 2's F stands as reported (sec 3).
4. **Za9 and the domain models are confounded with pool size** (nine against six, three or one), and for the wolf, removing or keeping the terrestrial sources also removes or keeps its congener.
5. **Domain models repeat their data heavily** at the fixed 20,000 updates. Passes per run for Zd3: marine *M. leonina* 65, *A. gazella* about 257, *E. chrysolophus* about 282; flying stork 4.5, *T. antarctica* about 81; terrestrial *C. elaphus* 6, *B. bison* and *C. latrans* about 23. For Zd1: stork 6.3, *C. elaphus* 9.2, *M. leonina* 96.5. But they keep Z's final checkpoint rule anyway, for comparability; and so we declare overfitting as a risk, and per species loss shares are reported. The marine domain model is entirely Southern Ocean, and predicting on unseen regions has been already recorded as a limitation (TFT paper). 
6. **Descriptor distance prediction is not carried** (sec 4).
7. **Coyote, a congener of the held-out wolf, is a source.**
8. **Every marine source is Southern Ocean** (SOES and the two added marine species, as is the added flying seabird *T. antarctica*), while the held-out *M. angustirostris* is North Pacific. So "marine" and "polar" coincide on the source side. Tier 2's Argos against Fastloc confound in the marine pair is unchanged.
9. **Descriptor distances keep the Tier 2 frozen standardisation**, even though some candidate data were rebuilt under later QC (A15). This keeps distances comparable across tiers, but the scaling constants do not describe the final 32 species pool. The mean distance objective selected three sources with fewer than 3,500 training windows each, so part of the measured coverage may reflect data quality rather than movement.
10. **The pool is imbalanced and small sources are oversampled.** Flying 46.1%, terrestrial 41.4%, marine 12.5% of draws, with three sources revisited 32-38 times per run.
11. **Species is confounded with study.** Holding out a species also holds out its tags, fix schedules and regions, so a transfer failure cannot be attributed to taxonomy alone. Coordinate precision also differs by study (for example, 44% of *Cathartes aura* longitudes lie on a 1e-4 degree grid), which gives the model a study fingerprint.
12. **White stork contains 254 zero displacement steps within frozen runs of 20 or more bins.** Their cause is not established, and the frozen Tier 2 preprocessing is not rebuilt (A2).
13. **The regime diagnostic has known limitations** (sec 5); the frozen boundary is applied regardless.
14. **Exploratory work runs in parallel.** Single species architecture ablations and a 40,000 step budget check run as Tier E on validation individuals only. Anything they motivate we be register here before the Tier 3 freeze, or it cant enter Tier 3.
15. **A9's plausibility drop reached only the rebuilt marine role species.** The four GPS role species' extracts predate A9 and still hold fixes carrying the speed flags A9 drops (over ceiling, spike or speed outlier), counted directly from the extract files on 16-09-2026: *C. ciconia* 110,909 of 161.5M, *L. fuscus* 4,625 of 27.6M, *C. elaphus* 47, *C. lupus* 45. Resampling keeps one fix per bin, so few survive into the modelled data: 2,166 of 1.25M stork segment rows (0.17%), 82 for *L. fuscus* (one in its test individuals), 22 for *C. elaphus*, and 1 for *C. lupus* (none in its test individuals). Our A2 forbids rebuilding them.

**Tier 4** trains on the 26 sources of sec 2 under the same configuration and budget. It is exploratory and is never reported as a tiered result.

**Recorded outside this list:** per species sample sizes, including *B. bison*'s 25 training individuals (source table in `registered/tier3_source_pool.json`); the A15 speed ceilings kept for stork and gull (A15); and the A16 follow up defects fixed on 16-09-2026, which changed no Tier 2 number. 

## GenAI reference

Section 1 was drafted with GenAI from decisions I made but asked GenAI to confirm; nested pool, mean distance objective with the minimum as the first tie break, the frozen Tier 2 metric, a 2,000 window floor, and congeners allowed but declared.
