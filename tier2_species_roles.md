# Tier 2 species roles frozen

Here we outline preregistration 1 gate 2 record, such as selected roles, pairwise descriptor distances, and selection reasons. This was committed before any Tier 2 model was trained and before any held-out partition was read. The roles are frozen under section, and the held-out set will carry over into Tier 3 with the same splits.

This final version (running file the last 5 days) was just completed, since we re derived every candidate through QC after correcting duplicate resolution and a few other minor issues, and refroze the descriptor standardisation over the cleaned table. This changed the descriptor values but did not change the selected roles or the A4 branch.

Added this modifcation on 16-08-26 under our A9, which applied the physical plausibility QC flags that had been computed but we never acted on; thus we rebuilt both marine species and regenerated their split manifests.

Updated again under A10, which put each descriptor on its natural measurement scale before standardising:

## 1. Roles

| domain      | source                    | held-out                        | pair distance |
| ----------- | ------------------------- | ------------------------------- | ------------: |
| flying      | *Ciconia ciconia*         | *Larus fuscus*                  |          3.21 |
| terrestrial | *Cervus elaphus*          | *Canis lupus*                   |          2.44 |
| marine      | *Mirounga leonina* (SOES) | *Mirounga angustirostris* (NES) |          1.88 |

Split manifests are `splits/stork_split_seed42.json` and the corresponding `larus_fuscus`, `cervus_elaphus`, `canis_lupus`, `mirounga_leonina`, and `mirounga_angustirostris` `_split_seed42.json` files.

## 2. Pairwise distances

Distances are Euclidean distances over the eight descriptors, each now transformed to its natural measurement scale under A10 and then standardised. The transform map, centres and scales are now all frozen together in `results/regime_descriptor_standardisation.json`. Raw descriptor values are in `results/regime_profiles.csv`.

|                  | stork | *L. fuscus* | *Cervus* | wolf | *M. leonina* | *M. ang.* |
| ---------------- | ----: | ----------: | -------: | ---: | -----------: | --------: |
| **stork**        |  0.00 |        3.21 |     5.72 | 4.43 |         6.43 |      7.24 |
| ***L. fuscus***  |  3.21 |        0.00 |     3.81 | 1.98 |         3.75 |      4.86 |
| ***Cervus***     |  5.72 |        3.81 |     0.00 | 2.44 |         2.94 |      3.64 |
| **wolf**         |  4.43 |        1.98 |     2.44 | 0.00 |         3.30 |      4.16 |
| ***M. leonina*** |  6.43 |        3.75 |     2.94 | 3.30 |         0.00 |      1.88 |
| ***M. ang.***    |  7.24 |        4.86 |     3.64 | 4.16 |         1.88 |      0.00 |

For the source pool of stork, *Cervus*, and *M. leonina*, the mean pairwise distance is 5.03 and the minimum is 2.94. Note we outlined 1 to use the source pool to maximise descriptor space coverage. Within domain pair distances serve a different purpose. Their range are 3.21, 2.44, and 1.88 allows secondary prediction 2 to test whether transfer varies with source target distance.

Note, these are the post A10 values. The A9 values, on raw z-scores, were 5.08 / 2.65 / 1.46 with source pool coverage 5.47/2.74, and the pre A9 values before that were 5.10 / 2.65 / 1.42 with coverage 5.62/3.42. The A4 branch and every role are still unchanged across all three.

## 3. Our reasons

* **Flying.** Stork is fixed as our source species by 1. *L. fuscus* is the largest remaining profiled flying candidate, with 330 individuals above the evaluation floor. It is also very different from stork in step scale, with a median step length is 4,115 m versus 611 m, while the corresponding tail ratios are 27 and 234.

* **Terrestrial.** *Cervus* and wolf provide a predator–prey comparison while keeping device type and region similar. Both datasets are GPS collars from Alberta and British Columbia. 

* **Marine.** Both species are *Mirounga*, which we already noted as a limitation of this pair. SOES and NES are geographically distinct: their median latitudes are −59.1° and +43.7°, so they occur in different hemispheres and ocean basins, and NES falls outside RAATD.

**The alternative marine candidates can no longer be fairly compared with this pair using distance, so we do not quote those numbers as if they were comparable.** A9 applied the physical plausibility QC flags only to the two *Mirounga* species because they are the species assigned roles. After removing implausible steps, their tail ratios dropped from 14.47 to 5.63 and 15.70 to 4.51. The other marine candidates in `results/regime_profiles.csv` are still pre QC, so their tail ratios like Weddell 21.13, Adelie 19.86, crabeater 55.85, and walrus 16.37, are still inflated by the same Argos artefacts removed from the selected pair. We'll get to these later. 

## 4. Declared deviations and confounds

**Terrestrial ordering differs from two registered tie breaks.** Note that 1 favours the species with the larger eligible test pool on the held-out side; *Cervus* has 323 eligible individuals compared with 110 for wolf. When we wrote 6, we favoured placing the better characterised dataset on the held-out side (pros and cons for this, but this ensures we can develop a sufficient S model for testing). 

So our selected ordering is *Cervus* as source and wolf as held out because source pool size changes substantially under the alternative. With wolf as the source, terrestrial data would contribute about 5% of source pool windows; 62,013 of 1,181,930. With *Cervus* as the source, it contributes about 35%: 593,874 of 1,713,791. So the former would make the pretraining corpus much more heavily weighted toward the flying and marine domains; which we obviously want to avoid.

(Those two pool totals were quoted as 1.24M and 1.77M before A9. The per species numerators reproduce exactly from the pipeline; the totals did not, by a near constant ~20,000 windows sitting in the stork term under every stork build on disk, and they have since moved again because A9 shrank the marine contribution. The figures above are the current pipeline values. The shares, and the argument, are unaffected.)

This justification was re checked under our now registered sampling rule (**A8**, weights revised by **A9**) before the splits were used, as required above. At the registered $\alpha = 0.5$ the terrestrial source contributes 39.8% of pretraining draws under the selected ordering against 16.1% under the reversed one, so the argument holds and the ordering stands. A species balanced rule would have made the two orderings indistinguishable and forced a revisit; $\alpha$ was fixed on grounds independent of the roles and is excluded from the tuning budget.

Descriptor coverage gives similar results under both orderings. The selected ordering gives a source pool mean/minimum distance of 5.03/2.94, compared with 4.72/3.30 for the reversed ordering. The reversed ordering therefore has a slightly larger minimum distance, as it did under the pre-A10 standardisation.

**Consequence.** The terrestrial specialist S trains on 99 individuals and 62,013 windows, about one tenth of the windows available from *Cervus*. S is the oracle used in
`p90_travel(Z) <= 1.25 x p90_travel(S)`. A weaker specialist raises this threshold and therefore makes the STRONG band easier for Z to satisfy. So this direction of bias does favour our study hypothesis and will be reported whenever we claim the STRONG band.

**Marine pair uses different device technologies.** SOES is entirely Argos based. NES carries both Argos and Fastloc GPS, with location class G on 228,343 of its 537,698 retained fixes (42.5%), against 158,329 class B and 97,182 class A. So marine transfer performance could reflect both species differences and device differences. And note that the G share rose under A9, from 38.3% to 42.5%, because the speed ceiling removes Argos fixes far more often than Fastloc ones; which is the same asymmetry stated above, now sitting inside the held-out species as well as across the pair.

