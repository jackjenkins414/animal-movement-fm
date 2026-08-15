# Tier 2 species roles frozen

Here we outline preregistration 1 gate 2 record, such as selected roles, pairwise descriptor distances, and selection reasons. This was committed before any Tier 2 model was trained and before any held-out partition was read. The roles are frozen under section, and the held-out set will carry over into Tier 3 with the same splits.

This final version (running file the last 5 days) was just completed, since we re derived every candidate through QC after correcting duplicate resolution and a few other minor issues, and refroze the descriptor standardisation over the cleaned table. This changed the descriptor values but did not change the selected roles or the A4 branch.

## 1. Roles

| domain      | source                    | held-out                        | pair distance |
| ----------- | ------------------------- | ------------------------------- | ------------: |
| flying      | *Ciconia ciconia*         | *Larus fuscus*                  |          5.10 |
| terrestrial | *Cervus elaphus*          | *Canis lupus*                   |          2.65 |
| marine      | *Mirounga leonina* (SOES) | *Mirounga angustirostris* (NES) |          1.42 |

Split manifests are `splits/stork_split_seed42.json` and the corresponding `larus_fuscus`, `cervus_elaphus`, `canis_lupus`, `mirounga_leonina`, and `mirounga_angustirostris` `_split_seed42.json` files.

## 2. Pairwise distances

Distances are Euclidean distances over the eight standardised descriptors. Standardisation is frozen in `results/regime_descriptor_standardisation.json`, with descriptor values in
`results/regime_profiles.csv`.

|                  | stork | *L. fuscus* | *Cervus* | wolf | *M. leonina* | *M. ang.* |
| ---------------- | ----: | ----------: | -------: | ---: | -----------: | --------: |
| **stork**        |  0.00 |        5.10 |     6.68 | 5.70 |         6.77 |      6.99 |
| ***L. fuscus***  |  5.10 |        0.00 |     3.65 | 1.65 |         3.31 |      3.72 |
| ***Cervus***     |  6.68 |        3.65 |     0.00 | 2.65 |         3.42 |      3.39 |
| **wolf**         |  5.70 |        1.65 |     2.65 | 0.00 |         3.66 |      3.66 |
| ***M. leonina*** |  6.77 |        3.31 |     3.42 | 3.66 |         0.00 |      1.42 |
| ***M. ang.***    |  6.99 |        3.72 |     3.39 | 3.66 |         1.42 |      0.00 |

For the source pool of stork, *Cervus*, and *M. leonina*, the mean pairwise distance is 5.62 and the minimum is 3.42. Note we outlined 1 to use the source pool to maximise descriptor space coverage. Within domain pair distances serve a different purpose. Their range are 5.10, 2.65, and 1.42 allows secondary prediction 2 to test whether transfer varies with source target distance.

## 3. Our reasons

* **Flying.** Stork is fixed as our source species by 1. *L. fuscus* is the largest remaining profiled flying candidate, with 330 individuals above the evaluation floor. It is also very different from stork in step scale, with a median step length is 4,115 m versus 611 m, while the corresponding tail ratios are 27 and 234.

* **Terrestrial.** *Cervus* and wolf provide a predator–prey comparison while keeping device type and region similar. Both datasets are GPS collars from Alberta and British Columbia. 

* **Marine.** Both species are *Mirounga*, which we did note already as a limitation of this pair. The alternatives are worse on the same measure. Once every candidate was taken
  through QC, Weddell seal sits at 1.00 from SOES, the closest of any marine species and closer than NES at 1.42, and it also shares RAATD Southern Ocean data with SOES, making dataset and region more similar. Walrus is likewise closer than NES at 1.43. Crabeater seal is more distant at 4.09 but holds only 41 eligible individuals, and walrus 53. SOES and NES differ geographicall; since their median latitudes are −59.1° and +43.7°, they occur in different hemispheres and ocean basins, and NES is outside RAATD.

## 4. Declared deviations and confounds

**Terrestrial ordering differs from two registered tie breaks.** Note that 1 favours the species with the larger eligible test pool on the held-out side; *Cervus* has 323 eligible individuals compared with 110 for wolf. When we wrote 6, we favoured placing the better characterised dataset on the held-out side (pros and cons for this, but this ensures we can develop a sufficient S model for testing). 

So our selected ordering is *Cervus* as source and wolf as held-out because source-pool size changes substantially under the alternative. With wolf as the source, terrestrial data would contribute about 5% of source pool windows; 62k of 1.24M. With *Cervus* as the source, it contributes about 34%: 594k of 1.77M. And so the former would make the pretraining corpus much more heavily weighted toward the flying and marine domains.

Descriptor coverage gives similar results under both orderings. The selected ordering gives a source pool mean/minimum distance of 5.62/3.42, compared with 5.38/3.66 for the reversed ordering. The reversed ordering therefore has a slightly larger minimum distance.

**Consequence.** The terrestrial specialist S trains on 99 individuals and 62,013 windows, about one tenth of the windows available from *Cervus*. S is the oracle used in
`p90_travel(Z) <= 1.25 x p90_travel(S)`. A weaker specialist raises this threshold and therefore makes the STRONG band easier for Z to satisfy. So this direction of bias does favour our study hypothesis and will be reported whenever we claim the STRONG band.

**Marine pair uses different device technologies.** SOES is entirely Argos based. NES carries both Argos and Fastloc GPS, with location class G on 238,315 of its fixes. So marine transfer performance could reflect both species differences and device differences.

