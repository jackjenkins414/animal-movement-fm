# Preregistration: Tier 3 source pool scaling

**Date: 16-09-2026. Draft; only sec 1 is complete.** This file will become the Tier 3 component of `preregistration_tier2.md`. Tier 3 differs from Tier 2 in one variable, the size and composition of the source pool (three sources growing to nine), and holds everything else fixed: the same three held-out species and frozen splits, metrics, success bands and protocol (`preregistration_tier2.md` §1). Sections beyond §1 are still to be written and will be completed before any Tier 3 model is trained.

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

**Congeners of held-out species are allowed and declared.** *Canis latrans* shares a genus with the held-out *Canis lupus*. It remains a candidate, and if selected the link is declared as a confound for the terrestrial result, as the *Mirounga* pair already is in Tier 2. (*Larus argentatus* was excluded at ingest for Tier 3's flying candidates; that exclusion predates this rule and is not revisited.)

**Distance.** Euclidean distance over the eight gate 2 descriptors, each transformed and standardised with the **frozen Tier 2 standardisation** `registered/tier2_descriptor_standardisation.json` (A10 transforms, A16 freeze). One fixed metric is used across tiers rather than refitting over the Tier 3 table. That standardisation was fitted over the 21 species as they stood at the Tier 2 freeze, before A15's QC rebuilt the candidates, so its centres and scales reflect pre-QC marine tail ratios; that is a declared property of the metric, not a reason to refit it.

**Objective.** Among all admissible pools, choose the one with the **largest mean of the 36 pairwise distances** among its nine species. Ties (difference $\leq$ 1e-9) go, in order, to:

1. the larger minimum pairwise distance
2. the larger total training window count across the nine
3. the pool whose sorted species names come first alphabetically

Model performance is never used, and neither is any held-out species' descriptor row or distance.

## GenAI reference

Section 1 was drafted with GenAI from decisions I made but asked GenAI to confirm; nested pool, mean distance objective with the minimum as the first tie break, the frozen Tier 2 metric, a 2,000 window floor, and congeners allowed but declared.
