# Preregistration: Tier 2 cross-species generalisation

**Date: 27-07-2026.**

This purpose of our writing here is to fix the evaluation protocol, the metrics, and what will count as success, in advance of running anything. Ultimately, we decide here how the experiment will be judged before we know whether the model succeeds or not. It exists because Tier 2 evaluates only a small number of held out species, so its estimated performance will be sensitive to the particular species, datasets, and training runs included. Fixing the criteria beforehand prevents us from selecting (after seeing the results) whichever metric or framing presents the most favourable outcome. 

### Literature basis

We follow the logic of preregistration for predictive modelling described by Hofman et al. (2023) (arXiv:2311.18807v1), where the prediction problem, data partitions, transformations, evaluation metrics and baselines are all declared before we observe final test performance (even before we begin the modelling). This is important because a conventional train/validation/test split does not by itself remove researcher degrees of freedom in how a prediction task is framed, tuned and reported.

Specifically, Hofman et al. propose a **two-phase** protocol, given by: 

- Phase (A), completed before any model training or validation, defines the problem, which is the research question, variables, how the training and test sets are constructed, planned transformations, metrics and baselines. 
- Phase(B), completed after model selection but *before testing*, records the finalised model; the prediction method, training details (seeds, hyperparameters, computational budget), whether the test data has previously been accessed, and any adjustments to phase (A). Sections 1–4 of this document are our phase (A); the final pre-test freeze in (2) is our phase (B).

Hofman et al. also draw a line between exploring the *properties of a dataset* (such as structure, missingness, feature ranges), which is permitted before phase (A), and analysis that models relations between features and outcomes, which is not. Our species selection procedure (`experiments/data/regime_profile.py`) (this script is uncommitted but almost finished) computes descriptive movement statistics only; such as step scale, intermittency, directional persistence, diel and seasonal amplitude, and fits nothing.
Thus it is simply a dataset property exploration under that distinction. 

We also follow the Adaptive Preregistration approach proposed for ecological model based research by Gould et al. (2026) (DOI: 10.1111/2041-210x.70311), which targets the failure modes most relevant here: *model fishing* (trialling multiple model structures and reporting only the best fitting of those models), *s-hacking* (testing several performance metrics and selectively reporting the favourable ones), and *sample curation* (dropping inconvenient sites or periods). Decisions that can be fixed now are done so here; anything genuinely data dependent is documented in a dated amendment rather than introduced silently after seeing the result.

**Amendment policy.** Adjustments are permitted; silent adjustments are not. Every post commit change is recorded in **Amendments** with its date, reason and status relative to test access. Superseded wording is either quoted there or identified by the exact pre-change commit, whose diff preserves it. And obviously a change made after seeing a sealed result must say so explicitly. Unrecorded changes to success criteria would invalidate the whole purpose of us doing this.

**Correction rule.** If a defect is discovered during or after a sealed evaluation (such as a data bug, a broken metric, a mis-specified split) a correction is permitted **only if it is independent of the direction of the result**, (it would have been made identically had the numbers come out the other way). Such corrections are logged as amendments, with the defect, the fix, and the affected numbers before and after. Changes motivated by the result itself are not corrections and we will not employ them. 

## 1. The claim under test

A movement forecasting model pretrained on multiple source species generalises zero-shot to a new species: Z beats persistence on that species' sealed test partition without using any of that species' data for Z pretraining or model selection.

**Tier 2 pool: six species, in two fixed and disjoint roles.**

| role | count | composition |
| --- | --- | --- |
| **source** | 3 | one per domain: flying, terrestrial, marine. White stork is the flying source (our existing corpus). |
| **held out** | 3 | one per domain: flying, terrestrial, marine. Excluded from Z pretraining and model selection; test targets remain sealed until final evaluation. |

No species does double duty; a source species is never held out, and a held out species is never a source.

**Why six rather than three.** With only three species evaluated leave one out, holding out the marine species leaves nothing marine in training. A failure could then be taxonomic distance, or it could simply be that the model never saw that domain at all, and we would have no way to separate the two. With one source species per domain, every held out species has its own domain represented in training *by a different species*, so the test is genuinely "a new species" rather than "a new domain". The domain question does not disappear; we ask it separately and deliberately, through the ablation arm in Section 2.

**The held out set is fixed across tiers.** The same three held out species carry into Tier 3 unchanged, along with their frozen splits. Tier 3 then differs from Tier 2 in exactly one variable: the size and composition of the source pool (three species, one per domain, growing to nine, three per domain). Everything else is held constant, being the same test species, the same test individuals, the same metrics and the same protocol. That makes our two tiers a single controlled scaling study rather than two disconnected experiments, and it lets us report held out performance as a function of source pool size. There is no established law for how many source domains a foundation model needs, so this is the axis we are best placed to measure.

Our selection of which species fill which role proceeds through two separate gates, in this order, and they must not be blended:

1. **Eligibility: a data quality filter.** Applied before regime profiling. A species must retain at least 20 training, 10 validation and 10 test individuals after continuity filtering; training individuals need at least one valid window and validation/test individuals at least 100. Roles are frozen after this filter; a selected species that later becomes unusable is not replaced.
2. **Selection: the scientific criterion.** Among eligible species, choose one source and one held-out species per domain to maximise movement regime coverage using the fixed descriptors from `experiments/data/regime_profile.py`. Before modelling, commit the candidate table, pairwise descriptor distances, chosen roles and reasons. Ties are resolved by better device/region matching within domain, then larger eligible test count; model performance is never used.

An important note on keeping these separate matters: selecting on cadence feasibility alone would be exactly the convenience driven choice this document is meant to rule out, whereas excluding a species that cannot form windows at all is a precondition, not a preference.

**Pairing within a domain.** For each domain, the source and the held out species should differ in *species identity* and, as far as the data permits, not in device technology or geographic region. Where that cannot be achieved, the difference is recorded explicitly for that pair so a transfer result can be attributed rather than merely observed. Our marine pair is the known risk:, since if both marine species come from RAATD then both are Southern Ocean, and "marine" and "polar" become the same variable on both sides of the comparison. Drawing the held out marine species from a different ocean basin (for example Movebank Argos walrus, Arctic) is the intended mitigation.

**Fallback, declared in advance.** If six species have not cleared eligibility, this design reverts to sealed leave one species out over three species, one per domain; each species held out in turn and trained on the other two, giving three sealed results and no ablation arm. Taking the fallback will be recorded as a dated amendment. Registering it now is the point; choosing between two designs is something we will know after we begin our analysis. 

## 2. Protocol (fixed in advance)

**Outer loop: sealed evaluation.** Z is pretrained and selected using source species only, then evaluated on each held-out test partition. Before sealing, test data may undergo only registered model independent ingest, QC and continuity/window count checks. After the manifests and preprocessed data are frozen, test targets and model performance remain unseen until final evaluation. Held-out training/validation data may be used for descriptive selection, the preregistered regime threshold, S and F; they cannot influence Z.

**Domain ablation arm.** For each held-out species, Za is Z retrained without the source species from the matching domain and evaluated on that held-out species only. For example, marine Za is trained on the flying and terrestrial sources. Z versus Za tests the value of adding the matching-domain source, although it cannot separate domain coverage from the accompanying increase from two to three source species.

**Inner loop: pseudo targets.** Hyperparameters are selected by holding out each source species in turn as a pseudo target. Before the first tuning run, a committed `preregistration_tier2_tuning.md` will fix the configuration budget, selection metric, rule for combining pseudo target results and tie break. The selected FM configuration is then frozen for Z, Za, S and F; B's conventional architecture and training specification are fixed in the same record. No unregistered configuration may enter the sealed evaluation.

**Leakage controls.**

- Split individuals FIRST, then window. Windowing before splitting leaks near identical 24 fix contexts across train and test. We do already do this, but its stated here because it is the primary leakage mode in this literature.
- After resampling, construct modelling windows only within runs of occupied bins whose `bin_index` increases by exactly one. The complete context plus target window must therefore be contiguous at the shared modelling cadence; a window cannot cross even a tolerated internal gap within a coarse segment (this was a previous bug we had).
- Apply the continuity rule to both model training and evaluation. A training individual remains eligible if it contributes at least one valid contiguous window. Validation and test individuals must contribute at least 100 valid contiguous windows because their per individual estimates would otherwise be too noisy. These filters are applied to the already frozen manifest: individuals are never reassigned, replaced or drawn from another split to restore its size.
- Group by individual, which subsumes deployment, since an individual's deployments cannot be separated by an individual level split.
- Splits are read from committed manifests (`splits/*.json`), never recomputed.
- Any QC threshold derived from data (speed quantiles, error calibration) is computed on **train only** or strictly per track. Corpus wide thresholds leak test information into preprocessing.

**Robustness checks.** Blocked by study and blocked by region evaluation inside species, to show that a "cross-species" result is not a geography or tag settings shortcut.

### Final pre-test freeze

After source species training and pseudo target selection, but before any sealed test result is examined, a final dated record will freeze the model specification. This is our phase (B) in the sense of Hofman et al. (2023).

**The freeze is another committed file, `preregistration_tier2_freeze.md`,** which we will commit before the first sealed evaluation is run. It will record:

- final architecture, layer/width configuration and parameter count
- selected hyperparameters and the number of inner loop configurations actually evaluated, against the declared budget
- random seeds and the checkpoint selection rule
- computational budget consumed
- **an explicit, dated attestation that after the registered mechanical preprocessing and sealing, no held-out test target, plot, prediction or performance result was accessed**
- any difference from this document, listed item by item

## 3. Metrics (fixed in advance)

Our primary error is **great circle (geodesic) error in metres** on the held out species' test windows, reported as **median, p90, p99**; recall we found heavy right skew makes the mean uninformative.

Reported **stratified by regime**, where the regime boundary is scale free and species specific: a step is stationary if it is below 10% of that species' own median step, where the median is computed on that species' training individuals only, over steps spanning **exactly one bin** of our common modelling cadence after resampling. Requiring exactly one bin matters since segments tolerate internal gaps of up to 8 bins, and those longer steps are systematically larger, so admitting them inflates the median and loosens the threshold. `regime_profile.py` implements this definition; on white stork's training individuals it gives a threshold of 60.9 m against the 60 m hysteresis cut we used in Tier 1, and a stationary fraction of 31.1% against the ~33% derived independently there. `regime_profile.py` will give the production figure. If our implementation and this text ever disagree, this text governs (if we encounter a bug for example).

We will also report:

- **persistence skill** = 1 − model_error / persistence_error, per regime and per species
- **paired 95% percentile bootstrap intervals**, using 2,000 resamples of test individuals with replacement and seed 42; paired models use the same resamples

The forecast is one step, so ADE and FDE both equal the geodesic error and are not reported as separate metrics. For trained models, the headline metric is the median across the fixed training seeds. "Beats" means a strictly lower headline point estimate; confidence intervals and seed spread are reported but do not create a second pass/fail rule.

**Comparators.**

| Symbol | Model |
| --- | --- |
| P | persistence: predict no movement |
| Z | zero-shot: the full FM, pretrained on all three source species and evaluated unchanged on a species completely unseen by Z. Tests whether knowledge transfers across species. |
| Za | domain-ablated zero-shot: Z retrained without the source species matching the held-out species' domain. Tests the value of adding matched-domain source data. |
| B | pooled-source baseline: trained on the same source pool as Z but stripped of the FM-specific package (like pretraining objective, species conditioning and specialised architecture)|
| S | specialist: the same architecture and frozen configuration as Z, trained only on the held-out species' own training individuals. An oracle target species benchmark. Defined precisely below. |
| F | fine-tuned: Z further trained on 1%, 10% or 100% of the held-out species' training individuals. Measures how target data changes the value of pretraining. Defined precisely below. |

### On the specialist (S)

The 25% threshold in Section 4 is defined against S, so leaving S vague would leave a numeric success criterion resting on an ambiguous comparator. So we fix the following now:

- **Architecture and configuration.** S uses the same architecture and the same frozen configuration as Z, taken from the pre-test freeze. The only thing that differs between Z and S is the data each was trained on, so any gap between them is attributable to training data rather than to model capacity or tuning effort.
- **Training data.** Only the held out species' own training individuals, read from that species' frozen split manifest. Never the test partition.
- **Seeds.** The same number of training seeds as Z, so the comparison does not inherit some variance asymmetry.
- **Tuning privilege, stated openly.** S may early stop and select checkpoints on the held-out species' validation individuals. Z cannot. This deliberate advantage makes S an oracle style target species benchmark, although not a guaranteed mathematical upper bound. S receives no species specific hyperparameter search; its configuration was frozen in the source-only inner loop.

### Fine-tuning (F), and why it is here

Tier 2 as originally written tested zero-shot only, but one of the things we are investigating in our work is pretrain **then finetune**, and some of the transfer literature we have reviewed thus far reports that a specialist can overtake a zero-shot model with as little as ~2% of target data. Zero-shot is therefore the hardest single point on a curve rather than the whole claim, so we register the curve.

- **Definition.** F_k is Z, further trained on a fraction k of the held out species' training individuals.
- **Fractions, which we fix in advance.** k $\in$ {1%, 10%, 100%} of the held out species' training individuals. We log space them deliberately; the fine-tuning curve is steep at the low end and flat near the top, and the transfer literature puts the crossover at around 2% of target data, so densifying the 25–100% stretch would buy resolution in the region we expect to be flattest.
- **Repeat draws at the low end.** At small k the subset is tiny in absolute terms (1% of the stork's 630 training individuals is 6 birds, and fewer on the other two species), so *which* animals are drawn will move the result more than the fine-tuning does. We therefore run **three independent draws at k = 1% and three at k = 10%**, and a single run at k = 100% since there is nothing left to vary. That totals to 7 fine-tuning runs per held out species.
- **Floor on subset size.** A fraction must yield at least **5 individuals**; where it does not, we use 5. Otherwise "1%" on a species with few training individuals is two or three animals and the label claims more than the data supports. The realised individual count and window count are reported alongside every k, since the same fraction means different absolute amounts of data on each species.
- **Sampling and nesting.** Sampling is by *individual*, never by window, drawn from that species' frozen split manifest with a fixed seed per draw. Each draw is nested: its 1% subset sits inside its 10% subset, which sits inside the full training set. This makes the comparison across k *within* a draw paired rather than independent, so the curve is not confounded by subset composition changing between points.
- **Configuration.** The same architecture and frozen configuration as Z and S; the fine-tuning budget (epochs, learning rate schedule etc) is declared in our pre-test freeze. F may early stop on the held-out species' validation individuals, on the same terms as S, so that F and S remain directly comparable.
- **Sealing is unaffected.** F touches training individuals only. The held out species' test partition stays sealed for every model in this document.
- **Reporting.** A curve of travel regime p90 against k, per held out species, with Z as the k = 0 point and S as the asymptote. At k = 1% and k = 10% we report the **spread across the three draws** (median and range) rather than a single number, because a point estimate at those subset sizes would not be interpretable. This spread is a different quantity from the paired bootstrap intervals in Section 3: those resample *test* individuals, whereas the draw spread measures sensitivity to which *training* animals happened to be selected.

The comparison doing the most work here is **F_100 against S**: both see all of the held out species' training data, and they differ only in whether the weights were pretrained on the other species first. That isolates the value of pretraining with training data held constant, which is honestly the cleanest statement of what a foundation model is supposed to buy us (hopefully). 

**This does not move the goalposts.** The success bands in Section 4 are defined on Z alone and are unchanged by the addition of F. A good fine-tuning curve does not rescue a WEAK zero-shot result and we will not report it as though it does. F is registered here, before any Tier 2 run, precisely so that adding it cannot later be mistaken for widening the reporting surface after the fact.

## 4. Success bands

We apply these bands in order: Strong first, then Acceptable; every remaining result is Weak/Null. "Travel regime" means steps at or above the stationary threshold defined above. Pooled performance cannot elevate a result that fails the species-specific travel criteria.

**STRONG.** For **every** held out species: Z beats P on travel regime p90 **and** p99 (persistence skill > 0 on both), and Z is within 25% of the specialist on travel regime p90; which we can state as follows to remove ambiguity:

    p90_travel(Z)  <=  1.25 x p90_travel(S)

**ACCEPTABLE.** If Strong is not met, Z beats P and B on travel regime p90 for **all three** held-out species. Z may fail on p99 or trail S by more than 25%.

**WEAK / NULL.** Every other result, including improvement only on pooled metrics or inconsistent travel effects across species.

A WEAK outcome will be reported as a null result with its mechanism, in the same form as our Rung 2 state head null. A null here is informative because it would say that movement primitives do not transfer across taxa at this cadence and data scale...

## 5. Secondary preregistered predictions

1. **Trait conditioning: deliberately NOT registered as a Tier 2 prediction.**

   The natural prediction would be: trait conditioning helps zero-shot but not within species, because trait and taxonomic metadata are constant within a species and so carry no
   within species information; an ablation that improved *both* would indicate a confound rather than a mechanism. That prediction is **not testable at n = 3** and so we register it as a **Tier 3** prediction, not a Tier 2 one.

   The reason: with three source species, the trait vector takes only **three distinct values** across the whole of training (two, in the domain ablated runs). It is therefore perfectly collinear with species identity, and a learned projection over three distinct inputs is a three point lookup; "trait conditioning" and "learned species embedding" are empirically indistinguishable at this n, whatever the descriptors are (traits, taxonomic rank, or text derived taxon vectors as in MoveFormer). Testing it anyway would yield a three point extrapolation dressed as a mechanism. Note that moving to six species does not help here, because the three held out species contribute no training signal at all; what matters is the number of distinct conditioning vectors *seen during pretraining*, which is the size of the source pool. This is the specific thing Tier 3 fixes by taking the source pool to nine.

   The conditioning mechanism will still be implemented and its effect reported **descriptively**, with no success or failure claim attached.

   *Conditional registration:* individual level traits (e.g. mass at deployment) vary **within** a species and so break the collinearity. If any become available for a source species, a prediction will be registered here as a dated amendment **before** those traits are used.

2. **Transfer will be uneven across the three held out species**, tracking regime distance from the source pool as measured by `regime_profile.py`, rather than being uniform.
3. **Persistence remains hard to beat on stationary steps** for every species. Zero-shot gains, if any, will be concentrated in the travel regime.
4. **Pretraining helps even when the held out species' full training data is available.** F_100 beats S on travel regime p90 for at least two of the three held out species. If instead F_100 is indistinguishable from S everywhere, then pretraining contributes nothing beyond initialisation once target data exists and its value is confined to the low data regime, which is a reportable result rather than a failure. If F_100 is consistently *worse* than S, that is negative transfer, which the literature reports is common, and we will report it as such rather than bury it.
5. **Adding a matching domain source helps.** Z beats Za on travel p90 for at least two of the three held-out species. Because Z also has one more source species, this supports the value of the matched domain addition but does not isolate domain coverage from source-pool size.

## 6. Declared limitations (stated before results, not after)

- **n = 3 held out species.** Six species are involved but only three are ever evaluated, so this remains an existence proof rather than a transfer study. It is too few for the transfer matrix and negative transfer auditing the literature recommends; those need Tier 3 scale. So no claim of general cross-taxa transfer will be made from three held out species.
- **Source and held out roles are disjoint, which is deliberate but costly.** Six species of ingest and QC buy three sealed results, where leave one out would have bought three from three. We accept that cost for the domain coverage and the fixed held out set. The consequence is that there is no rotation available: if a held out species turns out to be unusable after ingest, that result is lost rather than recoverable by reassignment, so the better quality data should sit on the held out side.
- **The fixed held out set propagates.** Because the same three species are carried into Tier 3, any way in which they are unrepresentative is inherited by both tiers rather than averaged out across them. The scaling comparison is clean precisely because the test set does not change, and that is the same reason a poor choice of test species cannot be corrected later without breaking the comparison.
- **The domain ablation is n = 1 per domain and changes pool size.** Z versus Za gives one comparison per domain and changes the source pool from two species to three. It tests the value of adding the matched domain source, not a pure domain effect, and remains a secondary prediction.
- **Single data split** per species (frozen manifests) and a small number of training seeds; effects smaller than our seed spread measured in the 22-07-2026 multi-seed sweep will not be treated as resolved.
- **Confounded taxon and technology.** The marine species are Argos while the others are GPS, so "marine" and "different error model" cannot be fully separated. With two marine species this confound now sits on both sides of the marine pair, which is why the pairing rule in Section 1 asks for the difference to be recorded per pair.
- Tier 1 results were computed in degree space; Tier 2 will use local ENU metres (currently in progress right now). The two are bridged by an explicit representation ablation rather than assumed comparable.

### Versioning and reporting

This document and the pre-test freeze are version controlled. Operative sections are kept current, while each amendment preserves the superseded wording and records the date and reason. The final report will identify preregistered analyses and every deviation.

## Amendments

Appended in order. Each entry gives the date, reason, test-access status and either the superseded wording or its exact commit.

### A1; 28-07-2026: stationary threshold definition disambiguated, stork figures corrected

**Status when made:** no Tier 2 model had been trained and no sealed evaluation had been run, so this is a correction under our outlined rules above rather than a change of criteria. It is independent of the direction of any result, because no result exists.

**Reason.** We previously worded that the median step is computed "on tracks resampled to the common modelling cadence", which is somewhat ambiguous. Resampled segments tolerate internal gaps of up to 8 bins, so "a step in a resampled track" could mean any consecutive pair within a segment, or only pairs spanning exactly one bin. The two readings give different answers, and the multi bin steps are systematically longer, so admitting them inflates the median and loosens the threshold:

| reading | steps used | median step | 10% threshold | stationary fraction |
| --- | ---: | ---: | ---: | ---: |
| any consecutive pair in a segment | 916,002 | 625.9 m | 62.6 m | 30.8% |
| exactly one bin apart | 879,559 | 609.5 m | 60.9 m | 31.1% |

(white stork, training individuals only, frozen manifest `splits/stork_split_seed42.json`)

The previous figure we quoted of 62.6 m corresponded to the reading we are *not* implementing. Left uncorrected, `regime_profile.py` would have produced 60.9 m against a document asserting 62.6 m, and since this text governs the implementation, the implementation would have been deemed wrong for computing the intended quantity. The exactly one-bin reading is also the correct one on its merits, since a step spanning three bins is not a step at the modelling cadence.

**Superseded text:**

> Reported **stratified by regime**, where the regime boundary is scale free and species specific: a step is stationary if it is below 10% of that species' own median step, where the median is computed on that species' training individuals only, on tracks resampled to the common modelling cadence. `regime_profile.py` implements this definition; on white stork's training individuals it gives a threshold of 62.6 m against the 60 m hysteresis cut we used in Tier 1, and a stationary fraction of 30.8% against the ~33% derived independently there. `regime_profile.py` will give the production figure. If the implementation and this text ever disagree, this text governs (if we encounter a bug for example).

**What changed.** The definition now specifies steps spanning exactly one bin, with a sentence explaining why that matters; the stork figures are updated from 62.6 m / 30.8% to 60.9 m / 31.1%. Nothing else in Section 3 changed, and no metric, comparator, success band or secondary prediction was altered.

### A2; 29-07-2026: modelling windows made fully contiguous; split eligibility clarified

**Status when made:** no Tier 2 model had been trained and no sealed Tier 2 evaluation had been run. The defect was found while auditing the data pipeline before Tier 2. Existing Tier 1 real data results had already been produced, so those results are treated as provisional as of now and will be recomputed; the synthetic ladder is unaffected obviously.

**Reason.** The resampler deliberately preserves internal gaps of up to 8 bins inside a coarse `segment_id`. Window construction grouped only by that identifier, so a 24-fix context and its target could silently cross a missing 6-hour bin. The model therefore treated some 12–48-hour displacements as if they were single steps at the shared cadence. On the frozen stork test split, 3,997 of 121,656 targets (3.29%) were not one bin targets and 14,922 windows (12.27%) contained a gap in their history or target.

The correction preserves each coarse segment but derives contiguous runs within it, splitting whenever `diff(bin_index) != 1`, and constructs windows only inside those runs. This is applied structurally to training, evaluation and state datasets. The frozen manifest remains authoritative: it is not regenerated or refilled after corrected counts are known. Training requires at least one valid contiguous window because any such window supplies valid supervision; validation and test retain the 100 window floor because it protects the stability of per-individual evaluation estimates.

Under the corrected rule, the frozen stork manifest contains 625 of 630 eligible training individuals, 100 of 105 eligible validation individuals, and 91 of 105 eligible test individuals, contributing 774,627, 139,714 and 105,862 windows respectively.

**Superseded text:**

> - Split individuals FIRST, then window. Windowing before splitting leaks near identical 24 fix contexts across train and test. We do already do this, but its stated here because it is the primary leakage mode in this literature.
> - Group by individual, which subsumes deployment, since an individual's deployments cannot be separated by an individual level split.
> - Splits are read from committed manifests (`splits/*.json`), never recomputed.

**What changed.** The leakage controls now require complete one bin continuity across every context + target window, apply the rule to both training and evaluation, distinguish training inclusion from evaluation eligibility, and state explicitly that the frozen split is filtered rather than recomputed or replenished. This correction does not alter the Tier 2 species roles, models, metrics, comparators, success bands or regime-threshold rule.

### A3; 03-08-2026: Tier 2 decision rules completed

**Status when made:** no Tier 2 model had been trained and no sealed test result had been accessed.

**Reason.** Our original protocol left several choices open: it conflated a held-out species with its sealed test partition, did not fix species level eligibility or the pre-tuning decision record, left B ambiguous, left some outcomes outside the success bands, did not state how seeds determined "beats", and overstated what Z versus Za isolates. These are now fixed.

**Superseded wording:** the affected certain parts of of Sections 1–6 and the amendment policy are preserved in the previous commit to this file (add that later). A1 and A2 above separately govern the regime threshold and continuity corrections.

## GenAI Reference 

I initially used GenAI by giving it this preregistration, which I had written myself, and asking whether the success criteria seemed reasonable for the scope of my project (which I outlined in the chat). It agreed that the criteria was appropriate, but pointed out one issue I had overlooked; trait conditioning is not meaningfully testable in Tier 2 because the model sees only three source species, and therefore only three distinct trait vectors. At that scale, the traits are perfectly confounded with species identity, so the model could simply use them as a species lookup rather than learn a genuinely transferable relationship between biological traits and movement.

I had initially assumed that because traits describe meaningful biological differences, they should naturally help zero-shot transfer. GenAI’s suggestion made me realise that the problem was not whether the traits were biologically relevant, but whether the experiment could distinguish trait based generalisation from species memorisation. I then developed and wrote the Tier 2 limitation and Tier 3 justification myself, including the decision to report trait conditioning descriptively in Tier 2 rather than treat it as a formal prediction.