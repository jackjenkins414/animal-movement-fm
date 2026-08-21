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

1. **Eligibility: a data quality filter.** Applied before regime profiling. Splitting is **split-specific**, fixed by Amendment A7: validation and test individuals are drawn from the pool clearing the evaluation window floor F, set to 25 by Amendment A5, and every remaining individual holding at least one valid window goes to training. A species is eligible if that pool holds at least 20 individuals and at least 40 hold at least one valid window, which yields the required 20 training, 10 validation and 10 test individuals. Roles are frozen after this filter; a selected species that later becomes unusable is not replaced.
2. **Selection: the scientific criterion.** Among eligible species, choose one source and one held-out species per domain to maximise movement regime coverage using the fixed descriptors from `experiments/data/regime_profile.py`. Before modelling, commit the candidate table, pairwise descriptor distances, chosen roles and reasons. Ties are resolved by better device/region matching within domain, then larger eligible test count; model performance is never used.

An important note on keeping these separate matters: selecting on cadence feasibility alone would be exactly the convenience driven choice this document is meant to rule out, whereas excluding a species that cannot form windows at all is a precondition, not a preference.

**Pairing within a domain.** For each domain, the source and the held out species should differ in *species identity* and, as far as the data permits, not in device technology or geographic region. Where that cannot be achieved, the difference is recorded explicitly for that pair so a transfer result can be attributed rather than merely observed. Our marine pair is the known risk:, since if both marine species come from RAATD then both are Southern Ocean, and "marine" and "polar" become the same variable on both sides of the comparison. Drawing the held out marine species from a different ocean basin (for example Movebank Argos walrus, Arctic) is the intended mitigation.

**Fallback, declared in advance.** If six species have not cleared eligibility, this design reverts to sealed leave one species out over three species, one per domain; each species held out in turn and trained on the other two, giving three sealed results and no ablation arm. Taking the fallback will be recorded as a dated amendment. Registering it now is the point; choosing between two designs is something we will know after we begin our analysis. 

## 2. Protocol (fixed in advance)

**Outer loop: sealed evaluation.** Z is pretrained and selected using source species only, then evaluated on each held-out test partition. Before sealing, test data may undergo only registered model independent ingest, QC and continuity/window count checks. After the manifests and preprocessed data are frozen, test targets and model performance remain unseen until final evaluation. Held-out training/validation data may be used for descriptive selection, the preregistered regime threshold, S and F; they cannot influence Z.

**Domain ablation arm.** For each held-out species, Za is Z retrained without the source species from the matching domain and evaluated on that held-out species only. For example, marine Za is trained on the flying and terrestrial sources. Z versus Za tests the value of adding the matching-domain source, although it cannot separate domain coverage from the accompanying increase from two to three source species.

**Inner loop: pseudo targets.** Hyperparameters are selected by holding out each source species in turn as a pseudo target. Before the first tuning run, a committed `preregistration_tier2_tuning.md` will fix the configuration budget, selection metric, rule for combining pseudo target results and tie break. The selected FM configuration is then frozen for Z, Za, S and F; B's conventional architecture and training specification are fixed in the same record. No unregistered configuration may enter the sealed evaluation.

**Modelling window.** All species share one modelling cadence of 6 hours and one window shape. The context length is now **C = 12** history bins, fixed by our Amendment A6, so a valid window spans **13 consecutive 6h bins**: 12 observed positions and the one step target. Windows per individual are therefore sum over maximal contiguous runs of $max(0, L − C)$, summed across runs, which means the context is subtracted once per run rather than once per individual. This applies identically to every species, every comparator and every tier; a change to C is an amendment, not a hyperparameter, because it changes both the eligible window population and the counts on which eligibility is judged.

**Sampling.** Pretraining windows are drawn hierarchically, **species → individual → window**: a source species is drawn with probability proportional to the square root of its training partition window count at the registered context length, and a window is then drawn uniformly from that species' training windows, so individuals are weighted by the number of windows they contribute. The weights are fixed numerically in **Amendment A8**, revised for the rebuilt marine data by **Amendment A9**, and are excluded from the configuration budget, so they are never tuned. Draws are with replacement and training length is declared in optimiser steps rather than epochs. Validation, pseudo target scoring and sealed evaluation are **not** sampled: every eligible window is scored exactly once, and the weighting in Section 3 governs what is reported.

**Leakage controls.**

- We split individuals FIRST, then window. Windowing before splitting leaks near identical context windows across train and test, since adjacent windows differ by only a single fix. We do already do this, but its stated here because it is somewhat the primary leakage mode. 
- After resampling, construct modelling windows only within runs of occupied bins whose `bin_index` increases by exactly one. The complete context plus target window must therefore be contiguous at the shared modelling cadence; a window cannot cross even a tolerated internal gap within a coarse segment (this was a previous bug we had).
- Apply the continuity rule to both model training and evaluation. A training individual remains eligible if it contributes at least one valid contiguous window. Validation and test individuals must contribute at least **25** valid contiguous windows, the evaluation window floor F fixed by Amendment A5, because their per individual estimates would otherwise rest on a handful of forecasts. These filters are applied to the already frozen manifest; individuals are never reassigned, replaced or drawn from another split to restore its size.
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

**Weighting, fixed by Amendment A5.** Species median and p90 are computed from the **individual weighted** empirical distribution, not by pooling every window, so that the longest deployments cannot dominate a species' reported error. **p99 is reported at species level only and never per individual**, because no per individual window floor we could impose without emptying the split would support a p99. Every individual entering these statistics clears the evaluation window floor F = 25; sensitivity at F $\in$ ${10, 25, 50, 100}$ is reported too alongside.

Reported **stratified by regime**, where the regime boundary is scale free and species specific. The two strata are called **low displacement** and **travel**; they are a measurement boundary on step length, not a claim about biological stationarity or behavioural state.

The boundary is a **single universal formula** applied identically to every species, with the anchoring quantile selected mechanically by the decision tree registered in **Amendment A4**. The default rule is `0.10 × p50`, where p50 is the median step of that species' training individuals over steps spanning **exactly one bin** of our common modelling cadence after resampling. Requiring exactly one bin matters since segments tolerate internal gaps of up to 8 bins, and those longer steps are systematically larger, so admitting them inflates the median and loosens the threshold. If the registered guards fire, the anchor moves to p75 and then p90 **for every species at once**; the data may select the anchor, but it may never select a different formula per species. On white stork's training individuals the default rule gives a boundary of 60.9 m against the 60 m hysteresis cut we used in Tier 1, and a low displacement fraction of 31.1% against the ~33% derived independently there. `regime_profile.py` implements this and will give the production figure. If our implementation and this text ever disagree, this text governs (if we encounter a bug for example).

For Argos derived species the low displacement class may quite possibly reflect the observation error floor rather than genuine near immobility. Because this result may arise almost automatically from Argos measurement error in the marine data, Secondary Prediction 3 will be treated as a methodological limitation rather than evidence of genuine biological behaviour.

We will also report:

- **persistence skill** = 1 − model_error / persistence_error, per regime and per species
- **paired 95% percentile bootstrap intervals**, using 2,000 resamples of test individuals with replacement and seed 42; paired models use the same resamples

The forecast is one step, so ADE and FDE both equal the geodesic error and are not reported as separate metrics. For trained models, the headline metric is the median across the fixed training seeds. "Beats" means a strictly lower headline point estimate; confidence intervals and seed spread are reported but do not create a second pass/fail rule.

**Comparators.**

| Symbol | Model |
| --- | --- |
| P | persistence: predict no movement |
| V | constant velocity: extrapolate the last observed displacement. `models/averaging.DisplacementAveraging` at k = 1, the CVM already in the Tier-1 ladder. Reported, never used for selection. |
| Z | zero-shot: the full FM, pretrained on all three source species and evaluated unchanged on a species completely unseen by Z. Tests whether knowledge transfers across species. |
| Za | domain-ablated zero-shot: Z retrained without the source species matching the held-out species' domain. Tests the value of adding matched-domain source data. |
| B | pooled source baseline: trained on the same source pool as Z but stripped of the FM-specific package (the masked reconstruction objective, the context derived conditioning path and the FM specific architecture) |
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

**Superseded wording:** the affected parts of Sections 1–6 and the amendment policy are preserved in the pre change version of this file at commit `1e31524`. A1 and A2 above separately govern the regime threshold and continuity corrections.

### A4; 03-08-2026: the low displacement boundary is made robust to its anchor

**Status when made:** no Tier 2 model has been trained, no sealed test partition has been accessed, and **no candidate species has been profiled**. No per-species segments file exists other than white stork's, and `results/regime_profiles.csv` holds no candidate row. The procedure below is therefore registered before any distribution that could influence its outcome has been seen.

**Reason.** The registered `0.10 × p50` rule is reproducible, but it is anchored on the median and the anchor is what fails. Where a species spends most of its time below the cadence's detection scale, the median itself falls inside the low displacement mode and the rule *inverts*: a species that is genuinely ~80% low displacement scores ~5.6%. The defect is in the anchor, not the constant, so a 5/10/20% sensitivity band does **not** detect it; all three multipliers return the same broken answer. A fallback must therefore move the **quantile**. It is registered now because once candidate distributions are visible, choosing an anchor is choosing a result.

**The registered procedure.**

1. Profile every mechanically eligible candidate on **training individuals only**, over steps spanning **exactly one bin**: p50, p75, p90; the `0.10 × p50` boundary and the fraction below it; and, where supported, a two component lognormal mixture crossing and the fraction below it.
2. A mixture is **supported** only with all of: at least 100 one bin steps; a converged fit; at least 5% weight in each component; a density crossing strictly between the component means; and BIC at least 10 lower than a one component lognormal. Otherwise it is reported unavailable and its guard is not applied; an unsupported mixture is not evidence either way.
3. **Retain `0.10 × p50` for every species** if every candidate has **< 45.0%** of steps below its own boundary, **and** no supported mixture places its p50 inside the low component (`p50 <= crossing`).
4. **Otherwise move every species to a p75 anchor**, with the multiplier fixed from the stork training distribution so the stork boundary is unchanged at 60.9 m: `c75 = 60.9 m / p75_stork`, shared rule `c75 × p75_species`. Calibrating on stork keeps Tier 1 and Tier 2 numerically comparable and leaves no freedom in the constant.
5. **If any supported mixture still places p75 inside the low component, move every species to p90** the same way: `c90 = 60.9 m / p90_stork`, shared rule `c90 × p90_species`.
6. **If a supported mixture places even p90 inside the low component for a species**, we do not invent a higher quantile and do not give that species its own rule. It is reported **pooled only**, recorded as having no defensible registered split.

Whichever branch fires applies **once, across all species**. The data select the anchoring quantile; they cannot select a per species formula, and no species is dropped for being inconvenient. The mixture crossing stays a diagnostic and is never the evaluation boundary, since a per species fitted boundary would make the reported strata depend on a fit whose quality varies between species.

**Terminology.** The strata are renamed "stationary" and "travelling" to **low displacement** and **travel**. The old names asserted a behavioural interpretation a step length threshold cannot carry, especially on Argos data where short steps are substantially observation error. The boundary is computed identically; only the name changes.

**Superseded text (Section 3, as committed at `86afad1`):**

> Reported **stratified by regime**, where the regime boundary is scale free and species specific: a step is stationary if it is below 10% of that species' own median step, where the median is computed on that species' training individuals only, over steps spanning **exactly one bin** of our common modelling cadence after resampling. Requiring exactly one bin matters since segments tolerate internal gaps of up to 8 bins, and those longer steps are systematically larger, so admitting them inflates the median and loosens the threshold. `regime_profile.py` implements this definition; on white stork's training individuals it gives a threshold of 60.9 m against the 60 m hysteresis cut we used in Tier 1, and a stationary fraction of 31.1% against the ~33% derived independently there. `regime_profile.py` will give the production figure. If our implementation and this text ever disagree, this text governs (if we encounter a bug for example).

**What changed.** Section 3 now states the boundary as one universal formula whose anchoring quantile is chosen by the procedure above, uses the new stratum names, and carries the Argos limitation. The stork figures (60.9 m, 31.1%) are unchanged and the default branch is the rule already registered, so if no guard fires the evaluation is unaffected. No metric, comparator, success band, species role or secondary prediction is altered.

### A5; 05-08-2026: evaluation window floor corrected, and p99 moved to species level

**Status when made:** no Tier 2 model has been trained and no held-out species performance exists. The evidence is persistence error on source data only, which is a property of the data and needs no model.

**Reason 1, independent of species selection.** One hundred windows cannot support a p99. Under independence 36.6% of individuals hold no observation above the population p99 (`0.99^100`), so their reported p99 is really just their sample maximum, biased low with no way to detect it; a two sided distribution free interval needs roughly 368 observations. Both figures assume independence, which flatters the floor, since adjacent windows share every position but one. No per individual floor we could impose without emptying the split would support a p99, which is why the fix is change (3) below rather than a higher threshold.

**Reason 2: 100 was inherited, not derived for Tier 2.** It came from our Tier 1 stork split, where it excluded 6 of 846 birds and 0.05% of windows (we had so much data there). Costless on a densely tracked migratory bird, but expensive on species with shorter deployments, and never checked against the precision it was supposed to buy. `experiments/data/window_floor_sensitivity.py` measures that on training individuals only, drawing simulated test sets of n individuals and computing the cluster bootstrap interval for the individual weighted travel p90:

| species | floor | n=10 | n=20 | n=30 |
| --- | ---: | ---: | ---: | ---: |
| white stork | 10 | 17.9 km | 20.0 | 15.0 |
| white stork | 100 | 20.3 km | 19.4 | 15.0 |
| southern elephant seal | 25 | 61.4 km | 46.5 | 46.9 |
| southern elephant seal | 100 | 50.2 km | 45.1 | 34.0 |

Raising the floor from 10 to 100 changes stork width by nothing at n = 30, and the ~27% it buys on the seal costs 89 individuals. Width is governed by how many individuals are tested and how much they differ, not by how many windows each contributes.

**What changes.**

1. The evaluation floor becomes **25** valid one step forecast windows per individual, summed across runs satisfying the registered continuity rule and applied identically to every species. We note that is serves as a pragmatic inclusion floor, and not an optimal or literature derived precision threshold (we couldn't find one for this problem anyway). We keep a floor because every included individual must contribute more than a handful of forecasts, and choose 25 because the sensitivity analysis found no consistent p90 precision gain from raising it to 100 while 100 imposed heavy duration based exclusions. Sensitivity at floors of 10, 25, 50 and 100 will be reported.
2. Species median and p90 are computed from the **individual weighted** empirical distribution rather than by pooling all windows, since pooling lets the longest deployments dominate (for example, Wolf A contributes 1000 forecasts to Wolf B's 25; if we pool all windows, Wolf A contributes ~97%). 
3. **p99 is reported at species level only**, never per individual.

**Declared consequences.** Individuals clearing the floor, at 100 then 25: stork 784 -> 835, southern elephant seal 240 -> 329, Adelie penguin 5 -> 64, Weddell seal 11 -> 50, walrus 13 -> 36, humpback 28 -> 34. This materially affects which species can be evaluated and is a consequence, not the justification. Our supposed 95% uncertainty intervals only contained the true value about 80-92.5% of the time. This appears to be a limitation of using a cluster bootstrap when there are relatively few individuals, and requiring more windows per individual does not solve it.

**The boundary this does not cross.** Eligibility thresholds are properties of the data and may be corrected under the dataset exploration Hofman et al. permit. **Success criteria are not.** The bands in Section 4 will not be revised on data availability grounds, and since Section 4 compares p99 per species rather than per individual, change (3) leaves them untouched.

**Superseded text (Section 2, as amended by A2):**

> - Apply the continuity rule to both model training and evaluation. A training individual remains eligible if it contributes at least one valid contiguous window. Validation and test individuals must contribute at least 100 valid contiguous windows because their per individual estimates would otherwise be too noisy. These filters are applied to the already frozen manifest: individuals are never reassigned, replaced or drawn from another split to restore its size.

**What changed.** The evaluation floor moves from 100 to 25; median and p90 become individual weighted; p99 becomes species level. The training floor of at least one valid window is unchanged, as is the rule that frozen manifests are filtered rather than replenished. No comparator, species role, success band or secondary prediction is altered.

### A6; 10-08-2026: context length reduced from 24 to 12

**Status when made:** no Tier 2 model has been trained, no species roles have been frozen, and no held-out species has been evaluated. The evidence comes entirely from the Tier 1 white stork corpus on its frozen split `splits/stork_split_seed42.json`, which contains no Tier 2 held-out species. Our decision metric, the tolerance and the action under each outcome were written into `experiments/reporting/context_ablation.py` and committed at `491320d` **before** the C=12 runs were read; the script records this as `rule_set_before_reading` and writes the verdict mechanically to `results/context_ablation/decision.json`.

**Reason.** We inherited C=24 from Tier 1, chosen against stork tracks that are long and near-continuous at 6h. Windows per individual are sum over maximal contiguous runs of $max(0, L − C)$, so the context is subtracted **once per contiguous run**, not once per individual, and a species whose tracks fragment into many short runs (like all of our marine ones) pay that cost repeatedly. Measured window multipliers from C=24 to C=12 are now

| species | window multiplier, C=24 -> C=12 |
| --- | ---: |
| white stork | 1.04x |
| southern elephant seal (SOES) | 1.70x |
| northern elephant seal (NES) | 1.57x |
| Adélie penguin | 1.80x |
| Weddell seal | 2.62x |

This is the largest data lever available to our marine leg without changing the cadence or going $\Delta$ t aware, both ruled out elsewhere. It is therefore worth testing, but only against a rule fixed in advance, because "shorten the context until the marine species become eligible" is exactly the choice that this document was created to prevent. 

**The registered rule, committed before the numbers were read.** Every C=24 target is also a C=12 target, so the comparison is paired: C=12 is scored on the C=24 targets and inherits the C=24 regime labels, and that arm is the decision number. Our chosen metric is **moving regime p99 on the paired targets**, test split, mean over 5 seeds, `time_season` cell. Judged **preserved** if and only if

    |p99(C=12 paired) − p99(C=24)| $\leq$ 2 x pooled seed sd,   for BOTH architectures

with the actions fixed in advance: *preserved* -> register C=12 and rebuild Tier 2 eligibility at C=12; *degraded* -> retain C=24 and report a null.

**Result.** Moving regime, mean across 5 seeds, seed sd in parentheses:

| arch | metric | C=24 | C=12 paired | change |
| --- | --- | ---: | ---: | ---: |
| LSTM | mean | 10.303 (0.098) km | 10.245 (0.048) km | −0.56% |
| | median | 2.465 (0.116) km | 2.246 (0.041) km | **−8.89%** |
| | p90 | 26.705 (0.344) km | 26.795 (0.318) km | +0.34% |
| | p99 | 121.839 (0.791) km | 123.299 (0.569) km | +1.20% |
| Transformer | mean | 9.495 (0.013) km | 9.483 (0.029) km | −0.13% |
| | median | 1.924 (0.012) km | 1.922 (0.014) km | −0.09% |
| | p90 | 25.658 (0.201) km | 25.612 (0.161) km | −0.18% |
| | p99 | 114.875 (0.667) km | 115.770 (0.781) km | +0.78% |

On the decision metric LSTM p99 rose 1.46 km against a 1.9 km tolerance and Transformer p99 rose 0.90 km against 2.1 km. Both pass, and so our verdict is **preserved**, and the registered action follows.

**What changes.** The context length now becomes **C = 12**, so a valid window is **13 consecutive 6h bins** in place of 25. Section 2 now states the modelling window explicitly, which it previously did not. 

**Caveats.**

1. **"Indistinguishable", not "improved".** Both p99 deltas are positive: this is a ~1% cost in the extreme tail, bought in exchange for windows. The LSTM's 8.89% median gain lands elsewhere in the distribution and does not net against the tail.
2. **Pooled sd is the less powerful test.** The arms share seeds and score identical targets, so a paired by seed test might resolve the LSTM's +1.20% as real. We ran the test we registered; a stronger one could plausibly have failed this arm. A limitation of the rule, not grounds for re testing after the fact.
3. **Stork only.** The fragmented marine tracks that stand to gain most were not in the ablation. That C=12 holds on SOES, NES, Adele penguin or Weddell seal is an **assumption**, declared as one rather than reported as a result.
4. **A5 was reasoned under C=24.** Its overlap argument holds at any C; at 12 of 13 rather than 24 of 25 each window is marginally more informative than A5 assumed, which strengthens the floor of 25 rather than weakening it. A5's consequence table stays at the counts at which it was measured.

**Superseded text.** No operative sentence stated the context length before this amendment, which is the defect it corrects; the only numeric trace was "near identical 24 fix contexts" in the first leakage control bullet, now reworded to be context length independent. Amendments A1–A5 are dated records of decisions taken under C=24 and are **not** retrospectively edited.

**What changed.** Context length 24 -> 12, window 25 -> 13 bins, Section 2 gains an explicit modelling window statement. No metric, comparator, success band, species role, secondary prediction or evaluation floor is altered. The eligibility rebuild is a consequence, not a change of criteria.

### A7; 10-08-2026: splitting fixed to split-specific

**Status when made:** no Tier 2 model has been trained, no species roles have been frozen, and no eligibility count has yet been rebuilt at C=12. The policy is therefore registered before the counts it governs exist.

**Reason.** Section 1 fixed how many individuals each split needs but we never said where they are drawn from, and two policies satisfied it equally:

- **filter then split**; discard everyone below F, divide the survivors 75/12.5/12.5; entry gate $n(\geq F) \geq 80$.
- **split specific**: draw validation and test from the $\geq$ F pool, then send every other individual holding at least one valid window to training; entry gate $n(\geq F) \geq 20$ and $n(\geq 1) \geq 40$.

We choose **split specific**, because the two roles ask different things of an individual and should not share one threshold. An individual too short to support a stable per individual evaluation estimate can still supply valid supervision, so discarding it costs training data and buys no evaluation precision. On SOES that is roughly 366 training individuals against 247. The entry gate is the larger effect, 20 above the floor rather than 80, and most of our marine candidates sit in the 20-80 band, so the two policies disagree about which species are eligible at all; that is a declared consequence, not the justification.

**Superseded text (Section 1, gate 1, as amended by A5):**

> A species must retain at least 20 training, 10 validation and 10 test individuals after continuity filtering; training individuals need at least one valid window and validation/test individuals at least F, the evaluation window floor fixed at 25 by Amendment A5 as of right now.

**What changed.** Gate 1 now states the split policy and its entry gate. No metric, comparator, success band, species role, secondary prediction, window floor or context length is altered.

**Note on fixed versus proportional evaluation size.** At the time of this amendment we also considered whether validation and test size should scale with the number of eligible individuals, rather than remain fixed at 10 each. We retain the fixed 10/10 rule here because a percentage split would permit very small species to enter evaluation with only one or two test individuals, undermining the individual level species estimates and bootstrap uncertainty. However, the converse issue also remains; for species with many eligible individuals, restricting evaluation to only 10 test animals may discard useful independent replication. We therefore leave open a future preregistered amendment, made before the per species split manifests are committed (not merely before sealed evaluation, since our A2 outline forbids regenerating a frozen manifest) to increase the test count under a fixed mechanical rule or cap if justified by an individual count sensitivity analysis. Any such change would alter evaluation precision only and would not be introduced to make a particular species eligible.

### A8; 14-08-2026: the pretraining sampling rule is registered

**Status.** No Tier 2 model trained, no configuration evaluated, no held-out partition read. Species roles were frozen in `tier2_species_roles.md` at commit `a947127` before this amendment.

**Defect.** Recall that in 2 we fixed the modelling window, the split policy and the leakage controls, but never said how training windows are actually drawn. The default any dataloader implements; a uniform over the pooled window set, puts the marine source at **5.7%** of draws. Thus, a weak NES result would then be unattributable between either source scarcity and taxonomic distance, and Secondary Prediction 2 would not be testable.

**The rule.** Species -> individual -> window. A source species is drawn with probability proportional to $n_s^{\alpha}$ at $\alpha = 0.5$, where $n_s$ is its training partition window count at C = 12; a window is then drawn **uniformly within that species**, so individuals are weighted by how many windows they contribute. That is a training weight only — A5 governs what is reported regardless. Note, that draws are with replacement, so training length is declared in optimiser steps rather than epochs. Weights are computed once from the frozen manifests and read from `results/source_pool_weights.json`, and never re derived at runtime.

Registered draw probabilities: *C. ciconia* **46.491%**, *C. elaphus* **38.619%**, *M. leonina* **14.889%**, from $n_s$ = 806,595 / 556,579 / 82,731. Superseded by A9; Za and the pseudo target folds renormalise $\sqrt{n_s}$ over the two species that remain.

**$\alpha$ is excluded from the configuration budget and is never searched.** $\alpha = 1$ gives the 5.7% above; $\alpha = 0$ resamples *M. leonina* 9.75 times per stork pass, converting a coverage problem into an overfitting one. 0.5 is the conventional interior value, fixed by convention rather than by search.

**Scope.** S and F train on one species, where the species level is degenerate. Validation, pseudo target scoring and sealed evaluation are not sampled at all; every eligible window is scored exactly once. B inherits the rule unchanged, since a sampling difference would confound the comparison B exists to make.

**Unchanged.** No metric, comparator, success band, species role, secondary prediction, evaluation floor, window floor or context length. Derivation, the resampling ratios and the §4 reconciliation are in `notes.md` (16-08-2026) and `methods.md`.

### A9; 16-08-2026: physical plausibility QC applied to the extract; marine data rebuilt; A8 weights revised

**Status.** No Tier 2 model trained, no configuration evaluated, no held-out partition read. We found this defect while measuring window scale statistics, not while looking at any model output.

**Defect.** `data.qc` computes seven flags; but our `extract_species_fixes.py` acted on none of the four physical plausibility ones. Correct for a GPS corpus, where every species flags zero fixes; but wrong for an Argos one. The largest retained 6h step was **12,959 km** on *M. leonina*; 2,160 km/h and eleven SOES fixes sat at positive latitude against a declared bound of −90..0.

**Correction.** The drop mask is now built from a declared `DROP_FLAGS` tuple covering all seven flags, applied identically to every species. No per species rule is introduced.

| | *M. leonina* | *M. angustirostris* |
| --- | ---: | ---: |
| fixes kept | 787,410 -> **556,106** | 621,467 -> **537,698** |
| train / val / test individuals | 417/10/10 -> **397/10/10** | 353/10/10 -> **331/10/10** |
| windows at C = 12 | 87,433 -> **55,822** | 58,000 -> **48,587** |

| species | $n_s$ (train windows) | draw probability |
| --- | ---: | ---: |
| *Ciconia ciconia* | 806,595 | **47.911%** |
| *Cervus elaphus* | 556,579 | **39.799%** |
| *Mirounga leonina* | 53,076 | **12.290%** |

Subpools: without stork, *Cervus* 76.406% / SOES 23.594%; without *Cervus*, stork 79.585% / SOES 20.415%; without SOES, stork 54.624% / *Cervus* 45.376%. $\alpha$ remains 0.5 and remains excluded from the budget.

**Gate 2.** Standardisation refrozen over the 21 species table and the A4 decision re run; the branch is still `0.10 × p50` and **no species role changes**. Pair distances 5.10 -> **5.08** / 2.65 / 1.42 -> **1.46**, coverage 5.62/3.42 -> **5.47/2.74**.

**Split manifests regenerated, not filtered.** A2's no regeneration rule protects a manifest whose splits have been used; but these two were committed on 15-08-2026, no partition of either had been read, and the population underneath them changed. Filtering would have discarded individuals without replacing them and could have taken val or test below the fixed 10. Regenerated at seed 42 under our unchanged A7 policy with F = 25, so **which individuals sit in validation and test changes**.

**Declared limitation.** The interval aware ceiling operates on raw cadence, so the largest surviving 6h step is still **3,822 km** (SOES) and **1,822 km** (NES). A plausibility guard on the resampled grid would catch what a raw cadence guard cannot; that is new QC, is not registered here, and we declare it here as a limitation of the marine data in both tiers. The marine draw share falls to 12.290%; adding a second marine source to compensate was considered and rejected, since 1 selects sources for regime coverage and Tier 3 is fixed as differing in source pool size alone.

### A10; 18-08-2026: descriptor transforms made consistent by measurement scale

**Change.** Each gate 2 descriptor is put on its natural measurement scale before centring and scaling; ratio scale strictly positive -> **log10** (tail ratio, step CV, diel and seasonal amplitude); already logged -> none (median and p90 step); proportion in [0,1] -> **logit** (intermittency); correlation in [−1,1] -> none (rho).

**Reason.** Four descriptors of the same kind as the two logged ones were z-scored raw. `tail_ratio` runs 3.6 to 234.4 and stork *is* the 234, sitting at z = +4.35 while the other twenty compress into 1.08 SD — a near binary "is this the stork" indicator. Each pair distance was consequently dominated by one descriptor: flying 71% tail ratio, terrestrial 78% intermittency, marine 92% rho.

**Effect.** Pair distances 5.08 / 2.65 / 1.46 -> **3.21 / 2.44 / 1.88**; worst single descriptor share 92% -> **59%**; source pool coverage 5.47/2.74 -> **5.03/2.94** against **4.72/3.30** reversed, so 4 reads as before. **No role changes.** The A4 branch cannot change: `select_threshold_rule` never reads these descriptors, asserted in `tests/test_regime_profile.py` rather than established by re-running.

**Status.** No Tier 2 model trained, no configuration evaluated, no held-out partition read. `results/regime_descriptor_standardisation.json` is regenerated over the same 21 species and now records the transform it was taken under.

### A11; 18-08-2026: constant velocity added as a reported comparator

**Change.** A comparator **V** joins P in the reported tables; constant velocity, extrapolating the last observed displacement. It is `models/averaging.DisplacementAveraging` at k = 1, already in the Tier 1 ladder, so this adds a column and no new machinery.

**Reason.** P predicts no movement, which is a weak floor in exactly the stratum 4's bands are defined on; travel regime is by definition the steps where the animal moved. Reporting only P leaves the headline open to the objection that the baseline was chosen to be beatable.

**Unchanged.** Rcall 4 bands remain defined against P, B and S, and the inner loop selection metric remains persistence skill on travel regime p90. Adding a stronger comparator and then moving the bands onto it would be like changing the goalposts; using it to select would be changing the search. If V beats Z on any held-out species that is reported as such.

**Status.** Registered before any Tier 2 model trained, configuration evaluated, or held-out test partition read.

### A12; 18-08-2026: auxiliary objective named accurately, masked tokens replaced, loss made radial

**Terminology.** The masked reconstruction term trains **jointly with the forecast term in the same optimiser step**. There is no preceding self-supervised phase, so it is an **auxiliary objective**, not a pretraining stage, and 3s comparator table is updated to match. 

**Masked tokens.** A masked token's projected embedding is replaced by a **learned mask embedding** rather than having its movement channels zeroed. Zeroing was ambiguous — heading is already gated to zero below 10 m and roughly a third of steps are low displacement, so a hidden token and a stationary animal produced the same input. Since the objective is part of what distinguishes Z from B, an ill posed objective weakens that comparison directly. Elapsed time, window scale and the clock features stay visible, as before.

**Loss form.** Now the **radial pseudo-Huber** on the error magnitude, $\delta^2\left(\sqrt{1 + \frac{\lVert e\rVert^2}{\delta^2}} - 1\right)$, replacing the component wise Huber, which was axis dependent while every number 3 reports is a geodesic distance. ** $\delta$ remains 1.0** and is better supported under the radial form: the measured median normalised errors of 0.797 / 0.982 / 0.996 are error *magnitudes*, which is what the radial loss takes. The earlier claim that this held "by construction" is withdrawn — it is the measurement, not the algebra.

**Status.** No Tier 2 model trained, no configuration evaluated, no held-out partition read.

## GenAI Reference 

I initially used GenAI by giving it this preregistration, which I had written myself, and asking whether the success criteria seemed reasonable for the scope of my project (which I outlined in the chat). It agreed that the criteria was appropriate, but pointed out one issue I had overlooked; trait conditioning is not meaningfully testable in Tier 2 because the model sees only three source species, and therefore only three distinct trait vectors. At that scale, the traits are perfectly confounded with species identity, so the model could simply use them as a species lookup rather than learn a genuinely transferable relationship between biological traits and movement.

I had initially assumed that because traits describe meaningful biological differences, they should naturally help zero-shot transfer. GenAI’s suggestion made me realise that the problem was not whether the traits were biologically relevant, but whether the experiment could distinguish trait based generalisation from species memorisation. I then developed and wrote the Tier 2 limitation and Tier 3 justification myself, including the decision to report trait conditioning descriptively in Tier 2 rather than treat it as a formal prediction.