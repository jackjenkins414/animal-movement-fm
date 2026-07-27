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

**Amendment policy.** Adjustments are permitted; silent adjustments however, are not. This follows Hofman et al.'s phase (B) design, whose stated objective is to avoid over constraining the researcher by allowing changes to phase (A), while requiring that those changes be explicitly stated so a reader can judge their impact. Accordingly: any change to this document after it is committed will be recorded in an **Amendments** section appended at the end of this file, with the date, the reason, and the original text quoted verbatim. The first such amendment starts that section; until that point, this document is v1 fully. A change made *after* a sealed test result has been seen must additionally say so explicitly. Changing the success criteria after seeing results, without recording it, would invalidate the the whole purpose of our work here (obviously).

**Correction rule.** If a defect is discovered during or after a sealed evaluation (such as a data bug, a broken metric, a mis-specified split) a correction is permitted **only if it is independent of the direction of the result**, (it would have been made identically had the numbers come out the other way). Such corrections are logged as amendments, with the defect, the fix, and the affected numbers before and after. Changes motivated by the result itself are not corrections and we will not employ them. 

## 1. The claim under test

A movement forecasting model pretrained on multiple species generalises to a species held out entirely from training, i.e, it beats trivial baselines on that species without ever having seen it.

**Tier 2 pool: six species, in two fixed and disjoint roles.**

| role | count | composition |
| --- | --- | --- |
| **source** | 3 | one per domain: flying, terrestrial, marine. White stork is the flying source (our existing corpus). |
| **held out** | 3 | one per domain: flying, terrestrial, marine. Never trained on, never tuned on, never inspected until the sealed evaluation. |

No species does double duty; a source species is never held out, and a held out species is never a source.

**Why six rather than three.** With only three species evaluated leave one out, holding out the marine species leaves nothing marine in training. A failure could then be taxonomic distance, or it could simply be that the model never saw that domain at all, and we would have no way to separate the two. With one source species per domain, every held out species has its own domain represented in training *by a different species*, so the test is genuinely "a new species" rather than "a new domain". The domain question does not disappear; we ask it separately and deliberately, through the ablation arm in Section 2.

**The held out set is fixed across tiers.** The same three held out species carry into Tier 3 unchanged, along with their frozen splits. Tier 3 then differs from Tier 2 in exactly one variable: the size and composition of the source pool (three species, one per domain, growing to nine, three per domain). Everything else is held constant, being the same test species, the same test individuals, the same metrics and the same protocol. That makes our two tiers a single controlled scaling study rather than two disconnected experiments, and it lets us report held out performance as a function of source pool size. There is no established law for how many source domains a foundation model needs, so this is the axis we are best placed to measure.

Our selection of which species fill which role proceeds through two separate gates, in this order, and they must not be blended:

1. **Eligibility: a data quality filter.** Can the species produce well populated windows at our modelling cadence, at usable individual counts? Applied mechanically to every candidate, before any regime descriptor is inspected.
2. **Selection: the scientific criterion.** Among eligible species, choose for movement regime coverage as measured by `experiments/data/regime_profile.py`, not by taxonomic convenience. Our reasoning here is that this makes the Tier 2 test a harder and more meaningful question; does the pretrained model transfer across genuinely different forms of movement? 

An important note on keeping these separate matters: selecting on cadence feasibility alone would be exactly the convenience driven choice this document is meant to rule out, whereas excluding a species that cannot form windows at all is a precondition, not a preference.

**Pairing within a domain.** For each domain, the source and the held out species should differ in *species identity* and, as far as the data permits, not in device technology or geographic region. Where that cannot be achieved, the difference is recorded explicitly for that pair so a transfer result can be attributed rather than merely observed. Our marine pair is the known risk:, since if both marine species come from RAATD then both are Southern Ocean, and "marine" and "polar" become the same variable on both sides of the comparison. Drawing the held out marine species from a different ocean basin (for example Movebank Argos walrus, Arctic) is the intended mitigation.

**Fallback, declared in advance.** If six species have not cleared eligibility, this design reverts to sealed leave one species out over three species, one per domain; each species held out in turn and trained on the other two, giving three sealed results and no ablation arm. Taking the fallback will be recorded as a dated amendment. Registering it now is the point; choosing between two designs is something we will know after we begin our analysis. 

## 2. Protocol (fixed in advance)

**Outer loop: sealed evaluation on the held out species.** The model is pretrained once on the three source species, then evaluated on each of the three held out species in turn. Each held out species is used for NOTHING except that final evaluation, not model selection, not early stopping, not threshold tuning, not feature decisions.

**Domain ablation arm.** To ask separately whether *domain* coverage in the source pool matters, we re-pretrain three further models. Each drops the one source species sharing a domain with a given held out species, and is then evaluated on that held out species only. Comparing Z against Za on the same held out species, with everything else identical, measures what having any data at all from that domain was worth. This is the within versus across domain question asked directly, rather than inferred from a leave one out design that cannot separate it from taxonomic distance.

**Inner loop: pseudo targets.** Hyperparameters are selected by holding out one of the three *source* species as a pseudo target, so tuning is against genuine unseen species performance rather than random validation error. The configuration selected this way is then frozen and used for the final pretraining on all three sources, and for every comparator below. **The inner loop is budgeted**: the number of configurations evaluated is bounded in advance and recorded in the pre-test freeze. An unbounded inner loop would let our (researcher) degrees of freedom back in through the side door, since enough pseudo target attempts will eventually select a configuration that happens to suit the outer fold.

**Leakage controls.**

- Split individuals FIRST, then window. Windowing before splitting leaks near identical 24 fix contexts across train and test. We do already do this, but its stated here because it is the primary leakage mode in this literature.
- Group by individual, which subsumes deployment, since an individual's deployments cannot be separated by an individual level split.
- Splits are read from committed manifests (`splits/*.json`), never recomputed.
- Any QC threshold derived from data (speed quantiles, error calibration) is computed on **train only** or strictly per track. Corpus wide thresholds leak test information into preprocessing.

**Robustness checks.** Blocked by study and blocked by region evaluation inside species, to show that a "cross-species" result is not a geography or tag settings shortcut.

### Final pre-test freeze

After source species training and pseudo target model selection are complete, but before any sealed held out species test result is examined, a second dated record will freeze the final model specification. This is our phase (B) in the sense of Hofman et al. (2023), which separates problem definition from the model specification made immediately before testing.

**The freeze is another committed file, `preregistration_tier2_freeze.md`,** which we will commit before the first sealed evaluation is run. It will record:

- final architecture, layer/width configuration and parameter count
- selected hyperparameters and the number of inner loop configurations actually evaluated, against the declared budget
- random seeds and the checkpoint selection rule
- computational budget consumed
- **an explicit, dated attestation that the held out species' test partition has not been accessed in any way**: not for exploratory plotting, not for sanity checks, not for
  debugging. Stated as a claim rather than left implicit, so it is something a reader can hold us to.
- any difference from this document, listed item by item

## 3. Metrics (fixed in advance)

Our primary error is **great circle (geodesic) error in metres** on the held out species' test windows, reported as **median, p90, p99**; recall we found heavy right skew makes the mean uninformative.

Reported **stratified by regime**, where the regime boundary is scale free and species specific: a step is stationary if it is below 10% of that species' own median step, where the median is computed on that species' training individuals only, on tracks resampled to the common modelling cadence. `regime_profile.py` implements this definition; on white stork's training individuals it gives a threshold of 62.6 m against the 60 m hysteresis cut we used in Tier 1, and a stationary fraction of 30.8% against the ~33% derived independently there. `regime_profile.py` will give the production figure. If the implementation and this text ever disagree, this text governs (if we encounter a bug for example).

We will also report:

- **persistence skill** = 1 − model_error / persistence_error, per regime and per species
- **ADE / FDE** for comparability with the trajectory forecasting literature
- **paired bootstrap confidence intervals over individuals**, not over windows (windows within an individual are strongly correlated, so window level intervals are far too narrow)

**Comparators.**

| Symbol | Model |
| --- | --- |
| P | persistence: predict no movement |
| Z | zero-shot: pretrained on all **three source species**, never saw the held out species |
| Za | zero-shot, domain ablated: pretrained on only **two** source species, dropping the one that shares a domain with this held out species. Everything else identical to Z. Z versus Za isolates the value of domain coverage in the source pool. |
| B | pooled source baseline: the same architecture trained on the pooled source species with **no** metadata/trait conditioning and no cross-species machinery, i.e, Z stripped of everything the FM framing claims to contribute. Isolates whether any transfer is attributable to the FM design rather than to simply having seen more movement data. |
| S | specialist: the same architecture and frozen configuration as Z, but trained only on the held out species' own training individuals. An oracle upper bound, not a target to beat. Defined precisely below. |
| F | fine-tuned: Z, further trained on a fixed fraction of the held out species' training individuals. Reported as a curve over fractions rather than as a single number. Defined precisely below. |

### On the specialist (S)

The 25% threshold in Section 4 is defined against S, so leaving S vague would leave a numeric success criterion resting on an ambiguous comparator. So we fix the following now:

- **Architecture and configuration.** S uses the same architecture and the same frozen configuration as Z, taken from the pre-test freeze. The only thing that differs between Z and S is the data each was trained on, so any gap between them is attributable to training data rather than to model capacity or tuning effort.
- **Training data.** Only the held out species' own training individuals, read from that species' frozen split manifest. Never the test partition.
- **Seeds.** The same number of training seeds as Z, so the comparison does not inherit some variance asymmetry.
- **Tuning privilege, stated openly.** S is allowed to early stop and select checkpoints on a validation split of the held out species' *training* individuals. Z structurally cannot do this, because for Z that species does not exist during training. The asymmetry is deliberate and is exactly what makes S an oracle: the Z versus S number is an upper bound on what having the target species' data buys us, not a like for like contest. We are not tuning per species hyperparameters for S, because the configuration is frozen in the inner loop against pseudo targets, and giving S its own tuning budget would stack an architecture and tuning confound on top of the data difference we actually want to measure.

### Fine-tuning (F), and why it is here

Tier 2 as originally written tested zero-shot only, but one of the things we are investigating in our work is pretrain **then finetune**, and some of the transfer literature we have reviewed thus far reports that a specialist can overtake a zero-shot model with as little as ~2% of target data. Zero-shot is therefore the hardest single point on a curve rather than the whole claim, so we register the curve.

- **Definition.** F_k is Z, further trained on a fraction k of the held out species' training individuals.
- **Fractions, which we fix in advance.** k $\in$ {1%, 10%, 100%} of the held out species' training individuals. We log space them deliberately; the fine-tuning curve is steep at the low end and flat near the top, and the transfer literature puts the crossover at around 2% of target data, so densifying the 25–100% stretch would buy resolution in the region we expect to be flattest.
- **Repeat draws at the low end.** At small k the subset is tiny in absolute terms (1% of the stork's 630 training individuals is 6 birds, and fewer on the other two species), so *which* animals are drawn will move the result more than the fine-tuning does. We therefore run **three independent draws at k = 1% and three at k = 10%**, and a single run at k = 100% since there is nothing left to vary. That totals to 7 fine-tuning runs per held out species.
- **Floor on subset size.** A fraction must yield at least **5 individuals**; where it does not, we use 5. Otherwise "1%" on a species with few training individuals is two or three animals and the label claims more than the data supports. The realised individual count and window count are reported alongside every k, since the same fraction means different absolute amounts of data on each species.
- **Sampling and nesting.** Sampling is by *individual*, never by window, drawn from that species' frozen split manifest with a fixed seed per draw. Each draw is nested: its 1% subset sits inside its 10% subset, which sits inside the full training set. This makes the comparison across k *within* a draw paired rather than independent, so the curve is not confounded by subset composition changing between points.
- **Configuration.** The same architecture and frozen configuration as Z and S; the fine-tuning budget (epochs, learning rate schedule etc) is declared in our pre-test freeze. F may early stop on a validation split of the held out species' training individuals, on the same terms as S, so that F and S remain directly comparable.
- **Sealing is unaffected.** F touches training individuals only. The held out species' test partition stays sealed for every model in this document.
- **Reporting.** A curve of travel regime p90 against k, per held out species, with Z as the k = 0 point and S as the asymptote. At k = 1% and k = 10% we report the **spread across the three draws** (median and range) rather than a single number, because a point estimate at those subset sizes would not be interpretable. This spread is a different quantity from the paired bootstrap intervals in Section 3: those resample *test* individuals, whereas the draw spread measures sensitivity to which *training* animals happened to be selected.

The comparison doing the most work here is **F_100 against S**: both see all of the held out species' training data, and they differ only in whether the weights were pretrained on the other species first. That isolates the value of pretraining with training data held constant, which is honestly the cleanest statement of what a foundation model is supposed to buy us (hopefully). 

**This does not move the goalposts.** The success bands in Section 4 are defined on Z alone and are unchanged by the addition of F. A good fine-tuning curve does not rescue a WEAK zero-shot result and we will not report it as though it does. F is registered here, before any Tier 2 run, precisely so that adding it cannot later be mistaken for widening the reporting surface after the fact.

## 4. Success bands

We also declare these in advance. "Travel regime" means steps at or above the stationary threshold defined above; the regime where forecasting is scientifically interesting and where persistence is not optimal by construction.

**STRONG.** For **every** held out species: Z beats P on travel regime p90 **and** p99 (persistence skill > 0 on both), and Z is within 25% of the specialist on travel regime p90; which we can state as follows to remove ambiguity:

    p90_travel(Z)  <=  1.25 x p90_travel(S)

**ACCEPTABLE.** Z beats P on travel regime p90 for **all three** held out species with a consistent sign, and `p90_travel(Z) < p90_travel(B)` for all three; Z may trail S by more
than 25%.

**WEAK / NULL.** Z beats P only on pooled metrics but not on travel regime metrics, **or** the sign of the effect is inconsistent across held out species.

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
5. **Domain coverage in the source pool matters more than source pool size.** Z beats Za on travel regime p90 for at least two of the three held out species, i.e, dropping the same domain hurts more than dropping one species' worth of data would on its own. If Z and Za are indistinguishable, then what transfers is generic movement structure rather than anything domain specific, which would be a genuinely interesting null and would also predict that Tier 3's gains come from sheer species count rather than from stratified domain coverage.

## 6. Declared limitations (stated before results, not after)

- **n = 3 held out species.** Six species are involved but only three are ever evaluated, so this remains an existence proof rather than a transfer study. It is too few for the transfer matrix and negative transfer auditing the literature recommends; those need Tier 3 scale. So no claim of general cross-taxa transfer will be made from three held out species.
- **Source and held out roles are disjoint, which is deliberate but costly.** Six species of ingest and QC buy three sealed results, where leave one out would have bought three from three. We accept that cost for the domain coverage and the fixed held out set. The consequence is that there is no rotation available: if a held out species turns out to be unusable after ingest, that result is lost rather than recoverable by reassignment, so the better quality data should sit on the held out side.
- **The fixed held out set propagates.** Because the same three species are carried into Tier 3, any way in which they are unrepresentative is inherited by both tiers rather than averaged out across them. The scaling comparison is clean precisely because the test set does not change, and that is the same reason a poor choice of test species cannot be corrected later without breaking the comparison.
- **The domain ablation is n = 1 per domain.** Z versus Za is a single paired comparison for each of the three domains, so it can indicate a direction but cannot resolve a small effect. It is registered as a secondary prediction, not a success criterion.
- **Single data split** per species (frozen manifests) and a small number of training seeds; effects smaller than our seed spread measured in the 22-07-2026 multi-seed sweep will not be treated as resolved.
- **Confounded taxon and technology.** The marine species are Argos while the others are GPS, so "marine" and "different error model" cannot be fully separated. With two marine species this confound now sits on both sides of the marine pair, which is why the pairing rule in Section 1 asks for the difference to be recorded per pair.
- Tier 1 results were computed in degree space; Tier 2 will use local ENU metres (currently in progress right now). The two are bridged by an explicit representation ablation rather than assumed comparable.

### Versioning and reporting

This document and the pre-test freeze are version controlled in the thesis repository, so each revision carries a commit timestamp independent of any date written inside the file. As we stated, amendments are appended to this file rather than overwriting the text they revise, so the superseded wording stays readable and our git history shows when each change was made. Our final thesis report will state which analyses were preregistered, identify any deviations, and give the reason for each one (which is the point of our exercise here), since a preregistration that is never compared against the realised path does no work.

## GenAI Reference 

I initially used GenAI by giving it this preregistration, which I had written myself, and asking whether the success criteria seemed reasonable for the scope of my project (which I outlined in the chat). It agreed that the criteria was appropriate, but pointed out one issue I had overlooked; trait conditioning is not meaningfully testable in Tier 2 because the model sees only three source species, and therefore only three distinct trait vectors. At that scale, the traits are perfectly confounded with species identity, so the model could simply use them as a species lookup rather than learn a genuinely transferable relationship between biological traits and movement.

I had initially assumed that because traits describe meaningful biological differences, they should naturally help zero-shot transfer. GenAI’s suggestion made me realise that the problem was not whether the traits were biologically relevant, but whether the experiment could distinguish trait based generalisation from species memorisation. I then developed and wrote the Tier 2 limitation and Tier 3 justification myself, including the decision to report trait conditioning descriptively in Tier 2 rather than treat it as a formal prediction.