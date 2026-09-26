# Preregistration: bird-only foundation model

**Written 26-09-2026**, after the 14 new species were extracted, gated, split and profiled, and before any pool weights were computed or any bird model trained. Tier 2–4 machinery is unchanged and referenced, not restated.

## 1. Question

Here we ask, does a foundation model pretrained only on flying birds transfer to held-out flying birds, zero-shot and few-shot, better than persistence and than from scratch specialists?

## 2. Held-out birds

| bird | split (train / val / test) | travel boundary | status |
|---|---|---:|---|
| *Larus fuscus* | 332 / 10 / 10 | 411.451 m | frozen A2 split; its test partition was read by Tiers 2 and 3, and will be by Tier 4 |
| *Anser albifrons* | 299 / 10 / 10 | 82.139 m | released from the Tier 3 sec 2 reservation for this study only; still excluded from Tier 4; test partition never read |
| *Streptopelia turtur* | 88 / 10 / 10 | 25.806 m | rule: most individuals at >= F among eligible flying birds never used in any tier, excluding Laridae and Anatidae. Applied 26-09-2026 to built values: 108 (next *P. occidentalis*, 69). Test partition never read |

Travel boundaries are A4's 0.1 x p50 on training individuals; the gull's registered in `registered/tier2_regime_decision.json`, the goose's and the dove's from their profile rows (the goose's on all 299 training birds). All three are pinned at full precision in `bird_train.py`, checked against our profile tables on every load, and imported by the evaluator. 

## 3. Pools

**Main pool (22).** Every eligible flying bird (A7 gate 1 at C = 12) except the three held out and their congeners: *C. ciconia*, *C. aura*, *H. ostralegus*, *M. americana*, *C. corone*, *B. canadensis*, *F. naumanni*, *P. apivorus*, *M. milvus*, *G. himalayensis*, *D. exulans*, *T. chrysostoma*, *T. antarctica*, *A. platyrhynchos*, *B. leucopsis*, *H. caspia*, *P. occidentalis*, *F. tinnunculus*, *P. roseus*, *P. leucorodia*, *M. bassanus*, *A. alba*.

**Congener pool (27).** The main pool plus *L. argentatus* (the gull's congener) and *A. caerulescens*, *A. fabalis*, *A. anser*, *A. brachyrhynchus* (the goose's). The dove has no congener among eligible birds.

**Excluded.** The "*Chen caerulescens*" and genus only "*Anser*" rows (the same snow geese as the *A. caerulescens* study), the Movebank copy of *T. antarctica* (RAATD is used), penguins, and Argos only Movebank studies (GPS only, as Tiers 2–4).

Weights follow A8 at alpha = 0.5, as in Tier 4 (no window floor). Both pools are written once to `registered/birds_source_pool.json`.

**Training-set exclusions.** The *A. albifrons* birds 490431939, 662388807, 1056306620, 662388728, 133992693 and 133992680 (at least 100 fixes duplicated in a val or test bird) are left out of every goose training set: S, SB, Sk, LSTM-S, XGB-S and the Fb subsets. The dove and the gull were checked the same way on 26-09-2026 and are clean. The six are recorded in `registered/birds_source_pool.json`, which the driver reads its exclusions from.

## 4. Arms

Configuration and budget as Tier 3/4 (`tier4_exploratory_freeze.md` sec 2): grid 11, 20,000 updates at batch 256, seeds 0–4, final checkpoint.

| arm | what | runs |
|---|---|---:|
| Zb, Bb | full package and plain trunk on the 22 bird pool | 10 |
| Zbc, Bbc | the same with the 5 congeners added back (27 birds) | 10 |
| Fb | Zb seed 0 fine-tuned on 1% / 10% / 100% of each held-out bird, as F9 | 21 |
| LSTM-Bb, XGB-Bb | pooled baseline rungs | 10 |
| goose and dove specialists | S, SB, Sk, LSTM-S, XGB-S | 44 |

95 runs. P, V, ARIMA and AVG3/10 are computed for the goose and the dove without training. **Carried for the gull:** P, V, S (Tier 2); ARIMA, AVG3, AVG10, SB, Sk, LSTM-S, XGB-S (Tier 3).

## 5. Reading rules

The statistic, seeds and bootstrap are as `tier4_exploratory_freeze.md` sec 3. A gap within +/- 0.05 counts as the same. Each held-out bird is read separately.

| comparison | result | reading |
|---|---|---|
| Zb, Bb persistence skill | above +0.05 / within / below −0.05 | beats / matches / loses to persistence zero-shot |
| Zb, Bb vs S | within 0.05 or above / S ahead | zero-shot matches the specialist / the specialist wins |
| Zb vs Bb | Zb ahead / within / Bb ahead | the package helps / no difference / the package hurts |
| $\Delta$_Z = Zbc − Zb, $\Delta$_B = Bbc − Bb | both above +0.05 | congeners carry the transfer |
| | both within noise | congeners add nothing |
| | one above +0.05, the other within noise | the package and the congeners interact |
| | anything else | unresolved |
| Fb vs Sk, same draw (Tier 3 freeze sec 6.1) | Sk ahead / within / Fb ahead | pretraining loses / no difference / pretraining helps |
| ladder with LSTM-Bb and XGB-Bb | — | descriptive, as Tier 3 sec 3.1 |

The dove has no congener, so its $\Delta$ rows measure only the effect of adding five unrelated birds to the pool.

## 6. Predictions

Tier 3 numbers are from `results.md` sec 18. Tier 4 loss shares are from its pre-seal diagnostic run (training records only).

- **Zero-shot vs persistence.** Zb and Bb within +/- 0.05 of persistence on all three. On the gull every pooled transformer in Tier 3 sat within 0.03 of it (Z9 −0.0074, Zd3 −0.0035, Zd1 −0.0049, sec 18.3). Tier 4's loss concentrated in long-range soaring fliers (Z26: turkey vulture 27.8% of loss from 7.6% of draws, stork 18.0% from 14.9%, *T. chrysostoma* 5.8% from 0.86%, *D. exulans* 3.1% from 0.97%), and all four are in the bird pool. By domain, Z26's loss was flying 93.7 / terrestrial 4.8 / marine 1.5% from draws of 65.4 / 25.0 / 9.6% (median across seeds, rescaled to 100%).
- **Zero-shot vs S.** The specialist wins on the gull by more than 0.05 (S +0.0498, LSTM-S +0.1616, against Z9 −0.0074, sec 18.4), and on the goose and the dove.
- **Zb vs Bb.** Within noise on all three. The package does not change which sources dominate the loss: stork 64% of Z9's and 67% of B9's (Tier 3 freeze sec 7.1); turkey vulture 27.8% of Z26's and 25.0% of B26's.
- **Congeners.** The goose falls in the genus row (both $\Delta$ above +0.05), because Tier 3's largest effect was within a genus (Zd1, SOES alone, +0.1798 on NES, sec 18.3 and 18.6) and four *Anser* congeners enter at once. Against that, the five congeners are geese and a gull, not the soaring fliers that took Tier 4's loss. The gull (one congener) and the dove (no congener) are within noise.
- **Fb vs Sk.** Sk wins on most draws for all three birds. F9 lost to Sk on 17 of 18 draws (sec 18.2, prediction 3), and pool composition moved the initialisation, not the outcome (prediction 4).

## 7. As trained

WILL FILL after training. The evaluator refuses to open any test partition until this section is filled and committed.

## 8. Attestation

This file was committed before any bird study model was trained or any test partition of this study was read.

