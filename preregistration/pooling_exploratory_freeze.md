# Pooling on the 26 source pool (exploratory)

**GenAI reference:** GenAI was used to fill in the tables from what we gave it, but prose is our own. 

Exploratory and never reported as a tiered result. Written 01-10-2026, after the Tier 4 and bird rotation seals, and before any pooling run. Committed before any run, then committed again with sec 7 filled before our sealed evaluation. 

## 1. Question and status

At 26 sources the plain trunk sits at persistence on all three held-out species, and on NES it loses the skill a narrower pool gives it. Here we ask does one dense trunk smooth over diverse behaviours, and does domain balanced sampling, MoE routing, or both, stop it?

| | |
|---|---|
| held-out species | *L. fuscus* (gull), *C. lupus* (wolf), *M. angustirostris* (NES), as in our Tiers 2–4 |
| test partitions | already opened; this study scores new arms on them again (table below) |
| reserved | *O. dalli* and *P. adeliae* are in no pool and stay unread |
| pool | Tier 4's 26 sources, `registered/tier4_source_pool.json`, unchanged |

| species | opened before | this study |
|---|---|---:|
| gull | Tier 2, Tier 3, Tier 4, bird study | 5th |
| wolf | Tier 2, Tier 3, Tier 4 | 4th |
| NES | Tier 2, Tier 3, Tier 4 | 4th |

The bird rotation study carried fold 0 from the bird study's sealed predictions and did not re-read the gull's test partition, so it is not counted.

## 2. Pool and sampling

Two rules over the same 26 sources and the same training windows. **A8** is Tier 4's: $\propto \sqrt{n}$ over the pool. **Domain balanced**: each domain gets a third of the draws, split within the domain by A8. The balanced weights are the registered Tier 4 domain sub-pools (`sub_pools.domain_*`) divided by 3. These are the same as the full pool renormalised within each domain (agreement to 1e-16), so we have written nothing new to `registered/`. The driver pins them below and checks them on every load. Domain labels set draw probabilities only; the model never sees them, so every arm is zero-shot on the held-out species. 

| domain | sources | A8 | balanced |
|---|---:|---:|---:|
| flying | 14 | 65.4% | 33.3% |
| terrestrial | 5 | 25.0% | 33.3% |
| marine | 7 | 9.6% | 33.3% |

Passes are draws per training window over one run, 20,000 x 256 draws times the draw share, divided by the windows.

| domain | source | training windows | A8 % | balanced % | passes, A8 | passes, balanced |
|---|---|---:|---:|---:|---:|---:|
| flying | *C. ciconia* | 806,595 | 14.949 | 7.619 | 1 | 0 |
| flying | *C. aura* | 210,939 | 7.645 | 3.896 | 2 | 1 |
| flying | *A. caerulescens* | 199,754 | 7.439 | 3.792 | 2 | 1 |
| flying | *M. americana* | 137,954 | 6.182 | 3.151 | 2 | 1 |
| flying | *H. ostralegus* | 127,354 | 5.940 | 3.028 | 2 | 1 |
| flying | *C. corone* | 111,895 | 5.568 | 2.838 | 3 | 1 |
| flying | *B. canadensis* | 99,724 | 5.256 | 2.679 | 3 | 1 |
| flying | *F. naumanni* | 31,995 | 2.977 | 1.518 | 5 | 2 |
| flying | *M. milvus* | 25,234 | 2.644 | 1.348 | 5 | 3 |
| flying | *P. apivorus* | 22,509 | 2.497 | 1.273 | 6 | 3 |
| flying | *G. himalayensis* | 9,621 | 1.633 | 0.832 | 9 | 4 |
| flying | *D. exulans* | 3,393 | 0.970 | 0.494 | 15 | 7 |
| flying | *T. chrysostoma* | 2,645 | 0.856 | 0.436 | 17 | 8 |
| flying | *T. antarctica* | 2,564 | 0.843 | 0.430 | 17 | 9 |
| terrestrial | *C. elaphus* | 556,579 | 12.418 | 16.555 | 1 | 2 |
| terrestrial | *C. taurinus* | 51,417 | 3.774 | 5.032 | 4 | 5 |
| terrestrial | *B. bison* | 37,729 | 3.233 | 4.310 | 4 | 6 |
| terrestrial | *C. latrans* | 37,253 | 3.213 | 4.283 | 4 | 6 |
| terrestrial | *L. rufus* | 20,206 | 2.366 | 3.154 | 6 | 8 |
| marine | *M. leonina* | 53,076 | 3.835 | 13.320 | 4 | 13 |
| marine | *L. weddellii* | 7,718 | 1.462 | 5.079 | 10 | 34 |
| marine | *A. gazella* | 5,414 | 1.225 | 4.254 | 12 | 40 |
| marine | *E. chrysolophus* | 2,818 | 0.884 | 3.069 | 16 | 56 |
| marine | *O. rosmarus* | 2,460 | 0.826 | 2.868 | 17 | 60 |
| marine | *L. carcinophaga* | 2,286 | 0.796 | 2.764 | 18 | 62 |
| marine | *A. patagonicus* | 1,170 | 0.569 | 1.978 | 25 | 87 |

2,570,302 training windows in all. Balanced draws are probabilistic, not stratified within a batch.

## 3. Arms and runs

A 2 x 2 on the plain trunk (B: no conditioning, no reconstruction head): sampling {A8, domain-balanced} x FFN {dense, MoE}. Tier 4's configuration and budget: grid 11 (`d_model` 256, 4 layers, LR 1e-3), 20,000 applied updates at batch 256, final checkpoint, no validation selection.

| arm | sampling | FFN | seeds | plan indices | parameters | runs |
|---|---|---|---|---|---:|---:|
| B26 (carried) | A8 | dense | 0–4 | Tier 4 5–9 | 2,112,258 | — |
| B26bal | balanced | dense | 0–4 | 0–4 | 2,112,258 | 5 |
| B26moe | A8 | MoE | 0–4 | 5–9 | 5,279,490 | 5 |
| B26balmoe | balanced | MoE | 0–4 | 10–14 | 5,279,490 | 5 |
| B26wide (conditional, sec 5) | A8 | dense, width 2,048 | 0–4 | 15–19, appended | 5,264,130 | 0 or 5 |

B26 is our A8 dense cell, which is carried from Tier 4's sealed prediction files. B26bal's configuration is B26's exactly; only the draw probabilities differ. The MoE arms differ from B26 in the trunk and nothing else.

**MoE.**

| | |
|---|---|
| where | the feedforward block of all four encoder layers; attention, norms, dropout and the rest of the trunk as B26 |
| experts, k | 8 experts, top 2 per token; gates are the top 2 router probabilities renormalised to sum to 1 |
| expert | Linear(256, 256), ReLU, dropout, Linear(256, 256); hidden 256 = B26's 512 / k |
| routing | token level; the router is Linear(256, 8), no bias, on the token state only; no species or domain input |
| parameters per layer | dense FFN 262,912; MoE 1,054,720 in total, 265,216 active per token (+0.9% on dense) |
| auxiliary loss | $E \cdot \sum_e f_e P_e$ per layer, averaged over layers; $f_e$ is expert $e$'s share of the top k assignments and $P_e$ its mean router probability; 1 at uniform routing |
| auxiliary weight | 1e-2; following Fedus et al (2022), who use $\alpha = 10^{-2}$ for the Switch Transformer load balancing loss |
| initialisation | the four layers start as copies of one layer, as `nn.TransformerEncoder` does for B26; experts within a layer start independently |
| computation | every expert on every token, gated; no capacity limit, no dropped tokens |
| routing record | per source, 2,048 training windows evenly spaced through its training windows; per layer, mean router probability and top k share per expert, and the share of a window's tokens on its most common top-1 expert |

| | |
|---|---|
| driver | `experiments/training/pooling_train.py`, `.pbs`; records tagged `pooling`, `exploratory: true`; model in `src/animal_movement_fm/models/moe.py` |
| hardware | gpuhopper, 1 GPU, 12 CPUs, fp32 (no autocast, no TF32 setting), `math` attention backend, as Tier 4 trained B26; walltime 01:30:00 per subjob (Tier 4's longest run 21m12s) |
| order | index 5 (B26moe seed 0) alone first, to measure the MoE trunk's cost; then the rest in blocks of at most ten |
| compute | about 210–350 SU: B26bal about 70 (5 x 13.9, Tier 4's mean per run); MoE runs 1–2 x that each, uncalibrated until index 5 reports. B26wide, if run, about 70–140 more (its FFN does the same work as the MoE's all-expert pass) |

## 4. Statistic

As `tier4_exploratory_freeze.md` sec 3: travel p90 skill against persistence at F = 25, median of five seeds, paired bootstrap of 2,000 resamples at seed 42, a gap within $\pm$ 0.05 we will read as no difference. Comparators are read from the Tier 3 and Tier 4 sealed prediction files.

## 5. Readings (descriptive, no pass/fail)

Every row is per species: gull, wolf, NES. A difference is left minus right, with intervals from the paired bootstrap.

| reading | what is reported |
|---|---|
| each arm against P | skill, with interval |
| each arm − B26 | the gap, with interval |
| sampling main effect | B26bal − B26 |
| MoE main effect | B26moe − B26 |
| interaction | (B26balmoe − B26bal) − (B26moe − B26), interval from the same resamples |
| NES against narrower and other rungs | each arm against BdM +0.1457, B9 +0.1272, LSTM-B26 +0.1228, XGB-B26 +0.1728 |
| expert use by domain | from the routing record, training windows only: top k share per expert, per layer, per domain (each domain's sources weighted equally) |
| dilution | per source and per domain loss shares against draw shares, as Tier 4 freeze sec 4, for the three new arms |


## 6. Pre run predictions


| reading | prediction | basis |
|---|---|---|
| B26bal − B26 | above +0.05 on NES; gull and wolf within $\pm$ 0.05 | marine draws rise from 9.6% to 33.3%; B26 took 1.2% of its loss from marine and the all marine BdM reaches +0.1457 on NES |
| B26moe − B26 | within $\pm$ 0.05 on all three | the router sees the same window as the dense trunk, and A8 still gives flying about 95% of the loss, so experts would split flying behaviour rather than serve marine. Of the Transformer arms on all 26 sources, only B26dom moved beyond noise, on NES (+0.0762 against B26's +0.0016) |

## 7. As trained

WILL FILL.

## 8. Limitations

- **Repeat passes under balancing.** Small marine sources are seen many more times per window: *A. patagonicus* about 87 times per run (25 under A8), *L. carcinophaga* 62, *O. rosmarus* 60, *E. chrysolophus* 56. A balanced gain on NES may carry some memorisation of these few animals. 
- **MoE has 2.5 x B26's total parameters** (5,279,490 against 2,112,258) at matched active parameters. A MoE gain may be capacity rather than routing; the sec 5 control exists for that case.
- **`loss_history` includes the auxiliary term** for the MoE arms (about 0.01 at balanced routing), so it is not directly comparable with B26's. The dilution loss shares are the forecast loss only.
- **Layers start as identical copies in both trunks.** This is `nn.TransformerEncoder`'s behaviour and holds for every arm built on `MovementFM`'s Transformer trunk; the MoE trunk follows it so the two trunks share the convention.
- **Equivalence of B26's path is shown on CPU, not Gadi.** The golden test runs on the Mac at one thread; CPU initialisation depends on the thread count, and Gadi builds under `OMP_NUM_THREADS=12`.
- **Test partitions re-opened** Just is the way it is. 

## 9. Note

No pooling run existed when this file was first committed. Only the pooling sealed evaluator (not yet written) opens the held-out test partitions, after this file is committed with sec 6 and sec 7 filled.
