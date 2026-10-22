# Repaired transformer on the Tier 2–4 pools (exploratory)

This is exploratory and never a tiered result. Written on 23-10-2026, before any run of this plan. Committed before any run, and then committed again with sec 7 filled before the sealed evaluation.

## 1. What we already know

- The 1 h studies showed the default transformer (Post-LN, input x 16) collapses in training, and that our repair (Pre-LN, no x 16, signed log1p on the step inputs) fixes it. Repaired, it matched LSTM-Bb on the 1 h birds (`results/birds_1h_bbf_sealed/`).
- Every Tier 2–4 transformer used the old encoder. In Tier 4, B26 sat at persistence on all three held-out species, while LSTM-B26 and XGB-B26 reached the gull and NES (`results.md` sec 19).
- All Tier 2–4 sealed results are known. The gull, wolf and NES test partitions have been opened before, and this study opens them again.
- Before this freeze we ran a dry run and one calibration (B26F seed 0), and neither opened a test partition. Every config differs from its twin only in the three switches, and starting weights match the twin's at each seed. Retraining B26 seed 0 for 1,000 updates reproduced its Tier 4 record exactly.

## 2. Question

Was the Tier 2–4 transformer's failure to transfer the training collapse, or is it real? With the repair on, does the pooled transformer move off persistence and catch up with LSTM and XGBoost on the same pools?

## 3. Design

We retrain five pooled plain-trunk transformers with the repair switched on. Each arm keeps its twin's pool and recipe, and only the encoder changes.

- **B3F**: Tier 2's B pool (3 sources), scored on gull, wolf and NES; twin B.
- **B9F**: Tier 3's B9 pool (9 sources), scored on all three; twin B9.
- **B26F**: Tier 4's B26 pool (26 sources), scored on all three; twin B26.
- **BdMF**: Tier 4's 7 marine sources, scored on NES; twin BdM.
- **BdTF**: Tier 4's 5 terrestrial sources (Zd-T's pool) as a plain trunk, scored on the wolf. It has no twin, so it is read against P, B26F and Zd-T only.

That makes 25 runs (5 seeds each). Everything else is the Tier 3/4 recipe: 6 h grid, grid index 11, C = 12, 20,000 updates at batch 256, final checkpoint, gpuhopper, FP32 with TF32 off. The switches are the repair probe's V4, the same as BbF. One difference from the 1 h BbF: the 6 h data has no $\pm$ 1,000 clip, so signed log1p sees the raw scaled steps. The twins saw the same unclipped data, so the comparison with them stays clean. The largest scaled step is about 13,000 before the transform and about 9.5 after.

Only these 25 runs are trained. P, the specialists, LSTM-B26, XGB-B26 and every twin are read from their sealed prediction files. Driver: `experiments/training/tier4_repaired_train.py`.

## 4. Scoring

Tier 4's scoring, unchanged: travel p90 skill against P at F = 25, median of 5 seeds, paired bootstrap of 2,000 at seed 42, and $\pm$ 0.05 counts as no difference. Each carried arm is tied back to its own sealed report.

## 5. Check before the seal

After training and before the sealed evaluation, a diagnostic runs on training and validation windows only. It gives loss against the zero predictor, how much the output moves when the last token is swapped, and |pred| / |actual|, for the new arms and their twins. We read an arm as still collapsed if its loss is within 1% of the zero predictor's and swapping the last token barely moves its output, which is what the 1 h Bb looked like. This is descriptive and does not gate the evaluation.

## 6. Expectations

- The repair lifts each arm above its old-encoder twin by more than 0.05 on the gull and NES (B3F − B, B9F − B9, B26F − B26, and BdMF − BdM on NES).
- B26F is within $\pm$ 0.05 of LSTM-B26 on the gull and NES.
- It does not improve as the pool grows: B3F, B9F and B26F are within $\pm$ 0.05 of each other.
- On NES, BdMF (marine only) is above B26F by more than 0.05.
- The wolf stays flat: every arm is within $\pm$ 0.05 of P on the wolf.

## 7. As trained

TO FILL from the diagnostic before the sealed evaluation.

## 8. Limitations

- The repair was chosen on one fold and one seed of the 1 h birds, and this is its first use at 6 h.
- BdTF has no plain twin, so its repair effect can't be separated from the change of package.
- The held-out test partitions have been opened several times before; there is no multiplicity correction.
- NES and the wolf have 10 test animals each, and the wolf's own specialist only reaches +0.024, so there is little skill on the wolf for any model to find.

Only this study's sealed evaluator scores test partitions for it.
