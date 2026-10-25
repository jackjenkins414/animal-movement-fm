# Repaired transformer on the rotating held-out birds (exploratory)

Exploratory, never a tiered result. Written on 25-10-2026, before any run.

## 1. What we already know

- All rotation (`results.md` sec 22), BbF and Tier 2–4 rerun (sec 25) results are known.
- Every test partition scored here was opened before.
- An exploratory follow-up, not confirmatory evidence. Design, switches, comparisons and diagnostic are fixed before training.
- Committed before any run, and again with sec 7 filled before the seal.
- Dry run (25-10): all ten pools and counts as registered, configs differ from the originals in the switches alone, initial weights equal but for encoder.norm, and every original's record, checkpoint and sealed predictions present. Calibration (fold 9, seed 0, 1,000 updates): BbF6 and ZbF6 train, and their originals rerun on H200 match the V100 records closely but not exactly (max logged-loss difference 0.012 and 0.058). Neither opened a test partition.

## 2. Question

Repaired, how do the full and plain transformers compare on the 30 rotation birds with persistence, their originals, each other and pooled LSTM and XGBoost?

## 3. Design

- BbF6 and ZbF6 are Bb and Zb with the repair switches: no $\times \sqrt{d}$ input scale, Pre-LN with a final LayerNorm, signed log1p on the two step channels.
- 60 runs: folds 0–9, seeds 0–2, each arm.
- Fold 0 trains on the bird study's main pool, as its originals did.
- Otherwise as `birds_rotation_exploratory_freeze.md` secs 2–3.
- The transform acts on inputs only; ZbF6's reconstruction targets and losses are unchanged.
- gpuhopper, FP32, TF32 off.
- Compute: 0.025 s (BbF6) and 0.030 s (ZbF6) per update, about 0.14–0.17 h per run, roughly 1,000–1,200 SU for the 60.

## 4. Scoring

Scoring is the rotations, unchanged (its freeze, secs 4 and 7). Per bird:

1. BbF6 and ZbF6 against P.
2. BbF6 − Bb and ZbF6 − Zb.
3. ZbF6 − BbF6.
4. BbF6 and ZbF6 against LSTM-Bb and XGB-Bb.
5. LSTM-Bb − LSTM-S and XGB-Bb − XGB-S, from the rotation report, for context.

Fold 0's originals enter at seeds 0–2 (XGB-Bb seed 0); the published five seed values are shown apart.

## 5. Check before the seal

A diagnostic on training and validation windows only gives loss against the zero predictor, the response to swapping the last token, and |pred| / |actual|, for the 60 runs and their originals. A run is still collapsed if its loss is within 1% of the zero predictor's and the swap barely moves its output. Any change to the repair or recipe needs a dated amendment here before test evaluation.

## 6. Expectations

- Both repairs help: BbF6 − Bb and ZbF6 − Zb are above +0.05 on most of the 30 birds.
- At C = 12, repaired plain does not catch the pooled baselines: the median of BbF6 − LSTM-Bb and of BbF6 − XGB-Bb across the 30 birds is below −0.05. (I expect longer contexts would close this, but they are not tested here.)
- The package still costs: ZbF6 − BbF6 is negative on most of the 30 birds.

## 7. As trained

To fill after training, before the sealed evaluation.

## 8. Limitations

- The repair was chosen on one fold and seed at 1 h.
- ZbF6 reconstructs raw-scale steps from log compressed inputs: a harder task, same objective.
- Originals ran on V100, repaired arms on H200; repaired − original differs in hardware too.
- Three seeds; intervals conditional on headline fits omit training seed uncertainty.
- Test partitions reopened; no multiplicity correction.
