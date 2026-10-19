# Repaired transformer (BbF) for the 1 h pooled birds (exploratory)

**GenAI reference:** I specified the design, the comparisons, the name, the C = 96 fallback and the hard stop, and wrote the sec 6 expectations, which are mine. GenAI  wrote the tables and numbers taken from the repair probe, the 1 h freeze and the env freeze. 

To adhere to our prereg, this is exploratory and never reported as a tiered result. BbF is **Bb with the repaired encoder (the repair probe's V4 switches)**. Written on 19-10-2026, before any run of this plan. Committed before any run, then committed again with sec 7 filled before the sealed evaluation.

## 1. Status and what has been opened

| | |
|---|---|
| status | exploratory; no number here is a Tier 2 or Tier 3 result |
| runs of this plan trained | none. No calibration was run for this plan (sec 3) |
| how the repair was chosen | the transformer repair probe (`notes.md` 17-10 plan, 19-10 results; `results/transformer_repair/validation/transformer_repair.md`): fold 6, seed 0, 14 runs on gpuhopper FP32, validation windows only, no test partition read. V4 was chosen because it was the strongest on that fold's validation windows, so the selection is one fold and one seed. Read at selection: 6 h validation travel p90 skill against P of +0.12 / +0.12 (pool / held-out) at C = 12 and +0.19 / +0.27 at C = 96, against LSTM-Bb's +0.10 / +0.11 and +0.15 / +0.18; at C = 12 V4's held-out validation loss was 1.01 of the zero predictor's (pool 0.976) |
| known before this freeze | the 1 h main study's and ablation's sealed results (`results/birds_1h_sealed/`), in which Bb collapsed toward persistence; the transformer collapse diagnosis (`notes.md` 17-10; `results/transformer_diagnosis/`, validation only); the repair probe above; the env study's design, freeze (secs 1–6 at `3886c05`, sec 7 at `b698cfa`) and run records |
| env study | at this commit, the env sealed evaluation had been submitted, but its results had not been read. Its BbR, LSTMBbR and LSTMBbE predictions are comparators here (sec 3) |
| test partitions | no test ids, windows or targets were read for this study. All 30 birds' test partitions were opened before it (1 h freeze sec 1) |
| other studies | the 1 h and env studies are sealed separately; this study edits none of their files and reads only their saved predictions, run records and reports |
| reserved | *O. dalli* and *P. adeliae* are not in this set and stay unread |

## 2. Question

With the collapse removed, does the pooled transformer beat LSTM-Bb and XGB-Bb on test at C = 12 and C = 96? How much does the repair move Bb on test? And does the repaired transformer gain more from ERA5 weather than LSTM-Bb does?

## 3. Design

Folds, pools, exclusions, recipes, encoding, k_h, heads, seeds, the 20,000 update budget and the final checkpoint are the 1 h main study's (`birds_1h_exploratory_freeze.md` secs 2 and 3), unchanged. BbF differs from Bb only in the three switches (sec 5). BbFE is BbF plus the env study's 15 ERA5 channels, exactly as that study's BbE (`TrainConfig.env_channels` = 15). LSTMBbR96 is LSTM-Bb at C = 96, unchanged, rerun on gpuhopper.

| arm | C | base (1 h study) | per | seeds | plan indices | runs | queue | compared with |
|---|---:|---|---|---|---|---:|---|---|
| BbF | 12 | Bb | fold (0–9) | 0–2 | 0–29 | 30 | gpuhopper | BbR (env study, gpuhopper) |
| BbF | 96 | Bb, C = 96 (ablation) | fold | 0–2 | 30–59 | 30 | gpuhopper | Bb, C = 96 (ablation, gpuvolta) |
| BbFE | 12 | Bb + env (env study's BbE) | fold | 0–2 | 60–89 | 30 | gpuhopper | BbF |
| LSTMBbR96 | 96 | LSTM-Bb, C = 96 (ablation) | fold | 0–2 | 90–119 | 30 | gpuhopper | LSTM-Bb, C = 96 (ablation, gpuvolta) |
| **total** | | | | | | **120** | | |

*GenAI wrote this below to save me time*
- **Precision.** Every run here is full FP32 (TF32 off in cuDNN and matmuls, recorded per run), as in our env study.
- **Comparators are reloaded.** Comparators are reloaded, not retrained or re-predicted. Every comparator comes from its own study's saved test predictions: Bb, LSTM-Bb and XGB-Bb at C = 12 from the 1 h main study, at C = 96 from the 1 h ablation, and BbR, LSTMBbR and LSTMBbE from the env study. Each file's sha256 is recorded at prediction, and each reused arm's skill is tied back to its own sealed report (sec 4).
- **BbFE's twin.** BbFE's pool, env files and env_stats must equal its BbE twin's (env study, same fold and seed); the driver refuses otherwise.
- **Repeats.** BbF at fold 6, seed 0 (indices 18 and 48) and LSTMBbR96 at fold 6, seed 0 (index 108) have exactly the configs and seeds of the probe's BbV4, BbV4 at C = 96 and LSTM-Bb V0 at C = 96. They are retrained, not reused, and their logged losses are compared with the probe's in sec 7 (no pass/fail).
- **Fallback.** LSTMBbR96 is submitted last. If it is not complete by 25-10-2026, the C = 96 LSTM comparison uses the 1 h ablation's LSTM-Bb (V100) runs, and the report says so. The evaluator's `--c96-fallback` is refused before 25-10 and when LSTMBbR96 is complete, and is recorded in the report.

| | |
|---|---|
| driver | `experiments/training/bird_1h_bbf_train.py`, `bird_1h_bbf_train.pbs` (gpuhopper). It imports the repair driver's V4 (pinned in the driver), the env driver's pool, statistics and guards, and the 1 h driver's configs. Records are tagged `birds_1h_bbf`, `exploratory: true`, `opened_test_partition: false`, with the switches, `repeats`, `env_reference`, the TF32 flags and the freeze attestation. The driver refuses test partitions, the wrong queue, TF32 on, any step budget but 20,000, and any run before this freeze is committed (except `--dry-run` and `--calibrate`) |
| resources | walltime 01:00:00, 48 GB, 12 CPUs, 1 GPU (gpuhopper) |
| compute | at most about 3.2 KSU for the 120 runs (below); evaluation uncalibrated, one gpuhopper job at a provisional 04:00:00 (≤ 360 SU) |

**Compute estimate.** No calibration was run for this plan. The figures come from earlier calibrations at fold 6 (the largest pool), at 90.0 SU per walltime hour on gpuhopper (env freeze sec 3):
- BbFE takes the env study's BbE H200 calibration.
- BbF at C = 12 takes the same figure as a bound; it reads 12 channels, not 27.
- BbF and LSTMBbR96 at C = 96 take the 1 h study's gpuvolta C = 96 extrapolations (1 h freeze sec 3), as upper bounds.

The probe trained all three configs on gpuhopper; its PBS walltimes are not in the run records and go in sec 7.

| arm | runs | h per run | SU per run | SU | source |
|---|---:|---:|---:|---:|---|
| BbF, C = 12 | 30 | ≤ 0.22 | ≤ 19.7 | ≤ 592 | env BbE calibration, H200 |
| BbF, C = 96 | 30 | ≤ 0.52 | ≤ 46.8 | ≤ 1,404 | 1 h Bb C = 96 extrapolation, V100 |
| BbFE | 30 | 0.22 | 19.7 | 592 | env BbE calibration, H200 |
| LSTMBbR96 | 30 | ≤ 0.21 | ≤ 18.9 | ≤ 567 | 1 h LSTM-Bb C = 96 extrapolation, V100 |
| **total** | **120** | | | **≤ 3,155** | |

## 4. Comparisons and scoring

Scoring is as in our 1 h freeze sec 4, unchanged; travel p90 skill against P per bird and horizon, individual weighted, median across seeds (XGBoost single seed); paired bootstrap of 2,000 resamples at seed 42; $\pm$ 0.05 read as no difference. The floor, scored rule, common set, travel boundary, latitude rule and scoring rules are the same, so the scored birds are the 1 h main study's: 28 at 1 h, 27 at 6 h, 28 at 24 h, and the common set is 27. Every arm is scored on the 1 h main eval's test_scored windows. The C = 96 arms are scored on the same windows, and P is checked bitwise identical across C and against the reused studies' P. Evaluator: `experiments/reporting/bird_1h_bbf_sealed_eval.py`, which imports the 1 h and env evaluators for all of the above.

| reading | definition | hardware |
|---|---|---|
| BbF − LSTM-Bb, C = 12 | BbF − LSTMBbR (env study); BbF − LSTM-Bb (main study) alongside | same (H200 FP32); the main-study row is V100, read with the env study's hardware check |
| BbF − LSTM-Bb, C = 96 | BbF − LSTMBbR96; BbF − LSTM-Bb (ablation) alongside, or as the only row under the fallback | same (H200 FP32); the ablation row is V100-trained (predicted on H200 FP32) |
| BbF − XGB-Bb | at C = 12 (main study) and C = 96 (ablation) | XGBoost on the CPU queue |
| repair on test | BbF − BbR at C = 12 (same hardware), BbF − Bb (main) alongside; BbF − Bb at C = 96 (ablation, V100) | as stated |
| weather | BbFE − BbF | same |
| contrast | (BbFE − BbF) − (LSTMBbE − LSTMBbR), the interval from the same resamples (each arm's headline seed, the same draws and cached p90s) | same |
| hardware, differences only | LSTMBbR − LSTM-Bb and BbR − Bb at C = 12; LSTMBbR96 − LSTM-Bb at C = 96 | gpuhopper rerun − V100 run; medians only, no bands, no pass/fail |
| skill against P | every arm above, per bird × horizon | — |
| tie-back | each reused arm's skill here − its own sealed report's (C = 96: the 1 h ablation's "all" stratum) | max \|diff\| and median per arm; no pass/fail |

Each comparison is reported per bird x horizon with metre and skill intervals. Across birds we use the median, the counts above +0.05 / within / below −0.05, and the intervals excluding 0, over the scored birds and the common set. Medians are repeated over both sets without fold 6's three held-out birds (*A. platyrhynchos*, *M. americana*, *P. occidentalis*), because fold 6 is where the repaired encoder was chosen. That repeat is descriptive only. No threshold or p value decides any reading.

## 5. The repaired encoder and covariates

| switch | every earlier arm | BbF and BbFE |
|---|---|---|
| `input_scale` | input projection × √d_model (× 16) | `none` |
| `norm_first` | Post-LN encoder layers | Pre-LN layers, plus a final LayerNorm before the heads |
| `step_transform` | `none` | `signed_log1p`: sign(x) log1p(\|x\|) on `d_east_scaled` and `d_north_scaled` of every token, after the ±1,000 clip; every other channel, the env channels included, is untouched |

The loss, the heads, k_h and every other config field are Bb's in the 1 h main study. `signed_log1p` acts on the model's inputs only; the targets are unchanged.

BbFE's covariates are the env study's (env freeze sec 5), unchanged: 15 channels at the fix each token steps to; env_stats from the run's training pool only (window weight p_s / n_s); every standardised channel clipped at $\pm$ 10 SD. They are appended after the 12 movement channels, so the step transform never touches them.

## 6. Expectations

| comparison (6 h unless stated) | expectation | reason |
|---|---|---|
| BbF − BbR (C = 12) and BbF − Bb (C = 96) | above +0.05 | the repair removes a collapse; the probe gained +0.09 and +0.19 |
| BbF − LSTM-Bb, C = 12 and C = 96 | above +0.05 | with the collapse removed, attention over the full history should use it better than the LSTM's single recurrent state; the probe was ahead at C = 96 (+0.04 pool, +0.09 held-out) |
| BbF − XGB-Bb, C = 12 and C = 96 | within $\pm$ 0.05 | the probe was within or near $\pm$ 0.05 on one fold |
| any comparison at 1 h or 24 h | within $\pm$ 0.05 | little skill exists at those horizons |
| BbFE − BbF | within $\pm$ 0.05 at 1 h; median #\geq$ 0 at 6 and 24 h | as the env freeze |
| (BbFE − BbF) − (LSTMBbE − LSTMBbR) | above +0.05 at 6 h; within $\pm$ 0.05 at 1 h and 24 h | attention can weigh each token's weather against its movement, where the LSTM folds both into one state, so the repaired transformer should gain more from weather |

## 7. As trained

To fill from `bird_1h_bbf_sealed_eval.py --diagnostic` before the sealed evaluation, with the PBS walltimes and SU from the job logs and the probe's.

## 8. Limitations and hard stop

- **Selection.** The repaired encoder was chosen on one fold's validation windows (fold 6, seed 0). Test here is new data for the choice, but the choice itself is not cross-validated. The fold 6 rows in sec 4 show how much the scored medians lean on that fold.
- **Hardware.** The C = 96 Bb and XGB-Bb comparators, and the main-study C = 12 rows, were trained on other hardware (gpuvolta, or the CPU queue for XGBoost). Under the fallback, the C = 96 LSTM comparison is V100 against H200, and the env study's hardware check was measured at C = 12 only.
- **Untested.** The loss itself as a pull toward zero is untested; the repair changes the encoder and the input scale, not the objective.
- **Probe numbers.** The probe's validation skills are on a capped sample without the floor, and are not comparable to any sealed number.
- **BbFE.** BbFE inherits the env study's limitations (env freeze sec 8).

## 9. Note

No run of this plan existed when this file was first written. Only `bird_1h_bbf_sealed_eval.py` scores test partitions for this study.
