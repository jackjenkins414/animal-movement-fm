# GenAI reference: I used GenAI to help draft the training entry point for NCI Gadi and it mirrors my previous
# established run_transformer_real.py pipeline (same split, same segment_id grouping), but is adapted for GPU 
# execution and the first convergence fix iteration. Specifically I wrote the docs, congfigs and a rough main
# method which was polished and check by GenAI. 
"""
Transformer training on Gadi (real stork data, full scale); convergence iteration 1.

Purpose
-------
As of now, our Transformer does not converge at 6h under the current constant-LR, unclipped recipe. This run applies the 
standard transformer-stabilisation trio as a first principled attempt:

  - lower peak LR      (3e-4, down from 1e-3)
  - warmup + cosine    (warmup_frac=0.1; re-enabled)
  - gradient clipping  (max global norm 1.0; new capability)

Architecture is held identical to the previous runs (d_model=64, nhead=4, num_layers=2) so any change in behaviour is 
attributable to the optimisation recipe, not the model.

This is iteration 1, not a claimed fix. The signal to watch is the per-epoch `train_mse` in the log: the LSTM converges 
to ~0.0179, so the target is to reach <= that and keep descending smoothly (no divergence / oscillation). The end of run 
test summary is a secondary, human-readable check.

Differences from run_transformer_real.py (deliberate for Gadi):
  - device auto-selects cuda (the whole point of running here).
  - paths are repo-relative, so they resolve on /g/data rather than $HOME.
  - OMP_NUM_THREADS is NOT pinned to 1 (that guard is for the torch+xgboost test path only).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(message)s")

# repo relative paths; this file is <repo>/experiments/train_transformer_gadi.py, so parents[1] is the repo root wherever
# the checkout lives (Mac or /g/data on Gadi).
REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
CKPT = REPO / "checkpoints" / "transformer_stork_convergence_iter1.pt"

# convergence fix hyperparameters (iteration 1)
CONFIG = dict(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    context_length=24,
    epochs=80,
    batch_size=256, # larger than the old 64: fewer, less-noisy updates -> steadier descent, better GPU use
    lr=3e-4,   # lower peak LR
    warmup_frac=0.1, # linear warmup then cosine decay
    grad_clip=1.0, # cap global grad norm
)


def main() -> None:
    print("=" * 70)
    print("Transformer convergence run (iteration 1)")
    print(f"device available : cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu              : {torch.cuda.get_device_name(0)}")
    print(f"data             : {SEGMENTS}")
    print(f"checkpoint out   : {CKPT}")
    print("config           :")
    for k, v in CONFIG.items():
        print(f"    {k:<16}= {v}")
    print("=" * 70)

    if not SEGMENTS.exists():
        raise SystemExit(f"FAIL: data not found at {SEGMENTS}")

    # Same split as the established real runs: reproducible seed, gap-respecting segments.
    counts = window_counts_per_individual(SEGMENTS)
    split = make_split(counts, strata=None, min_windows=100, seed=42)

    print("loading train and test frames...")
    train_frame = load_split_frame(SEGMENTS, split.train)
    test_frame = load_split_frame(SEGMENTS, split.test)
    print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
    print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

    # device left as default (None) -> the model auto-selects cuda on Gadi.
    model = Transformer(**CONFIG)

    print(f"\ntraining for {CONFIG['epochs']} epochs...")
    t0 = time.time()
    model.fit(train_frame, group_col="segment_id")   # gap-respecting grouping
    fit_min = (time.time() - t0) / 60
    print(f"  fit done in {fit_min:.1f} min on device={model.device}")

    model.save(CKPT)
    print(f"  saved checkpoint -> {CKPT}")

    print("\nevaluating on held-out test storks...")
    results = evaluate(
        model,
        test_frame,
        context_length=CONFIG["context_length"],
        individual_col="segment_id",
        show_progress=False,     # no tqdm bar in a batch log
        extra_cols=("individual_id",),
    )
    print(f"\n{len(results):,} predictions")
    print(stratified_summary(results).to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
