"""
LSTM training on Gadi (real stork data, full scale); matched recipe run for the head to head.

Purpose
-------
Produce a converged LSTM checkpoint trained under a recipe matched to the converged Transformer (iter-1), so the raw-sequence 
Transformer vs LSTM comparison is apples to apples and any difference is then attributable to architecture (attention vs 
recurrence), not to training budget. The existing on disk checkpoint (`lstm_stork_h64_l1_e60.pt`) was trained pre-Gadi at 60 
epochs / batch 64, so it is not budget matched to the 80 epoch / batch 256 Transformer.

What is held identical to the Transformer run:
  - data + split (same seed=42 make_split, grouped by segment_id)
  - context_length=24, epochs=80, batch_size=256
  - Adam, MSE loss, GPU preload in fit()

What differs, deliberately, and is the LSTM's conventional setup:
  - constant learning rate (no warmup/cosine schedule)
  - no gradient clipping
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.lstm import LSTM

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
CKPT = REPO / "checkpoints" / "lstm_stork_convergence_matched.pt"


CONFIG = dict(
    hidden_dim=64,
    num_layers=1,
    context_length=24,
    epochs=80,   # matched to the Transformer run
    batch_size=256,  # matched to the Transformer run (was 64 pre-Gadi)
    lr=1e-3,    # the LSTM's established constant LR; no warmup/cosine, no clipping
)

# Note this main method was copied from tran_transformer_gadi.py and just modified slightly for the LSTM 
def main() -> None:
    print("=" * 70)
    print("LSTM matched-recipe convergence run (for the raw-sequence head-to-head)")
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

    counts = window_counts_per_individual(SEGMENTS)
    split = make_split(counts, strata=None, min_windows=100, seed=42)

    print("loading train and test frames...")
    train_frame = load_split_frame(SEGMENTS, split.train)
    test_frame = load_split_frame(SEGMENTS, split.test)
    print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
    print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

    model = LSTM(**CONFIG)

    print(f"\ntraining for {CONFIG['epochs']} epochs...")
    t0 = time.time()
    model.fit(train_frame, group_col="segment_id")  
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
        show_progress=False,    
        extra_cols=("individual_id",),
    )
    print(f"\n{len(results):,} predictions")
    print(stratified_summary(results).to_string(index=False))
    print("\nDONE.")


if __name__ == "__main__":
    main()
