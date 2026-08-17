# GenAI reference: Again the main Transformer training/evaluation logic I implemented and just used GenAI 
# to improve print statements, checkpoint reporting, and some comments.
"""Train the Transformer on the real stork train split (60 epochs), evaluate on test.

Matches the locked synthetic recipe exactly (60 epochs, warmup_frac=0.1 cosine schedule) so the real-data 
Transformer is directly comparable to the synthetic ladder AND to the real LSTM run; the head-to-head that 
tests whether attention separates from recurrence at real-data scale (916k windows vs ~7k synthetic). 
Checkpointed after fit again too. 
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"   # segafault guard addition

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.transformer import Transformer

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"
CKPT = Path.home() / "animal-movement-fm" / "checkpoints" / "transformer_stork_d64_l2_constlr_e60.pt"

counts = window_counts_per_individual(SEGMENTS, context_length=24)  # Tier 1, pre A6
split = make_split(counts, strata=None, min_windows=100, seed=42)

print("loading train and test frames...")
train_frame = load_split_frame(SEGMENTS, split.train)
test_frame = load_split_frame(SEGMENTS, split.test)
print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

# identical to the synthetic ladder Transformer
model = Transformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                    dropout=0.1, context_length=24, epochs=60, batch_size=64,
                    lr=1e-3, warmup_frac=0.0, device="cpu")  # was 0.1; now constant LR

print("\ntraining Transformer for 60 epochs (~45+ min expected)...")
t0 = time.time()
model.fit(train_frame, group_col="segment_id") # gap-respecting grouping
fit_min = (time.time() - t0) / 60
print(f"  60-epoch fit done in {fit_min:.1f} min")

model.save(CKPT)
print(f"  saved checkpoint -> {CKPT}")

print("evaluating on held-out test storks...")
results = evaluate(model, test_frame, context_length=24,
                   individual_col="segment_id", show_progress=True,
                   extra_cols=("individual_id",))

print(f"\n{len(results):,} predictions")
print(stratified_summary(results).to_string(index=False))