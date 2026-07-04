# GenAI reference: Again, I used GenAI to refine progress print statements and make minor readability improvements
"""Smoke test: confirm the LSTM trains and evaluates on real stork data.

NOT a real benchmark -- 2 epochs only. Success = (1) no segfault with torch+xgboost imported, (2) train loss drops 
epoch 1 -> 2, (3) evaluate returns a finite median. The post-2-epoch error will be bad; that is expected and fine.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1" # segfault guard

import logging
import time
from pathlib import Path

# Surface the per-epoch train_mse lines from lstm.py so we can see loss drop.
logging.basicConfig(level=logging.INFO, format="%(message)s")

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.lstm import LSTM

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"

counts = window_counts_per_individual(SEGMENTS)
split = make_split(counts, strata=None, min_windows=100, seed=42)

print("loading train and test frames...")
train_frame = load_split_frame(SEGMENTS, split.train)
test_frame = load_split_frame(SEGMENTS, split.test)
print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

# 2 epochs only — smoke test.
model = LSTM(hidden_dim=64, num_layers=1, context_length=24,
             epochs=2, batch_size=64, lr=1e-3, device="cpu")

print("\ntraining LSTM for 2 epochs (smoke test)...")
t0 = time.time()
model.fit(train_frame, group_col="segment_id") # gap-respecting grouping
print(f"  2-epoch fit done in {time.time() - t0:.0f}s")

print("evaluating on held-out test storks...")
results = evaluate(model, test_frame, context_length=24,
                   individual_col="segment_id", show_progress=True,
                   extra_cols=("individual_id",))

print(f"\n{len(results):,} predictions (2-epoch model — expect poor numbers)")
print(stratified_summary(results).to_string(index=False))