# GenAI reference: I implemented the core LSTM training/evaluation logic and asked GenAI to improve terminal progress 
# messages, checkpoint reporting, and minor code/comment cleanup
"""Train the LSTM on the real stork train split (60 epochs), evaluate on held-out test.

60 epochs matches the locked synthetic recipe so the real-data LSTM is directly comparable to the synthetic 
ladder and to the Transformer run that follows. The trained model is checkpointed after fit so 
evaluation/fine-tuning never needs a retrain.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"   # before torch import (fix bc of segfault we had on mac)

import logging
import time
from pathlib import Path

# surface lstm.py's per-epoch "train_mse" lines so we can watch convergence
logging.basicConfig(level=logging.INFO, format="%(message)s")

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.lstm import LSTM

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"
CKPT = Path.home() / "animal-movement-fm" / "checkpoints" / "lstm_stork_h64_l1_e60.pt"

counts = window_counts_per_individual(SEGMENTS)
split = make_split(counts, strata=None, min_windows=100, seed=42)

print("loading train and test frames...")
train_frame = load_split_frame(SEGMENTS, split.train)
test_frame = load_split_frame(SEGMENTS, split.test)
print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

# full recipe: 60 epochs, matching the synthetic ladder
model = LSTM(hidden_dim=64, num_layers=1, context_length=24,
             epochs=60, batch_size=64, lr=1e-3, device="cpu")

print("\ntraining LSTM for 60 epochs (~45 min expected)...")
t0 = time.time()
model.fit(train_frame, group_col="segment_id") # gap-respecting grouping
fit_min = (time.time() - t0) / 60
print(f"  60-epoch fit done in {fit_min:.1f} min")

# Checkpoint immediately so the trained weights survive regardless of what follows.
model.save(CKPT)
print(f"  saved checkpoint -> {CKPT}")

print("evaluating on held-out test storks...")
results = evaluate(model, test_frame, context_length=24,
                   individual_col="segment_id", show_progress=True,
                   extra_cols=("individual_id",))

print(f"\n{len(results):,} predictions")
print(stratified_summary(results).to_string(index=False))