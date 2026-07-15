# GenAI reference: Again, here I used GenAI to add clearer terminal output messages and minor code/comment cleanup.
"""Train XGBoost on the real stork train split, evaluate on held-out test storks."""

# MUST be set before numpy/torch/xgboost import — resolves the macOS OpenMP
# segfault from torch and XGBoost shipping separate OpenMP runtimes.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from pathlib import Path

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.xgboost import XGBoost

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"

counts = window_counts_per_individual(SEGMENTS)
split = make_split(counts, strata=None, min_windows=100, seed=42)

# Load both splits up front. Train is large (~630 birds); test matches the baselines.
print("loading train and test frames...")
train_frame = load_split_frame(SEGMENTS, split.train)
test_frame = load_split_frame(SEGMENTS, split.test)
print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
print(f"  test : {len(test_frame):,} rows / {test_frame['segment_id'].nunique()} segments")

# Train on segments (so feature windows never cross a transmission gap), keyed on segment_id.
model = XGBoost(n_lags=5, use_time_features=False)
print(f"\ntraining {model.name} on train split...")
t0 = time.time()
model.fit(train_frame, individual_col="segment_id")
print(f"  fit done in {time.time() - t0:.0f}s")

# Evaluate on held-out test, identical harness path as the baselines.
print("evaluating on held-out test storks...")
results = evaluate(
    model,
    test_frame,
    context_length=24,
    individual_col="segment_id",
    show_progress=True,
    extra_cols=("individual_id",),
)

print(f"\n{len(results):,} predictions")
print(stratified_summary(results).to_string(index=False))