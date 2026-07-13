# GenAI reference: I used GenAI to help with the main method print statemets (it minorly changed the structure too). 
"""
LSTM + time of day training on Gadi (real stork data)

The aim here is to propagate the local solar time of day feature to the neural rungs, as its own isolated increment, now that
the raw sequence head to head is done. This trains the time augmented LSTM so it can be compared against the raw sequence
LSTM it mirrors.

Clean ablation recipe
---------------------

The raw LSTM we carry forward is the convergence matched LSTM (80 epochs, batch 256, constant lr=1e-3, no warmup/cosine and no
clipping). So this run uses the matched recipe unchanged and turns time ON as the single difference, so LSTM time vs LSTM raw is
a one variable ablation:

From features/time.py: each displacement is now tagged with the local solar (sin, cos) of the fix it lands on (destination tagging),
so the 4 channel input lets the recurrence condition on time of day at each step. Only the two displacement channels are standardised;
the sin/cos are already bounded. On the 07-09 XGBoost note the diurnal lift lands on the median/body and leaves the p99 tail
(launch onset) essentially unchanged, so we will see how the stationary/moving median numbers (not the moving/launch tail) look here.
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

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
CKPT = REPO / "checkpoints" / "lstm_stork_time.pt"

# matched-LSTM recipe 
CONFIG = dict(
    hidden_dim=64,
    num_layers=1,
    context_length=24,
    epochs=80,          
    batch_size=256,     
    lr=1e-3,            
    use_time_features=True, # per pos local solar time 
)

def main() -> None:
    print("=" * 70)
    print("LSTM + time-of-day run (Design B; time 2x2)")
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
