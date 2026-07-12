# GenAI reference: I used GenAI to help draft the training entry point for NCI Gadi and it mirrors my previous
# established run_transformer_real.py pipeline (same split, same segment_id grouping), but is adapted for GPU 
# execution and the first convergence fix iteration. Specifically I wrote the docs, congfigs and a rough main
# method which was polished and check by GenAI. 
"""
Transformer training on Gadi (real stork data, full scale); convergence iteration 2.

Purpose
-------
Iteration 1 converged smoothly (train_mse 0.0482 -> 0.0214) but plateaued above the LSTM's 0.0179. The iter-1 recipe was
doubly conservative: it lowered the peak LR (1e-3 -> 3e-4) and raised the batch (64 -> 256) at once, then cosine-decayed
the LR to ~0. By the linear scaling rule a 4x larger batch wants a bigger peak LR, not a smaller one, so the optimiser was
under driven and stalled before reaching the LSTM.

Iteration 2 keeps everything from iter-1 except the peak LR:

  - raise peak LR      (1e-3, up from 3e-4)   OUR ONLY CHANGE 
  - warmup + cosine    (warmup_frac=0.1; unchanged)
  - gradient clipping  (max global norm 1.0; unchanged)
  - batch_size 256, epochs 80 (unchanged)

Warmup + grad clipping are what make peak 1e-3 safe now; the original naive 1e-3 run (neither) diverged. Architecture is held
identical to the previous runs (d_model=64, nhead=4, num_layers=2) so any change in behaviour is attributable to the LR, not
the model.

Also new this run: Transformer.fit() now preloads the training tensors onto the GPU once and indexes batches on-device, rather
than streaming through a DataLoader with per-batch CPU->GPU copies. Iter-1 ran 72 min at ~15% GPU util (V100 starved); the
whole train set is ~170 MB and fits easily in 32 GB, so this should cut the run to minutes without changing the maths.

Note our the target is to reach <= the LSTM's 0.0179 and keep descending smoothly. 

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
CKPT = REPO / "checkpoints" / "transformer_stork_convergence_iter2.pt"

# convergence fix hyperparameters (iteration 2)
CONFIG = dict(
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    context_length=24,
    epochs=80,
    batch_size=256, # larger than the old 64: fewer, less-noisy updates -> steadier descent, better GPU use
    lr=1e-3,   # raised from iter-1's 3e-4: linear-scaling rule for the 4x batch; warmup + grad_clip keep it safe
    warmup_frac=0.1, # linear warmup then cosine decay
    grad_clip=1.0, # cap global grad norm
)


def main() -> None:
    print("=" * 70)
    print("Transformer convergence run (iteration 2)")
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
