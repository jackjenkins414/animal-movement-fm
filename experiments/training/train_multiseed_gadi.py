# GenAI reference: the majority of this script I wrote; GenAI cleaned up the main method and some minor comments
"""
Multi-seed covariate sweep on Gadi: one PBS array job runs all 30 cells.

Why
---
Every neural result so far (the raw head to head, the time of day 2x2) rests on a SINGLE training seed, so the
small effects (~6% tail moves, the LSTM's minor negatives) sit inside plausible seed noise and cannot be leaned
on. This script runs the full grid in one submission:

    3 feature cells  x  2 architectures  x  5 training seeds  =  30 runs

  cells : raw          no covariates, the plain displacement sequence
          time         + local solar time of day (per position)
          time_season  + day of year as well (per position)
  archs : lstm, transformer
  seeds : 0..4, applied to torch (weight init + batch shuffling)

The DATA split seed stays fixed at 42 throughout. That is the point: every run sees identical windows, so the
only thing varying within a cell is the training seed.

Index decoding
--------------
PBS sets $PBS_ARRAY_INDEX in 0..29, which maps to a unique (cell, arch, seed):

    cell = idx // 10        arch = (idx % 10) // 5        seed = idx % 5

Concurrency
-----------
All 30 subjobs run at once, so nothing here appends to a shared file; that would race. Each run writes its own
checkpoint, its own predictions parquet and its own summary JSON, keyed by run id. Combining them is a separate
analysis step (see the aggregation script), which also keeps inference and analysis decoupled. 

We run inference over BOTH the val and test splits and tag each row with which it came from, so model selection
can use val without ever touching test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import torch

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import (
    apply_window_eligibility,
    stork_split,
    window_counts_per_individual,
)
from animal_movement_fm.eval.harness import evaluate, stratified_summary
from animal_movement_fm.models.lstm import LSTM
from animal_movement_fm.models.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
CKPT_DIR = REPO / "checkpoints" / "multiseed"
OUT_DIR = REPO / "results" / "multiseed"

SPLIT_SEED = 42   # data split; FIXED across all runs
CONTEXT_LENGTH = 24
SEEDS = [0, 1, 2, 3, 4]  # training seeds

# feature cells: (name, use_time_features, use_season_features)
CELLS = [
    ("raw", False, False),
    ("time", True, False),
    ("time_season", True, True),
]

# context length ablation arm (indices 30-39); time_season x 2 archs x 5 seeds at C=12, riding along 
# with the C=24 grid rather than costing a second submission. C=24 targets are a strict subset of C=12, 
# so compare only the paired numbers - see context_ablation.py.
ABLATION_CONTEXT = 12
ABLATION_CELL = ("time_season", True, True)

# established per-architecture recipes, unchanged from the single-seed runs so these results stay
# comparable to them; transformer keeps lr=3e-4. context_length is injected per run (see
# decode_index) rather than baked in, because the ablation arm varies it.
BASE_CONFIG = {
    "lstm": dict(
        hidden_dim=64, num_layers=1,
        epochs=80, batch_size=256, lr=1e-3,
    ),
    "transformer": dict(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1,
        epochs=80, batch_size=256,
        lr=3e-4, warmup_frac=0.1, grad_clip=1.0,
    ),
}
ARCHS = ["lstm", "transformer"]
MODEL_CLS = {"lstm": LSTM, "transformer": Transformer}

N_BASE = len(CELLS) * len(ARCHS) * len(SEEDS)          # 30: indices 0-29, context 24
N_ABLATION = len(ARCHS) * len(SEEDS)                   # 10: indices 30-39, context 12
N_RUNS = N_BASE + N_ABLATION                           # 40


def decode_index(idx: int) -> tuple[tuple[str, bool, bool], str, int, int]:
    """
    Map a flat array index in [0, 40) to its (cell, arch, seed, context_length).

    0-29 is the covariate grid at C=24, decoding unchanged from the pre-ablation script so previously submitted indices 
    still mean the same thing. 30-39 is time_season at C=12.
    """
    if not 0 <= idx < N_RUNS:
        raise SystemExit(f"FAIL: array index {idx} outside 0..{N_RUNS - 1}")

    if idx < N_BASE:
        cell = CELLS[idx // 10]
        arch = ARCHS[(idx % 10) // 5]
        seed = SEEDS[idx % 5]
        return cell, arch, seed, CONTEXT_LENGTH

    j = idx - N_BASE
    arch = ARCHS[j // len(SEEDS)]
    seed = SEEDS[j % len(SEEDS)]
    return ABLATION_CELL, arch, seed, ABLATION_CONTEXT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=int, default=None,
        help="run index 0..29; defaults to $PBS_ARRAY_INDEX (override for local testing)",
    )
    args = parser.parse_args()

    idx = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if idx is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX in the environment")
    idx = int(idx)

    (cell_name, use_time, use_season), arch, seed, context_length = decode_index(idx)

    # run ids stay byte identical for the C=24 arm so existing checkpoints, prediction parquets and the aggregation script 
    # are untouched; only the ablation arm carries the context suffix.
    run_id = f"{arch}_{cell_name}_s{seed}"
    if context_length != CONTEXT_LENGTH:
        run_id = f"{run_id}_c{context_length}"

    config = dict(BASE_CONFIG[arch])
    config["context_length"] = context_length
    config["use_time_features"] = use_time
    config["use_season_features"] = use_season

    arm = "context ablation" if context_length != CONTEXT_LENGTH else "registered context"
    print("=" * 70)
    print(f"multi-seed sweep run {idx}/{N_RUNS - 1}: {run_id}")
    print(f"device available : cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu              : {torch.cuda.get_device_name(0)}")
    print(f"cell             : {cell_name} (time={use_time}, season={use_season})")
    print(f"architecture     : {arch}")
    print(f"context_length   : {context_length}  [{arm}]")
    print(f"training seed    : {seed}   (split seed fixed at {SPLIT_SEED})")
    for k, v in config.items():
        print(f"    {k:<20}= {v}")
    print("=" * 70)

    if not SEGMENTS.exists():
        raise SystemExit(f"FAIL: data not found at {SEGMENTS}")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Seed torch before the model exists: this covers weight init AND the randperm batch shuffling in fit().
    torch.manual_seed(seed)

    # read the frozen manifest; never recompute the split here (make_split is order dependent and gave different splits before (painful));
    # counts must use THIS run's context - eligibility is context-dependent, so C=24 counts would drop individuals that are usable at C=12
    counts = window_counts_per_individual(SEGMENTS, context_length=context_length)
    split = stork_split()

    # drop only individuals that yield no window under A2 contiguity. No eval floor here; it is a per individual filter, so applying it 
    # downstream in aggregate_multiseed.py is metric identical and lets our A5 10/25/50/100 sensitivity come from one set of runs
    split = apply_window_eligibility(
        split, counts, train_min_windows=1, eval_min_windows=1
    )

    n_individuals = {
        "train": len(split.train), "val": len(split.val), "test": len(split.test),
    }
    print(f"frozen manifest after A2 eligibility: "
          f"train {n_individuals['train']} / val {n_individuals['val']} / test {n_individuals['test']}")

    print("loading frames...")
    train_frame = load_split_frame(SEGMENTS, split.train)
    frames = {
        "val": load_split_frame(SEGMENTS, split.val),
        "test": load_split_frame(SEGMENTS, split.test),
    }
    print(f"  train: {len(train_frame):,} rows / {train_frame['segment_id'].nunique()} segments")
    for name, f in frames.items():
        print(f"  {name:<5}: {len(f):,} rows / {f['segment_id'].nunique()} segments")

    model = MODEL_CLS[arch](**config, name=run_id)

    print(f"\ntraining for {config['epochs']} epochs...")
    t0 = time.time()
    model.fit(train_frame, group_col="segment_id")
    fit_min = (time.time() - t0) / 60
    print(f"  fit done in {fit_min:.1f} min on device={model.device}")

    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    model.save(ckpt_path)
    print(f"  saved checkpoint -> {ckpt_path}")

    # Inference over val AND test; tag rows so the analysis phase can slice by split.
    print("\nrunning inference...")
    parts = []
    summary = {
        "run_id": run_id, "index": idx, "arch": arch, "cell": cell_name,
        "seed": seed, "split_seed": SPLIT_SEED,
        "context_length": context_length,
        "arm": "ablation_c12" if context_length != CONTEXT_LENGTH else "registered_c24",
        "use_time_features": use_time, "use_season_features": use_season,
        "fit_minutes": round(fit_min, 2), "config": config,
        # Provenance for the corrected re-run. `eval_floor` is null by design: the floor is applied
        # downstream, so these summary figures are UNFILTERED and are not comparable to the
        # pre-correction tables, which were computed at floor 100. Read the floored numbers from
        # aggregate_multiseed.py, not from here.
        "split_source": "frozen_manifest:splits/stork_split_seed42.json",
        "continuity_corrected": True,
        "train_min_windows": 1,
        "eval_floor": None,
        "n_individuals": n_individuals,
    }
    for split_name, frame in frames.items():
        results = evaluate(
            model, frame,
            context_length=context_length,
            individual_col="segment_id",
            show_progress=False,
            extra_cols=("individual_id",),
        )
        results.insert(0, "split", split_name)
        results.insert(0, "run_id", run_id)
        results.insert(0, "context_length", context_length)
        parts.append(results)

        stats = stratified_summary(results).iloc[0].to_dict()
        summary[split_name] = {k: float(v) for k, v in stats.items()}
        print(f"  {split_name:<5}: {len(results):,} preds  "
              f"median={stats['median_m'] / 1000:.3f} km  p99={stats['p99_m'] / 1000:.3f} km")

    preds_path = OUT_DIR / f"preds_{run_id}.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(preds_path, index=False)

    summary_path = OUT_DIR / f"summary_{run_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\npredictions -> {preds_path}")
    print(f"summary     -> {summary_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
