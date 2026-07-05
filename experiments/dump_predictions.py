# GenAI reference: I used GenAI to help with this script; specifically I wrote and verified the experiment 
# logic, including the same held out split, segment_id grouping, and evaluate() path as the headline runs. 
# GenAI helped with the parquet export structure, checkpoint reloading loop, vectorised prev position/step
# length calc and added the clean result printing. I verified the output by checking that prediction counts 
# and error summaries matched the headline benchmark outputs, saved parquet files had the expected columns
# and that true_step_km values were spot checked against haversine distances between consecutive targets 
# within a segment.
"""
Dump per-window predictions + errors to parquet for the diagnostic phase.

Our benchmark harness already returns one row per prediction; run_ladder just collapses it with stratified_summary 
and discards the rows. The diagnostic analyses (error distribution, p90 vs p99, regime split, per track ranking) 
need those rows. So we added this script to re-run evaluate on the held out test storks and write the full 
per-prediction frame, adding the following:

  prev_true_lon/lat : previous true position within the segment
  true_step_km      : haversine(prev_true, target); the step the bird actually made, used to tag each prediction's 
                      regime (foraging/moving)
  error_km          : error_m / 1000        going to km here

Identical windowing to the headline runs: same evaluate(), same segment_id grouping, same split. No harness changes.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.models.lstm import LSTM
from animal_movement_fm.models.transformer import Transformer
from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.metrics import haversine_distance
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.averaging import DisplacementAveraging

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"
CHECKPOINTS = REPO / "checkpoints"
LSTM_CKPT = CHECKPOINTS / "lstm_stork_h64_l1_e60.pt"
TRANSFORMER_CKPT = CHECKPOINTS / "transformer_stork_d64_l2_e60.pt"
CONTEXT_LENGTH = 24


def add_regime_fields(results: pd.DataFrame) -> pd.DataFrame:
    """
    Attach prev true position and true step length per prediction.

    Predictions are ordered within each segment (evaluate walks each segment_id in time order), so shifting 
    the true target position by one row inside a segment gives the previous true position; which is exactly 
    history[-1] for that window. The first prediction of each segment has no predecessor and gets NaN step 
    (dropped only where regime matters, never from error stats).
    """
    results = results.copy()
    results["prev_true_lon"] = results.groupby("segment_id")["target_lon"].shift(1)
    results["prev_true_lat"] = results.groupby("segment_id")["target_lat"].shift(1)

    has_prev = results["prev_true_lon"].notna()
    step_km = np.full(len(results), np.nan)
    step_km[has_prev.to_numpy()] = haversine_distance(
        results.loc[has_prev, "prev_true_lon"].to_numpy(),
        results.loc[has_prev, "prev_true_lat"].to_numpy(),
        results.loc[has_prev, "target_lon"].to_numpy(),
        results.loc[has_prev, "target_lat"].to_numpy(),
    ) / 1000.0
    results["true_step_km"] = step_km
    results["error_km"] = results["error_m"] / 1000.0
    return results


def dump_predictor(predictor, frame: pd.DataFrame, name: str) -> None:
    """Run evaluate on one predictor and write the enriched per prediction parquet."""
    results = evaluate(
        predictor,
        frame,
        context_length=CONTEXT_LENGTH,
        individual_col="segment_id", # gap safe windowing, matches headline runs
        show_progress=True,
        extra_cols=("individual_id",), # real bird id for per-track ranking
    )
    results = add_regime_fields(results)
    results.insert(0, "model", name)

    out = RESULTS / f"preds_{name}.parquet"
    results.to_parquet(out, index=False)
    med = results["error_km"].median()
    p99 = results["error_km"].quantile(0.99)
    print(f"  {name:<14} {len(results):>8,} preds  "
          f"median={med:.3f} km  p99={p99:.3f} km  -> {out.name}")

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    counts = window_counts_per_individual(SEGMENTS)
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    frame = load_split_frame(SEGMENTS, split.test)
    print(f"loaded {len(frame):,} rows across {frame['segment_id'].nunique()} "
          f"test segments\n")

    # baselines that are traininng free
    baselines = [
        (Persistence(), "persistence"),
        (DisplacementAveraging(k=1), "avg_k1"),
        (DisplacementAveraging(k=3), "avg_k3"),
    ]
    for predictor, name in baselines:
        dump_predictor(predictor, frame, name)

    # trained neural models, reloaded from checkpoint (no retrain) from_checkpoint restores weights + train-stat 
    # normalisation, so predict() produces predictions in the same space as the headline runs
    neural = [
        (LSTM_CKPT, LSTM, "lstm"),
        (TRANSFORMER_CKPT, Transformer, "transformer"),
    ]
    for ckpt_path, cls, name in neural:
        if not ckpt_path.exists():
            print(f"  {name:<14} checkpoint missing ({ckpt_path.name}); skipped")
            continue
        model = cls.from_checkpoint(ckpt_path, device="cpu")
        dump_predictor(model, frame, name)

    print("\nall available prediction dumps written to experiments/results/")


if __name__ == "__main__":
    main()