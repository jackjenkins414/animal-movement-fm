"""
Run a predictor on real resampled stork data, respecting the train/val/test split.

Bridges stork_segments.parquet + a Split (data.split) into the existing evaluation harness (eval.harness). 
Windows are built per segment_id, so no window ever crosses a transmission gap; the real individual_id is 
carried through for per-bird summaries.

No model or harness changes: the segments store absolute (lon, lat) positions, which is exactly what the 
ladder's predictors consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb

from animal_movement_fm.eval.harness import Predictor, evaluate, stratified_summary


def load_split_frame(
    segments_parquet,
    individual_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Load segment rows for a given set of individuals, shaped for the harness.

    Returns a long frame with columns [segment_id, individual_id, bin_index, timestamp, lon, lat], filtered to `individual_ids` 
    and sorted so the harness can derive one bin contiguous runs within each segment.
    """

    con = duckdb.connect()
    # register the id list as a table so DuckDB filters efficiently.
    id_df = pd.DataFrame({"individual_id": individual_ids.astype("int64")})
    con.register("split_ids", id_df)

    frame = con.execute(f"""
        SELECT s.segment_id, s.individual_id, s.bin_index, s.timestamp, s.lon, s.lat
        FROM '{segments_parquet}' s
        SEMI JOIN split_ids ids ON s.individual_id = ids.individual_id
        ORDER BY s.segment_id, s.timestamp
    """).df()
    return frame


def evaluate_on_split(
    predictor: Predictor,
    segments_parquet,
    individual_ids: np.ndarray,
    context_length: int = 24,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate one predictor on one split's individuals.

    Windows are grouped by segment_id (gap-respecting); individual_id is carried as an extra column for per-bird stratified 
    summaries. Returns the per-prediction results frame from the harness.
    """
    frame = load_split_frame(segments_parquet, individual_ids)
    return evaluate(
        predictor,
        frame,
        context_length=context_length,
        individual_col="segment_id",  # window within segments, never across gaps
        show_progress=show_progress,
        extra_cols=("individual_id",), # keep the real bird id for summaries
    )

def run_ladder(
    predictors: list[Predictor],
    segments_parquet,
    individual_ids,
    context_length: int = 24,
) -> pd.DataFrame:
    """
    Evaluate each predictor on the same test individuals; return one summary per model.

    The test frame is loaded once and reused, so every model sees identical windows. Returns a table with one row per predictor, 
    sorted by median error.
    """
    frame = load_split_frame(segments_parquet, individual_ids)
    print(f"loaded {len(frame):,} rows across {frame['segment_id'].nunique()} segments\n")

    rows = []
    for predictor in predictors:
        name = getattr(predictor, "name", type(predictor).__name__)
        results = evaluate(
            predictor,
            frame,
            context_length=context_length,
            individual_col="segment_id",
            show_progress=False,
            extra_cols=("individual_id",),
        )
        summary = stratified_summary(results).iloc[0].to_dict()
        summary = {"model": name, **summary}
        rows.append(summary)
        print(f"  {name:<18} median={summary['median_m']:>8.1f}m  "
              f"mean={summary['mean_m']:>9.1f}m  p99={summary['p99_m']:>11.1f}m")

    table = pd.DataFrame(rows).sort_values("median_m").reset_index(drop=True)
    return table


if __name__ == "__main__":
    from pathlib import Path

    from animal_movement_fm.data.split import (
        make_split,
        window_counts_per_individual,
    )
    from animal_movement_fm.eval.harness import stratified_summary
    from animal_movement_fm.models.persistence import Persistence

    SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"

    counts = window_counts_per_individual(SEGMENTS)
    split = make_split(counts, strata=None, min_windows=100, seed=42)

    print(f"evaluating persistence on {len(split.test)} held-out test storks...")
    results = evaluate_on_split(Persistence(), SEGMENTS, split.test)

    print(f"\n{len(results):,} predictions")
    print(stratified_summary(results).to_string(index=False))