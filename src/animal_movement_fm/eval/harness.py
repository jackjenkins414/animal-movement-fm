"""
Evaluation harness for next-step movement prediction. 

Design
-----
Every model in this project, from persistence to transformers, implements a 
single method: predict(history) -> next postion. The aim here is to have a
harness that does not know or care which model it is calling (a Strategy 
Design pattern: every model presents the same interface (the `Predictor` 
protocol) and the harness treats them as black boxes).
This decoupling means: 

- Adding a new model is one new file in `src/.../models/` and zero changes here. 
- All models are evaluated on identical inputs, so reported numbers are then 
  directly comparable. 
- The harness is testable in isolation using a trivial dummy predictor. 

Evaluation task
----- 
Single-step prediction: given the previous `context_length` positions, predict 
the next position. We default to single-step for now (as this is what the 
literature focuses on), but hope to pursue multi-step forecasting (5-step, 10-step) 
as a future extension. Note, the result schema already includes a `step_ahead` 
column to make that addition trivial. 
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dataclasses import dataclass, field
from typing import Any

import numpy as np 
import pandas as pd 
from tqdm import tqdm

from animal_movement_fm.eval.metrics import(
    euclidean_distance_degrees, 
    haversine_distance, 
    summarise_errors,
)

# ------------------------------
# Interface every model implements 
# ------------------------------

# The Strategy interface is the `Predictor` Protocol class
# And persistence, averaging, ARIMA, XGBoost, LSTM etc are each of 
# our "Concrete Strategy" from the Strategy design pattern 
@runtime_checkable
class Predictor(Protocol):
    """
    A movement-prediction model.

    Implementations only need `predict`. The optional `name` attribute is used in results tables; if absent, 
    the class name is used.

    `predict` takes a PredictionContext (history + optional side info) and returns a (2,) array predicting the 
    (lon, lat) of the next observation. Predictors are stateless across calls; any learnable state lives inside
    the implementing object and is established before evaluation (e.g. via a `fit` method, which is NOT part
    of this protocol because not all models have one, such as persistence).
    """

    def predict(self, context: "PredictionContext") -> np.ndarray: ...


@dataclass(frozen=True)
class PredictionContext:
    """Everything a model needs to make one prediction.

    Bundles the history window with any side information a model may consume, so new inputs (timestamps, 
    environmental covariates etc) are added as fields here rather than by changing every model's 
    predict signature.

    Fields
    ------
    history : np.ndarray
        Shape (T, 2), columns (lon, lat), T == context_length. The past positions; every model uses this.
    timestamp : pd.Timestamp | None
        Time of the target step (the one being predicted), for time of day features. None when unavailable
        (e.g. synthetic data without real times).
    history_timestamps : np.ndarray | None
        Shape (T,), the timestamp of each fix in `history`, row aligned with it (history_timestamps[i] is the
        time of history[i]). Populated by `evaluate` alongside `timestamp`; None when times are unavailable.
        This is what a per position time of day feature needs: the clock at every fix in the window, not just
        the target's. Kept separate from `history` so the position array stays purely (lon, lat).
    extra : dict[str, Any]
        Open slot for future side information (covariates, individual metadata) without another signature
        change. Empty for now.
    """
    history: np.ndarray
    timestamp: "pd.Timestamp | None" = None
    history_timestamps: "np.ndarray | None" = None
    extra: dict[str, Any] = field(default_factory=dict)



# ------------------------------
# Sliding window iteration 
# ------------------------------

def iter_windows(
        trajectory: pd.DataFrame, 
        context_length: int, 
) -> "list[tuple[np.ndarray, pd.Timestamp, int]]":
    """
    Generate sliding-window training examples for one trajectory. 

    For a trajectory of length T with context window W, this generates T - W
    windows, each of which is (W, 2) history paired with a (2,) target. 
    """

    # Convert our df to numpy arrays since repreated slicing is faster in numpy 
    coords = trajectory[["lon", "lat"]].to_numpy()
    timestamps = trajectory["timestamp"].to_numpy()
    T = len(trajectory)

    out = []

    # A trajectory shorter than the context window cannot produce a single valid
    # training example, so just return early. 
    if T <= context_length: 
        return out
    
    # Window at pos t uses observations [t - W, t] as history and observation t as 
    # the target. So t ranges from W (first valid target) to T-1 inclusive
    for t in range(context_length, T): 
        history = coords[t - context_length : t] # (W, 2)
        target = coords[t] # (2,)
        out.append((history, target, timestamps[t], t))
    return out 

# ------------------------------
# Main entry point 
# ------------------------------

# This is the "Context" of our Strategy design pattern, where we have written 
# `evaluate()` to need a prediction algorithm but never specified which one. 
# The context delegates to a Strategy object: pred = predictor.predict(history)
def evaluate(
        predictor: Predictor, 
        df: pd.DataFrame, 
        context_length: int = 24, 
        *, 
        individual_col: str = "individual_id", 
        show_progress: bool = True, 
        extra_cols: tuple[str, ...] = (), 
) -> pd.DataFrame: 
    """
    Evaluate a predictor on a movement DataFrame. 

    Parameters
    -----
    predictor: an object implementing the Predictor protocol. 
    df: long-format DataFrame with at least columns [individual_id, timestamp, 
       lon, lat] sorted by (individual_id, timestamp). Note, synthetic data won't 
       contain things like species, class, sex, etc, but real data will. 
    context_length: number of past observations supplied as input to each prediction. 
       Start with assumption that 24 is a reasonable estimate for hourly data, which 
       is 1 day of history. But tune per dataset. 
    individual_col: column name identifying trajectories. Each individual is processed
       independently; no information leaks across them. 
    show_progress: show a tqdm bar over individuals. 
    extra_cols: additional columns from `df` to carry through into the results. These 
       include things like species and class. Values taken from the target row of 
       each prediction. 
    
    Returns 
    -----
    DataFrame with one row per prediction, columns [individual_id, timestamp, target_lon
    target_lat, pred_lon, pred_lat, error_m, error_deg] + any extra cols. For summaries 
    we use `.groupyby(...).agg(...)` on the returned DataFrame to produce stratified 
    summaries. 
    """

    # A bug that was caught; if the caller hands an empty DataFrame, we still 
    # return a DataFrame with correct schema so downstream code does not fail
    if df.empty: 
        return pd.DataFrame(
            columns=[
                individual_col, 
                "timestamp", 
                "target_lon", 
                "target_lat", 
                "pred_lon", 
                "pred_lat",
                "error_m",
                "error_deg",
                *extra_cols, # col names to col list, so we dont literally add this col name
            ]
        )
    
    # Assume DataFrame is sorted, but sort here as safety net
    # Sort in place to avoid silently doubling memory 
    df = df.sort_values([individual_col, "timestamp"]).reset_index(drop=True)

    # One dict per prediction 
    records: list[dict] = []

    individuals = df.groupby(individual_col, sort=False)
    iterator = tqdm(individuals, disable=not show_progress, desc="Evaluating") if show_progress else individuals

    for individual_id, traj in iterator: 
        windows = iter_windows(traj, context_length=context_length)

        # Trajectory shorter than context length 
        if not windows: 
            continue

        # pull the per target rows from the original trajectory once
        # so we can carry through extra_cols (cheaper this way)
        target_indices = [w[3] for w in windows]
        target_rows = traj.iloc[target_indices]

        # whole trajectory timestamp array, sliced per window below. iter_windows already gives us each window's target index, and 
        # history is coords[idx-W:idx], so the row aligned per fix times are the same slice of this array. Lets models that use a 
        # per-position time of day feature see every fix's clock, not just the target's, at zero extra windowing cost.
        traj_timestamps = traj["timestamp"].to_numpy()

        # The prediction loop. Each model handles its own batching internally if it wants to; the protocol contract is one prediction 
        # at a time. This keeps the protocol simple at the cost of some inefficiency for batched models. Note, this is what we can 
        # optimise in future if it becomes a bottleneck
        for (history, target, ts, _idx), (_, target_row) in zip(windows, target_rows.iterrows()):
            context = PredictionContext(
                history=history,
                timestamp=pd.Timestamp(ts),
                history_timestamps=traj_timestamps[_idx - context_length : _idx],
            )
            pred = predictor.predict(context)
            err_m = haversine_distance(pred[0], pred[1], target[0], target[1])
            err_deg = euclidean_distance_degrees(pred[0], pred[1], target[0], target[1])

            # Build a record dict
            record = {
                individual_col: individual_id, 
                "timestamp": ts, 
                "target_lon": target[0], 
                "target_lat": target[1],
                "pred_lon": float(pred[0]), 
                "pred_lat": float(pred[1]), 
                "error_m": float(err_m), 
                "error_deg": float(err_deg), 
            }
            for col in extra_cols:
                record[col] = target_row[col]
            records.append(record)

    return pd.DataFrame(records)

# GenAI reference: Helper function drafted with GenAI to keep `evaluate`focused on 
# per-prediction evaluation only. Since `evaluate` returns one row per prediction, 
# while this helper converts those raw errors into summary tables overall or stratified 
# by columns such as species, class, individual_id, etc. 
def stratified_summary(
    results: pd.DataFrame,
    by: str | list[str] | None = None,
) -> pd.DataFrame:
    """
    Produce a summary table from the per-prediction results of `evaluate`.

    If `by` is None, returns a single row table with overall stats.
    If `by` is a column name (or list), returns one row per group.

    Use the returned DataFrame as the basis for thesis tables.
    """
    # ---------- Mode 1: overall summary ----------
    # No grouping requested. Compute one row of stats over all predictions.
    if by is None:
        summary = summarise_errors(results["error_m"].to_numpy())
        return pd.DataFrame([summary])

    # ---------- Mode 2: grouped summary ----------
    # Normalise `by` to a list of column names. This lets us treat the
    # single-column and multi-column cases uniformly below.
    rows = []
    group_cols = [by] if isinstance(by, str) else list(by)

    # Iterate each group, compute stats, and stash them along with the
    # grouping key(s).
    for keys, group in results.groupby(group_cols):
        summary = summarise_errors(group["error_m"].to_numpy())

        # When grouping by multiple columns, pandas gives us a tuple of
        # keys; for a single column, just the value. Handle both.
        if isinstance(keys, tuple):
            for col, val in zip(group_cols, keys):
                summary[col] = val
        else:
            summary[group_cols[0]] = keys

        rows.append(summary)

    # Reorder columns so the grouping keys come first (more natural to
    # read in printed tables).
    summary_df = pd.DataFrame(rows)
    return summary_df[
        group_cols + [c for c in summary_df.columns if c not in group_cols]
    ]