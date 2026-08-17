# GenAI reference: Again I did the basics (imports/constants etc), max_gap_bins_for(), metric/skill helpers, and most of 
# build_segments_for_ids(), while GenAI helped complete parts of the segment-building and wrote the main method to check 
# the sweep logic, reuse the canonical split/cache files consistently, and produce clear progress/summary output. The 
# script was verified by confirming the 6h persistence denominator re-matched the canonical result, checking train/test 
# segment caches matched the expected cadence labels, and ensured time features stayed disabled. Also checked the saved 
# CSV against the printed XGBoost skill table.
"""
Cadence sweep: XGBoost skill relative to same-cadence persistence.

The first pass that trains. XGBoost fits once per cadence on the split.train birds resampled at that cadence, then predicts 
on the same 105 split.test birds every other pass used. Reports the same relative skill metric as the averaging pass, so 
this table stacks directly under it:

    skill = (persistence_error - model_error) / persistence_error   (per cadence, per pct)

What this pass tests: the averaging pass showed the migration TAIL becomes more tractable at fine cadence (p99 skill rose toward 
30min) while the typical step got worse (CV injects motion into a growing stationary fraction). Averaging is a weak model, so 
that tail signal could be a quirk of flat velocity extrapolation. XGBoost is the first model that can actually exploit structure;
nonlinearity, regime separation in feature space. If its p99 skill ALSO rises toward fine cadence, the tail tractability finding 
is real; if only averaging showed it, it was an artefact.

Scope: 6h / 3h / 1h / 30 min, run locally. 24h/12h are dropped (coarse train birds starve on the 120-bin floor and aren't really headline 
comparable anyway).
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"  

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from animal_movement_fm.data.resample import resample_trajectory
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.data.benchmark import load_split_frame, evaluate_on_split
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.xgboost import XGBoost


REPO = Path(__file__).resolve().parents[2]
STORK_FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
SEGMENTS_6H = REPO / "data" / "processed" / "stork_segments.parquet"  # for the canonical split
RESULTS = REPO / "results"

# label -> bin seconds
CADENCE_GRID = {"6h": 21600, "3h": 10800, "1h": 3600, "30min": 1800}

SPLIT_DAYS = 2  # >2d gap 
MIN_SEGMENT_BINS = 120 # held in bins 
CONTEXT_LENGTH = 24 # held in bins


def max_gap_bins_for(bin_seconds: int) -> int:
    """Convert the >2d split rule to a bin count at this cadence (6h -> 8)."""
    return round(SPLIT_DAYS * 86400 / bin_seconds)

def build_segments_for_ids(
    con: duckdb.DuckDBPyConnection,
    fixes_parquet: Path,
    individual_ids: np.ndarray,
    bin_seconds: int,
    max_gap_bins: int,
) -> pd.DataFrame:
    """Resample the given individuals at `bin_seconds` into tidy segment rows.

    One (individual_id, deployment_id) trajectory read at a time from the sorted staging parquet so memory stays bounded 
    to one trajectory's raw fixes. Mirrors build_segments.py pass two, cadence-parameterised. Same routine the test bird 
    sweep used, applied to train.
    """
    id_df = pd.DataFrame({"individual_id": np.asarray(individual_ids, dtype="int64")})
    con.register("keep_ids", id_df)
    traj_keys = con.execute(f"""
        SELECT DISTINCT f.individual_id, f.deployment_id
        FROM '{fixes_parquet}' f
        SEMI JOIN keep_ids k ON f.individual_id = k.individual_id
        ORDER BY f.individual_id, f.deployment_id
    """).fetchall()

    frames = []
    for individual_id, deployment_id in traj_keys:
        raw = con.execute(f"""
            SELECT timestamp, location_lat, location_long
            FROM '{fixes_parquet}'
            WHERE individual_id = {individual_id}
              AND deployment_id = {deployment_id}
            ORDER BY timestamp
        """).df()

        segments = resample_trajectory(
            raw["timestamp"].to_numpy(),
            raw["location_lat"].to_numpy(),
            raw["location_long"].to_numpy(),
            bin_seconds=bin_seconds,
            max_gap_bins=max_gap_bins,
            min_segment_bins=MIN_SEGMENT_BINS,
        )
        for k, seg in enumerate(segments):
            frames.append(pd.DataFrame({
                "segment_id": f"{individual_id}_{deployment_id}_{k}",
                "individual_id": individual_id,
                "deployment_id": deployment_id,
                "bin_index": seg.bin_index,
                "timestamp": seg.timestamp,
                "lat": seg.lat,
                "lon": seg.lon,
                "centre_offset_s": seg.centre_offset_s,
            }))

    cols = ["segment_id", "individual_id", "deployment_id", "bin_index",
            "timestamp", "lat", "lon", "centre_offset_s"]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


def err_km(results: pd.DataFrame) -> np.ndarray:
    return results["error_m"].to_numpy() / 1000.0


def ladder(err: np.ndarray) -> dict:
    return {
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "p99_km": float(np.percentile(err, 99)),
    }


def skill(base: dict, model: dict) -> dict:
    return {
        f"skill_{p}": (base[f"{p}_km"] - model[f"{p}_km"]) / base[f"{p}_km"]
        for p in ("median", "p90", "p99")
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # --- Canonical split, identical to every other pass ---
    counts = window_counts_per_individual(SEGMENTS_6H, context_length=24)  # Tier 1, pre A6
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    print(f"canonical split: {len(split.train)} train birds, "
          f"{len(split.test)} test birds (reused across cadences)\n")

    rows = []
    for label, bin_seconds in CADENCE_GRID.items():
        mgb = max_gap_bins_for(bin_seconds)
        print(f"=== {label} (max_gap_bins={mgb}) ===")

        # --- train segments (cache the expensive resample) ---
        train_seg_path = RESULTS / f"segments_sweep_train_{label}.parquet"
        if train_seg_path.exists():
            print(f"  train segments cached ({train_seg_path.name})")
        else:
            print("  resampling train birds from staging parquet...")
            train_seg = build_segments_for_ids(
                con, STORK_FIXES, split.train, bin_seconds, mgb)
            train_seg.to_parquet(train_seg_path, index=False)
            print(f"  {train_seg['segment_id'].nunique()} train segments, "
                  f"{len(train_seg):,} bins -> {train_seg_path.name}")

        # --- test segments (built by cadence_sweep.py pass 1) ---
        test_seg_path = RESULTS / f"segments_sweep_{label}.parquet"
        if not test_seg_path.exists():
            print(f"  MISSING test segments ({test_seg_path.name}); "
                  f"rerun cadence_sweep.py -- skipping {label}\n")
            continue

        # --- fit XGBoost on train, time features OFF, windowed by segment_id ---
        train_frame = load_split_frame(train_seg_path, split.train)
        print(f"  fitting XGBoost on {len(train_frame):,} train rows...")
        model = XGBoost(use_time_features=False).fit(
            train_frame, individual_col="segment_id")

        # --- persistence denominator + XGBoost, same 105 test birds ---
        base = ladder(err_km(evaluate_on_split(
            Persistence(), test_seg_path, split.test,
            context_length=CONTEXT_LENGTH, show_progress=False)))

        print(f"  evaluating XGBoost on {len(split.test)} test birds "
              f"(per-window harness loop, this is the slow part)...")
        xgb = ladder(err_km(evaluate_on_split(
            model, test_seg_path, split.test,
            context_length=CONTEXT_LENGTH, show_progress=False)))

        guard = (f"  <- 6h persistence re-match (want ~0.599): {base['median_km']:.3f}"
                 if label == "6h" else "")
        s = skill(base, xgb)
        print(f"  persistence median={base['median_km']:.3f} km  "
              f"p99={base['p99_km']:.1f} km{guard}")
        print(f"  xgboost     median={xgb['median_km']:.3f} km  "
              f"p99={xgb['p99_km']:.1f} km")
        print(f"  SKILL       median={s['skill_median']:+.1%}  "
              f"p90={s['skill_p90']:+.1%}  p99={s['skill_p99']:+.1%}\n")

        rows.append({
            "cadence": label, "model": "xgboost_lag5",
            "persist_median_km": base["median_km"], "persist_p99_km": base["p99_km"],
            "model_median_km": xgb["median_km"], "model_p99_km": xgb["p99_km"],
            **s,
        })

    table = pd.DataFrame(rows)
    out_path = RESULTS / "cadence_sweep_xgboost.csv"
    table.to_csv(out_path, index=False)

    print("XGBoost skill over persistence, fine -> coarse:")
    order = {"1h": 0, "3h": 1, "6h": 2}
    for _, r in table.assign(_o=table["cadence"].map(order)).sort_values("_o").iterrows():
        print(f"  {r['cadence']:>3}  median {r['skill_median']:+.1%}   "
              f"p90 {r['skill_p90']:+.1%}   p99 {r['skill_p99']:+.1%}")
    print(f"\nsummary -> {out_path}")
    print("\nRead the p99 column fine->coarse: rising p99-skill toward 1h confirms the "
          "averaging tail-tractability trend with a model that can exploit it.")


if __name__ == "__main__":
    main()