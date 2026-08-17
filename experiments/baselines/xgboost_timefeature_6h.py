# GenAI reference: I implemented the experiment framing, imports/constants, segment rebuild helpers, cache handling, 
# and metric ladder and got GenAI to write the main() orchestration to sanity check the file level logic, keep the 
# A/B comparison clean, and produce readable terminal summaries/exports. I verified it by confirming  the time-off 
# run reproduced the cached 6h XGBoost sweep result, checking that sin_hour appeared only in the time on feature 
# layout, and comparing the saved CSV against the printed median/p90/p99 values.
"""
Time of day feature A/B on 6h XGBoost. 

Isolated increment: fit XGBoost twice on the SAME 6h split (once with time features off, once on) and evaluate both on the
same 105 test birds. The only thing that changes is use_time_features, so any difference is the diurnal feature's effect 
and nothing else.

Baseline reference: the cadence sweep's 6h XGBoost (features off) was median 1.007 km, p99 97.4 km; the "off" run here 
should reproduce that (same split, same segments), which is the coherence check that this A/B sits on the sweep's foundation.

Note: XGBoost's time of day feature is local solar corrected inside _build_features (UTC hour shifted by longitude/15, since 
storks span ~3h of longitude and UTC hour otherwise smears the diurnal rhythm across birds). So the "time_on" run here is the 
corrected feature, not raw UTC. Result to date: solar correction lifts the body (median/p90 to +4-5% vs off) while leaving the 
p99 migration tail basically unchanged (~+2.4%); the diurnal benefit is in day to day movement, not the onset tail, which is a 
seasonal/calendar event time of day perhaps cannot reach and another possible extension...
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
from animal_movement_fm.models.xgboost import XGBoost


REPO = Path(__file__).resolve().parents[2]
STORK_FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
SEGMENTS_6H = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "results"

BIN_SECONDS = 21600 # 6h
MAX_GAP_BINS = 8  # >2d at 6h
MIN_SEGMENT_BINS = 120
CONTEXT_LENGTH = 24


def build_segments_for_ids(con, fixes_parquet, individual_ids) -> pd.DataFrame:
    """Resample the given individuals at 6h (rebuild fallback if the cache is gone)."""
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
            WHERE individual_id = {individual_id} AND deployment_id = {deployment_id}
            ORDER BY timestamp
        """).df()
        segments = resample_trajectory(
            raw["timestamp"].to_numpy(),
            raw["location_lat"].to_numpy(),
            raw["location_long"].to_numpy(),
            bin_seconds=BIN_SECONDS,
            max_gap_bins=MAX_GAP_BINS,
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
    return pd.concat(frames, ignore_index=True)


def ensure_segments(con, path, ids) -> Path:
    """Use the sweeps cached 6h segments if present; else rebuild and cache."""
    if path.exists():
        print(f"  using cached {path.name}")
        return path
    print(f"  {path.name} missing; rebuilding...")
    build_segments_for_ids(con, STORK_FIXES, ids).to_parquet(path, index=False)
    return path


def ladder(results: pd.DataFrame) -> dict:
    err = results["error_m"].to_numpy() / 1000.0
    return {
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "p99_km": float(np.percentile(err, 99)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    counts = window_counts_per_individual(SEGMENTS_6H, context_length=24)  # Tier 1, pre A6
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    print(f"split: {len(split.train)} train, {len(split.test)} test birds\n")

    train_seg = ensure_segments(con, RESULTS / "segments_sweep_train_6h.parquet", split.train)
    test_seg = ensure_segments(con, RESULTS / "segments_sweep_6h.parquet", split.test)
    train_frame = load_split_frame(train_seg, split.train)
    print()

    rows = {}
    for use_time in (False, True):
        tag = "time_on" if use_time else "time_off"
        print(f"[{tag}] fitting XGBoost on {len(train_frame):,} rows "
              f"(use_time_features={use_time})...")
        model = XGBoost(use_time_features=use_time).fit(
            train_frame, individual_col="segment_id")
        # Confirm the layout actually reflects the toggle.
        has_time = "sin_hour" in (model.feature_names_ or [])
        res = evaluate_on_split(model, test_seg, split.test,
                                context_length=CONTEXT_LENGTH, show_progress=False)
        rows[tag] = ladder(res)
        rows[tag]["time_in_layout"] = has_time
        print(f"         median={rows[tag]['median_km']:.3f} km  "
              f"p90={rows[tag]['p90_km']:.2f} km  p99={rows[tag]['p99_km']:.1f} km  "
              f"(sin_hour in features: {has_time})\n")

    off, on = rows["time_off"], rows["time_on"]
    print(f"6h XGBoost time-off  <- coherence check vs sweep (want ~1.007 / 97.4): "
          f"{off['median_km']:.3f} / {off['p99_km']:.1f}")
    print("\nEffect of UTC time-of-day feature (on vs off):")
    for pct in ("median_km", "p90_km", "p99_km"):
        delta = (off[pct] - on[pct]) / off[pct]
        print(f"  {pct:<10} off={off[pct]:8.3f}  on={on[pct]:8.3f}  "
              f"change={delta:+.1%}")

    pd.DataFrame([
        {"variant": "time_off", **off},
        {"variant": "time_on", **on},
    ]).to_csv(RESULTS / "xgboost_timefeature_6h.csv", index=False)
    print(f"\nsummary -> {RESULTS / 'xgboost_timefeature_6h.csv'}")
    print("\nReminder: UTC timestamps => smeared diurnal signal. This is a LOWER BOUND; "
          "local-solar correction is the next increment if this shows promise.")

if __name__ == "__main__":
    main()