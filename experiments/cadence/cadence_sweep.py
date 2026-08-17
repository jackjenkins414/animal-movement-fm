# GenAI reference: Again I did the diagnostic framing, imports/constants, max_gap_bins_for(), summarise_km(), and most 
# of build_test_segments(), and used GenAI helping refine parts of the segment-building loop plus writing main() to check
#  the cadence sweep flow and produce clear outputs. The script was verified by confirming the 6h row reproduced the 
# canonical split/build behaviour, checking max_gap_bins matched the 2 day rule at each cadence, ensuring the same held 
# out test birds were reused across cadences, and comparing saved segment/persistence summary outputs against the printed table.
"""
Cadence sweep: persistence only across the cadence grid.

Resamples the held out TEST storks at each candidate cadence, builds segments with a time-based gap threshold and evaluates persistence 
through the existing harness. This pass now does:

  1. Validates the 'resample at C machinery'. The 6h row should reproduce the canonical segment build (~1,215 segments corpus wide; here,
     the test bird subset of it).
  2. Quantifies the step-scaling problem: persistence error in km SHRINKS at finer cadence for trivial reasons (smaller true steps), which 
     is exactly why absolute km is not comparable across cadences and why we pass reports skill RELATIVE to same cadence persistence instead. 
     The per cadence persistence numbers here are that relative metric's denominator.
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
from animal_movement_fm.data.benchmark import evaluate_on_split
from animal_movement_fm.models.persistence import Persistence


REPO = Path(__file__).resolve().parents[2]
STORK_FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
SEGMENTS_6H = REPO / "data" / "processed" / "stork_segments.parquet"  
RESULTS = REPO / "results"

# coarse -> fine, so the cheap but starved runs validate the loop before the dense ones
CADENCE_GRID = {
    "24h": 86400,
    "12h": 43200,
    "6h": 21600,
    "3h": 10800,
    "1h": 3600,
    "30min": 1800,
}

SPLIT_DAYS = 2   # >2d gap 
MIN_SEGMENT_BINS = 120 # held in bins
CONTEXT_LENGTH = 24  # held in bins (identical model input shape across cadences)


def max_gap_bins_for(bin_seconds: int) -> int:
    """Convert the >2d split rule to a bin count at this cadence (6h -> 8)."""
    return round(SPLIT_DAYS * 86400 / bin_seconds)


def build_test_segments(
    con: duckdb.DuckDBPyConnection,
    fixes_parquet: Path,
    individual_ids: np.ndarray,
    bin_seconds: int,
    max_gap_bins: int,
) -> pd.DataFrame:
    """Resample the given individuals at `bin_seconds`, return tidy segment rows.

    One (individual_id, deployment_id) trajectory read at a time from the sorted staging parquet, so memory stays bounded. 
    Mirrors build_segments.py pass two, cadence parameterised.
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


def summarise_km(results: pd.DataFrame) -> dict:
    """Persistence error ladder in km from the harness per-prediction frame."""
    err_km = results["error_m"].to_numpy() / 1000.0
    return {
        "n_pred": int(err_km.size),
        "median_km": float(np.median(err_km)),
        "p90_km": float(np.percentile(err_km, 90)),
        "p99_km": float(np.percentile(err_km, 99)),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # GenAI comment I have left in:
    # ---- Canonical individual-level split (seed=42), derived once from the 6h build and
    # reused at every cadence so the SAME test birds are compared throughout. This
    # matters: performance is dominated by the individual (~900x spread), so varying the
    # test set across cadences would confound the comparison. ---- 
    counts = window_counts_per_individual(SEGMENTS_6H, context_length=24)  # Tier 1, pre A6
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    test_ids = split.test
    print(f"canonical split: {len(test_ids)} held-out test birds "
          f"(reused across all cadences)\n")

    rows = []
    for label, bin_seconds in CADENCE_GRID.items():
        mgb = max_gap_bins_for(bin_seconds)
        seg = build_test_segments(con, STORK_FIXES, test_ids, bin_seconds, mgb)

        n_seg = seg["segment_id"].nunique()
        n_birds = seg["individual_id"].nunique()
        note = "  <- reproduces canonical 6h build" if label == "6h" else ""
        print(f"[{label:>5}]  max_gap_bins={mgb:>3}  "
              f"segments={n_seg:>5}  birds={n_birds:>4}  bins={len(seg):>9,}{note}")

        if n_seg == 0:
            print(f"          no surviving segments at {label} "
                  f"(120-bin floor = {120 * bin_seconds / 86400:.0f} days) -- skipping\n")
            rows.append({"cadence": label, "seconds": bin_seconds, "max_gap_bins": mgb,
                         "n_segments": 0, "n_test_birds": 0, "n_pred": 0,
                         "median_km": np.nan, "p90_km": np.nan, "p99_km": np.nan})
            continue

        seg_path = RESULTS / f"segments_sweep_{label}.parquet"
        seg.to_parquet(seg_path, index=False)

        results = evaluate_on_split(
            Persistence(), seg_path, test_ids,
            context_length=CONTEXT_LENGTH, show_progress=False,
        )
        summ = summarise_km(results)
        print(f"          persistence:  median={summ['median_km']:.3f} km  "
              f"p90={summ['p90_km']:.2f} km  p99={summ['p99_km']:.1f} km  "
              f"(n={summ['n_pred']:,})\n")

        rows.append({"cadence": label, "seconds": bin_seconds, "max_gap_bins": mgb,
                     "n_segments": n_seg, "n_test_birds": n_birds, **summ})

    table = pd.DataFrame(rows)
    out_path = RESULTS / "cadence_sweep_persistence.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nsummary -> {out_path}")
    print("\nNote: median_km falls at finer cadence for trivial step-scaling reasons; "
          "this is the per-cadence persistence baseline, NOT a cross-cadence ranking. "
          "The learner pass reports skill relative to these numbers.")


if __name__ == "__main__":
    main()