# GenAI reference: I used GenAI to help draft the DuckDB SQL queries in this file. I validated it by checking
# survivor counts against the cadence analysis, confirming the staged parquet contained only the expected stork
# trajectories, and also verified that per trajectory resampling produced the expected number of 6h bins (it did).
"""
Build the stork training set; so we resample all 6h survivor trajectories into segments.

Applies the validated 6h resampler (data.resample) across every stork trajectory that survives the 6h-cadence 
/ >=120-point filter, and writes the resulting segments to one tidy parquet (one row per 6h bin). This is the 
per-species dataset the Tier 1 specialist ladder and the FM pretraining read.

Run once; the expensive binning is materialised so downstream modelling loads ready-made segments. Paths are module 
constants for now; they graduate to the project config later.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from animal_movement_fm.data.resample import (
    BIN_SECONDS,
    MAX_GAP_BINS,
    MIN_SEGMENT_BINS,
    resample_trajectory,
)

# paths local dev; absolute so the script is cwd-independent)
DATA_DIR = Path.home() / "moveformer-data"
RAW_PARQUET = DATA_DIR / "locations_gps.parquet"
TRAJ_SUMMARY = DATA_DIR / "trajectory_summary.parquet"

OUT_DIR = Path.home() / "animal-movement-fm" / "data" / "processed"
STORK_FIXES = OUT_DIR / "stork_fixes.parquet"  # staging: raw fixes, sorted
STORK_SEGMENTS = OUT_DIR / "stork_segments.parquet"  # final: resampled segments

# build species + survivor criteria (must match the cadence analysis)
SPECIES = "Ciconia ciconia"
CADENCE_MIN = 360 # 6h native-cadence ceiling (no upsampling)
MIN_POINTS = 120  # >=120 points after downsampling to 6h


def trajectory_segment_frames(
    timestamp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    individual_id,
    deployment_id=None,
    bin_seconds: int = BIN_SECONDS,
    min_segment_bins: int = MIN_SEGMENT_BINS,
    max_gap_bins: int = MAX_GAP_BINS,
    extra: dict | None = None,
) -> list[pd.DataFrame]:
    """
    Resample one trajectory into segment DataFrames.

    Supports integer or string individual IDs and optional deployment IDs. When a deployment ID is present, we 
    include it in the segment ID so separate deployments are never joined.

    `extra` can be used to attach individual level metadata such as device type, sex, or taxon to every output row; yet to
    make use of it yet, but it will be something we will explore later. 
    """

    segments = resample_trajectory(
        timestamp, lat, lon, bin_seconds=bin_seconds, min_segment_bins=min_segment_bins, max_gap_bins=max_gap_bins
    )
    if not segments:
        return []

    prefix = f"{individual_id}" if deployment_id is None else f"{individual_id}_{deployment_id}"
    frames = []
    for k, seg in enumerate(segments):
        frame = pd.DataFrame({
            "segment_id": f"{prefix}_{k}",
            "individual_id": individual_id,
            "bin_index": seg.bin_index,
            "timestamp": seg.timestamp,
            "lat": seg.lat,
            "lon": seg.lon,
            "centre_offset_s": seg.centre_offset_s,
        })
        if deployment_id is not None:
            frame.insert(2, "deployment_id", deployment_id)
        for column, value in (extra or {}).items():
            frame[column] = value
        frames.append(frame)
    return frames


def segments_from_fixes(
    fixes: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    bin_seconds: int = BIN_SECONDS,
    min_segment_bins: int = MIN_SEGMENT_BINS,
    max_gap_bins: int = MAX_GAP_BINS,
    carry_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame | None, dict]:
    """
    Resample an in memory table of raw fixes into one segments DataFrame.

    This is a convenience wrapper around `trajectory_segment_frames` for per-species datasets. Returns 
    `(segments, stats)`, with `segments=None` when no trajectory meets `min_segment_bins`.
    """

    required = {"individual_id", "timestamp", lat_col, lon_col}
    missing = required - set(fixes.columns)
    if missing:
        raise ValueError(f"fixes is missing required columns: {sorted(missing)}")

    fixes = fixes.copy()
    fixes["individual_id"] = fixes["individual_id"].astype(str).str.strip()
    fixes["timestamp"] = pd.to_datetime(fixes["timestamp"], errors="coerce")
    fixes = fixes[fixes["timestamp"].notna()]

    group_cols = ["individual_id"]
    has_deployment = "deployment_id" in fixes.columns and fixes["deployment_id"].notna().any()
    if has_deployment:
        fixes["deployment_id"] = fixes["deployment_id"].astype(str).str.strip()
        group_cols.append("deployment_id")

    ordered = fixes.sort_values(group_cols + ["timestamp"])

    # native cadence, reported as a diagnostic and is never enforced here; a track much coarser than the bin resamples 
    # into mostly empty bins and fails the window floor at the eligibility gate, which is where that decision belongs 
    # and gets recorded
    deltas = ordered.groupby(group_cols, sort=False)["timestamp"].diff().dt.total_seconds().dropna()
    median_gap_h = float(np.median(deltas) / 3600) if len(deltas) else np.nan
    share_within_bin = float((deltas <= bin_seconds).mean()) if len(deltas) else np.nan

    carried = {
        column: fixes.groupby("individual_id")[column].first()
        for column in carry_columns
        if column in fixes.columns
    }

    frames: list[pd.DataFrame] = []
    n_trajectories = 0
    n_empty = 0
    for keys, track in ordered.groupby(group_cols, sort=True):
        n_trajectories += 1
        keys = keys if isinstance(keys, tuple) else (keys,)
        produced = trajectory_segment_frames(
            track["timestamp"].to_numpy(),
            track[lat_col].to_numpy(dtype=float),
            track[lon_col].to_numpy(dtype=float),
            individual_id=keys[0],
            deployment_id=keys[1] if has_deployment else None,
            bin_seconds=bin_seconds,
            min_segment_bins=min_segment_bins,
            max_gap_bins=max_gap_bins,
        )
        if not produced:
            n_empty += 1
        frames.extend(produced)

    stats = {
        "fixes_in": len(fixes),
        "individuals_in": fixes["individual_id"].nunique(),
        "trajectories": n_trajectories,
        "trajectories_no_segments": n_empty,
        "median_native_gap_h": round(median_gap_h, 3),
        "share_gaps_within_bin": round(share_within_bin, 4),
        "min_segment_bins": min_segment_bins,
        "max_gap_bins": max_gap_bins,
        "segments": 0,
        "bins": 0,
        "individuals_out": 0,
        "first_bin": pd.NaT,
        "last_bin": pd.NaT,
    }
    if not frames:
        return None, stats

    out = pd.concat(frames, ignore_index=True)
    for column, values in carried.items():
        out[column] = out["individual_id"].map(values)

    stats.update({
        "segments": out["segment_id"].nunique(),
        "bins": len(out),
        "individuals_out": out["individual_id"].nunique(),
        "first_bin": out["timestamp"].min(),
        "last_bin": out["timestamp"].max(),
    })
    return out, stats


def stage_survivor_fixes(con: duckdb.DuckDBPyConnection) -> tuple[int, int]:
    """One pass over the 8GB parquet: write stork survivors' raw fixes, globally
    sorted by (individual_id, deployment_id, timestamp) so the per-trajectory
    reads in the next stage prune row groups instead of rescanning.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            WITH survivors AS (
                SELECT individual_id, deployment_id
                FROM '{TRAJ_SUMMARY}'
                WHERE taxon_canonical_name = '{SPECIES}'
                  AND date_diff('minute', first_obs, last_obs)
                      / nullif(n_observations - 1, 0) <= {CADENCE_MIN}
                  AND date_diff('minute', first_obs, last_obs) / {CADENCE_MIN} >= {MIN_POINTS}
            )
            SELECT l.individual_id, l.deployment_id,
                   l.timestamp, l.location_lat, l.location_long
            FROM '{RAW_PARQUET}' l
            JOIN survivors s
              ON l.individual_id = s.individual_id
             AND l.deployment_id = s.deployment_id
            ORDER BY l.individual_id, l.deployment_id, l.timestamp
        ) TO '{STORK_FIXES}' (FORMAT PARQUET)
    """)
    n_rows = con.execute(f"SELECT COUNT(*) FROM '{STORK_FIXES}'").fetchone()[0]
    n_traj = con.execute(
        f"SELECT COUNT(*) FROM (SELECT DISTINCT individual_id, deployment_id "
        f"FROM '{STORK_FIXES}')"
    ).fetchone()[0]
    return n_rows, n_traj

def build_segments(con: duckdb.DuckDBPyConnection) -> "pd.DataFrame":
    """Pass two: resample each staged trajectory into segments, return tidy rows.

    Reads one (individual_id, deployment_id) trajectory at a time from the sorted
    staging parquet, applies the 6h resampler, and tags each surviving segment with
    a composite segment_id. Memory stays bounded to one trajectory's fixes at a time.
    """
    import pandas as pd

    # the list of trajectories to walk, in the staged sort order.
    traj_keys = con.execute(f"""
        SELECT DISTINCT individual_id, deployment_id
        FROM '{STORK_FIXES}'
        ORDER BY individual_id, deployment_id
    """).fetchall()

    frames = []
    n_segments = 0
    n_dropped_traj = 0 # trajectories that yielded zero surviving segments

    for individual_id, deployment_id in traj_keys:
        raw = con.execute(f"""
            SELECT timestamp, location_lat, location_long
            FROM '{STORK_FIXES}'
            WHERE individual_id = {individual_id}
              AND deployment_id = {deployment_id}
            ORDER BY timestamp
        """).df()

        produced = trajectory_segment_frames(
            raw["timestamp"].to_numpy(),
            raw["location_lat"].to_numpy(),
            raw["location_long"].to_numpy(),
            individual_id=individual_id,
            deployment_id=deployment_id,
        )

        if not produced:
            n_dropped_traj += 1
            continue

        frames.extend(produced)
        n_segments += len(produced)

    out = pd.concat(frames, ignore_index=True)
    print(f"{n_segments} segments from {len(traj_keys)} trajectories "
          f"({n_dropped_traj} yielded none) | {len(out):,} total 6h bins")
    return out

