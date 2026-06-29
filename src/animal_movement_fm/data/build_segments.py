# GenAI reference: We used GenAI to help draft the DuckDB SQL queries in this file. We validated by checking 
# survivor counts against the cadence analysis, confirming the staged parquet contained only the expected stork
# trajectories, and also verified that per-trajectory resampling produced the expected number of 6h bins (it did). 
"""
Build the stork training set; so we resample all 6h survivor trajectories into segments.

Applies the validated 6h resampler (data.resample) across every stork trajectory that survives 
the 6h-cadence / >=120-point filter, and writes the resulting segments to one tidy parquet 
(one row per 6h bin). This is the per-species dataset the Tier-1 specialist ladder and the FM
pretraining read.

Run once; the expensive binning is materialised so downstream modelling loads
ready-made segments. Paths are module constants for now; they graduate to the
project config later.
"""

from pathlib import Path
import time


import duckdb
import pandas as pd 

from animal_movement_fm.data.resample import resample_trajectory

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

        segments = resample_trajectory(
            raw["timestamp"].to_numpy(),
            raw["location_lat"].to_numpy(),
            raw["location_long"].to_numpy(),
        )

        if not segments:
            n_dropped_traj += 1
            continue

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
            n_segments += 1

    out = pd.concat(frames, ignore_index=True)
    print(f"{n_segments} segments from {len(traj_keys)} trajectories "
          f"({n_dropped_traj} yielded none) | {len(out):,} total 6h bins")
    return out

