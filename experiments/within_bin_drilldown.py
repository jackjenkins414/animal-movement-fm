# GenAI reference: I implemented the whole file at first (whole main workflow). GenAI had to make a number of corrections to 
# suspect_bins() and classify() after my initial logic risked computing suspect criteria at the wrong row/bin level. The 
# revised version was verified by checking that raw fixes were joined to the correct individual_id/bin_index, confirming
# step_km was constant within each suspect bin, and inspecting labelled REAL/NOISE samples for biologically plausibility.
# And minor print statement fixes in main too. 
"""
Calling this a 'Drill-down': is the low-step / high-spread cloud real movement or bad fixes? 

Within bin variance from before showed bins with near-zero 6h STEP (representative points nearly co-located => labelled 
"stationary") yet large within-bin SPREAD (a raw fix tens of km from the representative). Two explanations:

  REAL  : a smooth out and back foraging loop that the 6h grid can't see -> many fixes, sane consecutive hops, 
          spread ~ median distance
  NOISE : one wild GPS fix among an otherwise tight cluster -> spread >> median distance, one impossible consecutive hop

Discriminators, per suspect bin:
  spread_km        max raw fix distance from bin representative 
  median_dist_km   median raw fix distance from representative
  spike_ratio      spread /median_dist  (huge => single outlier => NOISE)
  max_hop_km       largest distance between consecutive raw fixes in time order (>~300 km in 6h is biologically impossible => NOISE)
  max_hop_kmh      that hop as an implied speed

Prints a labelled sample so we can eyeball, plus an aggregate real/noise split.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from animal_movement_fm.eval.metrics import haversine_distance

REPO = Path(__file__).resolve().parents[1]
FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"

BIN_SECONDS = 6 * 60 * 60
REGIME_CUT_KM = 0.06  # foraging/moving trough 
SUSPECT_STEP_KM = 0.06 # "stationary" by step length
SUSPECT_SPREAD_KM = 30.0 # but large within bin spread
STORK_MAX_KMH = 80.0 # generous single hop ceiling for a stork? check lit 
N_SAMPLE = 25 # bins to print individually


def suspect_bins() -> pd.DataFrame:
    """
    Bins with tiny 6h step but large within-bin spread; the cloud in question.

    Recomputes step length from segments (LAG) and joins the raw-fix spread we already dumped in within_bin_variance.parquet is 
    avoided here; instead we recompute spread from raw fixes so this script is self-contained.
    """
    return duckdb.sql(f"""
        WITH raw AS (
            SELECT individual_id, timestamp,
                   location_long AS lon, location_lat AS lat,
                   CAST(floor(epoch(timestamp) / {BIN_SECONDS}) AS BIGINT) AS bin_index
            FROM '{FIXES}'
        ),
        stepped AS (
            SELECT individual_id, bin_index, lon, lat,
                   LAG(lon) OVER w AS plon, LAG(lat) OVER w AS plat
            FROM '{SEGMENTS}'
            WINDOW w AS (PARTITION BY segment_id ORDER BY bin_index)
        )
        SELECT s.individual_id, s.bin_index, s.plon, s.plat, s.lon AS rep_lon, s.lat AS rep_lat,
               r.timestamp, r.lon, r.lat
        FROM stepped s
        JOIN raw r
          ON s.individual_id = r.individual_id AND s.bin_index = r.bin_index
        WHERE s.plon IS NOT NULL
        ORDER BY s.individual_id, s.bin_index, r.timestamp
    """).df()


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """One row per suspect bin with the discriminators."""

    # step length of each bin (rep vs prev rep)
    df = df.copy()
    step = haversine_distance(df["plon"].to_numpy(), df["plat"].to_numpy(),
                              df["lon"].to_numpy(), df["lat"].to_numpy()) / 1000.0
    df["step_km"] = step

    # distance of each raw fix from the bin representative
    df["dist_km"] = haversine_distance(
        df["lon"].to_numpy(), df["lat"].to_numpy(),
        df["rep_lon"].to_numpy(), df["rep_lat"].to_numpy()) / 1000.0

    rows = []
    for (ind, b), g in df.groupby(["individual_id", "bin_index"]):
        if len(g) < 3:
            continue
        spread = g["dist_km"].max()
        median_dist = g["dist_km"].median()
        step_km = g["step_km"].iloc[0]
        if step_km >= SUSPECT_STEP_KM or spread < SUSPECT_SPREAD_KM:
            continue # only the low step / high spread cloud

        # consecutive hop in time order
        gg = g.sort_values("timestamp")
        lon = gg["lon"].to_numpy(); lat = gg["lat"].to_numpy()
        ts = gg["timestamp"].to_numpy()
        hops_km = haversine_distance(lon[:-1], lat[:-1], lon[1:], lat[1:]) / 1000.0
        dt_h = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0
        dt_h = np.where(dt_h <= 0, np.nan, dt_h)
        max_hop_i = int(np.nanargmax(hops_km)) if len(hops_km) else 0
        max_hop_km = float(hops_km[max_hop_i]) if len(hops_km) else 0.0
        max_hop_kmh = float(hops_km[max_hop_i] / dt_h[max_hop_i]) if len(hops_km) else 0.0

        rows.append({
            "individual_id": ind, "bin_index": b, "n_fixes": len(g),
            "step_km": step_km, "spread_km": spread, "median_dist_km": median_dist,
            "spike_ratio": spread / max(median_dist, 1e-3),
            "max_hop_km": max_hop_km, "max_hop_kmh": max_hop_kmh,
        })
    return pd.DataFrame(rows)


def label(res: pd.DataFrame) -> pd.DataFrame:
    """NOISE if an impossible hop OR a lone spike; else REAL."""
    impossible_hop = res["max_hop_kmh"] > STORK_MAX_KMH
    lone_spike = res["spike_ratio"] > 10.0
    res = res.copy()
    res["verdict"] = np.where(impossible_hop | lone_spike, "NOISE", "REAL")
    return res


def main() -> None:
    res = label(classify(suspect_bins()))
    if not len(res):
        print("no suspect bins matched the lowstep / high-spread criteria.")
        return

    n = len(res)
    n_noise = int((res["verdict"] == "NOISE").sum())
    print(f"{n:,} suspect bins (step < {SUSPECT_STEP_KM} km, spread > {SUSPECT_SPREAD_KM} km)\n")
    print(f"  NOISE (bad fix): {n_noise:,} ({n_noise/n:.1%})")
    print(f"  REAL  (loop)   : {n-n_noise:,} ({(n-n_noise)/n:.1%})\n")

    # sample sorted so the reader sees a spread of both verdicts
    sample = pd.concat([
        res[res["verdict"] == "REAL"].head(N_SAMPLE // 2),
        res[res["verdict"] == "NOISE"].head(N_SAMPLE // 2),
    ])
    cols = ["individual_id", "bin_index", "n_fixes", "spread_km",
            "median_dist_km", "spike_ratio", "max_hop_km", "max_hop_kmh", "verdict"]
    with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
        print(sample[cols].to_string(index=False))


if __name__ == "__main__":
    main()