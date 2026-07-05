# GenAI reference: Here I implemented the load_binned_fixes, per_bin_spread, add_step_length, and the docstrings
# imports etc. I used GenAI to help write the report, plot and main method, and I verified its outpiut by checking 
# that the raw fixes joined only to surviving (individual_id, bin_index) bins using the exact resampler bin formula, 
# confirming step_km was computed within segment boundaries, and checking the parquet/scatter outputs.
"""
Within 6h bin variance: how lossy is the resampler's one point per-bin collapse? (Again for obsidian)

Goes back to the raw fixes (stork_fixes.parquet) and asks, for each 6h bin that survived into stork_segments.parquet: 
how far did the bird actually range DURING those 6 hours, before we collapsed it to a single representative point?

Bins are reconstructed with resample.py's exact formula
    bin_index = floor(epoch_seconds / 21600)
so raw fixes join to segment bins by exact integer equality on (individual_id, bin_index); identical bin edges.

Reports, per surviving bin:
  n_fixes    raw fixes that fell in the bin (1 => nothing was thrown away / recorded)
  spread_km  max distance of any raw fix from the bin representative point
  path_km    summed consecutive travel through the raw fixes (actual in-window path)

Headline: within bin spread vs the bin's own step length. If spread climbs in the long step (migration) regime, our 
6h grid is lossy exactly where movement is fast; which argues for a finer cadence (will do this in future). 

note: n_fixes distribution is reported FIRST. If most bins hold one raw fix, within bin movement was never recorded and 
the analysis is bounded by raw cadence; itself a finding.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"  

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from animal_movement_fm.eval.metrics import haversine_distance

REPO = Path(__file__).resolve().parents[1]
FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"

BIN_SECONDS = 6 * 60 * 60  # we match resample.py
REGIME_CUT_KM = 0.06  # foraging/moving trough from step-displacement 


def load_binned_fixes() -> pd.DataFrame:
    """
    Raw fixes tagged with resample.py's bin_index, restricted to surviving bins.

    SEMI JOIN against the (individual_id, bin_index) pairs actually present in stork_segments.parquet, so we 
    only measure spread for bins the model uses. Also pulls each bin's representative lon/lat (from segments) so 
    spread can be measured against the exact point the pipeline kept.
    """
    return duckdb.sql(f"""
        WITH raw AS (
            SELECT
                individual_id,
                timestamp,
                location_long AS lon,
                location_lat  AS lat,
                CAST(floor(epoch(timestamp) / {BIN_SECONDS}) AS BIGINT) AS bin_index
            FROM '{FIXES}'
        ),
        bins AS (
            SELECT
                individual_id,
                bin_index,
                lon AS rep_lon,
                lat AS rep_lat
            FROM '{SEGMENTS}'
        )
        SELECT
            r.individual_id,
            r.bin_index,
            r.timestamp,
            r.lon, r.lat,
            b.rep_lon, b.rep_lat
        FROM raw r
        JOIN bins b
          ON r.individual_id = b.individual_id
         AND r.bin_index     = b.bin_index
        ORDER BY r.individual_id, r.bin_index, r.timestamp
    """).df()


def per_bin_spread(binned: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw fixes to one row per bin with n_fixes, spread_km, path_km."""
    # distance of each raw fix from its bin representative point
    binned = binned.copy()
    binned["dist_to_rep_km"] = haversine_distance(
        binned["lon"].to_numpy(), binned["lat"].to_numpy(),
        binned["rep_lon"].to_numpy(), binned["rep_lat"].to_numpy(),
    ) / 1000.0

    # consecutive within-bin travel (path length), computed per bin in time order
    def path_km(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return 0.0
        lon = g["lon"].to_numpy(); lat = g["lat"].to_numpy()
        seg = haversine_distance(lon[:-1], lat[:-1], lon[1:], lat[1:]) / 1000.0
        return float(seg.sum())

    grouped = binned.groupby(["individual_id", "bin_index"])
    out = grouped.agg(
        n_fixes=("timestamp", "size"),
        spread_km=("dist_to_rep_km", "max"),
    ).reset_index()
    out["path_km"] = grouped.apply(path_km, include_groups=False).values
    return out


def add_step_length(spread: pd.DataFrame) -> pd.DataFrame:
    """Attach each bins OUTGOING 6h step length to split by regime."""
    reps = duckdb.sql(f"""
        WITH ordered AS (
            SELECT
                individual_id, segment_id, bin_index, lon, lat,
                LAG(lon) OVER w AS plon,
                LAG(lat) OVER w AS plat
            FROM '{SEGMENTS}'
            WINDOW w AS (PARTITION BY segment_id ORDER BY bin_index)
        )
        SELECT individual_id, bin_index, plon, plat, lon, lat
        FROM ordered WHERE plon IS NOT NULL
    """).df()
    reps["step_km"] = haversine_distance(
        reps["plon"].to_numpy(), reps["plat"].to_numpy(),
        reps["lon"].to_numpy(), reps["lat"].to_numpy(),
    ) / 1000.0
    reps = reps[["individual_id", "bin_index", "step_km"]]
    return spread.merge(reps, on=["individual_id", "bin_index"], how="inner")


def report(spread: pd.DataFrame) -> None:
    n = len(spread)
    print(f"{n:,} surviving bins with raw fixes\n")

    print("n_fixes per bin (does the raw data even let us see within-bin motion?):")
    vc = spread["n_fixes"].value_counts().sort_index()
    for k in [1, 2, 3, 4, 5]:
        c = int(vc.get(k, 0))
        print(f"  {k} fix{'es' if k > 1 else '  '}: {c:>9,} ({c/n:6.1%})")
    multi = int((spread["n_fixes"] >= 2).sum())
    print(f"  >=2 fixes: {multi:,} ({multi/n:.1%})  <- only these have measurable spread")
    print(f"  median n_fixes = {spread['n_fixes'].median():.0f}, "
          f"max = {spread['n_fixes'].max()}\n")

    measurable = spread[spread["n_fixes"] >= 2]
    if len(measurable):
        print("within-bin spread_km (bins with >=2 fixes):")
        for p in [50, 90, 99]:
            print(f"  p{p}: {np.percentile(measurable['spread_km'], p):.3f} km")
        print()
        forag = measurable[measurable["step_km"] < REGIME_CUT_KM]
        migr  = measurable[measurable["step_km"] >= REGIME_CUT_KM]
        print(f"foraging bins (step < {REGIME_CUT_KM} km): "
              f"median spread {forag['spread_km'].median():.3f} km  (n={len(forag):,})")
        print(f"moving   bins (step >= {REGIME_CUT_KM} km): "
              f"median spread {migr['spread_km'].median():.3f} km  (n={len(migr):,})")


def plot(spread: pd.DataFrame, out_path: Path) -> None:
    measurable = spread[spread["n_fixes"] >= 2]
    if not len(measurable):
        print("no multi-fix bins; skipping scatter")
        return
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.clip(measurable["step_km"], 1e-3, None)
    ax.scatter(x, np.clip(measurable["spread_km"], 1e-3, None),
               s=4, alpha=0.15, color="#4C72B0", edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.axvline(REGIME_CUT_KM, color="#C44E52", linestyle="--", linewidth=1.5,
               label=f"regime cut \u2248 {REGIME_CUT_KM*1000:.0f} m")
    ax.set_xlabel("bin 6h step length (km)")
    ax.set_ylabel("within-bin spread (km)")
    ax.set_title("Within-bin spread vs step length\n(is 6h lossy where movement is fast?)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    binned = load_binned_fixes()
    print(f"joined {len(binned):,} raw fixes into surviving bins\n")

    spread = add_step_length(per_bin_spread(binned))
    report(spread)

    out_parquet = RESULTS / "within_bin_variance.parquet"
    spread.to_parquet(out_parquet, index=False)
    plot_path = RESULTS / "within_bin_spread_vs_step.png"
    plot(spread, plot_path)
    print(f"\nper-bin spread -> {out_parquet}")
    print(f"scatter        -> {plot_path}")


if __name__ == "__main__":
    main()