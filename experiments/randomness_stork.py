# GenAI reference: Here I did the core directional persistence analysis and real stork adaptation and used GenAI 
# to refine explanatory comments, clean terminal output (main method modific.), and make minor plot labelling improvements.
"""Directional-persistence (randomness) distribution on the 6h stork segments.

Adapts the synthetic phase randomness index to real (lon, lat) stork data. Two corrections vs the synthetic version, 
both required for the number to be valid:

  1. Turning angles are computed in LOCAL METRES, not raw degrees. Storks span ~50N->Africa, where a degree of longitude 
     is much shorter than a degree of latitude; planar angles on raw (lon, lat) are distorted. Thus we project each
     step to metres via an equirectangular scaling (lon *= cos(mean_lat)) before computing angles; accurate for per-track 
     turning-angle stats and consistent with the haversine metres the rest of the pipeline uses.

  2. rho is computed per SEGMENT (segment_id), the 6h gap-clean unit; never across a transmission gap, and at the fixed 
     6h analysis cadence (rho is not cadence-invariant, so it must be measured at the cadence models see).
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

MIN_STEPS = 10   # need enough steps for a meaningful per-segment rho


def directional_persistence_metres(lon: np.ndarray, lat: np.ndarray,
                                   *, eps: float = 1e-9) -> float:
    """Mean cosine of turning angles, computed in local metres.

    Equirectangular projection about the track's mean latitude: x = lon*cos(lat0), y = lat, both then scaled to metres. 
    Angles are invariant to the common metre scaling, so we only need the cos(lat0) longitude correction to undo the 
    degree-anisotropy before taking turning angles.
    """
    lat0 = np.deg2rad(np.mean(lat))
    x = lon * np.cos(lat0) # longitude corrected for latitude shrink
    y = lat # latitude unchanged
    pos = np.column_stack([x, y])

    steps = np.diff(pos, axis=0) # (T-1, 2)
    lengths = np.hypot(steps[:, 0], steps[:, 1]) # (T-1,)
    valid = lengths > eps
    steps, lengths = steps[valid], lengths[valid]
    if len(steps) < 2:
        return np.nan

    units = steps / lengths[:, None]
    cos_turns = np.sum(units[:-1] * units[1:], axis=1)  # cos of turning angle
    return float(np.mean(cos_turns))


def main() -> None:
    con = duckdb.connect()

    # Pull one segment at a time's (lon, lat) in bin order. Segments are already
    # the 6h gap-clean unit, so no gap can fall inside a track here.
    seg_ids = con.execute(f"""
        SELECT DISTINCT segment_id FROM '{SEGMENTS}'
    """).df()["segment_id"].tolist()
    print(f"{len(seg_ids)} segments to process", flush=True)

    rows = []
    for i, sid in enumerate(seg_ids, 1):
        g = con.execute(f"""
            SELECT lon, lat, individual_id
            FROM '{SEGMENTS}'
            WHERE segment_id = ?
            ORDER BY bin_index
        """, [sid]).df()

        if len(g) - 1 < MIN_STEPS:
            continue

        rho = directional_persistence_metres(g["lon"].to_numpy(), g["lat"].to_numpy())
        rows.append({
            "segment_id": sid,
            "individual_id": g["individual_id"].iloc[0],
            "n_steps": len(g) - 1,
            "persistence": rho,
            "randomness": 1.0 - rho if rho == rho else np.nan,  # NaN-safe
        })
        if i % 100 == 0:
            print(f"  [{i}/{len(seg_ids)}]", flush=True)

    res = pd.DataFrame(rows).dropna(subset=["persistence"])
    res.to_csv("results/randomness_stork.csv", index=False)

    print("\nstork directional persistence (rho), per 6h segment:")
    print(f"  segments: {len(res)}")
    print(f"  mean   : {res['persistence'].mean():.3f}")
    print(f"  median : {res['persistence'].median():.3f}")
    print(f"  sd     : {res['persistence'].std():.3f}")
    print(f"  min/max: {res['persistence'].min():.3f} / {res['persistence'].max():.3f}")

    # Plot: stork rho distribution, with the synthetic regime anchors for context.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 41)
    ax.hist(res["persistence"], bins=bins, alpha=0.7, color="C2",
            label=f"white stork (6h segments, n={len(res)})")

    # Synthetic anchors from the validation run (BM~0, switching~0.6, CRW~0.86)
    for x, name in [(0.0, "BM ~0"), (0.6, "switching ~0.6"), (0.86, "CRW ~0.86")]:
        ax.axvline(x, color="grey", ls="--", lw=1, alpha=0.6)
        ax.text(x, ax.get_ylim()[1] * 0.92, name, rotation=90,
                va="top", ha="right", fontsize=8, color="grey")

    ax.set_xlabel("directional persistence  (rho = mean cos turning angle, metres)")
    ax.set_ylabel("number of segments")
    ax.set_title("Where real storks sit on the randomness spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "randomness_stork.png", dpi=150)
    print(f"\nsaved {OUT / 'randomness_stork.png'}")


if __name__ == "__main__":
    main()