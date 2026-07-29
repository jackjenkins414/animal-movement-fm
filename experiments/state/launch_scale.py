# GenAI reference: I wrote the outline, imports, moving_bouts(), and the underlying bout definition logic, then used GenAI 
# to finish log_histogram() and write main() with clear summaries, plotting and exports. I verified it by checking some 
# moving bouts against the labelled sequence, confirming bouts stayed within segment_id boundaries, checking that path_km summed 
# the bout's step_km values, and examining the plot for plausible local vs migratory separation
"""
Refining 'launch': separating migratory launches from ordinary local ones.

The 4-state labeller marks EVERY foraging->moving flip as a launch (which came out to ~16% of fixes, ~157 per segment). Those 
are mostly ordinary daily starts of movement, and not the rare migration departures that dominate the error tail. To make 'launch' 
mean the event we actually care about, we look one level up: at the MOVING BOUT each launch opens (the maximal run of base moving 
fixes from a launch to the next settle).

Per bout we measure:
  net_km        straight-line distance from the pre-launch position to the bout's last moving fix
  path_km       summed step length over the bout (includes the launch step)
  straightness  net_km / path_km  (1 = straight transit, ~0 = out-and-back local trip)
  n_steps       bout length in 6 h fixes (duration proxy)

We then look for a trough in the bout net-displacement distribution that separates local bouts from migratory ones; the same move 
that turned our prev 60m step trough into the foraging/moving cut (Note: a 'migration launch' will be a launch whose bout exceeds 
the chosen cut. This script only characterises; the cut is we choose from the literature. 
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from animal_movement_fm.analysis.regime import label_contiguous_regimes
from animal_movement_fm.eval.metrics import haversine_distance

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "results"


def moving_bouts(lab: pd.DataFrame) -> pd.DataFrame:
    """
    One row per moving bout with net/path/straightness/length.

    A bout is a maximal run of base_regime == 'moving' within a segment. `lab` must be the output of `label_regimes` (sorted by 
    segment/timestamp, reset index), so positional index == row order.
    """
    seg = lab["segment_id"].to_numpy()
    run = lab["contiguous_run_id"].to_numpy()
    lon = lab["lon"].to_numpy()
    lat = lab["lat"].to_numpy()
    step_km = lab["step_km"].to_numpy()
    is_mov = (lab["base_regime"] == "moving").to_numpy()

    # a bout starts at a moving fix whose predecessor in the same contiguous run is not moving
    prev_mov = np.zeros(len(lab), dtype=bool)
    prev_mov[1:] = is_mov[:-1] & (run[1:] == run[:-1])
    new_bout = is_mov & ~prev_mov
    bout_id = np.cumsum(new_bout) # constant within a bout; only read on moving fixes

    mov = np.where(is_mov)[0]
    mdf = pd.DataFrame({
        "bout": bout_id[mov], "idx": mov, "seg": seg[mov], "step_km": step_km[mov],
    })
    g = mdf.groupby("bout", sort=True)
    first_idx = g["idx"].first().to_numpy()
    last_idx = g["idx"].last().to_numpy()
    n_steps = g.size().to_numpy()
    path_km = g["step_km"].sum().to_numpy()
    bout_seg = g["seg"].first().to_numpy()

    # anchor = the fix just before the bout, if it exists in the same run
    prev_idx = first_idx - 1
    same_run = (first_idx > 0) & (
        run[np.maximum(prev_idx, 0)] == run[first_idx]
    )
    anchor_idx = np.where(same_run, prev_idx, first_idx)

    net_km = haversine_distance(
        lon[anchor_idx], lat[anchor_idx], lon[last_idx], lat[last_idx]
    ) / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        straightness = np.where(path_km > 0, net_km / path_km, np.nan)

    return pd.DataFrame({
        "segment_id": bout_seg, "n_steps": n_steps,
        "net_km": net_km, "path_km": path_km, "straightness": straightness,
    })


def log_histogram(net_km: np.ndarray) -> None:
    """Print a log-spaced histogram of bout net displacement, to reveal a local/migratory trough."""
    v = net_km[net_km > 0]
    edges = np.logspace(np.log10(max(v.min(), 1e-3)), np.log10(v.max()), 25)
    counts, _ = np.histogram(v, bins=edges)
    peak = counts.max()
    print("\nbout net displacement (log-spaced histogram):")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        bar = "#" * int(40 * c / peak) if peak else ""
        print(f"  {lo:8.3f}-{hi:8.3f} km | {c:6d} {bar}")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(
        SEGMENTS,
        columns=["segment_id", "individual_id", "bin_index", "timestamp", "lat", "lon"],
    )
    lab = label_contiguous_regimes(frame)
    bouts = moving_bouts(lab)
    print(f"{len(bouts):,} moving bouts over {bouts['segment_id'].nunique():,} segments")

    # Distributions.
    print("\nnet_km percentiles:")
    for q in [10, 25, 50, 75, 90, 95, 99, 99.9]:
        print(f"  p{q:<5}: {np.nanpercentile(bouts['net_km'], q):10.3f} km")
    print("\nstraightness percentiles (net/path):")
    for q in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{q:<5}: {np.nanpercentile(bouts['straightness'], q):6.3f}")

    log_histogram(bouts["net_km"].to_numpy())

    # How rare is a 'migration launch' under candidate net-displacement cuts?
    print("\nfraction of bouts (= launches) above candidate migratory cuts:")
    for cut in [5, 10, 25, 50, 100, 200]:
        m = bouts["net_km"] >= cut
        print(f"  net >= {cut:>3} km : {m.sum():6,} bouts ({m.mean():6.2%})  "
              f"median straightness {bouts.loc[m, 'straightness'].median():.2f}, "
              f"median length {bouts.loc[m, 'n_steps'].median():.0f} fixes")

    # Figure: net displacement (log) + straightness. 
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    v = bouts["net_km"][bouts["net_km"] > 0]
    a.hist(np.log10(v), bins=60, color="#4C72B0")
    a.set_xlabel("log10 bout net displacement (km)"); a.set_ylabel("bouts")
    a.set_title("Moving-bout net displacement")
    b.hist(bouts["straightness"].dropna(), bins=60, color="#C44E52")
    b.set_xlabel("straightness (net / path)"); b.set_ylabel("bouts")
    b.set_title("Moving-bout straightness")
    fig.tight_layout()
    fig.savefig(RESULTS / "launch_scale.png", dpi=140); plt.close(fig)

    bouts.to_parquet(RESULTS / "moving_bouts.parquet", index=False)
    print(f"\nbouts -> {RESULTS / 'moving_bouts.parquet'}")
    print(f"plot  -> {RESULTS / 'launch_scale.png'}")


if __name__ == "__main__":
    main()
