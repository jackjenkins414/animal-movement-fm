# GenAI reference: I used GenAI for small parts of this; specifically I implemented the step distance calculation
# summary logic, docstrings and main analysis intent. GenAI helped build most of plot_hist(), corrected the 
# DuckDB query, and cleaned the terminal print formatting around the existing summaries. I verified GenAI's 
# output by checking that LAG only paired consecutive rows (which was my error GenAI fixed) within each 
# segment transitions, spot checking 2 sampled step_km values with a manual haversine calculation. 

"""
Step-displacement distribution for resampled stork segments.

Model-independent: reads stork_segments.parquet only. For every pair of consecutive stored bins WITHIN a segment 
(never across a segment boundary), computes the great-circle step distance in km, then characterises the 
distribution corpus-wide and per individual.

We aim to answer three diagnostic questions:
  - Two-regime structure: is the step-length distribution bimodal (near-stationary foraging lump + 
    directed migration shoulder)?
  - Biological ceiling: what is the largest 6h step a stork actually makes? (feeds the physical-plausibility 
    check on the p99 error tail, §2)
  - Normality: are step sizes normal / log-normal, or a two-regime mixture?

Steps are counted the way the harness windows the data: between consecutive STORED bins within a segment. `step_bins` 
records how many 6h bins each step spans (1 = a clean 6h step; >1 = a within-segment gap the resampler kept).
The clean-6h subset (step_bins == 1) is used for the biological ceiling.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"  # macOS OpenMP guard 

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from animal_movement_fm.eval.metrics import haversine_distance


REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"

STATIONARY_KM = 0.1  # steps below this (100 m) are treated as effectively stationary


def load_steps(segments_parquet: Path) -> pd.DataFrame:
    """
    One row per within-segment step, with endpoints and the bin-gap.

    LAG over (segment_id ORDER BY bin_index) pairs each bin with its predecessor; the first bin of each 
    segment has a null predecessor and is dropped, so no step ever crosses a segment boundary.
    """
    frame = duckdb.sql(f"""
        WITH ordered AS (
            SELECT
                segment_id,
                individual_id,
                bin_index,
                lon, lat,
                LAG(lon)       OVER w AS prev_lon,
                LAG(lat)       OVER w AS prev_lat,
                LAG(bin_index) OVER w AS prev_bin_index
            FROM '{segments_parquet}'
            WINDOW w AS (PARTITION BY segment_id ORDER BY bin_index)
        )
        SELECT
            segment_id,
            individual_id,
            bin_index - prev_bin_index AS step_bins,
            prev_lon, prev_lat, lon, lat
        FROM ordered
        WHERE prev_lon IS NOT NULL
        ORDER BY segment_id, bin_index
    """).df()
    return frame


def add_step_km(steps: pd.DataFrame) -> pd.DataFrame:
    """Great-circle step distance in km, via the package's canonical haversine."""
    metres = haversine_distance(
        steps["prev_lon"].to_numpy(),
        steps["prev_lat"].to_numpy(),
        steps["lon"].to_numpy(),
        steps["lat"].to_numpy(),
    )
    steps["step_km"] = metres / 1000.0
    return steps


def summarise(step_km: np.ndarray, label: str) -> dict:
    """Percentile summary in km (km-not-metres reporting convention)."""
    return {
        "subset": label,
        "n": int(step_km.size),
        "median_km": float(np.median(step_km)),
        "mean_km": float(np.mean(step_km)),
        "p90_km": float(np.percentile(step_km, 90)),
        "p95_km": float(np.percentile(step_km, 95)),
        "p99_km": float(np.percentile(step_km, 99)),
        "max_km": float(np.max(step_km)),
        "frac_stationary": float((step_km < STATIONARY_KM).mean()),
    }


def normality(step_km: np.ndarray) -> None:
    """Skew, kurtosis, D'Agostino-Pearson normality test on raw and log steps."""
    nonzero = step_km[step_km > 1e-3]
    log_km = np.log10(nonzero)

    for name, x in (("raw step_km", step_km), ("log10 step_km (>1 m)", log_km)):
        _, p = stats.normaltest(x)
        print(f"  {name:<22}  skew={stats.skew(x):+.2f}  "
              f"kurtosis={stats.kurtosis(x):+.2f}  normaltest p={p:.1e}")
    print("  (p far below 0.05 => not normal; a two-regime mixture is expected)")


def plot_hist(step_km: np.ndarray, out_path: Path) -> None:
    """Two-panel histogram: clipped linear km, and log10 km with regime cut."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Linear panel, clipped to the readable body (drops the impossible tail).
    LIN_CLIP_KM = 60.0
    body = step_km[step_km <= LIN_CLIP_KM]
    frac_shown = body.size / step_km.size
    ax_lin.hist(body, bins=200, color="#4C72B0")
    ax_lin.set_xlabel("6h step (km)")
    ax_lin.set_ylabel("count")
    ax_lin.set_title(f"Step length — linear (\u2264{LIN_CLIP_KM:.0f} km, "
                     f"{frac_shown:.1%} of steps)")
    ax_lin.set_xlim(0, LIN_CLIP_KM)

    # Log panel with the empirical foraging/moving regime cut marked.
    REGIME_CUT_KM = 0.06  # the trough between the two peaks (~60 m)
    nonzero = step_km[step_km > 1e-3]
    ax_log.hist(np.log10(nonzero), bins=200, color="#4C72B0")
    ax_log.axvline(np.log10(REGIME_CUT_KM), color="#C44E52",
                   linestyle="--", linewidth=1.5,
                   label=f"regime cut \u2248 {REGIME_CUT_KM*1000:.0f} m")
    ax_log.set_xlabel("log10 6h step (km)")
    ax_log.set_ylabel("count")
    ax_log.set_title("Step length — log (nonzero steps)")
    ax_log.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_individual(steps: pd.DataFrame) -> pd.DataFrame:
    """Per-bird step summary; feeds trajectory ranking (§1) and regime split (§6)."""
    g = steps.groupby("individual_id")["step_km"]
    out = pd.DataFrame({
        "n_steps": g.size(),
        "median_km": g.median(),
        "p99_km": g.quantile(0.99),
        "max_km": g.max(),
        "frac_stationary": g.apply(lambda s: (s < STATIONARY_KM).mean()),
    }).reset_index()
    return out.sort_values("median_km").reset_index(drop=True)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    steps = add_step_km(load_steps(SEGMENTS))
    print(f"loaded {len(steps):,} steps across "
          f"{steps['segment_id'].nunique():,} segments, "
          f"{steps['individual_id'].nunique():,} individuals\n")

    # clean 6h vs multi-bin (within-segment gaps the resampler kept)
    clean = steps[steps["step_bins"] == 1]
    print(f"step_bins == 1 (clean 6h): {len(clean):,} "
          f"({len(clean) / len(steps):.1%})  |  multi-bin steps: "
          f"{len(steps) - len(clean):,}\n")

    summaries = pd.DataFrame([
        summarise(steps["step_km"].to_numpy(), "all consecutive-bin steps"),
        summarise(clean["step_km"].to_numpy(), "clean 6h steps (step_bins==1)"),
    ])
    print(summaries.to_string(index=False))
    print()

    print("Normality (all consecutive-bin steps):")
    normality(steps["step_km"].to_numpy())
    print()

    plot_path = RESULTS / "step_displacement_hist.png"
    plot_hist(steps["step_km"].to_numpy(), plot_path)
    print(f"histogram -> {plot_path}")

    per_ind = per_individual(steps)
    per_ind_path = RESULTS / "step_per_individual.parquet"
    per_ind.to_parquet(per_ind_path, index=False)
    print(f"per-individual summary -> {per_ind_path}  ({len(per_ind)} birds)")

    summ_path = RESULTS / "step_displacement_summary.csv"
    summaries.to_csv(summ_path, index=False)
    print(f"summary -> {summ_path}")


if __name__ == "__main__":
    main()