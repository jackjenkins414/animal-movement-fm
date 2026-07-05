# GenAI reference: I did all of this; GenAI was used only for minor terminal print formatting edits and slightly tweaked my plot in the main method.
"""
Step 4b supplement (of notes hence the naming): hour-of-day activity rhythm.

Confirms the lag4 / lag8 spikes in the step-magnitude autocorrelation are a 24h diurnal cycle (since 
4 six-hour bins = one day), and characterises the rhythm directly: median step magnitude by hour-of-day bin. This is (a) the
de-confusing figure for the 4a sawtooth, and (b) the baseline signal a time-of-day feature would hand the model.

Uses the bin timestamp's UTC hour. Because bins are anchored to the 6h grid (00/06/12/18 UTC), hour-of-day collapses to 
four slots: 0, 6, 12, 18.

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
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"
REGIME_CUT_KM = 0.06


def load_steps_with_hour() -> pd.DataFrame:
    """
    Per within-segment step: step length (km) and the UTC hour-of-day of the
    step's ENDPOINT bin (the target of that transition).
    """
    return duckdb.sql(f"""
        WITH ordered AS (
            SELECT segment_id, bin_index, timestamp, lon, lat,
                   LAG(lon) OVER w AS plon,
                   LAG(lat) OVER w AS plat
            FROM '{SEGMENTS}'
            WINDOW w AS (PARTITION BY segment_id ORDER BY bin_index)
        )
        SELECT plon, plat, lon, lat,
               CAST(hour(timestamp) AS INTEGER) AS hour_utc
        FROM ordered
        WHERE plon IS NOT NULL
    """).df()


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_steps_with_hour()
    df["step_km"] = haversine_distance(
        df["plon"].to_numpy(), df["plat"].to_numpy(),
        df["lon"].to_numpy(), df["lat"].to_numpy()) / 1000.0

    print(f"{len(df):,} steps\n")
    print("step magnitude by UTC hour-of-day bin:")
    g = df.groupby("hour_utc")["step_km"]
    tbl = pd.DataFrame({
        "n": g.size(),
        "median_km": g.median(),
        "mean_km": g.mean(),
        "frac_stationary": g.apply(lambda s: (s < REGIME_CUT_KM).mean()),
    })
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(tbl.to_string())

    # amplitude of the rhythm: ratio of busiest to quietest slot (median)
    med = tbl["median_km"]
    print(f"\ndiurnal amplitude (max/min median): {med.max() / med.max()*0 + med.max()/med.min():.1f}x")
    print(f"  busiest slot: {int(med.idxmax()):02d}:00 UTC ({med.max():.3f} km)")
    print(f"  quietest slot: {int(med.idxmin()):02d}:00 UTC ({med.min():.3f} km)")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    hours = tbl.index.to_numpy()
    ax.plot(hours, tbl["median_km"], "o-", color="#4C72B0", label="median step")
    ax.plot(hours, tbl["mean_km"], "s--", color="#C44E52", label="mean step")
    ax.set_xlabel("UTC hour-of-day bin"); ax.set_ylabel("step length (km)")
    ax.set_xticks(hours)
    ax.set_title("Diurnal activity rhythm\n(confirms the lag-4 autocorrelation spike = 24h cycle)")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "step4_hourofday.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nplot -> {out}")


if __name__ == "__main__":
    main()