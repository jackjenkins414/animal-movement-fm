# GenAI reference: I used a small amount of GenAI assistence here. I implemented percentile_ladder() and 
# plausibility_split() logic, along with the docstrings, comments imports etc. GenAI helped write the 
# main method, and the error distribution plotting function, and the final csv/plot export section and 
# lightly ammended the printed table output. I verified this by checking that preds_*.parquet row counts 
# matched the earlier prediction dumps, confirming the percentile ladder reproduced the headline 
# p50/p99 values, and manually checking implied_kmh as error_km / (step_bins * 6h) for 2 persistence
# rows before trusting the plausibility split.
"""
Step 3: error distribution, percentile ladder.

Pure diagnostics on the prediction dumps (preds_*.parquet). Nothing is written back to the pipeline; the biological 
speed ceiling is used only to TAG existing errors, not to filter data (we'll get to that step later).

We do the following here:
  3a  error distribution per baseline (linear-clipped + log), km
  3b  percentile ladder p50/p75/p90/p95/p99 in km; shows where the tail explodes, so maybe p99 -> p90 is our go to metric 
  3c  Level-A plausibility split of the persistence error tail:
      implied 6h-step ground speed vs a 100 km/h ceiling (90/108 sensitivity). Persistence error == true step, so an 
      error's implied speed is the speed the bird would have needed. > ceiling => biologically impossible => so the "error"
      is a data artefact, not a model miss.

Speed uses each step's REAL duration (step_bins * 6h), joined from segments, so multi-bin gap steps aren't wrongly flagged.

note: called step 3 just because we are following steps outlined in my obsidian notes 
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"

BASELINES = ["persistence", "avg_k1", "avg_k3"]
BIN_HOURS = 6.0
CEILINGS_KMH = {"conservative_90": 90.0, "default_100": 100.0, "permissive_108": 108.0}
LIN_CLIP_KM = 60.0  # readable body for the linear panel; we got this from step 1 


def load_preds(name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / f"preds_{name}.parquet")


def step_bins_lookup() -> pd.DataFrame:
    """
    Duration (in 6h bins) of each prediction's step, keyed to match the dumps.

    The dump ordered predictions within each segment in bin order, and each prediction's target is a bin; 
    step_bins = this bin_index; previous bin_index within the segment. We recompute it from segments and 
    join on (segment_id, target bin) via timestamp, which the dump carries.
    """
    return duckdb.sql(f"""
        WITH ordered AS (
            SELECT segment_id, timestamp, bin_index,
                   bin_index - LAG(bin_index) OVER w AS step_bins
            FROM '{SEGMENTS}'
            WINDOW w AS (PARTITION BY segment_id ORDER BY bin_index)
        )
        SELECT segment_id, timestamp, step_bins
        FROM ordered
        WHERE step_bins IS NOT NULL
    """).df()


def percentile_ladder(preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in preds.items():
        e = df["error_km"].to_numpy()
        rows.append({
            "model": name,
            "n": e.size,
            "p50_km": np.percentile(e, 50),
            "p75_km": np.percentile(e, 75),
            "p90_km": np.percentile(e, 90),
            "p95_km": np.percentile(e, 95),
            "p99_km": np.percentile(e, 99),
            "p90_to_p99_jump": np.percentile(e, 99) / np.percentile(e, 90),
        })
    return pd.DataFrame(rows)


def plausibility_split(persist: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each persistence error as plausible-migration vs impossible-artefact.

    implied_kmh = error_km / (step_bins * 6h). error_km == true step for the tail cases we care about; 
    comparing to the ceiling asks 'could the bird have moved this far in the available time?'.
    """
    dur_h = persist["step_bins"].to_numpy() * BIN_HOURS
    implied_kmh = np.where(dur_h > 0, persist["error_km"].to_numpy() / dur_h, np.nan)
    persist = persist.assign(implied_kmh=implied_kmh)

    rows = []
    n = np.isfinite(implied_kmh).sum()
    for label, ceil in CEILINGS_KMH.items():
        impossible = np.nansum(implied_kmh > ceil)
        # error mass above vs below the ceiling
        above_mass = persist.loc[implied_kmh > ceil, "error_km"].sum()
        total_mass = persist["error_km"].sum()
        rows.append({
            "ceiling": label,
            "ceiling_kmh": ceil,
            "n_impossible": int(impossible),
            "frac_impossible": impossible / n,
            "error_mass_above_ceiling_frac": above_mass / total_mass,
        })
    return pd.DataFrame(rows)


def plot_error_dist(preds: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5))
    colours = {"persistence": "#4C72B0", "avg_k1": "#DD8452", "avg_k3": "#55A868"}

    for name, df in preds.items():
        e = df["error_km"].to_numpy()
        body = e[e <= LIN_CLIP_KM]
        ax_lin.hist(body, bins=200, histtype="step", linewidth=1.4,
                    color=colours.get(name), label=name)
        nz = e[e > 1e-3]
        ax_log.hist(np.log10(nz), bins=200, histtype="step", linewidth=1.4,
                    color=colours.get(name), label=name)

    ax_lin.set_xlabel("error (km)"); ax_lin.set_ylabel("count")
    ax_lin.set_title(f"Error distribution — linear (\u2264{LIN_CLIP_KM:.0f} km)")
    ax_lin.set_xlim(0, LIN_CLIP_KM); ax_lin.legend()

    ax_log.set_xlabel("log10 error (km)"); ax_log.set_ylabel("count")
    ax_log.set_title("Error distribution — log")
    ax_log.legend()

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    preds = {name: load_preds(name) for name in BASELINES}
    for name, df in preds.items():
        print(f"{name:<12} {len(df):>8,} preds")
    print()

    # 3b percentile ladder
    ladder = percentile_ladder(preds)
    print("Percentile ladder (km):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
        print(ladder.to_string(index=False))
    print()

    # 3c plausibility split (persistence only; error == true step there)
    persist = preds["persistence"].merge(
        step_bins_lookup(), on=["segment_id", "timestamp"], how="left"
    )
    missing = persist["step_bins"].isna().sum()
    if missing:
        print(f"note: {missing:,} preds had no step_bins match (first-in-segment); "
              f"excluded from plausibility.\n")
    split = plausibility_split(persist.dropna(subset=["step_bins"]))
    print("Level-A plausibility split of persistence error (implied 6h-step speed):")
    with pd.option_context("display.float_format", lambda v: f"{v:,.5f}"):
        print(split.to_string(index=False))
    print()

    # 3a distribution plot
    plot_path = RESULTS / "step3_error_dist.png"
    plot_error_dist(preds, plot_path)
    ladder.to_csv(RESULTS / "step3_percentile_ladder.csv", index=False)
    split.to_csv(RESULTS / "step3_plausibility_split.csv", index=False)
    print(f"distribution -> {plot_path}")
    print(f"ladder       -> {RESULTS / 'step3_percentile_ladder.csv'}")
    print(f"split        -> {RESULTS / 'step3_plausibility_split.csv'}")


if __name__ == "__main__":
    main()