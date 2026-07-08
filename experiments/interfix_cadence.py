# GenAI reference: Here I wrote the diagnostic framing, imports/constants, fmt_duration(), and cadence_support(); and I used GenAI 
# to mainly aid with the DuckDB SQL for build_delta_table()/summary queries, the main() orchestration/output/export flow, and plot_hist(). 
# I verfied its output by checking that LAG intervals were computed only within (individual_id, deployment_id), confirming duplicate
# or non-positive intervals were counted then excluded from summaries and inspecting the histogram to ensure it looked plausible. 
"""
Inter-fix cadence characterisation for the stork survivor corpus.

Model-independent; no resample, no rebuild. Reads stork_fixes.parquet (the staged, stork-only survivor raw fixes) and measures 
the native sampling interval; the time between consecutive raw fixes WITHIN a trajectory (individual_id, deployment_id), never 
across a deployment boundary.

Note; it does not pick the best cadence; that is the sweep's job, via true bin occupancy. Support here is an optimistic proxy: 
a duty cycled tag samples finely in bursts yet still leaves fine-cadence bins empty.
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
STORK_FIXES = REPO / "data" / "processed" / "stork_fixes.parquet"
RESULTS = REPO / "experiments" / "results"

# cadences we might sweep; the current build cadence (6h) is included for reference
CANDIDATE_CADENCES_S = {
    "30min": 1800,
    "1h": 3600,
    "3h": 10800,
    "6h": 21600,
    "12h": 43200,
    "24h": 86400,
}

# first run; a fast sample of birds. Can set to None to run the full survivor corpus
SAMPLE_INDIVIDUALS: int | None = None


def fmt_duration(seconds: float) -> str:
    """Human-readable duration across the s->months range the intervals span."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"


def build_delta_table(
    con: duckdb.DuckDBPyConnection,
    fixes_parquet: Path,
    sample_individuals: int | None,
) -> tuple[int, int, int]:
    """Materialise consecutive-fix intervals into a temp table `deltas` 

    LAG over (individual_id, deployment_id ORDER BY timestamp) gives each fix its predecessor's time; the first fix of each 
    trajectory has a null predecessor and is dropped (correctly not a cadence). Zero/negative deltas (duplicate or reordered 
    timestamps) are kept here and filtered downstream so they can be counted. Returns (n_pairs, n_nonpositive, n_trajectories).
    """
    sample_filter = ""
    if sample_individuals:
        sample_filter = f"""
            WHERE individual_id IN (
                SELECT individual_id
                FROM (SELECT DISTINCT individual_id FROM '{fixes_parquet}')
                ORDER BY hash(individual_id)
                LIMIT {sample_individuals}
            )
        """

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE deltas AS
        WITH src AS (
            SELECT individual_id, deployment_id, timestamp
            FROM '{fixes_parquet}'
            {sample_filter}
        ),
        d AS (
            SELECT
                individual_id, deployment_id,
                (epoch_us(timestamp) - epoch_us(LAG(timestamp) OVER w)) / 1e6 AS delta_s
            FROM src
            WINDOW w AS (PARTITION BY individual_id, deployment_id ORDER BY timestamp)
        )
        SELECT individual_id, deployment_id, delta_s
        FROM d
        WHERE delta_s IS NOT NULL
    """)

    n_pairs = con.execute("SELECT count(*) FROM deltas").fetchone()[0]
    n_nonpos = con.execute("SELECT count(*) FROM deltas WHERE delta_s <= 0").fetchone()[0]
    n_traj = con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT individual_id, deployment_id FROM deltas)"
    ).fetchone()[0]
    return n_pairs, n_nonpos, n_traj


def pooled_summary(con: duckdb.DuckDBPyConnection) -> pd.Series:
    """Corpus-wide interval percentile ladder, in seconds (all positive deltas)."""
    return con.execute("""
        SELECT
            min(delta_s)                 AS min_s,
            median(delta_s)              AS median_s,
            avg(delta_s)                 AS mean_s,
            quantile_cont(delta_s, 0.90) AS p90_s,
            quantile_cont(delta_s, 0.99) AS p99_s,
            max(delta_s)                 AS max_s
        FROM deltas WHERE delta_s > 0
    """).df().iloc[0]


def per_trajectory_cadence(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One native-cadence summary per trajectory; the spread is the tag-program mixture."""
    return con.execute("""
        SELECT
            individual_id, deployment_id,
            count(*)                     AS n_deltas,
            median(delta_s)              AS median_s,
            quantile_cont(delta_s, 0.90) AS p90_s
        FROM deltas WHERE delta_s > 0
        GROUP BY individual_id, deployment_id
        ORDER BY median_s
    """).df()


def cadence_support(
    con: duckdb.DuckDBPyConnection,
    per_traj: pd.DataFrame,
    candidates: dict[str, int],
) -> pd.DataFrame:
    """For each candidate C: fraction of fixes and of trajectories natively at least as fine as C.

    frac_fixes_le is data volume weighted (pulled toward the dominant high freq study); frac_traj_le is animal weighted (each bird counts once). 
    Both are optimistic proxies for honest support; true occupancy comes from an actual resample at C.
    """
    frac_exprs = ",\n            ".join(
        f'avg(CASE WHEN delta_s <= {sec} THEN 1.0 ELSE 0.0 END) AS "{name}"'
        for name, sec in candidates.items()
    )
    fix_frac = con.execute(
        f"SELECT {frac_exprs} FROM deltas WHERE delta_s > 0"
    ).df().iloc[0]

    rows = [
        {
            "cadence": name,
            "seconds": sec,
            "frac_fixes_le": float(fix_frac[name]),
            "frac_traj_le": float((per_traj["median_s"] <= sec).mean()),
        }
        for name, sec in candidates.items()
    ]
    return pd.DataFrame(rows)


def log_histogram(con: duckdb.DuckDBPyConnection, bin_width: float = 0.1) -> pd.DataFrame:
    """Binned counts over log10(interval seconds), computed DuckDB-side for plotting."""
    return con.execute(f"""
        SELECT
            floor(log10(delta_s) / {bin_width}) * {bin_width} AS log_s,
            count(*) AS n
        FROM deltas WHERE delta_s > 0
        GROUP BY 1 ORDER BY 1
    """).df()


def plot_hist(
    hist: pd.DataFrame,
    candidates: dict[str, int],
    out_path: Path,
    bin_width: float = 0.1,
) -> None:
    """Log-axis histogram of native cadence, with candidate sweep cadences marked."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(hist["log_s"], hist["n"], width=bin_width, align="edge", color="#4C72B0")

    top = ax.get_ylim()[1]
    for name, sec in candidates.items():
        ax.axvline(np.log10(sec), color="#C44E52", linestyle="--", linewidth=1.2)
        ax.text(np.log10(sec), top * 0.97, name, rotation=90,
                va="top", ha="right", fontsize=8, color="#C44E52")

    refs = {"1 s": 1, "1 min": 60, "1 h": 3600, "6 h": 21600,
            "1 d": 86400, "7 d": 604800, "30 d": 2592000}
    ax.set_xticks([np.log10(s) for s in refs.values()])
    ax.set_xticklabels(list(refs.keys()))
    ax.set_xlabel("inter-fix interval (log scale)")
    ax.set_ylabel("count")
    ax.set_title("Native inter-fix cadence \u2014 stork survivor corpus")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    n_pairs, n_nonpos, n_traj = build_delta_table(con, STORK_FIXES, SAMPLE_INDIVIDUALS)
    scope = f"SAMPLE of {SAMPLE_INDIVIDUALS} individuals" if SAMPLE_INDIVIDUALS else "FULL corpus"
    print(f"[{scope}]  {n_traj:,} trajectories, {n_pairs:,} consecutive-fix pairs "
          f"({n_nonpos:,} duplicate/zero-\u0394t dropped)\n")

    pooled = pooled_summary(con)
    print("Pooled inter-fix interval:")
    for key in ("min_s", "median_s", "mean_s", "p90_s", "p99_s", "max_s"):
        print(f"  {key[:-2]:<7} {fmt_duration(pooled[key])}")
    print()

    per_traj = per_trajectory_cadence(con)
    med = per_traj["median_s"]
    print("Per-trajectory native cadence (median interval per trajectory):")
    print(f"  finest    {fmt_duration(med.min())}")
    print(f"  p25       {fmt_duration(med.quantile(0.25))}")
    print(f"  median    {fmt_duration(med.median())}")
    print(f"  p75       {fmt_duration(med.quantile(0.75))}")
    print(f"  coarsest  {fmt_duration(med.max())}")
    print(f"  -> {len(per_traj):,} trajectories; the spread is the tag-program mixture\n")

    support = cadence_support(con, per_traj, CANDIDATE_CADENCES_S)
    print("Candidate-cadence support (optimistic proxy; true occupancy comes from the sweep):")
    print(support.to_string(index=False, formatters={
        "frac_fixes_le": "{:.1%}".format,
        "frac_traj_le": "{:.1%}".format,
    }))
    print()

    hist = log_histogram(con)
    plot_path = RESULTS / "interfix_cadence_hist.png"
    plot_hist(hist, CANDIDATE_CADENCES_S, plot_path)
    print(f"histogram -> {plot_path}")

    per_traj_path = RESULTS / "interfix_per_trajectory.parquet"
    per_traj.to_parquet(per_traj_path, index=False)
    print(f"per-trajectory cadence -> {per_traj_path}  ({len(per_traj)} trajectories)")

    support_path = RESULTS / "interfix_cadence_support.csv"
    support.to_csv(support_path, index=False)
    print(f"support table -> {support_path}")


if __name__ == "__main__":
    main()