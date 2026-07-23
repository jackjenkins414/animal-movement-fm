"""
Aggregate the 30 run multi-seed sweep into the regime decomposed error with seed spread.

This is the analysis half of the decoupled inference/analysis split: the Gadi array job wrote one
predictions parquet + summary JSON per run (results/multiseed/); this script reads them and produces
the headline table as MEAN +/- SD across the 5 training seeds.  
So now our prev small effects that were single seed results from before (raw attention vs recurrence; 
whether time/season move the launch tail) now can have with error bars.

What it does
------------
1. For each run's test-split predictions, recompute the true step the bird made (haversine between
   consecutive target positions within a segment) and tag each prediction's regime with the canonical
   labeller (analysis.regime.tag_regime, step_col="true_step_km"); identical to our Step 6 / time 2x2
   decomposition, so these numbers are directly comparable our prev single seed ones.
2. Per run, compute per regime error (median / p90 / p99 in km) plus an aggregate over all regimes.
3. Aggregate across the 5 seeds within each (architecture, feature cell): mean and sd.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from animal_movement_fm.analysis.regime import tag_regime
from animal_movement_fm.eval.metrics import haversine_distance

REPO = Path(__file__).resolve().parents[2]
MULTI = REPO / "results" / "multiseed"

REGIMES = ["foraging", "moving", "launch", "settle"]
CELL_ORDER = ["raw", "time", "time_season"]
ARCH_ORDER = ["lstm", "transformer"]
METRICS = ["median_km", "p90_km", "p99_km"]


def add_regime(preds: pd.DataFrame) -> pd.DataFrame:
    """Recompute true_step_km within each segment and tag the 4-way regime, matching Step 6."""
    df = preds.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    prev_lon = df.groupby("segment_id")["target_lon"].shift(1)
    prev_lat = df.groupby("segment_id")["target_lat"].shift(1)
    has_prev = prev_lon.notna()

    step_km = np.full(len(df), np.nan)
    step_km[has_prev.to_numpy()] = haversine_distance(
        prev_lon[has_prev].to_numpy(), prev_lat[has_prev].to_numpy(),
        df.loc[has_prev, "target_lon"].to_numpy(), df.loc[has_prev, "target_lat"].to_numpy(),
    ) / 1000.0
    df["true_step_km"] = step_km
    df["error_km"] = df["error_m"] / 1000.0
    return tag_regime(df, group_col="segment_id", step_col="true_step_km")

# GenAI reference: I wrote the analysis framing, imports/constants, and add_regime() logic above, then gave the partial script 
# to GenAI to finish. I asked for it comlete the regime summary, run loading, aggregation, printed pivots, and plotting; all of 
# which I could have done but I wanted to speed up the process with scripts like this. I had to fix a file naming bug in the 
# generated code so each summary_*.json uses meta["run_id"] to load the matching preds_{run_id}.parquet file, rather than 
# constructing an incorrect name from the summary filename. I also verified the output by checking the recomputed aggregate
# medians against each runs summary JSON and p99 plot matched the printed tables. All was fine. 
def regime_stats(df: pd.DataFrame) -> list[dict]:
    """Per-regime and aggregate error stats (km) for one run's tagged predictions."""
    rows = []
    for reg in REGIMES:
        e = df.loc[df["regime"] == reg, "error_km"].to_numpy()
        if e.size:
            rows.append({"regime": reg, "n": int(e.size), "median_km": float(np.median(e)),
                         "p90_km": float(np.percentile(e, 90)), "p99_km": float(np.percentile(e, 99))})
    e = df.loc[df["regime"] != "unknown", "error_km"].to_numpy()  # Aggregate excludes segment-start.
    rows.append({"regime": "all", "n": int(e.size), "median_km": float(np.median(e)),
                 "p90_km": float(np.percentile(e, 90)), "p99_km": float(np.percentile(e, 99))})
    return rows


def load_runs(split: str = "test") -> pd.DataFrame:
    """One row per (run, regime): [arch, cell, seed, regime, n, median_km, p90_km, p99_km]."""
    summaries = sorted(MULTI.glob("summary_*.json"))
    if not summaries:
        raise SystemExit(f"no summary_*.json in {MULTI}; run the sweep first (or rsync it here)")

    rows = []
    for spath in summaries:
        meta = json.loads(spath.read_text())
        run_id = meta["run_id"]
        preds = pd.read_parquet(MULTI / f"preds_{run_id}.parquet")
        preds = preds[preds["split"] == split]
        tagged = add_regime(preds)

        # Coherence check: recomputed aggregate median must match the run's own summary. 
        agg_med = float(np.median(tagged["error_km"]))
        reported = meta[split]["median_m"] / 1000.0
        if abs(agg_med - reported) > 1e-3:
            print(f"  WARN {run_id}: recomputed median {agg_med:.4f} km != summary {reported:.4f} km")

        for r in regime_stats(tagged):
            rows.append({"arch": meta["arch"], "cell": meta["cell"], "seed": meta["seed"], **r})
    return pd.DataFrame(rows)


def summarise(by_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and sd of each metric across the 5 seeds, per (arch, cell, regime)."""
    g = by_run.groupby(["arch", "cell", "regime"])
    out = g[METRICS].agg(["mean", "std"]).reset_index()
    out.columns = ["arch", "cell", "regime"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    out["n_seeds"] = g.size().to_numpy()
    return out


def cell_key(cell: str) -> int:
    return CELL_ORDER.index(cell) if cell in CELL_ORDER else len(CELL_ORDER)


def fmt(mean: float, std: float) -> str:
    return f"{mean:6.1f} +/- {std:4.1f}"


def print_pivot(summ: pd.DataFrame, regime: str, metric: str, title: str) -> None:
    """arch x cell table of 'mean +/- sd' for one regime/metric."""
    print(f"\n{title}  [{regime} {metric}, km, test]")
    sub = summ[summ["regime"] == regime]
    header = "  " + "cell".ljust(14) + "".join(a.ljust(18) for a in ARCH_ORDER)
    print(header)
    for cell in CELL_ORDER:
        line = "  " + cell.ljust(14)
        for arch in ARCH_ORDER:
            row = sub[(sub["cell"] == cell) & (sub["arch"] == arch)]
            if len(row):
                line += fmt(row[f"{metric}_mean"].iloc[0], row[f"{metric}_std"].iloc[0]).ljust(18)
            else:
                line += "-".ljust(18)
        print(line)


def main() -> None:
    by_run = load_runs(split="test")
    by_run = by_run.sort_values(["cell", "arch", "seed"], key=lambda s: s.map(cell_key) if s.name == "cell" else s)
    by_run.to_csv(MULTI / "regime_by_run.csv", index=False)

    summ = summarise(by_run)
    summ = summ.sort_values(["regime", "cell", "arch"], key=lambda s: s.map(cell_key) if s.name == "cell" else s)
    summ.to_csv(MULTI / "regime_summary.csv", index=False)

    print(f"loaded {by_run['seed'].nunique()} seeds x {by_run[['arch','cell']].drop_duplicates().shape[0]} "
          f"(arch,cell) cells = {by_run.groupby(['arch','cell']).ngroups * by_run['seed'].nunique()} runs")

    # The three questions this sweep exists to answer:
    print_pivot(summ, "moving", "p99_km", "Q1/2  moving-tail p99 (attention vs recurrence x covariate)")
    print_pivot(summ, "launch", "p99_km", "Q3    launch-onset p99 (does any covariate move the hard tail?)")
    print_pivot(summ, "all", "median_km", "        aggregate median")

    # Attention vs recurrence on the raw cell, with overlap check.
    print("\nattention vs recurrence, RAW cell (moving p99, km):")
    raw = summ[(summ.cell == "raw") & (summ.regime == "moving")].set_index("arch")
    if {"lstm", "transformer"} <= set(raw.index):
        l, t = raw.loc["lstm"], raw.loc["transformer"]
        gap = t["p99_km_mean"] - l["p99_km_mean"]
        pooled = np.hypot(l["p99_km_std"], t["p99_km_std"])
        verdict = "within seed noise" if abs(gap) <= pooled else "separated beyond 1 sd"
        print(f"  lstm {fmt(l['p99_km_mean'], l['p99_km_std'])}   "
              f"transformer {fmt(t['p99_km_mean'], t['p99_km_std'])}   "
              f"gap {gap:+.1f} km ({verdict})")

    # Season effect: time_season vs time, per arch, on launch p99.
    print("\nseason effect (time_season - time), launch p99, km:")
    for arch in ARCH_ORDER:
        s = summ[(summ.arch == arch) & (summ.regime == "launch")].set_index("cell")
        if {"time", "time_season"} <= set(s.index):
            d = s.loc["time_season", "p99_km_mean"] - s.loc["time", "p99_km_mean"]
            pooled = np.hypot(s.loc["time", "p99_km_std"], s.loc["time_season", "p99_km_std"])
            tag = "within noise" if abs(d) <= pooled else "beyond 1 sd"
            print(f"  {arch:<12} {d:+6.1f} km  (pooled sd {pooled:.1f}; {tag})")

    _plot(summ)
    print(f"\nby-run   -> {MULTI / 'regime_by_run.csv'}")
    print(f"summary  -> {MULTI / 'regime_summary.csv'}")


def _plot(summ: pd.DataFrame) -> None:
    """Grouped bars with seed-sd error bars: moving p99 and launch p99, by cell x arch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    x = np.arange(len(CELL_ORDER))
    w = 0.38
    colours = {"lstm": "#DA8BC3", "transformer": "#C44E52"}
    for ax, regime in zip(axes, ["moving", "launch"]):
        sub = summ[summ["regime"] == regime]
        for i, arch in enumerate(ARCH_ORDER):
            means = [sub[(sub.cell == c) & (sub.arch == arch)]["p99_km_mean"].pipe(
                lambda s: s.iloc[0] if len(s) else np.nan) for c in CELL_ORDER]
            errs = [sub[(sub.cell == c) & (sub.arch == arch)]["p99_km_std"].pipe(
                lambda s: s.iloc[0] if len(s) else np.nan) for c in CELL_ORDER]
            ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=4,
                   label=arch, color=colours[arch])
        ax.set_xticks(x); ax.set_xticklabels(CELL_ORDER)
        ax.set_ylabel("p99 error (km)"); ax.set_title(f"{regime} p99 (mean +/- sd, 5 seeds)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(MULTI / "multiseed_regime_p99.png", dpi=140)
    plt.close(fig)
    print(f"plot     -> {MULTI / 'multiseed_regime_p99.png'}")


if __name__ == "__main__":
    main()
