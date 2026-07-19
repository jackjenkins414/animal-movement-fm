# GenAI reference: Here i wrote the diagnostic docstring, function structure/type signatures, and initial implementations, then 
# used GenAI to finish the script (asked it to tighten the control flow, and add clear terminal print summaries). I verified it
# by checking the state counts against label_regimes(), confirming split assignment by individual_id, and a quick spot check of 
# launch/settle counts a sample of individual segments.
"""
State labelling: composition + transition structure (Part B, validation gate).

Before building a state classifier we characterise the labels the classifier will have to predict. This script labels every fix 
in the segment product with `analysis.regime.label_regimes` and reports:

  1. Class composition (foraging / moving / launch / settle / unknown) over the whole corpus and broken down by the modelling 
     split (train / val / test; 'dropped' = birds below min_windows).
  2. The majority class baseline accuracy per split; and thus the floor any classifier must beat
  3. Per-segment transition structure; launches and settles per segment, and how rare launch is

Launch is the class that matters (migration onset, the error tail) and is expected to be rare, so we report it explicitly: a 
classifier can score high accuracy by ignoring launch entirely, which is why we look at per class support here, and not just 
the aggregate.

Note: 'unknown' is the first fix of each segment (no incoming step); it is not a real class and is excluded from the base rate/accuracy 
figures. Labels come straight from our canonical labeller, so they match the Step 6 decomposition (tho validated separately)
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.analysis.regime import label_regimes, REGIMES
from animal_movement_fm.data.split import make_split, window_counts_per_individual

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "results"

STATES = list(REGIMES) + ["unknown"]  # display order


def split_of(split, individual_ids: np.ndarray) -> dict[int, str]:
    """Map each individual_id to 'train'/'val'/'test', or 'dropped' if below min_windows."""
    lookup = {int(i): "train" for i in split.train}
    lookup.update({int(i): "val" for i in split.val})
    lookup.update({int(i): "test" for i in split.test})
    return {int(i): lookup.get(int(i), "dropped") for i in individual_ids}


def composition_table(lab: pd.DataFrame) -> pd.DataFrame:
    """Counts of each state per split (rows = state, cols = split), plus a 'full' column."""
    tab = (
        lab.groupby(["split", "regime"]).size().unstack("regime", fill_value=0).T
        .reindex(STATES, fill_value=0)
    )
    tab["full"] = tab.sum(axis=1)
    cols = [c for c in ["full", "train", "val", "test", "dropped"] if c in tab.columns]
    return tab[cols]


def majority_baseline(lab: pd.DataFrame) -> pd.DataFrame:
    """Per-split majority-class accuracy over the real (non-unknown) classes."""
    real = lab[lab["regime"] != "unknown"]
    rows = []
    for split_name, g in real.groupby("split"):
        shares = g["regime"].value_counts(normalize=True)
        rows.append({
            "split": split_name,
            "n_labelled": len(g),
            "majority_class": shares.index[0],
            "majority_acc": shares.iloc[0],
            "launch_frac": shares.get("launch", 0.0),
        })
    return pd.DataFrame(rows).set_index("split")


def transition_structure(lab: pd.DataFrame) -> pd.DataFrame:
    """Per-segment launch/settle counts and the resulting per-segment summary."""
    per_seg = (
        lab.assign(is_launch=lab["regime"].eq("launch"), is_settle=lab["regime"].eq("settle"))
        .groupby("segment_id")
        .agg(n_fix=("regime", "size"),
             n_launch=("is_launch", "sum"),
             n_settle=("is_settle", "sum"))
    )
    return per_seg


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(
        SEGMENTS, columns=["segment_id", "individual_id", "timestamp", "lat", "lon"]
    )
    print(f"loaded {len(frame):,} fixes / {frame['segment_id'].nunique():,} segments / "
          f"{frame['individual_id'].nunique():,} individuals")

    lab = label_regimes(frame)

    # tag each fix with its modelling split (single labelling pass, then map by bird)
    counts = window_counts_per_individual(SEGMENTS)
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    id_split = split_of(split, frame["individual_id"].unique())
    lab["split"] = lab["individual_id"].map(id_split)

    # 1. composition
    comp = composition_table(lab)
    print("\nstate composition (fix counts):")
    print(comp.to_string())
    print("\nstate composition (% of split's fixes):")
    pct = comp.div(comp.sum(axis=0), axis=1) * 100.0
    print(pct.to_string(float_format=lambda v: f"{v:5.2f}"))

    # 2. majority baseline
    base = majority_baseline(lab)
    print("\nmajority-class baseline (over real classes, unknown excluded):")
    print(base.to_string(float_format=lambda v: f"{v:.4f}"))

    # 3. transition structure
    per_seg = transition_structure(lab)
    modelling_segs = lab.loc[lab["split"].isin(["train", "val", "test"]), "segment_id"].unique()
    ps = per_seg.loc[per_seg.index.isin(modelling_segs)]
    print(f"\nper-segment transition structure (modelling splits, {len(ps):,} segments):")
    print(f"  launches / segment : mean {ps['n_launch'].mean():.2f}, "
          f"median {ps['n_launch'].median():.0f}, max {ps['n_launch'].max():.0f}")
    print(f"  settles  / segment : mean {ps['n_settle'].mean():.2f}, "
          f"median {ps['n_settle'].median():.0f}, max {ps['n_settle'].max():.0f}")
    print(f"  segments with >=1 launch: {(ps['n_launch'] >= 1).mean():.1%}")
    print(f"  total launches: {int(ps['n_launch'].sum()):,}  "
          f"total settles: {int(ps['n_settle'].sum()):,}")

    # persist the composition for the write-up / downstream reference
    out = RESULTS / "state_composition.csv"
    comp.to_csv(out)
    print(f"\ncomposition table -> {out}")


if __name__ == "__main__":
    main()
