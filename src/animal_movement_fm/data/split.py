"""
Assign stork individuals to train/val/test splits, by individual (no leakage).

Splitting is at the individual level: all of one animal's windows go entirely to one split. Windows from 
the same bird are heavily autocorrelated (adjacent windows share history), so a window-level split would 
leak near-duplicates across train and test and measure memorisation, not generalisation.

The split is stratified by a key (taxon by default) so the class balance is preserved across splits; 
trivial within single-species stork, but load-bearing at Tier 2 where whole species are held out. 
Reproducible via a fixed seed.
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass


def window_counts_per_individual(
    segments_parquet,
    context_length: int = 24,
) -> pd.DataFrame:
    """Per-individual window counts, to set the minimum-track threshold before splitting.

    Counts windows exactly as the benchmark will: the harness windower runs per
    segment (so windows never cross a transmission gap), and counts are summed per
    individual across its segments. Returns one row per individual_id, sorted ascending
    so the short-track tail is visible.
    """
    import duckdb

    from animal_movement_fm.eval.harness import iter_windows

    con = duckdb.connect()
    seg_df = con.execute(f"""
        SELECT segment_id, individual_id, timestamp, lat, lon
        FROM '{segments_parquet}'
        ORDER BY segment_id, timestamp
    """).df()

    counts: dict[int, int] = {}
    for _, g in seg_df.groupby("segment_id", sort=False):
        individual_id = int(g["individual_id"].iloc[0])
        n_windows = len(iter_windows(g, context_length=context_length))
        counts[individual_id] = counts.get(individual_id, 0) + n_windows

    df = (
        pd.DataFrame({"individual_id": list(counts), "n_windows": list(counts.values())})
        .sort_values("n_windows")
        .reset_index(drop=True)
    )
    return df

@dataclass(frozen=True)
class Split:
    """Train/val/test assignment at the individual level.

    Each array holds individual_ids. A window belongs to the split its bird is in; no individual appears in more
    than one array (no leakage by construction).
    """

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(
    counts: pd.DataFrame,
    strata: pd.DataFrame | None = None,
    min_windows: int = 100,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    seed: int = 42,
) -> Split:
    """Split individuals into train/val/test, stratified by key, reproducibly.

    `counts` is the per-individual window table from window_counts_per_individual. `strata` maps individual_id -> a 
    stratification key (e.g. taxon); if None, all individuals share one stratum (correct for single-species stork). 
    Individuals with fewer than `min_windows windows are dropped before splitting.
    """
    eligible = counts[counts["n_windows"] >= min_windows].copy()

    if strata is None:
        eligible["stratum"] = "all"
    else:
        eligible = eligible.merge(strata, on="individual_id", how="left")
        eligible["stratum"] = eligible["stratum"].fillna("unknown")

    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []

    # split within each stratum so class balance is preserved across splits
    for _, group in eligible.groupby("stratum", sort=True):
        ids = group["individual_id"].to_numpy()
        ids = rng.permutation(ids) # seeded shuffle
        n = len(ids)
        n_test = round(n * test_frac)
        n_val = round(n * val_frac)
        test_ids.append(ids[:n_test])
        val_ids.append(ids[n_test : n_test + n_val])
        train_ids.append(ids[n_test + n_val :])

    return Split(
        train=np.concatenate(train_ids),
        val=np.concatenate(val_ids),
        test=np.concatenate(test_ids),
    )

# GenAI ref: GenAI was used to help draft the summary code and the `__main__` summary printout (I had to be somewhere
# and just wanted the rest of the file finsihed before then; I could have written it since it is straightforward, just 
# tedious). We did validate the output by checking the reported individual/window counts against the `counts` table, 
# confirming train/val/test IDs are disjoint, and verifying that the reported window fractions sum to ~100%. Apprx 
def summarise_split(split: Split, counts: pd.DataFrame) -> None:
    """Print individuals and windows per split, plus the window-fraction check."""
    w = counts.set_index("individual_id")["n_windows"]
    total_w = sum(int(w.reindex(ids).sum()) for ids in (split.train, split.val, split.test))

    for name, ids in [("train", split.train), ("val", split.val), ("test", split.test)]:
        n_ind = len(ids)
        n_win = int(w.reindex(ids).sum())
        print(f"  {name:<5}: {n_ind:>4} individuals | {n_win:>8,} windows "
              f"({n_win / total_w * 100:>4.1f}% of windows)")
        

if __name__ == "__main__":
    from pathlib import Path

    SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"
    counts = window_counts_per_individual(SEGMENTS)

    n = len(counts)
    print(f"{n} individuals produced windows | {counts['n_windows'].sum():,} total windows\n")
    print("window-count distribution per individual:")
    for q in [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]:
        print(f"  p{int(q*100):>3}: {counts['n_windows'].quantile(q):>10,.0f}")
    print()
    for thresh in [10, 25, 50, 100, 200]:
        kept = (counts["n_windows"] >= thresh).sum()
        lost_w = counts.loc[counts["n_windows"] < thresh, "n_windows"].sum()
        print(f"  drop <{thresh:>3} windows: keep {kept:>4}/{n} individuals, "
              f"lose {lost_w:>8,} windows ({lost_w / counts['n_windows'].sum() * 100:.1f}%)")
        
    # --- build and report the split at the chosen threshold ---
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    print("\nsplit (min_windows=100, seed=42, stratum=all):")
    summarise_split(split, counts)