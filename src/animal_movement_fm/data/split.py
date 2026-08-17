"""
Assign stork individuals to train/val/test splits, by individual (no leakage).

Splitting is at the individual level: all of one animal's windows go entirely to one split. Windows from
the same bird are heavily autocorrelated (adjacent windows share history), so a window level split would
leak near-duplicates across train and test and measure memorisation, not generalisation.

The split is stratified by a key (taxon by default) so the class balance is preserved across splits;
trivial within single-species stork, but load-bearing at Tier 2 where whole species are held out.
Reproducible via a fixed seed.

Frozen manifests: how splits are actually consumed
--------------------------------------------------
Experiment scripts should NOT call `make_split`; they should read a committed id manifest via `load_split` (or the
project convenience wrapper `stork_split`). `make_split` remains the generator, and is fine for exploratory work.

Two reasons, one historical and one structural.

Historically, `make_split` seeded its shuffle but applied it to whatever row order it was handed, and that order came
from an unstable sort over a table with hundreds of tied window counts. Identical data and seed therefore produced
DIFFERENT splits on different machines (Mac vs Gadi shared only 76/105 test individuals), which silently made
cross-machine results incomparable. Ids are now sorted canonically before the permutation, so that specific defect is
fixed.

Structurally, though, determinism is not enough once the corpus grows. A computed split is a function of the whole id
set, so adding a species for Tier 2 changes that set and reshuffles the EXISTING individuals too; Tier 1 results would
quietly stop being comparable with no error raised. Pinning each dataset's assignment to its own manifest makes tiers
independent, and is the same reason public benchmarks ship explicit split files rather than a seed.

Convention: one manifest per dataset, `splits/<name>_split_seed<seed>.json`, generated once by
`experiments/data/make_split_manifest.py` and committed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def window_counts_per_individual(
    segments_parquet,
    *,
    context_length: int,
) -> pd.DataFrame:
    """Per individual window counts, to set the minimum track threshold before splitting.

    Counts windows exactly as the benchmark will: the harness windower derives one bin contiguous runs within each
    coarse segment, and counts are summed per individual.
    """
    import duckdb

    from animal_movement_fm.eval.harness import iter_windows

    con = duckdb.connect()
    seg_df = con.execute(f"""
        SELECT segment_id, individual_id, bin_index, timestamp, lat, lon
        FROM '{segments_parquet}'
        ORDER BY segment_id, bin_index
    """).df()

    # keyed on the raw id, NOT int(), because RAATD individual_ids are strings such as
    # "CAR2013_sel_a_m_03"; had int cast error raised
    counts: dict = {}
    for _, g in seg_df.groupby("segment_id", sort=False):
        individual_id = g["individual_id"].iloc[0]
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

# GenAI reference: I designed the eligibility policy and began this function; GenAI completed most of the table building implementation.
# It was verified with tests covering eligible and excluded individuals, missing count rows, preserved split order, and no reassignment
# between frozen splits.
def split_eligibility_table(
    split: Split,
    counts: pd.DataFrame,
    *,
    train_min_windows: int = 1,
    eval_min_windows: int = 100,
) -> pd.DataFrame:
    """
    Audit eligibility without changing a frozen split assignment.

    Training individuals need at least one valid contiguous window because every such window supplies legitimate supervision. Validation and
    test individuals need ``eval_min_windows`` because they support per individual estimates and individual level bootstrap inference.

    Individuals are never reassigned or replaced. Missing count rows are treated as zero windows and surfaced in the returned table.
    """
    if train_min_windows < 1:
        raise ValueError("train_min_windows must be >= 1")
    if eval_min_windows < 1:
        raise ValueError("eval_min_windows must be >= 1")
    required = {"individual_id", "n_windows"}
    missing = required - set(counts.columns)
    if missing:
        raise ValueError(f"counts is missing required columns: {sorted(missing)}")
    if counts["individual_id"].duplicated().any():
        raise ValueError("counts must contain at most one row per individual_id")

    # keyed on the canonical id so a manifest and a counts table may spell ids differently
    count_by_key = {
        id_key(i): int(n)
        for i, n in zip(counts["individual_id"], counts["n_windows"], strict=True)
    }
    rows = []
    for split_name, ids, minimum in (
        ("train", split.train, train_min_windows),
        ("val", split.val, eval_min_windows),
        ("test", split.test, eval_min_windows),
    ):
        for individual_id in ids:
            n_windows = count_by_key.get(id_key(individual_id), 0)
            rows.append(
                {
                    "individual_id": individual_id,
                    "split": split_name,
                    "n_windows": n_windows,
                    "min_windows": minimum,
                    "eligible": n_windows >= minimum,
                    "reason": (
                        "eligible"
                        if n_windows >= minimum
                        else f"fewer_than_{minimum}_valid_windows"
                    ),
                }
            )
    return pd.DataFrame(rows)


def apply_window_eligibility(
    split: Split,
    counts: pd.DataFrame,
    *,
    train_min_windows: int = 1,
    eval_min_windows: int = 100,
) -> Split:
    """
    Return the eligible subset of a frozen assignment, without reshuffling.

    Array order is preserved. The input ``split`` and its manifest are not modified; excluded individuals simply contribute no training
    or evaluation windows and are documented by :func:`split_eligibility_table`.
    """
    table = split_eligibility_table(
        split,
        counts,
        train_min_windows=train_min_windows,
        eval_min_windows=eval_min_windows,
    )

    eligible_by_split = {
        name: {
            id_key(i)
            for i in table.loc[(table["split"] == name) & table["eligible"], "individual_id"]
        }
        for name in ("train", "val", "test")
    }

    def keep(ids: np.ndarray, name: str) -> np.ndarray:
        """
        Filter in place order, preserving the source dtype so an integer manifest stays int64 and a string manifest stays object 
        even when nothing survives.
        """
        kept = [i for i in ids if id_key(i) in eligible_by_split[name]]
        return np.asarray(kept, dtype=ids.dtype)

    return Split(
        train=keep(split.train, "train"),
        val=keep(split.val, "val"),
        test=keep(split.test, "test"),
    )


def id_key(value) -> str:
    """
    Canonical comparison key for one individual id.

    Individual ids are not one type across the corpus. Movebank ids are integers but reach us as int64, as float64 after a parquet 
    round trip, or as text; RAATD ids are strings such as `"CAR2013_sel_a_m_03". Added this for robustness. 
    """

    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if np.isfinite(value) and float(value).is_integer() else str(value)
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def canonical_manifest_ids(ids) -> list:
    """
    Sorted, JSON serialisable ids for a manifest.

    Where every id is integral they are written as JSON NUMBERS and sorted numerically, which is what keeps our frozen stork manifest byte 
    identical; otherwise they are written as strings and sorted lexicographically. A mixed set is treated as strings, so the file never 
    contains two spellings of the same id.
    """

    values = list(np.asarray(ids, dtype=object).ravel())
    if not values:
        return []

    keys = [id_key(v) for v in values]
    if all(k.lstrip("-").isdigit() for k in keys):
        return sorted(int(k) for k in keys)
    return sorted(keys)


def make_split(
    counts: pd.DataFrame,
    strata: pd.DataFrame | None = None,
    min_windows: int = 100,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    seed: int = 42,
) -> Split:
    """Split individuals into train/val/test, stratified by key, reproducibly.

    `counts` is the per individual window table from window_counts_per_individual. `strata` maps individual_id -> a
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
        # Bug catch (caught by GenAI; fixed by me)
        # CANONICAL ORDER BEFORE THE SHUFFLE. The seed makes the permutation reproducible, but a permutation is
        # applied to whatever order it is handed, so the split silently inherited the row order of `counts`. That
        # order is not canonical: `window_counts_per_individual` ends with an UNSTABLE `sort_values("n_windows")`
        # and 393 of the 840 eligible stork individuals share an n_windows value with another bird, so tied birds
        # could come out in different orders on different platforms. The result was a split that differed by
        # machine (Mac vs Gadi shared only 76/105 test individuals) despite an identical seed and identical data.
        # Sorting the ids first makes the split a pure function of (individual_ids, seed) and nothing else.
        ids = np.sort(group["individual_id"].to_numpy())
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

def make_split_specific(
    counts: pd.DataFrame,
    *,
    eval_min_windows: int = 25,
    train_min_windows: int = 1,
    n_val: int = 10,
    n_test: int = 10,
    seed: int = 42,
) -> Split:
    """
    Split individuals using the A7 split specific policy.

    Validation and test are drawn from individuals meeting the evaluation window floor. Every remaining individual meeting the 
    lower training floor is assigned to training, with IDs sorted before shuffling so the split is reproducible across machines
    and we don't run into the mac vs gadi error again. 
    """
    if n_val < 0 or n_test < 0:
        raise ValueError("n_val and n_test must be non-negative")
    if train_min_windows > eval_min_windows:
        raise ValueError(
            "train_min_windows above eval_min_windows would make the training pool a subset "
            "of the evaluation pool, which inverts the policy"
        )

    eval_pool = np.sort(
        counts.loc[counts["n_windows"] >= eval_min_windows, "individual_id"].to_numpy()
    )
    if len(eval_pool) < n_val + n_test:
        raise ValueError(
            f"evaluation pool holds {len(eval_pool)} individuals at >= {eval_min_windows} "
            f"windows, fewer than the {n_val + n_test} needed for validation and test"
        )

    shuffled = np.random.default_rng(seed).permutation(eval_pool)
    test_ids = shuffled[:n_test]
    val_ids = shuffled[n_test : n_test + n_val]

    held_out = set(test_ids.tolist()) | set(val_ids.tolist())
    train_pool = np.sort(
        counts.loc[counts["n_windows"] >= train_min_windows, "individual_id"].to_numpy()
    )
    train_ids = np.array([i for i in train_pool if i not in held_out], dtype=train_pool.dtype)

    return Split(train=train_ids, val=val_ids, test=test_ids)


def save_split(split: Split, path: str | Path, *, note: str = "") -> None:
    """
    Save a train, validation, and test split as a JSON file.

    The IDs are sorted to keep the file readable and make changes easy to review. Parent directories are created automatically.

    Parameters
    ----------
    split : Split
        The split to save.
    path : str | Path
        Path to the output JSON file.
    note : str
        Optional note describing how or why the split was created.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": note,
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "train": canonical_manifest_ids(split.train),
        "val": canonical_manifest_ids(split.val),
        "test": canonical_manifest_ids(split.test),
    }
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: str | Path) -> Split:
    """
    Load a train, validation, and test split from a JSON file.

    The three sets are checked for overlapping IDs to prevent individuals from appearing in more than one split.

    Parameters
    ----------
    path : str | Path
        Path to the split JSON file.

    Returns
    -------
    Split
        The loaded train, validation, and test assignments.
    """
    payload = json.loads(Path(path).read_text())

    def as_array(values) -> np.ndarray:
        """Integer manifests load as int64, preserving every existing caller; string manifests
        (RAATD) load as object so their ids survive intact."""
        if len(values) == 0:
            return np.empty(0, dtype=np.int64)
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return np.asarray(values, dtype=np.int64)
        return np.asarray([str(v) for v in values], dtype=object)

    train, val, test = (as_array(payload[k]) for k in ("train", "val", "test"))

    keyed = {name: {id_key(i) for i in ids}
             for name, ids in (("train", train), ("val", val), ("test", test))}
    overlap = (
        (keyed["train"] & keyed["val"])
        | (keyed["train"] & keyed["test"])
        | (keyed["val"] & keyed["test"])
    )
    if overlap:
        raise ValueError(f"split manifest {path} has individuals in more than one split: {sorted(overlap)[:10]}")

    return Split(train=train, val=val, test=test)


# the project's frozen stork split; committed, so every machine reads the SAME individuals
STORK_SPLIT_FILE = "splits/stork_split_seed42.json"


def stork_split(repo_root: str | Path | None = None) -> Split:
    """
    Load the project's fixed white stork split.

    Experiment scripts should use this function rather than recomputing the split, ensuring that the same individuals are used on every machine.

    Parameters
    ----------
    repo_root : str | Path | None
        Root directory of the repository. If omitted, it is inferred from the location of this file.
    Returns
    -------
    Split
        The fixed white-stork train, validation, and test assignments.

    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    path = root / STORK_SPLIT_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"frozen split manifest not found at {path}. It is committed to the repo; if it is missing, "
            f"regenerate it with save_split(make_split(...), '{STORK_SPLIT_FILE}') and note that a regenerated "
            f"split will NOT match the one existing results were computed on."
        )
    return load_split(path)

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
    counts = window_counts_per_individual(SEGMENTS, context_length=24)  # Tier 1, pre-A6

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
