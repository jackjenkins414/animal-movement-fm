# GenAI reference: As with other scripts in the repo, I used GenAI to draft this after I found that 3 pool species have tracks
# copied between the training and validation/test sets. So I outlined the following to GenAI; a training individual with 100+ 
# fixes copied into a val/test individual of the same species, matched on (timestamp, lon, lat) exactly as stored; counts for 
# >= 1, >= 10 and >= 100 shared fixes, with exclusion at >= 100; plus a cross-species check between each fold's pool training 
# fixes and its held-out birds' val/test fixes. I checked the output and haven't changed anything in it, since it functioned as
# intended and adhered to what we use GenAI for in this project. 
"""
Duplicate fix check for the rotating held-out bird study, over the 30 birds of `registered/birds_source_pool.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from animal_movement_fm.data.split import id_key  # noqa: E402
from experiments.data import source_pool_weights as SPW  # noqa: E402

FIX = ["timestamp", "lon", "lat"]
PARTITIONS = ("train", "val", "test")
THRESHOLDS = (1, 10, 100)
EXCLUDE_AT = 100 
OUT = SPW.RESULTS / "birds_rotation" / "duplicate_fixes.csv"


def load_fixes(stem: str, keep_unassigned: bool = False) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    One species' fixes with a partition label, and each id key's spelling in the manifest.
    """
    path = SPW.SPLITS / f"{SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json"
    if keep_unassigned and not path.exists():
        manifest = {part: [] for part in PARTITIONS}
    else:
        manifest = json.loads(path.read_text())
    raw = {id_key(i): i for part in PARTITIONS for i in manifest[part]}
    partition = {id_key(i): part for part in PARTITIONS for i in manifest[part]}
    frame = pd.read_parquet(SPW.PROCESSED / f"{stem}_segments.parquet", columns=["individual_id", *FIX])
    keys = {value: id_key(value) for value in frame["individual_id"].unique()}
    frame["individual"] = frame["individual_id"].map(keys)
    frame["partition"] = frame["individual"].map(partition)
    if keep_unassigned:
        frame["partition"] = frame["partition"].fillna("unassigned")
    frame = frame.dropna(subset=["partition"]).drop_duplicates(["individual", *FIX])
    frame["species"] = stem
    return frame[["species", "individual", "partition", *FIX]].reset_index(drop=True), raw


def shared_fixes(train: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    """Distinct (timestamp, lon, lat) shared by each (training individual, val or test individual) pair."""
    hits = train.merge(held, on=FIX, suffixes=("_train", "_eval"))
    keys = ["species_train", "individual_train", "species_eval", "individual_eval", "partition_eval"]
    return hits.groupby(keys, sort=True).size().rename("shared_fixes").reset_index()


def worst_pair(pairs: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Each training individual's largest overlap with any one val or test individual, within the `by` grouping."""
    return pairs.groupby([*by, "individual_train"], sort=True)["shared_fixes"].max().reset_index()


def threshold_counts(worst: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """How many training individuals reach each threshold, per `by` group."""
    table = worst.groupby(by)["shared_fixes"].agg(
        **{f">={t}": (lambda s, t=t: int((s >= t).sum())) for t in THRESHOLDS}
    )
    return table.reset_index()


def fold_pools(birds: list[str]) -> tuple[dict[str, int], dict[int, set[str]]]:
    fold_of = {n: k for k, members in SPW.BIRDS_ROTATION_FOLDS.items() for n in members}
    if sorted(fold_of) != sorted(birds):
        raise SystemExit("FAIL: the pinned folds do not cover the 30 birds")
    return fold_of, {k: set(SPW.rotation_pool(m, birds)) for k, m in SPW.BIRDS_ROTATION_FOLDS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help=f"print only; do not write {OUT.name}")
    args = parser.parse_args()

    birds = SPW.rotation_birds(json.loads(SPW.BIRDS_OUT.read_text()))
    frames, spelling = [], {}
    for stem in birds:
        frame, raw = load_fixes(stem)
        frames.append(frame)
        spelling[stem] = raw
    fixes = pd.concat(frames, ignore_index=True)
    train = fixes[fixes["partition"] == "train"]
    held = fixes[fixes["partition"] != "train"]
    print(f"{len(birds)} birds: {len(train):,} training fixes, {len(held):,} val and test fixes")

    pairs = shared_fixes(train, held)
    fold_of, pools = fold_pools(birds)
    within = pairs[pairs["species_train"] == pairs["species_eval"]]
    in_pool = pd.Series(
        [s in pools[fold_of[e]] for s, e in zip(pairs["species_train"], pairs["species_eval"])],
        index=pairs.index, dtype=bool,
    )
    cross = pairs[(pairs["species_train"] != pairs["species_eval"]) & in_pool]

    worst = worst_pair(within, ["species_train"])
    table = pd.DataFrame({"species_train": birds}).merge(
        threshold_counts(worst, ["species_train"]), how="left", on="species_train"
    ).fillna(0)
    flagged = worst[worst["shared_fixes"] >= EXCLUDE_AT]
    exclusions = {
        species: sorted((spelling[species][k] for k in group["individual_train"]), key=str)
        for species, group in flagged.groupby("species_train")
    }
    table["excluded at >=100"] = [len(exclusions.get(s, [])) for s in table["species_train"]]
    print("\nwithin species: training individuals sharing fixes with one val or test individual")
    print(table.rename(columns={"species_train": "bird"}).astype({f">={t}": int for t in THRESHOLDS}).to_string(index=False))

    registered = {id_key(i) for i in SPW.BIRDS_TRAINING_EXCLUSIONS["anser_albifrons"]}
    found = {id_key(i) for i in exclusions.get("anser_albifrons", [])}
    print(f"\nregistered goose six: {len(found & registered)} found by exact match; "
          f"not found {sorted(registered - found)}, carried as registered")
    for species, ids in exclusions.items():
        known = {id_key(i) for i in SPW.BIRDS_TRAINING_EXCLUSIONS.get(species, ())}
        new = [i for i in ids if id_key(i) not in known]
        if new:
            print(f"NEW exclusion, {species}: {', '.join(map(str, new))}")

    print("\nacross species: pool training individuals sharing fixes with one of the fold's val or test individuals")
    if cross.empty:
        print("  none at >= 1 shared fix, in any fold")
    else:
        worst_cross = worst_pair(cross, ["species_eval", "species_train"])
        summary = threshold_counts(worst_cross, ["species_eval", "species_train"])
        summary.insert(0, "fold", summary["species_eval"].map(fold_of))
        print(summary.sort_values(["fold", "species_eval", "species_train"]).to_string(index=False))
        for row in worst_cross[worst_cross["shared_fixes"] >= EXCLUDE_AT].itertuples():
            print(f"  >= {EXCLUDE_AT}: {row.species_train} {spelling[row.species_train][row.individual_train]} "
                  f"against held-out {row.species_eval} ({row.shared_fixes} fixes)")

    if args.no_write:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([within.assign(kind="within"), cross.assign(kind="across")], ignore_index=True)
    out["fold"] = out["species_eval"].map(fold_of)
    out.to_csv(OUT, index=False)  # ids and counts only
    print(f"\nwrote {OUT.relative_to(REPO)} ({len(out)} pairs)")


if __name__ == "__main__":
    main()
