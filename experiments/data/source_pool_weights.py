"""
Derive the Tier 2 pretraining sampling weights from A8.

Species are sampled proportional to n^alpha, where n is training window count at the registered context length. We rederive counts  
from frozen manifests using the production windowing code and checked against registered guard values.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path

import pandas as pd

from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import window_counts_per_individual

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
RESULTS = REPO / "results"

ALPHA = 0.5  # fixed by A8

# segments stem, split stem, domain, role
ROLES: tuple[tuple[str, str, str, str], ...] = (
    ("ciconia_ciconia", "stork", "flying", "source"),
    ("larus_fuscus", "larus_fuscus", "flying", "held-out"),
    ("cervus_elaphus", "cervus_elaphus", "terrestrial", "source"),
    ("canis_lupus", "canis_lupus", "terrestrial", "held-out"),
    ("mirounga_leonina", "mirounga_leonina", "marine", "source"),
    ("mirounga_angustirostris", "mirounga_angustirostris", "marine", "held-out"),
)

# registered window count checks.
GUARDS: dict[str, dict[str, int]] = {
    "ciconia_ciconia": {"train": 806595, "all": 1064095},
    "larus_fuscus": {"train": 597407},
    "cervus_elaphus": {"train": 556579, "all": 593874},
    "canis_lupus": {"train": 47136, "all": 62013},
    "mirounga_leonina": {"train": 53076, "all": 55822},
    "mirounga_angustirostris": {"train": 43815, "all": 48587},
}


def species_window_counts(segments_stem: str, split_stem: str) -> dict[str, int]:
    """Return window counts for each frozen split of one species."""
    counts = window_counts_per_individual(
        str(PROCESSED / f"{segments_stem}_segments.parquet"),
        context_length=CONTEXT_LENGTH,
    )
    counts["individual_id"] = counts["individual_id"].astype(str)
    per_individual = counts.set_index("individual_id")["n_windows"]

    manifest_path = SPLITS / f"{split_stem}_split_seed42.json"
    manifest = json.loads(manifest_path.read_text())

    out: dict[str, int] = {}

    for split in ("train", "val", "test"):
        ids = [str(i) for i in manifest[split]]
        windows = per_individual.reindex(ids).fillna(0)

        out[split] = int(windows.sum())
        out[f"n_{split}"] = len(ids)

    out["all"] = out["train"] + out["val"] + out["test"]
    return out


def draw_probabilities(
    counts: dict[str, int],
    alpha: float = ALPHA,
) -> dict[str, float]:
    "Add after"
    weights = {
    species: n_windows**alpha
    for species, n_windows in counts.items()}
    
    total = sum(weights.values())

    return {
        species: weight / total
        for species, weight in weights.items()}


def check_guards(table: pd.DataFrame) -> list[str]:
    """Return count mismatches."""
    failures = []
    indexed = table.set_index("species")

    for species, expected in GUARDS.items():
        if species not in indexed.index:
            continue

        for key, want in expected.items():
            got = int(indexed.loc[species, key])
            if got != want:
                failures.append(f"{species} {key} = {got}, expected {want}")

    return failures


# GenAI (16-09-2026): the Tier 3 mode below was written by GenAI to apply preregistration_tier3.md sec 1 mechanically
TIER3_FIXED = {"ciconia_ciconia": "flying", "cervus_elaphus": "terrestrial", "mirounga_leonina": "marine"}
TIER3_PROFILE_KEY = {"ciconia_ciconia": "white_stork"}  # the stork's profile row predates the binomial stems
TIER3_MIN_TRAIN_WINDOWS = 2_000  # preregistration_tier3.md sec 1, candidate condition 4
TIER3_OUT = REPO / "registered" / "tier3_source_pool.json"


def tier3(args: argparse.Namespace) -> None:
    """Select the nine Tier 3 sources under preregistration_tier3.md sec 1 and weight them under A8."""
    from animal_movement_fm.analysis.regime_profile import DESCRIPTORS, coverage_distances, select_source_pool

    role_stems = {stem for stem, *_ in ROLES}
    eligibility = pd.read_csv(RESULTS / "species_eligibility.csv")
    passing = set(eligibility.loc[eligibility["clears_gate1"].astype(bool), "species"])
    profiles = pd.read_csv(RESULTS / "regime_profiles.csv")
    complete = profiles[profiles[DESCRIPTORS].notna().all(axis=1)].set_index("species")

    counts: dict[str, int] = {}
    for stem in TIER3_FIXED:
        split_stem = next(split for seg, split, *_ in ROLES if seg == stem)
        counts[stem] = species_window_counts(stem, split_stem)["train"]
        if counts[stem] != GUARDS[stem]["train"]:
            raise SystemExit(f"FAIL: {stem} has {counts[stem]} training windows, registered {GUARDS[stem]['train']}")

    table = []
    candidates: dict[str, str] = {}
    for stem in sorted(complete.index):
        if stem in role_stems or stem in TIER3_PROFILE_KEY.values() or stem == "stork":
            continue
        manifest = SPLITS / f"{stem}_split_seed42.json"
        if stem not in passing or not manifest.exists():
            table.append({"species": stem, "domain": complete.loc[stem, "domain"], "reason": "fails A7 gate 1"})
            continue
        counts[stem] = species_window_counts(stem, stem)["train"]
        ok = counts[stem] >= TIER3_MIN_TRAIN_WINDOWS
        table.append({"species": stem, "domain": complete.loc[stem, "domain"], "train_windows": counts[stem],
                      "reason": "candidate" if ok else f"below {TIER3_MIN_TRAIN_WINDOWS} training windows"})
        if ok:
            candidates[stem] = str(complete.loc[stem, "domain"])

    keyed = complete.rename(index={v: k for k, v in TIER3_PROFILE_KEY.items()})
    rows = keyed.loc[list(TIER3_FIXED) + sorted(candidates)].reset_index()
    frozen = json.loads((REPO / "registered" / "tier2_descriptor_standardisation.json").read_text())
    distances = coverage_distances(rows, standardisation=frozen)

    result = select_source_pool(distances, TIER3_FIXED, candidates, counts)
    pool = result["pool"]
    weights = draw_probabilities({name: counts[name] for name in pool})

    print("candidates: " + ", ".join(f"{d} {sum(1 for v in candidates.values() if v == d)}"
                                      for d in ("flying", "terrestrial", "marine")))
    print(f"pools scored: {result['n_pools_scored']:,}; tied on mean: {result['n_tied_on_mean']}")
    print(f"selected, mean {result['mean_distance']:.4f}, min {result['min_distance']:.4f}:")
    domain_of = {**TIER3_FIXED, **candidates}
    for name in sorted(pool, key=lambda n: (domain_of[n], n)):
        print(f"  {domain_of[name]:12s} {name:26s} n={counts[name]:9,d}  p={weights[name] * 100:7.3f}%")
    print("runners up:")
    for item in result["runners_up"]:
        print(f"  mean {item['mean_distance']:.4f} min {item['min_distance']:.4f}  added {item['added']}")

    payload = {
        "_comment": [
            "Tier 3 source pool selected mechanically under preregistration_tier3.md sec 1, with A8 draw",
            "probabilities (alpha 0.5) over the selected nine. Written by source_pool_weights.py --tier3.",
            "Sub-pools (Za, inner-loop folds) are not included; they depend on choices the Tier 3 prereg has not fixed yet.",
        ],
        "rule": "preregistration_tier3.md sec 1",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "min_train_windows": TIER3_MIN_TRAIN_WINDOWS,
        "standardisation": "registered/tier2_descriptor_standardisation.json",
        "fixed": TIER3_FIXED,
        "candidate_table": table,
        **{k: result[k] for k in ("pool", "added", "mean_distance", "min_distance", "train_windows",
                                  "n_pools_scored", "n_tied_on_mean", "candidates_per_domain",
                                  "distance_matrix", "runners_up")},
        "train_window_counts": {name: counts[name] for name in pool},
        "source_pool": weights,
    }

    if TIER3_OUT.exists():
        previous = json.loads(TIER3_OUT.read_text())
        same = previous["pool"] == pool and all(
            abs(previous["source_pool"][k] - weights[k]) <= 1e-12 for k in pool
        )
        if not same:
            raise SystemExit(f"FAIL: {TIER3_OUT.name} already holds a different selection; the rule is run once")
        print(f"reproduced {TIER3_OUT.relative_to(REPO)}")
        return
    if args.no_write:
        return
    TIER3_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {TIER3_OUT.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print only; do not write results",
    )
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="select the nine Tier 3 sources (preregistration_tier3.md sec 1) and write registered/tier3_source_pool.json",
    )
    args = parser.parse_args()
    if args.tier3:
        tier3(args)
        return

    rows = []
    for segments_stem, split_stem, domain, role in ROLES:
        counts = species_window_counts(segments_stem, split_stem)
        rows.append(
            {
                "species": segments_stem,
                "domain": domain,
                "role": role,
                "n_train": counts["n_train"],
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
                "all": counts["all"],
            }
        )

    # GenAI reference: used to polish reporting/output code below
    table = pd.DataFrame(rows)

    failures = check_guards(table)
    if failures:
        raise SystemExit("FAIL: guard mismatch\n  " + "\n  ".join(failures))

    sources = table[table["role"] == "source"]
    source_counts = dict(zip(sources["species"], sources["train"].astype(int)))

    full = draw_probabilities(source_counts)
    sub_pools = {
        f"without_{held}": draw_probabilities(
            {s: n for s, n in source_counts.items() if s != held}
        )
        for held in source_counts
    }

    pd.set_option("display.width", 200)

    print(f"C={CONTEXT_LENGTH}; alpha={ALPHA}")
    print()
    print(table.to_string(index=False))

    print("\nsource pool:")
    for species, p in full.items():
        print(f"  {species:24s} n={source_counts[species]:8d}  p={p * 100:7.3f}%")

    print("\nsub-pools:")
    for name, probs in sub_pools.items():
        inner = ", ".join(f"{s} {p * 100:.3f}%" for s, p in probs.items())
        print(f"  {name:32s} {inner}")

    print(f"\nguards: {sum(len(v) for v in GUARDS.values())} passed")

    # GenAI catch: the training scripts now read the tracked registered/tier2_source_pool_weights.json, not
    # the file this script writes; the recomputed weights must still reproduce it exactly
    registered = json.loads((REPO / "registered" / "tier2_source_pool_weights.json").read_text())
    drift = [
        f"{pool}.{species}: {value!r} vs registered {registered_pool[species]!r}"
        for pool, computed_pool, registered_pool in [("source_pool", full, registered["source_pool"])]
        + [(f"sub_pools.{k}", v, registered["sub_pools"][k]) for k, v in sub_pools.items()]
        for species, value in computed_pool.items()
        if abs(value - registered_pool[species]) > 1e-12
    ]
    if drift:
        raise SystemExit("FAIL: recomputed weights differ from the registered file\n  " + "\n  ".join(drift))
    print("matches registered/tier2_source_pool_weights.json")

    if args.no_write:
        return

    RESULTS.mkdir(exist_ok=True)

    payload = {
        "registered_by": "preregistration_tier2.md amendment A8",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "tunable": False,
        "counts": table.to_dict(orient="records"),
        "source_pool": full,
        "sub_pools": sub_pools,
    }

    out = RESULTS / "source_pool_weights.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()