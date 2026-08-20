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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print only; do not write results",
    )
    args = parser.parse_args()

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