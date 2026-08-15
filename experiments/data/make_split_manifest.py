"""
Create a fixed train/validation/test split for a dataset.

The split is saved as a JSON file containing the individual IDs assigned to each subset. Experiment scripts 
can then load this file instead of generating the split again, ensuring that the same individuals are used 
across machines and runs.

A separate manifest should be created for each dataset or training tier. This is useful when the available 
data changes over time, since adding new individuals to a dataset could otherwise change a newly generated split.

Examples
--------
Create a split for the white-stork dataset:

    uv run python experiments/data/make_split_manifest.py \
        --segments data/processed/stork_segments.parquet \
        --name stork

Create a stratified split for a multi-species dataset:

    uv run python experiments/data/make_split_manifest.py \
        --segments data/processed/marine_segments.parquet \
        --name marine_multispecies \
        --strata data/processed/marine_taxa.parquet

The output is then written to:

    splits/<name>_split_seed<seed>.json

Note; an existing manifest will not be overwritten unless ``--force`` is provided.
"""

# GenAI reference: made minor modifications to this file when checking it
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import (
    make_split,
    make_split_specific,
    save_split,
    summarise_split,
    window_counts_per_individual,
)

REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--segments",
        required=True,
        help="path to the dataset's segment parquet file",
    )
    p.add_argument(
        "--name",
        required=True,
        help="dataset name used in the output manifest filename",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used to generate the split",
    )
    p.add_argument(
        "--policy",
        choices=("split-specific", "filter-then-split"),
        default="split-specific",
        help="A7 fixed this at split-specific; filter-then-split is retained only so the "
             "frozen stork manifest stays reproducible from this script",
    )
    p.add_argument(
        "--eval-individuals",
        type=int,
        default=10,
        help="individuals drawn into validation and into test under the A7 policy",
    )
    p.add_argument(
        "--min-windows",
        type=int,
        default=25,
        help="exclude individuals with fewer than this many windows",
    )
    p.add_argument(
        "--context-length",
        type=int,
        default=CONTEXT_LENGTH,
        help="context length used when counting available windows",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.125,
        help="fraction of eligible individuals assigned to validation",
    )
    p.add_argument(
        "--test-frac",
        type=float,
        default=0.125,
        help="fraction of eligible individuals assigned to testing",
    )
    p.add_argument(
        "--strata",
        default=None,
        help=(
            "optional parquet or CSV file with individual_id and stratum columns; "
            "splitting is performed separately within each stratum"
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite the output file if it already exists",
    )
    return p.parse_args()


def main() -> None:
    """Generate and save the split manifest."""
    args = parse_args()
    segments = Path(args.segments)

    if not segments.exists():
        raise SystemExit(f"Segments file not found: {segments}")

    out = REPO / "splits" / f"{args.name}_split_seed{args.seed}.json"

    if out.exists() and not args.force:
        raise SystemExit(
            f"Split manifest already exists: {out}\n"
            "Use --force to replace it. Replacing the file may change which "
            "individuals belong to each split, so any affected results should "
            "be recomputed."
        )

    strata = None
    if args.strata:
        sp = Path(args.strata)
        strata = pd.read_parquet(sp) if sp.suffix == ".parquet" else pd.read_csv(sp)

        missing = {"individual_id", "stratum"} - set(strata.columns)
        if missing:
            raise SystemExit(
                "The --strata file must contain individual_id and stratum "
                f"columns. Missing: {missing}"
            )

    print(f"Counting windows with context length {args.context_length}...")
    counts = window_counts_per_individual(
        segments,
        context_length=args.context_length,
    )

    eligible = int((counts["n_windows"] >= args.min_windows).sum())
    print(
        f"  {len(counts):,} individuals produced windows; "
        f"{eligible:,} have at least {args.min_windows}"
    )

    if args.policy == "split-specific":
        # A7. Validation and test take a fixed count from the >=F pool; every remaining individual with at least one valid 
        # window goes to training
        split = make_split_specific(
            counts,
            eval_min_windows=args.min_windows,
            train_min_windows=1,
            n_val=args.eval_individuals,
            n_test=args.eval_individuals,
            seed=args.seed,
        )
    else:
        # The superseded filter then split policy, retained only so the frozen stork manifest remains reproducible from this script
        split = make_split(
            counts,
            strata=strata,
            min_windows=args.min_windows,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )

    print()
    summarise_split(split, counts)

    if args.policy == "split-specific":
        note = (
            f"Split for '{args.name}' generated from {segments.name} under the A7 "
            f"split-specific policy: seed={args.seed}, evaluation floor F={args.min_windows} "
            f"(A5), context_length={args.context_length} (A6), "
            f"{args.eval_individuals} validation and {args.eval_individuals} test drawn from "
            f"the >=F pool, training taking every remaining individual with >=1 valid window."
        )
    else:
        note = (
            f"Split for '{args.name}' generated from {segments.name} under the SUPERSEDED "
            f"filter-then-split policy: seed={args.seed}, min_windows={args.min_windows}, "
            f"context_length={args.context_length}, val_frac={args.val_frac}, "
            f"test_frac={args.test_frac}, "
            f"strata={'yes' if strata is not None else 'none'}."
        )

    save_split(split, out, note=note)

    print(f"\nSaved split manifest to {out}")
    print("Commit this file so future runs use the same split.")


if __name__ == "__main__":
    main()
