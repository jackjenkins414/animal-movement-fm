"""
Audit whether candidate species have enough valid windows for Tier 2.

For each species, this script counts valid modelling windows per individual using the same
contiguity rules as the benchmark. It reports the pool of individuals eligible for training
(>= 1 window) and validation/test (>= 100 windows), and projects a 75/12.5/12.5 split over the
validation/test pool.

This is an audit only. It reports the observed eligibility pools and a filter-then-split
projection; it does not choose a split construction policy or assign species roles.

Usage
-----
uv run python experiments/data/species_eligibility.py
uv run python experiments/data/species_eligibility.py --species soes walrus stork
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data.split import window_counts_per_individual

REPO = Path(__file__).resolve().parents[2]
SEGMENTS_DIR = REPO / "data" / "processed"
RESULTS_DIR = REPO / "results"

VAL_FRAC = 0.125
TEST_FRAC = 0.125


def audit(
    path: Path,
    species: str,
    context_length: int,
    eval_min_windows: int,
    train_min_windows: int,
):
    """Calculate eligibility statistics for one species."""

    counts = window_counts_per_individual(
        path,
        context_length=context_length,
    ).rename(columns={"n_windows": "windows"})

    counts["species"] = species
    counts["segments_file"] = str(path.resolve())
    counts["context_length"] = context_length
    counts["train_min_windows"] = train_min_windows
    counts["eval_min_windows"] = eval_min_windows

    windows = counts["windows"].to_numpy()
    n_individuals = len(counts)

    n_train_pool = int((windows >= train_min_windows).sum())
    n_eval_pool = int((windows >= eval_min_windows).sum())

    # project the same 75/12.5/12.5 split we used for the stork dataset (this might change later)
    n_test = round(n_eval_pool * TEST_FRAC)
    n_val = round(n_eval_pool * VAL_FRAC)
    n_train = n_eval_pool - n_val - n_test

    # record device types because some species contain both PTT and GPS individuals
    segments = pd.read_parquet(path)
    device_counts = {}

    if "device_type" in segments.columns:
        device_counts = (
            segments.groupby("individual_id")["device_type"]
            .first()
            .value_counts()
            .to_dict()
        )

    row = {
        "species": species,
        "segments_file": str(path.resolve()),
        "context_length": context_length,
        "train_min_windows": train_min_windows,
        "eval_min_windows": eval_min_windows,
        "individuals_total": n_individuals,
        f"individuals_ge{train_min_windows}_window": n_train_pool,
        f"individuals_ge{eval_min_windows}_windows": n_eval_pool,
        "windows_total": int(windows.sum()),
        "windows_median_per_individual": (
            float(np.median(windows)) if n_individuals else 0.0
        ),
        "windows_max": int(windows.max()) if n_individuals else 0,
        "projected_train": n_train,
        "projected_val": n_val,
        "projected_test": n_test,
        "device_mix": ", ".join(
            f"{device}:{count}"
            for device, count in sorted(device_counts.items())
        ),
    }

    return row, counts


def species_name(path: Path) -> str:
    """Return the species name encoded by a segment parquet filename."""
    return path.stem.removesuffix("_segments")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=SEGMENTS_DIR,
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="species to audit; default is all *_segments.parquet files",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--eval-min-windows",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--train-min-windows",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory for the summary and per-individual CSV files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing output files",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    paths = sorted(args.segments_dir.glob("*_segments.parquet"))

    if args.species:
        available = {species_name(path).lower(): path for path in paths}
        wanted = list(dict.fromkeys(species.lower() for species in args.species))
        missing = sorted(set(wanted) - available.keys())
        if missing:
            raise SystemExit(
                "No segment file found for requested species: " + ", ".join(missing)
            )
        paths = [available[species] for species in wanted]

    if not paths:
        raise SystemExit(f"No segment files found in {args.segments_dir}")

    summary_output = args.output_dir / "species_eligibility.csv"
    individual_output = args.output_dir / "species_eligibility_per_individual.csv"
    existing_outputs = [path for path in (summary_output, individual_output) if path.exists()]

    if existing_outputs and not args.force:
        existing = ", ".join(str(path) for path in existing_outputs)
        raise SystemExit(f"Output file already exists: {existing}. Use --force to replace it.")

    rows = []
    individual_tables = []

    print("Auditing species eligibility...\n")

    for path in paths:
        species = species_name(path)

        row, counts = audit(
            path,
            species,
            args.context_length,
            args.eval_min_windows,
            args.train_min_windows,
        )

        rows.append(row)
        individual_tables.append(counts)

        eval_pool = row[f"individuals_ge{args.eval_min_windows}_windows"]

        print(
            f"{species}: "
            f"{row['individuals_total']} individuals, "
            f"{eval_pool} with >= {args.eval_min_windows} windows, "
            f"{row['windows_total']:,} windows total"
        )

    eval_pool_col = f"individuals_ge{args.eval_min_windows}_windows"

    table = (
        pd.DataFrame(rows)
        .sort_values(eval_pool_col, ascending=False)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(summary_output, index=False)
    pd.concat(individual_tables, ignore_index=True).to_csv(
        individual_output,
        index=False,
    )

    pool_columns = [
        "species",
        "individuals_total",
        f"individuals_ge{args.train_min_windows}_window",
        eval_pool_col,
        "windows_total",
        "windows_median_per_individual",
    ]

    split_columns = [
        "species",
        "projected_train",
        "projected_val",
        "projected_test",
        "device_mix",
    ]

    print("\nEligibility pools")
    print("-----------------")
    print(table[pool_columns].to_string(index=False))

    print(
        f"\nProjected filter-then-split 75/12.5/12.5 split "
        f"(individuals with >= {args.eval_min_windows} windows)"
    )
    print("------------------------------------------------------------")
    print(table[split_columns].to_string(index=False))

    print(
        "\nThis audit reports observed pools and a filter-then-split projection. "
        "It does not choose a split construction policy or assign species roles."
    )

    print("\nSaved results:")
    print(f"  Species summary: {summary_output}")
    print(f"  Per-individual counts: {individual_output}")


if __name__ == "__main__":
    main()
