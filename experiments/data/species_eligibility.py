"""
Audit whether candidate species have enough valid windows for Tier 2.

For each species, this script counts valid modelling windows per individual using the same contiguity rules as the benchmark. 
It reports the pool of individuals eligible for training (>= 1 window) and validation/test (>= F windows), and applies the A7 
split specific policy: validation and test take 10 individuals each from the >= F pool, training takes every remaining 
individual with at least one valid window. Gate 1 is `n(>=F) >= 20` and `n(>=1) >= 40`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import window_counts_per_individual

REPO = Path(__file__).resolve().parents[2]
SEGMENTS_DIR = REPO / "data" / "processed"
RESULTS_DIR = REPO / "results"

# A7 split-specific policy. Validation and test take EVAL_INDIVIDUALS each from >=F pool; training takes every remaining 
# individual with at least one valid window; gate 1  needs 2*EVAL_INDIVIDUALS above the floor and GATE1_MIN_TOTAL with 
# at least one window
EVAL_INDIVIDUALS = 10
GATE1_MIN_TOTAL = 40

# A5 fixed the evaluation floor at 25 and requires sensitivity at these values
FLOOR_SENSITIVITY = (10, 25, 50, 100)


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

    # A7;s splitting is split specific 
    n_val = min(EVAL_INDIVIDUALS, max(n_eval_pool - EVAL_INDIVIDUALS, 0))
    n_test = min(EVAL_INDIVIDUALS, max(n_eval_pool - n_val, 0))
    n_train = max(n_train_pool - n_val - n_test, 0)
    clears_gate1 = bool(n_eval_pool >= 2 * EVAL_INDIVIDUALS and n_train_pool >= GATE1_MIN_TOTAL)

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
        "split_train": n_train,
        "split_val": n_val,
        "split_test": n_test,
        "clears_gate1": clears_gate1,
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
        default=CONTEXT_LENGTH,
        help="history bins per window; A6 fixed this at 12",
    )
    parser.add_argument(
        "--eval-min-windows",
        type=int,
        default=25,
        help="evaluation window floor F; A5 fixed this at 25",
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
        "split_train",
        "split_val",
        "split_test",
        "clears_gate1",
        "device_mix",
    ]

    print("\nEligibility pools")
    print("-----------------")
    print(table[pool_columns].to_string(index=False))

    print(
        f"\nA7 split-specific allocation "
        f"(individuals with >= {args.eval_min_windows} windows)"
    )
    print("------------------------------------------------------------")
    print(table[split_columns].to_string(index=False))

    print(
        "\nSplit sizes follow the A7 split-specific policy: 10 validation and 10 test "
        "drawn from the >=F pool, training taking every remaining individual with >=1 window. "
        "It does not choose a split construction policy or assign species roles."
    )

    print("\nSaved results:")
    print(f"  Species summary: {summary_output}")
    print(f"  Per-individual counts: {individual_output}")


if __name__ == "__main__":
    main()
