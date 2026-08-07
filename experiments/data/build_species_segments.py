"""
Build a 6 hour segment file from extracted species fixes.

The input parquet is resampled using the shared segment building pipeline, with only segments meeting `--min-segment-bins` 
retained. Selected per individual metadata is carried into the output where available.

Theis script writes the resulting segment parquet and a JSON summary containing the input, output, resampling settings, and 
build statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from animal_movement_fm.data.build_segments import segments_from_fixes
from animal_movement_fm.data.resample import BIN_SECONDS


# per individual metadata to keep in the segment file
# `location_quality` is per fix and must be handled before resampling
CARRY_COLUMNS = (
    "device_type",
    "sex",
    "age_class",
    "scientific_name",
    "deployment_site",
)


def positive_integer(value: str) -> int:
    """Parse a positive integer."""
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="extracted fixes parquet",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="segment parquet to create",
    )
    parser.add_argument(
        "--min-segment-bins",
        required=True,
        type=positive_integer,
        help="minimum number of occupied bins retained in a segment",
    )
    return parser.parse_args()


def to_json_value(value: object) -> object:
    """Convert pandas and NumPy values to JSON compatible types."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    """Build the segment parquet and summary file."""
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must differ")

    fixes = pd.read_parquet(args.input)

    segments, stats = segments_from_fixes(
        fixes,
        bin_seconds=BIN_SECONDS,
        min_segment_bins=args.min_segment_bins,
        carry_columns=CARRY_COLUMNS,
    )

    if segments is None:
        raise SystemExit(
            f"No segments reached the {args.min_segment_bins}-bin minimum; "
            "no files were written."
        )

    # from here below is GenAI; I asked it to finish the main method of this file; I checked the output and it was correct
    args.output.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(args.output, index=False)

    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "bin_seconds": BIN_SECONDS,
        "carry_columns": list(CARRY_COLUMNS),
        **{key: to_json_value(value) for key, value in stats.items()},
    }

    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote {stats['segments']:,} segments "
        f"({stats['bins']:,} bins from {stats['individuals_out']:,} individuals) "
        f"to {args.output}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()