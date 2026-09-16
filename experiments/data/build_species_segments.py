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

import numpy as np
import pandas as pd

from animal_movement_fm.data.build_segments import segments_from_fixes
from animal_movement_fm.data.resample import BIN_SECONDS
from animal_movement_fm.data.split import ROLE_STEMS


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
        "--batch",
        type=int,
        default=50,
        help="individuals resampled per batch; lower it if memory is tight to avoid the before",
    )
    parser.add_argument(
        "--allow-role-species",
        action="store_true",
        help="permit rewriting a Tier 2 role species' segments; refused by default because their split manifests "
             "are frozen and used (A2)",
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


def cadence_diagnostics(path: Path) -> dict:
    """
    Compute species wide cadence diagnostics in one DuckDB pass.

    These statistics span all individuals, so they are calculated outside the batching loop to keep the reported values independent of batch size.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    return con.execute(
        f"""
        WITH d AS (
            SELECT epoch(timestamp) - lag(epoch(timestamp))
                     OVER (PARTITION BY individual_id ORDER BY timestamp) AS gap
            FROM read_parquet('{path}')
        )
        SELECT median(gap) / 3600.0 AS median_native_gap_h,
               avg((gap <= {BIN_SECONDS})::INT) AS share_gaps_within_bin
        FROM d WHERE gap IS NOT NULL
        """
    ).df().iloc[0].to_dict()


def build(args) -> tuple[pd.DataFrame | None, dict]:
    """
    Resample individuals in batches and merge their segment statistics.

    Batching keeps memory use manageable for very large species while preserving the same per individual resampling behaviour as 
    a single full-data pass.
    """
    import duckdb

    # keys that must not be summed across batches; a setting is carried, a span is spanned
    SETTINGS = {"min_segment_bins"}
    SPANS = {"first_bin": min, "last_bin": max}

    con = duckdb.connect()
    con.execute("PRAGMA threads=4; PRAGMA memory_limit='4GB'")
    ids = con.execute(
        f"SELECT DISTINCT individual_id FROM read_parquet('{args.input}') ORDER BY 1"
    ).df()["individual_id"].tolist()

    frames, merged = [], {}
    for start in range(0, len(ids), args.batch):
        chunk = ids[start : start + args.batch]
        fixes = con.execute(
            f"SELECT * FROM read_parquet('{args.input}') "
            "WHERE individual_id IN (SELECT unnest($ids))",
            {"ids": chunk},
        ).df()
        segments, stats = segments_from_fixes(
            fixes,
            bin_seconds=BIN_SECONDS,
            min_segment_bins=args.min_segment_bins,
            carry_columns=CARRY_COLUMNS,
        )
        # GenAI assisted in the casing here but I implented this
        for key, value in stats.items():
            if key in SETTINGS:
                # a setting, not a count. `min_segment_bins` is an int, so summing it would report 13 x the number of batches
                merged.setdefault(key, value)
            elif key in SPANS:
                # batches are ordered by individual, not by time, so the first batch's range is an arbitrary subset of the species'
                current = merged.get(key)
                if pd.isna(value):
                    merged.setdefault(key, value)
                elif current is None or pd.isna(current):
                    merged[key] = value
                else:
                    merged[key] = SPANS[key](current, value)
            elif isinstance(value, (int, np.integer)):
                merged[key] = merged.get(key, 0) + int(value)
            else:
                merged.setdefault(key, value)
        if segments is not None:
            frames.append(segments)
        del fixes, segments

    if not frames:
        return None, merged
    out = pd.concat(frames, ignore_index=True)
    merged.update(cadence_diagnostics(args.input))
    merged["segments"] = int(out["segment_id"].nunique())
    merged["batched_individuals_per_batch"] = args.batch
    return out, merged


def main() -> None:
    """Build the segment parquet and summary file."""
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output paths must differ")

    # GenAI bug find: same A2 refusal extract_species_fixes.py already applies one step upstream
    stem = args.output.name.removesuffix(".parquet").removesuffix("_segments")
    if stem in ROLE_STEMS and not args.allow_role_species:
        raise SystemExit(
            f"FAIL: {stem} holds a Tier 2 role and its frozen split manifest has been used. Rewriting its segments "
            "moves the population underneath a sealed manifest, which A2 forbids. Pass --allow-role-species only "
            "if that is genuinely what you mean."
        )

    segments, stats = build(args)

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