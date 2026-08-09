"""
Test how much precision is gained by requiring a minimum number of valid windows per individual.

The preregistration requires each validation or test individual to contribute at least 100 valid
windows (as of now at least). Because that threshold was chosen by our judgement rather than from
a calculation, this script checks the trade-off between retaining individuals and improving
precision.

Only source data are used. No held-out species or sealed partitions are read, and no model is
trained. Precision is measured using persistence error, the registered comparator P, so this can be
run before species selection. The result should be frozen before any held-out performance is examined.

For each window floor:
1. Keep training individuals with at least that many valid windows.
2. Calculate the individual-weighted p90 persistence error over the full eligible pool. This is the
   reference value for that floor because changing the floor also changes the population represented.
3. Repeatedly draw simulated test sets of `n` individuals without replacement.
4. Within each test set, cluster-bootstrap individuals with replacement, keeping each individual's
   windows together.
5. Record the 95% bootstrap interval width and whether it contains the reference p90.

Quantiles are individual weighted rather than window weighted so long tracks do not dominate. This
corresponds to choosing an individual first, then one of its windows.

The output CSV records the input files and every simulation setting needed to reproduce the run.
An existing output is not replaced unless `--force` is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data.split import load_split

REPO = Path(__file__).resolve().parents[2]
SEGMENTS_DIR = REPO / "data" / "processed"
DEFAULT_OUTPUT = REPO / "results" / "window_floor_sensitivity.csv"

EARTH_RADIUS_M = 6_371_000.0


def positive_integer(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be least 1")
    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a non-negative integer."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great circle distance in metres."""

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = np.radians(lon2) - np.radians(lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def persistence_errors(
    segments: pd.DataFrame,
    context_length: int,
) -> dict:
    """
    Return persistence errors for every valid window, grouped by individual.

    Windows are restricted to contiguous one-bin runs. For persistence, the prediction error is
    simply the one bin displacement from the final context fix to the target.
    """

    errors: dict = {}
    segments = segments.sort_values(["segment_id", "bin_index"])

    for _, segment in segments.groupby("segment_id", sort=False):
        bins = segment["bin_index"].to_numpy()
        lat = segment["lat"].to_numpy(dtype=float)
        lon = segment["lon"].to_numpy(dtype=float)
        individual_id = segment["individual_id"].iloc[0]

        breaks = np.flatnonzero(np.diff(bins) != 1) + 1
        lat_runs = np.split(lat, breaks)
        lon_runs = np.split(lon, breaks)

        for run_lat, run_lon in zip(lat_runs, lon_runs, strict=True):
            if len(run_lat) <= context_length:
                continue

            step_errors = haversine_m(
                run_lat[:-1],
                run_lon[:-1],
                run_lat[1:],
                run_lon[1:],
            )

            # a run of length L contributes L - context_length valid targets
            valid_errors = step_errors[context_length - 1 :]
            errors.setdefault(individual_id, []).append(valid_errors)

    return {
        individual_id: np.sort(np.concatenate(individual_errors))
        for individual_id, individual_errors in errors.items()
    }


def weighted_quantile(arrays: list[np.ndarray], q: float) -> float:
    """Calculate a quantile giving each individual equal total weight."""

    n_individuals = len(arrays)

    values = np.concatenate(arrays)
    weights = np.concatenate(
        [
            np.full(len(values_i), 1.0 / (n_individuals * len(values_i)))
            for values_i in arrays
        ]
    )

    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]

    index = np.searchsorted(np.cumsum(weights), q)
    return float(values[index])


def simulate_study(
    pool: list[np.ndarray],
    n_test: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Simulate one test set and return its bootstrap interval and point estimate."""

    chosen_indices = rng.choice(len(pool), size=n_test, replace=False)
    chosen = [pool[i] for i in chosen_indices]

    point = weighted_quantile(chosen, 0.90)

    replicates = np.empty(n_boot)
    for i in range(n_boot):
        bootstrap_indices = rng.integers(0, n_test, size=n_test)
        bootstrap_sample = [chosen[j] for j in bootstrap_indices]
        replicates[i] = weighted_quantile(bootstrap_sample, 0.90)

    lo, hi = np.percentile(replicates, [2.5, 97.5])
    return float(lo), float(hi), point


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--floors",
        type=positive_integer,
        nargs="+",
        default=[10, 25, 50, 100],
    )
    parser.add_argument(
        "--n-test",
        type=positive_integer,
        nargs="+",
        default=[10, 20, 30],
    )
    parser.add_argument("--context-length", type=positive_integer, default=24)
    parser.add_argument("--studies", type=positive_integer, default=120)
    parser.add_argument("--boot", type=positive_integer, default=300)
    parser.add_argument("--seed", type=nonnegative_integer, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file to create",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the output file if it already exists",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"Output file already exists: {args.output}. Use --force to replace it.")

    sources = {
        "stork": (
            SEGMENTS_DIR / "stork_segments.parquet",
            REPO / "splits/stork_split_seed42.json",
        ),
        "soes": (
            SEGMENTS_DIR / "soes_segments.parquet",
            None,
        ),
    }

    rows = []

    for species, (segments_path, split_path) in sources.items():
        if not segments_path.is_file():
            raise SystemExit(f"Segments file not found: {segments_path}")
        if split_path is not None and not split_path.is_file():
            raise SystemExit(f"Split manifest not found: {split_path}")

        segments = pd.read_parquet(segments_path)

        if split_path is not None:
            train_ids = {str(i) for i in load_split(split_path).train}
            segments = segments[
                segments["individual_id"].astype(str).isin(train_ids)
            ]
            scope = f"{len(train_ids)} frozen training individuals"
        else:
            # SOES has no frozen split, so there is no sealed partition to leak
            scope = "all individuals (no split frozen yet; nothing sealed to leak)"

        errors = persistence_errors(segments, args.context_length)

        print(
            f"\n{species}: {scope}; "
            f"{len(errors)} individuals with >= 1 window"
        )

        for floor in args.floors:
            pool = [
                individual_errors
                for individual_errors in errors.values()
                if len(individual_errors) >= floor
            ]

            if not pool:
                print(f"  floor {floor:3d}: no eligible individuals, skipped")
                continue

            truth = weighted_quantile(pool, 0.90)

            for n_test in args.n_test:
                if len(pool) < n_test:
                    print(
                        f"  floor {floor:3d}  n={n_test:3d}: "
                        f"pool {len(pool)} too small, skipped"
                    )
                    continue

                rng = np.random.default_rng(args.seed)

                widths = []
                points = []
                covered = 0

                for _ in range(args.studies):
                    lo, hi, point = simulate_study(
                        pool,
                        n_test,
                        args.boot,
                        rng,
                    )

                    widths.append(hi - lo)
                    points.append(point)
                    covered += int(lo <= truth <= hi)

                median_width = float(np.median(widths))
                coverage = covered / args.studies

                rows.append(
                    {
                        "species": species,
                        "segments_path": str(segments_path.resolve()),
                        "split_manifest": str(split_path.resolve()) if split_path else "",
                        "population_scope": scope,
                        "context_length": args.context_length,
                        "studies": args.studies,
                        "bootstrap_replicates": args.boot,
                        "seed": args.seed,
                        "floor": floor,
                        "n_test": n_test,
                        "pool": len(pool),
                        "truth_p90_m": round(truth, 1),
                        "median_ci_width_m": round(median_width, 1),
                        "relative_width": round(median_width / truth, 3),
                        "coverage": round(coverage, 3),
                        "point_sd_m": round(float(np.std(points)), 1),
                    }
                )

                print(
                    f"  floor {floor:3d}  "
                    f"n={n_test:3d}  "
                    f"pool {len(pool):4d}  "
                    f"truth p90 {truth / 1000:7.2f} km  "
                    f"CI width {median_width / 1000:6.2f} km "
                    f"({median_width / truth:5.1%})  "
                    f"coverage {coverage:5.1%}"
                )

    if not rows:
        raise SystemExit("No simulations could be run with the requested settings")

    table = pd.DataFrame(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)

    print(f"\n-> {args.output}")
    print(
        "\nSOURCE DATA ONLY, persistence error, "
        "no model trained and no sealed partition read."
    )
    print(
        "Freeze this result before any held-out species performance is examined."
    )


if __name__ == "__main__":
    main()
