"""
Build the 1 h segments for the 30 rotation birds, with the values the 1 h driver will pin.

Writes data/processed_1h/{bird}_segments.parquet and results/birds_1h_build/. Boundaries and r_h use training individuals (after exclusions) only; 
validation and test are counted, never measured.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.eda.cadence_horizons as ch
import experiments.eda.cadence_survey as cs
from animal_movement_fm.data.build_segments import segments_from_fixes
from animal_movement_fm.data.enu import SCALE_FLOOR_M, enu_steps, window_scale
from animal_movement_fm.data.resample import MIN_SEGMENT_BINS
from animal_movement_fm.data.split import id_key
from experiments.data.build_species_segments import CARRY_COLUMNS
from experiments.data.source_pool_weights import draw_probabilities, rotation_probabilities

OUT_SEGMENTS = REPO / "data" / "processed_1h"
OUT_RESULTS = REPO / "results" / "birds_1h_build"
HORIZON_COUNTS = cs.OUT / "horizon_counts.csv"
REGIME_DECISION = REPO / "registered" / "tier2_regime_decision.json"
GRID_S = ch.GRID_S
MAX_GAP_BINS = ch.SPLIT_BINS  # 48 bins, two days
HORIZONS = ch.HORIZONS
CONTEXT = ch.CONTEXT
CHECKED = ("individuals", "ge1", "geF", "ge_equal_time", "windows")


def output_dirs(subset: bool) -> tuple[Path, Path]:
    """
    Segments and results folders; a --species run writes under subset/, never over the full build.
    """
    return (OUT_SEGMENTS / "subset", OUT_RESULTS / "subset") if subset else (OUT_SEGMENTS, OUT_RESULTS)


def build_bird(bird: str) -> pd.DataFrame:
    """
    One bird's 1 h segments, every individual in its fixes file, in the 6 h builder's batches and columns.
    """
    path = cs.PROCESSED / f"{bird}_fixes.parquet"
    con = duckdb.connect()
    ids = con.execute(f"SELECT DISTINCT individual_id FROM read_parquet('{path}') ORDER BY 1").df()[
        "individual_id"
    ]

    frames = []
    for start in range(0, len(ids), cs.BATCH):
        fixes = con.execute(
            f"SELECT * FROM read_parquet('{path}') WHERE individual_id IN (SELECT unnest($ids))",
            {"ids": ids[start : start + cs.BATCH].tolist()},
        ).df()
        segments, _ = segments_from_fixes(
            fixes,
            bin_seconds=GRID_S,
            min_segment_bins=MIN_SEGMENT_BINS,
            max_gap_bins=MAX_GAP_BINS,
            carry_columns=CARRY_COLUMNS,
        )
        if segments is not None:
            frames.append(segments)

    return pd.concat(frames, ignore_index=True)


def segment_arrays(segments: pd.DataFrame):
    """
    (individual key, bin_index, lon, lat) per segment, in bin order.
    """
    for _, segment in segments.sort_values(["segment_id", "bin_index"]).groupby("segment_id", sort=False):
        yield (
            id_key(segment["individual_id"].iloc[0]),
            segment["bin_index"].to_numpy(),
            segment["lon"].to_numpy(),
            segment["lat"].to_numpy(),
        )


def target_index(bins: np.ndarray, anchors: np.ndarray, h: int) -> np.ndarray:
    """
    Position of bin anchor + h in `bins`, or -1 where it is empty.
    """
    targets = np.minimum(np.searchsorted(bins, bins[anchors] + h), len(bins) - 1)
    return np.where(bins[targets] == bins[anchors] + h, targets, -1)


def pair_displacements(bins: np.ndarray, lon: np.ndarray, lat: np.ndarray, h: int) -> np.ndarray:
    """
    Metres between every occupied pair exactly h bins apart as the targets are measured.
    """
    targets = target_index(bins, np.arange(len(bins)), h)
    pairs = np.stack([np.flatnonzero(targets >= 0), targets[targets >= 0]], axis=1)
    return np.linalg.norm(enu_steps(lon[pairs], lat[pairs])[:, 0, :], axis=-1)


def scaled_targets(bins: np.ndarray, lon: np.ndarray, lat: np.ndarray, h: int) -> np.ndarray:
    """
    r_h = |target| / window scale over a segment's direct rule windows at h.
    """
    anchors = np.flatnonzero(cs.run_position(bins) >= CONTEXT - 1)
    targets = target_index(bins, anchors, h)
    anchors, targets = anchors[targets >= 0], targets[targets >= 0]

    if not len(anchors):
        return np.empty(0)

    context = anchors[:, None] + np.arange(-CONTEXT + 1, 1)[None, :]
    scale = window_scale(enu_steps(lon[context], lat[context]))
    pairs = np.stack([anchors, targets], axis=1)
    return np.linalg.norm(enu_steps(lon[pairs], lat[pairs])[:, 0, :], axis=-1) / scale


def survey_segments(
    bird: str, segments: pd.DataFrame, parts: dict, multiplier: float
) -> tuple[dict, list, list]:
    """
    Window counts per individual from the written segments; boundaries and r_h from training individuals.
    """
    counts = {"raw": {}}
    steps = {h: [] for h in HORIZONS}
    ratios = {h: [] for h in HORIZONS}

    for key, bins, lon, lat in segment_arrays(segments):
        for measure, n in ch.tally(ch.targets_hit(bins, CONTEXT, HORIZONS)).items():
            counts["raw"][key, measure] = counts["raw"].get((key, measure), 0) + n

        # only training individuals contribute boundaries and reference ratios
        if key in parts["train"]:
            for h in HORIZONS:
                steps[h].append(pair_displacements(bins, lon, lat, h))
                ratios[h].append(scaled_targets(bins, lon, lat, h))

    boundaries, ratio_rows = [], []
    for h in HORIZONS:
        displacements, scaled = np.concatenate(steps[h]), np.concatenate(ratios[h])
        p50 = float(np.percentile(displacements, 50))
        boundaries.append(
            {"species": bird, "horizon": h, "pairs": len(displacements), "p50_m": p50}
            | floored(multiplier * p50)
        )
        ratio_rows.append(
            {"species": bird, "horizon": h, "windows": len(scaled), "r_p50": float(np.median(scaled))}
        )

    return counts, boundaries, ratio_rows


def floored(a4_m: float) -> dict:
    """
    The A4 boundary, floored at SCALE_FLOOR_M: below it displacement can't be told from location error (07-10).
    """
    return {"a4_m": a4_m, "boundary_m": max(a4_m, SCALE_FLOOR_M), "floor_binds": a4_m < SCALE_FLOOR_M}


def check_counts(bird: str, counts: dict, parts: dict, expected: pd.DataFrame) -> list[str]:
    """
    Every partition and horizon's counts from the written segments equal horizon_counts.csv.
    """
    from collections import defaultdict

    survey = {"parts": parts, "counts": {"raw": defaultdict(int, counts["raw"]), "split8": defaultdict(int)}}
    built_counts = pd.DataFrame(ch.count_rows(bird, survey)).set_index(["horizon", "partition"])
    expected_counts = expected[expected.species == bird].set_index(["horizon", "partition"])

    failures = [
        f"{h} {p} {c}: {built_counts.loc[(h, p), c]} built, {expected_counts.loc[(h, p), c]} expected"
        for h, p in expected_counts.index
        for c in CHECKED
        if built_counts.loc[(h, p), c] != expected_counts.loc[(h, p), c]
    ]

    for p in cs.PARTITIONS:
        built, expected_windows = (
            built_counts.loc[("any", p), "windows_all_three"],
            expected_counts.loc[("any", p), "windows_all_three"],
        )
        if built != expected_windows:
            failures.append(f"any {p} all three: {built} built, {expected_windows} expected")

    return failures


def fold0_probabilities(folds: dict) -> dict[str, float]:
    """
    A8 on fold 0s registered pool and post exclusion counts by the rule that gave folds 1-9 (checked first).
    """
    recomputed = rotation_probabilities(folds["pools"], folds["train_window_counts"])
    drift = [
        f"{k}.{n}"
        for k, probs in recomputed.items()
        for n, p in probs.items()
        if abs(folds["probabilities"][k][n] - p) > 1e-12
    ]

    if drift or set(recomputed) != set(folds["probabilities"]):
        raise SystemExit(f"FAIL: the rule no longer reproduces the registered folds 1-9: {drift}")

    return draw_probabilities({n: folds["train_window_counts"][n] for n in folds["pools"]["0"]})


def save_results(
    out_results: Path,
    boundaries: list,
    ratio_rows: list,
    fold0: dict,
    multiplier: float,
    started: float,
    summary: dict,
    failures: dict,
) -> None:
    """
    Write boundaries, reference ratios, sampling probabilities and the build summary.
    """
    pd.DataFrame(boundaries).to_csv(out_results / "boundaries.csv", index=False)
    pd.DataFrame(ratio_rows).to_csv(out_results / "r_reference.csv", index=False)
    (out_results / "fold0_probabilities.json").write_text(json.dumps(fold0, indent=2) + "\n")
    (out_results / "build_summary.json").write_text(
        json.dumps(
            {
                "grid_s": GRID_S,
                "max_gap_bins": MAX_GAP_BINS,
                "min_segment_bins": MIN_SEGMENT_BINS,
                "a4_multiplier": multiplier,
                "seconds": round(time.time() - started, 1),
                "birds": summary,
                "failures": {k: v for k, v in failures.items() if v},
            },
            indent=2,
        )
        + "\n"
    )

# GenAI reference: despite me first doing this implementation below for results and main method; I just got GenAI to complete it and that is what is below. 
def print_results(
    out_dirs: tuple[Path, Path], boundaries: list, ratio_rows: list, fold0: dict, failures: dict, started: float
) -> None:
    """Print the travel boundaries, reference ratios and fold 0 probabilities."""
    boundary_table = pd.DataFrame(boundaries).pivot(index="species", columns="horizon", values="boundary_m")
    ratio_table = pd.DataFrame(ratio_rows).pivot(index="species", columns="horizon", values="r_p50")
    print(
        f"\ntravel boundary m (max(0.1 x p50, {SCALE_FLOOR_M:g}), training after exclusions) | median r_h (reference)"
    )
    for bird in boundary_table.index:
        print(
            f"{bird:<26} "
            + " ".join((f"{boundary_table.loc[bird, h]:10.2f}" for h in HORIZONS))
            + "  | "
            + " ".join((f"{ratio_table.loc[bird, h]:7.2f}" for h in HORIZONS))
        )

    print("\nfold 0 A8 probabilities (6 h post-exclusion counts)")
    for name, p in sorted(fold0.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<26} {p:.6f}")

    passed = sum(not v for v in failures.values())
    print(
        f"\nwrote {out_dirs[0].relative_to(REPO)}/ and {out_dirs[1].relative_to(REPO)}/  "
        f"({passed} of {len(failures)} pass, {time.time() - started:.0f} s)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="*", help="a subset of the 30 birds (default: all)")
    args = parser.parse_args()

    folds = json.loads(cs.FOLDS.read_text())
    results = json.loads(cs.ROTATION_RESULTS.read_text())
    decision = json.loads(REGIME_DECISION.read_text())
    if int(decision["anchor_quantile"]) != 50:
        raise SystemExit("FAIL: the frozen A4 rule is no longer anchored on p50")
    multiplier = float(decision["multiplier"])
    expected = pd.read_csv(HORIZON_COUNTS, dtype={"horizon": str})
    birds = args.species or folds["birds"]

    out_segments, out_results = output_dirs(bool(args.species))
    out_segments.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)
    started = time.time()
    boundaries, ratio_rows, summary, failures = [], [], {}, {}

    for bird in birds:
        bird_started = time.time()
        segments = build_bird(bird)
        path = out_segments / f"{bird}_segments.parquet"
        segments.to_parquet(path, index=False)
        parts = cs.partitions(bird, folds, results)
        counts, bird_boundaries, bird_ratios = survey_segments(bird, pd.read_parquet(path), parts, multiplier)
        boundaries += bird_boundaries
        ratio_rows += bird_ratios
        failures[bird] = check_counts(bird, counts, parts, expected)
        summary[bird] = {
            "rows": len(segments),
            "segments": int(segments["segment_id"].nunique()),
            "individuals": int(segments["individual_id"].nunique()),
            "seconds": round(time.time() - bird_started, 1),
        }
        status = "PASS" if not failures[bird] else "FAIL: " + "; ".join(failures[bird][:3])
        print(
            f"{bird:<26} {summary[bird]['rows']:>10,} rows {summary[bird]['seconds']:7.1f} s  {status}",
            flush=True,
        )

    fold0 = fold0_probabilities(folds)
    save_results(out_results, boundaries, ratio_rows, fold0, multiplier, started, summary, failures)
    print_results((out_segments, out_results), boundaries, ratio_rows, fold0, failures, started)

    if any(failures.values()):
        raise SystemExit("FAIL: built windows differ from horizon_counts.csv; see above")


if __name__ == "__main__":
    main()
