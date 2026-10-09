"""
Horizon counts for the 1 h rebuild (descriptive, no models): bin occupancy only, for every partition.
Deets later. 
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.eda.cadence_survey as cs
from animal_movement_fm.data.resample import MIN_SEGMENT_BINS, _to_bins, resample_trajectory

GRID_H = 1
GRID_S = GRID_H * 3600
CONTEXT = cs.CONTEXT_FIXED
HORIZONS = (1, 6, 24)
SPLIT_BINS = 48  # split segments at two days
CURRENT_SPLIT_BINS = 8  # the 6 h setting becomes eight hours on this grid
EQUAL_TIME = cs.EQUAL_TIME_H / GRID_H  # 150 windows on the 1 h grid
SURVEY_COUNTS = cs.OUT / "cadence_counts.csv"
MEASURES = ("ge1", "geF", "ge_equal_time", "windows")


def targets_hit(bin_index: np.ndarray, context: int, horizons: tuple[int, ...]) -> np.ndarray:
    """
    (anchors, horizons) bool: anchor t ends C one-bin contiguous context bins; True where bin t + h is occupied.
    """
    anchors = bin_index[cs.run_position(bin_index) >= context - 1]
    return np.stack([np.isin(anchors + h, bin_index) for h in horizons], axis=1).reshape(
        len(anchors), len(horizons)
    )


def segment_targets_hit(segments: list, context: int, horizons: tuple[int, ...]) -> np.ndarray:
    """
    As `targets_hit`, with context and target inside one segment.
    """
    hits = [targets_hit(segment.bin_index, context, horizons) for segment in segments]
    return np.concatenate(hits) if hits else np.zeros((0, len(horizons)), dtype=bool)


def tally(hit: np.ndarray) -> dict[str, int]:
    """
    Windows per horizon, anchors with any target and anchors with all of them.
    """
    windows = {h: int(hit[:, j].sum()) for j, h in enumerate(HORIZONS)}
    return windows | {"any": int(hit.any(axis=1).sum()), "all": int(hit.all(axis=1).sum())}


def survey_horizons(bird: str, folds: dict, results: dict) -> dict:
    parts = cs.partitions(bird, folds, results)
    assigned = set().union(*parts.values())
    path = cs.PROCESSED / f"{bird}_fixes.parquet"
    raw_ids = duckdb.sql(f"SELECT DISTINCT individual_id FROM read_parquet('{path}') ORDER BY 1").df()[
        "individual_id"
    ]
    raw_ids = raw_ids.tolist()

    counts = defaultdict(lambda: defaultdict(int))  # rule -> (individual, measure) -> windows
    for start in range(0, len(raw_ids), cs.BATCH):
        fixes = cs.read_fixes(path, raw_ids[start : start + cs.BATCH])
        for key, _, _, ts, lat, lon in cs.tracks(fixes):
            if key not in assigned:
                continue
            bins, *_ = _to_bins(ts, lat, lon, bin_seconds=GRID_S)
            counts["closed_form"][key, 1] += cs.windows(cs.run_lengths(bins), CONTEXT)
            rules = {"raw": targets_hit(bins, CONTEXT, HORIZONS)}
            for name, max_gap in (("split48", SPLIT_BINS), ("split8", CURRENT_SPLIT_BINS)):
                segments = resample_trajectory(
                    ts, lat, lon, bin_seconds=GRID_S, max_gap_bins=max_gap, min_segment_bins=MIN_SEGMENT_BINS
                )
                rules[name] = segment_targets_hit(segments, CONTEXT, HORIZONS)
            for name, hit in rules.items():
                for measure, n in tally(hit).items():
                    counts[name][key, measure] += n
        del fixes

    return {"parts": parts, "counts": counts}


def check_species(bird: str, survey: dict, survey_counts: pd.DataFrame) -> list[str]:
    """
    h = 1 reproduces the survey (per individual and per partition); a 48 bin split changes no window.
    """
    failures = []
    counts, assigned = survey["counts"], set().union(*survey["parts"].values())

    if any((counts["raw"][i, 1] != counts["closed_form"][i, 1] for i in assigned)):
        failures.append("h = 1 differs from the survey's closed form")

    for measure in (*HORIZONS, "any", "all"):
        changed = [i for i in assigned if counts["raw"][i, measure] != counts["split48"][i, measure]]
        if changed:
            failures.append(f"48-bin split changes {measure} for {len(changed)} individuals")

    reference = survey_counts[
        (survey_counts.species == bird)
        & (survey_counts.grid_h == GRID_H)
        & (survey_counts.phase == "edge")
        & (survey_counts.context_kind == "c12")
    ]
    built = {row["partition"]: row for row in count_rows(bird, survey) if row["horizon"] == "1"}
    for _, row in reference.iterrows():
        differences = [m for m in ("individuals", *MEASURES) if built[row.partition][m] != row[m]]
        if differences:
            failures.append(f"h = 1 {row.partition} differs from cadence_counts.csv on {differences}")

    return failures


def count_rows(bird: str, survey: dict) -> list[dict]:
    counts, rows = survey["counts"], []
    for horizon in (*HORIZONS, "any"):
        for partition in cs.PARTITIONS:
            ids = survey["parts"][partition]
            windows = np.array([counts["raw"][i, horizon] for i in ids])
            row = {
                "species": bird,
                "horizon": str(horizon),
                "partition": partition,
                "individuals": len(ids),
                "ge1": int((windows >= 1).sum()),
                "geF": int((windows >= cs.FLOOR).sum()),
                "ge_equal_time": int((windows >= EQUAL_TIME).sum()),
                "windows": int(windows.sum()),
                "windows_split8": sum((counts["split8"][i, horizon] for i in ids)),
            }
            if horizon == "any":
                row["windows_all_three"] = sum((counts["raw"][i, "all"] for i in ids))
            rows.append(row)

    return rows


def eligible_birds(counts: pd.DataFrame, horizon: str, measure: str) -> int:
    individuals = counts[counts.horizon == horizon].pivot(
        index="species", columns="partition", values=measure
    )
    return int(((individuals["val"] >= 5) & (individuals["test_scored"] >= 5)).sum())


def horizon_rows(counts: pd.DataFrame) -> list[str]:
    """
    Format one summary row per horizon.
    """
    lines = []
    for horizon in (*map(str, HORIZONS), "any"):
        totals = counts[counts.horizon == horizon].groupby("partition").sum(numeric_only=True)
        windows = " / ".join((f"{int(totals.loc[p, 'windows']):,}" for p in ("train", "val", "test_scored")))
        all_three = (
            " / ".join(
                (
                    f"{totals.loc[p, 'windows_all_three'] / totals.loc[p, 'windows']:.2f}"
                    for p in ("train", "val", "test_scored")
                )
            )
            if horizon == "any"
            else ""
        )
        all_partitions = totals.loc[["train", "val", "test"]]
        lost = 1 - all_partitions["windows_split8"].sum() / all_partitions["windows"].sum()
        label = f"{horizon} h" if horizon != "any" else "any"
        lines.append(
            f"| {label} | {eligible_birds(counts, horizon, 'geF')} | "
            f"{eligible_birds(counts, horizon, 'ge_equal_time')} | {windows} | {all_three} | {lost:.3f} |"
        )

    return lines


def bird_rows(counts: pd.DataFrame) -> list[str]:
    """
    Format the per bird counts at each horizon.
    """
    lines = []
    for bird in counts.species.unique():
        bird_counts = counts[counts.species == bird].set_index(["horizon", "partition"])
        cells = [
            f"{bird_counts.loc[(h, 'train'), 'ge1']} / "
            f"{bird_counts.loc[(h, 'val'), 'ge_equal_time']} / "
            f"{bird_counts.loc[(h, 'test_scored'), 'ge_equal_time']}"
            for h in (*map(str, HORIZONS), "any")
        ]
        lines.append(f"| {bird} | " + " | ".join(cells) + " |")

    return lines

# GenAI reference: the below is done by GenAI from all my above work
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="*", help="a subset of the 30 birds (default: all)")
    args = parser.parse_args()

    folds = json.loads(cs.FOLDS.read_text())
    results = json.loads(cs.ROTATION_RESULTS.read_text())
    survey_counts = pd.read_csv(SURVEY_COUNTS)
    birds = args.species or folds["birds"]
    unknown = sorted(set(birds) - set(folds["birds"]))
    if unknown:
        raise SystemExit(f"FAIL: not in the rotation's 30 birds: {unknown}")

    rows, checks = [], {}
    for bird in birds:
        started = time.time()
        survey = survey_horizons(bird, folds, results)
        checks[bird] = check_species(bird, survey, survey_counts)
        rows += count_rows(bird, survey)
        status = "PASS" if not checks[bird] else "FAIL: " + "; ".join(checks[bird])
        print(f"{bird:<26} {time.time() - started:7.1f} s  {status}", flush=True)

    counts = pd.DataFrame(rows)
    counts.to_csv(cs.OUT / "horizon_counts.csv", index=False)

    passed = sum(not v for v in checks.values())
    lines = [
        "# Horizon counts at 1 h, C = 12 (descriptive)",
        "",
        (
            "`experiments/eda/cadence_horizons.py`. Direct rule: an anchor t ends 12 one-bin contiguous"
            " context bins; a window at horizon h needs bin t + h occupied, the bins between not "
            "required. Bin occupancy only, for every partition. Edge phase. F = 25; equal time = 150 "
            "windows (150 h). test_scored is test minus the rotation's val–test and identity drops."
        ),
        "",
        f"Checks (h = 1 reproduces the survey; a 48-bin split leaves every count unchanged): {passed} of "
        f"{len(checks)} birds pass.",
        "",
        "## Per horizon",
        "",
        (
            "Birds: ≥ 5 val and ≥ 5 test_scored individuals at the floor. `any`: anchors with at least "
            "one of the three targets; all three: the share of those with every target. Lost to 8-bin "
            "split: share of windows (all partitions) lost if segments split at the current 8 bins (8 h"
            " at 1 h) instead of 48."
        ),
        "",
        (
            "| horizon | birds, F | birds, equal time | windows train / val / test_scored | all three, "
            "train / val / test_scored | lost to 8-bin split |"
        ),
        "|---|---:|---:|---|---|---:|",
    ]

    lines += horizon_rows(counts)
    lines += [
        "",
        "## Per bird",
        "",
        "train ≥ 1 window / val ≥ 150 / test_scored ≥ 150 (equal-time floor).",
        "",
        "| bird | 1 h | 6 h | 24 h | any |",
        "|---|---:|---:|---:|---:|",
    ]

    lines += bird_rows(counts)
    (cs.OUT / "horizon_counts.md").write_text("\n".join(lines) + "\n")

    print(
        f"wrote {cs.OUT.relative_to(REPO)}/horizon_counts.*  ({passed} of "
        f"{len(checks)} pass)"
    )

    if any(checks.values()):
        raise SystemExit("FAIL: a reproduction or split check failed; see above")


if __name__ == "__main__":
    main()
