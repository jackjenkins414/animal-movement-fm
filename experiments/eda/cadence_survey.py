# GenAI reference: As with other experiment scripts, I got GenAI to write this from my spec; being survey the 30 rotation
# birds across 1/2/3/6 h grids, using ~3 day contexts and C=12, with stats from training birds only and just counts for
# val/test. It also checks the 6 h setup, phase shift sensitivity, step lengths vs the 30 m floor, and misleading grid
# diagnostics. I gave it the real segment builder and windowing code among other files to base this off. 
"""
Cadence survey of the 30 flying birds (descriptive, no models).
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from animal_movement_fm.analysis.regime_profile import local_en_metres  # noqa: E402
from animal_movement_fm.data.build_segments import segments_from_fixes  # noqa: E402
from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import MIN_SEGMENT_BINS, _to_bins  # noqa: E402
from animal_movement_fm.data.split import id_key, load_split, window_counts_per_individual  # noqa: E402
from experiments.data.source_pool_weights import PROCESSED, SPLIT_STEM, SPLITS  # noqa: E402
from experiments.data.species_eligibility import EVALUATION_FLOOR  # noqa: E402

FOLDS = REPO / "registered" / "birds_rotation_folds.json"
ROTATION_RESULTS = REPO / "results" / "birds_rotation_sealed" / "birds_rotation_results.json"
ROTATION_MANIFEST = REPO / "results" / "birds_rotation_sealed" / "birds_rotation_sealed_manifest.json"
OUT = REPO / "results" / "cadence"

GRIDS_H = (1, 2, 3, 6)
CONTEXT_3D = {1: 72, 2: 36, 3: 24, 6: 12}  # about 3 days of history at each grid
CONTEXT_FIXED = 12
PHASES = ("edge", "centred")  # edge: bins start on whole hours (the pipeline); centred: a sensitivity
FLOOR = EVALUATION_FLOOR
EQUAL_TIME_H = FLOOR * 6  # F windows at 6 h, in hours: the floor held at equal time across grids
THRESHOLDS_H = (1, 2, 3, 6)
PHASE_HOLE_BINS = 1.5  # a run break whose raw interval across the hole is at most this many bins
DT_BAND = (0.5, 1.5)  # adjacent-bin steps whose real elapsed time falls outside this band, in bins
LONG_GAP_H = 3
NIGHT_H = (20, 4)  # local solar hours, wrapping midnight
DAY_H = (8, 16)
MIN_MONTH_INTERVALS = 50
SCHEDULE_RATIO = 2.0
BATCH = 50
PARTITIONS = ("train", "val", "test", "test_scored")

# Reference palette: categorical slot 1, blue ordinal ramp (steps 250/400/550/700), recessive neutrals.
BLUE = "#2a78d6"
GRID_COLOURS = {1: "#86b6ef", 2: "#3987e5", 3: "#1c5cab", 6: "#0d366b"}
SURFACE, INK, INK_2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"


# ---------------------------------------------------------------------------------------------------------------------
# Runs and windows


def run_lengths(bin_index: np.ndarray) -> np.ndarray:
    """Lengths of the one-bin contiguous runs in a sorted array of occupied bins."""
    if len(bin_index) == 0:
        return np.empty(0, dtype=np.int64)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(bin_index) != 1) + 1, [len(bin_index)]))
    return np.diff(bounds)


def run_position(bin_index: np.ndarray) -> np.ndarray:
    """Each bin's position inside its run (0 at the run start)."""
    starts = np.concatenate(([True], np.diff(bin_index) != 1))
    run_start = np.maximum.accumulate(np.where(starts, np.arange(len(bin_index)), 0))
    return np.arange(len(bin_index)) - run_start


def windows(lengths: np.ndarray, context: int) -> int:
    """Windows a set of runs yields at `context`: a run of L bins gives L - C, the closed form of `iter_windows`."""
    return int(np.maximum(lengths - context, 0).sum())


def usable_hours(lengths: np.ndarray, context: int, grid_h: int) -> float:
    """Hours inside runs long enough to hold at least one window; comparable across grids."""
    return float(lengths[lengths > context].sum() * grid_h)


def segment_windows(segments: pd.DataFrame, contexts: tuple[int, ...]) -> pd.DataFrame:
    """Windows per individual from a segments frame, counting one-bin runs inside each segment."""
    bins = segments["bin_index"].to_numpy()
    seg = segments["segment_id"].to_numpy()
    new_run = np.ones(len(bins), dtype=bool)
    new_run[1:] = (seg[1:] != seg[:-1]) | (np.diff(bins) != 1)
    run_id = np.cumsum(new_run) - 1
    lengths = np.bincount(run_id)
    individual = segments["individual_id"].to_numpy()[new_run]
    out = pd.DataFrame({"key": [id_key(i) for i in individual]})
    for c in contexts:
        out[c] = np.maximum(lengths - c, 0)
    return out.groupby("key").sum()


# ---------------------------------------------------------------------------------------------------------------------
# Raw-fix diagnostics


def phase_holes(epoch_s: np.ndarray, bin_index_raw: np.ndarray, grid_s: float) -> tuple[int, int]:
    """(run breaks, breaks a tag did not miss): the raw interval across the empty bin is at most PHASE_HOLE_BINS bins."""
    occupied, first = np.unique(bin_index_raw, return_index=True)
    last = np.concatenate((first[1:] - 1, [len(bin_index_raw) - 1]))
    breaks = np.flatnonzero(np.diff(occupied) > 1)
    across = epoch_s[first[breaks + 1]] - epoch_s[last[breaks]]
    return len(breaks), int((across <= PHASE_HOLE_BINS * grid_s).sum())


def solar_hour(epoch_s: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return (epoch_s / 3600.0 + lon / 15.0) % 24.0


def in_hours(hour: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    lo, hi = span
    return (hour >= lo) & (hour < hi) if lo < hi else (hour >= lo) | (hour < hi)


def schedule_changed(months: np.ndarray, intervals: np.ndarray) -> float:
    """1.0 if monthly median intervals differ by SCHEDULE_RATIO or more, 0.0 if not, NaN with < 2 qualifying months."""
    frame = pd.DataFrame({"m": months, "dt": intervals})
    sizes = frame.groupby("m")["dt"].size()
    medians = frame.groupby("m")["dt"].median()[sizes >= MIN_MONTH_INTERVALS]
    if len(medians) < 2:
        return np.nan
    return float(medians.max() / medians.min() >= SCHEDULE_RATIO)


def interval_summary(intervals_s: np.ndarray) -> dict:
    h = intervals_s / 3600.0
    q25, q50, q75 = np.quantile(h, (0.25, 0.5, 0.75))
    out = {"n_intervals": len(h), "median_h": q50, "q25_h": q25, "q75_h": q75}
    out |= {f"share_le_{t}h": float((h <= t).mean()) for t in THRESHOLDS_H}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Inputs


def partitions(bird: str, folds: dict, results: dict) -> dict[str, set[str]]:
    """Partition id_keys: training minus the registered exclusions; test_scored is test minus the rotation's rule drops."""
    split = load_split(SPLITS / f"{SPLIT_STEM.get(bird, bird)}_split_seed42.json")
    excluded = {id_key(i) for i in folds["training_exclusions"].get(bird) or []}
    dropped = {id_key(i) for i in results["scoring_rules"][bird]["dropped"]}
    test = {id_key(i) for i in split.test}
    return {
        "train": {id_key(i) for i in split.train} - excluded,
        "val": {id_key(i) for i in split.val},
        "test": test,
        "test_scored": test - dropped,
    }


def read_fixes(path: Path, ids: list) -> pd.DataFrame:
    con = duckdb.connect()
    names = set(con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 0").df().columns)
    cols = [c for c in ("individual_id", "deployment_id", "study_id", "timestamp", "lat", "lon") if c in names]
    return con.execute(
        f"SELECT {', '.join(cols)} FROM read_parquet('{path}') WHERE individual_id IN (SELECT unnest($ids))",
        {"ids": ids},
    ).df()


def tracks(fixes: pd.DataFrame):
    """(individual key, study, epoch seconds, timestamps, lat, lon) per track, grouped as `segments_from_fixes` does."""
    group = ["individual_id"]
    if "deployment_id" in fixes.columns and fixes["deployment_id"].notna().any():
        group.append("deployment_id")
    ordered = fixes.sort_values(group + ["timestamp"])
    for keys, t in ordered.groupby(group, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ts = t["timestamp"].to_numpy().astype("datetime64[us]")
        study = t["study_id"].iloc[0] if "study_id" in t.columns else None
        yield (id_key(keys[0]), study, ts.astype(np.int64) / 1e6, ts,
               t["lat"].to_numpy(dtype=float), t["lon"].to_numpy(dtype=float))


# ---------------------------------------------------------------------------------------------------------------------
# One species


def contexts_for(grid_h: int) -> tuple[int, ...]:
    return tuple(sorted({CONTEXT_3D[grid_h], CONTEXT_FIXED}))


def survey_species(bird: str, folds: dict, results: dict) -> dict:
    parts = partitions(bird, folds, results)
    assigned = set().union(*parts.values())
    path = PROCESSED / f"{bird}_fixes.parquet"
    raw_ids = duckdb.sql(f"SELECT DISTINCT individual_id FROM read_parquet('{path}') ORDER BY 1").df()["individual_id"]
    raw_ids = raw_ids.tolist()

    counts = defaultdict(float)  # (key, grid, phase, context, measure) -> value
    seg_counts = {g: [] for g in GRIDS_H}
    rebuilt_6h = []
    train_ind, pooled_dt = [], []
    diag = defaultdict(float)  # (grid, measure) -> value, training only
    steps = {g: [] for g in GRIDS_H}
    step_medians = {g: [] for g in GRIDS_H}
    months = {g: np.zeros(12) for g in GRIDS_H}
    study_of = {}

    for start in range(0, len(raw_ids), BATCH):
        fixes = read_fixes(path, raw_ids[start:start + BATCH])
        for g in GRIDS_H:
            segments, _ = segments_from_fixes(fixes, bin_seconds=g * 3600, min_segment_bins=MIN_SEGMENT_BINS)
            if segments is None:
                continue
            seg_counts[g].append(segment_windows(segments, contexts_for(g)))
            if g == 6:
                rebuilt_6h.append(segments)

        per_ind = defaultdict(lambda: defaultdict(list))
        for key, study, epoch_s, ts, lat, lon in tracks(fixes):
            if key not in assigned:
                continue
            is_train = key in parts["train"]
            for g in GRIDS_H:
                grid_s = g * 3600
                for phase in PHASES:
                    shift = grid_s // 2 if phase == "centred" else 0
                    shifted = ts + np.timedelta64(shift, "s")
                    bins, rep_ts, rep_lat, rep_lon, _ = _to_bins(shifted, lat, lon, bin_seconds=grid_s)
                    lengths = run_lengths(bins)
                    for c in contexts_for(g):
                        counts[(key, g, phase, c, "windows")] += windows(lengths, c)
                        counts[(key, g, phase, c, "hours")] += usable_hours(lengths, c, g)
                    if not is_train:
                        continue
                    n_breaks, n_holes = phase_holes(epoch_s, np.floor((epoch_s + shift) / grid_s), grid_s)
                    diag[(g, f"breaks_{phase}")] += n_breaks
                    diag[(g, f"holes_{phase}")] += n_holes
                    if phase == "centred":
                        continue
                    adjacent = np.diff(bins) == 1
                    rep_epoch = rep_ts.astype("datetime64[us]").astype(np.int64) / 1e6
                    dt_bins = np.diff(rep_epoch)[adjacent] / grid_s
                    diag[(g, "adjacent")] += len(dt_bins)
                    diag[(g, "dt_off")] += int(((dt_bins < DT_BAND[0]) | (dt_bins > DT_BAND[1])).sum())
                    east, north = local_en_metres(rep_lon, rep_lat)
                    per_ind[key][f"steps_{g}"].append(np.hypot(east, north)[adjacent].astype(np.float32))
                    target = run_position(bins) >= CONTEXT_FIXED
                    months[g] += np.bincount(rep_ts[target].astype("datetime64[M]").astype(np.int64) % 12,
                                             minlength=12)
            if is_train:
                dt = np.diff(epoch_s)
                hour = solar_hour(epoch_s, lon)
                study_of[key] = study
                per_ind[key]["dt"].append(dt)
                per_ind[key]["month"].append(ts[:-1].astype("datetime64[M]").astype(np.int64))
                per_ind[key]["night"].append(np.array([in_hours(hour, NIGHT_H).sum()]))
                per_ind[key]["day"].append(np.array([in_hours(hour, DAY_H).sum()]))
                long_gap = dt > LONG_GAP_H * 3600
                per_ind[key]["long"].append(np.array([long_gap.sum()]))
                per_ind[key]["long_night"].append(np.array([(long_gap & in_hours(hour[:-1], NIGHT_H)).sum()]))

        for key, acc in per_ind.items():
            dt = np.concatenate(acc["dt"])
            if len(dt) == 0:
                continue
            pooled_dt.append(dt.astype(np.float32))
            day = np.concatenate(acc["day"]).sum()
            train_ind.append({
                "species": bird, "individual": key, "study_id": study_of[key],
                **interval_summary(dt),
                "night_fixes": int(np.concatenate(acc["night"]).sum()), "day_fixes": int(day),
                "long_gaps": int(np.concatenate(acc["long"]).sum()),
                "long_gaps_night": int(np.concatenate(acc["long_night"]).sum()),
                "schedule_change": schedule_changed(np.concatenate(acc["month"]), dt),
            })
            for g in GRIDS_H:
                s = np.concatenate(acc[f"steps_{g}"]) if acc[f"steps_{g}"] else np.empty(0, np.float32)
                steps[g].append(s)
                if len(s):
                    step_medians[g].append(float(np.median(s)))
        del fixes

    return {
        "parts": parts, "counts": counts, "seg_counts": seg_counts, "rebuilt_6h": pd.concat(rebuilt_6h),
        "train_ind": pd.DataFrame(train_ind), "pooled_dt": np.concatenate(pooled_dt), "diag": diag,
        "steps": {g: np.concatenate(v) for g, v in steps.items()}, "step_medians": step_medians, "months": months,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Tables


def per_individual(counts: dict, grid_h: int, phase: str, context: int, measure: str) -> dict[str, float]:
    return {k[0]: v for k, v in counts.items() if k[1:] == (grid_h, phase, context, measure)}


def count_rows(bird: str, survey: dict) -> list[dict]:
    rows = []
    for g in GRIDS_H:
        for phase in PHASES:
            # at 6 h the two contexts coincide, so the row appears under both kinds
            for kind, c in (("3d", CONTEXT_3D[g]), ("c12", CONTEXT_FIXED)):
                n_win = per_individual(survey["counts"], g, phase, c, "windows")
                hours = per_individual(survey["counts"], g, phase, c, "hours")
                equal_time = EQUAL_TIME_H / g
                for part in PARTITIONS:
                    ids = survey["parts"][part]
                    w = np.array([n_win.get(i, 0) for i in ids])
                    rows.append({
                        "species": bird, "grid_h": g, "phase": phase, "context": c,
                        "context_kind": kind, "partition": part,
                        "individuals": len(ids), "ge1": int((w >= 1).sum()), "geF": int((w >= FLOOR).sum()),
                        "ge_equal_time": int((w >= equal_time).sum()), "windows": int(w.sum()),
                        "usable_hours": float(sum(hours.get(i, 0) for i in ids)),
                    })
    return rows


def interval_row(bird: str, survey: dict) -> dict:
    ind = survey["train_ind"]
    medians = ind["median_h"]
    flagged = ind["schedule_change"].dropna()
    return {
        "species": bird, "train_individuals": len(ind),
        **interval_summary(survey["pooled_dt"].astype(np.float64)),
        "ind_median_p10_h": medians.quantile(0.1), "ind_median_p50_h": medians.median(),
        "ind_median_p90_h": medians.quantile(0.9),
        **{f"ind_share_median_le_{t}h": float((medians <= t).mean()) for t in THRESHOLDS_H},
        "night_day_ratio": ind["night_fixes"].sum() / ind["day_fixes"].sum(),
        "night_day_ratio_ind_median": (ind["night_fixes"] / ind["day_fixes"].replace(0, np.nan)).median(),
        "long_gap_night_share": ind["long_gaps_night"].sum() / max(ind["long_gaps"].sum(), 1),
        "schedule_change_share": flagged.mean() if len(flagged) else np.nan,
        "schedule_change_n": len(flagged),
    }


def diagnostic_rows(bird: str, survey: dict) -> list[dict]:
    d, train = survey["diag"], survey["parts"]["train"]
    base_month = survey["months"][6] / survey["months"][6].sum()
    win_6h = per_individual(survey["counts"], 6, "edge", CONTEXT_FIXED, "windows")
    base = {i for i in train if win_6h.get(i, 0) >= 1}
    study = survey["train_ind"].set_index("individual")["study_id"]
    rows = []
    for g in GRIDS_H:
        s = survey["steps"][g]
        month = survey["months"][g] / max(survey["months"][g].sum(), 1)
        row = {
            "species": bird, "grid_h": g,
            "phase_hole_share_edge": d[(g, "holes_edge")] / max(d[(g, "breaks_edge")], 1),
            "phase_hole_share_centred": d[(g, "holes_centred")] / max(d[(g, "breaks_centred")], 1),
            "breaks_edge": int(d[(g, "breaks_edge")]),
            "dt_off_share": d[(g, "dt_off")] / max(d[(g, "adjacent")], 1),
            "step_median_m": float(np.median(s)) if len(s) else np.nan,
            "step_median_ind_median_m": float(np.median(survey["step_medians"][g])) if survey["step_medians"][g]
            else np.nan,
            "step_share_below_floor": float((s < SCALE_FLOOR_M).mean()) if len(s) else np.nan,
            "month_tvd_vs_6h": 0.5 * float(np.abs(month - base_month).sum()),
        }
        for kind, c in (("c12", CONTEXT_FIXED), ("3d", CONTEXT_3D[g])):
            w = per_individual(survey["counts"], g, "edge", c, "windows")
            kept = {i for i in train if w.get(i, 0) >= 1}
            row[f"n_studies_{kind}"] = study.reindex(sorted(kept)).nunique()
            row[f"retained_share_{kind}"] = len(kept & base) / max(len(base), 1)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Checks


def check_species(bird: str, survey: dict, folds: dict, manifest: dict) -> list[str]:
    """The 6 h reproduction and the gap invariance; returns failures, empty when everything holds."""
    failures = []
    cols = ["segment_id", "bin_index", "timestamp", "lat", "lon", "centre_offset_s"]
    on_disk_path = PROCESSED / f"{bird}_segments.parquet"
    on_disk = pd.read_parquet(on_disk_path, columns=cols).sort_values(["segment_id", "bin_index"])
    rebuilt = survey["rebuilt_6h"][cols].sort_values(["segment_id", "bin_index"])
    if not rebuilt.reset_index(drop=True).equals(on_disk.reset_index(drop=True)):
        failures.append(f"6 h rebuild differs from {on_disk_path.name} ({len(rebuilt):,} vs {len(on_disk):,} rows)")

    disk = window_counts_per_individual(str(on_disk_path), context_length=CONTEXT_FIXED)
    disk = dict(zip(disk["individual_id"].map(id_key), disk["n_windows"]))
    ours = per_individual(survey["counts"], 6, "edge", CONTEXT_FIXED, "windows")
    assigned = set().union(*survey["parts"].values())
    bad = [i for i in assigned if disk.get(i, 0) != ours.get(i, 0)]
    if bad:
        failures.append(f"6 h C = 12 windows differ from window_counts_per_individual for {len(bad)} individuals")

    for g in GRIDS_H:
        frames = survey["seg_counts"][g]
        seg = pd.concat(frames).groupby(level=0).sum() if frames else pd.DataFrame()
        for c in contexts_for(g):
            raw = per_individual(survey["counts"], g, "edge", c, "windows")
            if any(raw.get(i, 0) != (seg[c].get(i, 0) if c in seg else 0) for i in assigned):
                failures.append(f"gap rule changes windows at {g} h, C = {c}")

    train_total = sum(ours.get(i, 0) for i in survey["parts"]["train"])
    if train_total != folds["train_window_counts"][bird]:
        failures.append(f"training windows {train_total:,} != registered {folds['train_window_counts'][bird]:,}")

    below = sorted(i for i in survey["parts"]["test_scored"] if ours.get(i, 0) < FLOOR)
    expected_below = sorted(id_key(i) for i in manifest["below_floor"].get(bird, []))
    expected_scored = manifest["eligibility"][bird]["scored"] - len(expected_below)
    scored = len(survey["parts"]["test_scored"]) - len(below)
    if below != expected_below or scored != expected_scored:
        failures.append(f"test at F = {FLOOR}: {scored} scored, below {below}; rotation: {expected_scored}, "
                        f"below {expected_below}")
    return failures


# ---------------------------------------------------------------------------------------------------------------------
# Outputs


def figure(intervals: pd.DataFrame, individuals: pd.DataFrame, counts: pd.DataFrame, path: Path) -> None:
    order = intervals.sort_values("ind_median_p50_h")["species"].tolist()
    label = {s: f"{s.split('_')[0][0].upper()}. {s.split('_')[1]}" for s in order}
    y = {s: i for i, s in enumerate(order)}
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 0.32 * len(order) + 2.2), sharey=True, layout="constrained",
                                      gridspec_kw={"width_ratios": (1.4, 1)}, facecolor=SURFACE)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=INK_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(RULE)

    jitter = np.random.default_rng(42).uniform(-0.25, 0.25, len(individuals))
    left.scatter(individuals["median_h"], individuals["species"].map(y) + jitter, s=8, color=BLUE, alpha=0.5,
                 linewidths=0)
    for t in GRIDS_H:
        left.axvline(t, color=RULE, lw=1, zorder=0)
    left.set_xscale("log")
    top = left.secondary_xaxis("top")
    top.set_xticks(GRIDS_H, [f"{t} h" for t in GRIDS_H])
    top.tick_params(colors=INK_2, labelsize=8, length=0)
    top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    labels = top.get_xticklabels()  # 2 h and 3 h sit close on a log axis: push them apart
    labels[GRIDS_H.index(2)].set_horizontalalignment("right")
    labels[GRIDS_H.index(3)].set_horizontalalignment("left")
    top.spines["top"].set_visible(False)
    left.set_ylim(len(order) - 0.4, -0.6)  # inverted, so rows run top-down in the md table's order
    left.set_yticks(range(len(order)), [label[s] for s in order], fontstyle="italic")
    left.set_xlabel("median fix interval per training individual (h, log scale)", color=INK_2, fontsize=9)

    train = counts[(counts["partition"] == "train") & (counts["phase"] == "edge") & (counts["context_kind"] == "3d")]
    base = train[train["grid_h"] == 6].set_index("species")["ge1"]
    offset = dict(zip(GRIDS_H, np.linspace(-0.24, 0.24, len(GRIDS_H))))  # one row per bird, a lane per grid
    for g in GRIDS_H:
        share = train[train["grid_h"] == g].set_index("species")["ge1"] / base
        right.scatter(share.reindex(order), np.arange(len(order)) + offset[g], s=24, color=GRID_COLOURS[g],
                      label=f"{g} h", edgecolors=SURFACE, linewidths=1, zorder=3)
    right.set_xlim(-0.03, 1.08)
    right.set_xlabel("training individuals with ≥ 1 window at a 3-day context,\nas a share of those at 6 h",
                     color=INK_2, fontsize=9)
    right.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(GRIDS_H),
                 labelcolor=INK_2, title="grid", title_fontsize=8)
    right.grid(axis="x", color=RULE, lw=0.6)
    fig.suptitle("Fix cadence and survival on finer grids, 30 flying birds (training individuals)", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fmt(x: float, digits: int = 2) -> str:
    return "—" if pd.isna(x) else f"{x:.{digits}f}"


def sig(x: float) -> str:
    """Three significant figures; pooled intervals run from 1 s bursts to days."""
    return "—" if pd.isna(x) else f"{x:.3g}"


def markdown(intervals: pd.DataFrame, counts: pd.DataFrame, diagnostics: pd.DataFrame, checks: dict) -> str:
    lines = [
        "# Cadence survey, 30 flying birds (descriptive)",
        "",
        "`experiments/eda/cadence_survey.py`. Statistics from training individuals only (training split minus the "
        "registered exclusions); validation and test are counts only. Not a registered result; no model is run.",
        "",
        f"Checks (6 h reproduction, gap invariance): {sum(not v for v in checks.values())} of {len(checks)} birds pass.",
        "",
        "## Fix intervals (training individuals)",
        "",
        "Pooled over intervals; `ind p50` is the median of individual medians. Night/day: fixes at local solar 20–04 "
        "over 08–16 (1 for an even schedule). Long gaps at night: share of > 3 h intervals starting 20–04 (0.33 if "
        "timing were even). Schedule change: share of individuals whose monthly median interval varies ≥ 2×.",
        "",
        "| bird | ind | median h | IQR h | ≤ 1 h | ≤ 2 h | ≤ 3 h | ≤ 6 h | ind p50 h | night/day | long gaps at night "
        "| schedule change |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in intervals.sort_values("ind_median_p50_h").itertuples():
        lines.append(
            f"| {r.species} | {r.train_individuals} | {sig(r.median_h)} | {sig(r.q25_h)}–{sig(r.q75_h)} "
            f"| {fmt(r.share_le_1h)} | {fmt(r.share_le_2h)} | {fmt(r.share_le_3h)} | {fmt(r.share_le_6h)} "
            f"| {sig(r.ind_median_p50_h)} | {fmt(r.night_day_ratio)} | {fmt(r.long_gap_night_share)} "
            f"| {fmt(r.schedule_change_share)} (n {r.schedule_change_n}) |")

    for kind, title in (("3d", "about 3 days (C = 72 / 36 / 24 / 12)"), ("c12", "C = 12")):
        lines += [
            "", f"## Individuals per partition at {title}, edge phase", "",
            f"train ≥ 1 window / val ≥ F / test_scored ≥ F (F = {FLOOR}); in brackets the equal-time floor "
            f"({EQUAL_TIME_H} h of windows). test_scored is test minus the rotation's val–test and identity drops.",
            "", "| bird | " + " | ".join(f"{g} h" for g in GRIDS_H) + " |", "|---|" + "---:|" * len(GRIDS_H),
        ]
        sub = counts[(counts["phase"] == "edge") & (counts["context_kind"] == kind)].set_index(
            ["species", "grid_h", "partition"])
        for bird in intervals.sort_values("ind_median_p50_h")["species"]:
            cells = []
            for g in GRIDS_H:
                tr, va, te = (sub.loc[(bird, g, p)] for p in ("train", "val", "test_scored"))
                cells.append(f"{tr.ge1} / {va.geF} ({va.ge_equal_time}) / {te.geF} ({te.ge_equal_time})")
            lines.append(f"| {bird} | " + " | ".join(cells) + " |")

    lines += [
        "", "## Grid diagnostics (training individuals, edge phase unless named)", "",
        "Phase holes: share of run breaks where the raw interval across the empty bin is ≤ 1.5 bins (the tag did not "
        "miss). Δt off: adjacent-bin steps whose real elapsed time is outside 0.5–1.5 bins. Month TVD: total-variation "
        "distance of C = 12 window targets' calendar months from 6 h. Retained: share of 6 h-eligible training "
        f"individuals with ≥ 1 window at a 3-day context. Below floor: one-bin steps < {SCALE_FLOOR_M:.0f} m "
        "(`SCALE_FLOOR_M`, `src/animal_movement_fm/data/enu.py`).",
        "", "| bird | grid | holes edge | holes centred | Δt off | step p50 m | below floor | month TVD | studies (3d) "
        "| retained (3d) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in diagnostics.itertuples():
        lines.append(
            f"| {r.species} | {r.grid_h} h | {fmt(r.phase_hole_share_edge)} | {fmt(r.phase_hole_share_centred)} "
            f"| {fmt(r.dt_off_share, 3)} | {fmt(r.step_median_m, 0)} | {fmt(r.step_share_below_floor)} "
            f"| {fmt(r.month_tvd_vs_6h)} | {r.n_studies_3d} | {fmt(r.retained_share_3d)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="*", help="a subset of the 30 birds (default: all)")
    args = parser.parse_args()

    folds = json.loads(FOLDS.read_text())
    results = json.loads(ROTATION_RESULTS.read_text())
    manifest = json.loads(ROTATION_MANIFEST.read_text())
    birds = args.species or folds["birds"]
    unknown = sorted(set(birds) - set(folds["birds"]))
    if unknown:
        raise SystemExit(f"FAIL: not in the rotation's 30 birds: {unknown}")

    OUT.mkdir(parents=True, exist_ok=True)
    intervals, individuals, counts, diagnostics, checks = [], [], [], [], {}
    for bird in birds:
        t0 = time.time()
        survey = survey_species(bird, folds, results)
        checks[bird] = check_species(bird, survey, folds, manifest)
        intervals.append(interval_row(bird, survey))
        individuals.append(survey["train_ind"])
        counts += count_rows(bird, survey)
        diagnostics += diagnostic_rows(bird, survey)
        status = "PASS" if not checks[bird] else "FAIL: " + "; ".join(checks[bird])
        print(f"{bird:<26} {time.time() - t0:7.1f} s  {status}", flush=True)

    intervals, individuals = pd.DataFrame(intervals), pd.concat(individuals, ignore_index=True)
    counts, diagnostics = pd.DataFrame(counts), pd.DataFrame(diagnostics)
    intervals.to_csv(OUT / "cadence_intervals.csv", index=False)
    individuals.to_csv(OUT / "cadence_individuals.csv", index=False)
    counts.to_csv(OUT / "cadence_counts.csv", index=False)
    diagnostics.to_csv(OUT / "cadence_diagnostics.csv", index=False)
    (OUT / "cadence_survey.md").write_text(markdown(intervals, counts, diagnostics, checks))
    figure(intervals, individuals, counts, OUT / "cadence_survey.png")
    print(f"wrote {OUT.relative_to(REPO)}/  ({sum(not v for v in checks.values())} of {len(checks)} pass)")
    if any(checks.values()):
        raise SystemExit("FAIL: a reproduction or invariance check failed; see above")


if __name__ == "__main__":
    main()
