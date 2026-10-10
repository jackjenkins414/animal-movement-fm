# GenAI reference: I used GenAI for this whole file, but have check it. Specifically I specified the built 1 h segments, for each bird and C = 24, 48, 
# 72, 96, how full the extended history t - C + 1 ... t - 12 is behind the main study's anchors, anchors per occupancy stratum per partition, and the 
# size of extended tokens' scaled steps (each from the previous observed fix) against the core's.
# GenAI reference: on 09-10 I specified the two ablation strata (gappy < 0.9, full >= 0.9) and the gap tokens' p99 under
# the per-bin encoding (step / bins spanned); GenAI added them.
"""
Extended history occupancy for the 1 h context ablation (descriptive, no models).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import experiments.data.build_1h_segments as b1  # noqa: E402
import experiments.eda.cadence_horizons as ch  # noqa: E402
import experiments.eda.cadence_survey as cs  # noqa: E402
from animal_movement_fm.data.enu import EARTH_RADIUS_M, enu_steps, window_scale, wrap_longitude  # noqa: E402

CORE = ch.CONTEXT  # 12 bins, 11 step tokens
C_VALUES = (24, 48, 72, 96)
CUTS = (0.5, 0.9)
STRATA = ("none", "below_0.5", "0.5_to_0.9", "from_0.9")  # 0 | (0, 0.5) | [0.5, 0.9) | [0.9, 1]
COUNTED = ("train", "val", "test_scored")
ABLATION = ("gappy", "full")  # share < 0.9 | >= 0.9, the ablation's two strata (09-10)
EDGES = np.logspace(-4, 7, 2201)  # |scaled step| histogram, about 1.2% per bin, for the p99
OUT_CSV = cs.OUT / "history_occupancy.csv"
OUT_MD = cs.OUT / "history_occupancy.md"


def output_paths(subset: bool) -> tuple[Path, Path]:
    """The .csv and .md; a --species run writes under subset/, never over the full count."""
    return (cs.OUT / "subset" / OUT_CSV.name, cs.OUT / "subset" / OUT_MD.name) if subset else (OUT_CSV, OUT_MD)


# ---------------------------------------------------------------------------------------------------------------------
# Windows on the history grid


def ablation_anchors(bins: np.ndarray) -> np.ndarray:
    """The main study's training anchors: 12 contiguous bins ending at t and at least one target."""
    anchors = np.flatnonzero(cs.run_position(bins) >= CORE - 1)
    return anchors[ch.targets_hit(bins, CORE, ch.HORIZONS).any(axis=1)]


def history_grid(bins: np.ndarray, anchors: np.ndarray, context: int) -> np.ndarray:
    """(anchors, C) positions of bins t - C + 1 ... t in `bins`, oldest first; -1 where the bin is empty."""
    want = bins[anchors][:, None] + np.arange(-context + 1, 1)[None, :]
    j = np.minimum(np.searchsorted(bins, want), len(bins) - 1)
    return np.where(bins[j] == want, j, -1)


def occupancy(grid: np.ndarray) -> np.ndarray:
    """Share of the extended bins t - C + 1 ... t - 12 that are observed."""
    return (grid[:, : grid.shape[1] - CORE] >= 0).mean(axis=1)


def stratum(share: np.ndarray) -> np.ndarray:
    return np.select([share == 0, share < CUTS[0], share < CUTS[1]], [0, 1, 2], 3)


def ablation_strata(counts: np.ndarray) -> dict[str, int]:
    """The four strata's anchor counts folded into the ablation's two: gappy (< 0.9) and full (>= 0.9)."""
    return {"gappy": int(counts[:3].sum()), "full": int(counts[3])}


def core_scale(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """The main study's window scale, from the 12 core positions only."""
    core = grid[:, -CORE:]
    return window_scale(enu_steps(lon[core], lat[core]))


def extended_steps(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray, scale: np.ndarray):
    """(|scaled step|, bins spanned) for observed extended tokens, each step from the previous observed fix.

    Extended tokens end at bins t - C + 2 ... t - 11 (columns 1 ... C - 12); a token is observed when its bin and an
    earlier history bin are. The tangent plane is anchored at t's latitude, as `enu_steps` anchors the core. A span
    above 1 is a gap token; under the per-bin encoding (09-10) its step is divided by the span.
    """
    n, context = grid.shape
    column = np.arange(context)
    last = np.maximum.accumulate(np.where(grid >= 0, column, -1), axis=1)
    previous = np.concatenate((np.full((n, 1), -1), last[:, :-1]), axis=1)
    ends = column[1 : context - CORE + 1]
    valid = (grid[:, ends] >= 0) & (previous[:, ends] >= 0)
    rows = np.broadcast_to(np.arange(n)[:, None], valid.shape)[valid]
    dest = grid[:, ends][valid]
    src = grid[rows, previous[:, ends][valid]]
    anchor_lat = np.radians(lat[grid[rows, -1]])
    east = EARTH_RADIUS_M * np.cos(anchor_lat) * wrap_longitude(np.radians(lon[dest] - lon[src]))
    north = EARTH_RADIUS_M * np.radians(lat[dest] - lat[src])
    span = np.broadcast_to(ends, valid.shape)[valid] - previous[:, ends][valid]
    return np.hypot(east, north) / scale[rows], span


def core_steps(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray, scale: np.ndarray) -> np.ndarray:
    core = grid[:, -CORE:]
    return (np.linalg.norm(enu_steps(lon[core], lat[core]), axis=-1) / scale[:, None]).ravel()


# ---------------------------------------------------------------------------------------------------------------------
# One species


def histogram(values: np.ndarray) -> np.ndarray:
    return np.histogram(np.clip(values, EDGES[0], EDGES[-1]), bins=EDGES)[0]


def p99(hist: np.ndarray) -> float:
    """Upper edge of the bin holding the 99th percentile."""
    cumulative = np.cumsum(hist)
    return float(EDGES[1 + np.searchsorted(cumulative, 0.99 * cumulative[-1])]) if cumulative[-1] else np.nan


def survey_bird(bird: str, folds: dict, results: dict) -> list[dict]:
    parts = cs.partitions(bird, folds, results)
    part_of = {key: p for p in COUNTED for key in parts[p]}
    segments = pd.read_parquet(b1.OUT_SEGMENTS / f"{bird}_segments.parquet",
                               columns=["segment_id", "individual_id", "bin_index", "lon", "lat"])
    strata = {(c, p): np.zeros(len(STRATA), dtype=np.int64) for c in C_VALUES for p in COUNTED}
    shares = {c: [] for c in C_VALUES}
    ext_all = {c: np.zeros(len(EDGES) - 1, dtype=np.int64) for c in C_VALUES}
    ext_gap = {c: np.zeros(len(EDGES) - 1, dtype=np.int64) for c in C_VALUES}
    ext_gap_bin = {c: np.zeros(len(EDGES) - 1, dtype=np.int64) for c in C_VALUES}  # per-bin encoding
    ext_max = {c: 0.0 for c in C_VALUES}
    ext_n = {c: [0, 0] for c in C_VALUES}  # observed extended tokens, of which bridging a gap
    core_hist = np.zeros(len(EDGES) - 1, dtype=np.int64)
    for key, bins, lon, lat in b1.segment_arrays(segments):
        part = part_of.get(key)
        if part is None:
            continue
        anchors = ablation_anchors(bins)
        if not len(anchors):
            continue
        for c in C_VALUES:
            grid = history_grid(bins, anchors, c)
            share = occupancy(grid)
            strata[(c, part)] += np.bincount(stratum(share), minlength=len(STRATA))
            if part != "train":
                continue
            shares[c].append(share.astype(np.float32))
            scale = core_scale(grid, lon, lat)
            if c == C_VALUES[0]:
                core_hist += histogram(core_steps(grid, lon, lat, scale))
            mag, span = extended_steps(grid, lon, lat, scale)
            gap = span > 1
            ext_all[c] += histogram(mag)
            ext_gap[c] += histogram(mag[gap])
            ext_gap_bin[c] += histogram(mag[gap] / span[gap])
            ext_max[c] = max(ext_max[c], float(mag.max(initial=0.0)))
            ext_n[c][0] += len(mag)
            ext_n[c][1] += int(gap.sum())

    rows = []
    for c in C_VALUES:
        s = np.concatenate(shares[c]) if shares[c] else np.empty(0)
        q = np.quantile(s, (0.1, 0.5, 0.9)) if len(s) else (np.nan,) * 3
        for p in COUNTED:
            row = {"species": bird, "context": c, "partition": p, "anchors": int(strata[(c, p)].sum())}
            row |= dict(zip(STRATA, strata[(c, p)].tolist())) | ablation_strata(strata[(c, p)])
            if p == "train":
                row |= {"share_p10": q[0], "share_p50": q[1], "share_p90": q[2],
                        "ext_tokens": ext_n[c][0], "ext_gap_share": ext_n[c][1] / max(ext_n[c][0], 1),
                        "ext_p99": p99(ext_all[c]), "ext_gap_p99": p99(ext_gap[c]),
                        "ext_gap_bin_p99": p99(ext_gap_bin[c]), "ext_max": ext_max[c],
                        "core_p99": p99(core_hist)}
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------------
# Outputs


def check_anchors(table: pd.DataFrame) -> list[str]:
    """Anchors per partition equal the main study's anchors with any target (horizon_counts.csv), at every C."""
    ref = pd.read_csv(ch.cs.OUT / "horizon_counts.csv", dtype={"horizon": str})
    ref = ref[ref.horizon == "any"].set_index(["species", "partition"])["windows"]
    return [f"{r.species} C = {r.context} {r.partition}: {r.anchors} anchors, {ref[(r.species, r.partition)]} expected"
            for r in table.itertuples() if r.anchors != ref[(r.species, r.partition)]]


def markdown(table: pd.DataFrame) -> str:
    train = table[table.partition == "train"]
    lines = [
        "# Extended-history occupancy for the 1 h context ablation (descriptive)", "",
        "`experiments/eda/history_occupancy.py`, on `data/processed_1h/`. Anchors are the main study's (12 contiguous "
        "bins ending at t, at least one target). Share: observed bins among t − C + 1 … t − 12, within the segment. "
        "Training individuals for shares and magnitudes; val and test_scored are anchor counts. Strata: 0 | (0, 0.5) | "
        "[0.5, 0.9) | [0.9, 1]. |scaled step|: extended tokens' step from the previous observed fix over the core's "
        "window scale; gap = the previous observed fix is more than one bin back; gap per bin = the gap token's step "
        "divided by the bins it spans (the 09-10 encoding); core = the 11 core tokens.", "",
        "## Training anchors", "",
        "| bird | C | share p10 / p50 / p90 | strata, share of anchors | gap tokens | p99 ext / gap / gap per bin / "
        "core | max ext |",
        "|---|---:|---|---|---:|---|---:|",
    ]
    for _, r in train.iterrows():
        shares = " / ".join(f"{r[q]:.2f}" for q in ("share_p10", "share_p50", "share_p90"))
        counts = r[list(STRATA)].to_numpy(dtype=float)
        strata = " / ".join(f"{x:.2f}" for x in counts / max(counts.sum(), 1))
        lines.append(f"| {r.species} | {r.context} | {shares} | {strata} | {r.ext_gap_share:.2f} | "
                     f"{r.ext_p99:.1f} / {r.ext_gap_p99:.1f} / {r.ext_gap_bin_p99:.1f} / {r.core_p99:.1f} | "
                     f"{r.ext_max:,.0f} |")
    lines += ["", "## Anchors per stratum (val / test_scored)", "",
              "| bird | C | val: 0 / <0.5 / 0.5–0.9 / ≥0.9 | test_scored: 0 / <0.5 / 0.5–0.9 / ≥0.9 |", "|---|---:|---|---|"]
    held = table.set_index(["species", "context", "partition"])
    for _, r in train.iterrows():
        cells = [" / ".join(f"{int(held.loc[(r.species, r.context, p), s]):,}" for s in STRATA)
                 for p in ("val", "test_scored")]
        lines.append(f"| {r.species} | {r.context} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Ablation strata (gappy < 0.9 / full ≥ 0.9), anchors", "",
              "| bird | C | train | val | test_scored |", "|---|---:|---|---|---|"]
    for _, r in train.iterrows():
        cells = [" / ".join(f"{int(held.loc[(r.species, r.context, p), s]):,}" for s in ABLATION) for p in COUNTED]
        lines.append(f"| {r.species} | {r.context} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", nargs="*", help="a subset of the 30 birds (default: all)")
    args = parser.parse_args()

    folds = json.loads(cs.FOLDS.read_text())
    results = json.loads(cs.ROTATION_RESULTS.read_text())
    birds = args.species or folds["birds"]
    out_csv, out_md = output_paths(bool(args.species))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    started, rows = time.time(), []
    for bird in birds:
        t0 = time.time()
        rows += survey_bird(bird, folds, results)
        print(f"{bird:<26} {time.time() - t0:7.1f} s", flush=True)
    table = pd.DataFrame(rows)
    failures = check_anchors(table)
    table.to_csv(out_csv, index=False)
    out_md.write_text(markdown(table))
    print(f"wrote {out_csv.relative_to(REPO)} and .md  ({time.time() - started:.0f} s)")
    if failures:
        raise SystemExit("FAIL: anchors differ from horizon_counts.csv\n  " + "\n  ".join(failures[:10]))


if __name__ == "__main__":
    main()
