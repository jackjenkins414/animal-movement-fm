# GenAI reference: I knew this was going to be a long file, which I could write (for the most part) but for the sake of time, 
# I got GenAI to produce the initial implementation of this file from my specification of the evaluation protocol; comparator 
# set, A5 weighting and floor, A4 strata, A3 seed aggregation, the bootstrap design and the 4 success bands. I haved added 
# specific comments to its unchanged code; functions without them I have edited or added in myself. 
"""
Sealed Tier 2 evaluation on the three held-out species' test partitions.

`--predict` runs the available comparators once and saves per-window predictions. `--report` reads those files to compute the registered 
metrics, bootstrap intervals, floor sensitivity and success band verdict without reopening the test partitions.

This is the only script that will open held-out test data. It runs only after `preregistration_tier2_freeze.md` exists.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tier2_final_train import (  # the run plan and species tables live in one place # type: ignore
    EVALUATION_FLOOR,
    HELD_OUT,
    SEEDS,
    boundary_m,
    held_out_entry,
)

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import load_split
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.selection import individual_weighted_quantile
from animal_movement_fm.models.averaging import DisplacementAveraging
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.train.loop import TrainConfig

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
FINAL = REPO / "results" / "tier2_final"
SEALED = REPO / "results" / "tier2_sealed"
PREDICTIONS = SEALED / "predictions"
FREEZE = REPO / "preregistration_tier2_freeze.md"

# registered evaluation statistics and A5 floor sensitivity
QUANTILES = {"p50": 0.50, "p90": 0.90, "p99": 0.99}
FLOOR_SENSITIVITY = (10, 25, 50, 100)
STRATA = ("travel", "low_displacement")

# reported statistics per comparator and stratum; added mean for tft paper comparison 
STATISTICS = (
    "mean_m_individual_weighted",
    "p50_m",
    "p90_m",
    "p99_m_individual_weighted",
    "mean_m_pooled",
    "p99_m_pooled",
)

# bootstrap the registered travel p90 and the tft mean
BOOTSTRAP_SERIES = (("travel", "p90"), ("all", "mean"))

# registered paired bootstrap over test individuals
N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 42

# 4; Z is within 25% of the specialist on travel regime p90
SPECIALIST_MULTIPLIER = 1.25

UNTRAINED = ("P", "V")
TRAINED = ("Z", "B", "S", "Za")
SECONDARY_ONLY = ("F",)

# registered comparator order for printed and written tables
COMPARATOR_ORDER = ("P", "V", "Z", "Za", "B", "S", "F")


# Seal checks
# GenAI reference: assisted with implementing the git backed freeze/attestation checks from my requirement that sealed evaluation 
# cannot begin before the committed freeze.
def assert_freeze_committed() -> dict:
    """
    Verify that the preregistration freeze is committed before opening test data.
    
    The freeze must exist, be tracked by git and have no uncommitted changes. The returned commit and blob
    identifiers are recorded in the sealed manifest so results can be tied to the exact frozen version.
    """
    if not FREEZE.exists():
        raise SystemExit(
            f"FAIL: {FREEZE.name} does not exist. 2 requires the pre-test freeze to be written and "
            "committed before the first sealed evaluation. Nothing here runs until it is."
        )

    def git(*args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True
        )

    if git("rev-parse", "--git-dir").returncode != 0:
        raise SystemExit("FAIL: not a git checkout, so the freeze cannot be shown to be committed")
    if git("ls-files", "--error-unmatch", FREEZE.name).returncode != 0:
        raise SystemExit(
            f"FAIL: {FREEZE.name} exists but is not tracked by git, so it is not committed."
        )
    if git("diff", "--quiet", "HEAD", "--", FREEZE.name).returncode != 0:
        raise SystemExit(
            f"FAIL: {FREEZE.name} has uncommitted changes. Commit it before any sealed partition "
            "is opened; an editable freeze is not a freeze."
        )

    commit = git("rev-parse", "HEAD").stdout.strip()
    blob = git("rev-parse", f"HEAD:{FREEZE.name}").stdout.strip()
    print(f"freeze           : {FREEZE.name} committed at {commit[:12]} (blob {blob[:12]})")
    return {
        "freeze_commit": commit,
        "freeze_blob": blob,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def test_frame(species: str) -> pd.DataFrame:
    """
    Load one held-out species' test individuals.
    
    Only be called after `assert_freeze_committed`; no other Tier 2 script should open this partition.
    """
    stem, split_stem, _, _, _ = held_out_entry(species)
    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    frame = load_split_frame(PROCESSED / f"{stem}_segments.parquet", split.test)
    return frame[["individual_id", "bin_index", "timestamp", "lon", "lat"]].reset_index(drop=True)


# comparator discovery and prediction
def load_trained(run_id: str):
    """
    Rebuild a trained model from the checkpoint reported by its training run.
    
    The saved configuration reconstructs the architecture, and non finite parameters are rejected before the model is allowed into 
    sealed evaluation.
    """
    path = FINAL / f"{run_id}_reported.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainConfig(**state["config"]).build()
    model.load_state_dict(state["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit(f"FAIL: {path} holds non-finite weights; that run diverged and cannot be scored")
    return model.eval()


def available_runs() -> dict[tuple[str, str, int], str]:
    """
    Index completed training results by `(arm, species, seed)`.
    """
    found = {}
    for path in sorted(FINAL.glob("*.json")):
        payload = json.loads(path.read_text())
        found[(payload["arm"], payload.get("species") or "", int(payload["seed"]))] = payload["run_id"]
    return found


def comparator_jobs(
    species: str,
    runs: dict,
) -> list[tuple[str, int | None, object]]:
    """
    Return all available comparator jobs for one held-out species.
    
    P and V are untrained. Z and B are shared across species, while S and Za are attached to a specific held-out species.
    """
    jobs: list[tuple[str, int | None, object]] = [
        ("P", None, Persistence()),
        ("V", None, DisplacementAveraging(k=1)),
    ]
    for arm in TRAINED:
        key_species = "" if arm in {"Z", "B"} else species
        for seed in SEEDS:
            run_id = runs.get((arm, key_species, seed))
            if run_id is not None:
                jobs.append((arm, seed, run_id))
    return jobs


def prediction_path(species: str, model: str, seed: int | None) -> Path:
    tag = model if seed is None else f"{model}_s{seed}"
    return PREDICTIONS / f"{species}__{tag}.parquet"


# GenAI reference: I asked for GenAI to help with sealed run orchestration, comparator discovery, skip/overwrite handling, manifest 
# writing and progress output around my evaluation protocol; I checked the below. 
def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    """
    Evaluate every available comparator on each sealed test partition.
    
    Predictions are saved per `(species, model, seed)` so reporting, floor sensitivity and bootstrap analysis can be rerun without 
    reopening the test data.
    """
    attestation = assert_freeze_committed()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    runs = available_runs()
    manifest = {"attestation": attestation, "species": {}}

    for species, *_ in HELD_OUT:
        frame = test_frame(species)
        print(
            f"\n{species}: {len(frame):,} test fixes, "
            f"{frame['individual_id'].nunique()} individuals",
            flush=True,
        )
        written = []
        for model, seed, handle in comparator_jobs(species, runs):
            out = prediction_path(species, model, seed)
            if out.exists() and not overwrite:
                print(f"  {out.name}  (exists, skipped)")
                written.append(out.name)
                continue
            started = time.time()
            if isinstance(handle, str):
                trained = load_trained(handle)
                if trained is None:
                    print(f"  {model} seed {seed}: no reported checkpoint, skipped")
                    continue
                results = evaluate_batched(
                    MovementFMPredictor(trained, device=device),
                    frame,
                    context_length=CONTEXT_LENGTH,
                )
            else:
                results = evaluate(
                    handle, frame, context_length=CONTEXT_LENGTH, show_progress=False
                )
            results.to_parquet(out, index=False)
            written.append(out.name)
            print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
        manifest["species"][species] = written

    missing = sorted(
        arm for arm in (*TRAINED, *SECONDARY_ONLY)
        if not any(f"__{arm}_s" in name for names in manifest["species"].values() for name in names)
    )
    manifest["deferred_or_missing_arms"] = missing
    if missing:
        print(
            f"\nNOTE: no results for {', '.join(missing)}. These are registered comparators; their "
            "absence is a reporting-completeness gap that must be stated in the freeze with its "
            "reason, not a silent omission."
        )
    SEALED.mkdir(parents=True, exist_ok=True)
    (SEALED / "tier2_sealed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# Metrics and bootstrap
def eligible_mask(results: pd.DataFrame, floor: int) -> np.ndarray:
    counts = results.groupby("individual_id").size()
    keep = set(counts[counts >= floor].index)
    return results["individual_id"].isin(keep).to_numpy()


def stratum_masks(persistence: pd.DataFrame, boundary: float) -> dict[str, np.ndarray]:
    """
    Split windows into A4 travel and low displacement strata.
    
    Persistence error is the realised one step displacement, so the same run supplies both the comparator and the stratum labels.
    """
    step = persistence["error_m"].to_numpy(float)
    travel = step >= float(boundary)
    return {
        "travel": travel,
        "low_displacement": ~travel,
        "all": np.ones_like(travel, dtype=bool),
    }


def summarise(
    results: pd.DataFrame,
    persistence: pd.DataFrame,
    boundary: float,
    floor: int = EVALUATION_FLOOR,
) -> dict:
    """
    Compute the registered metrics for one comparator on one species.
    
    Median, p90 and the individual weighted mean give each animal equal total weight. p99 and mean are also carried in pooled 
    form for reporting/comparison, while the registered success bands continue to use the preregistered statistics.
    """
    if len(results) != len(persistence):
        raise ValueError("comparator and persistence frames describe diff window sets")

    keep = eligible_mask(persistence, floor)
    masks = stratum_masks(persistence, boundary)
    out: dict = {
        "floor": floor,
        "n_individuals": int(persistence.loc[keep, "individual_id"].nunique()),
    }
    # GenAI finished the below off (until out)
    for stratum, mask in masks.items():
        selected = keep & mask
        rows = results.loc[selected]
        errors = rows["error_m"].to_numpy(float)
        entry = {"n_windows": int(selected.sum())}
        if entry["n_windows"]:
            entry["p50_m"] = individual_weighted_quantile(rows, QUANTILES["p50"])
            entry["p90_m"] = individual_weighted_quantile(rows, QUANTILES["p90"])
            entry["p99_m_individual_weighted"] = individual_weighted_quantile(rows, QUANTILES["p99"])
            entry["p99_m_pooled"] = float(np.percentile(errors, 99))
            entry["mean_m_individual_weighted"] = individual_weighted_mean(rows)
            entry["mean_m_pooled"] = float(errors.mean())
        out[stratum] = entry
    return out


def individual_weighted_mean(
    rows: pd.DataFrame,
    *,
    value_col: str = "error_m",
    individual_col: str = "individual_id",
) -> float:
    """
    Return the mean of each individual's mean error.
    
    This gives every animal equal total weight, matches our A5 weighting principle and providing the trajectory 
    weighted analogue of the mean error reported in the TFT literature.
    """
    per_individual = rows.groupby(individual_col)[value_col].mean()
    return float(per_individual.mean()) if len(per_individual) else float("nan")


# GenAI reference: I used it here to help with the NumPy bootstrap helpers below so the registered individual 
# weighted calculations can be repeated efficiently across many resamples
def _weighted_quantile_from_blocks(picked: list[np.ndarray], quantile: float) -> float:
    """
    Compute the individual weighted quantile from pre- plit error blocks.
    
    This reproduces `individual_weighted_quantile` using NumPy arrays so repeated bootstrap evaluations do not 
    require expensive pandas grouping.
    """
    k = len(picked)
    values = np.concatenate(picked)
    # Repeated draws remain separate equal-weight clusters.
    weights = np.concatenate(
        [np.full(block.size, 1.0 / (k * block.size)) for block in picked]
    )
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    position = int(np.searchsorted(np.cumsum(weights), quantile))
    return float(values[min(position, values.size - 1)])


def _weighted_mean_from_blocks(picked: list[np.ndarray]) -> float:
    """
    Return the mean of per individual means for pre-split error blocks.
    """
    return float(np.mean([block.mean() for block in picked]))

# GenAI reference: this whole function below it gave me and I added it unchanged, bar the docstirng which is mine. The code 
# is correct. 
def bootstrap_paired(
    frames: dict[str, pd.DataFrame],
    persistence: pd.DataFrame,
    boundary: float,
    floor: int = EVALUATION_FLOOR,
) -> dict:
    """
    Run the paired individual level bootstrap for the registered report series.
    
    Each series uses 2,000 individual resamples at the registered seed, shared across comparators within that series. Travel 
    p90 supports the registered comparison, while all window mean is reported for comparison with TFT paper (SOA).  
    """
    keep = eligible_mask(persistence, floor)
    masks = stratum_masks(persistence, boundary)
    out: dict = {
        "n_resamples": N_BOOTSTRAP,
        "seed": BOOTSTRAP_SEED,
        "series": {},
    }

    for stratum, statistic in BOOTSTRAP_SERIES:
        key = f"{stratum}_{statistic}_m"
        selected = keep & masks[stratum]
        ids = persistence.loc[selected, "individual_id"].to_numpy()
        unique = np.unique(ids)
        if unique.size < 2:
            out["series"][key] = {
                "error": f"fewer than two eligible individuals in {stratum}"
            }
            continue

        rows_by_individual = [np.flatnonzero(ids == value) for value in unique]
        draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
            0, unique.size, size=(N_BOOTSTRAP, unique.size)
        )
        blocks = {
            name: [
                frame["error_m"].to_numpy(float)[selected][rows]
                for rows in rows_by_individual
            ]
            for name, frame in frames.items()
        }
        samples: dict[str, list[float]] = {name: [] for name in blocks}
        for draw in draws:
            for name, per_individual in blocks.items():
                picked = [per_individual[index] for index in draw]
                samples[name].append(
                    _weighted_mean_from_blocks(picked)
                    if statistic == "mean"
                    else _weighted_quantile_from_blocks(picked, QUANTILES[statistic])
                )

        entry: dict = {
            "n_individuals": int(unique.size),
            "interval": {
                name: {
                    "lo": float(np.percentile(values, 2.5)),
                    "hi": float(np.percentile(values, 97.5)),
                }
                for name, values in samples.items()
            },
        }
        if "Z" in samples:
            entry["paired_difference_vs_Z_m"] = {}
            for name, values in samples.items():
                if name == "Z":
                    continue
                difference = np.asarray(samples["Z"]) - np.asarray(values)
                lo, hi = np.percentile(difference, 2.5), np.percentile(difference, 97.5)
                entry["paired_difference_vs_Z_m"][name] = {
                    "lo": float(lo),
                    "hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0),
                }
        out["series"][key] = entry
    return out


def headline(
    per_seed: dict[int, dict],
    stratum: str,
    statistic: str,
) -> tuple[float, int | None]:
    """
    Return the median across seeds headline and the seed that realised it.
    
    With five seeds the median is an observed run, so bootstrap intervals can be taken from that run's predictions 
    rather than from a synthetic aggregate.
    """
    pairs = [
        (payload[stratum][statistic], seed)
        for seed, payload in sorted(per_seed.items())
        if statistic in payload.get(stratum, {})
    ]
    if not pairs:
        return float("nan"), None
    values = sorted(value for value, _ in pairs)
    point = float(median(values))
    for value, seed in pairs:
        if value == point:
            return point, seed
    return point, None


# GenAI reference: I asked for it to help assemble saved predictions into the registered headline, skill, bootstrap and 
# floor sensitivity report structure which it did; and I checked it
def build_report(floor: int = EVALUATION_FLOOR) -> dict:
    """Build the sealed report from saved prediction files.
    
    The report includes per-model headlines, persistence skill, paired bootstrap intervals, floor
    sensitivity and the registered success-band verdict without reopening the test partitions.
    """
    if not PREDICTIONS.exists():
        raise SystemExit(f"FAIL: {PREDICTIONS} does not exist; run --predict first")

    report: dict = {
        "evaluation_floor": floor,
        "floor_sensitivity": list(FLOOR_SENSITIVITY),
        "seed_aggregation": "median_across_training_seeds",
        "species": {},
    }
    manifest_path = SEALED / "tier2_sealed_manifest.json"
    if manifest_path.exists():
        report["attestation"] = json.loads(manifest_path.read_text()).get("attestation")

    for species, *_ in HELD_OUT:
        _, _, profile_key, domain, _ = held_out_entry(species)
        boundary = boundary_m(profile_key)
        persistence_path = prediction_path(species, "P", None)
        if not persistence_path.exists():
            raise SystemExit(f"FAIL: {persistence_path} is missing; P defines the strata and the skill")
        persistence = pd.read_parquet(persistence_path)

        loaded: dict[str, pd.DataFrame] = {}
        per_seed: dict[str, dict[int, dict]] = {}
        for path in sorted(PREDICTIONS.glob(f"{species}__*.parquet")):
            tag = path.stem.split("__", 1)[1]
            model, _, seed_text = tag.partition("_s")
            frame = pd.read_parquet(path)
            statistics = summarise(frame, persistence, boundary, floor)
            if seed_text:
                per_seed.setdefault(model, {})[int(seed_text)] = statistics
            else:
                per_seed.setdefault(model, {})[-1] = statistics
            loaded[tag] = frame

        entry: dict = {
            "domain": domain,
            "regime_boundary_m": boundary,
            "n_test_individuals": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "comparators": {},
            "floor_sensitivity": {},
        }

        headline_frames: dict[str, pd.DataFrame] = {}
        ordered = sorted(
            per_seed.items(),
            key=lambda item: (
                COMPARATOR_ORDER.index(item[0]) if item[0] in COMPARATOR_ORDER else len(COMPARATOR_ORDER),
                item[0],
            ),
        )
        for model, seeds in ordered:
            record: dict = {
                "n_seeds": len([s for s in seeds if s >= 0]) or None,
                "per_seed": {},
            }
            for stratum in (*STRATA, "all"):
                _, seed = headline(seeds, stratum, "p90_m")
                # Window counts belong to the shared stratum, not the comparator.
                counts = {
                    payload[stratum].get("n_windows")
                    for payload in seeds.values()
                    if stratum in payload
                }
                record.setdefault("headline", {})[stratum] = {
                    "n_windows": counts.pop() if len(counts) == 1 else None,
                    # Apply the same A3 median-across-seeds aggregation to every statistic.
                    **{name: headline(seeds, stratum, name)[0] for name in STATISTICS},
                    # Bootstrap the run that realises the median travel p90.
                    "headline_seed": seed if seed is not None and seed >= 0 else None,
                }
            for seed, statistics in sorted(seeds.items()):
                if seed >= 0:
                    record["per_seed"][seed] = statistics
            if list(seeds) == [-1]:
                record["n_seeds"] = None
                record["per_seed"] = {}
                record["untrained"] = True
            entry["comparators"][model] = record

            seed = record["headline"]["travel"]["headline_seed"]
            tag = model if seed is None else f"{model}_s{seed}"
            if tag in loaded:
                headline_frames[model] = loaded[tag]

        # Persistence skill from headline point estimates.
        for model, record in entry["comparators"].items():
            if model == "P":
                continue
            record["persistence_skill"] = {}
            for stratum in (*STRATA, "all"):
                reference = entry["comparators"]["P"]["headline"][stratum]
                for statistic in STATISTICS:
                    denominator = reference.get(statistic)
                    value = record["headline"][stratum].get(statistic)
                    if denominator and np.isfinite(denominator) and np.isfinite(value):
                        record["persistence_skill"][f"{stratum}_{statistic}"] = 1.0 - value / denominator

        entry["bootstrap"] = bootstrap_paired(headline_frames, persistence, boundary, floor)

        for alternative in FLOOR_SENSITIVITY:
            entry["floor_sensitivity"][alternative] = {
                model: summarise(
                    headline_frames[model],
                    persistence,
                    boundary,
                    alternative,
                )["travel"].get("p90_m")
                for model in sorted(headline_frames)
            }

        report["species"][species] = entry

    report["verdict"] = apply_success_bands(report)
    return report


def apply_success_bands(report: dict) -> dict:
    """
    Apply the registered STRONG, ACCEPTABLE and WEAK/NULL criteria in order.
    
    Uses headline point estimates only. Bootstrap intervals and seed spread are reported alongside the result but 
    do not introduce another pass/fail rule.
    """
    per_species = {}
    deferred = []
    for species, entry in report["species"].items():
        comparators = entry["comparators"]
        available = set(comparators)
        for arm in ("Za", "F"):
            if arm not in available:
                deferred.append(f"{arm}@{species}")

        def travel(model, statistic="p90_m"):
            return comparators.get(model, {}).get("headline", {}).get("travel", {}).get(statistic)

        z_p90, p_p90, b_p90, s_p90 = travel("Z"), travel("P"), travel("B"), travel("S")
        z_p99 = travel("Z", "p99_m_individual_weighted")
        p_p99 = travel("P", "p99_m_individual_weighted")

        checks = {
            "Z_beats_P_travel_p90": _lt(z_p90, p_p90),
            "Z_beats_P_travel_p99": _lt(z_p99, p_p99),
            "Z_within_25pc_of_S_travel_p90": _lt(
                z_p90,
                None if s_p90 is None else SPECIALIST_MULTIPLIER * s_p90,
            ),
            "Z_beats_B_travel_p90": _lt(z_p90, b_p90),
        }
        per_species[species] = {
            "checks": checks,
            "travel_p90_m": {
                "Z": z_p90,
                "P": p_p90,
                "B": b_p90,
                "S": s_p90,
            },
            "travel_p99_m_individual_weighted": {
                "Z": z_p99,
                "P": p_p99,
            },
        }
    # GenAI gave this which I added for clarity since it is more reabable. 
    def all_true(name):
        return all(entry["checks"].get(name) is True for entry in per_species.values())

    if (
        all_true("Z_beats_P_travel_p90")
        and all_true("Z_beats_P_travel_p99")
        and all_true("Z_within_25pc_of_S_travel_p90")
    ):
        band = "STRONG"
    elif all_true("Z_beats_P_travel_p90") and all_true("Z_beats_B_travel_p90"):
        band = "ACCEPTABLE"
    else:
        band = "WEAK/NULL"

    return {
        "band": band,
        "per_species": per_species,
        "deferred_comparators": sorted(set(deferred)),
        "note": (
            "Bands are defined on Z alone and evaluated on the median-across-seeds headline. A "
            "deferred comparator does not change the band, but its absence is a reporting-"
            "completeness gap that the freeze must state with its reason."
        ),
    }


def _lt(value, reference) -> bool | None:
    """
    Return a strict lower than comparison, or `None` if either value is unavailable.
    """
    if value is None or reference is None:
        return None
    if not (np.isfinite(value) and np.isfinite(reference)):
        return None
    return bool(value < reference)


# Reporting
# GenAI reference: helped to produce human readable report/table formatting below
def write_markdown(report: dict, path: Path) -> None:
    """Write the human-readable sealed-evaluation summary.
    """
    lines = [
        "# Tier 2 sealed evaluation",
        "",
        f"Evaluation floor F = {report['evaluation_floor']}; median and p90 individual-weighted (A5); "
        "headline = median across the five training seeds (A3).",
        "",
        "p99 is reported here on the individual-weighted distribution, for consistency with the "
        "weighting A5 applies to the median and p90. A5 moved p99 to species level without saying "
        "which species-level form it meant, so the pooled figure is carried alongside it in "
        "`tier2_verdict.json` (`p99_m_pooled`). **The freeze must state which one the §4 p99 "
        "criterion is read on.**",
        "",
        f"**Verdict: {report['verdict']['band']}**",
        "",
    ]
    if report["verdict"]["deferred_comparators"]:
        lines += [
            "Deferred comparators, registered and owed: "
            + ", ".join(report["verdict"]["deferred_comparators"]),
            "",
        ]
    for species, entry in report["species"].items():
        lines += [
            f"## {species} ({entry['domain']})",
            "",
            f"Regime boundary {entry['regime_boundary_m']:.1f} m; "
            f"{entry['n_test_individuals']} test individuals, {entry['n_test_windows']:,} windows.",
            "",
        ]
        series = entry.get("bootstrap", {}).get("series", {})
        for stratum, title in (
            ("all", "All windows"),
            ("travel", "Travel regime"),
            ("low_displacement", "Low displacement"),
        ):
            interval_key = {"all": "all_mean_m", "travel": "travel_p90_m"}.get(stratum)
            intervals = series.get(interval_key, {}).get("interval", {}) if interval_key else {}
            n_windows = entry["comparators"]["P"]["headline"][stratum].get("n_windows")
            lines += [
                f"### {title}" + (f" — {n_windows:,} windows" if n_windows else ""),
                "",
                "| model | mean | p50 | p90 | p99 | skill (p90) |"
                # Name the bootstrapped statistic in the table header.
                + (f" 95% CI ({interval_key.split('_')[1]}) |" if intervals else ""),
                "|---|---:|---:|---:|---:|---:|" + ("---:|" if intervals else ""),
            ]
            for model, record in entry["comparators"].items():
                values = record["headline"][stratum]
                skill = record.get("persistence_skill", {}).get(f"{stratum}_p90_m")
                row = (
                    f"| {model} | {_m(values.get('mean_m_individual_weighted'))} | "
                    f"{_m(values.get('p50_m'))} | {_m(values.get('p90_m'))} | "
                    f"{_m(values.get('p99_m_individual_weighted'))} | "
                    + (f"{skill:+.4f} |" if skill is not None else "— |")
                )
                if intervals:
                    band = intervals.get(model)
                    row += (
                        f" [{_m(band['lo'])}, {_m(band['hi'])}] |" if band else " — |"
                    )
                lines.append(row)
            lines.append("")
    lines += [
        "---",
        "",
        "Mean, p50, p90 and p99 are all individual-weighted (A5), so every animal contributes equal "
        "weight regardless of deployment length. Pooled forms of the mean and p99 are in "
        "`tier2_verdict.json`. The 95% intervals are the registered paired percentile bootstrap — "
        "2,000 resamples of test individuals at seed 42, shared across comparators — taken on "
        "all-windows mean and travel-regime p90; per-comparator differences against Z, including "
        "whether each excludes zero, are in the JSON.",
        "",
        "§3 registers median, p90 and p99 and treats the mean as uninformative under this much "
        "right skew. The mean is reported anyway for comparability with the TFT movement-forecasting "
        "literature, whose headline is mean great-circle error; **no §4 success band reads it**.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _m(value) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:,.0f} m"


# GenAI reference: helped here with CLI and output writing scaffolding for our 2 phase sealed evaluation; the evaluation 
# protocol and registered criteria were specified separately.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predict",
        action="store_true",
        help="open the sealed partitions and score every comparator",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="compute metrics, intervals and the §4 verdict",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-score comparators that already have a prediction file",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=EVALUATION_FLOOR,
        help="primary evaluation window floor",
    )
    args = parser.parse_args()

    if not (args.predict or args.report):
        raise SystemExit("FAIL: pass --predict, --report, or both")

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        run_predictions(device, overwrite=args.overwrite)

    if args.report:
        report = build_report(args.floor)
        SEALED.mkdir(parents=True, exist_ok=True)
        (SEALED / "tier2_verdict.json").write_text(
            json.dumps(report, indent=2, default=float) + "\n"
        )
        write_markdown(report, SEALED / "tier2_results.md")
        print(f"\nverdict: {report['verdict']['band']}")
        for species, entry in report["verdict"]["per_species"].items():
            passed = [name for name, value in entry["checks"].items() if value is True]
            failed = [name for name, value in entry["checks"].items() if value is False]
            unknown = [name for name, value in entry["checks"].items() if value is None]
            print(
                f"  {species}: pass {passed or '-'}; fail {failed or '-'}; "
                f"undetermined {unknown or '-'}"
            )
        print(f"\nwrote {SEALED / 'tier2_verdict.json'}")
        print(f"wrote {SEALED / 'tier2_results.md'}")


if __name__ == "__main__":
    main()
