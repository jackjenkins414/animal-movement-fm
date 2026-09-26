# GenAI reference: This is a GenAI written file; I asked it to implement this Tier 3 sealed evaluator from my `preregistration_tier3.md` 
# which fixes the arms, ladder, predictions, per draw reporting etc. I asked it to reuse my Tier 2 evaluation machinery directly; I 
# reviewed the implementation before opening a test partition.
"""
Sealed Tier 3 evaluation on the three held-out species.

`--predict` writes Tier 3 per-window predictions. `--report` reads those files together with the existing Tier 2 sealed predictions and 
computes the registered metrics, predictions and baseline ladder.

Tier 2 is never rerun, and nothing here opens test data before `preregistration_tier3_freeze.md` is committed.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tier2_final_train as T  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402
import tier3_train as T3  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import (  # noqa: E402
    MovementFMPredictor,
    WindowModelPredictor,
    evaluate_batched,
)
from animal_movement_fm.eval.harness import evaluate  # noqa: E402
from animal_movement_fm.models.arima import ARIMA  # noqa: E402
from animal_movement_fm.models.averaging import DisplacementAveraging  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TIER3_RESULTS = REPO / "results" / "tier3"
TIER_E_RESULTS = REPO / "results" / "tier_e"
SEALED = REPO / "results" / "tier3_sealed"
PREDICTIONS = SEALED / "predictions"
TIER2_PREDICTIONS = REPO / "results" / "tier2_sealed" / "predictions"
FREEZE = REPO / "preregistration_tier3_freeze.md"

REPORT_JSON = SEALED / "tier3_results.json"
REPORT_MD = SEALED / "tier3_results.md"
MANIFEST = SEALED / "tier3_sealed_manifest.json"

FLOOR = S2.EVALUATION_FLOOR
STRATA = S2.STRATA
STATISTICS = S2.STATISTICS
FLOOR_SENSITIVITY = S2.FLOOR_SENSITIVITY
# Single-seed skill gaps below this are reported as unresolved.
NOISE_SCALE = 0.05
# Arms included in the A5 floor sensitivity.
FLOOR_SENSITIVITY_ARMS = ("Z9", "B9", "Za9", "Zd3", "Zd1", "Z", "B", "S")

# Section 3.1 training-free rungs. V is carried from Tier 2 rather than recomputed.
LADDER_FREE = (
    ("ARIMA", lambda: ARIMA(order=(2, 0, 2))),
    ("AVG3", lambda: DisplacementAveraging(k=3)),
    ("AVG10", lambda: DisplacementAveraging(k=10)),
)

# Tier 2 comparators carried from their sealed prediction files.
CARRIED = ("P", "V", "Z", "B", "S", "Za")

# Registered specialist plain-trunk rung, trained under Tier E and scored here.
TIER_E_TEST_ARMS = ("SB",)

# Tier 2 E1embed carried forward and scored under the matching domain row, with no retraining.
DOMAIN_ROW_CARRIED = ("E1embed",)

# Reporting order; unlisted arms sort alphabetically afterwards.
ARM_ORDER = (
    "P", "V", "ARIMA", "AVG3", "AVG10",
    "Z", "B", "Za", "S", "SB",
    "Z9", "B9", "Za9", "Zd3", "Zd1", "B9dom", "E1embed_dom",
    "LSTMS", "LSTMB3", "LSTMB9", "XGBS", "XGBB3", "XGBB9",
    "Sk", "F3c", "F9",
)


# Loading
def tier3_runs() -> list[dict]:
    """
    Return completed registered Tier 3 runs from their result records.

    The result record, not the filename, defines the arm and whether the run is exploratory.
    """
    found = []
    for path in sorted(TIER3_RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("exploratory") or str(record.get("tier")) != "3":
            print(f"  skipped {path.name}: not a registered Tier 3 record")
            continue
        if record["arm"] not in T3.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record['arm']}, which is not a Tier 3 arm")
        found.append(record)
    return found


def prediction_path(species: str, tag: str) -> Path:
    return PREDICTIONS / f"{species}__{tag}.parquet"


def load_frame(path: Path) -> pd.DataFrame:
    """Read one prediction file in the fixed row order used for paired comparisons."""
    frame = pd.read_parquet(path)
    return frame.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)


def assert_aligned(frame: pd.DataFrame, persistence: pd.DataFrame, label: str) -> None:
    """Refuse prediction files that are not aligned to the persistence windows."""
    if len(frame) != len(persistence):
        raise SystemExit(f"FAIL: {label} has {len(frame):,} windows; persistence has {len(persistence):,}")
    same = bool(
        (frame["individual_id"].to_numpy() == persistence["individual_id"].to_numpy()).all()
        and np.allclose(frame["target_lon"].to_numpy(float), persistence["target_lon"].to_numpy(float))
        and np.allclose(frame["target_lat"].to_numpy(float), persistence["target_lat"].to_numpy(float))
    )
    if not same:
        raise SystemExit(
            f"FAIL: {label} does not describe the same windows as persistence, so it cannot be paired with it. "
            "Re-run --predict for that arm with --overwrite."
        )


def load_reported(results_dir: Path, run_id: str):
    """Rebuild a reported model from its checkpoint, refusing non-finite weights."""
    path = results_dir / f"{run_id}_reported.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainConfig(**state["config"]).build()
    model.load_state_dict(state["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit(f"FAIL: {path} holds non-finite weights; that run diverged and cannot be scored")
    return model.eval(), state


def domain_row(state: dict, domain: str, run_id: str) -> int:
    """Return the learned domain row used to score an unseen held-out species."""
    rows = list(state.get("domain_rows") or [])
    if not rows:
        pool = list(state.get("pool_species") or [])
        raise SystemExit(
            f"FAIL: {run_id} carries a species lookup indexed by {pool or 'nothing recorded'}, not by domain, so it "
            "is not zero-shot on an unseen species"
        )
    if domain not in rows:
        raise SystemExit(f"FAIL: {run_id} has no row for domain {domain}; it records {rows}")
    return rows.index(domain)


# Prediction
def predict_species(species: str, records: list[dict], device: torch.device, overwrite: bool) -> list[str]:
    _, _, _, domain, matching_source = T.held_out_entry(species)
    frame = S2.test_frame(species)
    print(
        f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals",
        flush=True,
    )
    written: list[str] = []

    def done(tag: str) -> bool:
        out = prediction_path(species, tag)
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            written.append(out.name)
            return True
        return False

    def save(tag: str, results: pd.DataFrame, started: float) -> None:
        out = prediction_path(species, tag)
        results.to_parquet(out, index=False)
        written.append(out.name)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)

    # Training-free ladder rungs; ARIMA is fitted per window at prediction time.
    for tag, build in LADDER_FREE:
        if done(tag):
            continue
        started = time.time()
        save(tag, evaluate(build(), frame, context_length=CONTEXT_LENGTH, show_progress=False), started)

    # Score pooled runs on every species and attached runs only on their own species.
    for record in records:
        attached = record.get("species")
        if attached and attached != species:
            continue
        tag = record["run_id"]
        if done(tag):
            continue
        started = time.time()
        if record.get("model") == "WindowXGBoost":
            from animal_movement_fm.models.xgboost import WindowXGBoost

            model = WindowXGBoost.load(TIER3_RESULTS / f"{tag}_xgb")
            predictor = WindowModelPredictor(model, name=tag, sigma=SCALE_FLOOR_M)
        else:
            loaded = load_reported(TIER3_RESULTS, tag)
            if loaded is None:
                print(f"  {tag}: no reported checkpoint, skipped")
                continue
            model, state = loaded
            row = domain_row(state, domain, tag) if state["config"].get("species_embed_dim") else None
            predictor = MovementFMPredictor(model, device=device, sigma=SCALE_FLOOR_M, species_index=row)
        save(tag, evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH), started)

    # Specialist plain-trunk rung carried from Tier E.
    for arm in TIER_E_TEST_ARMS:
        for seed in T.SEEDS:
            tag = f"{arm}_s{seed}"
            if done(tag):
                continue
            loaded = load_reported(TIER_E_RESULTS, f"{arm}_{species}_s{seed}")
            if loaded is None:
                print(f"  {arm}_{species}_s{seed}: no reported checkpoint, skipped")
                continue
            model, state = loaded
            started = time.time()
            save(
                tag,
                evaluate_batched(
                    MovementFMPredictor(model, device=device, sigma=SCALE_FLOOR_M),
                    frame,
                    context_length=CONTEXT_LENGTH,
                ),
                started,
            )

    # Tier 2 domain-label arm scored under the held-out species' matching domain row.
    for arm in DOMAIN_ROW_CARRIED:
        for seed in T.SEEDS:
            tag = f"{arm}_dom_s{seed}"
            if done(tag):
                continue
            loaded = load_reported(TIER_E_RESULTS, f"{arm}_s{seed}")
            if loaded is None:
                print(f"  {arm}_s{seed}: no reported checkpoint, skipped")
                continue
            model, state = loaded
            pool = list(state.get("pool_species") or [])
            if matching_source not in pool:
                raise SystemExit(f"FAIL: {arm}_s{seed} has no row for {matching_source} (table covers {pool})")
            started = time.time()
            save(
                tag,
                evaluate_batched(
                    MovementFMPredictor(
                        model, device=device, sigma=SCALE_FLOOR_M, species_index=pool.index(matching_source)
                    ),
                    frame,
                    context_length=CONTEXT_LENGTH,
                ),
                started,
            )
    return written


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    S2.FREEZE = FREEZE  # Reuse the Tier 2 seal check against the Tier 3 freeze.
    attestation = S2.assert_freeze_committed()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    records = tier3_runs()
    print(f"tier 3 results   : {len(records)} of {len(T3.PLAN)} registered runs on disk")

    manifest = {"attestation": attestation, "n_tier3_records": len(records), "species": {}}
    for species, *_ in T.HELD_OUT:
        manifest["species"][species] = predict_species(species, records, device, overwrite)
    missing = sorted({arm for arm in T3.ARMS if not T3.ARMS[arm].get("tier_e")} - {r["arm"] for r in records})
    manifest["missing_arms"] = missing
    if missing:
        print(
            f"\nNOTE: no trained runs for {', '.join(missing)}. These are registered arms; their absence is a "
            "reporting-completeness gap and must be stated with its reason, not passed over."
        )
    SEALED.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# Metrics
def species_frames(species: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """Load one species' persistence frame, available predictions and Tier 3 run records."""
    persistence_path = TIER2_PREDICTIONS / f"{species}__P.parquet"
    if not persistence_path.exists():
        raise SystemExit(
            f"FAIL: {persistence_path} is missing. P defines the strata and the skill denominator, and Tier 3 reads "
            "Tier 2's sealed persistence run rather than recomputing it."
        )
    persistence = load_frame(persistence_path)

    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(TIER2_PREDICTIONS.glob(f"{species}__*.parquet")):
        tag = path.stem.split("__", 1)[1]
        if tag.partition("_s")[0] not in CARRIED:
            continue
        frames[f"tier2:{tag}"] = load_frame(path)
    for path in sorted(PREDICTIONS.glob(f"{species}__*.parquet")):
        frames[path.stem.split("__", 1)[1]] = load_frame(path)
    for tag, frame in frames.items():
        assert_aligned(frame, persistence, f"{species} {tag}")

    # the XGBoost arms write a metadata file beside their result record, and it carries no run_id; the records are the
    # run files only, keyed by run id
    records = {
        record["run_id"]: record
        for record in (json.loads(path.read_text()) for path in sorted(TIER3_RESULTS.glob("*.json")))
        if "run_id" in record and "arm" in record
    }
    return persistence, frames, records


def parse_tag(tag: str, records: dict[str, dict]) -> dict | None:
    """Decode a prediction tag into its arm, seed and optional subset draw."""
    if tag.startswith("tier2:"):
        arm, _, seed = tag.removeprefix("tier2:").partition("_s")
        return {"arm": arm, "seed": int(seed) if seed else None, "k": None, "draw": None}
    if tag in records:
        record = records[tag]
        return {
            "arm": record["arm"], "seed": record["seed"],
            "k": record.get("k_percent"), "draw": record.get("draw"),
        }
    arm, _, seed = tag.partition("_s")
    known = {name for name, _ in LADDER_FREE} | {f"{name}_dom" for name in DOMAIN_ROW_CARRIED} | set(TIER_E_TEST_ARMS)
    if arm in known:
        return {"arm": arm, "seed": int(seed) if seed else None, "k": None, "draw": None}
    print(f"  ignored {tag}: no Tier 3 result record and not a recognised carried arm")
    return None


def summarise_all(
    frames: dict[str, pd.DataFrame], persistence: pd.DataFrame, boundary: float, records: dict, parse=None
) -> dict[str, dict]:
    """Compute the registered metrics and group them by arm, seed or subset draw."""
    # GenAI reference: GenAI helped my diagnose and fix this bug;`parse` added so tier4_sealed_eval.py can decode its `tier3:`-prefixed carried tags
    # with its own parse_tagl if left as None, Tier 3s own parse_tag is used exactly as before
    out: dict[str, dict] = {}
    for tag, frame in frames.items():
        parsed = (parse or parse_tag)(tag, records)
        if parsed is None:
            continue
        statistics = S2.summarise(frame, persistence, boundary, FLOOR)
        entry = out.setdefault(parsed["arm"], {"by_seed": {}, "by_draw": {}, "tags": {}})
        if parsed["k"] is None:
            key = parsed["seed"] if parsed["seed"] is not None else -1
            entry["by_seed"][key] = statistics
            entry["tags"][key] = tag
        else:
            key = (parsed["k"], parsed["draw"])
            entry["by_draw"][key] = statistics
            entry["tags"][key] = tag
    return out


def skill(value, reference) -> float:
    """Return persistence skill: 1 - model / persistence."""
    if value is None or reference is None:
        return float("nan")
    if not (np.isfinite(value) and np.isfinite(reference) and reference):
        return float("nan")
    return 1.0 - value / reference


def headline_block(entry: dict, persistence_headline: dict) -> dict:
    """Build one seeded arm's A3 median headline and persistence skill by stratum."""
    block: dict = {"n_seeds": len([k for k in entry["by_seed"] if k >= 0]) or None, "headline": {}, "skill": {}}
    for stratum in (*STRATA, "all"):
        values = {name: S2.headline(entry["by_seed"], stratum, name)[0] for name in STATISTICS}
        _, seed = S2.headline(entry["by_seed"], stratum, "p90_m")
        block["headline"][stratum] = {**values, "headline_seed": seed if seed is not None and seed >= 0 else None}
        block["skill"][stratum] = {
            name: skill(values[name], persistence_headline[stratum].get(name)) for name in STATISTICS
        }
    block["per_seed"] = {str(seed): payload for seed, payload in sorted(entry["by_seed"].items()) if seed >= 0}
    return block


def draw_block(entry: dict, persistence_headline: dict) -> dict:
    """Build a subset-trained arm's point estimates separately for each `(k, draw)`."""
    out: dict = {}
    for (k, draw), statistics in sorted(entry["by_draw"].items()):
        strata = [stratum for stratum in (*STRATA, "all") if stratum in statistics]
        out[f"k{k:03d}_d{draw}"] = {
            "k_percent": k,
            "draw": draw,
            "headline": {stratum: statistics[stratum] for stratum in strata},
            "skill": {
                stratum: {
                    name: skill(statistics[stratum].get(name), persistence_headline[stratum].get(name))
                    for name in STATISTICS
                }
                for stratum in strata
            },
        }
    return out


def tag_key(key) -> str:
    """Return the report key for a seed or `(k, draw)` subset."""
    if isinstance(key, tuple):
        k, draw = key
        return f"k{k:03d}_d{draw}"
    return str(key)


def travel_skill(arms: dict, arm: str) -> float:
    return arms.get(arm, {}).get("skill", {}).get("travel", {}).get("p90_m", float("nan"))


def headline_frame(arms: dict, frames: dict[str, pd.DataFrame], arm: str):
    """Return the prediction frame behind an arm's reported median travel p90."""
    block = arms.get(arm)
    if not block or not block.get("headline"):
        return None
    seed = block["headline"]["travel"].get("headline_seed")
    tag = block["tags"].get(tag_key(seed if seed is not None else -1))
    return frames.get(tag) if tag else None


# Paired bootstrap
def paired_delta(
    a: pd.DataFrame, b: pd.DataFrame, persistence: pd.DataFrame, boundary: float, labels: tuple[str, str]
) -> dict:
    """
    Run the registered paired bootstrap for one arm comparison.

    Differences are `left - right` in metres, so negative means the left arm has lower error.
    """
    keep = S2.eligible_mask(persistence, FLOOR)
    masks = S2.stratum_masks(persistence, boundary)
    out: dict = {"n_resamples": S2.N_BOOTSTRAP, "seed": S2.BOOTSTRAP_SEED, "series": {}}
    for stratum, statistic in S2.BOOTSTRAP_SERIES:
        key = f"{stratum}_{statistic}_m"
        selected = keep & masks[stratum]
        ids = persistence.loc[selected, "individual_id"].to_numpy()
        unique = np.unique(ids)
        if unique.size < 2:
            out["series"][key] = {"error": f"fewer than two eligible individuals in {stratum}"}
            continue
        rows_by_individual = [np.flatnonzero(ids == value) for value in unique]
        draws = np.random.default_rng(S2.BOOTSTRAP_SEED).integers(
            0, unique.size, size=(S2.N_BOOTSTRAP, unique.size)
        )
        blocks = {
            name: [frame["error_m"].to_numpy(float)[selected][rows] for rows in rows_by_individual]
            for name, frame in ((labels[0], a), (labels[1], b))
        }
        samples: dict[str, list[float]] = {name: [] for name in blocks}
        for draw in draws:
            for name, per_individual in blocks.items():
                picked = [per_individual[index] for index in draw]
                samples[name].append(
                    S2._weighted_mean_from_blocks(picked)
                    if statistic == "mean"
                    else S2._weighted_quantile_from_blocks(picked, S2.QUANTILES[statistic])
                )
        difference = np.asarray(samples[labels[0]]) - np.asarray(samples[labels[1]])
        lo, hi = float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))
        out["series"][key] = {
            "n_individuals": int(unique.size),
            "interval": {
                name: {"lo": float(np.percentile(values, 2.5)), "hi": float(np.percentile(values, 97.5))}
                for name, values in samples.items()
            },
            "difference_m": {
                "label": f"{labels[0]} - {labels[1]}",
                "lo": lo,
                "hi": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            },
        }
    return out


def resolved(delta: float) -> bool | None:
    """Return whether a skill gap exceeds the single-seed noise scale."""
    if delta is None or not np.isfinite(delta):
        return None
    return bool(abs(delta) > NOISE_SCALE)


def compare_arms(
    left: str, right: str, arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float
) -> dict:
    """Compare two arms on travel-p90 skill and their paired headline predictions."""
    out: dict = {"left": left, "right": right}
    missing = [name for name in (left, right) if not np.isfinite(travel_skill(arms, name))]
    if missing:
        out["error"] = f"{', '.join(missing)} has no scored runs for this species"
        return out
    left_skill, right_skill = travel_skill(arms, left), travel_skill(arms, right)
    out |= {
        f"{left}_travel_p90_skill": left_skill,
        f"{right}_travel_p90_skill": right_skill,
        "skill_difference": left_skill - right_skill,
        "left_better": bool(left_skill > right_skill),
        "resolved_against_noise_scale": resolved(left_skill - right_skill),
        "noise_scale": NOISE_SCALE,
    }
    a, b = headline_frame(arms, frames, left), headline_frame(arms, frames, right)
    if a is not None and b is not None:
        out["bootstrap"] = paired_delta(a, b, persistence, boundary, (left, right))
    return out


def compare_draws(
    left: str, right: str, k: int, arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float
) -> dict:
    """
    Compare two few-shot arms at one k, pairing only matching draws.

    Across-draw summaries are descriptive; bootstrap intervals stay within each draw.
    """
    out: dict = {"left": left, "right": right, "k_percent": k, "draws": {}}
    missing = [name for name in (left, right) if not arms.get(name, {}).get("by_draw")]
    if missing:
        out["error"] = f"{', '.join(missing)} has no subset runs for this species"
        return out
    deltas = []
    # Match subset runs by the shared `k001_d0` key.
    right_draws = arms[right]["by_draw"]
    for key, detail in sorted(arms[left]["by_draw"].items()):
        if detail["k_percent"] != k or key not in right_draws:
            continue
        left_p90 = detail["headline"]["travel"].get("p90_m", float("nan"))
        right_p90 = right_draws[key]["headline"]["travel"].get("p90_m", float("nan"))
        entry = {
            f"{left}_travel_p90_m": left_p90,
            f"{right}_travel_p90_m": right_p90,
            "difference_m": left_p90 - right_p90,
            "left_better": bool(left_p90 < right_p90),
        }
        a = frames.get(arms[left]["tags"].get(key))
        b = frames.get(arms[right]["tags"].get(key))
        if a is not None and b is not None:
            entry["bootstrap"] = paired_delta(a, b, persistence, boundary, (left, right))
        out["draws"][f"d{detail['draw']}"] = entry
        deltas.append(left_p90 - right_p90)
    if deltas:
        out["across_draws"] = {
            "n_draws": len(deltas),
            "mean_difference_m": float(np.mean(deltas)),
            "range_m": [float(np.min(deltas)), float(np.max(deltas))],
            "left_better_in": int(sum(1 for value in deltas if value < 0)),
            "descriptive": True,
        }
    return out


def four_way(arms: dict) -> dict:
    """Return the registered pool-composition comparison table."""
    rows = ("Z9", "Za9", "Zd3", "Zd1", "Z", "Za")
    return {
        "read_as": {
            "Z9": "nine sources, all domains",
            "Za9": "six sources, other domains only",
            "Zd3": "three sources, same domain only",
            "Zd1": "one source, same domain",
            "Z": "Tier 2: three sources, one per domain",
            "Za": "Tier 2: two sources, other domains",
        },
        "travel_p90_skill": {arm: travel_skill(arms, arm) for arm in rows if arm in arms},
    }


def floor_sensitivity(
    arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float
) -> dict[str, dict[str, float]]:
    """Recompute travel p90 at the registered A5 alternative floors."""
    out: dict[str, dict[str, float]] = {}
    for arm in FLOOR_SENSITIVITY_ARMS:
        frame = headline_frame(arms, frames, arm)
        if frame is None:
            continue
        out[arm] = {
            str(floor): S2.summarise(frame, persistence, boundary, floor)["travel"].get("p90_m")
            for floor in FLOOR_SENSITIVITY
        }
    return out


def arima_fallback_rate(frames: dict[str, pd.DataFrame]) -> dict[str, float] | None:
    """
    Estimate ARIMA's per-axis fallback-to-persistence rate from the saved predictions.

    Exact zero-displacement forecasts are indistinguishable from fallbacks in this check.
    """
    arima, persistence = frames.get("ARIMA"), frames.get("tier2:P")
    if arima is None or persistence is None:
        return None
    lon = np.isclose(arima["pred_lon"].to_numpy(float), persistence["pred_lon"].to_numpy(float))
    lat = np.isclose(arima["pred_lat"].to_numpy(float), persistence["pred_lat"].to_numpy(float))
    return {
        "lon_axis": float(lon.mean()),
        "lat_axis": float(lat.mean()),
        "both_axes": float((lon & lat).mean()),
        "either_axis": float((lon | lat).mean()),
    }


# Report
def registered_bands(report: dict) -> dict:
    """
    Apply the unchanged Tier 2 success bands to Z9/B9 with carried S and P.

    Tier 3 arms are renamed only for the existing Tier 2 band implementation.
    """
    mapping = {"Z9": "Z", "B9": "B", "S": "S", "P": "P"}
    renamed = {"evaluation_floor": FLOOR, "species": {}}
    for species, entry in report["species"].items():
        comparators = {}
        for source, target in mapping.items():
            block = entry["arms"].get(source)
            if not block or not block.get("headline"):
                continue
            comparators[target] = {"headline": block["headline"], "per_seed": block.get("per_seed", {})}
        # Express Tier 3 completeness in the Tier 2 band implementation's vocabulary.
        scored = set(comparators)
        scored |= {"Za"} if "Za9" in entry["arms"] else set()
        scored |= {"F"} if "F9" in entry["arms"] else set()
        renamed["species"][species] = {"comparators": comparators, "scored_arms": sorted(scored)}
    return {
        "read_as": "Z is Z9 and B is B9; S and P are Tier 2's carried runs, and Za/F completeness reads Za9/F9",
        "bands": S2.apply_success_bands(renamed),
    }


def across_species(report: dict) -> dict:
    """Summarise the registered two-of-three predictions across held-out species."""
    out: dict = {}
    for key, label in (
        ("1_pool_scaling_zero_shot", "Z9 beats Tier 2's Z"),
        ("2_other_domains_only", "Z9 beats Za9"),
        ("5_domain_bound_transfer", "Z9 beats Zd3"),
    ):
        wins, resolved_count, detail = 0, 0, {}
        for species, entry in report["species"].items():
            comparison = entry["predictions"].get(key, {})
            if "left_better" not in comparison:
                detail[species] = comparison.get("error", "not scored")
                continue
            wins += bool(comparison["left_better"])
            resolved_count += bool(comparison.get("resolved_against_noise_scale"))
            detail[species] = {
                "skill_difference": comparison["skill_difference"],
                "left_better": comparison["left_better"],
                "resolved_against_noise_scale": comparison.get("resolved_against_noise_scale"),
            }
        out[key] = {
            "statement": label,
            "n_species_better": wins,
            "n_species_resolved": resolved_count,
            "meets_two_of_three": bool(wins >= 2),
            "per_species": detail,
        }
    return out


def ladder_table(report: dict) -> dict:
    """Build the descriptive Section 3.1 baseline ladder."""
    blocks = {
        "training_free": ("P", "V", "ARIMA", "AVG3", "AVG10"),
        "specialist": ("XGBS", "LSTMS", "SB", "S"),
        "pooled": ("XGBB3", "XGBB9", "LSTMB3", "LSTMB9", "B", "B9"),
        "foundation_model": ("Z", "Z9", "Za9", "Zd3", "Zd1", "B9dom", "E1embed_dom"),
    }
    out: dict = {"descriptive": True, "statistic": "travel_p90_skill", "blocks": {}}
    for block, rungs in blocks.items():
        out["blocks"][block] = {
            rung: {
                species: travel_skill(entry["arms"], rung)
                for species, entry in report["species"].items()
                if rung in entry["arms"]
            }
            for rung in rungs
        }
    return out


def build_report() -> dict:
    if not PREDICTIONS.exists():
        raise SystemExit(f"FAIL: {PREDICTIONS} does not exist; run --predict first")
    report: dict = {
        "tier": "3",
        "registered_by": "preregistration_tier3.md",
        "evaluation_floor": FLOOR,
        "floor_sensitivity": list(FLOOR_SENSITIVITY),
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "tier2_numbers_read_from": str(TIER2_PREDICTIONS.relative_to(REPO)),
        "species": {},
    }
    if MANIFEST.exists():
        report["attestation"] = json.loads(MANIFEST.read_text()).get("attestation")

    for species, *_ in T.HELD_OUT:
        _, _, profile_key, domain, matching_source = T.held_out_entry(species)
        boundary = T.boundary_m(profile_key)
        persistence, frames, records = species_frames(species)
        summaries = summarise_all(frames, persistence, boundary, records)
        if "P" not in summaries:
            raise SystemExit(f"FAIL: {species} has no persistence summary; P is the skill denominator")
        persistence_headline = {
            stratum: {name: S2.headline(summaries["P"]["by_seed"], stratum, name)[0] for name in STATISTICS}
            for stratum in (*STRATA, "all")
        }

        arms: dict[str, dict] = {}
        for arm, entry in summaries.items():
            block = (
                headline_block(entry, persistence_headline)
                if entry["by_seed"]
                else {"n_seeds": None, "headline": {}, "skill": {}, "per_seed": {}}
            )
            if entry["by_draw"]:
                block["by_draw"] = draw_block(entry, persistence_headline)
            block["tags"] = {tag_key(key): tag for key, tag in entry["tags"].items()}
            arms[arm] = block

        entry = {
            "domain": domain,
            "matching_tier2_source": matching_source,
            "regime_boundary_m": boundary,
            "n_test_individuals": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "scored_arms": sorted(arms),
            "arms": arms,
            "arima_fallback_rate": arima_fallback_rate(frames),
            "floor_sensitivity": floor_sensitivity(arms, frames, persistence, boundary),
        }
        entry["predictions"] = {
            "1_pool_scaling_zero_shot": compare_arms("Z9", "Z", arms, frames, persistence, boundary),
            "2_other_domains_only": compare_arms("Z9", "Za9", arms, frames, persistence, boundary),
            "3_pretrained_vs_scratch_1pc": compare_draws("F9", "Sk", 1, arms, frames, persistence, boundary),
            "3_higher_data_point_10pc": compare_draws("F9", "Sk", 10, arms, frames, persistence, boundary),
            "4_pool_scaling_few_shot_1pc": compare_draws("F9", "F3c", 1, arms, frames, persistence, boundary),
            "4_higher_data_point_10pc": compare_draws("F9", "F3c", 10, arms, frames, persistence, boundary),
            "5_domain_bound_transfer": compare_arms("Z9", "Zd3", arms, frames, persistence, boundary),
            "5_four_way_table": four_way(arms),
        }
        report["species"][species] = entry

    report["across_species"] = across_species(report)
    report["verdict"] = registered_bands(report)
    report["ladder"] = ladder_table(report)
    return report


# Markdown
def _s(value) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:+.4f}"


def _m(value) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:,.0f}"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# Tier 3 sealed results",
        "",
        f"Registered by `preregistration_tier3.md`. Evaluation floor F = {report['evaluation_floor']}, median "
        f"across training seeds, paired bootstrap of {S2.N_BOOTSTRAP:,} test-individual resamples at seed "
        f"{S2.BOOTSTRAP_SEED}. A gap smaller than the {report['noise_scale']} single-seed noise scale is reported "
        "as unresolved, and no interval creates a second pass/fail rule.",
        "",
        f"Tier 2's numbers are read from `{report['tier2_numbers_read_from']}` and are not recomputed here.",
        "",
    ]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"Tier 3 freeze committed at `{attestation['freeze_commit'][:12]}`.", ""]

    lines += ["## Registered predictions", ""]
    for key, entry in report["across_species"].items():
        lines += [
            f"**{entry['statement']}** ({key}): better for {entry['n_species_better']} of 3 species, "
            f"{entry['n_species_resolved']} resolved against the noise scale. Two-of-three rule "
            f"{'met' if entry['meets_two_of_three'] else 'not met'}.",
            "",
            "| species | skill difference | better | resolved |",
            "| --- | ---: | --- | --- |",
        ]
        for species, detail in entry["per_species"].items():
            if isinstance(detail, str):
                lines.append(f"| {species} | -- | {detail} | -- |")
                continue
            lines.append(
                f"| {species} | {_s(detail['skill_difference'])} | "
                f"{'yes' if detail['left_better'] else 'no'} | "
                f"{'yes' if detail['resolved_against_noise_scale'] else 'no'} |"
            )
        lines.append("")

    lines += ["## Few-shot comparisons, per draw", ""]
    for species, entry in report["species"].items():
        for key in (
            "3_pretrained_vs_scratch_1pc",
            "3_higher_data_point_10pc",
            "4_pool_scaling_few_shot_1pc",
            "4_higher_data_point_10pc",
        ):
            comparison = entry["predictions"].get(key, {})
            if not comparison.get("draws"):
                lines += [f"**{species}, {key}**: {comparison.get('error', 'not scored')}", ""]
                continue
            left, right = comparison["left"], comparison["right"]
            lines += [
                f"**{species}, {left} against {right} at k = {comparison['k_percent']}%** ({key})",
                "",
                f"| draw | {left} travel p90 (m) | {right} travel p90 (m) | difference (m) | interval excludes 0 |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
            for draw, detail in comparison["draws"].items():
                interval = (
                    detail.get("bootstrap", {}).get("series", {}).get("travel_p90_m", {}).get("difference_m", {})
                )
                lines.append(
                    f"| {draw} | {_m(detail[f'{left}_travel_p90_m'])} | {_m(detail[f'{right}_travel_p90_m'])} | "
                    f"{_m(detail['difference_m'])} | {'yes' if interval.get('excludes_zero') else 'no'} |"
                )
            across = comparison.get("across_draws", {})
            if across:
                lines += [
                    "",
                    f"Descriptive across draws: mean {_m(across['mean_difference_m'])} m, range "
                    f"{_m(across['range_m'][0])} to {_m(across['range_m'][1])} m; {left} better in "
                    f"{across['left_better_in']} of {across['n_draws']} draws.",
                ]
            lines.append("")

    rows = ("Z9", "Za9", "Zd3", "Zd1", "Z", "Za")
    lines += [
        "## Pool composition, travel p90 skill",
        "",
        "| species | " + " | ".join(rows) + " |",
        "| --- | " + " | ".join(["---:"] * len(rows)) + " |",
    ]
    for species, entry in report["species"].items():
        table = entry["predictions"]["5_four_way_table"]["travel_p90_skill"]
        lines.append(f"| {species} | " + " | ".join(_s(table.get(arm)) for arm in rows) + " |")
    lines.append("")

    species_order = list(report["species"])
    lines += [
        "## Baseline ladder, travel p90 skill (descriptive)",
        "",
        "| block | rung | " + " | ".join(species_order) + " |",
        "| --- | --- | " + " | ".join(["---:"] * len(species_order)) + " |",
    ]
    for block, rungs in report["ladder"]["blocks"].items():
        for rung, values in rungs.items():
            if not values:
                continue
            lines.append(
                f"| {block} | {rung} | " + " | ".join(_s(values.get(species)) for species in species_order) + " |"
            )
    lines.append("")

    verdict = report["verdict"]
    lines += [
        "## Success bands",
        "",
        f"Tier 2 section 4 unchanged. {verdict['read_as']}.",
        "",
        f"**Band: {verdict['bands']['band']}**",
        "",
        "| species | " + " | ".join(sorted(next(iter(verdict["bands"]["per_species"].values()))["checks"])) + " |",
        "| --- | " + " | ".join(
            ["---"] * len(next(iter(verdict["bands"]["per_species"].values()))["checks"])
        ) + " |",
    ]
    for species, detail in verdict["bands"]["per_species"].items():
        lines.append(
            f"| {species} | "
            + " | ".join(
                {True: "yes", False: "no", None: "--"}[detail["checks"][check]]
                for check in sorted(detail["checks"])
            )
            + " |"
        )
    if verdict["bands"].get("deferred_comparators"):
        lines += ["", f"Deferred or missing: {', '.join(verdict['bands']['deferred_comparators'])}."]
    lines.append("")

    lines += ["## Per-species detail", ""]
    order = {arm: index for index, arm in enumerate(ARM_ORDER)}
    for species, entry in report["species"].items():
        lines += [
            f"### {species} ({entry['domain']})",
            "",
            f"{entry['n_test_individuals']} test individuals, {entry['n_test_windows']:,} windows, regime boundary "
            f"{entry['regime_boundary_m']:,.1f} m."
            + (
                " ARIMA fell back to persistence on both axes in "
                f"{entry['arima_fallback_rate']['both_axes']:.1%} of windows and on at least one axis in "
                f"{entry['arima_fallback_rate']['either_axis']:.1%}."
                if entry.get("arima_fallback_rate") is not None
                else ""
            ),
            "",
            "| arm | seeds | travel p90 (m) | travel p90 skill | all-window mean skill |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm in sorted(entry["arms"], key=lambda name: (order.get(name, len(ARM_ORDER)), name)):
            block = entry["arms"][arm]
            if not block.get("headline"):
                continue
            lines.append(
                f"| {arm} | {block.get('n_seeds') or '--'} | {_m(block['headline']['travel'].get('p90_m'))} | "
                f"{_s(block['skill']['travel'].get('p90_m'))} | "
                f"{_s(block['skill']['all'].get('mean_m_individual_weighted'))} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")

# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="score the arms on test and save predictions")
    parser.add_argument("--report", action="store_true", help="build the report from the saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score arms that already have prediction files")
    args = parser.parse_args()
    if not (args.predict or args.report):
        raise SystemExit("pass --predict, --report, or both")

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        run_predictions(device, overwrite=args.overwrite)
    if args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report()
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}\n")
        print(f"band             : {report['verdict']['bands']['band']}")
        for key, entry in report["across_species"].items():
            print(
                f"{key:<30} better for {entry['n_species_better']}/3, {entry['n_species_resolved']} resolved, "
                f"two-of-three {'met' if entry['meets_two_of_three'] else 'not met'}"
            )


if __name__ == "__main__":
    main()
