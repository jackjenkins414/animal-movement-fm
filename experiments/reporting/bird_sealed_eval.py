# GenAI reference: as with other scripts in the project, we used GenAI to generate this, but detailed our it from preregistration_birds.md: 
# the gull's carried arms from Tiers 2-3, P/V/ARIMA/AVG computed for the goose and dove, the Fb-Sk draw identity check, the goose exclusion 
# check, the five reading rules with their clarifications etc; GenAI added some extra things as well. We checked the code and added our own 
# comments or docstrings here and there, but largely kepted the original comments and docs from GenAI. 
"""
Sealed evaluation of the bird-only study (`preregistration_birds.md`).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_train as BT  # noqa: E402
import tier2_final_train as T  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402
import tier3_sealed_eval as E3  # noqa: E402
import tier3_train as T3  # noqa: E402
import tier4_sealed_eval as E4  # noqa: E402

from animal_movement_fm.data.benchmark import load_split_frame  # noqa: E402
from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.data.split import load_split  # noqa: E402
from animal_movement_fm.eval.fm_predictor import (  # noqa: E402
    MovementFMPredictor,
    WindowModelPredictor,
    evaluate_batched,
)
from animal_movement_fm.eval.harness import evaluate  # noqa: E402

PREREG = REPO / "preregistration_birds.md"
AS_TRAINED_HEADING = "As trained"

PLACEHOLDER = re.compile(r"\b(?:to|will)\s+fill\b|\bdraft\b", re.IGNORECASE)

BIRD_RESULTS = BT.RESULTS
TIER2_PREDICTIONS = S2.PREDICTIONS
TIER3_PREDICTIONS = E3.PREDICTIONS
TIER3_RESULTS = E3.TIER3_RESULTS
SEALED = REPO / "results" / "birds_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_results.json"
REPORT_MD = SEALED / "birds_results.md"
MANIFEST = SEALED / "birds_sealed_manifest.json"

FLOOR = E3.FLOOR
NOISE_SCALE = E3.NOISE_SCALE

GULL, GOOSE, DOVE = BT.HELD_OUT
# preregistration_birds.md sec 4: carried for the gull, never re-predicted
CARRIED_TIER2 = ("P", "V", "S")
CARRIED_TIER3 = ("ARIMA", "AVG3", "AVG10", "SB", "Sk", "LSTMS", "XGBS")
if set(CARRIED_TIER2) | set(CARRIED_TIER3) != BT.CARRIED_GULL_ARMS:
    raise RuntimeError("the evaluator's gull carry list disagrees with bird_train.CARRIED_GULL_ARMS")
# computed here, for the goose and the dove only
TRAINING_FREE = ("P", "V", "ARIMA", "AVG3", "AVG10")
# every bird is read against the same arms: its own or carried specialists, the rungs, and the study's trained arms
REQUIRED_ARMS = frozenset(BT.ARMS) | frozenset(TRAINING_FREE)
SUBSET_ARMS = ("Sk", "Fb")
PAIRED_FRACTIONS = T3.SK_FRACTIONS  # Fb at 100% has no Sk: descriptive only, beside S
DILUTION_ARMS = ("Zb", "Bb", "Zbc", "Bbc")

# From the PBS job logs, as reported on 27-09-2026; none of these is in the run records.
AS_TRAINED_FROM_PBS = {
    "exit status": "0 for every run",
    "queue and precision": "gpuvolta, fp32, torch 2.5.1+cu121",
    "compute": "458.0 SU; longest run 11m11s",
}

# Decided before the seal (27-09-2026); printed into the §7 block by --diagnostic.
CLARIFICATIONS = (
    ("Zb, Bb vs S", "S is the transformer specialist: Tier 2's S for the gull, the bird S for the goose and dove. "
                    "LSTM-S, XGB-S and SB appear in the ladder only."),
    ("vs S outcomes", "Two, as written: within 0.05 or above = matches; S ahead by more than 0.05 = specialist "
                      "wins. The gap is always printed."),
    ("Fb vs Sk", "Read per draw; the count across draws is descriptive (Tier 3 freeze sec 6.1)."),
    ("Fb at k = 100%", "Descriptive only, shown beside S; there is no Sk at 100%."),
    ("dove Δ rows", "adding five unrelated birds helps / changes nothing / interacts with the package / unresolved; "
                    "same thresholds."),
    ("floor sensitivity", "Not reported."),
)

CONGENER_WORDING = {
    "carry": "congeners carry the transfer",
    "nothing": "congeners add nothing",
    "interact": "the package and the congeners interact",
    "unresolved": "unresolved",
}
DOVE_WORDING = {
    "carry": "adding five unrelated birds helps",
    "nothing": "adding five unrelated birds changes nothing",
    "interact": "adding five unrelated birds interacts with the package",
    "unresolved": "unresolved",
}

LADDER_BLOCKS = {
    "training_free": ("P", "V", "ARIMA", "AVG3", "AVG10"),
    "specialist": ("XGBS", "LSTMS", "SB", "S"),
    "pooled": ("XGBBb", "LSTMBb", "Bb", "Bbc"),
    "foundation_model": ("Zb", "Zbc"),
}


# Seal
def check_as_trained(body: str) -> None:
    """Refuse an empty §7, or one that still carries placeholder wording."""
    if not body.strip():
        raise SystemExit(f"FAIL: {PREREG.name} '{AS_TRAINED_HEADING}' is empty in the committed version.")
    found = PLACEHOLDER.search(body)
    if found:
        raise SystemExit(
            f"FAIL: {PREREG.name} '{AS_TRAINED_HEADING}' still says {found.group(0)!r} in the committed version. Fill "
            "it from --diagnostic, commit it, then run the sealed evaluation."
        )


def as_trained_section(text: str) -> str:
    try:
        return E4.as_trained_section(text)
    except SystemExit:
        raise SystemExit(f"FAIL: {PREREG.name} has no '{AS_TRAINED_HEADING}' heading.") from None


def assert_as_trained_filled() -> str:
    """§7 as committed at HEAD, not as in the working tree."""
    shown = E4.git("show", f"HEAD:{PREREG.name}")
    if shown.returncode != 0:
        raise SystemExit(f"FAIL: cannot read {PREREG.name} from HEAD, so it is not committed: {shown.stderr.strip()}")
    body = as_trained_section(shown.stdout)
    check_as_trained(body)
    return body


def assert_sealed() -> dict:
    """
    The preregistration is tracked, unchanged against HEAD and has §7 filled. Run by --predict only.

    A manifest from an earlier predict run keeps its attestation: a later run under a different blob is refused rather
    than silently re-attesting the partitions to another version of the file.
    """
    previous_freeze = S2.FREEZE
    S2.FREEZE = PREREG
    try:
        attestation = S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous_freeze
    attestation["as_trained_lines"] = len(assert_as_trained_filled().splitlines())
    if MANIFEST.exists():
        earlier = json.loads(MANIFEST.read_text()).get("attestation") or {}
        if earlier.get("freeze_blob") and earlier["freeze_blob"] != attestation["freeze_blob"]:
            raise SystemExit(
                f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but {PREREG.name} is now "
                f"blob {attestation['freeze_blob'][:12]}. Rebuild the report with --report instead."
            )
    return attestation


# Records
def is_birds(record: dict) -> bool:
    return record.get("tier") == "birds" and "run_id" in record and "arm" in record


def read_records() -> list[dict]:
    """Every bird run record; XGBoost metadata files are skipped silently."""
    found = []
    for path in sorted(BIRD_RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if not is_birds(record):
            if not path.name.endswith("_xgb_meta.json"):
                print(f"  skipped {path.name}: not a bird run record (tier {record.get('tier')!r})")
            continue
        if record["arm"] not in BT.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record['arm']}, which is not a bird arm")
        found.append(record)
    return found


def record_map(records: list[dict] | None = None) -> dict[str, dict]:
    return {record["run_id"]: record for record in (read_records() if records is None else records)}


def run_problems(run, record: dict | None) -> list[str]:
    """What stops one planned run counting as finished cleanly."""
    if record is None:
        return ["no record"]
    if BT.ARMS[run.arm].get("xgb"):
        boosters = [BIRD_RESULTS / f"{run.run_id}_xgb_{axis}.ubj" for axis in ("east", "north")]
        return [] if all(path.exists() for path in boosters) else ["booster missing"]
    problems = []
    steps = (record.get("config") or {}).get("steps")
    if record.get("applied_updates") != BT.REGISTERED_STEPS or steps != BT.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {steps}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (BIRD_RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def completeness(records: dict[str, dict] | None = None) -> dict:
    """Every planned run has a record and a checkpoint, and every neural run applied its full budget cleanly."""
    records = record_map() if records is None else records
    by_arm: dict[str, dict] = {}
    problems = []
    for run in BT.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id}: {'; '.join(issues)}")
    neural = [run for run in BT.PLAN if not BT.ARMS[run.arm].get("xgb")]
    return {
        "n_planned": len(BT.PLAN),
        "n_records": sum(row["records"] for row in by_arm.values()),
        "n_clean": sum(row["clean"] for row in by_arm.values()),
        "n_neural": len(neural),
        "n_xgb": len(BT.PLAN) - len(neural),
        "by_arm": by_arm,
        "problems": problems,
        "complete": not problems,
    }


# Boundaries
def check_boundaries() -> dict[str, float]:
    """The pinned boundaries of prereg §2, checked by bird_train against its profile tables, never Gadi's shared one."""
    if BT.IL.PROFILES_CSV in BT.PROFILE_TABLES.values():
        raise SystemExit("FAIL: a bird boundary would be checked against results/regime_profiles.csv")
    return BT.check_boundaries()


# Draw identity and goose exclusions: training splits and records only
def subset_metadata(subset) -> dict:
    return {
        "draw_seed": int(subset.seed),
        "n_individuals": int(subset.n_individuals),
        "n_individuals_available": int(subset.n_available),
    }


def tier3_record(run_id: str) -> dict | None:
    path = TIER3_RESULTS / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def assert_draw_identity(payload: dict, records: dict[str, dict]) -> dict:
    """
    Fb and Sk train on identical individuals at every paired (k, draw), and every record says what was recomputed.

    Membership is not stored as ids anywhere; it follows from the frozen split, the registered exclusions and
    nested_subsets(draw). The gull's Sk is Tier 3's, recomputed with Tier 3's own subset function and read from its
    Tier 3 record.
    """
    problems, pairs, fb_checked = [], 0, 0
    for species in BT.HELD_OUT:
        for k, draws in T.FINETUNE_DRAWS_PER_FRACTION.items():
            for draw in range(draws):
                label = f"{species} k{k} d{draw}"
                fb_run = BT.Run("Fb", species, T.FINETUNE_INIT_SEED, k, draw)
                fb = BT.finetune_subset(fb_run, payload)
                checks = [("Fb", fb, records.get(fb_run.run_id))]
                if k in PAIRED_FRACTIONS:
                    if species == GULL:
                        sk_run = T3.Run("Sk", species, 0, k, draw)
                        sk, sk_record = T.finetune_subset(sk_run), tier3_record(sk_run.run_id)
                    else:
                        sk_run = BT.Run("Sk", species, 0, k, draw)
                        sk, sk_record = BT.finetune_subset(sk_run, payload), records.get(sk_run.run_id)
                    if [str(i) for i in fb.individuals] != [str(i) for i in sk.individuals]:
                        problems.append(f"{label}: Fb and Sk train on different individuals")
                    checks.append(("Sk", sk, sk_record))
                    pairs += 1
                for name, subset, record in checks:
                    if record is None:
                        problems.append(f"{label}: no {name} record")
                        continue
                    stored = record.get("subset") or {}
                    expected = subset_metadata(subset)
                    got = {key: stored.get(key) for key in expected}
                    if got != expected:
                        problems.append(f"{label}: {name} record says {got}, recomputed {expected}")
                fb_checked += 1
    if problems:
        raise SystemExit("FAIL: draw identity\n  " + "\n  ".join(problems))
    return {"status": "PASS", "paired_draws": pairs, "fb_draws": fb_checked}


def assert_goose_exclusions(payload: dict, records: dict[str, dict]) -> dict:
    """
    None of the six registered goose birds is in any goose training set (prereg §3).

    Records hold no id lists, so the evidence is: each record names the six, each subset drew from the 293, and each
    full-set run's window count is the count without the six, which differs from the count with them.
    """
    six = {str(i) for i in payload["training_exclusions"].get(GOOSE, [])}
    if len(six) != 6:
        raise SystemExit(f"FAIL: the registered goose exclusions hold {len(six)} ids, not 6")
    kept = BT.held_out_train_ids(GOOSE, payload)  # refuses an exclusion that is not a training bird
    if six & {str(i) for i in kept}:
        raise SystemExit("FAIL: an excluded goose is still in the goose training list")
    full = list(load_split(T.SPLITS / f"{GOOSE}_split_seed42.json").train)
    kept_windows = T._load_species_windows(GOOSE, GOOSE, "train", kept).n_windows
    full_windows = T._load_species_windows(GOOSE, GOOSE, "train", full).n_windows
    if kept_windows == full_windows:
        raise SystemExit("FAIL: the six carry no windows, so window counts cannot show they were left out")

    problems = []
    runs = [run for run in BT.PLAN if run.species == GOOSE]
    for run in runs:
        record = records.get(run.run_id)
        if record is None:
            problems.append(f"{run.run_id}: no record")
            continue
        if {str(i) for i in record.get("excluded_individuals") or []} != six:
            problems.append(f"{run.run_id}: record names {record.get('excluded_individuals')}, not the six")
        spec = BT.ARMS[run.arm]
        if spec["pool"] == "held_out_subset":
            if six & {str(i) for i in BT.finetune_subset(run, payload).individuals}:
                problems.append(f"{run.run_id}: an excluded goose is in the recomputed subset")
            available = (record.get("subset") or {}).get("n_individuals_available")
            if available != len(kept):
                problems.append(f"{run.run_id}: subset drawn from {available} birds, not {len(kept)}")
        else:
            field = "training_windows" if spec.get("xgb") else "pool_windows"
            if record.get(field) != kept_windows:
                problems.append(f"{run.run_id}: {field} {record.get(field)}, expected {kept_windows} "
                                f"({full_windows} with the six)")
    if problems:
        raise SystemExit("FAIL: goose exclusions\n  " + "\n  ".join(problems))
    return {
        "status": "PASS", "runs": len(runs), "training_birds": len(kept),
        "windows_without_the_six": kept_windows, "windows_with_the_six": full_windows,
    }


# Prediction
def test_frame(species: str) -> pd.DataFrame:
    """The only reader of a test partition. Called from predict_species, after assert_sealed."""
    if species not in BT.HELD_OUT:
        raise SystemExit(f"FAIL: {species} is not a held-out bird")
    split = load_split(S2.SPLITS / f"{species}_split_seed42.json")
    frame = load_split_frame(S2.PROCESSED / f"{species}_segments.parquet", split.test)
    return frame[["individual_id", "bin_index", "timestamp", "lon", "lat"]].reset_index(drop=True)


def training_free_models() -> list[tuple[str, object]]:
    """P and V as Tier 2 built them, ARIMA and AVG3/10 as Tier 3 did."""
    first = [(tag, model) for tag, _, model in S2.comparator_jobs("", {})]
    return first + [(tag, build()) for tag, build in E3.LADDER_FREE]


def prediction_path(species: str, tag: str) -> Path:
    return PREDICTIONS / f"{species}__{tag}.parquet"


def predict_species(species: str, records: list[dict], device: torch.device, overwrite: bool) -> list[str]:
    frame = test_frame(species)
    print(f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
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

    if species != GULL:  # the gull's training-free rungs are carried from Tiers 2 and 3
        for tag, model in training_free_models():
            if not done(tag):
                started = time.time()
                save(tag, evaluate(model, frame, context_length=CONTEXT_LENGTH, show_progress=False), started)

    for record in records:
        attached = record.get("species")
        if attached and attached != species:
            continue
        tag = record["run_id"]
        if done(tag):
            continue
        started = time.time()
        if BT.ARMS[record["arm"]].get("xgb"):
            from animal_movement_fm.models.xgboost import WindowXGBoost

            predictor = WindowModelPredictor(WindowXGBoost.load(BIRD_RESULTS / f"{tag}_xgb"), name=tag,
                                             sigma=SCALE_FLOOR_M)
        else:
            loaded = E3.load_reported(BIRD_RESULTS, tag)
            if loaded is None:
                print(f"  {tag}: no reported checkpoint, skipped")
                continue
            predictor = MovementFMPredictor(loaded[0], device=device, sigma=SCALE_FLOOR_M)
        save(tag, evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH), started)
    return written


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    payload = BT.load_registered_pool()
    records = read_records()
    rmap = record_map(records)
    gap = completeness(rmap)
    identity = assert_draw_identity(payload, rmap)
    exclusions = assert_goose_exclusions(payload, rmap)
    check_boundaries()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = {"attestation": attestation, "completeness": gap, "draw_identity": identity,
                "goose_exclusions": exclusions, "species": {}}
    for species in BT.HELD_OUT:
        manifest["species"][species] = predict_species(species, records, device, overwrite)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames and decoding
def species_frames(species: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """P, the gull's carried predictions (gull only) and this study's predictions, all aligned to P."""
    home = TIER2_PREDICTIONS if species == GULL else PREDICTIONS
    persistence_path = home / f"{species}__P.parquet"
    if not persistence_path.exists():
        raise SystemExit(f"FAIL: {persistence_path} is missing; P defines the strata and the skill denominator")
    persistence = E3.load_frame(persistence_path)

    frames: dict[str, pd.DataFrame] = {}
    if species == GULL:
        for directory, prefix, carried in ((TIER2_PREDICTIONS, "tier2:", CARRIED_TIER2),
                                           (TIER3_PREDICTIONS, "tier3:", CARRIED_TIER3)):
            for path in sorted(directory.glob(f"{species}__*.parquet")):
                tag = path.stem.split("__", 1)[1]
                split = E4.split_carried_tag(tag)
                if split is not None and split[0] in carried:
                    frames[f"{prefix}{tag}"] = E3.load_frame(path)
    if PREDICTIONS.exists():
        for path in sorted(PREDICTIONS.glob(f"{species}__*.parquet")):
            frames[path.stem.split("__", 1)[1]] = E3.load_frame(path)
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{species} {tag}")
    return persistence, frames, record_map()


def parse_tag(tag: str, records: dict[str, dict]) -> dict | None:
    """Bird run ids from their records; the gull's carried tags through Tier 4's decoder; the goose's and dove's rungs."""
    if tag in records:
        record = records[tag]
        return {"arm": record["arm"], "seed": record["seed"], "k": record.get("k_percent"), "draw": record.get("draw")}
    if tag.startswith(("tier2:", "tier3:")):
        prefix, bare = tag.split(":", 1)
        split = E4.split_carried_tag(bare)
        allowed = CARRIED_TIER2 if prefix == "tier2" else CARRIED_TIER3
        if split is None or split[0] not in allowed:
            print(f"  ignored {tag}: not an arm carried for the gull (prereg sec 4)")
            return None
        return E4.parse_tag(tag, {})
    if tag in TRAINING_FREE:
        return {"arm": tag, "seed": None, "k": None, "draw": None}
    print(f"  ignored {tag}: not a bird run, a carried gull arm or a training-free rung")
    return None


# Guards
def assert_no_collisions(frames: dict, records: dict) -> None:
    E4.assert_no_collisions(frames, records, parse=parse_tag)


def missing_arms(summaries: dict, species: str) -> list[tuple[str, str]]:
    return [(arm, species) for arm in sorted(REQUIRED_ARMS) if arm not in summaries]


def registered_draws(arm: str, species: str) -> set[tuple[int, int]]:
    """The gull's Sk from Tier 3's plan (carried); every other subset run from bird_train's plan."""
    plan = T3.PLAN if (arm == "Sk" and species == GULL) else BT.PLAN
    return {(run.k_percent, run.draw) for run in plan if run.arm == arm and run.species == species}


def assert_draws_complete(summaries: dict, species: str) -> None:
    problems = []
    for arm in SUBSET_ARMS:
        expected = registered_draws(arm, species)
        if not expected:
            raise SystemExit(f"FAIL: no registered draws found for {arm} / {species}")
        present = set(summaries.get(arm, {}).get("by_draw", {}))
        if present != expected:
            problems.append(f"{arm} / {species}: missing {sorted(expected - present)}, "
                            f"unexpected {sorted(present - expected)}")
    if problems:
        raise SystemExit("FAIL: subset arms do not hold their registered draws:\n  " + "\n  ".join(problems))


# Readings, prereg sec 5. Every rule returns a reading string.
def band(gap: float, labels: tuple[str, str, str]) -> str:
    """Above +0.05 / within / below -0.05, with 0.05 itself counting as within."""
    if gap is None or not np.isfinite(gap):
        return "not scored"
    if not E3.resolved(gap):
        return labels[1]
    return labels[0] if gap > 0 else labels[2]


def read_vs_persistence(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    out = {}
    for arm in ("Zb", "Bb"):
        comparison = E3.compare_arms(arm, "P", arms, frames, persistence, boundary)
        skill = E3.travel_skill(arms, arm)
        comparison["reading"] = band(skill, ("beats persistence zero-shot", "matches persistence zero-shot",
                                             "loses to persistence zero-shot"))
        out[arm] = comparison
    return out


def read_vs_specialist(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    """Two outcomes as registered; the gap is always printed."""
    out = {}
    for arm in ("Zb", "Bb"):
        comparison = E3.compare_arms(arm, "S", arms, frames, persistence, boundary)
        gap = E3.travel_skill(arms, arm) - E3.travel_skill(arms, "S")
        if not np.isfinite(gap):
            reading = "not scored"
        else:
            reading = "zero-shot matches the specialist" if gap >= -NOISE_SCALE else "the specialist wins"
        comparison |= {"gap": gap, "reading": f"{reading} (gap {gap:+.4f})" if np.isfinite(gap) else reading}
        out[arm] = comparison
    return out


def read_package(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    comparison = E3.compare_arms("Zb", "Bb", arms, frames, persistence, boundary)
    comparison["reading"] = band(E4.delta(arms, "Zb", "Bb"),
                                 ("the package helps", "no difference", "the package hurts"))
    return comparison


def classify_congeners(delta_z: float, delta_b: float, species: str) -> str:
    """The four registered rows. Deltas are with the five added minus without, so positive means they help."""
    words = DOVE_WORDING if species == DOVE else CONGENER_WORDING
    if not (np.isfinite(delta_z) and np.isfinite(delta_b)):
        return "not scored"
    z_resolved, b_resolved = E3.resolved(delta_z), E3.resolved(delta_b)
    if z_resolved and b_resolved and delta_z > 0 and delta_b > 0:
        return words["carry"]
    if not z_resolved and not b_resolved:
        return words["nothing"]
    if (z_resolved and delta_z > 0 and not b_resolved) or (b_resolved and delta_b > 0 and not z_resolved):
        return words["interact"]
    return words["unresolved"]


def read_congeners(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float, species: str) -> dict:
    delta_z, delta_b = E4.delta(arms, "Zbc", "Zb"), E4.delta(arms, "Bbc", "Bb")
    return {
        "delta_Z": delta_z,
        "delta_B": delta_b,
        "Zbc_vs_Zb": E3.compare_arms("Zbc", "Zb", arms, frames, persistence, boundary),
        "Bbc_vs_Bb": E3.compare_arms("Bbc", "Bb", arms, frames, persistence, boundary),
        "reading": classify_congeners(delta_z, delta_b, species),
    }


def draw_skill(detail: dict) -> float:
    return detail.get("skill", {}).get("travel", {}).get("p90_m", float("nan"))


def read_few_shot(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    """Fb vs Sk read per draw on the skill gap; the count across draws is descriptive. k = 100% sits beside S."""
    fb_draws = arms.get("Fb", {}).get("by_draw", {})
    sk_draws = arms.get("Sk", {}).get("by_draw", {})
    out: dict = {}
    for k in PAIRED_FRACTIONS:
        comparison = E3.compare_draws("Fb", "Sk", k, arms, frames, persistence, boundary)
        readings = {}
        for key, fb in sorted(fb_draws.items()):
            if fb["k_percent"] != k or key not in sk_draws:
                continue
            gap = draw_skill(fb) - draw_skill(sk_draws[key])
            readings[key] = {
                "Fb_travel_p90_skill": draw_skill(fb),
                "Sk_travel_p90_skill": draw_skill(sk_draws[key]),
                "skill_gap": gap,
                "reading": band(gap, ("pretraining helps", "no difference", "pretraining loses")),
            }
        comparison["per_draw"] = readings
        counts = {label: 0 for label in ("pretraining helps", "no difference", "pretraining loses", "not scored")}
        for entry in readings.values():
            counts[entry["reading"]] += 1
        comparison["across_draws"] = {"counts": counts, "descriptive": True}
        out[f"k{k:03d}"] = comparison
    full = {key: fb for key, fb in fb_draws.items() if fb["k_percent"] == 100}
    out["k100"] = {
        "descriptive": True,
        "Fb_travel_p90_skill": {key: draw_skill(fb) for key, fb in full.items()},
        "S_travel_p90_skill": E3.travel_skill(arms, "S"),
        "reading": "descriptive only; no Sk at 100%, shown beside S",
    }
    return out


def ladder_table(report: dict) -> dict:
    out: dict = {"statistic": "travel_p90_skill", "reading": "descriptive; no pass/fail", "blocks": {}}
    for block, rungs in LADDER_BLOCKS.items():
        out["blocks"][block] = {
            rung: {species: E3.travel_skill(entry["arms"], rung) for species, entry in report["species"].items()}
            for rung in rungs
        }
    return out


# Report
def build_report() -> dict:
    if not PREDICTIONS.exists():
        raise SystemExit(f"FAIL: {PREDICTIONS} does not exist; run --predict first")
    boundaries = check_boundaries()
    report: dict = {
        "study": "birds",
        "registered_by": PREREG.name,
        "evaluation_floor": FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "clarifications": [{"item": item, "resolution": text} for item, text in CLARIFICATIONS],
        "completeness": completeness(),
        "species": {},
    }
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        report["attestation"] = manifest.get("attestation")
        report["draw_identity"] = manifest.get("draw_identity")
        report["goose_exclusions"] = manifest.get("goose_exclusions")

    for species in BT.HELD_OUT:
        boundary = boundaries[species]
        persistence, frames, records = species_frames(species)
        assert_no_collisions(frames, records)
        summaries = E3.summarise_all(frames, persistence, boundary, records, parse=parse_tag)
        missing = missing_arms(summaries, species)
        if missing:
            raise SystemExit("FAIL: arms with no summary, so readings would be scored without them:\n  "
                             + "\n  ".join(f"{arm} / {sp}" for arm, sp in missing))
        assert_draws_complete(summaries, species)
        print("\n".join(E4.placement_lines(summaries, species)), flush=True)

        persistence_headline = {
            stratum: {name: S2.headline(summaries["P"]["by_seed"], stratum, name)[0] for name in E3.STATISTICS}
            for stratum in (*E3.STRATA, "all")
        }
        arms: dict[str, dict] = {}
        for arm, entry in summaries.items():
            block = (
                E3.headline_block(entry, persistence_headline)
                if entry["by_seed"]
                else {"n_seeds": None, "headline": {}, "skill": {}, "per_seed": {}}
            )
            if entry["by_draw"]:
                block["by_draw"] = E3.draw_block(entry, persistence_headline)
            block["tags"] = {E3.tag_key(key): tag for key, tag in entry["tags"].items()}
            arms[arm] = block

        report["species"][species] = {
            "regime_boundary_m": boundary,
            "n_test_individuals": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "scored_arms": sorted(arms),
            "arms": arms,
            "readings": {
                "1_vs_persistence": read_vs_persistence(arms, frames, persistence, boundary),
                "2_vs_specialist": read_vs_specialist(arms, frames, persistence, boundary),
                "3_package": read_package(arms, frames, persistence, boundary),
                "4_congeners": read_congeners(arms, frames, persistence, boundary, species),
                "5_few_shot": read_few_shot(arms, frames, persistence, boundary),
            },
        }
    report["ladder"] = ladder_table(report)
    return report


def _s(value) -> str:
    return f"{value:+.4f}" if value is not None and np.isfinite(value) else "—"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Bird-only study: sealed evaluation", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{PREREG.name}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"Runs: {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean. "
              f"Statistic: travel p90 skill at F = {report['evaluation_floor']}, median of seeds; "
              f"±{report['noise_scale']} is no difference.", ""]
    for species, entry in report["species"].items():
        readings = entry["readings"]
        lines += [f"## {species}", "", "| rule | arm | value | reading |", "|---|---|---:|---|"]
        for arm, block in readings["1_vs_persistence"].items():
            lines.append(f"| vs persistence | {arm} | {_s(block.get(f'{arm}_travel_p90_skill'))} | {block['reading']} |")
        for arm, block in readings["2_vs_specialist"].items():
            lines.append(f"| vs S | {arm} | {_s(block.get('gap'))} | {block['reading']} |")
        package = readings["3_package"]
        lines.append(f"| package | Zb − Bb | {_s(package.get('skill_difference'))} | {package['reading']} |")
        congeners = readings["4_congeners"]
        lines.append(f"| congeners | Δ_Z / Δ_B | {_s(congeners['delta_Z'])} / {_s(congeners['delta_B'])} | "
                     f"{congeners['reading']} |")
        for k, block in readings["5_few_shot"].items():
            if k == "k100":
                values = ", ".join(_s(v) for v in block["Fb_travel_p90_skill"].values())
                lines.append(f"| Fb at 100% | Fb / S | {values} / {_s(block['S_travel_p90_skill'])} | "
                             f"{block['reading']} |")
                continue
            for key, draw in block["per_draw"].items():
                lines.append(f"| Fb vs Sk | {key} | {_s(draw['skill_gap'])} | {draw['reading']} |")
            counts = ", ".join(f"{n} {label}" for label, n in block["across_draws"]["counts"].items() if n)
            lines.append(f"| Fb vs Sk | {k} across draws | — | descriptive: {counts} |")
        lines.append("")
    lines += ["## Ladder (travel p90 skill; descriptive, no pass/fail)", "",
              "| block | rung | " + " | ".join(report["species"]) + " |",
              "|---|---|" + "---:|" * len(report["species"])]
    for block, rungs in report["ladder"]["blocks"].items():
        for rung, values in rungs.items():
            lines.append(f"| {block} | {rung} | " + " | ".join(_s(values.get(s)) for s in report["species"]) + " |")
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: training records, registered files and training splits only
def dilution_table(records: list[dict], payload: dict) -> dict:
    """Median across seeds of each source's loss share, against its A8 draw share, for Zb, Bb, Zbc and Bbc."""
    out: dict = {}
    for arm in DILUTION_ARMS:
        label = BT.ARMS[arm]["pool"]
        weights = payload["probabilities"][label]
        per_seed = [E4.loss_shares(record) for record in records if record["arm"] == arm]
        per_seed = [shares for shares in per_seed if shares]
        if not per_seed:
            out[arm] = {"pool": label, "error": "no loss shares on disk"}
            continue
        block = {
            "pool": label,
            "n_seeds": len(per_seed),
            "rows": {
                name: {
                    "draw_share": float(weights[name]),
                    "loss_share": float(np.nanmedian([shares.get(name, np.nan) for shares in per_seed])),
                }
                for name in weights
            },
        }
        if label == "congener":
            congeners = sorted(set(weights) & BT.SPW.BIRDS_CONGENERS)
            block["congeners_combined"] = {
                "draw_share": float(sum(weights[name] for name in congeners)),
                "loss_share": float(np.nanmedian([sum(shares.get(n, 0.0) for n in congeners) for shares in per_seed])),
            }
        out[arm] = block
    return out


def section7_markdown(gap: dict, identity: str, exclusions: str, table: dict) -> str:
    neural_clean = sum(row["clean"] for arm, row in gap["by_arm"].items() if not BT.ARMS[arm].get("xgb"))
    xgb_clean = sum(row["clean"] for arm, row in gap["by_arm"].items() if BT.ARMS[arm].get("xgb"))
    lines = [
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']}; exit status {AS_TRAINED_FROM_PBS['exit status']}; "
        f"{AS_TRAINED_FROM_PBS['queue and precision']} |",
        f"| neural ({gap['n_neural']}) | {neural_clean} clean: {BT.REGISTERED_STEPS:,} of {BT.REGISTERED_STEPS:,} "
        "updates applied, none withheld, no non-finite steps |",
        f"| XGBoost ({gap['n_xgb']}) | {xgb_clean} fit once, boosters present |",
        f"| compute | {AS_TRAINED_FROM_PBS['compute']} |",
        f"| draw identity | {identity} |",
        f"| goose exclusions | {exclusions} |",
        f"| deferred or missing | {'none' if gap['complete'] else '; '.join(gap['problems'])} |",
        "",
        "Loss share against draw share (training records; median across seeds, columns need not sum to 100%).",
        "",
        "| source | draw, main | Zb | Bb | draw, congener | Zbc | Bbc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def pct(arm: str, name: str, key: str) -> str:
        row = table.get(arm, {}).get("rows", {}).get(name)
        return f"{row[key] * 100:.1f}%" if row else "—"

    names = sorted(
        set(table.get("Zbc", {}).get("rows", {})) | set(table.get("Zb", {}).get("rows", {})),
        key=lambda n: -(table.get("Zb", {}).get("rows", {}).get(n, {}).get("loss_share") or -1.0),
    )
    for name in names:
        lines.append(f"| {name} | {pct('Zb', name, 'draw_share')} | {pct('Zb', name, 'loss_share')} | "
                     f"{pct('Bb', name, 'loss_share')} | {pct('Zbc', name, 'draw_share')} | "
                     f"{pct('Zbc', name, 'loss_share')} | {pct('Bbc', name, 'loss_share')} |")
    combined = {arm: table.get(arm, {}).get("congeners_combined") for arm in ("Zbc", "Bbc")}
    if combined["Zbc"]:
        lines.append(f"| five congeners combined | — | — | — | {combined['Zbc']['draw_share'] * 100:.1f}% | "
                     f"{combined['Zbc']['loss_share'] * 100:.1f}% | "
                     f"{(combined['Bbc'] or {}).get('loss_share', float('nan')) * 100:.1f}% |")
    lines += ["", "Clarifications before the seal.", "", "| item | resolution |", "|---|---|"]
    lines += [f"| {item} | {text} |" for item, text in CLARIFICATIONS]
    return "\n".join(lines)


def run_check(check, *args) -> tuple[str, bool]:
    """Run one pre-seal check, returning its PASS/FAIL line rather than stopping the diagnostic."""
    try:
        result = check(*args)
    except SystemExit as failure:
        return f"FAIL — {str(failure).removeprefix('FAIL: ')}", False
    detail = ", ".join(f"{key} {value}" for key, value in result.items() if key != "status")
    return f"PASS ({detail})", True


def diagnostic() -> None:
    """Completeness, the two pre-seal checks, loss shares and the §7 block. Exits non-zero if any check fails."""
    payload = BT.load_registered_pool()
    records = read_records()
    rmap = record_map(records)
    gap = completeness(rmap)
    identity, identity_ok = run_check(assert_draw_identity, payload, rmap)
    exclusions, exclusions_ok = run_check(assert_goose_exclusions, payload, rmap)
    table = dilution_table(records, payload)

    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<7} {row['records']:>2} of {row['planned']:>2}, {row['clean']:>2} clean")
    for problem in gap["problems"]:
        print(f"  {problem}")
    print(f"draw identity    : {identity}")
    print(f"goose exclusions : {exclusions}")
    print("\n--- paste into preregistration_birds.md sec 7 ---\n")
    print(section7_markdown(gap, identity, exclusions, table))
    if not (gap["complete"] and identity_ok and exclusions_ok):
        raise SystemExit("FAIL: the diagnostic found a problem; §7 must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score every arm on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score arms that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true", help="pre-seal checks and the §7 block; opens nothing")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --predict, --report, --diagnostic, or a combination")

    if args.diagnostic:
        diagnostic()
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
        for species, entry in report["species"].items():
            readings = entry["readings"]
            print(f"{species:<22} package: {readings['3_package']['reading']}; "
                  f"congeners: {readings['4_congeners']['reading']}")


if __name__ == "__main__":
    main()
