# GenAI reference: I outlined this sealed evaluator for the repaired transformer across the bird rotation; only score the 60 new runs after the freeze 
# is locked, reuse/check all original sealed predictions, apply the rotation scoring and preregistered comparisons across birds etc; GenAI wrote it this
# and I checked it.
"""
Sealed evaluation of the repaired transformer on the rotating held-out birds
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO / "experiments" / "diagnostics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_repaired_diagnostic as RD  # noqa: E402
import bird_rotation_repaired_train as D  # noqa: E402
import bird_rotation_sealed_eval as RS  # noqa: E402
import tier4_repaired_sealed_eval as R4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402

S2, E3, E4, BE, BT, RT, T, env = RS.S2, RS.E3, RS.E4, RS.BE, RS.BT, RS.RT, D.T, D.env

FREEZE = REPO / "preregistration" / D.FREEZE
SEALED = REPO / "results" / "birds_rotation_repaired_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_rotation_repaired_results.json"
REPORT_MD = SEALED / "birds_rotation_repaired_results.md"
FIGURE_JSON = SEALED / "birds_rotation_repaired_figure_5_3.json"
MANIFEST = SEALED / "birds_rotation_repaired_sealed_manifest.json"

FLOOR, NOISE_SCALE = RS.FLOOR, RS.NOISE_SCALE
FOLD0, ROTATION, BIRDS, FOLD_OF = RS.FOLD0, RS.ROTATION, RS.BIRDS, RS.FOLD_OF
PROBE_FOLD = 6  # the repair probe's fold (fold 6, seed 0, at 1 h): the repair was chosen there
BIRD_SETS = {
    "all": BIRDS,
    "not_fold0": ROTATION,
    "not_fold6": tuple(b for b in BIRDS if FOLD_OF[b] != PROBE_FOLD),
    "not_fold0_or_fold6": tuple(b for b in ROTATION if FOLD_OF[b] != PROBE_FOLD),
}
if [len(birds) for birds in BIRD_SETS.values()] != [30, 27, 27, 24]:
    raise RuntimeError("the bird sets no longer hold 30, 27, 27 and 24 birds")

NEW = tuple(D.ARMS)  # BbF6, ZbF6: predicted here
ORIGINALS = RS.ARMS[1:]  # Bb, LSTMBb, XGBBb, Zb, LSTMS, XGBS: read from their sealed files
LABELS = ("P", *NEW, "Bb", "Zb", "LSTMBb", "XGBBb", "LSTMS", "XGBS")
if set(LABELS) != set(RS.ARMS) | set(NEW) or set(D.ARMS.values()) - set(ORIGINALS):
    raise RuntimeError("the evaluator's arms no longer match the rotation's and the driver's")

# Freeze sec 4, in its order: (comparison number, left, right); right "P" is skill against P.
SEC4: tuple[tuple[int, str, str], ...] = (
    (1, "BbF6", "P"), (1, "ZbF6", "P"),
    (2, "BbF6", "Bb"), (2, "ZbF6", "Zb"),
    (3, "ZbF6", "BbF6"),
    (4, "BbF6", "LSTMBb"), (4, "BbF6", "XGBBb"), (4, "ZbF6", "LSTMBb"), (4, "ZbF6", "XGBBb"),
    (5, "LSTMBb", "LSTMS"), (5, "XGBBb", "XGBS"),
)
FAMILIES = RS.FAMILIES  # comparison 5, which the rotation report also holds

# Freeze sec 6, in its order, read on all 30 birds. "Most" is more than MOST of the 30.
MOST = 15
EXPECTATIONS: tuple[dict, ...] = (
    {"rule": "most_above", "comparisons": ("BbF6 - Bb", "ZbF6 - Zb")},  # above +0.05 on more than 15
    {"rule": "median_below", "comparisons": ("BbF6 - LSTMBb", "BbF6 - XGBBb")},  # median below -0.05
    {"rule": "most_negative", "comparisons": ("ZbF6 - BbF6",)},  # below 0 on more than 15
)


def name(left: str, right: str) -> str:
    return f"{left} - {right}"


COMPARISON_NAMES = tuple(name(left, right) for _, left, right in SEC4)
if {c for e in EXPECTATIONS for c in e["comparisons"]} - set(COMPARISON_NAMES):
    raise RuntimeError("sec 6 reads a comparison sec 4 does not make")


# Seal
def freeze_committed() -> dict:
    """The freeze is tracked and unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    previous = S2.FREEZE
    S2.FREEZE = FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def committed_freeze() -> str:
    shown = E4.git("show", f"HEAD:{FREEZE.relative_to(REPO).as_posix()}")
    if shown.returncode != 0:
        raise SystemExit(f"FAIL: cannot read {FREEZE.name} from HEAD, so it is not committed: {shown.stderr.strip()}")
    return shown.stdout


def section_body(text: str, number: int) -> str:
    try:
        return RS.section_body(text, number)
    except SystemExit:
        raise SystemExit(f"FAIL: {FREEZE.name} has no section {number}") from None


def check_filled(number: int, text: str) -> None:
    """Refuse sec 7 under any heading but "As trained", and an empty section or one still carrying placeholder words."""
    if number == 7 and not re.search(r"^##\s+7\.\s+As trained\b", text, re.MULTILINE):
        raise SystemExit(f"FAIL: {FREEZE.name} sec 7 is not headed 'As trained'")
    body = section_body(text, number)
    if not body.strip():
        raise SystemExit(f"FAIL: {FREEZE.name} sec {number} is empty in the committed version.")
    found = RS.PLACEHOLDER.search(body)
    if found:
        raise SystemExit(f"FAIL: {FREEZE.name} sec {number} still says {found.group(0)!r} in the committed version.")


def assert_sealed() -> dict:
    """Committed, clean, and secs 6 and 7 filled in the committed version; an earlier manifest at the same blob. Run by
    --predict before any test partition is opened."""
    attestation = freeze_committed()
    text = committed_freeze()
    for number in (6, 7):
        check_filled(number, text)
    attestation["sections_filled"] = [6, 7]
    if MANIFEST.exists():
        earlier = json.loads(MANIFEST.read_text()).get("attestation") or {}
        if earlier.get("freeze_blob") and earlier["freeze_blob"] != attestation["freeze_blob"]:
            raise SystemExit(
                f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but {FREEZE.name} is now "
                f"blob {attestation['freeze_blob'][:12]}. Rebuild the report with --report instead."
            )
    return attestation


def checked_runs() -> tuple[dict, int, dict]:
    """(completeness, grid index, payloads), refusing unless all 60 runs are clean and every original is as registered."""
    payloads = D.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    gap = RD.completeness(RD.read_records(), grid_index, payloads)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"][:20]))
    originals = RD.original_check(grid_index, payloads)
    if originals:
        raise SystemExit("FAIL: originals differ from the registered\n  " + "\n  ".join(originals[:20]))
    return gap, grid_index, payloads


# Inputs from the rotation's seal
def rotation_inputs() -> dict:
    """The rotation's sealed manifest (scoring rules) and report (tie-back target), attested at the same freeze commit,
    with drops as its freeze sec 4 counts and as its report recorded them."""
    for path in (RS.MANIFEST, RS.REPORT_JSON):
        if not path.exists():
            raise SystemExit(f"FAIL: {path} does not exist; the scoring rules and the tie-back are read from it")
    manifest, report = json.loads(RS.MANIFEST.read_text()), json.loads(RS.REPORT_JSON.read_text())
    commit = (report.get("attestation") or {}).get("freeze_commit")
    if not commit or commit != (manifest.get("attestation") or {}).get("freeze_commit"):
        raise SystemExit(f"FAIL: {RS.REPORT_JSON.name} and {RS.MANIFEST.name} are not attested at one freeze commit")
    problems = drop_problems(manifest.get("scoring_rules") or {}, report)
    if problems:
        raise SystemExit("FAIL: the rotation's scoring rules\n  " + "\n  ".join(problems))
    return {"rules": manifest["scoring_rules"], "report": report, "attestation": report["attestation"],
            "report_sha256": R4.sha256(RS.REPORT_JSON), "manifest_sha256": R4.sha256(RS.MANIFEST)}


def drop_problems(rules: dict, report: dict) -> list[str]:
    problems = []
    for bird in BIRDS:
        entry = rules.get(bird)
        if entry is None:
            problems.append(f"{bird}: no scoring rules")
            continue
        expected = RS.SCORED_IN_FREEZE.get(bird, (10, 10))
        if (entry["test_birds"], entry["scored"]) != expected:
            problems.append(f"{bird}: {entry['scored']} of {entry['test_birds']} scored, the rotation freeze says "
                            f"{expected[1]} of {expected[0]}")
        if FOLD_OF[bird] == 0 and entry["dropped"]:
            problems.append(f"{bird}: fold 0 drops {entry['dropped']}; the rotation freeze says none")
        sealed = ((report.get("species") or {}).get(bird) or {}).get("dropped")
        if sorted(entry["dropped"]) != sorted(sealed or []):
            problems.append(f"{bird}: the manifest drops {entry['dropped']}, the report {sealed}")
    return problems


# Files
def prediction_path(bird: str, tag: str) -> Path:
    return PREDICTIONS / f"{bird}__{tag}.parquet"


def new_runs(bird: str) -> list:
    return [run for run in D.PLAN if bird in run.scored_on]


def read_runs(bird: str) -> dict[str, dict]:
    """
    Tag -> {path, arm, seed} for P and every original on this bird, from the plans that trained them. Fold 0: the bird
    study's sealed files at five seeds, the gull's P from Tier 2 and its specialists from Tier 3 (tagged `tier3:` as the
    rotation report tags them). Folds 1-9: the rotation's own sealed files.
    """
    fold = FOLD_OF[bird]
    if fold:
        out = {"P": {"path": RS.prediction_path(bird, "P"), "arm": "P", "seed": None}}
        for run in RT.PLAN:
            if run.arm in ORIGINALS and run.fold == fold and run.species in ("", bird):
                out[run.run_id] = {"path": RS.prediction_path(bird, run.run_id), "arm": run.arm, "seed": run.seed}
        return out
    home = S2.PREDICTIONS if bird == BE.GULL else BE.PREDICTIONS
    out = {"P": {"path": home / f"{bird}__P.parquet", "arm": "P", "seed": None}}
    for run in BT.PLAN:
        if run.arm in ORIGINALS and run.k_percent is None and run.species in ("", bird):
            out[run.run_id] = {"path": BE.PREDICTIONS / f"{bird}__{run.run_id}.parquet", "arm": run.arm, "seed": run.seed}
    if bird == BE.GULL:
        for run in BE.T3.PLAN:
            if run.arm in ("LSTMS", "XGBS") and run.k_percent is None and run.species == bird:
                out[f"tier3:{run.run_id}"] = {"path": E3.PREDICTIONS / f"{bird}__{run.run_id}.parquet",
                                              "arm": run.arm, "seed": run.seed}
    return out


def missing_reused() -> list[str]:
    return [str(entry["path"]) for bird in BIRDS for entry in read_runs(bird).values() if not entry["path"].exists()]


def tag_problems(report: dict) -> list[str]:
    """Each original's tags here against the rotation report's for that bird and arm, and its seeds against the plans."""
    problems = []
    for bird in BIRDS:
        sealed = ((report.get("species") or {}).get(bird) or {}).get("arms") or {}
        reads = read_runs(bird)
        for arm in ORIGINALS:
            here = {tag for tag, entry in reads.items() if entry["arm"] == arm}
            there = set(((sealed.get(arm) or {}).get("tags") or {}).values())
            if here != there:
                problems.append(f"{bird} {arm}: tags {sorted(here)} here, {sorted(there)} in the rotation report")
            seeds = {entry["seed"] for entry in reads.values() if entry["arm"] == arm}
            if seeds != RS.planned_seeds(bird, arm):
                problems.append(f"{bird} {arm}: seeds {sorted(seeds)}, planned {sorted(RS.planned_seeds(bird, arm))}")
    return problems


# Prediction
def test_frame(bird: str) -> pd.DataFrame:
    """One held-out bird's test partition, through the evaluator that sealed it. Called after assert_sealed."""
    return BE.test_frame(bird) if FOLD_OF[bird] == 0 else RS.test_frame(bird)


def predict_bird(bird: str, device, overwrite: bool, grid_index: int, payloads: dict) -> list[str]:
    """Score this bird's six new runs on its test partition, one cached file per run; the partition is read only if a
    file is missing."""
    runs = new_runs(bird)
    todo = [run for run in runs if overwrite or not prediction_path(bird, run.run_id).exists()]
    for run in runs:
        if run not in todo:
            print(f"  {prediction_path(bird, run.run_id).name}  (exists, skipped)")
    if todo:
        frame = test_frame(bird)
        print(f"\n{bird}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
    for run in todo:
        out = prediction_path(bird, run.run_id)
        started = time.time()
        entry = next(e for e in RD.entries(run.fold) if e["run_id"] == run.run_id)
        model, record = RD.load_model(entry, grid_index, payloads)
        if bird in record["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
        predictor = MovementFMPredictor(model, device=device, sigma=SCALE_FLOOR_M)
        results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
        R4.write_atomic(results, out)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
    return [prediction_path(bird, run.run_id).name for run in runs]


def run_predictions(device, overwrite: bool = False, tf32: dict | None = None) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    gap, grid_index, payloads = checked_runs()
    rotation = rotation_inputs()
    problems = tag_problems(rotation["report"])
    missing = missing_reused()
    if problems or missing:
        raise SystemExit("FAIL: the read arms\n  " + "\n  ".join((problems + missing)[:20]))
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = {"attestation": attestation, "tf32": tf32 if tf32 is not None else env.tf32_flags(),
                "completeness": gap,
                "rotation": {key: rotation[key] for key in ("attestation", "report_sha256", "manifest_sha256")},
                "species": {}, "new": {}, "reused": {}}
    for bird in BIRDS:
        manifest["species"][bird] = predict_bird(bird, device, overwrite, grid_index, payloads)
        manifest["new"][bird] = {run.run_id: R4.sha256(prediction_path(bird, run.run_id)) for run in new_runs(bird)}
        manifest["reused"][bird] = {tag: R4.sha256(entry["path"]) for tag, entry in read_runs(bird).items()}
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def lookup(tag: str, parsed: dict) -> dict | None:
    return parsed.get(tag)


def load_frames(bird: str, manifest: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """
    (P, frames by tag, tag -> arm and seed) for one bird: every file checked against the sha256 recorded at --predict,
    nothing in this study's directory but its own runs, and every frame aligned to P.
    """
    reused = (manifest.get("reused") or {}).get(bird) or {}
    new = (manifest.get("new") or {}).get(bird) or {}
    frames, parsed = {}, {}
    for tag, entry in read_runs(bird).items():
        if not entry["path"].exists() or reused.get(tag) != R4.sha256(entry["path"]):
            raise SystemExit(f"FAIL: {entry['path']} is missing or differs from the sha256 recorded at --predict")
        frames[tag] = E3.load_frame(entry["path"])
        parsed[tag] = {"arm": entry["arm"], "seed": entry["seed"], "k": None, "draw": None}
    own = {run.run_id: run for run in new_runs(bird)}
    present = {path.stem.split("__", 1)[1] for path in PREDICTIONS.glob(f"{bird}__*.parquet")}
    if present - set(own):
        raise SystemExit(f"FAIL: {bird} holds predictions this study does not score: {sorted(present - set(own))}")
    if set(own) - present:
        raise SystemExit(f"FAIL: {bird} has no predictions for {sorted(set(own) - present)}; run --predict")
    for run_id, run in own.items():
        path = prediction_path(bird, run_id)
        if new.get(run_id) != R4.sha256(path):
            raise SystemExit(f"FAIL: {path} differs from the sha256 recorded at --predict")
        frames[run_id] = E3.load_frame(path)
        parsed[run_id] = {"arm": run.arm, "seed": run.seed, "k": None, "draw": None}
    persistence = frames["P"]
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{bird} {tag}")
    return persistence, frames, parsed


def expected_seeds(bird: str, published: bool = False) -> dict[str, set[int]]:
    """
    Seeds per arm. Matched (every fold): the new arms at 0-2 and each original at the rotation's own seed set (neural
    0-2, XGBoost 0). Published (fold 0 only): the originals at the seeds they were sealed at (five).
    """
    out = {arm: set(D.SEEDS) for arm in NEW}
    return out | {arm: RS.planned_seeds(bird, arm) if published else RS.rotation_seeds(arm) for arm in ORIGINALS}


def summarise(persistence: pd.DataFrame, frames: dict, parsed: dict, boundary: float,
              seeds: dict[str, set[int]], label: str) -> tuple[dict, dict]:
    """(the frames at `seeds`, arm blocks), refused unless every arm is present at exactly those seeds."""
    chosen = {tag: frame for tag, frame in frames.items()
              if parsed[tag]["seed"] is None or parsed[tag]["seed"] in seeds.get(parsed[tag]["arm"], set())}
    E4.assert_no_collisions(chosen, parsed, parse=lookup)
    summaries = E3.summarise_all(chosen, persistence, boundary, parsed, parse=lookup)
    problems = [arm for arm in LABELS if arm not in summaries]
    for arm in LABELS[1:]:
        found = {key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0}
        if arm in summaries and found != seeds[arm]:
            problems.append(f"{arm} has seeds {sorted(found)}, expected {sorted(seeds[arm])}")
    if problems:
        raise SystemExit(f"FAIL: {label}: " + "; ".join(problems))
    return chosen, RS.arm_blocks(summaries)


def compare_all(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    """Every arm against P and the sec 4 comparisons, with skill and metre intervals from one set of resamples."""
    resampler = RS.Resampler(persistence, boundary)
    return {
        "against_p": {arm: RS.compare(arm, "P", arms, frames, persistence, boundary, resampler) for arm in LABELS[1:]},
        "comparisons": {name(left, right): RS.compare(left, right, arms, frames, persistence, boundary, resampler)
                        for _, left, right in SEC4 if right != "P"},
    }


def reading(block: dict, comparison: str) -> dict:
    left, right = comparison.split(" - ")
    return (block.get("against_p") or {}).get(left, {}) if right == "P" else (block.get("comparisons") or {}).get(
        comparison, {})


# Tie-back
def _missing(value) -> bool:
    return value is None or not math.isfinite(value)


def _same(a, b) -> bool:
    """Equal to 1e-12, or missing on both sides (an empty stratum is NaN in both reports)."""
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def same_headline(mine: dict, theirs: dict) -> list[str]:
    """Every stored statistic of one arm, per stratum, equal to the sealed report's (the baselines evaluator's check,
    with a statistic missing on both sides counted as equal)."""
    return [f"{stratum} {stat}: {(mine.get(stratum) or {}).get(stat)} here, {(theirs.get(stratum) or {}).get(stat)} sealed"
            for stratum in (*E3.STRATA, "all") for stat in E3.STATISTICS
            if not _same((mine.get(stratum) or {}).get(stat), (theirs.get(stratum) or {}).get(stat))]


def _gap(arms: dict, pooled: str, specialist: str) -> float:
    return E3.travel_skill(arms, pooled) - E3.travel_skill(arms, specialist)


def tie_back(bird: str, primary: dict, matched: dict | None, sealed: dict, counts: dict) -> dict:
    """
    P and every original here against the rotation report: every headline statistic in every stratum, comparison 5's
    gaps and the scored counts (folds 1-9 at their seeds, fold 0 at five), and fold 0 at seeds 0-2 against the report's
    seeds 0-2 skills and gaps. Returns the problems and the largest travel p90 skill difference.
    """
    problems, diffs = [], []
    for arm in RS.ARMS:
        problems += [f"{arm} {p}" for p in same_headline(primary[arm]["headline"],
                                                         (sealed["arms"].get(arm) or {}).get("headline") or {})]
        diffs.append(E3.travel_skill(primary, arm) - E3.travel_skill(sealed["arms"], arm))
    for family, (pooled, specialist) in FAMILIES.items():
        here, there = _gap(primary, pooled, specialist), sealed["readings"]["gaps"][family].get("skill_difference")
        if not _same(here, there):
            problems.append(f"{family} gap {here} here, {there} sealed")
    for key, value in counts.items():
        if sealed.get(key) != value:
            problems.append(f"{key} {value} here, {sealed.get(key)} sealed")
    if matched is not None:
        early = sealed["seeds_0_2"]
        for arm in ORIGINALS:
            here = E3.travel_skill(matched, arm)
            diffs.append(here - early["skill"][arm])
            if not _same(here, early["skill"][arm]):
                problems.append(f"{arm} seeds 0-2 skill {here} here, {early['skill'][arm]} sealed")
        for family, (pooled, specialist) in FAMILIES.items():
            here = _gap(matched, pooled, specialist)
            if not _same(here, early["gaps"][family]):
                problems.append(f"{family} seeds 0-2 gap {here} here, {early['gaps'][family]} sealed")
    finite = [abs(d) for d in diffs if math.isfinite(d)]
    return {"problems": [f"{bird}: {p}" for p in problems], "max_abs_skill_diff": max(finite) if finite else None,
            "n_arms": len(RS.ARMS) + (len(ORIGINALS) if matched is not None else 0)}


# One bird
def bird_inputs(bird: str, manifest: dict, rules: dict, boundary: float) -> dict:
    """The bird's frames after the rotation's drops, and its arm blocks at the matched seeds (and fold 0's published)."""
    persistence, frames, parsed = load_frames(bird, manifest)
    dropped = rules[bird]["dropped"]
    persistence, frames = RS.drop_scored(persistence, frames, dropped)
    out = {"persistence": persistence, "parsed": parsed, "dropped": dropped,
           "counts": {"n_test_individuals_scored": int(persistence["individual_id"].nunique()),
                      "n_test_windows": int(len(persistence))}}
    out["matched"] = summarise(persistence, frames, parsed, boundary, expected_seeds(bird), f"{bird} matched")
    if FOLD_OF[bird] == 0:
        out["published"] = summarise(persistence, frames, parsed, boundary, expected_seeds(bird, True),
                                     f"{bird} published")
    return out


def check_bird(bird: str, inputs: dict, sealed: dict) -> dict:
    if FOLD_OF[bird] == 0:
        return tie_back(bird, inputs["published"][1], inputs["matched"][1], sealed, inputs["counts"])
    return tie_back(bird, inputs["matched"][1], None, sealed, inputs["counts"])


def provenance(bird: str, parsed: dict, manifest: dict) -> dict:
    """Tag -> arm, seed, file and sha256 for every frame behind this bird's rows."""
    shas = ((manifest.get("reused") or {}).get(bird) or {}) | ((manifest.get("new") or {}).get(bird) or {})
    reads = read_runs(bird)
    out = {}
    for tag, entry in parsed.items():
        path = reads[tag]["path"] if tag in reads else prediction_path(bird, tag)
        out[tag] = {"arm": entry["arm"], "seed": entry["seed"], "file": str(path.relative_to(REPO)),
                    "sha256": shas.get(tag)}
    return out


def bird_entry(bird: str, inputs: dict, boundary: float, payloads: dict, manifest: dict) -> dict:
    fold = FOLD_OF[bird]
    persistence = inputs["persistence"]
    entry = {
        "fold": fold,
        "carried": fold == 0,
        "regime_boundary_m": boundary,
        **inputs["counts"],
        "dropped": inputs["dropped"],
        "n_source_species": len(RD.pool_names(fold, payloads)),
        "provenance": provenance(bird, inputs["parsed"], manifest),
    }
    for seed_set in ("matched", "published"):
        if seed_set in inputs:
            frames, arms = inputs[seed_set]
            entry[seed_set] = {"seeds": {arm: sorted(s) for arm, s in expected_seeds(bird, seed_set == "published").items()},
                               "arms": arms} | compare_all(arms, frames, persistence, boundary)
    return entry


# Across birds and sec 6
def across_birds(report: dict) -> dict:
    """Per comparison and bird set: median, counts above +0.05 / within / below -0.05, and below 0. Matched seeds only:
    fold 0's published values never enter."""
    out = {}
    for comparison in COMPARISON_NAMES:
        out[comparison] = {}
        for set_name, birds in BIRD_SETS.items():
            values = [reading(report["species"][bird]["matched"], comparison).get("skill_difference") for bird in birds]
            finite = [v for v in values if v is not None and np.isfinite(v)]
            counts = Counter(RS.band(v) for v in values)
            out[comparison][set_name] = {
                "n_birds": len(birds), "n_scored": len(finite),
                "median": float(np.median(finite)) if finite else None,
                **{key: counts.get(key, 0) for key in ("above", "within", "below", "not scored")},
                "negative": sum(v < 0 for v in finite),
            }
    return out


def sec6_bullets(path: Path = FREEZE) -> list[str]:
    """The freeze's sec 6 expectations, quoted rather than restated; exactly one per reading."""
    bullets = [line[2:].strip() for line in section_body(path.read_text(), 6).splitlines() if line.startswith("- ")]
    if len(bullets) != len(EXPECTATIONS):
        raise SystemExit(f"FAIL: {path.name} sec 6 has {len(bullets)} expectations; the evaluator reads "
                         f"{len(EXPECTATIONS)}")
    return bullets


def expectation_value(rule: str, row: dict):
    return {"most_above": row["above"], "median_below": row["median"], "most_negative": row["negative"]}[rule]


def expectation_met(rule: str, row: dict) -> bool | None:
    """Read on all 30: None unless every bird is scored."""
    if row["n_scored"] != row["n_birds"]:
        return None
    if rule == "median_below":
        return bool(row["median"] < -NOISE_SCALE)
    return bool(expectation_value(rule, row) > MOST)


def expectations(report: dict, bullets: list[str] | None = None) -> list[dict]:
    out = []
    for text, expectation in zip(bullets or sec6_bullets(), EXPECTATIONS):
        rows = []
        for comparison in expectation["comparisons"]:
            sets = report["across_birds"][comparison]
            rows.append({"comparison": comparison, "value": expectation_value(expectation["rule"], sets["all"]),
                         "met": expectation_met(expectation["rule"], sets["all"]),
                         "elsewhere": {k: expectation_value(expectation["rule"], v) for k, v in sets.items()
                                       if k != "all"}})
        out.append({"expectation": text, "rule": expectation["rule"], "most": MOST, "noise_scale": NOISE_SCALE,
                    "rows": rows, "verdict": R4.verdict([row["met"] for row in rows])})
    return out


# Figure 5.3
def _ci(block: dict | None) -> tuple:
    return (block or {}).get("lo"), (block or {}).get("hi")


def _sources(entry: dict, block: dict, arm: str) -> list[dict]:
    tags = set((block["arms"].get(arm) or {}).get("tags", {}).values())
    return [{"tag": tag} | info for tag, info in entry["provenance"].items() if tag in tags]


def figure_rows(report: dict) -> dict:
    """One row per bird, seed set and arm (skill against P) or sec 4 contrast, with its interval, fits and files."""
    rows = []
    for bird, entry in report["species"].items():
        for seed_set in ("matched", "published"):
            block = entry.get(seed_set)
            if block is None:
                continue
            base = {"species": bird, "fold": entry["fold"], "carried": entry["carried"], "seed_set": seed_set,
                    "n_source_species": entry["n_source_species"],
                    "n_test_individuals": entry["n_test_individuals_scored"], "n_test_windows": entry["n_test_windows"]}
            for arm in LABELS[1:]:
                comparison = block["against_p"][arm]
                boot = comparison.get("skill_bootstrap") or {}
                lo, hi = _ci((boot.get("skill") or {}).get(arm))
                m_lo, m_hi = _ci(boot.get("difference_m"))
                rows.append(base | {"comparison": "skill_vs_p", "sec4": 1 if arm in NEW else None,
                                    "arm_or_family": arm, "left": arm, "right": "P",
                                    "estimate": comparison.get("skill_difference"), "ci95_lo": lo, "ci95_hi": hi,
                                    "metre_ci95_lo": m_lo, "metre_ci95_hi": m_hi,
                                    "left_n_fits": block["arms"][arm]["n_seeds"], "right_n_fits": None,
                                    "provenance": {arm: _sources(entry, block, arm)}})
            for number, left, right in SEC4:
                if right == "P":
                    continue
                comparison = block["comparisons"][name(left, right)]
                boot = comparison.get("skill_bootstrap") or {}
                lo, hi = _ci(boot.get("skill_difference"))
                m_lo, m_hi = _ci(boot.get("difference_m"))
                family = next((f for f, pair in FAMILIES.items() if pair == (left, right)), None)
                rows.append(base | {"comparison": "specialist_gap" if family else "contrast", "sec4": number,
                                    "arm_or_family": family or name(left, right), "left": left, "right": right,
                                    "estimate": comparison.get("skill_difference"), "ci95_lo": lo, "ci95_hi": hi,
                                    "metre_ci95_lo": m_lo, "metre_ci95_hi": m_hi,
                                    "left_n_fits": block["arms"][left]["n_seeds"],
                                    "right_n_fits": block["arms"][right]["n_seeds"],
                                    "provenance": {left: _sources(entry, block, left),
                                                   right: _sources(entry, block, right)}})
    return {
        "figure": "5.3", "study": report["study"], "exploratory": True, "registered_by": report["registered_by"],
        "attestation": report["attestation"], "rotation_attestation": report["rotation"]["attestation"],
        "statistic": "travel p90 skill against P", "evaluation_floor": FLOOR, "noise_scale": NOISE_SCALE,
        "n_bootstrap": S2.N_BOOTSTRAP, "bootstrap_seed": S2.BOOTSTRAP_SEED,
        "seed_sets": {"matched": "every fold; new arms at seeds 0-2, originals at the rotation's seeds (neural 0-2, "
                                 "XGBoost 0); the across-bird readings use these alone",
                      "published": "fold 0 only; originals at the five seeds they were sealed at; never pooled"},
        "rows": rows,
    }


# Report
def build_report(tf32: dict | None = None) -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    rotation = rotation_inputs()
    recorded = manifest.get("rotation") or {}
    if (recorded.get("report_sha256"), recorded.get("manifest_sha256")) != (rotation["report_sha256"],
                                                                            rotation["manifest_sha256"]):
        raise SystemExit("FAIL: the rotation's sealed report or manifest changed after --predict")
    gap, _, payloads = checked_runs()
    boundaries = RT.check_boundaries() | BE.check_boundaries()
    sealed = rotation["report"]["species"]

    # Pass 1, cheap: every bird's read arms tied back before any bootstrap is spent.
    ties, problems = {}, []
    for bird in BIRDS:
        ties[bird] = check_bird(bird, bird_inputs(bird, manifest, rotation["rules"], boundaries[bird]), sealed[bird])
        problems += ties[bird]["problems"]
    if problems:
        raise SystemExit("FAIL: read arms differ from the rotation's sealed report\n  " + "\n  ".join(problems[:30]))
    print(f"tie-back         : {sum(t['n_arms'] for t in ties.values())} arm readings on {len(BIRDS)} birds equal the "
          f"rotation report; max |skill diff| {max(t['max_abs_skill_diff'] for t in ties.values()):.1e}", flush=True)

    report: dict = {
        "study": "birds_rotation_repaired",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "rotation": {"report": str(RS.REPORT_JSON.relative_to(REPO)), **recorded},
        "switches": D.SWITCHES,
        "evaluation_floor": FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "n_bootstrap": S2.N_BOOTSTRAP,
        "bootstrap_seed": S2.BOOTSTRAP_SEED,
        "most": MOST,
        "tf32": {"predict": manifest.get("tf32"), "report": tf32 if tf32 is not None else env.tf32_flags()},
        "completeness": gap,
        "bird_sets": {key: list(birds) for key, birds in BIRD_SETS.items()},
        "species": {},
    }
    for bird in BIRDS:  # pass 2: the bootstrap
        started = time.time()
        inputs = bird_inputs(bird, manifest, rotation["rules"], boundaries[bird])
        report["species"][bird] = bird_entry(bird, inputs, boundaries[bird], payloads, manifest) | {
            "tie_back": ties[bird]}
        print(f"{bird}: fold {FOLD_OF[bird]} in {time.time() - started:.0f}s", flush=True)
    report["across_birds"] = across_birds(report)
    report["expectations"] = expectations(report)
    report["tie_back"] = {"against": str(RS.REPORT_JSON.relative_to(REPO)), "problems": 0,
                          "max_abs_skill_diff": max(t["max_abs_skill_diff"] for t in ties.values())}
    return report


def _cell(reading_: dict, key: str = "skill_difference") -> str:
    return f"{RS._s(reading_.get('skill_difference'))} {RS._interval((reading_.get('skill_bootstrap') or {}).get(key))}"


def _skill_cell(block: dict, arm: str) -> str:
    comparison = block["against_p"][arm]
    return f"{RS._s(comparison.get('skill_difference'))} " + RS._interval(
        ((comparison.get("skill_bootstrap") or {}).get("skill") or {}).get(arm))


SET_LABELS = {"all": "30", "not_fold0": "27, not fold 0", "not_fold6": "27, not fold 6",
              "not_fold0_or_fold6": "24, neither"}


def write_markdown(report: dict, path: Path) -> None:
    lines = ["**Exploratory — not a tiered result.**", "",
             "# Repaired transformer on the rotating held-out birds: sealed evaluation", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"BbF6 and ZbF6 are Bb and Zb with {report['switches']}. Runs scored: {gap['n_clean']} of "
              f"{gap['n_planned']} clean. Scored as the rotation: travel p90 skill against P at F = "
              f"{report['evaluation_floor']}, its drops and boundaries, median of seeds; paired bootstrap of "
              f"{report['n_bootstrap']:,} whole-individual resamples at seed {report['bootstrap_seed']}, P resampled "
              f"with them, 95% intervals in skill and metres. ±{report['noise_scale']} counts as no difference. "
              "Matched seeds: the new arms at 0–2 and every original at the rotation's seeds (neural 0–2, XGBoost 0), "
              "fold 0 included; fold 0's published five-seed values are in their own section and enter nothing else. "
              f"\"Most\" is more than {report['most']} of 30. TF32 at prediction: {report['tf32']['predict']}.", ""]

    lines += ["## Expectations (freeze sec 6, read on all 30 birds)", ""]
    for number, block in enumerate(report["expectations"], 1):
        lines += [f"### 6.{number} {block['verdict']}", "", f"> {block['expectation']}", "",
                  "| comparison | value on 30 | met | 27, not fold 0 | 27, not fold 6 | 24, neither |",
                  "|---|---:|---|---:|---:|---:|"]
        for row in block["rows"]:
            def show(v):
                return RS._s(v) if isinstance(v, float) else str(v)
            flag = {True: "yes", False: "no", None: "not scored"}[row["met"]]
            lines.append(f"| {row['comparison']} | {show(row['value'])} | {flag} | "
                         + " | ".join(show(row["elsewhere"][k]) for k in ("not_fold0", "not_fold6", "not_fold0_or_fold6"))
                         + " |")
        lines.append("")

    lines += ["## Across birds (matched seeds)", "",
              "| comparison | birds | scored | median | above +0.05 | within | below −0.05 | below 0 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for comparison, sets in report["across_birds"].items():
        for key, row in sets.items():
            lines.append(f"| {comparison} | {SET_LABELS[key]} | {row['n_scored']} | {RS._s(row['median'])} | "
                         f"{row['above']} | {row['within']} | {row['below']} | {row['negative']} |")

    lines += ["", "## Per bird: skill against P (matched seeds; 95% interval)", "",
              "| fold | bird | scored | " + " | ".join(LABELS[1:]) + " |", "|---:|---|---:|" + "---|" * (len(LABELS) - 1)]
    for bird, entry in report["species"].items():
        lines.append(f"| {entry['fold']} | {RS.short(bird)} | {entry['n_test_individuals_scored']} | "
                     + " | ".join(_skill_cell(entry["matched"], arm) for arm in LABELS[1:]) + " |")

    contrasts = [name(left, right) for _, left, right in SEC4 if right != "P"]
    lines += ["", "## Per bird: sec 4 comparisons 2–5 (matched seeds; skill difference, skill interval; metre interval)",
              "", "| fold | bird | " + " | ".join(contrasts) + " |", "|---:|---|" + "---|" * len(contrasts)]
    for bird, entry in report["species"].items():
        cells = [_cell(entry["matched"]["comparisons"][c]) + " " + RS._interval(
            (entry["matched"]["comparisons"][c].get("skill_bootstrap") or {}).get("difference_m"), "m")
                 for c in contrasts]
        lines.append(f"| {entry['fold']} | {RS.short(bird)} | " + " | ".join(cells) + " |")

    lines += ["", "## Fold 0 with the originals' published five seeds (apart; enters nothing above)", "",
              "| bird | " + " | ".join(LABELS[1:]) + " |", "|---|" + "---|" * (len(LABELS) - 1)]
    for bird in FOLD0:
        block = report["species"][bird]["published"]
        lines.append(f"| {RS.short(bird)} | " + " | ".join(_skill_cell(block, arm) for arm in LABELS[1:]) + " |")
    lines += ["", "| bird | " + " | ".join(contrasts) + " |", "|---|" + "---|" * len(contrasts)]
    for bird in FOLD0:
        block = report["species"][bird]["published"]
        lines.append(f"| {RS.short(bird)} | " + " | ".join(_cell(block["comparisons"][c]) for c in contrasts) + " |")

    tie = report["tie_back"]
    lines += ["", "## Read arms tied back", "",
              f"P and every original equal `{tie['against']}` in every headline statistic and stratum, with comparison "
              "5's gaps and the scored counts (fold 0 at five seeds, and at seeds 0–2 against its `seeds_0_2`); max "
              f"|travel p90 skill difference| {tie['max_abs_skill_diff']:.1e}. Every read file's sha256 is in the "
              "manifest and matched at report time."]
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: records, the originals and the files; no test partition or prediction is read
def diagnostic() -> None:
    try:
        freeze_committed()
        print("freeze           : committed and clean")
    except SystemExit as reason:
        print(f"freeze           : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    shown = E4.git("show", f"HEAD:{FREEZE.relative_to(REPO).as_posix()}")
    for number in (6, 7):
        try:
            if shown.returncode != 0:
                raise SystemExit("FAIL: not in HEAD")
            check_filled(number, shown.stdout)
            print(f"sec {number}            : filled in the committed version")
        except SystemExit as reason:
            print(f"sec {number}            : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    payloads = D.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    gap = RD.completeness(RD.read_records(), grid_index, payloads)
    originals = RD.original_check(grid_index, payloads)
    missing = missing_reused()
    n_read = sum(len(read_runs(bird)) for bird in BIRDS)
    n_new = sum(len(new_runs(bird)) for bird in BIRDS)
    present_new = sum(prediction_path(bird, run.run_id).exists() for bird in BIRDS for run in new_runs(bird))
    try:
        rotation, rotation_problem = rotation_inputs(), None
        tags = tag_problems(rotation["report"])
    except SystemExit as reason:
        rotation, rotation_problem, tags = None, str(reason), []

    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<6} {row['records']:>2} of {row['planned']:>2}, {row['clean']:>2} clean")
    for problem in gap["problems"][:10]:
        print(f"  {problem}")
    print("originals        : " + ("as registered" if not originals else f"{len(originals)} problems\n  "
                                   + "\n  ".join(originals[:10])))
    print(f"read files       : {n_read - len(missing)} of {n_read} present"
          + ("" if not missing else "\n  " + "\n  ".join(missing[:10])))
    print("rotation seal    : " + (rotation_problem or (
        f"report and manifest at {rotation['attestation']['freeze_commit'][:12]}; drops as its freeze sec 4 and "
        f"report")))
    print("read tags        : " + ("as the rotation report" if not tags else "\n  " + "\n  ".join(tags[:10])))
    print(f"new predictions  : {present_new} of {n_new} files present (not read)")
    if not gap["complete"] or originals or missing or rotation_problem or tags:
        raise SystemExit("FAIL: the diagnostic found a problem; fix it before --predict")
    print("diagnostic passed; no test partition or prediction file was read")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", action="store_true", help="records, originals and files only; no test read")
    parser.add_argument("--predict", action="store_true", help="check the seal, then score the 60 runs on test")
    parser.add_argument("--overwrite", action="store_true", help="re-score runs that already have prediction files")
    parser.add_argument("--report", action="store_true", help="build the report from the saved predictions")
    args = parser.parse_args()
    if not (args.diagnostic or args.predict or args.report):
        raise SystemExit("pass --diagnostic, --predict, --report, or a combination")
    flags = env.disable_tf32()  # before any model is built or loaded
    print(f"tf32             : {flags}", flush=True)

    if args.diagnostic:
        diagnostic()
    if args.predict:
        env.require_fp32()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        run_predictions(device, overwrite=args.overwrite, tf32=flags)
    if args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report(tf32=flags)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        FIGURE_JSON.write_text(json.dumps(figure_rows(report), indent=2, default=str) + "\n")
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}\nwrote {FIGURE_JSON}\n")
        for number, block in enumerate(report["expectations"], 1):
            print(f"sec 6.{number}  {block['verdict']}")


if __name__ == "__main__":
    main()
