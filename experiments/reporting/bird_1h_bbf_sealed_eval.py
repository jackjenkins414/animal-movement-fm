# GenAI reference: I outlined this evaluator for our repaired transformer study; that being score BbF (C = 12 and 96), BbFE and
# LSTMBbR96 on test with the 1 h study's scoring, reuse every comparator from the 1 h and env studies' saved predictions the 
# comparisons etc, and GenAI wrote it. I checked its behaviour to validate it.
# 26-10-2026: when our extension (freeze sec 10) was appended to the drivers plan, which GenAI scoped for the original 120 runs 
# (F.ORIGINAL), so the sealed BbF evaluation reads exactly the runs it was sealed on
# Also did an extension mode; its own seal on sec 10s two subsections, manifest and folder, predict only the 90 extension runs; 
# read BbF at C = 12 / 96 etc 
"""
Sealed evaluation of the repaired transformer study (`preregistration/birds_1h_bbf_exploratory_freeze.md`); with
--extension, of its sec 10 extension (BbF at C = 24, 48 and 72) into results/birds_1h_bbf_extension_sealed/.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_1h_bbf_train as F  # noqa: E402
import bird_1h_env_sealed_eval as V  # noqa: E402

from animal_movement_fm.data import era5  # noqa: E402

E, env, TR = V.E, V.env, F.TR
b1h, RS, S2, E3, E4 = E.b1h, E.RS, E.S2, E.E3, E.E4

FREEZE = REPO / "preregistration" / F.FREEZE
SEALED = REPO / "results" / "birds_1h_bbf_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_1h_bbf_results.json"
REPORT_MD = SEALED / "birds_1h_bbf_results.md"
MANIFEST = SEALED / "birds_1h_bbf_sealed_manifest.json"

BIRDS, FOLD_OF, HORIZONS, CORE, C96 = E.BIRDS, E.FOLD_OF, E.HORIZONS, E.CORE, F.C96
FLOOR, MIN_SCORED = E.FLOOR, E.MIN_SCORED
SELECTION_FOLD = TR.PROBE_FOLDS[0]  # the repaired encoder was chosen on this fold's validation windows
FALLBACK_FROM = dt.date(2026, 10, 25)  # freeze sec 3: LSTMBbR96 not complete by then -> the V100 ablation runs
DEFAULTS = {name: F.TrainConfig.__dataclass_fields__[name].default for name in F.SWITCHES}

# label -> (study, arm in that study's plan, C). "bbf" runs are predicted here; "1h" and "env" are reloaded.
ARMS: dict[str, tuple[str, str, int]] = {
    "BbF": ("bbf", "BbF", CORE),
    "BbFE": ("bbf", "BbFE", CORE),
    "BbR": ("env", "BbR", CORE),
    "LSTMBbR": ("env", "LSTMBbR", CORE),
    "LSTMBbE": ("env", "LSTMBbE", CORE),
    "Bb": ("1h", "Bb", CORE),
    "LSTMBb": ("1h", "LSTMBb", CORE),
    "XGBBb": ("1h", "XGBBb", CORE),
    "BbF_C96": ("bbf", "BbF", C96),
    "LSTMBbR_C96": ("bbf", "LSTMBbR", C96),
    "Bb_C96": ("1h", "Bb", C96),
    "LSTMBb_C96": ("1h", "LSTMBb", C96),
    "XGBBb_C96": ("1h", "XGBBb", C96),
}
FALLBACK_DROPS = ("LSTMBbR_C96",)
# freeze sec 4: left - right in travel p90 skill against P, with metre and skill intervals
COMPARISONS: tuple[tuple[str, str], ...] = (
    ("BbF", "LSTMBbR"),  # C = 12, same hardware
    ("BbF", "LSTMBb"),  # C = 12, the main study's V100 runs, alongside
    ("BbF", "XGBBb"),
    ("BbF", "BbR"),  # the repair on test, C = 12, same hardware
    ("BbF", "Bb"),  # the same against the main study's V100 runs, alongside
    ("BbFE", "BbF"),  # weather on the repaired transformer
    ("BbF_C96", "LSTMBbR_C96"),  # C = 96, same hardware (the fallback drops it)
    ("BbF_C96", "LSTMBb_C96"),  # C = 96, the ablation's V100 runs
    ("BbF_C96", "XGBBb_C96"),
    ("BbF_C96", "Bb_C96"),  # the repair on test, C = 96
)
# (first pair's difference) - (second pair's), from the same resamples
CONTRASTS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "(BbFE - BbF) - (LSTMBbE - LSTMBbR)": (("BbFE", "BbF"), ("LSTMBbE", "LSTMBbR")),
}
# differences only: gpuhopper reruns against the V100 runs they repeat
HARDWARE: tuple[tuple[str, str], ...] = (("LSTMBbR", "LSTMBb"), ("BbR", "Bb"), ("LSTMBbR_C96", "LSTMBb_C96"))
TIED_BACK = tuple(label for label, (study, _, _) in ARMS.items() if study != "bbf")
SETS = ("scored", "common", "scored_without_fold6", "common_without_fold6")
BANDS = V.BANDS

if {arm for study, arm, _ in ARMS.values() if study == "bbf"} != set(F.ARMS):
    raise RuntimeError("the evaluator's arms differ from the driver's")
PLAN_INDEX = {run.run_id: i for i, run in enumerate(F.PLAN)}


def name(pair: tuple[str, str]) -> str:
    return f"{pair[0]} - {pair[1]}"


def labels_of(fallback: bool) -> tuple[str, ...]:
    return tuple(label for label in ARMS if not (fallback and label in FALLBACK_DROPS))


def comparisons_of(fallback: bool) -> dict[str, tuple[str, str]]:
    labels = labels_of(fallback)
    return {name(p): p for p in COMPARISONS if set(p) <= set(labels)}


def hardware_of(fallback: bool) -> dict[str, tuple[str, str]]:
    labels = labels_of(fallback)
    return {name(p): p for p in HARDWARE if set(p) <= set(labels)}


def planned_runs(fallback: bool) -> list:
    return [run for run in F.ORIGINAL if not (fallback and run in F.LSTMBBR96)]


def today() -> dt.date:
    return dt.date.today()


@contextlib.contextmanager
def bound():
    """Point the rotation evaluator's freeze, manifest and floor, and Tier 3's floor, at this study while the block runs."""
    saved = RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR
    RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = FREEZE, MANIFEST, FLOOR, FLOOR
    try:
        yield
    finally:
        RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = saved


def fp32() -> dict:
    """Full FP32 for inference, as in training: TF32 off in cuDNN and matmuls, refused if either stays on."""
    return V.fp32()


# Seal
def freeze_committed() -> dict:
    with bound():
        return RS.freeze_committed()


def assert_sealed() -> dict:
    """This freeze committed, clean, sec 6 and sec 7 filled, and the blob any manifest was attested at."""
    with bound():
        return RS.assert_sealed()


def source_reports(pairs: tuple[tuple[str, Path], ...] | None = None) -> dict[str, dict]:
    """The 1 h and env sealed reports the reused predictions come from (or `pairs`); refused if missing or unattested."""
    out = {}
    for study, path in pairs or (("birds_1h", E.REPORT_JSON), ("birds_1h_env", V.REPORT_JSON)):
        if not path.exists():
            raise SystemExit(f"FAIL: {path} does not exist; the reused predictions need their sealed report")
        report = json.loads(path.read_text())
        if report.get("study") != study or not (report.get("attestation") or {}).get("freeze_commit"):
            raise SystemExit(f"FAIL: {path.name} is not an attested {study} sealed report")
        out[study] = report
    return out


def fallback_guard(fallback: bool, records: dict[str, dict]) -> dict:
    """The freeze sec 3 fallback: only from 25-10-2026, and only while LSTMBbR96 is not complete."""
    status = completeness(records, F.LSTMBBR96)
    out = {"used": bool(fallback), "lstmbbr96_clean": status["n_clean"], "lstmbbr96_planned": status["n_planned"]}
    if not fallback:
        return out
    if today() < FALLBACK_FROM:
        raise SystemExit(f"FAIL: the C = 96 fallback applies only from {FALLBACK_FROM:%d-%m-%Y} (freeze sec 3)")
    if status["complete"]:
        raise SystemExit("FAIL: LSTMBbR96 is complete; the fallback is not needed")
    return out | {"from": FALLBACK_FROM.isoformat(),
                  "reason": "LSTMBbR96 not complete by 25-10-2026; the C = 96 LSTM comparison uses the 1 h ablation's "
                            "LSTM-Bb (V100) runs (freeze sec 3)"}


# Records
def read_records() -> dict[str, dict]:
    found = {}
    for path in sorted(F.RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("tier") != F.TAG or "run_id" not in record:
            print(f"  skipped {path.name}: not a {F.TAG} run record (tier {record.get('tier')!r})")
            continue
        found[record["run_id"]] = record
    return found


def run_problems(run, record: dict | None) -> list[str]:
    """What stops one planned run counting as finished cleanly, its switches and env settings included."""
    if record is None:
        return ["no record"]
    problems = [
        f"{key} {record.get(key)!r}, expected {want!r}"
        for key, want in (
            ("plan_index", PLAN_INDEX[run.run_id]),
            ("arm", run.arm),
            ("fold", run.fold),
            ("seed", run.seed),
            ("context", run.context),
            ("switches", run.switches),
            ("env", run.env),
            ("env_channels", F.ENV_CHANNELS if run.env else 0),
            ("env_clip_sd", era5.CLIP_SD if run.env else None),
            ("queue", F.queue_of(PLAN_INDEX[run.run_id])),
            ("opened_test_partition", False),
        )
        if record.get(key) != want
    ]
    config = record.get("config") or {}
    switches = {key: config.get(key) for key in F.SWITCHES}
    if switches != (DEFAULTS | run.switches):
        problems.append(f"config switches {switches}")
    if config.get("env_channels") != (F.ENV_CHANNELS if run.env else 0):
        problems.append(f"config env_channels {config.get('env_channels')}")
    if run.env:
        stats = record.get("env_stats") or {}
        if stats.get("channels") != list(era5.ENV_CHANNELS[: era5.N_STANDARDISED]):
            problems.append("env_stats missing or on other channels")
        if sorted(record.get("env_files") or {}) != sorted(record.get("pool_species") or []):
            problems.append("env_files do not cover the pool")
    if any((record.get("tf32") or {"missing": True}).values()):
        problems.append(f"tf32 {record.get('tf32')}")
    if record.get("applied_updates") != F.REGISTERED_STEPS or config.get("steps") != F.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {config.get('steps')}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (F.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def arm_key(run) -> str:
    return run.arm if run.context == CORE else f"{run.arm}_C{run.context}"


def completeness(records: dict[str, dict], runs: list) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in runs:
        row = by_arm.setdefault(arm_key(run), {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (index {PLAN_INDEX[run.run_id]}): {'; '.join(issues)}")
    return {"n_planned": len(runs), "n_records": sum(r["records"] for r in by_arm.values()),
            "n_clean": sum(r["clean"] for r in by_arm.values()), "by_arm": by_arm, "problems": problems,
            "complete": not problems}


def training_exclusions(payload: dict, records: dict[str, dict], runs: list) -> dict:
    """Each record against its 1 h base run's exclusions, pool and pinned window count (the 1 h check), per arm."""
    checked = 0
    for key in sorted({arm_key(run) for run in runs}):
        group = [run for run in runs if arm_key(run) == key and run.run_id in records]
        as_base = {run.base.run_id: records[run.run_id] for run in group}
        checked += E.assert_training_exclusions(payload, as_base, [run.base for run in group])["runs"]
    return {"status": "PASS", "runs": checked, "species_with_exclusions": len(payload["training_exclusions"])}


def env_twin_problems(records: dict[str, dict]) -> list[str]:
    """Every BbFE record's env_stats and env sha256 against its BbE twin's record in the env study."""
    problems = []
    for run in F.ORIGINAL:
        record = records.get(run.run_id)
        if run.env and record is not None:
            problems += [f"{run.run_id}: {p}" for p in
                         F.env_reference_problems(run, record.get("env_stats"), record.get("env_files"))]
    return problems


def env_file_problems(records: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """Every bird's env parquet against the one sha256 the BbFE records hold for it, and the file on disk now."""
    seen: dict[str, set[str]] = {}
    for run in F.ORIGINAL:
        for stem, digest in ((records.get(run.run_id) or {}).get("env_files") or {}).items():
            seen.setdefault(stem, set()).add(digest)
    problems = [f"{bird}: no BbFE record holds its sha256" for bird in BIRDS if bird not in seen]
    problems += [f"{bird}: records hold {len(seen[bird])} different sha256" for bird in BIRDS if len(seen.get(bird, ())) > 1]
    present = [bird for bird in BIRDS if env.env_path(bird).exists()]
    problems += [f"{bird}: {env.env_path(bird)} is missing" for bird in BIRDS if bird not in present]
    on_disk = env.env_files(present)
    problems += [f"{bird}: sha256 {on_disk[bird][:12]} on disk, {next(iter(seen[bird]))[:12]} in the records"
                 for bird in present if len(seen.get(bird, ())) == 1 and on_disk[bird] not in seen[bird]]
    return on_disk, problems


def repeat_checks(records: dict[str, dict]) -> dict[str, dict]:
    """Fold 6, seed 0 runs against the repair probe run with the same config: logged losses, point by point. No verdict."""
    out = {}
    for run in F.ORIGINAL:
        record = records.get(run.run_id)
        if not run.repeats or record is None:
            continue
        path = TR.RESULTS / f"{run.repeats}.json"
        if not path.exists():
            out[run.run_id] = {"repeats": run.repeats, "error": "probe record missing"}
            continue
        probe = json.loads(path.read_text())
        mine, theirs = record.get("loss_history") or [], probe.get("loss_history") or []
        same_length = len(mine) == len(theirs)
        diff = np.abs(np.asarray(mine, float) - np.asarray(theirs, float)) if same_length and mine else None
        out[run.run_id] = {
            "repeats": run.repeats,
            "points": len(mine),
            "identical": bool(same_length and mine == theirs),
            "max_abs_diff": float(diff.max()) if diff is not None else None,
            "final_loss_diff": (record["final_loss"] - probe["final_loss"]
                                if record.get("final_loss") is not None and probe.get("final_loss") is not None
                                else None),
        }
    return out


def checked_state(records: dict[str, dict], payload: dict, fallback: bool) -> tuple[dict, dict, dict]:
    """(completeness, exclusions, env sha256), refusing on any problem in the runs this evaluation scores."""
    env.check_main_study()
    runs = planned_runs(fallback)
    gap = completeness(records, runs)
    if gap["problems"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    exclusions = training_exclusions(payload, records, runs)
    problems = env_twin_problems(records)
    digests, file_problems = env_file_problems(records)
    if problems or file_problems:
        raise SystemExit("FAIL: BbFE's env inputs differ from the env study's\n  " + "\n  ".join(problems + file_problems))
    return gap, exclusions, digests


# Prediction
def source_dir(study: str) -> Path:
    return {"bbf": PREDICTIONS, "env": V.PREDICTIONS, "1h": E.PREDICTIONS}[study]


def prediction_path(bird: str, tag: str, study: str = "bbf") -> Path:
    return source_dir(study) / f"{bird}__{tag}.parquet"


def own_runs(bird: str, runs: list) -> list:
    return [run for run in runs if run.fold == FOLD_OF[bird]]


def tags_at(bird: str, label: str, horizon: int) -> list[tuple[str, int]]:
    """(run id, seed) of one bird's own-fold runs for a label at h: every seed of a neural arm, XGBoost's run at h."""
    study, arm, context = ARMS[label]
    fold = FOLD_OF[bird]
    if study == "bbf":
        runs = [r for r in F.ORIGINAL if r.arm == arm and r.context == context and r.fold == fold]
    elif study == "env":
        runs = [r for r in env.PLAN if r.arm == arm and r.fold == fold]
    else:
        runs = [r for r in b1h.PLAN if r.arm == arm and r.context == context and not r.species and r.fold == fold
                and (not r.xgb or r.horizon == horizon)]
    return [(run.run_id, run.seed) for run in runs]


def reused_files(bird: str, fallback: bool) -> list[tuple[str, str]]:
    """(study, tag) of every saved prediction this study reads for one bird: the other studies' P and comparators."""
    out = [("1h", "P"), ("1h", E.persistence_tag(C96)), ("env", "P")]
    for label in labels_of(fallback):
        study = ARMS[label][0]
        if study != "bbf":
            out += sorted({(study, tag) for h in HORIZONS for tag, _ in tags_at(bird, label, h)})
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def missing_reused(fallback: bool) -> list[str]:
    return [str(prediction_path(bird, tag, study)) for bird in BIRDS for study, tag in reused_files(bird, fallback)
            if not prediction_path(bird, tag, study).exists()]


def load_models(runs: list, records: dict[str, dict], device) -> list:
    """(run, model, k_h) per run, each checkpoint checked against its record: k_h, C, switches and env settings."""
    neural = []
    for run in runs:
        record = records[run.run_id]
        loaded = E3.load_reported(F.RESULTS, run.run_id)
        if loaded is None:
            raise SystemExit(f"FAIL: {run.run_id} has no reported checkpoint")
        model, state = loaded
        want = (list(record["horizon_k"]), run.context, record["switches"], record["env_channels"],
                record["env_stats"], record["env_clip_sd"])
        got = (list(state["horizon_k"]), state["context"], state.get("switches"), state.get("env_channels"),
               state.get("env_stats"), state.get("env_clip_sd"))
        if got != want or {key: state["config"].get(key) for key in F.SWITCHES} != DEFAULTS | run.switches:
            raise SystemExit(f"FAIL: {run.run_id}'s checkpoint disagrees with its record on k_h, C, switches or env")
        neural.append((run, model.to(device), np.asarray(record["horizon_k"], dtype=float)))
    return neural


def write_scored(bird, windows, runs, predicted, context, share, written) -> None:
    for run in runs:
        out = prediction_path(bird, run.run_id)
        E.frame_table(windows, predicted[run.run_id], context, share).to_parquet(out, index=False)
        written.append(out.name)


def predict_bird(bird: str, dropped, records: dict, device, overwrite: bool, runs: list) -> dict:
    """P at C = 12 and 96 (checked against each other and the reused P), BbF on the plain windows, BbFE on the env windows."""
    own = own_runs(bird, runs)
    for run in own:
        if bird in records[run.run_id]["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
    started = time.time()
    ids = E.scored_ids(bird, dropped)
    plain = E.test_windows(bird, ids, CORE)
    share = E.history_share(plain, CORE)
    persistence = E.frame_table(plain, None, CORE, share, persistence=True)
    print(f"\n{bird}: {plain.n_windows:,} test_scored windows, {len(plain.individuals)} individuals", flush=True)
    written: list[str] = []

    def keep_p(frame, tag):
        out = prediction_path(bird, tag)
        if overwrite or not out.exists():
            frame.to_parquet(out, index=False)
        written.append(out.name)

    def todo(group):
        done = [run for run in group if prediction_path(bird, run.run_id).exists() and not overwrite]
        written.extend(prediction_path(bird, run.run_id).name for run in done)
        return [run for run in group if run not in done]

    keep_p(persistence, "P")
    for study in ("1h", "env"):
        E.assert_same_persistence(persistence, pd.read_parquet(prediction_path(bird, "P", study)), f"{bird}: {study} P")
    plain_runs = todo([run for run in own if run.context == CORE and not run.env])
    if plain_runs:
        write_scored(bird, plain, plain_runs,
                     E.score_context(plain, load_models(plain_runs, records, device), [], device), CORE, share, written)

    long = E.test_windows(bird, ids, C96)
    long_share = E.history_share(long, C96)
    long_p = E.frame_table(long, None, C96, long_share, persistence=True)
    E.assert_same_persistence(persistence, long_p, f"{bird} C = {C96}")
    E.assert_same_persistence(long_p, pd.read_parquet(prediction_path(bird, E.persistence_tag(C96), "1h")),
                              f"{bird}: 1h P_C{C96}")
    keep_p(long_p, E.persistence_tag(C96))
    long_runs = todo([run for run in own if run.context == C96])
    if long_runs:
        write_scored(bird, long, long_runs,
                     E.score_context(long, load_models(long_runs, records, device), [], device), C96, long_share,
                     written)

    covariates = todo([run for run in own if run.env])
    if covariates:
        env.check_env_files([bird])
        base = env.env_windows(bird, ids)
        V.assert_same_windows(plain, base, bird)
        for payload, group in V.stats_groups(covariates, records).items():
            windows = replace(base, env_stats=env.stats_from_payload(json.loads(payload)))
            E.assert_same_persistence(persistence, E.frame_table(windows, None, CORE, share, persistence=True),
                                      f"{bird} env windows")
            write_scored(bird, windows, group,
                         E.score_context(windows, load_models(group, records, device), [], device), CORE, share,
                         written)
    print(f"  {len(written) - 2} runs written or present ({time.time() - started:.0f}s)", flush=True)
    return {"files": written, "test_windows": plain.n_windows}


def run_predictions(device, overwrite: bool = False, fallback: bool = False, tf32: dict | None = None) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    source_reports()  # refuse now rather than after a GPU job
    payload = b1h.load_registered()
    records = read_records()
    fallback_state = fallback_guard(fallback, records)
    gap, exclusions, digests = checked_state(records, payload, fallback)
    missing = missing_reused(fallback)
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} reused prediction files are missing\n  " + "\n  ".join(missing[:20]))
    drops, _, eligible = E.scoring_state(raise_on_mismatch=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest |= {"attestation": attestation, "scoring_rules": drops, "eligibility": eligible, "completeness": gap,
                 "training_exclusions": exclusions, "env_files": digests, "c96_fallback": fallback_state,
                 "tf32": tf32 if tf32 is not None else env.tf32_flags(), "species": {}, "reused": {}}
    runs = planned_runs(fallback)
    for bird in BIRDS:
        manifest["species"][bird] = predict_bird(bird, drops[bird]["dropped"], records, device, overwrite, runs)
        manifest["reused"][bird] = {f"{study}:{tag}": sha256(prediction_path(bird, tag, study))
                                    for study, tag in reused_files(bird, fallback)}
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def load_tables(bird: str, manifest: dict, fallback: bool) -> dict[str, pd.DataFrame]:
    """
    This study's saved predictions for the bird (anything outside its own fold refused) and the reused ones, each checked
    against the sha256 recorded at --predict; the reused P are checked equal to this study's.
    """
    tables = {path.stem.split("__", 1)[1]: pd.read_parquet(path)
              for path in sorted(PREDICTIONS.glob(f"{bird}__*.parquet"))}
    allowed = {run.run_id for run in own_runs(bird, planned_runs(fallback))} | {"P", E.persistence_tag(C96)}
    strays = sorted(set(tables) - allowed)
    if strays:
        raise SystemExit(f"FAIL: {bird} holds predictions from models outside its own fold: {strays}")
    for tag in ("P", E.persistence_tag(C96)):
        if tag not in tables:
            raise SystemExit(f"FAIL: {bird} has no {tag} predictions; P defines the skill denominator")
    E.assert_same_persistence(tables["P"], tables[E.persistence_tag(C96)], f"{bird} C = {C96}")
    recorded = (manifest.get("reused") or {}).get(bird) or {}
    for study, tag in reused_files(bird, fallback):
        path = prediction_path(bird, tag, study)
        key = f"{study}:{tag}"
        if not path.exists() or recorded.get(key) != sha256(path):
            raise SystemExit(f"FAIL: {path} is missing or differs from the sha256 recorded at --predict")
        frame = pd.read_parquet(path)
        if tag.startswith("P"):
            E.assert_same_persistence(tables["P"], frame, f"{bird}: {key}")
            continue
        tables[tag] = frame
    return tables


# Statistic
def contrast(first: tuple[str, str], second: tuple[str, str], arms: dict, frames: dict, resampler) -> dict:
    """(a - b) - (c - d) in travel p90 skill; the interval from the cell's resamples, as the comparisons beside it."""
    (a, b), (c, d) = first, second
    skills = {arm: E3.travel_skill(arms, arm) for arm in (a, b, c, d)}
    point = (skills[a] - skills[b]) - (skills[c] - skills[d])
    out = {"label": f"({a} - {b}) - ({c} - {d})", "difference": point, "band": RS.band(point),
           "noise_scale": E3.NOISE_SCALE}
    headline = {arm: E3.headline_frame(arms, frames, arm) for arm in (a, b, c, d)}
    if resampler.draws is None:
        out["skill_bootstrap"] = {"error": "fewer than two eligible individuals in travel"}
    elif any(frame is None for frame in headline.values()):
        out["skill_bootstrap"] = {"error": "an arm has no headline frame"}
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = {arm: 1.0 - resampler.p90(arm, frame) / resampler.reference for arm, frame in headline.items()}
        out["skill_bootstrap"] = {"n_individuals": resampler.n_individuals, "n_resamples": S2.N_BOOTSTRAP,
                                  "seed": S2.BOOTSTRAP_SEED, "difference": V._interval((s[a] - s[b]) - (s[c] - s[d]))}
    return out


def cell_readings(persistence: pd.DataFrame, frames: dict, parsed: dict, boundary: float,
                  expected: dict[str, set[int]], label: str, fallback: bool) -> tuple[dict, dict]:
    """Arm blocks and readings for one bird at one horizon; each label must hold exactly its planned seeds."""
    labels = labels_of(fallback)
    with bound():
        E4.assert_no_collisions(frames, parsed, parse=E.parse_tag)
        summaries = E3.summarise_all(frames, persistence, boundary, parsed, parse=E.parse_tag)
        problems = [arm for arm in ("P", *labels) if arm not in summaries]
        for arm in labels:
            seeds = {key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0}
            if arm in summaries and seeds != expected[arm]:
                problems.append(f"{arm} has seeds {sorted(seeds)}, expected {sorted(expected[arm])}")
        if problems:
            raise SystemExit(f"FAIL: {label}: " + "; ".join(problems))
        arms = RS.arm_blocks(summaries)
        resampler = RS.Resampler(persistence, boundary)
        readings = {
            "against_p": {arm: E3.travel_skill(arms, arm) for arm in labels},
            "comparisons": {key: RS.compare(a, b, arms, frames, persistence, boundary, resampler)
                            for key, (a, b) in comparisons_of(fallback).items()},
            "contrasts": {key: contrast(first, second, arms, frames, resampler)
                          for key, (first, second) in CONTRASTS.items()},
            "hardware": {key: RS.compare(a, b, arms, frames, persistence, boundary, resampler)
                         for key, (a, b) in hardware_of(fallback).items()},
        }
    return arms, readings


def bird_cell(bird: str, horizon: int, tables: dict, eligible: dict, fallback: bool) -> dict:
    counts = eligible[str(horizon)]
    boundary = b1h.BOUNDARY_M[(bird, horizon)]
    entry = {"boundary_m": boundary, **counts}
    if not counts["scored"]:
        return entry
    persistence = E.at(tables["P"], horizon)
    keep = S2.eligible_mask(persistence, FLOOR)
    entry["share_in_travel"] = float(S2.stratum_masks(persistence, boundary)["travel"][keep].mean())
    frames, parsed, expected = {"P": persistence}, {}, {}
    for label in labels_of(fallback):
        pairs = tags_at(bird, label, horizon)
        expected[label] = {seed for _, seed in pairs}
        for tag, seed in pairs:
            if tag not in tables:
                raise SystemExit(f"FAIL: {bird} has no predictions for {tag}; run --predict")
            frames[tag] = E.at(tables[tag], horizon)
            E.assert_aligned(frames[tag], persistence, f"{bird} {horizon} h {tag}")
            parsed[tag] = {"arm": label, "seed": seed}
    entry["arms"], entry["readings"] = cell_readings(persistence, frames, parsed, boundary, expected,
                                                     f"{bird} {horizon} h", fallback)
    return entry


def across_birds(report: dict, fallback: bool) -> dict:
    """Per horizon, over the scored birds, the common set, and both without the selection fold's held-out birds."""
    common = [bird for bird, entry in report["species"].items()
              if all(entry["horizons"][str(h)]["scored"] for h in HORIZONS)]
    out = {"common_set": common, "selection_fold": SELECTION_FOLD, "horizons": {}}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        sets = {"scored": scored, "common": common,
                "scored_without_fold6": [b for b in scored if FOLD_OF[b] != SELECTION_FOLD],
                "common_without_fold6": [b for b in common if FOLD_OF[b] != SELECTION_FOLD]}
        block: dict = {"scored": scored, "skill": {}, "comparisons": {}, "contrasts": {}, "hardware": {}}
        for label, birds in sets.items():
            cells = [report["species"][bird]["horizons"][str(horizon)] for bird in birds]
            for arm in labels_of(fallback):
                block["skill"].setdefault(arm, {})[label] = V._median(c["readings"]["against_p"][arm] for c in cells)
            for key in comparisons_of(fallback):
                block["comparisons"].setdefault(key, {})[label] = V._effect_summary(
                    cells, "comparisons", key, ("skill_bootstrap", "skill_difference"))
            for key in CONTRASTS:
                block["contrasts"].setdefault(key, {})[label] = V._effect_summary(
                    cells, "contrasts", key, ("skill_bootstrap", "difference"))
            for key in hardware_of(fallback):
                summary = V._effect_summary(cells, "hardware", key, ("skill_bootstrap", "skill_difference"))
                block["hardware"].setdefault(key, {})[label] = {"difference": summary["difference"]}
        out["horizons"][str(horizon)] = block
    return out


def tie_back(report: dict, sources: dict[str, dict], fallback: bool) -> dict:
    """
    Each reused arm's skill here minus its source report's, per bird x horizon, with max |diff| and median. The 1 h
    report's C = 96 skill is its ablation's "all" stratum. Descriptive: no pass/fail.
    """
    out: dict = {"sources": {study: str(path.relative_to(REPO)) for study, path in
                             (("birds_1h", E.REPORT_JSON), ("birds_1h_env", V.REPORT_JSON))},
                 "attestations": {study: r.get("attestation") for study, r in sources.items()},
                 "cells": {}, "arms": {}, "scored_differs": []}
    labels = [label for label in TIED_BACK if label in labels_of(fallback)]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            theirs = {study: r["species"][bird]["horizons"][horizon] for study, r in sources.items()}
            for study, other in theirs.items():
                if bool(cell["scored"]) != bool(other["scored"]):
                    out["scored_differs"].append(f"{bird} {horizon} h ({study})")
            if not (cell["scored"] and all(other["scored"] for other in theirs.values())):
                continue
            diffs = {}
            for label in labels:
                study, arm, context = ARMS[label]
                if study == "env":
                    reference = theirs["birds_1h_env"]["readings"]["against_p"][arm]
                elif context == CORE:
                    reference = E3.travel_skill(theirs["birds_1h"]["arms"], arm)
                else:
                    ablation = (theirs["birds_1h"].get("ablation") or {}).get(str(context)) or {}
                    reference = (ablation.get("all") or {}).get(arm, {}).get("skill", float("nan"))
                diffs[label] = cell["readings"]["against_p"][label] - reference
            out["cells"].setdefault(bird, {})[horizon] = diffs
    for label in labels:
        values = np.array([c[label] for per_bird in out["cells"].values() for c in per_bird.values()], dtype=float)
        finite = values[np.isfinite(values)]
        out["arms"][label] = {"n": int(finite.size),
                              "max_abs_diff": float(np.abs(finite).max()) if finite.size else None,
                              "median_diff": float(np.median(finite)) if finite.size else None}
    return out


# Report
def build_report(fallback: bool = False, tf32: dict | None = None) -> dict:
    sources = source_reports()
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    used = (manifest.get("c96_fallback") or {}).get("used")
    if used is not bool(fallback):
        raise SystemExit(f"FAIL: --predict ran with the C = 96 fallback {used}; --report was asked for {bool(fallback)}")
    b1h.check_boundaries()
    records = read_records()
    gap = completeness(records, planned_runs(fallback))
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    report: dict = {
        "study": "birds_1h_bbf",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "bbf": "Bb with the repaired encoder (the repair probe's V4 switches)",
        "switches": F.SWITCHES,
        "evaluation_floor": FLOOR,
        "min_scored_individuals": MIN_SCORED,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": E3.NOISE_SCALE,
        "c96_fallback": manifest["c96_fallback"],
        "tf32": {"predict": manifest.get("tf32"), "report": tf32 if tf32 is not None else env.tf32_flags()},
        "arms": {label: {"study": s, "arm": a, "context": c} for label, (s, a, c) in ARMS.items()
                 if label in labels_of(fallback)},
        "comparisons": list(comparisons_of(fallback)),
        "contrasts": list(CONTRASTS),
        "hardware": list(hardware_of(fallback)),
        "completeness": gap,
        "scoring_rules": manifest["scoring_rules"],
        "species": {},
    }
    for bird in BIRDS:
        tables = load_tables(bird, manifest, fallback)
        entry = {"fold": FOLD_OF[bird], "dropped": manifest["scoring_rules"][bird]["dropped"], "horizons": {}}
        for horizon in HORIZONS:
            entry["horizons"][str(horizon)] = bird_cell(bird, horizon, tables, manifest["eligibility"][bird], fallback)
        report["species"][bird] = entry
    report["across_birds"] = across_birds(report, fallback)
    report["tie_back"] = tie_back(report, sources, fallback)
    return report


def _s(value) -> str:
    return RS._s(value)


def _with_interval(reading: dict, key: tuple[str, str], point: str = "skill_difference") -> str:
    block = (reading.get(key[0]) or {}).get(key[1])
    return f"{_s(reading.get(point))} {RS._interval(block)}"


def write_markdown(report: dict, path: Path) -> None:
    arms = list(report["arms"])
    lines = ["# Repaired transformer (BbF): sealed evaluation (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap, fallback = report["completeness"], report["c96_fallback"]
    lines += [f"BbF is {report['bbf']}: {report['switches']}. Runs scored: {gap['n_clean']} of {gap['n_planned']} "
              f"clean. Travel p90 skill against P on the 1 h main eval's test_scored windows, at the "
              f"{report['evaluation_floor']}-window floor, a bird scored at h with ≥ {report['min_scored_individuals']} "
              f"test_scored individuals there; median of seeds; paired bootstrap 2,000 at seed 42. C = 96 arms are "
              f"scored on the same windows. TF32 at prediction: {report['tf32']['predict']}.", ""]
    lines += [("C = 96 fallback used: " + fallback["reason"]) if fallback.get("used")
              else f"C = 96 fallback not used: LSTMBbR96 {fallback['lstmbbr96_clean']} of "
                   f"{fallback['lstmbbr96_planned']} clean.", ""]

    lines += ["## Per bird and horizon: skill against P", "",
              "| fold | bird | h | scored (at floor / with windows) | travel share | " + " | ".join(arms) + " |",
              "|---:|---|---:|---|---:|" + "---:|" * len(arms)]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            n = f"{cell['at_floor']} / {cell['individuals']}"
            if not cell["scored"]:
                lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | unscored, {n} |" + " — |" * (1 + len(arms)))
                continue
            lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | {n} | {cell['share_in_travel']:.3f} | "
                         + " | ".join(_s(cell["readings"]["against_p"][arm]) for arm in arms) + " |")

    columns = report["comparisons"] + report["contrasts"]
    lines += ["", "## Per bird and horizon: comparisons and contrast (skill, 95% interval)", "",
              "| fold | bird | h | " + " | ".join(columns) + " |", "|---:|---|---:|" + "---|" * len(columns)]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            if not cell["scored"]:
                continue
            r = cell["readings"]
            parts = [_with_interval(r["comparisons"][k], ("skill_bootstrap", "skill_difference"))
                     for k in report["comparisons"]]
            parts += [_with_interval(r["contrasts"][k], ("skill_bootstrap", "difference"), "difference")
                      for k in report["contrasts"]]
            lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | " + " | ".join(parts) + " |")

    across = report["across_birds"]
    lines += ["", f"## Across birds: skill against P (median (n); common set = scored at all three horizons, "
                  f"{len(across['common_set'])} birds; without fold {across['selection_fold']}, where the repaired "
                  "encoder was chosen)", "",
              "| h | arm | scored | common | scored without fold 6 | common without fold 6 |",
              "|---:|---|---:|---:|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for arm in arms:
            entry = block["skill"][arm]
            lines.append(f"| {horizon} | {arm} | " + " | ".join(E._median_cell(entry[s]) for s in SETS) + " |")
    lines += ["", "## Across birds: comparisons and contrast (median (n); above +0.05 / within / below −0.05 / not "
                  "scored; intervals excluding zero)", "",
              "| h | reading | scored | common | median without fold 6, scored / common |", "|---:|---|---|---|---|"]
    for horizon, block in across["horizons"].items():
        for kind in ("comparisons", "contrasts"):
            for key, entry in block[kind].items():
                cells = [f"{E._median_cell(entry[s]['difference'])}; "
                         + " / ".join(str(entry[s]["bands"][b]) for b in BANDS)
                         + f"; {entry[s]['intervals_excluding_zero']} excl. 0" for s in ("scored", "common")]
                without = " / ".join(E._median_cell(entry[s]["difference"]) for s in SETS[2:])
                lines.append(f"| {horizon} | {key} | {cells[0]} | {cells[1]} | {without} |")
    lines += ["", "## Hardware across birds (gpuhopper rerun − V100 run; median difference (n); no pass/fail)", "",
              "| h | difference | scored | common |", "|---:|---|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for key, entry in block["hardware"].items():
            lines.append(f"| {horizon} | {key} | {E._median_cell(entry['scored']['difference'])} | "
                         f"{E._median_cell(entry['common']['difference'])} |")
    tie = report["tie_back"]
    lines += ["", "## Tie-back of the reused arms to their sealed reports (skill here − source report; no pass/fail)",
              "", "| arm | cells | max \\|diff\\| | median diff |", "|---|---:|---:|---:|"]
    for arm, entry in tie["arms"].items():
        top = f"{entry['max_abs_diff']:.2e}" if entry["max_abs_diff"] is not None else "—"
        median = f"{entry['median_diff']:+.2e}" if entry["median_diff"] is not None else "—"
        lines.append(f"| {arm} | {entry['n']} | {top} | {median} |")
    if tie["scored_differs"]:
        lines += ["", "Scored in one report only: " + ", ".join(tie["scored_differs"]) + "."]
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: run records, the env and probe records and env files only; no test partition is opened
def section7_markdown(records: dict, gap: dict, exclusions: str, twins: list[str], digests: dict,
                      file_problems: list[str], repeats: dict, fallback: dict) -> str:
    present = [records[r.run_id] for r in F.ORIGINAL if r.run_id in records]
    devices = ", ".join(f"{d} {n}" for d, n in sorted(Counter(r.get("device") for r in present).items())) or "none"
    queues = ", ".join(f"{q} {n}" for q, n in sorted(Counter(r.get("pbs_queue") for r in present).items())) or "none"
    tf32_off = sum(not any((r.get("tf32") or {"missing": True}).values()) for r in present)
    clipped = sum(r.get("env_clip_sd") == era5.CLIP_SD for r in present if r.get("env"))
    by_arm = "; ".join(f"{arm} {row['clean']} of {row['planned']}" for arm, row in gap["by_arm"].items())
    repeat_text = "; ".join(
        f"{run_id} vs {c['repeats']}: " + ("identical" if c.get("identical") else
                                          c.get("error") or f"max |Δ| {c['max_abs_diff']:.3g} over {c['points']} points")
        for run_id, c in repeats.items()) or "none recorded"
    lines = [
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean ({by_arm}) |",
        f"| training | {F.REGISTERED_STEPS:,} updates applied, none withheld, no non-finite steps on every clean run; "
        f"TF32 off on {tf32_off}; devices {devices} |",
        f"| switches | BbF and BbFE {F.SWITCHES}; LSTMBbR96 none, as recorded on every clean run |",
        f"| PBS queues | {queues} |",
        f"| BbFE env inputs | {clipped} runs at ±{era5.CLIP_SD:g} SD; env_stats and env sha256 against the BbE twins: "
        + ("equal" if not twins else "differ: " + "; ".join(twins[:5])) + " |",
        f"| env files | {len(digests)} birds; sha256 " + ("as recorded" if not file_problems
                                                          else "differ: " + "; ".join(file_problems[:5])) + " |",
        f"| training exclusions | {exclusions} |",
        f"| repeats of the probe (fold 6, seed 0, logged losses) | {repeat_text} |",
        "| C = 96 fallback | " + ("used: " + fallback["reason"] if fallback.get("used") else
                                 f"not used; LSTMBbR96 {fallback['lstmbbr96_clean']} of "
                                 f"{fallback['lstmbbr96_planned']} clean") + " |",
        "| test-partition reads | none: the diagnostic reads run records, the env and probe records and env files only |",
        f"| deferred or missing | {'none' if gap['complete'] else '; '.join(gap['problems'])} |",
    ]
    return "\n".join(lines)


def diagnostic(fallback: bool = False) -> None:
    """Completeness, switches, BbFE's env inputs, exclusions, the probe repeats and the sec 7 block. Opens no test partition."""
    freeze_committed()
    with bound():
        text = RS.committed_freeze()
        for number in (6, 7):
            try:
                RS.check_filled(number, RS.section_body(text, number))
                print(f"sec {number}            : filled")
            except SystemExit as reason:
                print(f"sec {number}            : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    env.check_main_study()
    payload = b1h.load_registered()
    records = read_records()
    fallback_state = fallback_guard(fallback, records)
    runs = planned_runs(fallback)
    gap = completeness(records, runs)
    try:
        exclusions = json.dumps(training_exclusions(payload, records, runs))
        exclusions_ok = True
    except SystemExit as reason:
        exclusions, exclusions_ok = str(reason), False
    twins = env_twin_problems(records)
    digests, file_problems = env_file_problems(records)
    repeats = repeat_checks(records)

    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for key, row in gap["by_arm"].items():
        print(f"  {key:<12} {row['records']:>3} of {row['planned']:>3}, {row['clean']:>3} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print(f"exclusions       : {exclusions}")
    print(f"BbFE twins       : " + ("equal" if not twins else "\n  " + "\n  ".join(twins[:20])))
    print(f"env files        : {len(digests)} present; " + ("as recorded" if not file_problems
                                                             else "\n  " + "\n  ".join(file_problems[:20])))
    for run_id, check in repeats.items():
        print(f"repeat           : {run_id} vs {check['repeats']}: {json.dumps(check)}")
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(records, gap, exclusions, twins, digests, file_problems, repeats, fallback_state))
    if not (gap["complete"] and exclusions_ok and not twins and not file_problems):
        raise SystemExit("FAIL: the diagnostic found a problem; sec 7 must not be filled until it passes")


# Extension; BbF at C = 24, 48 and 72, with its own seal, manifest and folder. Only the 90 extension runs are predicted; BbF at C = 12 and 96 and 
# every comparator are read from sealed prediction files.
EXT_SEALED = REPO / "results" / "birds_1h_bbf_extension_sealed"
EXT_PREDICTIONS = EXT_SEALED / "predictions"
EXT_REPORT_JSON = EXT_SEALED / "birds_1h_bbf_extension_results.json"
EXT_REPORT_MD = EXT_SEALED / "birds_1h_bbf_extension_results.md"
EXT_MANIFEST = EXT_SEALED / "birds_1h_bbf_extension_sealed_manifest.json"
EXT_SECTION = 10
EXT_SUBSECTIONS = ("Extension expectations", "Extension as trained")
EXT_CONTEXTS = F.EXTENSION_CONTEXTS  # 24, 48, 72
LADDER = (*EXT_CONTEXTS, C96)  # every C > 12 of the ablation-shaped block
EXT_ABLATION_ARMS = ("BbF", *E.ABLATION_ARMS)  # BbF, then the 1 h ablation's Bb, LSTM-Bb and XGB-Bb
LSTM_COMPARATOR = ("the 1 h ablation's LSTM-Bb, trained on gpuvolta (V100) and predicted on gpuhopper (H200, FP32); "
                   "BbF is trained and predicted on gpuhopper (H200, FP32). Freeze sec 10: the hardware differs")


def ablation_label(arm: str, context: int) -> str:
    return arm if context == CORE else f"{arm}_C{context}"


# label -> (study, arm, C). "ext" runs are predicted here; "bbf" (the sealed BbF study) and "1h" are read.
EXT_ARMS: dict[str, tuple[str, str, int]] = {
    "BbF": ("bbf", "BbF", CORE),
    **{f"BbF_C{c}": ("ext", "BbF", c) for c in EXT_CONTEXTS},
    "BbF_C96": ("bbf", "BbF", C96),
    **{ablation_label(arm, c): ("1h", arm, c) for arm in E.ABLATION_ARMS for c in (CORE, *LADDER)},
}
# freeze sec 10, at each extension C: BbF against P, LSTM-Bb and XGB-Bb (skill difference, metre and skill intervals)
EXT_COMPARISONS: tuple[tuple[str, str], ...] = tuple(
    pair for c in EXT_CONTEXTS for pair in ((f"BbF_C{c}", "P"), (f"BbF_C{c}", f"LSTMBb_C{c}"),
                                            (f"BbF_C{c}", f"XGBBb_C{c}")))
EXT_TIED_BACK = tuple(label for label, (study, _, _) in EXT_ARMS.items() if study != "ext")

if LADDER != tuple(E.ABLATION_CONTEXTS) or any(ARMS[k] != EXT_ARMS[k] for k in ("BbF", "BbF_C96")):
    raise RuntimeError("the extension's ladder is not the 1 h ablation's, or its BbF labels are not the sealed study's")
if any(ablation_label(arm, c) not in EXT_ARMS for arm in EXT_ABLATION_ARMS for c in (CORE, *LADDER)):
    raise RuntimeError("every arm of the ablation-shaped block needs a label at C = 12 and at every C of the ladder")


@contextlib.contextmanager
def ext_bound():
    """As bound(), with the extension's manifest, so the extension never reads or writes the sealed BbF manifest."""
    saved = RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR
    RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = FREEZE, EXT_MANIFEST, FLOOR, FLOOR
    try:
        yield
    finally:
        RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = saved


# Extension seal
def subsection_body(section: str, heading: str) -> str:
    """The body under `### heading` in a section's text, up to the next ### or ## heading."""
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == f"### {heading}"), None)
    if start is None:
        raise SystemExit(f"FAIL: {FREEZE.name} sec {EXT_SECTION} has no '### {heading}'")
    body = []
    for line in lines[start + 1:]:
        if line.startswith(("### ", "## ")):
            break
        body.append(line)
    return "\n".join(body).strip()


def ext_section_problems(text: str) -> list[str]:
    """Sec 10 and both of its subsections in `text`, each filled and free of placeholder wording; empty if so."""
    with ext_bound():
        section = RS.section_body(text, EXT_SECTION)
    problems = []
    for heading in EXT_SUBSECTIONS:
        try:
            body = subsection_body(section, heading)
        except SystemExit as reason:
            problems.append(str(reason).removeprefix("FAIL: "))
            continue
        found = RS.PLACEHOLDER.search(body)
        if not body:
            problems.append(f"sec {EXT_SECTION} '{heading}' is empty in the committed version")
        elif found:
            problems.append(f"sec {EXT_SECTION} '{heading}' still says {found.group(0)!r} in the committed version")
    return problems


def ext_assert_sealed() -> dict:
    """
    The freeze committed and clean, secs 6 and 7 filled, both sec 10 subsections filled in the committed version, and the
    blob any extension manifest was attested at. Run by --extension --predict only.
    """
    with ext_bound():
        attestation = RS.assert_sealed()
        text = RS.committed_freeze()
    problems = ext_section_problems(text)
    if problems:
        raise SystemExit("FAIL: " + "; ".join(problems))
    attestation["sections_filled"] = [6, 7, *(f"{EXT_SECTION}: {heading}" for heading in EXT_SUBSECTIONS)]
    return attestation


def ext_source_reports() -> dict[str, dict]:
    """The 1 h and BbF sealed reports the read predictions come from; refused if missing or unattested."""
    return source_reports((("birds_1h", E.REPORT_JSON), ("birds_1h_bbf", REPORT_JSON)))


def ext_checked_state(records: dict[str, dict], payload: dict) -> tuple[dict, dict]:
    """(completeness, exclusions) of the 90 extension runs, refusing on any problem."""
    gap = completeness(records, F.EXTENSION)
    if gap["problems"]:
        raise SystemExit("FAIL: extension runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    return gap, training_exclusions(payload, records, F.EXTENSION)


# Extension files
def ext_path(bird: str, tag: str, study: str = "ext") -> Path:
    return {"ext": EXT_PREDICTIONS, "bbf": PREDICTIONS, "1h": E.PREDICTIONS}[study] / f"{bird}__{tag}.parquet"


def ext_own_runs(bird: str) -> list:
    return [run for run in F.EXTENSION if run.fold == FOLD_OF[bird]]


def ext_tags_at(bird: str, label: str, horizon: int) -> list[tuple[str, int]]:
    """(run id, seed) of one bird's own-fold runs for an extension label at h (XGBoost: its run at h)."""
    study, arm, context = EXT_ARMS[label]
    fold = FOLD_OF[bird]
    if study in ("ext", "bbf"):
        runs = [r for r in (F.EXTENSION if study == "ext" else F.ORIGINAL)
                if r.arm == arm and r.context == context and r.fold == fold]
    else:
        runs = [r for r in b1h.PLAN if r.arm == arm and r.context == context and not r.species and r.fold == fold
                and (not r.xgb or r.horizon == horizon)]
    return [(run.run_id, run.seed) for run in runs]


def ext_read_files(bird: str) -> list[tuple[str, str]]:
    """(study, tag) of every sealed prediction the extension reads for one bird: P at each C and every read label."""
    out = [("1h", "P"), *(("1h", E.persistence_tag(c)) for c in LADDER), ("bbf", "P"), ("bbf", E.persistence_tag(C96))]
    for label, (study, _, _) in EXT_ARMS.items():
        if study != "ext":
            out += sorted({(study, tag) for h in HORIZONS for tag, _ in ext_tags_at(bird, label, h)})
    return out


def ext_missing_reads() -> list[str]:
    return [str(ext_path(bird, tag, study)) for bird in BIRDS for study, tag in ext_read_files(bird)
            if not ext_path(bird, tag, study).exists()]


def assert_same_share(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    """The strata come from history_share, so a P at C read from another study must carry the same shares."""
    if not np.array_equal(reference["history_share"].to_numpy(float), other["history_share"].to_numpy(float),
                          equal_nan=True):
        raise SystemExit(f"FAIL: {label}: history_share differs from the sealed 1 h P at that C")


# Extension prediction
def ext_predict_bird(bird: str, dropped, records: dict, device, overwrite: bool) -> dict:
    """P at each extension C (checked against the sealed 1 h P and P_C), then the bird's own-fold BbF runs at that C."""
    own = ext_own_runs(bird)
    for run in own:
        if bird in records[run.run_id]["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
    ids = E.scored_ids(bird, dropped)
    reference = pd.read_parquet(ext_path(bird, "P", "1h"))
    written: list[str] = []
    print(f"\n{bird}: {len(own)} extension runs", flush=True)
    for context in EXT_CONTEXTS:
        started = time.time()
        tag = E.persistence_tag(context)
        windows = E.test_windows(bird, ids, context)
        share = E.history_share(windows, context)
        persistence = E.frame_table(windows, None, context, share, persistence=True)
        E.assert_same_persistence(reference, persistence, f"{bird} C = {context}")
        sealed = pd.read_parquet(ext_path(bird, tag, "1h"))
        E.assert_same_persistence(sealed, persistence, f"{bird}: 1h {tag}")
        assert_same_share(sealed, persistence, f"{bird} {tag}")
        out = ext_path(bird, tag)
        if overwrite or not out.exists():
            persistence.to_parquet(out, index=False)
        written.append(out.name)
        group = [run for run in own if run.context == context]
        todo = [run for run in group if overwrite or not ext_path(bird, run.run_id).exists()]
        written += [ext_path(bird, run.run_id).name for run in group if run not in todo]
        if todo:
            predicted = E.score_context(windows, load_models(todo, records, device), [], device)
            for run in todo:
                path = ext_path(bird, run.run_id)
                E.frame_table(windows, predicted[run.run_id], context, share).to_parquet(path, index=False)
                written.append(path.name)
        print(f"  C = {context}: {len(todo)} runs scored, {len(group) - len(todo)} existed "
              f"({time.time() - started:.0f}s)", flush=True)
    return {"files": written}


def ext_run_predictions(device, overwrite: bool = False, tf32: dict | None = None) -> dict:
    attestation = ext_assert_sealed()  # nothing before this line touches a test partition
    ext_source_reports()
    payload = b1h.load_registered()
    records = read_records()
    gap, exclusions = ext_checked_state(records, payload)
    missing = ext_missing_reads()
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} sealed prediction files are missing\n  " + "\n  ".join(missing[:20]))
    drops, _, eligible = E.scoring_state(raise_on_mismatch=True)
    EXT_PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(EXT_MANIFEST.read_text()) if EXT_MANIFEST.exists() else {}
    manifest |= {"attestation": attestation, "scoring_rules": drops, "eligibility": eligible, "completeness": gap,
                 "training_exclusions": exclusions, "tf32": tf32 if tf32 is not None else env.tf32_flags(),
                 "species": {}, "read": {}}
    for bird in BIRDS:
        manifest["species"][bird] = ext_predict_bird(bird, drops[bird]["dropped"], records, device, overwrite)
        manifest["read"][bird] = {f"{study}:{tag}": sha256(ext_path(bird, tag, study))
                                  for study, tag in ext_read_files(bird)}
    EXT_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Extension frames and statistic
def ext_load_tables(bird: str, manifest: dict) -> dict[str, pd.DataFrame]:
    """
    The extension's predictions for the bird (anything outside its own fold refused) and the sealed files it reads, each
    against the sha256 recorded at --predict. Every P equals the 1 h P; each P at C equals the 1 h P at that C, shares
    included. "P" and "P_C{C}" are the 1 h files.
    """
    tables = {path.stem.split("__", 1)[1]: pd.read_parquet(path)
              for path in sorted(EXT_PREDICTIONS.glob(f"{bird}__*.parquet"))}
    allowed = {run.run_id for run in ext_own_runs(bird)} | {E.persistence_tag(c) for c in EXT_CONTEXTS}
    strays = sorted(set(tables) - allowed)
    if strays:
        raise SystemExit(f"FAIL: {bird} holds extension predictions from models outside its own fold: {strays}")
    own_p = {}
    for context in EXT_CONTEXTS:
        tag = E.persistence_tag(context)
        if tag not in tables:
            raise SystemExit(f"FAIL: {bird} has no extension {tag}; run --extension --predict")
        own_p[tag] = tables.pop(tag)
    recorded = (manifest.get("read") or {}).get(bird) or {}
    for study, tag in ext_read_files(bird):
        path, key = ext_path(bird, tag, study), f"{study}:{tag}"
        if not path.exists() or recorded.get(key) != sha256(path):
            raise SystemExit(f"FAIL: {path} is missing or differs from the sha256 recorded at --extension --predict")
        frame = pd.read_parquet(path)
        if tag.startswith("P") and key != "1h:P":  # ext_read_files lists the 1 h P first
            E.assert_same_persistence(tables["P"], frame, f"{bird}: {key}")
        if study == "1h" or not tag.startswith("P"):
            tables[tag] = frame
    for tag, frame in own_p.items():
        E.assert_same_persistence(tables[tag], frame, f"{bird}: extension {tag}")
        assert_same_share(tables[tag], frame, f"{bird}: extension {tag}")
    return tables


def ext_ablation_cell(bird: str, horizon: int, tables: dict, frames: dict, persistence: pd.DataFrame,
                      keep: np.ndarray, boundary: float) -> dict:
    """
    The 1 h ablation_cell for BbF beside the ablation's arms: per C > 12 and stratum, skill against P at C and at C = 12
    on the same windows (median of the seeds' travel p90), and the difference. Strata from the share observed at C.
    """
    out: dict = {}
    for context in LADDER:
        at_c = E.at(tables[E.persistence_tag(context)], horizon)
        E.assert_same_persistence(persistence, at_c, f"{bird} {horizon} h C = {context}")
        share = at_c["history_share"].to_numpy(float)
        masks = {"all": keep, "gappy": keep & (share < E.FULL_SHARE), "full": keep & (share >= E.FULL_SHARE)}
        for stratum, mask in masks.items():
            reference, n_individuals, n_travel = E.travel_p90(persistence, persistence, mask, boundary)
            cell = {"n_individuals": n_individuals, "n_travel_windows": n_travel, "p_p90_m": reference}
            for arm in EXT_ABLATION_ARMS:
                skills = {}
                for key, c in (("C", context), ("C12", CORE)):
                    values = [E.travel_p90(frames[tag], persistence, mask, boundary)[0]
                              for tag, _ in ext_tags_at(bird, ablation_label(arm, c), horizon)]
                    skills[key] = E3.skill(float(np.median(values)) if values else float("nan"), reference)
                cell[arm] = {"skill": skills["C"], "skill_C12": skills["C12"], "difference": skills["C"] - skills["C12"]}
            out.setdefault(str(context), {})[stratum] = cell
    return out


def ext_cell(bird: str, horizon: int, tables: dict, eligible: dict) -> dict:
    """One bird at one horizon: each label's skill, the sec 10 comparisons, and the ablation-shaped block."""
    counts = eligible[str(horizon)]
    boundary = b1h.BOUNDARY_M[(bird, horizon)]
    entry = {"boundary_m": boundary, **counts}
    if not counts["scored"]:
        return entry
    persistence = E.at(tables["P"], horizon)
    keep = S2.eligible_mask(persistence, FLOOR)
    entry["share_in_travel"] = float(S2.stratum_masks(persistence, boundary)["travel"][keep].mean())
    frames, parsed, expected = {"P": persistence}, {}, {}
    for label in EXT_ARMS:
        pairs = ext_tags_at(bird, label, horizon)
        expected[label] = {seed for _, seed in pairs}
        for tag, seed in pairs:
            if tag not in tables:
                raise SystemExit(f"FAIL: {bird} has no predictions for {tag}")
            frames[tag] = E.at(tables[tag], horizon)
            E.assert_aligned(frames[tag], persistence, f"{bird} {horizon} h {tag}")
            parsed[tag] = {"arm": label, "seed": seed}
    with ext_bound():
        E4.assert_no_collisions(frames, parsed, parse=E.parse_tag)
        summaries = E3.summarise_all(frames, persistence, boundary, parsed, parse=E.parse_tag)
        problems = [arm for arm in ("P", *EXT_ARMS) if arm not in summaries]
        for label in EXT_ARMS:
            seeds = {key for key in summaries.get(label, {}).get("by_seed", {}) if key >= 0}
            if label in summaries and seeds != expected[label]:
                problems.append(f"{label} has seeds {sorted(seeds)}, expected {sorted(expected[label])}")
        if problems:
            raise SystemExit(f"FAIL: {bird} {horizon} h: " + "; ".join(problems))
        arms = RS.arm_blocks(summaries)
        resampler = RS.Resampler(persistence, boundary)
        entry["arms"] = arms
        entry["readings"] = {
            "against_p": {label: E3.travel_skill(arms, label) for label in EXT_ARMS},
            "comparisons": {name(pair): RS.compare(*pair, arms, frames, persistence, boundary, resampler)
                            for pair in EXT_COMPARISONS},
        }
    entry["ablation"] = ext_ablation_cell(bird, horizon, tables, frames, persistence, keep, boundary)
    return entry


def ext_across_birds(report: dict) -> dict:
    """Per horizon: each label's median skill and each comparison's summary, over the four sets of the BbF report."""
    common = [bird for bird, entry in report["species"].items()
              if all(entry["horizons"][str(h)]["scored"] for h in HORIZONS)]
    out = {"common_set": common, "selection_fold": SELECTION_FOLD, "horizons": {}}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        sets = {"scored": scored, "common": common,
                "scored_without_fold6": [b for b in scored if FOLD_OF[b] != SELECTION_FOLD],
                "common_without_fold6": [b for b in common if FOLD_OF[b] != SELECTION_FOLD]}
        block: dict = {"scored": scored, "skill": {}, "comparisons": {}}
        for set_label, birds in sets.items():
            cells = [report["species"][bird]["horizons"][str(horizon)] for bird in birds]
            for label in EXT_ARMS:
                block["skill"].setdefault(label, {})[set_label] = V._median(
                    c["readings"]["against_p"][label] for c in cells)
            for pair in EXT_COMPARISONS:
                block["comparisons"].setdefault(name(pair), {})[set_label] = V._effect_summary(
                    cells, "comparisons", name(pair), ("skill_bootstrap", "skill_difference"))
        out["horizons"][str(horizon)] = block
    return out


def ext_ablation_summary(report: dict, common: list[str]) -> dict:
    """The 1 h ablation_summary's shape: arm -> h -> C -> stratum -> scored / common -> median skill and difference."""
    out: dict = {}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        for set_label, birds in (("scored", scored), ("common", common)):
            for context in LADDER:
                for stratum in E.STRATA:
                    cells = [report["species"][b]["horizons"][str(horizon)]["ablation"][str(context)][stratum]
                             for b in birds]
                    for arm in EXT_ABLATION_ARMS:
                        node = out.setdefault(arm, {}).setdefault(str(horizon), {}).setdefault(str(context), {})
                        node.setdefault(stratum, {})[set_label] = {
                            "skill": E._median(c[arm]["skill"] for c in cells),
                            "difference": E._median(c[arm]["difference"] for c in cells),
                        }
    return out


def _spread(values) -> dict:
    values = np.asarray([v for v in values if v is not None], dtype=float)
    finite = values[np.isfinite(values)]
    return {"n": int(finite.size), "max_abs_diff": float(np.abs(finite).max()) if finite.size else None,
            "median_diff": float(np.median(finite)) if finite.size else None}


def ext_tie_back(report: dict, sources: dict[str, dict]) -> dict:
    """
    Every read label's skill here minus its sealed report's: BbF at C = 12 and 96 against the BbF report, the 1 h arms
    at C = 12 against its main study and at C > 12 against its ablation's "all" stratum. The ablation-shaped block's
    1 h arms are compared cell by cell (every C, stratum and value) with the 1 h report's, and BbF's C = 12 and 96
    values there with the BbF report. Descriptive: no pass/fail.
    """
    out: dict = {"sources": {"birds_1h": str(E.REPORT_JSON.relative_to(REPO)),
                             "birds_1h_bbf": str(REPORT_JSON.relative_to(REPO))},
                 "attestations": {study: r.get("attestation") for study, r in sources.items()},
                 "cells": {}, "arms": {}, "ablation": {}, "scored_differs": []}
    ablation_diffs: dict[str, list] = {}
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            theirs = {study: r["species"][bird]["horizons"][horizon] for study, r in sources.items()}
            for study, other in theirs.items():
                if bool(cell["scored"]) != bool(other["scored"]):
                    out["scored_differs"].append(f"{bird} {horizon} h ({study})")
            if not (cell["scored"] and all(other["scored"] for other in theirs.values())):
                continue
            one_h, bbf = theirs["birds_1h"], theirs["birds_1h_bbf"]["readings"]["against_p"]
            diffs = {}
            for label in EXT_TIED_BACK:
                study, arm, context = EXT_ARMS[label]
                if study == "bbf":
                    reference = bbf[label]
                elif context == CORE:
                    reference = E3.travel_skill(one_h["arms"], arm)
                else:
                    reference = one_h["ablation"][str(context)]["all"][arm]["skill"]
                diffs[label] = cell["readings"]["against_p"][label] - reference
            out["cells"].setdefault(bird, {})[horizon] = diffs
            for context in LADDER:
                for stratum in E.STRATA:
                    ours, other = cell["ablation"][str(context)][stratum], one_h["ablation"][str(context)][stratum]
                    ablation_diffs.setdefault("P", []).append(ours["p_p90_m"] - other["p_p90_m"])
                    for arm in E.ABLATION_ARMS:
                        ablation_diffs.setdefault(arm, []).extend(ours[arm][k] - other[arm][k]
                                                                  for k in ("skill", "skill_C12", "difference"))
            ablation_diffs.setdefault("BbF", []).extend((
                cell["ablation"][str(C96)]["all"]["BbF"]["skill"] - bbf["BbF_C96"],
                cell["ablation"][str(EXT_CONTEXTS[0])]["all"]["BbF"]["skill_C12"] - bbf["BbF"]))
    for label in EXT_TIED_BACK:
        out["arms"][label] = _spread(c[label] for per_bird in out["cells"].values() for c in per_bird.values())
    out["ablation"] = {arm: _spread(values) for arm, values in ablation_diffs.items()}
    return out


def ext_build_report(tf32: dict | None = None) -> dict:
    sources = ext_source_reports()
    if not EXT_MANIFEST.exists():
        raise SystemExit(f"FAIL: {EXT_MANIFEST} does not exist; run --extension --predict first")
    manifest = json.loads(EXT_MANIFEST.read_text())
    b1h.check_boundaries()
    gap = completeness(read_records(), F.EXTENSION)
    if not gap["complete"]:
        raise SystemExit("FAIL: extension runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    report: dict = {
        "study": "birds_1h_bbf_extension",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name} sec {EXT_SECTION}",
        "attestation": manifest.get("attestation"),
        "bbf": "Bb with the repaired encoder (the repair probe's V4 switches)",
        "switches": F.SWITCHES,
        "contexts": list(EXT_CONTEXTS),
        "ladder": [CORE, *LADDER],
        "evaluation_floor": FLOOR,
        "min_scored_individuals": MIN_SCORED,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": E3.NOISE_SCALE,
        "lstm_comparator": LSTM_COMPARATOR,
        "tf32": {"predict": manifest.get("tf32"), "report": tf32 if tf32 is not None else env.tf32_flags()},
        "arms": {label: {"study": s, "arm": a, "context": c} for label, (s, a, c) in EXT_ARMS.items()},
        "comparisons": [name(pair) for pair in EXT_COMPARISONS],
        "completeness": gap,
        "scoring_rules": manifest["scoring_rules"],
        "species": {},
    }
    compared = 0
    for bird in BIRDS:
        tables = ext_load_tables(bird, manifest)
        compared += len(ext_read_files(bird))
        entry = {"fold": FOLD_OF[bird], "dropped": manifest["scoring_rules"][bird]["dropped"], "horizons": {}}
        for horizon in HORIZONS:
            entry["horizons"][str(horizon)] = ext_cell(bird, horizon, tables, manifest["eligibility"][bird])
        report["species"][bird] = entry
    report["persistence_identical_across_C"] = {"status": "PASS", "files_checked": compared}
    report["across_birds"] = ext_across_birds(report)
    report["ablation"] = ext_ablation_summary(report, report["across_birds"]["common_set"])
    report["tie_back"] = ext_tie_back(report, sources)
    return report


# Extension report
EXT_SHOWN = ("BbF", *(f"BbF_C{c}" for c in EXT_CONTEXTS), "BbF_C96", *(f"LSTMBb_C{c}" for c in EXT_CONTEXTS),
             *(f"XGBBb_C{c}" for c in EXT_CONTEXTS))


def ext_write_markdown(report: dict, path: Path) -> None:
    lines = ["# Repaired transformer (BbF) at C = 24, 48 and 72: sealed evaluation of the extension (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}`, committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"BbF is {report['bbf']}: {report['switches']}. Extension runs scored: {gap['n_clean']} of "
              f"{gap['n_planned']} clean. BbF at C = 12 and 96 and every comparator are read from sealed prediction "
              f"files (sha256 recorded at prediction). Travel p90 skill against P on the 1 h main eval's test_scored "
              f"windows, at the {report['evaluation_floor']}-window floor, a bird scored at h with ≥ "
              f"{report['min_scored_individuals']} test_scored individuals there; median of seeds; paired bootstrap "
              f"2,000 at seed 42. Every C is scored on the same windows. TF32 at prediction: {report['tf32']['predict']}.",
              "", f"LSTM comparator: {report['lstm_comparator']}.", ""]

    lines += ["## Per bird and horizon: skill against P", "",
              "| fold | bird | h | scored (at floor / with windows) | " + " | ".join(EXT_SHOWN) + " |",
              "|---:|---|---:|---|" + "---:|" * len(EXT_SHOWN)]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            n = f"{cell['at_floor']} / {cell['individuals']}"
            if not cell["scored"]:
                lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | unscored, {n} |" + " — |" * len(EXT_SHOWN))
                continue
            lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | {n} | "
                         + " | ".join(_s(cell["readings"]["against_p"][label]) for label in EXT_SHOWN) + " |")

    columns = report["comparisons"]
    lines += ["", "## Per bird and horizon: comparisons (skill difference, 95% interval)", "",
              "| fold | bird | h | " + " | ".join(columns) + " |", "|---:|---|---:|" + "---|" * len(columns)]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            if cell["scored"]:
                parts = [_with_interval(cell["readings"]["comparisons"][k], ("skill_bootstrap", "skill_difference"))
                         for k in columns]
                lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | " + " | ".join(parts) + " |")

    across = report["across_birds"]
    lines += ["", f"## Across birds: skill against P (median (n); common set = scored at all three horizons, "
                  f"{len(across['common_set'])} birds; without fold {across['selection_fold']}, where the repaired "
                  "encoder was chosen)", "",
              "| h | arm | scored | common | scored without fold 6 | common without fold 6 |",
              "|---:|---|---:|---:|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for label in EXT_SHOWN:
            lines.append(f"| {horizon} | {label} | "
                         + " | ".join(E._median_cell(block["skill"][label][s]) for s in SETS) + " |")
    lines += ["", "## Across birds: comparisons (median (n); above +0.05 / within / below −0.05 / not scored; "
                  "intervals excluding zero)", "",
              "| h | reading | scored | common | median without fold 6, scored / common |", "|---:|---|---|---|---|"]
    for horizon, block in across["horizons"].items():
        for key, entry in block["comparisons"].items():
            cells = [f"{E._median_cell(entry[s]['difference'])}; "
                     + " / ".join(str(entry[s]["bands"][b]) for b in BANDS)
                     + f"; {entry[s]['intervals_excluding_zero']} excl. 0" for s in ("scored", "common")]
            without = " / ".join(E._median_cell(entry[s]["difference"]) for s in SETS[2:])
            lines.append(f"| {horizon} | {key} | {cells[0]} | {cells[1]} | {without} |")

    lines += ["", "## History gain on the same windows (median over scored birds of skill at C, and of skill at C − "
                  "skill at C = 12; n birds; as the 1 h ablation)", "",
              "| arm | h | C | all | gappy (< 0.9) | full (≥ 0.9) |", "|---|---:|---:|---|---|---|"]
    for arm, per_horizon in report["ablation"].items():
        for horizon, per_context in per_horizon.items():
            for context, per_stratum in per_context.items():
                cells = [f"{_s(per_stratum[s]['scored']['skill']['median'])}, "
                         f"Δ {_s(per_stratum[s]['scored']['difference']['median'])} "
                         f"({per_stratum[s]['scored']['difference']['n']})" for s in E.STRATA]
                lines.append(f"| {arm} | {horizon} | {context} | " + " | ".join(cells) + " |")

    tie = report["tie_back"]
    lines += ["", "## Tie-back of the read arms to their sealed reports (here − source report; no pass/fail)", "",
              "| arm | cells | max \\|diff\\| | median diff |", "|---|---:|---:|---:|"]
    for group in (tie["arms"], {f"ablation block: {k}": v for k, v in tie["ablation"].items()}):
        for arm, entry in group.items():
            top = f"{entry['max_abs_diff']:.2e}" if entry["max_abs_diff"] is not None else "—"
            median = f"{entry['median_diff']:+.2e}" if entry["median_diff"] is not None else "—"
            lines.append(f"| {arm} | {entry['n']} | {top} | {median} |")
    if tie["scored_differs"]:
        lines += ["", "Scored in one report only: " + ", ".join(tie["scored_differs"]) + "."]
    path.write_text("\n".join(lines) + "\n")


# Extension diagnostic: run records only; no test partition is opened
def ext_as_trained_bullets(records: dict, gap: dict, exclusions: str) -> str:
    present = [records[r.run_id] for r in F.EXTENSION if r.run_id in records]
    devices = ", ".join(f"{d} {n}" for d, n in sorted(Counter(r.get("device") for r in present).items())) or "none"
    queues = ", ".join(f"{q} {n}" for q, n in sorted(Counter(r.get("pbs_queue") for r in present).items())) or "none"
    tf32_off = sum(not any((r.get("tf32") or {"missing": True}).values()) for r in present)
    switched = sum(r.get("switches") == F.SWITCHES for r in present)
    by_arm = "; ".join(f"{arm} {row['clean']} of {row['planned']}" for arm, row in gap["by_arm"].items())
    return "\n".join([
        f"- **Runs.** {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean ({by_arm}).",
        f"- **Training.** {F.REGISTERED_STEPS:,} updates applied, none withheld, no non-finite steps on every clean "
        f"run; TF32 off on {tf32_off}; devices {devices}; PBS queues {queues}.",
        f"- **Switches.** {F.SWITCHES} recorded on {switched} runs.",
        f"- **Training exclusions.** {exclusions}.",
        "- **Test-partition reads.** None for the extension before this section: its diagnostic reads run records only.",
        f"- **Deferred or missing.** {'none' if gap['complete'] else '; '.join(gap['problems'])}.",
    ])


def ext_diagnostic() -> None:
    """The 90 extension runs' completeness, switches and exclusions, and bullets for sec 10. Opens no test partition."""
    freeze_committed()
    with ext_bound():
        text = RS.committed_freeze()
    problems = ext_section_problems(text)
    for heading in EXT_SUBSECTIONS:
        mine = [p for p in problems if heading in p]
        print(f"sec 10, {heading:<23}: " + (f"{mine[0]} (not enforced here)" if mine else "filled"))
    payload = b1h.load_registered()
    records = read_records()
    gap = completeness(records, F.EXTENSION)
    try:
        result = training_exclusions(payload, records, F.EXTENSION)
        exclusions = (f"{result['status']} for all {result['runs']} runs "
                      f"({result['species_with_exclusions']} species with exclusions, as recorded)")
        exclusions_ok = True
    except SystemExit as reason:
        exclusions, exclusions_ok = str(reason), False
    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for key, row in gap["by_arm"].items():
        print(f"  {key:<12} {row['records']:>3} of {row['planned']:>3}, {row['clean']:>3} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print(f"exclusions       : {exclusions}")
    print("\n--- for the freeze's sec 10, \"Extension as trained\" ---\n")
    print(ext_as_trained_bullets(records, gap, exclusions))
    if not (gap["complete"] and exclusions_ok):
        raise SystemExit("FAIL: the diagnostic found a problem; \"Extension as trained\" must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score this study's runs on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score runs that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true",
                        help="runs, switches, BbFE's env inputs, exclusions, probe repeats and the sec 7 block; no test")
    parser.add_argument("--c96-fallback", action="store_true",
                        help="from 25-10-2026, if LSTMBbR96 is not complete: score C = 96 against the V100 ablation runs")
    parser.add_argument("--extension", action="store_true",
                        help=f"freeze sec {EXT_SECTION}: BbF at C = 24, 48, 72, to {EXT_SEALED.relative_to(REPO)}/")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --diagnostic, --predict, --report, or a combination")
    if args.extension and args.c96_fallback:
        raise SystemExit("FAIL: --c96-fallback belongs to the original 120 runs, not the extension")
    flags = fp32()  # before any model is built or loaded
    print(f"tf32             : {flags}", flush=True)

    if args.diagnostic:
        ext_diagnostic() if args.extension else diagnostic(fallback=args.c96_fallback)
    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        if args.extension:
            ext_run_predictions(device, overwrite=args.overwrite, tf32=flags)
        else:
            run_predictions(device, overwrite=args.overwrite, fallback=args.c96_fallback, tf32=flags)
    if args.report and args.extension:
        EXT_SEALED.mkdir(parents=True, exist_ok=True)
        report = ext_build_report(tf32=flags)
        EXT_REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        ext_write_markdown(report, EXT_REPORT_MD)
        print(f"\nwrote {EXT_REPORT_JSON}\nwrote {EXT_REPORT_MD}")
    elif args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report(fallback=args.c96_fallback, tf32=flags)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}")


if __name__ == "__main__":
    main()
