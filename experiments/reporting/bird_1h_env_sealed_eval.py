# GenAI reference: I specified this sealed env study evaluator; so reuse the 1 h scoring setup, check runs/files/freeze, 
# score env vs main arms, report effects/contrasts, and tie the reruns back to the sealed 1 h results; GenAI wrote it and 
# I checked it.
"""
Sealed evaluation of the env study (`preregistration/birds_1h_env_exploratory_freeze.md`): ERA5 covariates for Bb,
LSTM-Bb and XGB-Bb on the 30 rotation birds at C = 12 and direct horizons 1, 6 and 24 h (exploratory).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
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

import bird_1h_env_train as env  # noqa: E402
import bird_1h_sealed_eval as E  # noqa: E402

from animal_movement_fm.data import era5  # noqa: E402
from animal_movement_fm.data.multispecies import HorizonWindows  # noqa: E402

b1h, RS, S2, E3, E4 = E.b1h, E.RS, E.S2, E.E3, E.E4

FREEZE = REPO / "preregistration" / env.FREEZE
SEALED = REPO / "results" / "birds_1h_env_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_1h_env_results.json"
REPORT_MD = SEALED / "birds_1h_env_results.md"
MANIFEST = SEALED / "birds_1h_env_sealed_manifest.json"
ONE_H_REPORT = E.REPORT_JSON  # results/birds_1h_sealed/birds_1h_results.json, for the tie-back

BIRDS, FOLD_OF, HORIZONS, CORE = E.BIRDS, E.FOLD_OF, E.HORIZONS, E.CORE
FLOOR, MIN_SCORED = E.FLOOR, E.MIN_SCORED

ENV_ARMS = ("BbE", "LSTMBbE", "XGBBbE", "BbR", "LSTMBbR")
MAIN_ARMS = ("Bb", "LSTMBb", "XGBBb")  # the main study's runs at C = 12, reloaded
ARMS = ("P", *ENV_ARMS, *MAIN_ARMS)
# family -> (with env, without env): BbE and LSTMBbE against their gpuhopper reruns, XGBBbE against the main study's XGB-Bb
EFFECTS = {"Bb": ("BbE", "BbR"), "LSTMBb": ("LSTMBbE", "LSTMBbR"), "XGBBb": ("XGBBbE", "XGBBb")}
# hypothesis: the transformer gains more from env than either other family; (BbE - BbR) - (family's env effect)
CONTRASTS = {"Bb_vs_LSTMBb": ("Bb", "LSTMBb"), "Bb_vs_XGBBb": ("Bb", "XGBBb")}
# hardware check, a difference only: the gpuhopper rerun against the main study's gpuvolta run
HARDWARE = {"Bb": ("BbR", "Bb"), "LSTMBb": ("LSTMBbR", "LSTMBb")}
BANDS = ("above", "within", "below", "not scored")

if set(ENV_ARMS) != set(env.ARMS) or {env.ARMS[a]["base"] for a in env.ARMS} != set(MAIN_ARMS):
    raise RuntimeError("the evaluator's arms differ from the env driver's")
PLAN_INDEX = {run.run_id: i for i, run in enumerate(env.PLAN)}
MAIN_RUNS = [run for run in b1h.PLAN[: E.PARTS["main"][1] + 1] if run.arm in MAIN_ARMS and run.context == CORE]
if len(MAIN_RUNS) != 90 or any(run.species for run in MAIN_RUNS):
    raise RuntimeError(f"expected 90 pooled main-study runs (Bb, LSTM-Bb, XGB-Bb at C = 12), found {len(MAIN_RUNS)}")


@contextlib.contextmanager
def bound():
    """
    Point the rotation evaluator's freeze, manifest and floor, and Tier 3's floor, at this study while the block runs;
    restored afterwards, as bird_1h_sealed_eval.bound does for the 1 h study.
    """
    saved = RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR
    RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = FREEZE, MANIFEST, FLOOR, FLOOR
    try:
        yield
    finally:
        RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = saved


def fp32() -> dict:
    """Full FP32 for inference, as in training: TF32 off in cuDNN and matmuls, refused if either stays on."""
    env.disable_tf32()
    return env.require_fp32()


def one_h_report() -> dict:
    """The 1 h sealed report the main study's re-predicted arms are tied back to; refused if missing or unattested."""
    if not ONE_H_REPORT.exists():
        raise SystemExit(f"FAIL: {ONE_H_REPORT} does not exist; the tie-back needs the 1 h sealed report")
    report = json.loads(ONE_H_REPORT.read_text())
    if report.get("study") != "birds_1h" or not (report.get("attestation") or {}).get("freeze_commit"):
        raise SystemExit(f"FAIL: {ONE_H_REPORT.name} is not an attested 1 h sealed report")
    return report


# Seal
def freeze_committed() -> dict:
    with bound():
        return RS.freeze_committed()


def assert_sealed() -> dict:
    """The env freeze committed, clean, sec 6 and sec 7 filled, and the blob any manifest was attested at."""
    with bound():
        return RS.assert_sealed()


# Records
def read_env_records() -> dict[str, dict]:
    """Every env run record; XGBoost metadata files are skipped."""
    found = {}
    for path in sorted(env.RESULTS.glob("*.json")):
        if path.name.endswith("_xgb_meta.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("tier") != env.TAG or "run_id" not in record:
            print(f"  skipped {path.name}: not an env run record (tier {record.get('tier')!r})")
            continue
        found[record["run_id"]] = record
    return found


def env_run_problems(run, record: dict | None) -> list[str]:
    """What stops one planned env run counting as finished cleanly, the clip value included."""
    if record is None:
        return ["no record"]
    problems = [
        f"{key} {record.get(key)!r}, expected {want!r}"
        for key, want in (
            ("plan_index", PLAN_INDEX[run.run_id]),
            ("arm", run.arm),
            ("fold", run.fold),
            ("seed", run.seed),
            ("horizon", run.horizon or None),
            ("context", CORE),
            ("env", run.env),
            ("env_channels", env.ENV_CHANNELS if run.env else 0),
            ("env_clip_sd", era5.CLIP_SD if run.env else None),
            ("queue", env.queue_of(PLAN_INDEX[run.run_id])),
            ("opened_test_partition", False),
        )
        if record.get(key) != want
    ]
    if run.env:
        stats = record.get("env_stats") or {}
        if stats.get("channels") != list(era5.ENV_CHANNELS[: era5.N_STANDARDISED]):
            problems.append("env_stats missing or on other channels")
        if sorted(record.get("env_files") or {}) != sorted(record.get("pool_species") or []):
            problems.append("env_files do not cover the pool")
    if run.xgb:
        stem = env.RESULTS / f"{run.run_id}_xgb"
        if not all(Path(f"{stem}_{part}").exists() for part in ("east.ubj", "north.ubj", "meta.json")):
            problems.append("booster missing")
        if record.get("allow_missing") is not False:
            problems.append(f"allow_missing {record.get('allow_missing')}")
        return problems
    if any((record.get("tf32") or {"missing": True}).values()):
        problems.append(f"tf32 {record.get('tf32')}")
    steps = (record.get("config") or {}).get("steps")
    if record.get("applied_updates") != env.REGISTERED_STEPS or steps != env.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {steps}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (env.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def env_completeness(records: dict[str, dict]) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in env.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = env_run_problems(run, record)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (index {PLAN_INDEX[run.run_id]}): {'; '.join(issues)}")
    return {
        "n_planned": len(env.PLAN),
        "n_records": sum(row["records"] for row in by_arm.values()),
        "n_clean": sum(row["clean"] for row in by_arm.values()),
        "by_arm": by_arm,
        "problems": problems,
        "complete": not problems,
    }


def env_training_exclusions(payload: dict, records: dict[str, dict]) -> dict:
    """Each env record against its main-study base run's exclusions, pool and pinned window count (the 1 h check)."""
    as_base = {run.base.run_id: records[run.run_id] for run in env.PLAN if run.run_id in records}
    return E.assert_training_exclusions(payload, as_base, [run.base for run in env.PLAN if run.run_id in records])


def env_file_problems(records: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    """
    Every bird's env parquet against the sha256 the env records hold for it: one digest per bird across all records
    (each bird is in the pool of the nine folds that do not hold it out), equal to the file on disk now.
    """
    seen: dict[str, set[str]] = {}
    for run in env.PLAN:
        record = records.get(run.run_id)
        for stem, digest in ((record or {}).get("env_files") or {}).items():
            seen.setdefault(stem, set()).add(digest)
    problems = [f"{bird}: no env record holds its sha256" for bird in BIRDS if bird not in seen]
    problems += [f"{bird}: records hold {len(seen[bird])} different sha256" for bird in BIRDS
                 if len(seen.get(bird, ())) > 1]
    present = [bird for bird in BIRDS if env.env_path(bird).exists()]
    problems += [f"{bird}: {env.env_path(bird)} is missing" for bird in BIRDS if bird not in present]
    on_disk = env.env_files(present)
    problems += [f"{bird}: sha256 {on_disk[bird][:12]} on disk, {next(iter(seen[bird]))[:12]} in the records"
                 for bird in present if len(seen.get(bird, ())) == 1 and on_disk[bird] not in seen[bird]]
    return on_disk, problems


def checked_state(records: dict[str, dict], payload: dict) -> tuple[dict, dict, dict, dict]:
    """
    (main records, completeness, exclusions, env file sha256), refusing on any problem: all 150 env runs and
    the 90 main-study runs present and clean, every env run at era5.CLIP_SD, exclusions as pinned, env files as recorded.
    """
    env.check_main_study()
    main_records = E.record_map()
    gaps = {"env": env_completeness(records), "main": E.completeness(main_records, MAIN_RUNS)}
    problems = [p for gap in gaps.values() for p in gap["problems"]]
    if problems:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(problems))
    exclusions = {"env": env_training_exclusions(payload, records),
                  "main": E.assert_training_exclusions(payload, main_records, MAIN_RUNS)}
    digests, file_problems = env_file_problems(records)
    if file_problems:
        raise SystemExit("FAIL: env files differ from the records\n  " + "\n  ".join(file_problems))
    return main_records, gaps, exclusions, digests


# Prediction
def own_env_runs(bird: str) -> list:
    return [run for run in env.PLAN if run.fold == FOLD_OF[bird]]


def own_main_runs(bird: str) -> list:
    return [run for run in MAIN_RUNS if run.fold == FOLD_OF[bird]]


def load_env_models(runs: list, records: dict[str, dict], device) -> tuple[list, list]:
    """(neural, xgb) lists of (run, model, k_h) for env runs, each checkpoint checked against its record."""
    from animal_movement_fm.models.xgboost import WindowXGBoost

    neural, xgb = [], []
    for run in runs:
        record = records[run.run_id]
        k = np.asarray(record["horizon_k"], dtype=float)
        if run.xgb:
            model = WindowXGBoost.load(env.RESULTS / f"{run.run_id}_xgb")
            if model.allow_missing:
                raise SystemExit(f"FAIL: {run.run_id}'s booster allows missing values at C = {CORE}")
            xgb.append((run, model, k))
            continue
        loaded = E3.load_reported(env.RESULTS, run.run_id)
        if loaded is None:
            raise SystemExit(f"FAIL: {run.run_id} has no reported checkpoint")
        model, state = loaded
        want = (list(record["horizon_k"]), CORE, record["env_channels"], record["env_stats"], record["env_clip_sd"])
        got = (list(state["horizon_k"]), state["context"], state["env_channels"], state.get("env_stats"),
               state.get("env_clip_sd"))
        if got != want:
            raise SystemExit(f"FAIL: {run.run_id}'s checkpoint disagrees with its record on k_h, C, env channels, "
                             "env_stats or env_clip_sd")
        neural.append((run, model.to(device), k))
    return neural, xgb


def assert_same_windows(plain: HorizonWindows, covariates: HorizonWindows, bird: str) -> None:
    """The env windows are the 1 h main eval's: the same fixes, keys, anchors, targets and individuals, bit for bit."""
    fields = ("lon", "lat", "epoch_s", "keys", "individuals", "individual_index", "anchors", "targets")
    differ = [name for name in fields if not np.array_equal(getattr(plain, name), getattr(covariates, name))]
    if differ or (plain.context, plain.core, plain.horizons, plain.clip) != (
            covariates.context, covariates.core, covariates.horizons, covariates.clip):
        raise SystemExit(f"FAIL: {bird}'s env test windows differ from the 1 h main eval's in {differ or 'settings'}")


def stats_groups(runs: list, records: dict[str, dict]) -> dict[str, list]:
    """Env runs grouped by their record's env_stats, so each group's windows are standardised once."""
    groups: dict[str, list] = {}
    for run in runs:
        groups.setdefault(json.dumps(records[run.run_id]["env_stats"], sort_keys=True), []).append(run)
    return groups


def prediction_path(bird: str, tag: str) -> Path:
    return PREDICTIONS / f"{bird}__{tag}.parquet"


def write_scored(bird, windows, runs, predicted, share, written) -> None:
    for run in runs:
        horizons = (run.horizon,) if run.xgb else HORIZONS
        out = prediction_path(bird, run.run_id)
        E.frame_table(windows, predicted[run.run_id], CORE, share, horizons).to_parquet(out, index=False)
        written.append(out.name)


def predict_bird(bird: str, dropped, records: dict, main_records: dict, device, overwrite: bool) -> dict:
    """P, the main study's three pooled arms and the reruns on the plain windows; each env group on the env windows."""
    env_runs, main_runs = own_env_runs(bird), own_main_runs(bird)
    for run in env_runs:
        if bird in records[run.run_id]["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
    for run in main_runs:
        if bird in main_records[run.run_id]["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
    started = time.time()
    ids = E.scored_ids(bird, dropped)
    plain = E.test_windows(bird, ids, CORE)
    share = E.history_share(plain, CORE)
    persistence = E.frame_table(plain, None, CORE, share, persistence=True)
    print(f"\n{bird}: {plain.n_windows:,} test_scored windows, {len(plain.individuals)} individuals", flush=True)
    written = []
    out = prediction_path(bird, "P")
    if overwrite or not out.exists():
        persistence.to_parquet(out, index=False)
    written.append(out.name)

    def todo(runs):
        done = [run for run in runs if prediction_path(bird, run.run_id).exists() and not overwrite]
        written.extend(prediction_path(bird, run.run_id).name for run in done)
        return [run for run in runs if run not in done]

    plain_main = todo(main_runs)
    plain_env = todo([run for run in env_runs if not run.env])
    neural, xgb = E.load_models(plain_main, main_records, device, CORE)
    rerun_neural, _ = load_env_models(plain_env, records, device)
    if plain_main or plain_env:
        predicted = E.score_context(plain, neural + rerun_neural, xgb, device)
        write_scored(bird, plain, plain_main + plain_env, predicted, share, written)

    covariates = todo([run for run in env_runs if run.env])
    if covariates:
        env.check_env_files([bird])
        base = env.env_windows(bird, ids)
        assert_same_windows(plain, base, bird)
        for payload, runs in stats_groups(covariates, records).items():
            windows = replace(base, env_stats=env.stats_from_payload(json.loads(payload)))
            E.assert_same_persistence(persistence, E.frame_table(windows, None, CORE, share, persistence=True),
                                      f"{bird} env windows")
            predicted = E.score_context(windows, *load_env_models(runs, records, device), device)
            write_scored(bird, windows, runs, predicted, share, written)
    print(f"  {len(written) - 1} runs written or present ({time.time() - started:.0f}s)", flush=True)
    return {"files": written, "test_windows": plain.n_windows}


def run_predictions(device, overwrite: bool = False, tf32: dict | None = None) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    one_h_report()  # refuse now rather than after a GPU job
    payload = b1h.load_registered()
    records = read_env_records()
    main_records, gaps, exclusions, digests = checked_state(records, payload)
    drops, _, eligible = E.scoring_state(raise_on_mismatch=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest |= {"attestation": attestation, "scoring_rules": drops, "eligibility": eligible,
                 "completeness": gaps, "training_exclusions": exclusions, "env_files": digests,
                 "env_clip_sd": era5.CLIP_SD, "tf32": tf32 if tf32 is not None else env.tf32_flags(), "species": {}}
    for bird in BIRDS:
        manifest["species"][bird] = predict_bird(bird, drops[bird]["dropped"], records, main_records, device,
                                                 overwrite)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def load_tables(bird: str) -> dict[str, pd.DataFrame]:
    """The bird's saved predictions; anything from outside its own fold, or an unknown tag, is refused."""
    tables = {path.stem.split("__", 1)[1]: pd.read_parquet(path)
              for path in sorted(PREDICTIONS.glob(f"{bird}__*.parquet"))}
    allowed = {run.run_id for run in own_env_runs(bird) + own_main_runs(bird)} | {"P"}
    strays = sorted(set(tables) - allowed)
    if strays:
        raise SystemExit(f"FAIL: {bird} holds predictions from models outside its own fold: {strays}")
    if "P" not in tables:
        raise SystemExit(f"FAIL: {bird} has no P predictions; P defines the skill denominator")
    return tables


def tags_at(bird: str, arm: str, horizon: int) -> list[str]:
    """One bird's run ids for an arm at h: every seed of a neural arm, the one XGBoost run at h."""
    runs = own_env_runs(bird) if arm in ENV_ARMS else own_main_runs(bird)
    return [run.run_id for run in runs if run.arm == arm and (not run.xgb or run.horizon == horizon)]


def expected_seeds(bird: str, horizon: int) -> dict[str, set[int]]:
    out = {}
    for arm in ARMS[1:]:
        runs = own_env_runs(bird) if arm in ENV_ARMS else own_main_runs(bird)
        out[arm] = {run.seed for run in runs if run.arm == arm and (not run.xgb or run.horizon == horizon)}
    return out


# Statistic
def _interval(values: np.ndarray) -> dict:
    lo, hi = float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))
    return {"lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def contrast(first: str, second: str, arms: dict, frames: dict, resampler) -> dict:
    """
    (first's env effect) - (second's env effect) in travel p90 skill. The interval comes from the cell's resamples: the
    same draws, and the same cached p90s per arm, as the env-effect intervals beside it.
    """
    (a, b), (c, d) = EFFECTS[first], EFFECTS[second]
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
                                  "seed": S2.BOOTSTRAP_SEED, "difference": _interval((s[a] - s[b]) - (s[c] - s[d]))}
    return out


def cell_readings(persistence: pd.DataFrame, frames: dict, records: dict, boundary: float,
                  expected: dict[str, set[int]], label: str) -> tuple[dict, dict]:
    """Arm blocks and the readings for one bird at one horizon; each arm must hold exactly its planned seeds."""
    with bound():
        E4.assert_no_collisions(frames, records, parse=E.parse_tag)
        summaries = E3.summarise_all(frames, persistence, boundary, records, parse=E.parse_tag)
        problems = [arm for arm in ARMS if arm not in summaries]
        for arm in ARMS[1:]:
            seeds = {key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0}
            if arm in summaries and seeds != expected[arm]:
                problems.append(f"{arm} has seeds {sorted(seeds)}, expected {sorted(expected[arm])}")
        if problems:
            raise SystemExit(f"FAIL: {label}: " + "; ".join(problems))
        arms = RS.arm_blocks(summaries)
        resampler = RS.Resampler(persistence, boundary)
        readings = {
            "against_p": {arm: E3.travel_skill(arms, arm) for arm in ARMS[1:]},
            "env_effect": {family: RS.compare(a, b, arms, frames, persistence, boundary, resampler)
                           for family, (a, b) in EFFECTS.items()},
            "contrasts": {name: contrast(first, second, arms, frames, resampler)
                          for name, (first, second) in CONTRASTS.items()},
            "hardware": {family: RS.compare(a, b, arms, frames, persistence, boundary, resampler)
                         for family, (a, b) in HARDWARE.items()},
        }
    return arms, readings


def bird_cell(bird: str, horizon: int, tables: dict, records: dict, eligible: dict) -> dict:
    """One bird at one horizon: scored or not with n, share in travel, each arm's skill, and the readings."""
    counts = eligible[str(horizon)]
    boundary = b1h.BOUNDARY_M[(bird, horizon)]
    entry = {"boundary_m": boundary, **counts}
    if not counts["scored"]:
        return entry
    persistence = E.at(tables["P"], horizon)
    keep = S2.eligible_mask(persistence, FLOOR)
    entry["share_in_travel"] = float(S2.stratum_masks(persistence, boundary)["travel"][keep].mean())
    frames = {"P": persistence}
    for arm in ARMS[1:]:
        for tag in tags_at(bird, arm, horizon):
            if tag not in tables:
                raise SystemExit(f"FAIL: {bird} has no predictions for {tag}; run --predict")
            frames[tag] = E.at(tables[tag], horizon)
            E.assert_aligned(frames[tag], persistence, f"{bird} {horizon} h {tag}")
    entry["arms"], entry["readings"] = cell_readings(persistence, frames, records, boundary,
                                                     expected_seeds(bird, horizon), f"{bird} {horizon} h")
    return entry


def _median(values) -> dict:
    return E._median(values)


def _bands(values) -> dict:
    counts = Counter(RS.band(v) for v in values)
    return {key: counts.get(key, 0) for key in BANDS}


def _effect_summary(cells: list[dict], kind: str, name: str, interval_key: tuple[str, str]) -> dict:
    """Median, bands and intervals excluding zero over cells, for one env effect, contrast or hardware difference."""
    readings = [c["readings"][kind][name] for c in cells]
    point_key = "difference" if kind == "contrasts" else "skill_difference"
    points = [r.get(point_key) for r in readings]
    excluding = sum(bool(((r.get(interval_key[0]) or {}).get(interval_key[1]) or {}).get("excludes_zero"))
                    for r in readings)
    return {"difference": _median(points), "bands": _bands(points), "intervals_excluding_zero": excluding}


def across_birds(report: dict) -> dict:
    """Per horizon, over the scored birds and the common set (scored at all three): medians, bands and interval counts."""
    common = [bird for bird, entry in report["species"].items()
              if all(entry["horizons"][str(h)]["scored"] for h in HORIZONS)]
    out = {"common_set": common, "horizons": {}}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        block: dict = {"scored": scored, "skill": {}, "env_effect": {}, "contrasts": {}, "hardware": {}}
        for label, birds in (("scored", scored), ("common", common)):
            cells = [report["species"][bird]["horizons"][str(horizon)] for bird in birds]
            for arm in ARMS[1:]:
                block["skill"].setdefault(arm, {})[label] = _median(c["readings"]["against_p"][arm] for c in cells)
            for family in EFFECTS:
                block["env_effect"].setdefault(family, {})[label] = _effect_summary(
                    cells, "env_effect", family, ("skill_bootstrap", "skill_difference"))
            for name in CONTRASTS:
                block["contrasts"].setdefault(name, {})[label] = _effect_summary(
                    cells, "contrasts", name, ("skill_bootstrap", "difference"))
            for family in HARDWARE:
                summary = _effect_summary(cells, "hardware", family, ("skill_bootstrap", "skill_difference"))
                block["hardware"].setdefault(family, {})[label] = {"difference": summary["difference"]}
        out["horizons"][str(horizon)] = block
    return out


def tie_back(report: dict, one_h: dict) -> dict:
    """
    Per bird x horizon, the re-predicted main-study arms' travel p90 skill minus the 1 h sealed report's, with max |diff|
    and median per arm. Descriptive: no pass/fail. A cell scored in one report and not the other is listed.
    """
    out: dict = {"source": str(ONE_H_REPORT.relative_to(REPO)), "attestation": one_h.get("attestation"),
                 "cells": {}, "arms": {}, "scored_differs": []}
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            theirs = one_h["species"][bird]["horizons"][horizon]
            if bool(cell["scored"]) != bool(theirs["scored"]):
                out["scored_differs"].append(f"{bird} {horizon} h")
            if not (cell["scored"] and theirs["scored"]):
                continue
            out["cells"].setdefault(bird, {})[horizon] = {
                arm: cell["readings"]["against_p"][arm] - E3.travel_skill(theirs["arms"], arm) for arm in MAIN_ARMS}
    for arm in MAIN_ARMS:
        diffs = np.array([c[arm] for per_bird in out["cells"].values() for c in per_bird.values()], dtype=float)
        finite = diffs[np.isfinite(diffs)]
        out["arms"][arm] = {"n": int(finite.size),
                            "max_abs_diff": float(np.abs(finite).max()) if finite.size else None,
                            "median_diff": float(np.median(finite)) if finite.size else None}
    return out


# Report
def build_report(tf32: dict | None = None) -> dict:
    one_h = one_h_report()
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    b1h.check_boundaries()
    records = read_env_records()
    gap = env_completeness(records)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    main_records = E.record_map()
    merged = main_records | records
    report: dict = {
        "study": "birds_1h_env",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "evaluation_floor": FLOOR,
        "min_scored_individuals": MIN_SCORED,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": E3.NOISE_SCALE,
        "env_clip_sd": era5.CLIP_SD,
        "tf32": {"predict": manifest.get("tf32"), "report": tf32 if tf32 is not None else env.tf32_flags()},
        "effects": {family: f"{a} - {b}" for family, (a, b) in EFFECTS.items()},
        "contrasts": {name: f"({EFFECTS[f][0]} - {EFFECTS[f][1]}) - ({EFFECTS[s][0]} - {EFFECTS[s][1]})"
                      for name, (f, s) in CONTRASTS.items()},
        "hardware": {family: f"{a} - {b}" for family, (a, b) in HARDWARE.items()},
        "completeness": {"env": gap, "main": E.completeness(main_records, MAIN_RUNS)},
        "scoring_rules": manifest["scoring_rules"],
        "species": {},
    }
    for bird in BIRDS:
        tables = load_tables(bird)
        entry = {"fold": FOLD_OF[bird], "dropped": manifest["scoring_rules"][bird]["dropped"], "horizons": {}}
        for horizon in HORIZONS:
            entry["horizons"][str(horizon)] = bird_cell(bird, horizon, tables, merged, manifest["eligibility"][bird])
        report["species"][bird] = entry
    report["across_birds"] = across_birds(report)
    report["tie_back"] = tie_back(report, one_h)
    return report


def _s(value) -> str:
    return RS._s(value)


def _with_interval(reading: dict, key: tuple[str, str], point: str = "skill_difference") -> str:
    block = (reading.get(key[0]) or {}).get(key[1])
    return f"{_s(reading.get(point))} {RS._interval(block)}"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Env study: sealed evaluation (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    env_gap, main_gap = report["completeness"]["env"], report["completeness"]["main"]
    lines += [f"Env runs: {env_gap['n_records']} of {env_gap['n_planned']} recorded, {env_gap['n_clean']} clean; "
              f"main-study runs reloaded: {main_gap['n_clean']} of {main_gap['n_planned']} clean. Travel p90 skill "
              f"against P at C = {CORE} on the 1 h main eval's test_scored windows, at the {report['evaluation_floor']}-"
              f"window floor, a bird scored at h with ≥ {report['min_scored_individuals']} test_scored individuals "
              f"there; median of seeds; paired bootstrap 2,000 at seed 42. Env channels clipped at ±"
              f"{report['env_clip_sd']:g} SD. TF32 at prediction: {report['tf32']['predict']}.", ""]
    lines += ["Env effect: " + "; ".join(f"{k} {v}" for k, v in report["effects"].items()) + ". Contrasts: "
              + "; ".join(f"{k} {v}" for k, v in report["contrasts"].items()) + ". Hardware (difference only): "
              + "; ".join(f"{k} {v}" for k, v in report["hardware"].items()) + ".", ""]

    lines += ["## Per bird and horizon: skill against P", "",
              "| fold | bird | h | scored (at floor / with windows) | travel share | " + " | ".join(ARMS[1:]) + " |",
              "|---:|---|---:|---|---:|" + "---:|" * len(ARMS[1:])]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            n = f"{cell['at_floor']} / {cell['individuals']}"
            if not cell["scored"]:
                lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | unscored, {n} |"
                             + " — |" * (1 + len(ARMS[1:])))
                continue
            skills = [_s(cell["readings"]["against_p"][arm]) for arm in ARMS[1:]]
            lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | {n} | {cell['share_in_travel']:.3f} | "
                         + " | ".join(skills) + " |")

    lines += ["", "## Per bird and horizon: env effect and contrasts (skill, 95% interval)", "",
              "| fold | bird | h | " + " | ".join(f"{f} env" for f in EFFECTS) + " | "
              + " | ".join(CONTRASTS) + " | " + " | ".join(f"{f} hardware" for f in HARDWARE) + " |",
              "|---:|---|---:|" + "---|" * (len(EFFECTS) + len(CONTRASTS) + len(HARDWARE))]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            if not cell["scored"]:
                continue
            r = cell["readings"]
            parts = [_with_interval(r["env_effect"][f], ("skill_bootstrap", "skill_difference")) for f in EFFECTS]
            parts += [_with_interval(r["contrasts"][c], ("skill_bootstrap", "difference"), "difference")
                      for c in CONTRASTS]
            parts += [_with_interval(r["hardware"][f], ("skill_bootstrap", "skill_difference")) for f in HARDWARE]
            lines.append(f"| {entry['fold']} | {E.short(bird)} | {horizon} | " + " | ".join(parts) + " |")

    across = report["across_birds"]
    lines += ["", f"## Across birds (median (n); common set = scored at all three horizons, "
                  f"{len(across['common_set'])} birds)", "",
              "| h | arm | skill, scored | skill, common |", "|---:|---|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for arm in ARMS[1:]:
            entry = block["skill"][arm]
            lines.append(f"| {horizon} | {arm} | {E._median_cell(entry['scored'])} | {E._median_cell(entry['common'])} |")
    lines += ["", "## Env effect and contrasts across birds (median (n); above +0.05 / within / below −0.05 / not "
                  "scored; intervals excluding zero)", "",
              "| h | reading | scored | common |", "|---:|---|---|---|"]
    for horizon, block in across["horizons"].items():
        for kind in ("env_effect", "contrasts"):
            for name, entry in block[kind].items():
                cells = [f"{E._median_cell(entry[label]['difference'])}; "
                         + " / ".join(str(entry[label]["bands"][key]) for key in BANDS)
                         + f"; {entry[label]['intervals_excluding_zero']} excl. 0" for label in ("scored", "common")]
                lines.append(f"| {horizon} | {kind.replace('_', ' ')} {name} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Hardware check across birds (median difference (n); no pass/fail)", "",
              "| h | family | scored | common |", "|---:|---|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for family, entry in block["hardware"].items():
            lines.append(f"| {horizon} | {family} | {E._median_cell(entry['scored']['difference'])} | "
                         f"{E._median_cell(entry['common']['difference'])} |")
    tie = report["tie_back"]
    lines += ["", f"## Tie-back to the 1 h sealed report (re-predicted skill − `{tie['source']}`; no pass/fail)", "",
              "| arm | cells | max \\|diff\\| | median diff |", "|---|---:|---:|---:|"]
    for arm, entry in tie["arms"].items():
        top = f"{entry['max_abs_diff']:.2e}" if entry["max_abs_diff"] is not None else "—"
        median = f"{entry['median_diff']:+.2e}" if entry["median_diff"] is not None else "—"
        lines.append(f"| {arm} | {entry['n']} | {top} | {median} |")
    if tie["scored_differs"]:
        lines += ["", "Scored in one report only: " + ", ".join(tie["scored_differs"]) + "."]
    lines += ["", "| fold | bird | h | " + " | ".join(MAIN_ARMS) + " |", "|---:|---|---:|" + "---:|" * len(MAIN_ARMS)]
    for bird, per_horizon in tie["cells"].items():
        for horizon, diffs in per_horizon.items():
            lines.append(f"| {FOLD_OF[bird]} | {E.short(bird)} | {horizon} | "
                         + " | ".join(f"{diffs[arm]:+.2e}" for arm in MAIN_ARMS) + " |")
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: run records and registered files only; no test partition is opened
def section7_markdown(records: dict, gaps: dict, exclusions: str, digests: dict, file_problems: list[str]) -> str:
    env_gap, main_gap = gaps["env"], gaps["main"]
    neural = [records[r.run_id] for r in env.PLAN if not r.xgb and r.run_id in records]
    xgb = [records[r.run_id] for r in env.PLAN if r.xgb and r.run_id in records]
    devices = ", ".join(f"{d} {n}" for d, n in sorted(Counter(r.get("device") for r in neural).items())) or "none"
    queues = ", ".join(f"{q} {n}" for q, n in sorted(Counter(r.get("pbs_queue") for r in neural + xgb).items())) or "none"
    tf32_off = sum(not any((r.get("tf32") or {"missing": True}).values()) for r in neural)
    clipped = sum(r.get("env_clip_sd") == era5.CLIP_SD for r in records.values() if r.get("env"))
    by_arm = "; ".join(f"{arm} {row['clean']} of {row['planned']}" for arm, row in env_gap["by_arm"].items())
    lines = [
        "| | |", "|---|---|",
        f"| env runs | {env_gap['n_records']} of {env_gap['n_planned']} recorded, {env_gap['n_clean']} clean ({by_arm}) |",
        f"| neural ({len(neural)}) | {env.REGISTERED_STEPS:,} updates applied, none withheld, no non-finite steps on "
        f"every clean run; TF32 off on {tf32_off}; devices {devices} |",
        f"| XGBoost ({len(xgb)}) | fit once, boosters present on every clean run |",
        f"| PBS queues | {queues} |",
        f"| env clip | {clipped} env runs at ±{era5.CLIP_SD:g} SD |",
        f"| env files | {len(digests)} birds; sha256 "
        + ("as recorded" if not file_problems else "differ: " + "; ".join(file_problems)) + " |",
        f"| main-study runs reloaded | {main_gap['n_clean']} of {main_gap['n_planned']} clean (Bb, LSTM-Bb, XGB-Bb at "
        f"C = {CORE}) |",
        f"| training exclusions | {exclusions} |",
        "| test-partition reads | none: the diagnostic reads run records and env files only |",
        f"| deferred or missing | {'none' if env_gap['complete'] and main_gap['complete'] else '; '.join(env_gap['problems'] + main_gap['problems'])} |",
    ]
    return "\n".join(lines)


def diagnostic() -> None:
    """Completeness, the clip value, exclusions and env files, and the sec 7 block. Opens no test partition."""
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
    records = read_env_records()
    main_records = E.record_map()
    gaps = {"env": env_completeness(records), "main": E.completeness(main_records, MAIN_RUNS)}
    try:
        exclusions = json.dumps({"env": env_training_exclusions(payload, records),
                                 "main": E.assert_training_exclusions(payload, main_records, MAIN_RUNS)})
        exclusions_ok = True
    except SystemExit as reason:
        exclusions, exclusions_ok = str(reason), False
    digests, file_problems = env_file_problems(records)

    for kind, gap in gaps.items():
        print(f"completeness     : {kind} {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
        for key, row in gap["by_arm"].items():
            print(f"  {key:<11} {row['records']:>3} of {row['planned']:>3}, {row['clean']:>3} clean")
        for problem in gap["problems"][:20]:
            print(f"  {problem}")
    print(f"exclusions       : {exclusions}")
    print(f"env files        : {len(digests)} present; " + ("as recorded" if not file_problems
                                                             else "\n  " + "\n  ".join(file_problems)))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(records, gaps, exclusions, digests, file_problems))
    if not (gaps["env"]["complete"] and gaps["main"]["complete"] and exclusions_ok and not file_problems):
        raise SystemExit("FAIL: the diagnostic found a problem; sec 7 must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score every arm on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score runs that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true",
                        help="runs, the clip, exclusions, env files and the sec 7 block; opens no test partition")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --diagnostic, --predict, --report, or a combination")
    flags = fp32()  # before any model is built or loaded
    print(f"tf32             : {flags}", flush=True)

    if args.diagnostic:
        diagnostic()
    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        run_predictions(device, overwrite=args.overwrite, tf32=flags)
    if args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report(tf32=flags)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}")


if __name__ == "__main__":
    main()
