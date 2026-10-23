# GenAI reference: I outlined this 6 h baseline evaluator for our 30 bird rotation; that being reuse the sealed fold 0 results, score 
# V/ARIMA/AVG3/AVG10 with the same rules/bootstrap, read P and specialist rows from the sealed outputs etc. GenAI wrote this, and I 
# checked its behaviour. 
"""
Training free baselines for the rotating held-out birds (`preregistration/birds_rotation_baselines_exploratory_freeze.md`).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import heapq
import io
import json
import math
import multiprocessing
import platform
import socket
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_sealed_eval as BR  # noqa: E402
import metric_tables as MT  # noqa: E402

from animal_movement_fm.data.benchmark import load_split_frame  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.data.split import load_split  # noqa: E402
from animal_movement_fm.eval.harness import evaluate  # noqa: E402

BE, RT, S2, E3, SPW = BR.BE, BR.RT, BR.S2, BR.E3, BR.SPW

FREEZE = REPO / "preregistration" / "birds_rotation_baselines_exploratory_freeze.md"
SEALED = REPO / "results" / "birds_rotation_baselines_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_rotation_baselines_results.json"
REPORT_MD = SEALED / "birds_rotation_baselines_results.md"
MANIFEST = SEALED / "birds_rotation_baselines_sealed_manifest.json"

FLOOR = E3.FLOOR
NOISE_SCALE = E3.NOISE_SCALE

FOLD0, ROTATION, BIRDS, FOLD_OF = BR.FOLD0, BR.ROTATION, BR.BIRDS, BR.FOLD_OF
BASELINES = ("V", "ARIMA", "AVG3", "AVG10")
ARMS = ("P", *BASELINES)
SPECIALISTS = ("LSTMS", "XGBS")  # read from the rotation report; no new comparison
if set(ARMS) != set(BE.TRAINING_FREE):
    raise RuntimeError("the baselines no longer match bird_sealed_eval.TRAINING_FREE")

# bird_sealed_eval job log (27-09-2026): goose 151 s over 1,878 windows, dove 469 s over 6,028, gpuhopper CPU
BIRD_STUDY_ARIMA_S_PER_WINDOW = (151 + 469) / (1_878 + 6_028)
WALLTIME_MARGIN = 1.5


# Seal
def freeze_committed() -> dict:
    """The freeze is tracked and unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    previous = S2.FREEZE
    S2.FREEZE = FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def assert_sealed() -> dict:
    """Committed and clean. A manifest attested at another blob of the freeze is refused. Run by --predict only."""
    attestation = freeze_committed()
    if MANIFEST.exists():
        earlier = json.loads(MANIFEST.read_text()).get("attestation") or {}
        if earlier.get("freeze_blob") and earlier["freeze_blob"] != attestation["freeze_blob"]:
            raise SystemExit(
                f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but {FREEZE.name} is now "
                f"blob {attestation['freeze_blob'][:12]}. Rebuild the report with --report instead."
            )
    return attestation


# Inputs from the rotation's seal
def rotation_manifest() -> dict:
    if not BR.MANIFEST.exists():
        raise SystemExit(f"FAIL: {BR.MANIFEST} does not exist; the scoring rules are read from it")
    manifest = json.loads(BR.MANIFEST.read_text())
    missing = sorted(set(BIRDS) - set(manifest.get("scoring_rules") or {}))
    if missing:
        raise SystemExit(f"FAIL: {BR.MANIFEST.name} has no scoring rules for {missing}")
    return manifest


def rotation_report() -> dict:
    if not BR.REPORT_JSON.exists():
        raise SystemExit(f"FAIL: {BR.REPORT_JSON} does not exist; P and the specialists are checked against it")
    return json.loads(BR.REPORT_JSON.read_text())


def boundaries() -> dict[str, float]:
    return RT.check_boundaries() | BE.check_boundaries()


def rotation_p_path(bird: str) -> Path:
    return BR.prediction_path(bird, "P")


def fold0_paths(bird: str) -> dict[str, Path]:
    """The sealed files fold 0's baselines are read from: the bird study's, or the gull's carried Tier 2 and 3 ones."""
    if bird == BE.GULL:
        return {"P": S2.PREDICTIONS / f"{bird}__P.parquet", "V": S2.PREDICTIONS / f"{bird}__V.parquet",
                **{tag: E3.PREDICTIONS / f"{bird}__{tag}.parquet" for tag in ("ARIMA", "AVG3", "AVG10")}}
    return {tag: BE.PREDICTIONS / f"{bird}__{tag}.parquet" for tag in ARMS}


def prediction_path(bird: str, tag: str) -> Path:
    return PREDICTIONS / f"{bird}__{tag}.parquet"


def n_rows(path: Path) -> int | None:
    """Row count from parquet metadata; no column is read."""
    return pq.ParquetFile(path).metadata.num_rows if path.exists() else None


# Scheduling
def schedule(costs: dict[str, float], workers: int) -> tuple[list[str], float]:
    """Largest first (the order jobs are submitted), and the makespan of greedy assignment to `workers` processes."""
    order = sorted(costs, key=lambda bird: (-costs[bird], bird))
    loads = [0.0] * max(1, workers)
    for bird in order:
        heapq.heapreplace(loads, loads[0] + costs[bird])
    return order, max(loads)


# Prediction
def baseline_models() -> list[tuple[str, object]]:
    """V as Tier 2 built it, ARIMA and AVG3/10 as Tier 3 did: bird_sealed_eval.training_free_models without P."""
    return [(tag, model) for tag, model in BE.training_free_models() if tag in BASELINES]


def persistence_model():
    return next(model for tag, model in BE.training_free_models() if tag == "P")


def p_identity(recomputed: pd.DataFrame, sealed: pd.DataFrame) -> dict:
    """
    Does P recomputed now equal the rotation's sealed P file, row for row and column for column?

    The recomputed frame goes through parquet and E3's row order first, as the sealed one did.
    """
    buffer = io.BytesIO()
    recomputed.to_parquet(buffer, index=False)
    buffer.seek(0)
    recomputed = pd.read_parquet(buffer).sort_values(["individual_id", "timestamp"]).reset_index(drop=True)
    out = {"n_windows": int(len(sealed)), "n_recomputed": int(len(recomputed))}
    try:
        pd.testing.assert_frame_equal(recomputed, sealed, check_exact=True, check_like=False)
        out["identical"] = True
    except AssertionError as reason:
        out["identical"] = False
        out["difference"] = str(reason).splitlines()[0][:300]
    return out


def predict_bird(bird: str, overwrite: bool) -> dict:
    """One worker: one bird's test partition, P checked against the sealed file, then the four baselines."""
    started = time.time()
    frame = BR.test_frame(bird)  # refuses anything outside the 27
    sealed = E3.load_frame(rotation_p_path(bird))
    identity = p_identity(evaluate(persistence_model(), frame, context_length=CONTEXT_LENGTH, show_progress=False),
                          sealed)
    if not identity["identical"]:
        raise SystemExit(f"FAIL: {bird}: P recomputed differs from the rotation's sealed P: {identity.get('difference')}")
    out = {"bird": bird, "n_fixes": int(len(frame)), "p_identity": identity, "written": [], "seconds": {}}
    for tag, model in baseline_models():
        path = prediction_path(bird, tag)
        if path.exists() and not overwrite:
            out["written"].append(f"{path.name} (exists, skipped)")
        else:
            tick = time.time()
            evaluate(model, frame, context_length=CONTEXT_LENGTH, show_progress=False).to_parquet(path, index=False)
            out["seconds"][tag] = round(time.time() - tick, 1)
            out["written"].append(path.name)
        E3.assert_aligned(E3.load_frame(path), sealed, f"{bird} {tag}")
    out["seconds"]["total"] = round(time.time() - started, 1)
    return out


def run_predictions(workers: int, overwrite: bool = False) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    rules = rotation_manifest()["scoring_rules"]
    boundaries()
    missing = [str(rotation_p_path(bird)) for bird in ROTATION if not rotation_p_path(bird).exists()]
    if missing:
        raise SystemExit("FAIL: the rotation's sealed P files are missing:\n  " + "\n  ".join(missing))
    order, _ = schedule({bird: float(n_rows(rotation_p_path(bird))) for bird in ROTATION}, workers)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    print(f"workers          : {workers}; birds largest first: {', '.join(order[:3])}, ...", flush=True)

    birds: dict[str, dict] = {}
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(predict_bird, bird, overwrite): bird for bird in order}
        for future in as_completed(futures):
            result = future.result()  # a worker's SystemExit stops the run here
            birds[result["bird"]] = result
            print(f"  {result['bird']:<26} {result['seconds']}  P identical", flush=True)
    manifest = {
        "attestation": attestation,
        "rotation_manifest": str(BR.MANIFEST.relative_to(REPO)),
        "rotation_attestation": json.loads(BR.MANIFEST.read_text()).get("attestation"),
        "dropped": {bird: rules[bird]["dropped"] for bird in BIRDS},
        "workers": workers,
        "host": socket.gethostname(),
        "statsmodels": version("statsmodels"),
        "fold0": "read from sealed files, never re-predicted",
        "species": {bird: birds[bird] for bird in ROTATION},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def parse_tag(tag: str, records: dict) -> dict | None:
    if tag in ARMS:
        return {"arm": tag, "seed": None, "k": None, "draw": None}
    print(f"  ignored {tag}: not a training-free baseline")
    return None


def rotation_frames(bird: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """The rotation's sealed P and this study's four baselines, all aligned to P."""
    persistence = E3.load_frame(rotation_p_path(bird))
    frames = {"P": persistence}
    for tag in BASELINES:
        path = prediction_path(bird, tag)
        if not path.exists():
            raise SystemExit(f"FAIL: {path} is missing; run --predict first")
        frames[tag] = E3.load_frame(path)
        E3.assert_aligned(frames[tag], persistence, f"{bird} {tag}")
    return persistence, frames


def fold0_frames(bird: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """The bird study's sealed P and baselines (the gull's carried ones), restricted to this study's arms."""
    persistence, frames, records = BE.species_frames(bird)
    with contextlib.redirect_stdout(io.StringIO()):
        kept = {tag: frame for tag, frame in frames.items()
                if (parsed := BE.parse_tag(tag, records)) is not None and parsed["arm"] in ARMS}
    return persistence, kept, records


# Checks against the sealed reports
def same_headline(mine: dict, theirs: dict) -> list[str]:
    """Every stored statistic of one arm, per stratum, equal to the sealed report's."""
    problems = []
    for stratum in (*E3.STRATA, "all"):
        for name in E3.STATISTICS:
            a = (mine.get(stratum) or {}).get(name)
            b = (theirs.get(stratum) or {}).get(name)
            if not (MT.finite(a) and MT.finite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)):
                problems.append(f"{stratum} {name}: {a} here, {b} sealed")
    return problems


def check_against(label: str, arms: dict, sealed_arms: dict, names) -> None:
    problems = [f"{arm} {p}" for arm in names
                for p in same_headline(arms[arm]["headline"], (sealed_arms.get(arm) or {}).get("headline") or {})]
    if problems:
        raise SystemExit(f"FAIL: {label}\n  " + "\n  ".join(problems[:10]))


# Report
def score_bird(bird: str, rules: dict, boundary: float) -> tuple[dict, dict, pd.DataFrame, dict]:
    """Arm blocks for P and the four baselines on the rotation's scored windows; the frames they came from."""
    dropped = rules[bird]["dropped"]
    if bird in FOLD0:
        if dropped:
            raise SystemExit(f"FAIL: the scoring rules drop fold-0 birds {dropped}; the rotation freeze says none")
        persistence, frames, records = fold0_frames(bird)
        parse = BE.parse_tag
    else:
        persistence, frames = rotation_frames(bird)
        records, parse = {}, parse_tag
    persistence, frames = BR.drop_scored(persistence, frames, dropped)
    with contextlib.redirect_stdout(io.StringIO()):
        summaries = E3.summarise_all(frames, persistence, boundary, records, parse=parse)
    missing = [arm for arm in ARMS if arm not in summaries]
    if missing:
        raise SystemExit(f"FAIL: {bird}: no summary for {missing}")
    return BR.arm_blocks(summaries), frames, persistence, {"dropped": dropped}


def specialist_rows(entry: dict) -> dict:
    """LSTM-S and XGB-S as the rotation report stored them, with the intervals it stored for them."""
    intervals, _, _ = MT.harvest(entry, set(SPECIALISTS))
    return {arm: {**{key: entry["arms"][arm][key] for key in ("n_seeds", "headline", "skill")},
                  "intervals": {series: (intervals.get(arm) or {}).get(series) for series in MT.INTERVALS}}
            for arm in SPECIALISTS}


def arima_fallback(frames: dict, persistence: pd.DataFrame) -> dict | None:
    arima = next((frame for tag, frame in frames.items() if tag.split(":")[-1] == "ARIMA"), None)
    return E3.arima_fallback_rate({"ARIMA": arima, "tier2:P": persistence}) if arima is not None else None


def band(skill) -> str:
    if not MT.finite(skill):
        return "not scored"
    return "within" if not E3.resolved(skill) else ("above" if skill > 0 else "below")


def across_birds(report: dict) -> dict:
    sets = {"30 birds": list(BIRDS), "27 outside fold 0": list(ROTATION)}
    out: dict = {}
    for name, members in sets.items():
        block = {}
        for arm in (*BASELINES, *SPECIALISTS):
            rows = [MT.metrics_of(report["species"][bird]["arms"].get(arm)
                                  or report["species"][bird]["specialists"][arm], "arms") for bird in members]
            block[arm] = {label: MT.median_block([row[label]["skill"] for row in rows]) for label, _, _ in MT.METRICS}
            counts = Counter(band(row["travel_p90"]["skill"]) for row in rows)
            block[arm]["travel_p90_bands"] = {key: counts.get(key, 0) for key in ("above", "within", "below")}
        out[name] = {"n_birds": len(members), "arms": block}
    return out


def build_report() -> dict:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    rules = rotation_manifest()["scoring_rules"]
    sealed_rotation = rotation_report()
    sealed_birds = json.loads(BE.REPORT_JSON.read_text())
    limits = boundaries()
    report: dict = {
        "study": "birds_rotation_baselines",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "rotation_report": {"path": str(BR.REPORT_JSON.relative_to(REPO)),
                            "freeze_commit": (sealed_rotation.get("attestation") or {}).get("freeze_commit")},
        "evaluation_floor": FLOOR,
        "noise_scale": NOISE_SCALE,
        "n_bootstrap": S2.N_BOOTSTRAP,
        "bootstrap_seed": S2.BOOTSTRAP_SEED,
        "species": {},
    }
    for bird in BIRDS:
        entry = bird_entry(bird, rules, limits[bird], sealed_rotation["species"][bird],
                           sealed_birds["species"].get(bird))
        report["species"][bird] = entry
        print(f"  {bird:<26} " + "  ".join(f"{arm} {MT._s(E3.travel_skill(entry['arms'], arm))}" for arm in BASELINES),
              flush=True)
    report["across_birds"] = across_birds(report)
    return report


def bird_entry(bird: str, rules: dict, boundary: float, sealed_entry: dict, sealed_bird: dict | None) -> dict:
    """One bird's report entry: the statistics, the checks against the sealed reports, and each baseline against P."""
    arms, frames, persistence, extra = score_bird(bird, rules, boundary)
    check_against(f"{bird}: P differs from the rotation report", arms, sealed_entry["arms"], ["P"])
    if bird in FOLD0:
        check_against(f"{bird}: fold-0 baselines differ from the bird study's report", arms, sealed_bird["arms"], ARMS)
    resampler = BR.Resampler(persistence, boundary)
    against_p = {arm: BR.compare(arm, "P", arms, frames, persistence, boundary, resampler) for arm in BASELINES}
    intervals, conflicts, _ = MT.harvest(against_p, set(ARMS))
    if conflicts:
        raise SystemExit(f"FAIL: {bird}: interval copies disagree: {conflicts[:2]}")
    for arm, block in arms.items():
        block["intervals"] = {series: (intervals.get(arm) or {}).get(series) for series in MT.INTERVALS}
    return {
        "fold": FOLD_OF[bird],
        "regime_boundary_m": boundary,
        "n_test_individuals_scored": int(persistence["individual_id"].nunique()),
        "n_test_windows": int(len(persistence)),
        **extra,
        "arms": arms,
        "against_p": against_p,
        "specialists": specialist_rows(sealed_entry),
        "arima_fallback_rate": arima_fallback(frames, persistence),
        "checks": {"p_matches_rotation_report": True, "fold0_matches_bird_study": True if bird in FOLD0 else None},
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Rotating held-out birds: training-free baselines (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    lines += [f"Scored on the rotation's windows, rules and floor (F = {report['evaluation_floor']}); P read from the "
              "rotation's sealed predictions. Each cell is metres (skill against P); means are individual-weighted. "
              f"Intervals: paired bootstrap, {report['n_bootstrap']:,} at seed {report['bootstrap_seed']}. LSTM-S and "
              "XGB-S are read from the rotation report; they are not compared here.", ""]

    for name, block in report["across_birds"].items():
        lines += [f"## Median over {name}", "",
                  "| arm | " + " | ".join(MT.TITLES[l] for l, _, _ in MT.METRICS)
                  + " | travel p90 above +0.05 / within / below |",
                  "|---|" + "---:|" * len(MT.METRICS) + "---|"]
        for arm, values in block["arms"].items():
            counts = values["travel_p90_bands"]
            lines.append(f"| {arm} | " + " | ".join(MT._s(values[l]["median"]) for l, _, _ in MT.METRICS)
                         + f" | {counts['above']} / {counts['within']} / {counts['below']} |")
        lines.append("")

    lines += ["## Per bird", ""]
    for bird, entry in report["species"].items():
        rate = entry.get("arima_fallback_rate") or {}
        lines += [f"### {MT.short(bird)}", "",
                  f"fold {entry['fold']}; boundary {entry['regime_boundary_m']:,.1f} m; "
                  f"{entry['n_test_individuals_scored']} scored test birds; {entry['n_test_windows']:,} windows"
                  + (f"; ARIMA falls back on either axis in {rate['either_axis']:.1%} of windows" if rate else "") + ".",
                  "",
                  "| arm | travel p90 | travel mean | travel p50 | all mean | all p50 | travel p90 95% (m) "
                  "| travel p90 95% (skill) | all mean 95% (m) |",
                  "|---|---:|---:|---:|---:|---:|---|---|---|"]
        rows = [(arm, block) for arm, block in entry["arms"].items()]
        rows += [(f"{arm} (rotation)", block) for arm, block in entry["specialists"].items()]
        for arm, block in rows:
            metrics = MT.metrics_of(block, "arms")
            bands = block["intervals"]
            lines.append(f"| {arm} | " + " | ".join(MT._cell(arm, metrics[l]) for l, _, _ in MT.METRICS)
                         + f" | {MT._band(bands['travel_p90_m'], 'm')} | {MT._band(bands['travel_p90_skill'], 'skill')} "
                         f"| {MT._band(bands['all_mean_m'], 'm')} |")
        lines.append("")
    lines += ["Fold 0 is read from the bird study's sealed files (the gull's from Tiers 2 and 3) and checked equal to "
              "the bird study's report; every bird's P is checked equal to the rotation report's.", ""]
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: sealed-file metadata, registered files and training windows only
def arima_probe(bird: str, windows: int) -> dict:
    """Seconds per window of ARIMA on this machine, over training individuals of `bird` (never test)."""
    stem, split_stem, _ = RT.rotation_attached_entry(bird)
    train = list(load_split(SPW.SPLITS / f"{split_stem}_split_seed42.json").train)
    model = next(model for tag, model in baseline_models() if tag == "ARIMA")
    scored, seconds, used = 0, 0.0, 0
    for individual in train:
        frame = load_split_frame(SPW.PROCESSED / f"{stem}_segments.parquet", [individual])
        frame = frame[["individual_id", "bin_index", "timestamp", "lon", "lat"]].head(400).reset_index(drop=True)
        tick = time.time()
        results = evaluate(model, frame, context_length=CONTEXT_LENGTH, show_progress=False)
        seconds += time.time() - tick
        scored += len(results)
        used += 1
        if scored >= windows:
            break
    return {"bird": bird, "partition": "train", "individuals": used, "windows": scored,
            "seconds": round(seconds, 1), "s_per_window": seconds / scored if scored else float("nan"),
            "host": socket.gethostname(), "processor": platform.processor() or platform.machine()}


def diagnostic(workers: int, probe_bird: str, probe_windows: int) -> None:
    """Files, counts, boundaries and the ARIMA timing probe; prints the block for the freeze. Opens no test target."""
    problems: list[str] = []
    print(f"freeze           : {FREEZE.relative_to(REPO)} {'exists' if FREEZE.exists() else 'does not exist yet'}; "
          "its commit is checked by --predict, not here")
    rules = rotation_manifest()["scoring_rules"]
    sealed_rotation = rotation_report()
    limits = boundaries()
    print(f"boundaries       : {len(limits)} pinned values checked")

    rows = {}
    for bird in ROTATION:
        p_rows = n_rows(rotation_p_path(bird))
        have = {tag: n_rows(prediction_path(bird, tag)) for tag in BASELINES}
        rows[bird] = p_rows
        if p_rows is None:
            problems.append(f"{bird}: rotation P missing ({rotation_p_path(bird).relative_to(REPO)})")
        for tag, n in have.items():
            if n is not None and n != p_rows:
                problems.append(f"{bird}: {tag} has {n} rows, P has {p_rows}")
    fold0 = {}
    for bird in FOLD0:
        counts = {tag: n_rows(path) for tag, path in fold0_paths(bird).items()}
        fold0[bird] = counts
        for tag, n in counts.items():
            if n is None:
                problems.append(f"{bird}: {tag} missing ({fold0_paths(bird)[tag].relative_to(REPO)})")
        if len({n for n in counts.values() if n is not None}) > 1:
            problems.append(f"{bird}: fold-0 row counts differ {counts}")

    print("\nrotation birds (rows from parquet metadata; scored windows from the rotation report)")
    print(f"  {'bird':<26} {'fold':>4} {'P rows':>8} {'scored':>8} {'dropped':>7} {'existing baselines':>18}")
    for bird in ROTATION:
        existing = sum(prediction_path(bird, tag).exists() for tag in BASELINES)
        print(f"  {bird:<26} {FOLD_OF[bird]:>4} {rows[bird] if rows[bird] is not None else '—':>8} "
              f"{sealed_rotation['species'][bird]['n_test_windows']:>8,} {len(rules[bird]['dropped']):>7} "
              f"{existing:>16}/4")
    print("\nfold 0 (read, never recomputed)")
    for bird, counts in fold0.items():
        print(f"  {bird:<26} " + "  ".join(f"{tag} {n if n is not None else 'missing'}" for tag, n in counts.items()))

    probe = arima_probe(probe_bird, probe_windows)
    print(f"\nARIMA probe      : {probe['windows']} training windows of {probe_bird} in {probe['seconds']} s "
          f"= {probe['s_per_window']:.4f} s/window on {probe['host']} ({probe['processor']})")
    print(f"bird study rate  : {BIRD_STUDY_ARIMA_S_PER_WINDOW:.4f} s/window (gpuhopper CPU, job log 27-09-2026)")
    known = {bird: n for bird, n in rows.items() if n is not None}
    if not known:  # no P on this machine: project from the scored windows the rotation report stored
        known = {bird: sealed_rotation["species"][bird]["n_test_windows"] for bird in ROTATION}
        print("projection       : from the rotation report's scored windows (no P files here; dropped birds excluded)")
    projections = {}
    for label, rate in (("probe", probe["s_per_window"]), ("bird study", BIRD_STUDY_ARIMA_S_PER_WINDOW)):
        order, makespan = schedule({bird: n * rate for bird, n in known.items()}, workers)
        total = sum(known.values()) * rate
        projections[label] = {"cpu_hours": total / 3600, "makespan_hours": makespan / 3600,
                              "largest": order[0], "largest_hours": known[order[0]] * rate / 3600}
        print(f"  at {label:<11} rate: {total / 3600:5.1f} CPU-h over {sum(known.values()):,} windows; "
              f"{order[0]} {known[order[0]] * rate / 3600:4.1f} h; makespan on {workers} workers "
              f"{makespan / 3600:4.1f} h")
    worst = max(p["makespan_hours"] for p in projections.values())
    walltime = math.ceil(WALLTIME_MARGIN * worst + 1)  # + 1 h for the report's bootstrap
    walltime = f"{walltime:02d}"

    if problems:
        print("\nproblems\n  " + "\n  ".join(problems))
    print("\n--- paste into the freeze ---\n")
    total_rows = sum(n for n in rows.values() if n is not None)
    missing_p = sum(n is None for n in rows.values())
    lines = [
        "| | |", "|---|---|",
        f"| rotation inputs | {27 - missing_p} of 27 sealed P files present, {total_rows:,} windows before the scoring "
        f"rules; {sum(len(rules[b]['dropped']) for b in ROTATION)} test birds dropped by the rotation's rules, read from "
        "its sealed manifest |",
        f"| fold 0 | {sum(n is not None for c in fold0.values() for n in c.values())} of 15 sealed files present "
        "(gull: Tier 2 P and V, Tier 3 ARIMA, AVG3, AVG10; goose and dove: the bird study's); read, never recomputed |",
        f"| boundaries | {len(limits)} pinned values, checked against the profile tables |",
        f"| statsmodels | {version('statsmodels')} |",
        f"| ARIMA probe | {probe['s_per_window']:.4f} s/window over {probe['windows']} training windows of "
        f"{MT.short(probe_bird).strip('*')} on {probe['host']}; bird study {BIRD_STUDY_ARIMA_S_PER_WINDOW:.4f} s/window |",
        f"| projection | {projections['probe']['cpu_hours']:.1f} CPU-h at the probe rate "
        f"({projections['bird study']['cpu_hours']:.1f} at the bird study's); "
        f"{MT.short(projections['probe']['largest']).strip('*')} {projections['probe']['largest_hours']:.1f} h; "
        f"makespan on {workers} workers {projections['probe']['makespan_hours']:.1f} h |",
        f"| PBS | normal queue, ncpus={workers}, one bird per worker, largest first; walltime {walltime}:00:00 |",
        "| test-partition reads | none: counts from parquet metadata of sealed prediction files and the sealed "
        "rotation report; the probe used training individuals |",
    ]
    if problems:
        lines.append(f"| problems | {len(problems)}, listed above the block |")
    print("\n".join(lines))
    if problems:
        raise SystemExit(f"\nFAIL: {len(problems)} problem(s); --predict would refuse or fail")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score the baselines on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score baselines that already have files")
    parser.add_argument("--diagnostic", action="store_true",
                        help="files, counts and the ARIMA timing probe; reads no test target")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PBS_NCPUS", "8")),
                        help="worker processes, one bird each (default $PBS_NCPUS or 8)")
    parser.add_argument("--probe-bird", default="ciconia_ciconia", choices=ROTATION, help="--diagnostic timing bird")
    parser.add_argument("--probe-windows", type=int, default=300, help="--diagnostic timing windows")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --diagnostic, --predict, --report, or a combination")

    if args.diagnostic:
        diagnostic(args.workers, args.probe_bird, args.probe_windows)
    if args.predict:
        run_predictions(args.workers, overwrite=args.overwrite)
    if args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report()
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}")


if __name__ == "__main__":
    main()
