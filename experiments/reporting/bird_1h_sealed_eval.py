# GenAI reference: As with out previous evals, I used GenAI to draft this, and didn't need to refine anything. I outlined the plan; the rotation
# evaluator' seal and scoring rules, per horizon summaries at the 150 window floor with the boundary per bird and horizon, the rotation's bootstrap, 
# predictions from HorizonWindows on test_scored individuals, the context ablation by stratum etc. I gave it Tier 2 and 3 evaluators too. 
"""
Sealed evaluation of the 1 h rebuild (`preregistration/birds_1h_exploratory_freeze.md`): the main study at C = 12 and the context ablation at C = 24-96, 
for the 30 rotation birds at direct horizons 1, 6 and 24 h.
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_1h_train as b1h  # noqa: E402
import bird_rotation_sealed_eval as RS  # noqa: E402

from animal_movement_fm.data.multispecies import HorizonWindows, history_positions  # noqa: E402
from animal_movement_fm.data.split import id_key, load_split  # noqa: E402

S2, E3, E4, SPW = RS.S2, RS.E3, RS.E4, RS.SPW

FREEZE = REPO / "preregistration" / b1h.FREEZE
SEALED = REPO / "results" / "birds_1h_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_1h_results.json"
REPORT_MD = SEALED / "birds_1h_results.md"
MANIFEST = SEALED / "birds_1h_sealed_manifest.json"
JOIN_JSON = SEALED / "birds_1h_rbar_join.json"
JOIN_MD = SEALED / "birds_1h_rbar_join.md"
RANDOMNESS_CSV = RS.RANDOMNESS_CSV

BIRDS, FOLD_OF, HORIZONS, CORE = b1h.BIRDS, b1h.FOLD_OF, b1h.HORIZONS, b1h.CORE
ABLATION_CONTEXTS = b1h.ABLATION_CONTEXTS
FLOOR = b1h.FLOOR  # 150 windows per test individual per horizon
MIN_SCORED = b1h.MIN_SCORED_INDIVIDUALS  # a bird is scored at h with this many test_scored individuals at the floor
FULL_SHARE = 0.9  # Ablation strata (09-10): gappy below 0.9 of extended-history bins observed, full at or above
STRATA = ("all", "gappy", "full")

ARMS = ("P", "Bb", "LSTMBb", "XGBBb", "LSTMS", "XGBS")
FAMILIES = {"LSTM": ("LSTMBb", "LSTMS"), "XGB": ("XGBBb", "XGBS")}
AGAINST_P = ("Bb", "LSTMBb", "XGBBb")
ABLATION_ARMS = ("Bb", "LSTMBb", "XGBBb")
JOIN_ARMS = ("LSTMS", "XGBS", "LSTMBb", "XGBBb")  # freeze sec 5, as the rotation
PARTS = {"main": (0, 269), "ablation": (270, 629)}
FRAME_KEYS = ["individual_id", "timestamp", "target_lon", "target_lat"]
PERSISTENCE_COLUMNS = [*FRAME_KEYS, "pred_lon", "pred_lat", "error_m", "horizon"]

if set(ARMS) != set(b1h.ARMS) | {"P"}:
    raise RuntimeError("the evaluator's arms differ from the 1 h driver's")
if any((run.context == CORE) != (i <= PARTS["main"][1]) for i, run in enumerate(b1h.PLAN)):
    raise RuntimeError("the main study is not indices 0-269 at C = 12 with the ablation after it")
PLAN_INDEX = {run.run_id: i for i, run in enumerate(b1h.PLAN)}


def short(stem: str) -> str:
    return RS.short(stem)


@contextlib.contextmanager
def bound():
    """
    Point the rotation evaluator's freeze, manifest and floor, and Tier 3's floor, at this study while the block runs, as
    bird_rotation_train points T at the rotation; both are restored afterwards, so their own callers are unaffected.
    """
    saved = RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR
    RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = FREEZE, MANIFEST, FLOOR, FLOOR
    try:
        yield
    finally:
        RS.FREEZE, RS.MANIFEST, RS.FLOOR, E3.FLOOR = saved


# Seal
def freeze_committed() -> dict:
    """The freeze exists, is tracked and is unchanged against HEAD."""
    with bound():
        return RS.freeze_committed()


def assert_sealed() -> dict:
    """Committed, clean, sec 6 and sec 7 filled in the committed version, and the blob any manifest was attested at."""
    with bound():
        return RS.assert_sealed()


# Records
def read_records() -> list[dict]:
    """Every 1 h run record; XGBoost metadata files are skipped silently."""
    found = []
    for path in sorted(b1h.RESULTS.glob("*.json")):
        if path.name.endswith("_xgb_meta.json"):
            continue
        record = json.loads(path.read_text())
        if record.get("tier") != "birds_1h" or "run_id" not in record or "arm" not in record:
            print(f"  skipped {path.name}: not a 1 h run record (tier {record.get('tier')!r})")
            continue
        if record["arm"] not in b1h.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record['arm']}, which is not a 1 h arm")
        found.append(record)
    return found


def record_map(records: list[dict] | None = None) -> dict[str, dict]:
    return {record["run_id"]: record for record in (read_records() if records is None else records)}


def part_runs(part: str) -> list:
    spans = PARTS.values() if part == "all" else (PARTS[part],)
    return [run for first, last in spans for run in b1h.PLAN[first : last + 1]]


def run_problems(run, record: dict | None) -> list[str]:
    """What stops one planned run counting as finished cleanly."""
    if record is None:
        return ["no record"]
    problems = [
        f"{key} {record.get(key)!r}, expected {want!r}"
        for key, want in (
            ("plan_index", PLAN_INDEX[run.run_id]),
            ("context", run.context),
            ("horizon", run.horizon or None),
            ("opened_test_partition", False),
        )
        if record.get(key) != want
    ]
    if run.xgb:
        stem = b1h.RESULTS / f"{run.run_id}_xgb"
        if not all(Path(f"{stem}_{part}").exists() for part in ("east.ubj", "north.ubj", "meta.json")):
            problems.append("booster missing")
        if record.get("allow_missing") != (run.context > CORE):
            problems.append(f"allow_missing {record.get('allow_missing')}")
        return problems
    steps = (record.get("config") or {}).get("steps")
    if record.get("applied_updates") != b1h.REGISTERED_STEPS or steps != b1h.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {steps}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (b1h.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def completeness(records: dict[str, dict], runs: list) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in runs:
        key = run.arm if run.context == CORE else f"{run.arm}_C{run.context}"
        row = by_arm.setdefault(key, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (index {PLAN_INDEX[run.run_id]}): {'; '.join(issues)}")
    return {
        "n_planned": len(runs),
        "n_records": sum(row["records"] for row in by_arm.values()),
        "n_clean": sum(row["clean"] for row in by_arm.values()),
        "n_neural": sum(not run.xgb for run in runs),
        "n_xgb": sum(run.xgb for run in runs),
        "by_arm": by_arm,
        "problems": problems,
        "complete": not problems,
    }


def assert_training_exclusions(payload: dict, records: dict[str, dict], runs: list) -> dict:
    """Every record names its registered exclusions and pool, and built exactly the pinned 1 h windows after them."""
    problems, checked = [], 0
    for run in runs:
        record = records.get(run.run_id)
        if record is None:
            continue  # completeness reports it
        names = [run.species] if run.species else sorted(b1h.fold_probabilities(run.fold, payload))
        expected = b1h.RT.excluded_in(names, payload)
        if record.get("excluded_individuals") != expected:
            problems.append(f"{run.run_id}: record names {record.get('excluded_individuals')}, expected {expected}")
        if record.get("pool_species") != names:
            problems.append(f"{run.run_id}: pool {record.get('pool_species')}, expected {names}")
        if run.xgb:
            field = "training_windows"
            want = (
                b1h.XGB_POOLED_DRAWS
                if not run.species
                else b1h.TRAIN_WINDOWS[run.species][1 + HORIZONS.index(run.horizon)]
            )
        else:
            field, want = "pool_windows", sum(b1h.TRAIN_WINDOWS[name][0] for name in names)
        if record.get(field) != want:
            problems.append(f"{run.run_id}: {field} {record.get(field)}, expected {want}")
        checked += 1
    if problems:
        raise SystemExit("FAIL: training exclusions\n  " + "\n  ".join(problems))
    return {"status": "PASS", "runs": checked, "species_with_exclusions": len(payload["training_exclusions"])}


# Scoring rules: the rotation's (freeze sec 4), then the 1 h floor per horizon. Ids and counts only leave this block.
def scored_ids(bird: str, dropped) -> list:
    """The bird's test individuals minus the rule drops, in manifest order (ids only)."""
    split = load_split(b1h.T.SPLITS / f"{SPW.SPLIT_STEM.get(bird, bird)}_split_seed42.json")
    gone = {id_key(i) for i in dropped}
    return [i for i in split.test if id_key(i) not in gone]


def test_windows(bird: str, ids, context: int) -> HorizonWindows:
    """
    A bird's test_scored windows at `context`, with the training encoder and clip. Reads the test partition: called after
    assert_sealed by --predict, and after the freeze commit by --diagnostic for counts only.
    """
    return b1h.horizon_windows(bird, ids, context)


def eligibility(windows: HorizonWindows) -> dict[str, dict]:
    """Per horizon: individuals with a window there, those at the floor, windows, and whether the bird is scored."""
    individuals = windows.individuals[windows.individual_index]
    out = {}
    for head, horizon in enumerate(HORIZONS):
        counts = pd.Series(individuals[windows.targets[:, head] >= 0]).value_counts()
        at_floor = int((counts >= FLOOR).sum())
        out[str(horizon)] = {
            "individuals": int(len(counts)),
            "at_floor": at_floor,
            "windows": int(counts.sum()),
            "scored": at_floor >= MIN_SCORED,
        }
    return out


def eligibility_mismatches(eligible: dict[str, dict]) -> list[str]:
    """Each bird's test_scored counts per horizon against horizon_counts.csv (written before the freeze)."""
    counts = pd.read_csv(b1h.HORIZON_COUNTS, dtype={"horizon": str})
    counts = counts[counts["partition"] == "test_scored"].set_index(["species", "horizon"])
    out = []
    for bird, per_horizon in eligible.items():
        for horizon, entry in per_horizon.items():
            row = counts.loc[(bird, horizon)]
            want = (int(row["ge1"]), int(row["ge_equal_time"]), int(row["windows"]))
            got = (entry["individuals"], entry["at_floor"], entry["windows"])
            if got != want:
                out.append(f"{bird} {horizon} h: (individuals, at floor, windows) {got}, horizon_counts.csv {want}")
    return out


def scoring_state(raise_on_mismatch: bool) -> tuple[dict, list[str], dict]:
    drops = RS.rule_drops(RS.load_corpus(), BIRDS)
    mismatches = RS.scored_mismatches(drops)
    eligible = {
        bird: eligibility(test_windows(bird, scored_ids(bird, drops[bird]["dropped"]), CORE)) for bird in BIRDS
    }
    mismatches += eligibility_mismatches(eligible)
    if mismatches and raise_on_mismatch:
        raise SystemExit("FAIL: scoring rules or test counts disagree with the freeze\n  " + "\n  ".join(mismatches))
    return drops, mismatches, eligible


# Prediction
def own_runs(bird: str) -> list:
    """The bird's own specialists and its own fold's pooled arms, at every context; never a model whose pool held it."""
    return [run for run in b1h.PLAN if run.species == bird or (not run.species and run.fold == FOLD_OF[bird])]


def history_share(windows: HorizonWindows, context: int) -> np.ndarray:
    """Per window, the share of extended-history bins t - C + 1 ... t - 12 observed within the segment; NaN at C = 12."""
    if context == CORE:
        return np.full(windows.n_windows, np.nan)
    grid = history_positions(windows.keys, windows.anchors, context)
    return (grid[:, : context - CORE] >= 0).mean(axis=1)


def frame_table(windows, pred_m, context: int, share: np.ndarray, horizons=HORIZONS, persistence: bool = False):
    """b1h.horizon_frames stacked over `horizons`, with horizon, context and history_share columns."""
    frames = b1h.horizon_frames(windows, pred_m)
    parts = []
    for head, horizon in enumerate(windows.horizons):
        if horizon not in horizons:
            continue
        rows = np.flatnonzero(windows.targets[:, head] >= 0)
        parts.append(frames[horizon][1 if persistence else 0].assign(
            horizon=horizon, context=context, history_share=share[rows]))
    return pd.concat(parts, ignore_index=True)


def assert_same_persistence(reference: pd.DataFrame, other: pd.DataFrame, label: str) -> None:
    """Persistence must be identical across C: same windows, same targets, same errors, bit for bit."""
    same = len(reference) == len(other) and all(
        np.array_equal(reference[column].to_numpy(), other[column].to_numpy()) for column in PERSISTENCE_COLUMNS
    )
    if not same:
        raise SystemExit(f"FAIL: persistence at {label} differs from C = {CORE}; every C is scored on the same windows")


def load_models(runs: list, records: dict[str, dict], device, context: int) -> tuple[list, list]:
    """(neural, xgb) lists of (run, model, k_h) for the runs at `context`, each checked against its record."""
    from animal_movement_fm.models.xgboost import WindowXGBoost

    neural, xgb = [], []
    for run in runs:
        if run.context != context:
            raise SystemExit(f"FAIL: {run.run_id} is a C = {run.context} run, not C = {context}")
        record = records[run.run_id]
        k = np.asarray(record["horizon_k"], dtype=float)
        if run.xgb:
            model = WindowXGBoost.load(b1h.RESULTS / f"{run.run_id}_xgb")
            if model.allow_missing != (context > CORE):
                raise SystemExit(f"FAIL: {run.run_id}'s booster has allow_missing {model.allow_missing} at C = {context}")
            xgb.append((run, model, k))
            continue
        loaded = E3.load_reported(b1h.RESULTS, run.run_id)
        if loaded is None:
            raise SystemExit(f"FAIL: {run.run_id} has no reported checkpoint")
        model, state = loaded
        if list(state["horizon_k"]) != list(record["horizon_k"]) or state["context"] != context:
            raise SystemExit(f"FAIL: {run.run_id}'s checkpoint disagrees with its record on k_h or C")
        neural.append((run, model.to(device), k))
    return neural, xgb


def score_context(windows: HorizonWindows, neural: list, xgb: list, device, chunk: int = 8_192) -> dict[str, np.ndarray]:
    """
    (n, H, 2) predicted east / north metres per run, every window gathered once for all the context's models. Neural:
    b1h.predictions_m's rule (output x k_h x scale, padding mask only for C > 12). XGBoost: its horizon only, on windows
    with a target there, unobserved bins NaN as in training; the other heads stay zero and are never scored.
    """
    observed = windows.context > CORE
    out = {run.run_id: np.zeros((windows.n_windows, len(HORIZONS), 2)) for run, *_ in neural + xgb}
    with torch.no_grad():
        for start in range(0, windows.n_windows, chunk):
            ids = np.arange(start, min(start + chunk, windows.n_windows))
            features, token_mask, _, target_mask, scale = windows.gather(ids)
            if neural:
                x = torch.as_tensor(features, dtype=torch.float32, device=device)
                mask = torch.as_tensor(token_mask, device=device) if observed else None
                for run, model, k in neural:
                    prediction = model(x, observed=mask).double().cpu().numpy()
                    out[run.run_id][ids] = prediction * k[None, :, None] * scale[:, None, None]
            for run, model, k in xgb:
                head = HORIZONS.index(run.horizon)
                rows = np.flatnonzero(target_mask[:, head])
                if rows.size:
                    x = np.where(token_mask[rows, :, None], features[rows], np.nan).astype(np.float32)
                    out[run.run_id][ids[rows], head] = model.predict(x).astype(float) * k[head] * scale[rows, None]
    return out


def prediction_path(bird: str, tag: str) -> Path:
    return PREDICTIONS / f"{bird}__{tag}.parquet"


def persistence_tag(context: int) -> str:
    return "P" if context == CORE else f"P_C{context}"


def predict_bird(bird: str, dropped, records: dict[str, dict], device, overwrite: bool, contexts) -> list[str]:
    """P at C = 12 and at every C scored (checked identical), then the bird's own runs at each C."""
    ids = scored_ids(bird, dropped)
    runs = [run for run in own_runs(bird) if run.context in contexts]
    for run in runs:
        if not run.species and bird in records[run.run_id]["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {bird}")
    written, reference = [], None
    for context in (CORE, *(c for c in contexts if c != CORE)):
        started = time.time()
        windows = test_windows(bird, ids, context)
        share = history_share(windows, context)
        persistence = frame_table(windows, None, context, share, persistence=True)
        if reference is None:
            reference = persistence
            print(f"\n{bird}: {windows.n_windows:,} test_scored windows, {len(windows.individuals)} individuals", flush=True)
        assert_same_persistence(reference, persistence, f"{bird} C = {context}")
        out = prediction_path(bird, persistence_tag(context))
        if overwrite or not out.exists():
            persistence.to_parquet(out, index=False)
        written.append(out.name)
        if context not in contexts:
            continue
        todo = [run for run in runs if run.context == context]
        skipped = [run for run in todo if prediction_path(bird, run.run_id).exists() and not overwrite]
        todo = [run for run in todo if run not in skipped]
        written += [prediction_path(bird, run.run_id).name for run in skipped]
        predicted = score_context(windows, *load_models(todo, records, device, context), device) if todo else {}
        for run in todo:
            horizons = (run.horizon,) if run.xgb else HORIZONS
            out = prediction_path(bird, run.run_id)
            frame_table(windows, predicted[run.run_id], context, share, horizons).to_parquet(out, index=False)
            written.append(out.name)
        print(f"  C = {context:>2}: {len(todo)} runs scored, {len(skipped)} existed ({time.time() - started:.0f}s)",
              flush=True)
    return written


def contexts_of(part: str) -> tuple[int, ...]:
    return {"main": (CORE,), "ablation": ABLATION_CONTEXTS, "all": (CORE, *ABLATION_CONTEXTS)}[part]


def run_predictions(device, overwrite: bool = False, part: str = "all") -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    payload = b1h.load_registered()
    records = record_map()
    runs = part_runs(part)
    gap = completeness(records, runs)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    exclusions = assert_training_exclusions(payload, records, runs)
    drops, _, eligible = scoring_state(raise_on_mismatch=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest |= {"attestation": attestation, "scoring_rules": drops, "eligibility": eligible}
    entry = {"completeness": gap, "training_exclusions": exclusions, "species": {}}
    for bird in BIRDS:
        entry["species"][bird] = predict_bird(bird, drops[bird]["dropped"], records, device, overwrite,
                                              contexts_of(part))
    manifest.setdefault("parts", {})[part] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def load_tables(bird: str, records: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """The bird's saved predictions; a file from a model outside its own fold, or an unknown tag, is refused."""
    tables = {path.stem.split("__", 1)[1]: pd.read_parquet(path) for path in sorted(PREDICTIONS.glob(f"{bird}__*.parquet"))}
    allowed = {run.run_id for run in own_runs(bird)} | {persistence_tag(c) for c in (CORE, *ABLATION_CONTEXTS)}
    strays = sorted(set(tables) - allowed)
    if strays:
        raise SystemExit(f"FAIL: {bird} holds predictions from models outside its own fold: {strays}")
    if "P" not in tables:
        raise SystemExit(f"FAIL: {bird} has no P predictions; P defines the strata and the skill denominator")
    return tables


def at(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One horizon's rows, in the fixed order used for paired comparisons."""
    rows = frame[frame["horizon"] == horizon]
    return rows.sort_values(["individual_id", "timestamp"], kind="stable").reset_index(drop=True)


def assert_aligned(frame: pd.DataFrame, persistence: pd.DataFrame, label: str) -> None:
    same = len(frame) == len(persistence) and all(
        np.array_equal(frame[column].to_numpy(), persistence[column].to_numpy()) for column in FRAME_KEYS
    )
    if not same:
        raise SystemExit(f"FAIL: {label} does not describe the same windows as persistence; re-run --predict --overwrite")


def parse_tag(tag: str, records: dict[str, dict]) -> dict | None:
    if tag == "P":
        return {"arm": "P", "seed": None, "k": None, "draw": None}
    if tag in records:
        return {"arm": records[tag]["arm"], "seed": records[tag]["seed"], "k": None, "draw": None}
    print(f"  ignored {tag}: not a 1 h run or P")
    return None


def tags_at(bird: str, arm: str, horizon: int, context: int) -> list[str]:
    """The run ids of one bird's arm at (horizon, C): every seed of a neural arm, the one XGBoost run at h."""
    return [run.run_id for run in own_runs(bird)
            if run.arm == arm and run.context == context and (not run.xgb or run.horizon == horizon)]


def expected_seeds(bird: str, arm: str, horizon: int) -> set[int]:
    return {run.seed for run in own_runs(bird)
            if run.arm == arm and run.context == CORE and (not run.xgb or run.horizon == horizon)}


# Statistic
def main_cell(bird: str, horizon: int, tables: dict, records: dict, eligible: dict) -> dict:
    """One bird at one horizon: scored or not with n, share in travel, each arm's skill, and the readings."""
    counts = eligible[str(horizon)]
    boundary = b1h.BOUNDARY_M[(bird, horizon)]
    entry = {"boundary_m": boundary, **counts}
    if not counts["scored"]:
        return entry
    persistence = at(tables["P"], horizon)
    keep = S2.eligible_mask(persistence, FLOOR)
    entry["share_in_travel"] = float(S2.stratum_masks(persistence, boundary)["travel"][keep].mean())
    frames = {"P": persistence}
    for arm in ARMS[1:]:
        for tag in tags_at(bird, arm, horizon, CORE):
            frames[tag] = at(tables[tag], horizon)
            assert_aligned(frames[tag], persistence, f"{bird} {horizon} h {tag}")
    with bound():
        E4.assert_no_collisions(frames, records, parse=parse_tag)
        summaries = E3.summarise_all(frames, persistence, boundary, records, parse=parse_tag)
        problems = [arm for arm in ARMS if arm not in summaries]
        for arm in ARMS[1:]:
            seeds = {key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0}
            if arm in summaries and seeds != expected_seeds(bird, arm, horizon):
                problems.append(f"{arm} has seeds {sorted(seeds)}, expected {sorted(expected_seeds(bird, arm, horizon))}")
        if problems:
            raise SystemExit(f"FAIL: {bird} {horizon} h: " + "; ".join(problems))
        arms = RS.arm_blocks(summaries)
        resampler = RS.Resampler(persistence, boundary)
        entry["arms"] = arms
        entry["readings"] = {
            "gaps": {family: RS.compare(pooled, specialist, arms, frames, persistence, boundary, resampler)
                     for family, (pooled, specialist) in FAMILIES.items()},
            "against_p": {arm: RS.compare(arm, "P", arms, frames, persistence, boundary, resampler) for arm in AGAINST_P},
        }
    return entry


def travel_p90(frame: pd.DataFrame, persistence: pd.DataFrame, mask: np.ndarray, boundary: float) -> tuple[float, int, int]:
    """Travel p90 on the masked windows (individuals already at the floor on the whole horizon), with n's."""
    summary = S2.summarise(frame.loc[mask].reset_index(drop=True), persistence.loc[mask].reset_index(drop=True),
                           boundary, floor=1)
    return summary["travel"].get("p90_m", float("nan")), summary["n_individuals"], summary["travel"]["n_windows"]


def ablation_cell(bird: str, horizon: int, tables: dict) -> dict:
    """
    Per arm, C and stratum: skill against P at C and at C = 12 on the same windows (median of seeds), and the
    difference. Individuals are held at the floor on the whole horizon; strata come from the share observed at that C.
    """
    boundary = b1h.BOUNDARY_M[(bird, horizon)]
    persistence = at(tables["P"], horizon)
    keep = S2.eligible_mask(persistence, FLOOR)
    out: dict = {}
    for context in ABLATION_CONTEXTS:
        at_c = at(tables[persistence_tag(context)], horizon)
        assert_same_persistence(persistence, at_c, f"{bird} {horizon} h C = {context}")
        share = at_c["history_share"].to_numpy(float)
        masks = {"all": keep, "gappy": keep & (share < FULL_SHARE), "full": keep & (share >= FULL_SHARE)}
        for stratum, mask in masks.items():
            reference, n_individuals, n_travel = travel_p90(persistence, persistence, mask, boundary)
            cell = {"n_individuals": n_individuals, "n_travel_windows": n_travel, "p_p90_m": reference}
            for arm in ABLATION_ARMS:
                skills = {}
                for label, c in (("C", context), ("C12", CORE)):
                    values = []
                    for tag in tags_at(bird, arm, horizon, c):
                        frame = at(tables[tag], horizon)
                        assert_aligned(frame, persistence, f"{bird} {horizon} h {tag}")
                        values.append(travel_p90(frame, persistence, mask, boundary)[0])
                    skills[label] = E3.skill(float(np.median(values)) if values else float("nan"), reference)
                cell[arm] = {"skill": skills["C"], "skill_C12": skills["C12"],
                             "difference": skills["C"] - skills["C12"]}
            out.setdefault(str(context), {})[stratum] = cell
    return out


def _median(values) -> dict:
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return {"n": len(finite), "median": float(np.median(finite)) if finite else None}


def across_birds(report: dict) -> dict:
    """Per horizon, over the scored birds and over the common set (scored at all three): median skill and gap bands."""
    common = [bird for bird, entry in report["species"].items()
              if all(entry["horizons"][str(h)]["scored"] for h in HORIZONS)]
    out = {"common_set": common, "horizons": {}}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        block = {"scored": scored, "skill": {}, "bands": {}}
        for label, birds in (("scored", scored), ("common", common)):
            cells = [report["species"][bird]["horizons"][str(horizon)] for bird in birds]
            for arm in ARMS[1:]:
                block["skill"].setdefault(arm, {})[label] = _median(E3.travel_skill(c["arms"], arm) for c in cells)
            for family in FAMILIES:
                counts = Counter(RS.band(c["readings"]["gaps"][family].get("skill_difference")) for c in cells)
                block["bands"].setdefault(family, {})[label] = {
                    key: counts.get(key, 0) for key in ("above", "within", "below", "not scored")}
        out["horizons"][str(horizon)] = block
    return out


def ablation_summary(report: dict, common: list[str]) -> dict:
    """Per arm, horizon, C and stratum: medians over scored birds and over the common set of skill and of C - C12."""
    out: dict = {}
    for horizon in HORIZONS:
        scored = [bird for bird, entry in report["species"].items() if entry["horizons"][str(horizon)]["scored"]]
        for label, birds in (("scored", scored), ("common", common)):
            for context in ABLATION_CONTEXTS:
                for stratum in STRATA:
                    cells = [report["species"][b]["horizons"][str(horizon)]["ablation"][str(context)][stratum]
                             for b in birds]
                    for arm in ABLATION_ARMS:
                        node = out.setdefault(arm, {}).setdefault(str(horizon), {}).setdefault(str(context), {})
                        node.setdefault(stratum, {})[label] = {
                            "skill": _median(c[arm]["skill"] for c in cells),
                            "difference": _median(c[arm]["difference"] for c in cells),
                        }
    return out


# Report
def build_report(part: str = "all") -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    b1h.check_boundaries()
    records = record_map()
    with_ablation = part in ("ablation", "all")
    report: dict = {
        "study": "birds_1h",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "evaluation_floor": FLOOR,
        "min_scored_individuals": MIN_SCORED,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": E3.NOISE_SCALE,
        "completeness": completeness(records, part_runs("all" if with_ablation else "main")),
        "scoring_rules": manifest["scoring_rules"],
        "species": {},
    }
    compared = 0
    for bird in BIRDS:
        tables = load_tables(bird, records)
        for context in ABLATION_CONTEXTS if with_ablation else ():
            assert_same_persistence(tables["P"], tables[persistence_tag(context)], f"{bird} C = {context}")
            compared += 1
        entry = {"fold": FOLD_OF[bird], "dropped": manifest["scoring_rules"][bird]["dropped"], "horizons": {}}
        for horizon in HORIZONS:
            cell = main_cell(bird, horizon, tables, records, manifest["eligibility"][bird])
            if with_ablation and cell["scored"]:
                cell["ablation"] = ablation_cell(bird, horizon, tables)
            entry["horizons"][str(horizon)] = cell
        report["species"][bird] = entry
    report["persistence_identical_across_C"] = {"status": "PASS", "comparisons": compared}
    report["across_birds"] = across_birds(report)
    if with_ablation:
        report["ablation"] = ablation_summary(report, report["across_birds"]["common_set"])
    return report


def _s(value) -> str:
    return f"{value:+.4f}" if value is not None and np.isfinite(value) else "—"


def _median_cell(block: dict) -> str:
    return f"{_s(block['median'])} ({block['n']})"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# 1 h rebuild: sealed evaluation (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"Runs: {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean. Travel p90 skill "
              f"against P at the {report['evaluation_floor']}-window floor, a bird scored at h with ≥ "
              f"{report['min_scored_individuals']} test_scored individuals there; median of seeds; paired bootstrap "
              f"2,000 at seed 42. Persistence identical across C: {report['persistence_identical_across_C']['status']} "
              f"({report['persistence_identical_across_C']['comparisons']} comparisons).", ""]

    lines += ["## Per bird and horizon (skill against P; family gaps with skill intervals)", "",
              "| fold | bird | h | scored (at floor / with windows) | travel share | Bb | LSTM-Bb | XGB-Bb | LSTM-S | "
              "XGB-S | LSTM-Bb − LSTM-S | XGB-Bb − XGB-S |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for bird, entry in report["species"].items():
        for horizon, cell in entry["horizons"].items():
            n = f"{cell['at_floor']} / {cell['individuals']}"
            if not cell["scored"]:
                lines.append(f"| {entry['fold']} | {short(bird)} | {horizon} | unscored, {n} |" + " — |" * 8)
                continue
            skills = [_s(E3.travel_skill(cell["arms"], arm)) for arm in ARMS[1:]]
            gaps = []
            for family in FAMILIES:
                comparison = cell["readings"]["gaps"][family]
                boot = (comparison.get("skill_bootstrap") or {}).get("skill_difference")
                gaps.append(f"{_s(comparison.get('skill_difference'))} {RS._interval(boot)}")
            lines.append(f"| {entry['fold']} | {short(bird)} | {horizon} | {n} | {cell['share_in_travel']:.3f} | "
                         + " | ".join(skills + gaps) + " |")

    across = report["across_birds"]
    lines += ["", f"## Across birds (median skill (n); common set = scored at all three horizons, "
                  f"{len(across['common_set'])} birds)", "",
              "| h | arm | scored | common |", "|---:|---|---:|---:|"]
    for horizon, block in across["horizons"].items():
        for arm in ARMS[1:]:
            entry = block["skill"][arm]
            lines.append(f"| {horizon} | {arm} | {_median_cell(entry['scored'])} | {_median_cell(entry['common'])} |")
    lines += ["", "## Family gap bands (above +0.05 / within / below −0.05 / not scored)", "",
              "| h | family | scored | common |", "|---:|---|---|---|"]
    for horizon, block in across["horizons"].items():
        for family, entry in block["bands"].items():
            cells = [" / ".join(str(entry[label][key]) for key in ("above", "within", "below", "not scored"))
                     for label in ("scored", "common")]
            lines.append(f"| {horizon} | {family} | {cells[0]} | {cells[1]} |")

    if "ablation" in report:
        lines += ["", "## Context ablation (median over scored birds of skill at C, and of skill at C − skill at C = 12 "
                      "on the same windows; n birds)", "",
                  "| arm | h | C | all | gappy (< 0.9) | full (≥ 0.9) |", "|---|---:|---:|---|---|---|"]
        for arm, per_horizon in report["ablation"].items():
            for horizon, per_context in per_horizon.items():
                for context, per_stratum in per_context.items():
                    cells = [f"{_s(per_stratum[s]['scored']['skill']['median'])}, "
                             f"Δ {_s(per_stratum[s]['scored']['difference']['median'])} "
                             f"({per_stratum[s]['scored']['difference']['n']})" for s in STRATA]
                    lines.append(f"| {arm} | {horizon} | {context} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


# Post-seal join
def rbar_join(report: dict, table: pd.DataFrame) -> dict:
    """Skill against Rbar_corr per horizon, over the birds scored there, for the four JOIN_ARMS (freeze sec 5)."""
    if not (report.get("attestation") or {}).get("freeze_commit"):
        raise SystemExit("FAIL: the report carries no freeze attestation; --join reads only a sealed report")
    rbar = table.set_index("species")["rbar_corr"]
    keys = {bird: SPW.TIER4_PROFILE_KEY.get(bird, bird) for bird in report["species"]}
    missing = sorted(key for key in keys.values() if key not in rbar.index)
    if missing:
        raise SystemExit(f"FAIL: {RANDOMNESS_CSV.name} has no row for {missing}")
    values = {bird: float(rbar[key]) for bird, key in keys.items()}
    out = {"rbar_corr": values, "horizons": {}}
    for horizon in HORIZONS:
        cells = {bird: entry["horizons"][str(horizon)] for bird, entry in report["species"].items()}
        scored = [bird for bird, cell in cells.items() if cell["scored"]]
        out["horizons"][str(horizon)] = {
            "birds": scored,
            "rho": {arm: RS.spearman([values[bird] for bird in scored],
                                     [E3.travel_skill(cells[bird]["arms"], arm) for bird in scored])
                    for arm in JOIN_ARMS},
        }
    return out


def write_join(join: dict) -> None:
    JOIN_JSON.write_text(json.dumps(join, indent=2) + "\n")
    lines = ["# 1 h rebuild: skill against Rbar_corr", "", "| h | arm | n | ρ |", "|---:|---|---:|---:|"]
    lines += [f"| {horizon} | {arm} | {value['n']} | {_s(value['rho'])} |"
              for horizon, block in join["horizons"].items() for arm, value in block["rho"].items()]
    JOIN_MD.write_text("\n".join(lines) + "\n")


# Diagnostic: run records, registered files, and test-bird fixes and windows for ids and counts only
def section7_markdown(gap: dict, exclusions: str, drops: dict, mismatches: list[str], eligible: dict) -> str:
    identity = "; ".join(f"{short(b)} {', '.join(sorted({h['test'] for h in e['identity']}))}"
                         for b, e in drops.items() if e["identity"]) or "none"
    val_test = ", ".join(f"{short(b)} {len({h['test'] for h in e['val_test']})}"
                         for b, e in drops.items() if e["val_test"]) or "none"
    scored = {h: sorted(b for b in BIRDS if eligible[b][str(h)]["scored"]) for h in HORIZONS}
    common = [b for b in BIRDS if all(b in scored[h] for h in HORIZONS)]
    unscored = "; ".join(
        f"{h} h: " + (", ".join(f"{short(b)} {eligible[b][str(h)]['at_floor']}" for b in BIRDS if b not in scored[h])
                      or "none")
        for h in HORIZONS)
    neural_clean = sum(row["clean"] for key, row in gap["by_arm"].items() if not key.startswith("XGB"))
    xgb_clean = sum(row["clean"] for key, row in gap["by_arm"].items() if key.startswith("XGB"))
    lines = [
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean |",
        f"| neural ({gap['n_neural']}) | {neural_clean} clean: {b1h.REGISTERED_STEPS:,} of {b1h.REGISTERED_STEPS:,} "
        "updates applied, none withheld, no non-finite steps |",
        f"| XGBoost ({gap['n_xgb']}) | {xgb_clean} fit once, boosters present |",
        f"| training exclusions | {exclusions} |",
        f"| identity rule | {identity} |",
        f"| val–test rule | {val_test} |",
        f"| scored test birds and counts | {'as sec 4 and horizon_counts.csv' if not mismatches else 'differ: ' + '; '.join(mismatches)} |",
        f"| birds scored at the {FLOOR}-window floor | " + ", ".join(f"{h} h {len(scored[h])}" for h in HORIZONS)
        + f"; all three {len(common)} |",
        f"| unscored (individuals at the floor) | {unscored} |",
        "| test-partition reads | the diagnostic read test-bird fixes and windows (ids and counts only) before this "
        "section was committed |",
        f"| deferred or missing | {'none' if gap['complete'] else '; '.join(gap['problems'])} |",
    ]
    return "\n".join(lines)


def diagnostic(part: str = "all") -> None:
    """Completeness, exclusions, the scoring rules and the per-horizon floor, and the sec 7 block. Scores nothing."""
    freeze_committed()  # committed and clean before any test-bird fixes are read
    with bound():
        text = RS.committed_freeze()
        for number in (6, 7):
            try:
                RS.check_filled(number, RS.section_body(text, number))
                print(f"sec {number}            : filled")
            except SystemExit as reason:
                print(f"sec {number}            : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    payload = b1h.load_registered()
    records = record_map()
    runs = part_runs(part)
    gap = completeness(records, runs)
    try:
        exclusions, exclusions_ok = assert_training_exclusions(payload, records, runs), True
    except SystemExit as reason:
        exclusions, exclusions_ok = str(reason), False
    drops, mismatches, eligible = scoring_state(raise_on_mismatch=False)

    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean ({part})")
    for key, row in gap["by_arm"].items():
        print(f"  {key:<11} {row['records']:>3} of {row['planned']:>3}, {row['clean']:>3} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print(f"exclusions       : {exclusions}")
    print("\nscored test birds per horizon (at the floor / with windows)")
    for bird in BIRDS:
        cells = "  ".join(f"{h} h {eligible[bird][str(h)]['at_floor']:>3}/{eligible[bird][str(h)]['individuals']:<3}"
                          + ("" if eligible[bird][str(h)]["scored"] else "*") for h in HORIZONS)
        print(f"  {bird:<26} fold {FOLD_OF[bird]}  {drops[bird]['scored']:>3} of {drops[bird]['test_birds']:>3}  {cells}")
    print("  * unscored at that horizon")
    print("\n" + ("scored counts match sec 4 and horizon_counts.csv" if not mismatches
                  else "MISMATCH:\n  " + "\n  ".join(mismatches)))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(gap, exclusions, drops, mismatches, eligible))
    if not (gap["complete"] and exclusions_ok and not mismatches):
        raise SystemExit("FAIL: the diagnostic found a problem; sec 7 must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score every arm on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score runs that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true",
                        help="runs, exclusions, scoring rules, the per-horizon floor and the sec 7 block; scores nothing")
    parser.add_argument("--part", choices=("main", "ablation", "all"), default="all",
                        help="main study (0-269), context ablation (270-629), or both")
    parser.add_argument("--join", action="store_true", help="post-seal: skill against Rbar_corr from the report")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic or args.join):
        raise SystemExit("pass --diagnostic, --predict, --report, --join, or a combination")

    if args.diagnostic:
        diagnostic(args.part)
    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device           : {device}", flush=True)
        run_predictions(device, overwrite=args.overwrite, part=args.part)
    if args.report:
        SEALED.mkdir(parents=True, exist_ok=True)
        report = build_report(args.part)
        REPORT_JSON.write_text(json.dumps(report, indent=2, default=str) + "\n")
        write_markdown(report, REPORT_MD)
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}")
    if args.join:
        if not REPORT_JSON.exists():
            raise SystemExit(f"FAIL: {REPORT_JSON} does not exist; run --report first")
        write_join(rbar_join(json.loads(REPORT_JSON.read_text()), pd.read_csv(RANDOMNESS_CSV)))
        print(f"wrote {JOIN_JSON}\nwrote {JOIN_MD}")


if __name__ == "__main__":
    main()
