# GenAI reference: I specified this evaluator for the rotating bird study, that being use the committed freeze with secs 6/7,
# apply the two sec 4 scoring rules and check their counts, include an ids/counts-only diagnostic, report the bird study
# statistic with skill and metre intervals, carry fold 0 from the sealed 5 seed results with a seeds 0-2 descriptive
# column etc. GenAI wrote the code from that, reusing helpers from our bird and Tier 2-4 evaluators.
"""
Sealed evaluation of the rotating held-out bird study (`preregistration/birds_rotation_exploratory_freeze.md`).

Exploratory: nothing this produces is a tiered result.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import io
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_train as RT  # noqa: E402
import bird_sealed_eval as BE  # noqa: E402

from animal_movement_fm.data.benchmark import load_split_frame  # noqa: E402
from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.data.split import id_key, load_split  # noqa: E402
from animal_movement_fm.eval.fm_predictor import (  # noqa: E402
    MovementFMPredictor,
    WindowModelPredictor,
    evaluate_batched,
)
from animal_movement_fm.eval.harness import evaluate  # noqa: E402
from experiments.data import bird_duplicate_check as D  # noqa: E402

S2, E3, E4, BT, SPW = BE.S2, BE.E3, BE.E4, BE.BT, RT.SPW

FREEZE = REPO / "preregistration" / "birds_rotation_exploratory_freeze.md"
SEALED = REPO / "results" / "birds_rotation_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "birds_rotation_results.json"
REPORT_MD = SEALED / "birds_rotation_results.md"
MANIFEST = SEALED / "birds_rotation_sealed_manifest.json"
JOIN_JSON = SEALED / "birds_rotation_rbar_join.json"
JOIN_MD = SEALED / "birds_rotation_rbar_join.md"
RANDOMNESS_CSV = REPO / "results" / "randomness" / "species_randomness.csv"

FLOOR = E3.FLOOR
NOISE_SCALE = E3.NOISE_SCALE
SHARED_AT = D.EXCLUDE_AT  # >= 100 exact fixes, freeze sec 4

FOLD0 = tuple(BT.HELD_OUT)  # gull, goose, dove: carried, never re-predicted
ROTATION = tuple(RT.HELD_OUT)
BIRDS = FOLD0 + ROTATION  # fold order
FOLD_OF = {bird: 0 for bird in FOLD0} | dict(RT.FOLD_OF)

ARMS = ("P", "Bb", "LSTMBb", "XGBBb", "Zb", "LSTMS", "XGBS")
FAMILIES = {"LSTM": ("LSTMBb", "LSTMS"), "XGB": ("XGBBb", "XGBS")}
AGAINST_P = ("Bb", "LSTMBb", "XGBBb", "Zb")
PLAN_RUNS = {"Bb": 27, "LSTMBb": 27, "XGBBb": 9, "LSTMS": 81, "XGBS": 27, "Zb": 27}  # freeze sec 3
if dict(Counter(run.arm for run in RT.PLAN)) != PLAN_RUNS:
    raise RuntimeError("the rotation plan no longer matches freeze sec 3")

# freeze sec 4: (test birds, scored) after both rules; every other bird is (10, 10), fold 0 included
SCORED_IN_FREEZE = {
    "ciconia_ciconia": (105, 100),
    "gyps_himalayensis": (10, 8),
    "cathartes_aura": (10, 9),
    "ardea_alba": (10, 9),
}

PLACEHOLDER = re.compile(BE.PLACEHOLDER.pattern + r"|\bjack\s+to\s+write\b", re.IGNORECASE)

# From the PBS job logs, as reported on 30-09-2026; none of these is in the run records.
AS_TRAINED_FROM_PBS = {
    "exit status": "0 for every run",
    "queue and precision": "gpuvolta, fp32",
    "compute": "493 SU against the ~955 SU estimated in sec 3; longest run 00:11:09",
}


def short(stem: str) -> str:
    genus, _, epithet = stem.partition("_")
    return f"*{genus[0].upper()}. {epithet}*"


# Seal
def section_body(text: str, number: int) -> str:
    """The body of freeze section `number`, found by its number prefix, up to the next `## ` heading."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(rf"##\s+{number}\.", line)), None)
    if start is None:
        raise SystemExit(f"FAIL: {FREEZE.name} has no section {number}")
    body = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip()


def check_filled(number: int, body: str) -> None:
    """Refuse an empty section, or one that still carries placeholder wording."""
    if not body.strip():
        raise SystemExit(f"FAIL: {FREEZE.name} sec {number} is empty in the committed version.")
    found = PLACEHOLDER.search(body)
    if found:
        raise SystemExit(f"FAIL: {FREEZE.name} sec {number} still says {found.group(0)!r} in the committed version.")


def committed_freeze() -> str:
    shown = E4.git("show", f"HEAD:{FREEZE.relative_to(REPO).as_posix()}")
    if shown.returncode != 0:
        raise SystemExit(f"FAIL: cannot read {FREEZE.name} from HEAD, so it is not committed: {shown.stderr.strip()}")
    return shown.stdout


def freeze_committed() -> dict:
    """The freeze is tracked and unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    previous = S2.FREEZE
    S2.FREEZE = FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def assert_sealed() -> dict:
    """Committed, clean, and sec 6 and sec 7 filled in the committed version. Run by --predict only."""
    attestation = freeze_committed()
    text = committed_freeze()
    for number in (6, 7):
        check_filled(number, section_body(text, number))
    attestation["sections_filled"] = [6, 7]
    if MANIFEST.exists():
        earlier = json.loads(MANIFEST.read_text()).get("attestation") or {}
        if earlier.get("freeze_blob") and earlier["freeze_blob"] != attestation["freeze_blob"]:
            raise SystemExit(
                f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but {FREEZE.name} is now "
                f"blob {attestation['freeze_blob'][:12]}. Rebuild the report with --report instead."
            )
    return attestation


# Records
def read_records() -> list[dict]:
    """Every rotation run record; XGBoost metadata files are skipped silently."""
    found = []
    for path in sorted(RT.RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("tier") != "birds_rotation" or "run_id" not in record or "arm" not in record:
            if not path.name.endswith("_xgb_meta.json"):
                print(f"  skipped {path.name}: not a rotation run record (tier {record.get('tier')!r})")
            continue
        if record["arm"] not in RT.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record['arm']}, which is not a rotation arm")
        found.append(record)
    return found


def record_map(records: list[dict] | None = None) -> dict[str, dict]:
    return {record["run_id"]: record for record in (read_records() if records is None else records)}


def run_problems(run, record: dict | None) -> list[str]:
    """What stops one planned run counting as finished cleanly."""
    if record is None:
        return ["no record"]
    if RT.ARMS[run.arm].get("xgb"):
        boosters = [RT.RESULTS / f"{run.run_id}_xgb_{axis}.ubj" for axis in ("east", "north")]
        return [] if all(path.exists() for path in boosters) else ["booster missing"]
    problems = []
    steps = (record.get("config") or {}).get("steps")
    if record.get("applied_updates") != RT.REGISTERED_STEPS or steps != RT.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {steps}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (RT.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def completeness(records: dict[str, dict]) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in RT.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (fold {run.fold}): {'; '.join(issues)}")
    neural = [run for run in RT.PLAN if not RT.ARMS[run.arm].get("xgb")]
    return {
        "n_planned": len(RT.PLAN),
        "n_records": sum(row["records"] for row in by_arm.values()),
        "n_clean": sum(row["clean"] for row in by_arm.values()),
        "n_neural": len(neural),
        "n_xgb": len(RT.PLAN) - len(neural),
        "by_arm": by_arm,
        "problems": problems,
        "complete": not problems,
    }


def assert_training_exclusions(payload: dict, records: dict[str, dict]) -> dict:
    """Every record names its registered exclusions and built exactly the registered windows after them."""
    counts, exclusions = payload["train_window_counts"], payload["training_exclusions"]
    problems, checked = [], 0
    for run in RT.PLAN:
        record = records.get(run.run_id)
        if record is None:
            continue  # completeness reports it
        names = [run.species] if run.species else payload["pools"][str(run.fold)]
        expected = {name: exclusions[name] for name in names if name in exclusions}
        if record.get("excluded_individuals") != expected:
            problems.append(f"{run.run_id}: record names {record.get('excluded_individuals')}, expected {expected}")
        if RT.ARMS[run.arm].get("xgb"):
            field = "training_windows"
            want = RT.XGB_POOLED_DRAWS if not run.species else counts[run.species]
        else:
            field, want = "pool_windows", sum(counts[name] for name in names)
        if record.get(field) != want:
            problems.append(f"{run.run_id}: {field} {record.get(field)}, expected {want}")
        checked += 1
    if problems:
        raise SystemExit("FAIL: training exclusions\n  " + "\n  ".join(problems))
    return {"status": "PASS", "runs": checked, "species_with_exclusions": len(exclusions)}


# Scoring rules, freeze sec 4. Test-bird fixes are read for identity only: ids and counts leave this block.
def corpus_stems() -> list[str]:
    return sorted(path.name.removesuffix("_segments.parquet") for path in SPW.PROCESSED.glob("*_segments.parquet"))


def species_of(stem: str) -> str:
    """The species a segments file holds; the `stork` extract is *C. ciconia*."""
    return SPW.TIER4_ELIGIBILITY_ALIAS.get(stem, stem)


def load_corpus() -> pd.DataFrame:
    """Every species in data/processed, every individual, unassigned ones included."""
    frames = []
    for stem in corpus_stems():
        frame, _ = D.load_fixes(stem, keep_unassigned=True)
        frames.append(frame.assign(stem=stem, species=species_of(stem)))
    return pd.concat(frames, ignore_index=True)


def rule_drops(fixes: pd.DataFrame, birds) -> dict[str, dict]:
    """
    The test birds each rule drops, per scored bird.

    `fixes` holds stem, species, individual, partition and (timestamp, lon, lat). A bird's scored test fixes are those of
    its own segments file. Identity: >= 100 exact fixes shared with any one individual of another species, any partition.
    Val-test: >= 100 shared with any one validation bird of the same species.
    """
    own = fixes[fixes["stem"] == fixes["species"]]
    test = own[own["species"].isin(birds) & (own["partition"] == "test")]
    val = own[own["species"].isin(birds) & (own["partition"] == "val")]

    identity = test.merge(fixes, on=D.FIX, suffixes=("", "_other"))
    identity = identity[identity["species"] != identity["species_other"]]
    identity = identity.groupby(["species", "individual", "stem_other", "individual_other"]).size().rename("n")
    identity = identity[identity >= SHARED_AT].reset_index()
    val_test = test.merge(val, on=["species", *D.FIX], suffixes=("", "_val"))
    val_test = val_test.groupby(["species", "individual", "individual_val"]).size().rename("n")
    val_test = val_test[val_test >= SHARED_AT].reset_index()

    out = {}
    for bird in birds:
        test_ids = sorted(test.loc[test["species"] == bird, "individual"].unique())
        hits_identity = identity[identity["species"] == bird]
        hits_val = val_test[val_test["species"] == bird]
        dropped = sorted(set(hits_identity["individual"]) | set(hits_val["individual"]))
        out[bird] = {
            "test_birds": len(test_ids),
            "identity": [{"test": r.individual, "other": f"{r.stem_other}:{r.individual_other}", "shared": int(r.n)}
                         for r in hits_identity.itertuples()],
            "val_test": [{"test": r.individual, "val": r.individual_val, "shared": int(r.n)}
                         for r in hits_val.itertuples()],
            "dropped": dropped,
            "scored": len(test_ids) - len(dropped),
        }
    return out


def scored_mismatches(drops: dict[str, dict]) -> list[str]:
    """Every bird whose (test birds, scored) differs from freeze sec 4, with the dropped ids."""
    out = []
    for bird, entry in drops.items():
        expected = SCORED_IN_FREEZE.get(bird, (10, 10))
        got = (entry["test_birds"], entry["scored"])
        if got != expected:
            out.append(f"{bird}: {got[1]} of {got[0]} scored, freeze sec 4 says {expected[1]} of {expected[0]}; "
                       f"dropped {entry['dropped']}")
    return out


def test_window_counts(bird: str) -> dict[str, int]:
    """Windows per test individual at C = 12, counts only."""
    counts = SPW.window_counts_per_individual(
        str(SPW.PROCESSED / f"{bird}_segments.parquet"), context_length=CONTEXT_LENGTH
    )
    per_individual = counts.assign(key=counts["individual_id"].map(id_key)).set_index("key")["n_windows"]
    split = load_split(SPW.SPLITS / f"{SPW.SPLIT_STEM.get(bird, bird)}_split_seed42.json")
    return {id_key(i): int(per_individual.get(id_key(i), 0)) for i in split.test}


def eligibility(drops: dict[str, dict], windows: dict[str, dict[str, int]]) -> dict[str, dict]:
    """Per bird, the scored test birds below F = 25 and the smallest window count among them."""
    out = {}
    for bird, entry in drops.items():
        scored = {key: n for key, n in windows[bird].items() if key not in set(entry["dropped"])}
        out[bird] = {
            "scored": len(scored),
            "below_floor": sorted(key for key, n in scored.items() if n < FLOOR),
            "min_windows": min(scored.values()) if scored else None,
        }
    return out


def scoring_state(raise_on_mismatch: bool) -> tuple[dict, list[str], dict]:
    drops = rule_drops(load_corpus(), BIRDS)
    mismatches = scored_mismatches(drops)
    if mismatches and raise_on_mismatch:
        raise SystemExit("FAIL: scoring rules disagree with freeze sec 4\n  " + "\n  ".join(mismatches))
    windows = {bird: test_window_counts(bird) for bird in BIRDS}
    return drops, mismatches, eligibility(drops, windows)


# Prediction
def test_frame(species: str) -> pd.DataFrame:
    """Read one rotation bird's test partition for scoring. Called from predict_bird, after assert_sealed."""
    if species not in ROTATION:
        raise SystemExit(f"FAIL: {species} is not a rotation held-out bird; fold 0 is never re-predicted")
    stem, split_stem, _ = RT.rotation_attached_entry(species)
    split = load_split(SPW.SPLITS / f"{split_stem}_split_seed42.json")
    frame = load_split_frame(SPW.PROCESSED / f"{stem}_segments.parquet", split.test)
    return frame[["individual_id", "bin_index", "timestamp", "lon", "lat"]].reset_index(drop=True)


def own_records(species: str, records: list[dict]) -> list[dict]:
    """The bird's own specialists and its own fold's pooled arms; never a model whose pool held this bird."""
    fold = RT.FOLD_OF[species]
    return [
        record for record in records
        if (record.get("species") == species and record.get("fold") == fold)
        or (record.get("species") is None and record.get("fold") == fold)
    ]


def prediction_path(species: str, tag: str) -> Path:
    return PREDICTIONS / f"{species}__{tag}.parquet"


def predict_bird(species: str, records: list[dict], device: torch.device, overwrite: bool) -> list[str]:
    frame = test_frame(species)
    print(f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
    written: list[str] = []
    jobs: list[tuple[str, object]] = [("P", next(model for tag, _, model in S2.comparator_jobs("", {}) if tag == "P"))]
    jobs += [(record["run_id"], record) for record in own_records(species, records)]
    for tag, job in jobs:
        out = prediction_path(species, tag)
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            written.append(out.name)
            continue
        started = time.time()
        if tag == "P":
            results = evaluate(job, frame, context_length=CONTEXT_LENGTH, show_progress=False)
        else:
            if RT.ARMS[job["arm"]].get("xgb"):
                from animal_movement_fm.models.xgboost import WindowXGBoost

                predictor = WindowModelPredictor(WindowXGBoost.load(RT.RESULTS / f"{tag}_xgb"), name=tag,
                                                 sigma=SCALE_FLOOR_M)
            else:
                loaded = E3.load_reported(RT.RESULTS, tag)
                if loaded is None:
                    raise SystemExit(f"FAIL: {tag} has no reported checkpoint")
                predictor = MovementFMPredictor(loaded[0], device=device, sigma=SCALE_FLOOR_M)
            results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
        results.to_parquet(out, index=False)
        written.append(out.name)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
    return written


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    payload = RT.load_registered_folds()
    records = read_records()
    rmap = record_map(records)
    gap = completeness(rmap)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    exclusions = assert_training_exclusions(payload, rmap)
    RT.check_boundaries()
    BE.check_boundaries()
    drops, _, floor = scoring_state(raise_on_mismatch=True)
    below = {bird: entry["below_floor"] for bird, entry in floor.items() if entry["below_floor"]}
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = {"attestation": attestation, "completeness": gap, "training_exclusions": exclusions,
                "scoring_rules": drops, "eligibility": floor, "below_floor": below, "species": {}}
    for species in ROTATION:
        manifest["species"][species] = predict_bird(species, records, device, overwrite)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def parse_tag(tag: str, records: dict[str, dict]) -> dict | None:
    if tag in records:
        record = records[tag]
        return {"arm": record["arm"], "seed": record["seed"], "k": None, "draw": None}
    if tag == "P":
        return {"arm": "P", "seed": None, "k": None, "draw": None}
    print(f"  ignored {tag}: not a rotation run or P")
    return None


def rotation_frames(species: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """P and the bird's own predictions, all aligned to P; a prediction from another fold's model is refused."""
    persistence_path = prediction_path(species, "P")
    if not persistence_path.exists():
        raise SystemExit(f"FAIL: {persistence_path} is missing; P defines the strata and the skill denominator")
    persistence = E3.load_frame(persistence_path)
    records = record_map()
    frames = {path.stem.split("__", 1)[1]: E3.load_frame(path) for path in sorted(PREDICTIONS.glob(f"{species}__*.parquet"))}
    allowed = {record["run_id"] for record in own_records(species, list(records.values()))} | {"P"}
    strays = sorted(set(frames) - allowed)
    if strays:
        raise SystemExit(f"FAIL: {species} holds predictions from models outside its own fold: {strays}")
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{species} {tag}")
    return persistence, frames, records


def fold0_frames(species: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """The bird study's sealed predictions (and the gull's carried ones), restricted to the rotation's arms."""
    persistence, frames, records = BE.species_frames(species)
    with contextlib.redirect_stdout(io.StringIO()):
        kept = {tag: frame for tag, frame in frames.items()
                if (parsed := BE.parse_tag(tag, records)) is not None and parsed["arm"] in ARMS}
    return persistence, kept, records


def drop_scored(persistence: pd.DataFrame, frames: dict[str, pd.DataFrame], dropped) -> tuple:
    """Remove the dropped test birds from P and every arm alike; the frames stay aligned."""
    keep = ~persistence["individual_id"].map(id_key).isin({id_key(i) for i in dropped}).to_numpy()
    return (persistence.loc[keep].reset_index(drop=True),
            {tag: frame.loc[keep].reset_index(drop=True) for tag, frame in frames.items()})


def seeds_up_to(frames: dict, records: dict, parse, highest: int) -> dict:
    """Frames whose seed is at most `highest`, or unseeded (P)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return {tag: frame for tag, frame in frames.items()
                if (parsed := parse(tag, records)) is not None and (parsed["seed"] is None or parsed["seed"] <= highest)}


# Statistic
class Resampler:
    """
    The registered paired bootstrap on travel p90 (the draws `E3.paired_delta` makes), with P on every resample too, so
    each comparison gets a skill-unit interval beside its metre one.
    """

    def __init__(self, persistence: pd.DataFrame, boundary: float):
        keep = S2.eligible_mask(persistence, FLOOR)
        self.selected = keep & S2.stratum_masks(persistence, boundary)["travel"]
        ids = persistence.loc[self.selected, "individual_id"].to_numpy()
        unique = np.unique(ids)
        self.n_individuals = int(unique.size)
        self.rows = [np.flatnonzero(ids == value) for value in unique]
        self.draws = (
            np.random.default_rng(S2.BOOTSTRAP_SEED).integers(0, unique.size, size=(S2.N_BOOTSTRAP, unique.size))
            if unique.size >= 2 else None
        )
        self._cache: dict[str, np.ndarray] = {}
        self.reference = self.p90("P", persistence) if self.draws is not None else None

    def p90(self, label: str, frame: pd.DataFrame) -> np.ndarray:
        if label not in self._cache:
            errors = frame["error_m"].to_numpy(float)[self.selected]
            blocks = [errors[rows] for rows in self.rows]
            self._cache[label] = np.array([
                S2._weighted_quantile_from_blocks([blocks[index] for index in draw], S2.QUANTILES["p90"])
                for draw in self.draws
            ])
        return self._cache[label]

    def skill_interval(self, left: str, a: pd.DataFrame, right: str, b: pd.DataFrame) -> dict:
        if self.draws is None:
            return {"error": "fewer than two eligible individuals in travel"}
        la, lb = self.p90(left, a), self.p90(right, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            sa, sb = 1.0 - la / self.reference, 1.0 - lb / self.reference
        gap = sa - sb

        def interval(values):
            return {"lo": float(np.nanpercentile(values, 2.5)), "hi": float(np.nanpercentile(values, 97.5))}

        skill_gap = interval(gap)
        return {
            "n_individuals": self.n_individuals,
            "n_resamples": S2.N_BOOTSTRAP,
            "seed": S2.BOOTSTRAP_SEED,
            "skill": {left: interval(sa), right: interval(sb)},
            "skill_difference": {"label": f"{left} - {right}", **skill_gap,
                                 "excludes_zero": bool(skill_gap["lo"] > 0 or skill_gap["hi"] < 0)},
            "difference_m": {"label": f"{left} - {right}", **interval(la - lb)},
        }


def arm_blocks(summaries: dict) -> dict[str, dict]:
    persistence_headline = {
        stratum: {name: S2.headline(summaries["P"]["by_seed"], stratum, name)[0] for name in E3.STATISTICS}
        for stratum in (*E3.STRATA, "all")
    }
    arms = {}
    for arm, entry in summaries.items():
        block = E3.headline_block(entry, persistence_headline)
        block["tags"] = {E3.tag_key(key): tag for key, tag in entry["tags"].items()}
        arms[arm] = block
    return arms


def summarise_bird(species, persistence, frames, records, parse, boundary, neural_seeds: int) -> dict:
    E4.assert_no_collisions(frames, records, parse=parse)
    summaries = E3.summarise_all(frames, persistence, boundary, records, parse=parse)
    problems = [arm for arm in ARMS if arm not in summaries]
    for arm in ARMS[1:]:
        want = 1 if arm.startswith("XGB") else neural_seeds
        seeds = [key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0]
        if arm in summaries and len(seeds) != want:
            problems.append(f"{arm} has {len(seeds)} seeds, expected {want}")
    if problems:
        raise SystemExit(f"FAIL: {species}: " + "; ".join(problems))
    return arm_blocks(summaries)


def compare(left, right, arms, frames, persistence, boundary, resampler: Resampler) -> dict:
    out = E3.compare_arms(left, right, arms, frames, persistence, boundary)
    a, b = E3.headline_frame(arms, frames, left), E3.headline_frame(arms, frames, right)
    if a is not None and b is not None:
        out["skill_bootstrap"] = resampler.skill_interval(left, a, right, b)
    return out


def bird_readings(arms, frames, persistence, boundary) -> dict:
    """Freeze sec 5, per bird: the two family gaps, the pooled arms and Zb against P, and Zb - Bb."""
    resampler = Resampler(persistence, boundary)
    return {
        "gaps": {family: compare(pooled, specialist, arms, frames, persistence, boundary, resampler)
                 for family, (pooled, specialist) in FAMILIES.items()},
        "against_p": {arm: compare(arm, "P", arms, frames, persistence, boundary, resampler) for arm in AGAINST_P},
        "zb_minus_bb": compare("Zb", "Bb", arms, frames, persistence, boundary, resampler),
    }


def point_readings(arms: dict) -> dict:
    """The same readings as numbers only, for the descriptive fold-0 seeds 0-2 column."""
    skill = {arm: E3.travel_skill(arms, arm) for arm in ARMS[1:]}
    return {
        "skill": skill,
        "gaps": {family: skill[pooled] - skill[specialist] for family, (pooled, specialist) in FAMILIES.items()},
        "zb_minus_bb": skill["Zb"] - skill["Bb"],
    }


def band(gap: float) -> str:
    if gap is None or not np.isfinite(gap):
        return "not scored"
    return "within" if not E3.resolved(gap) else ("above" if gap > 0 else "below")


def band_counts(report: dict) -> dict:
    """Gaps above +0.05 / within / below -0.05, per family, over the 30 birds and over the 27 outside fold 0."""
    out = {}
    for family in FAMILIES:
        for label, birds in (("30", BIRDS), ("27", ROTATION)):
            counts = Counter(band(report["species"][bird]["readings"]["gaps"][family].get("skill_difference"))
                             for bird in birds)
            out[f"{family}_{label}"] = {key: counts.get(key, 0) for key in ("above", "within", "below", "not scored")}
    return out


def spearman(xs, ys) -> dict:
    from scipy.stats import spearmanr

    pairs = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    rho = float(spearmanr(*zip(*pairs)).statistic) if len(pairs) > 2 else float("nan")
    return {"n": len(pairs), "rho": rho}


def spearman_windows(report: dict, counts: dict[str, int]) -> dict:
    """Each family's gap against its specialist's training windows (registered, after exclusions; fold 0 as trained)."""
    return {
        family: spearman([counts[bird] for bird in BIRDS],
                         [report["species"][bird]["readings"]["gaps"][family].get("skill_difference", float("nan"))
                          for bird in BIRDS])
        for family in FAMILIES
    }


# Report
def build_report() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    payload = RT.load_registered_folds()
    boundaries = RT.check_boundaries() | BE.check_boundaries()
    report: dict = {
        "study": "birds_rotation",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "evaluation_floor": FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "completeness": completeness(record_map()),
        "scoring_rules": manifest["scoring_rules"],
        "species": {},
    }
    for species in BIRDS:
        dropped = manifest["scoring_rules"][species]["dropped"]
        boundary = boundaries[species]
        if species in FOLD0:
            if dropped:
                raise SystemExit(f"FAIL: the scoring rules drop fold-0 birds {dropped}; freeze sec 4 says none")
            persistence, frames, records = fold0_frames(species)
            parse, neural_seeds = BE.parse_tag, 5
        else:
            persistence, frames, records = rotation_frames(species)
            parse, neural_seeds = parse_tag, 3
        persistence, frames = drop_scored(persistence, frames, dropped)
        arms = summarise_bird(species, persistence, frames, records, parse, boundary, neural_seeds)
        entry = {
            "fold": FOLD_OF[species],
            "regime_boundary_m": boundary,
            "n_test_individuals_scored": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "dropped": dropped,
            "arms": arms,
            "readings": bird_readings(arms, frames, persistence, boundary),
        }
        if species in FOLD0:
            early = seeds_up_to(frames, records, parse, 2)
            entry["seeds_0_2"] = point_readings(summarise_bird(species, persistence, early, records, parse, boundary, 3))
        report["species"][species] = entry
    report["band_counts"] = band_counts(report)
    report["spearman_gap_vs_training_windows"] = spearman_windows(report, payload["train_window_counts"])
    return report


def _s(value) -> str:
    return f"{value:+.4f}" if value is not None and np.isfinite(value) else "—"


def _interval(block: dict | None, unit: str = "skill") -> str:
    if not block or "lo" not in block:
        return "—"
    if unit == "m":
        return f"[{block['lo']:,.0f}, {block['hi']:,.0f}]"
    return f"[{block['lo']:+.4f}, {block['hi']:+.4f}]"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Rotating held-out birds: sealed evaluation (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"Runs: {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean. Travel p90 skill at "
              f"F = {report['evaluation_floor']}, median of seeds; paired bootstrap 2,000 at seed 42.", ""]

    lines += ["## Pooled − specialist (skill; skill interval; metre interval)", "",
              "| fold | bird | scored | LSTM-Bb − LSTM-S | skill | m | XGB-Bb − XGB-S | skill | m |",
              "|---:|---|---:|---:|---|---|---:|---|---|"]
    for bird, entry in report["species"].items():
        cells = []
        for family in FAMILIES:
            comparison = entry["readings"]["gaps"][family]
            boot = comparison.get("skill_bootstrap") or {}
            cells += [_s(comparison.get("skill_difference")), _interval(boot.get("skill_difference")),
                      _interval(boot.get("difference_m"), "m")]
        lines.append(f"| {entry['fold']} | {short(bird)} | {entry['n_test_individuals_scored']} | " + " | ".join(cells) + " |")

    lines += ["", "## Against P, and Zb − Bb (skill [skill interval])", "",
              "| fold | bird | Bb | LSTM-Bb | XGB-Bb | Zb | Zb − Bb |", "|---:|---|---|---|---|---|---|"]
    for bird, entry in report["species"].items():
        readings = entry["readings"]
        cells = []
        for arm in AGAINST_P:
            comparison = readings["against_p"][arm]
            boot = (comparison.get("skill_bootstrap") or {}).get("skill", {}).get(arm)
            cells.append(f"{_s(comparison.get('skill_difference'))} {_interval(boot)}")
        package = readings["zb_minus_bb"]
        cells.append(f"{_s(package.get('skill_difference'))} "
                     f"{_interval((package.get('skill_bootstrap') or {}).get('skill_difference'))}")
        lines.append(f"| {entry['fold']} | {short(bird)} | " + " | ".join(cells) + " |")

    lines += ["", "## Gap counts", "", "| family | birds | above +0.05 | within | below −0.05 | not scored |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, counts in report["band_counts"].items():
        family, birds = key.split("_")
        lines.append(f"| {family} | {birds} | {counts['above']} | {counts['within']} | {counts['below']} | "
                     f"{counts['not scored']} |")

    lines += ["", "## Spearman: gap against the specialist's training windows", "", "| family | n | ρ |", "|---|---:|---:|"]
    for family, value in report["spearman_gap_vs_training_windows"].items():
        lines.append(f"| {family} | {value['n']} | {_s(value['rho'])} |")

    lines += ["", "## Fold 0 at seeds 0–2 (descriptive)", "",
              "| bird | LSTM-Bb − LSTM-S | XGB-Bb − XGB-S | Bb | LSTM-Bb | XGB-Bb | Zb | Zb − Bb |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for bird in FOLD0:
        early = report["species"][bird]["seeds_0_2"]
        lines.append(f"| {short(bird)} | {_s(early['gaps']['LSTM'])} | {_s(early['gaps']['XGB'])} | "
                     + " | ".join(_s(early["skill"][arm]) for arm in AGAINST_P) + f" | {_s(early['zb_minus_bb'])} |")
    path.write_text("\n".join(lines) + "\n")


# Post-seal join
def rbar_join(report: dict, table: pd.DataFrame) -> dict:
    """Skill against Rbar_corr over the 30 birds, for LSTM-S, XGB-S, LSTM-Bb and XGB-Bb (freeze sec 5)."""
    rbar = table.set_index("species")["rbar_corr"]
    keys = {bird: SPW.TIER4_PROFILE_KEY.get(bird, bird) for bird in BIRDS}
    missing = sorted(key for key in keys.values() if key not in rbar.index)
    if missing:
        raise SystemExit(f"FAIL: {RANDOMNESS_CSV.name} has no row for {missing}")
    values = {bird: float(rbar[keys[bird]]) for bird in BIRDS}
    out = {"n_birds": len(BIRDS), "rbar_corr": values, "rho": {}}
    for arm in ("LSTMS", "XGBS", "LSTMBb", "XGBBb"):
        skills = [E3.travel_skill(report["species"][bird]["arms"], arm) for bird in BIRDS]
        out["rho"][arm] = spearman([values[bird] for bird in BIRDS], skills)
    return out


def write_join(join: dict) -> None:
    JOIN_JSON.write_text(json.dumps(join, indent=2) + "\n")
    lines = ["# Rotating held-out birds: skill against Rbar_corr", "", "| arm | n | ρ |", "|---|---:|---:|"]
    lines += [f"| {arm} | {value['n']} | {_s(value['rho'])} |" for arm, value in join["rho"].items()]
    JOIN_MD.write_text("\n".join(lines) + "\n")


# Diagnostic: training records, registered files, and test-bird fixes and window counts for ids and counts only
def section7_markdown(gap: dict, exclusions: str, drops: dict, mismatches: list[str], floor: dict) -> str:
    neural_clean = sum(row["clean"] for arm, row in gap["by_arm"].items() if not RT.ARMS[arm].get("xgb"))
    xgb_clean = sum(row["clean"] for arm, row in gap["by_arm"].items() if RT.ARMS[arm].get("xgb"))
    identity = "; ".join(f"{short(b)} {', '.join(sorted({h['test'] for h in e['identity']}))}"
                         for b, e in drops.items() if e["identity"]) or "none"
    val_test = ", ".join(f"{short(b)} {len({h['test'] for h in e['val_test']})}"
                         for b, e in drops.items() if e["val_test"]) or "none"
    below = sum(len(entry["below_floor"]) for entry in floor.values())
    smallest = min(entry["min_windows"] for entry in floor.values() if entry["min_windows"] is not None)
    lines = [
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']}; exit status {AS_TRAINED_FROM_PBS['exit status']}; "
        f"{AS_TRAINED_FROM_PBS['queue and precision']} |",
        f"| neural ({gap['n_neural']}) | {neural_clean} clean: {RT.REGISTERED_STEPS:,} of {RT.REGISTERED_STEPS:,} "
        "updates applied, none withheld, no non-finite steps |",
        f"| XGBoost ({gap['n_xgb']}) | {xgb_clean} fit once, boosters present |",
        f"| compute | {AS_TRAINED_FROM_PBS['compute']} |",
        f"| training exclusions | {exclusions} |",
        f"| identity rule | {identity} |",
        f"| val–test rule | {val_test} |",
        f"| scored test birds | {'as sec 4' if not mismatches else 'differ from sec 4: ' + '; '.join(mismatches)} |",
        f"| F = 25 | {below} scored test birds below 25 windows; smallest {smallest} |",
        "| test-partition reads | the diagnostic read test-bird fixes and window counts (ids and counts only) before this "
        "section was committed, as sec 4 allows |",
        f"| deferred or missing | {'none' if gap['complete'] else '; '.join(gap['problems'])} |",
    ]
    return "\n".join(lines)


def diagnostic() -> None:
    """Completeness, exclusions, the scoring rules and F = 25, and the sec 7 block. Scores nothing."""
    freeze_committed()  # committed and clean before any test-bird fixes are read
    text = committed_freeze()
    for number in (6, 7):
        try:
            check_filled(number, section_body(text, number))
            print(f"sec {number}            : filled")
        except SystemExit as reason:
            print(f"sec {number}            : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    payload = RT.load_registered_folds()
    rmap = record_map()
    gap = completeness(rmap)
    exclusions, exclusions_ok = BE.run_check(assert_training_exclusions, payload, rmap)
    drops, mismatches, floor = scoring_state(raise_on_mismatch=False)

    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<7} {row['records']:>2} of {row['planned']:>2}, {row['clean']:>2} clean")
    for problem in gap["problems"]:
        print(f"  {problem}")
    print(f"exclusions       : {exclusions}")
    print("\nscoring rules (test birds / scored; dropped ids)")
    for bird, entry in drops.items():
        marks = [f"identity {h['test']} ({h['other']}, {h['shared']})" for h in entry["identity"]]
        marks += [f"val-test {h['test']} (val {h['val']}, {h['shared']})" for h in entry["val_test"]]
        print(f"  {bird:<26} fold {FOLD_OF[bird]}  {entry['scored']:>3} of {entry['test_birds']:>3}"
              + (f"  {'; '.join(marks)}" if marks else ""))
    print("\n" + ("scored counts match freeze sec 4" if not mismatches else "MISMATCH with freeze sec 4:\n  "
                                                                            + "\n  ".join(mismatches)))
    print("\nF = 25 (scored test birds below the floor; smallest window count)")
    for bird, entry in floor.items():
        print(f"  {bird:<26} {len(entry['below_floor'])} below; smallest {entry['min_windows']}"
              + (f"  {entry['below_floor']}" if entry["below_floor"] else ""))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(gap, exclusions, drops, mismatches, floor))
    if not (gap["complete"] and exclusions_ok):
        raise SystemExit("FAIL: the diagnostic found a problem in the runs; sec 7 must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score every arm on test")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score arms that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true",
                        help="runs, exclusions, scoring rules, F = 25 and the sec 7 block; scores nothing")
    parser.add_argument("--join", action="store_true", help="post-seal: skill against Rbar_corr from the report")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic or args.join):
        raise SystemExit("pass --diagnostic, --predict, --report, --join, or a combination")

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
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}")
    if args.join:
        if not REPORT_JSON.exists():
            raise SystemExit(f"FAIL: {REPORT_JSON} does not exist; run --report first")
        join = rbar_join(json.loads(REPORT_JSON.read_text()), pd.read_csv(RANDOMNESS_CSV))
        write_join(join)
        print(f"wrote {JOIN_JSON}\nwrote {JOIN_MD}")


if __name__ == "__main__":
    main()
