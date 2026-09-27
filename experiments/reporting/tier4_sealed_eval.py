# GenAI reference: much like our previous sealed eval scripts, I specified what was required, gave GenAI the required files
# so it could draft this for me to save time. I have reviewed the file.
"""
Sealed Tier 4 evaluation on the three held-out species (`tier4_exploratory_freeze.md` sec 3-4).

Recall the point of this Tier is xxploratory; so no number this produces may be reported as a tiered result (`preregistration_tier3.md` sec 6).

Tier 2 and Tier 3 are never rerun, and nothing here opens test data before `tier4_exploratory_freeze.md` is committed with its sec 7 "As trained" 
filled in.

Certain comments and docs I have added or modified, others I have left unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import contextlib
import io
import re
import subprocess
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

import tier2_final_train as T  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402
import tier3_sealed_eval as E3  # noqa: E402
import tier4_train as T4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import (  # noqa: E402
    MovementFMPredictor,
    WindowModelPredictor,
    evaluate_batched,
)

TIER4_RESULTS = REPO / "results" / "tier4"
SEALED = REPO / "results" / "tier4_sealed"
PREDICTIONS = SEALED / "predictions"
TIER3_PREDICTIONS = REPO / "results" / "tier3_sealed" / "predictions"
TIER2_PREDICTIONS = REPO / "results" / "tier2_sealed" / "predictions"
FREEZE = REPO / "tier4_exploratory_freeze.md"
AS_TRAINED_HEADING = "As trained"  # matched by text, not by number, so renumbering cannot disable the check

REPORT_JSON = SEALED / "tier4_results.json"
REPORT_MD = SEALED / "tier4_results.md"
MANIFEST = SEALED / "tier4_sealed_manifest.json"

BANNER = (
    "**Exploratory — not a tiered result.** Tier 4 is exploratory under `preregistration_tier3.md` sec 6. "
    "No number in this report may be reported as a tiered result."
)

FLOOR = E3.FLOOR
STRATA = E3.STRATA
STATISTICS = E3.STATISTICS
NOISE_SCALE = E3.NOISE_SCALE

# Carried comparators, tier4_exploratory_freeze.md line 33. Nothing here is retrained or rescored.
CARRIED_TIER2 = ("P", "V", "Z", "Za", "B", "S")
CARRIED_TIER3 = (
    "ARIMA", "AVG3", "AVG10", "SB", "E1embed_dom",
    "Z9", "B9", "Za9", "Zd3", "Zd1", "B9dom",
    "LSTMS", "LSTMB3", "LSTMB9", "XGBS", "XGBB3", "XGBB9",
    "Sk", "F9", "F3c",
)
# longest first, so E1embed_dom is not split as E1embed
CARRIED_ARM_NAMES = tuple(sorted(set(CARRIED_TIER2) | set(CARRIED_TIER3), key=len, reverse=True))

# A5 floor sensitivity, descriptive only; the freeze does not register it.
FLOOR_SENSITIVITY_ARMS = ("Z26", "B26", "Za26", "ZdM", "ZdMnoMir", "BdM", "BdMnoMir")

# The dilution diagnostic of sec 4, and the pool each arm drew from.
DILUTION_ARMS = ("Z26", "B26", "ZdM", "BdM", "ZdMnoMir", "BdMnoMir")

NES = "mirounga_angustirostris"

ARM_ORDER = (
    "P", "V", "ARIMA", "AVG3", "AVG10",
    "Z", "B", "Za", "S", "SB",
    "Z9", "B9", "Za9", "Zd3", "Zd1", "B9dom", "E1embed_dom",
    "Z26", "B26", "Za26", "ZdF", "ZdT", "ZdM", "BdM", "ZdMnoMir", "BdMnoMir", "B26dom",
    "LSTMS", "LSTMB3", "LSTMB9", "LSTMB26", "XGBS", "XGBB3", "XGBB9", "XGBB26",
    "Sk", "F3c", "F9", "F26",
)


# Seal checks
def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def as_trained_section(text: str) -> str:
    """Return the body of the freeze's "As trained" section, by heading text rather than number."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("#") and AS_TRAINED_HEADING.lower() in line.lower()),
        None,
    )
    if start is None:
        raise SystemExit(
            f"FAIL: {FREEZE.name} has no '{AS_TRAINED_HEADING}' heading. That section records the runs as trained "
            "and must exist and be filled before a test partition is opened."
        )
    body = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip()


def assert_as_trained_filled() -> str:
    """
    Refuse to open a test partition while the freeze's "As trained" section is still a placeholder.

    Read from the committed blob rather than the working tree, so this gate does not rest on the diff check that
    precedes it; the two conditions the freeze sets are then verified independently.
    """
    shown = git("show", f"HEAD:{FREEZE.name}")
    if shown.returncode != 0:
        raise SystemExit(f"FAIL: cannot read {FREEZE.name} from HEAD, so it is not committed: {shown.stderr.strip()}")
    body = as_trained_section(shown.stdout)
    if not body:
        raise SystemExit(f"FAIL: {FREEZE.name} '{AS_TRAINED_HEADING}' is empty in the committed version.")
    if "to fill" in body.lower():
        raise SystemExit(
            f"FAIL: {FREEZE.name} '{AS_TRAINED_HEADING}' still says TO FILL in the committed version. Fill it with "
            "the runs completed, service units and anything deferred, commit it, then run the sealed evaluation."
        )
    print(f"as trained       : {FREEZE.name} '{AS_TRAINED_HEADING}' filled ({len(body.splitlines())} lines)")
    return body


def assert_sealed() -> dict:
    """Both gates: the freeze is committed and unmodified, and its "As trained" section is filled."""
    S2.FREEZE = FREEZE  # reuse the Tier 2 seal check against the Tier 4 freeze
    attestation = S2.assert_freeze_committed()
    attestation["as_trained_lines"] = len(assert_as_trained_filled().splitlines())
    return attestation


# Loading
def is_tier4(record: dict) -> bool:
    """
    The one predicate for a Tier 4 record, used on every path that reads records.

    Tier 4 records all carry `exploratory: true`, so the tier tag alone decides; filtering on `exploratory`, as the
    Tier 3 evaluator does, would reject every Tier 4 run.
    """
    return str(record.get("tier")) == "4"


def read_records() -> list[dict]:
    """Every Tier 4 result record on disk, with anything from another tier refused."""
    found = []
    for path in sorted(TIER4_RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if not is_tier4(record):
            print(f"  skipped {path.name}: not a Tier 4 record (tier {record.get('tier')!r})")
            continue
        if record.get("arm") not in T4.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record.get('arm')}, which is not a Tier 4 arm")
        found.append(record)
    return found


def record_map() -> dict[str, dict]:
    """Run id -> record, filtered the same way as `read_records`."""
    return {record["run_id"]: record for record in read_records() if "run_id" in record}


def completeness() -> dict:
    """
    Name the planned runs with no record or no checkpoint, so a gap is stated rather than passed over.

    Reporting continues with whatever exists; the gap travels into the report and the manifest.
    """
    records = record_map()
    missing_records, missing_checkpoints = [], []
    for run in T4.PLAN:
        if run.run_id not in records:
            missing_records.append(run.run_id)
            continue
        if T4.ARMS[run.arm].get("xgb"):
            if not list(TIER4_RESULTS.glob(f"{run.run_id}_xgb*")):
                missing_checkpoints.append(run.run_id)
        elif not (TIER4_RESULTS / f"{run.run_id}_reported.pt").exists():
            missing_checkpoints.append(run.run_id)
    gap = {
        "n_planned": len(T4.PLAN),
        "n_records": len(records),
        "missing_records": missing_records,
        "missing_checkpoints": missing_checkpoints,
        "complete": not missing_records and not missing_checkpoints,
    }
    if not gap["complete"]:
        print(
            f"\nNOTE: reporting-completeness gap. {len(missing_records)} of {len(T4.PLAN)} planned runs have no "
            f"record and {len(missing_checkpoints)} have no checkpoint. Reporting continues with what exists; the "
            "gap is recorded in the report and must be stated with its reason."
        )
    return gap


def prediction_path(species: str, tag: str) -> Path:
    return PREDICTIONS / f"{species}__{tag}.parquet"


def split_carried_tag(tag: str) -> tuple[str, str] | None:
    """Split a carried prediction tag into its arm and the remainder, longest arm name first."""
    for arm in CARRIED_ARM_NAMES:
        if tag == arm:
            return arm, ""
        if tag.startswith(f"{arm}_"):
            return arm, tag[len(arm) + 1:]
    return None


def parse_tag(tag: str, records: dict[str, dict]) -> dict | None:
    """Decode a prediction tag into its arm, seed and optional subset draw."""
    if tag in records:  # a Tier 4 run, already filtered to tier "4"
        record = records[tag]
        return {
            "arm": record["arm"], "seed": record["seed"],
            "k": record.get("k_percent"), "draw": record.get("draw"),
        }
    bare = tag.removeprefix("tier2:").removeprefix("tier3:")
    split = split_carried_tag(bare)
    if split is None:
        print(f"  ignored {tag}: no Tier 4 record and not a carried arm")
        return None
    arm, rest = split
    # Tier 3's subset files end in the seed (F9_<species>_k010_d1_s0); without the optional suffix they fell through to
    # the seed branch and every draw landed under seed 0. Corrected 27-09-2026 (GenAI), freeze sec 9.
    subset = re.fullmatch(r"(?:.*_)?k(\d+)_d(\d+)(?:_s(\d+))?", rest)
    if subset:
        seed = subset.group(3)
        return {
            "arm": arm, "seed": int(seed) if seed is not None else None,
            "k": int(subset.group(1)), "draw": int(subset.group(2)),
        }
    seeded = re.fullmatch(r"(?:.*_)?s(\d+)", rest)
    return {
        "arm": arm,
        "seed": int(seeded.group(1)) if seeded else None,
        "k": None,
        "draw": None,
    }


# GenAI reference: the bug I mentioned it helped catch; added the `parse=parse_tag` fix in build_report; job 179924605's report had silently
# dropped every carried Tier 3 arm, so a missing carried comparator now stops the report instead
def missing_carried(summaries: dict, species: str) -> list[tuple[str, str]]:
    """(arm, species) for every carried arm of tier4_exploratory_freeze.md sec 2 with no summary for this species."""
    return [(arm, species) for arm in sorted(set(CARRIED_TIER2) | set(CARRIED_TIER3)) if arm not in summaries]


# The arms trained on (k, draw) subsets. Their registered draws are read off the plans that defined the runs.
SUBSET_ARMS = ("Sk", "F9", "F3c", "F26")


def summary_key(parsed: dict) -> tuple:
    """The slot E3.summarise_all files a decoded tag under: (arm, seed) or (arm, (k, draw)); a shared slot overwrites."""
    if parsed["k"] is None:
        return parsed["arm"], parsed["seed"] if parsed["seed"] is not None else -1
    return parsed["arm"], (parsed["k"], parsed["draw"])


def assert_no_collisions(frames: dict, records: dict, parse=None) -> None:
    """Fail if two prediction tags decode to the same summary slot, where the later file would silently win."""
    # GenAI reference: `parse` added 27-09-2026 so bird_sealed_eval.py can run this check with its own decoder. Left as
    # None, Tier 4's parse_tag is used exactly as before.
    slots: dict[tuple, list[str]] = {}
    with contextlib.redirect_stdout(io.StringIO()):  # summarise_all reports ignored tags itself
        decoded = {tag: (parse or parse_tag)(tag, records) for tag in frames}
    for tag, parsed in decoded.items():
        if parsed is not None:
            slots.setdefault(summary_key(parsed), []).append(tag)
    collisions = {slot: tags for slot, tags in slots.items() if len(tags) > 1}
    if collisions:
        raise SystemExit(
            "FAIL: prediction tags decode to the same summary slot, so one would overwrite another:\n  "
            + "\n  ".join(f"{slot}: {', '.join(tags)}" for slot, tags in sorted(collisions.items(), key=str))
        )


def registered_draws(arm: str, species: str) -> set[tuple[int, int]]:
    """The (k, draw) pairs the plan that trained `arm` registers for `species`: Tier 4's plan for F26, Tier 3's otherwise."""
    plan = T4.PLAN if arm in T4.ARMS else T4.T3.PLAN
    return {(run.k_percent, run.draw) for run in plan if run.arm == arm and run.species == species}


def assert_draws_complete(summaries: dict, species: str) -> None:
    """Fail unless each subset arm holds exactly its registered (k, draw) set for this species."""
    problems = []
    for arm in SUBSET_ARMS:
        expected = registered_draws(arm, species)
        if not expected:
            raise SystemExit(f"FAIL: no registered draws found for {arm} / {species}")
        present = set(summaries.get(arm, {}).get("by_draw", {}))
        if present != expected:
            problems.append(
                f"{arm} / {species}: missing {sorted(expected - present)}, unexpected {sorted(present - expected)}"
            )
    if problems:
        raise SystemExit("FAIL: subset arms do not hold their registered draws:\n  " + "\n  ".join(problems))


def placement_lines(summaries: dict, species: str) -> list[str]:
    """One line per arm: how many seeds and subset draws the report found for it."""
    lines = []
    for arm in sorted(summaries, key=lambda a: (ARM_ORDER.index(a) if a in ARM_ORDER else len(ARM_ORDER), a)):
        entry = summaries[arm]
        seeds = len([key for key in entry["by_seed"] if key >= 0])
        single = "  (one unseeded run)" if -1 in entry["by_seed"] else ""
        lines.append(f"  {species:<24} {arm:<12} seeds {seeds:>2}  draws {len(entry['by_draw']):>2}{single}")
    return lines


def species_frames(species: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """Load one species' persistence frame, the Tier 4 and carried predictions, and the Tier 4 records."""
    persistence_path = TIER2_PREDICTIONS / f"{species}__P.parquet"
    if not persistence_path.exists():
        raise SystemExit(
            f"FAIL: {persistence_path} is missing. P defines the strata and the skill denominator, and Tier 4 reads "
            "Tier 2's sealed persistence run rather than recomputing it."
        )
    persistence = E3.load_frame(persistence_path)

    frames: dict[str, pd.DataFrame] = {}
    for directory, prefix, carried in (
        (TIER2_PREDICTIONS, "tier2:", CARRIED_TIER2),
        (TIER3_PREDICTIONS, "tier3:", CARRIED_TIER3),
    ):
        for path in sorted(directory.glob(f"{species}__*.parquet")):
            tag = path.stem.split("__", 1)[1]
            split = split_carried_tag(tag)
            if split is None or split[0] not in carried:
                continue
            frames[f"{prefix}{tag}"] = E3.load_frame(path)
    if PREDICTIONS.exists():
        for path in sorted(PREDICTIONS.glob(f"{species}__*.parquet")):
            frames[path.stem.split("__", 1)[1]] = E3.load_frame(path)
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{species} {tag}")

    return persistence, frames, record_map()


# Prediction
def predict_species(species: str, records: list[dict], device: torch.device, overwrite: bool) -> list[str]:
    """Score every Tier 4 run on one held-out species' test partition. Carried arms are read, never rescored."""
    _, _, _, domain, _ = T.held_out_entry(species)
    frame = S2.test_frame(species)
    print(f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
    written: list[str] = []

    for record in records:
        attached = record.get("species")
        if attached and attached != species:
            continue
        tag = record["run_id"]
        out = prediction_path(species, tag)
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            written.append(out.name)
            continue
        started = time.time()
        if record.get("model") == "WindowXGBoost":
            from animal_movement_fm.models.xgboost import WindowXGBoost

            model = WindowXGBoost.load(TIER4_RESULTS / f"{tag}_xgb")
            predictor = WindowModelPredictor(model, name=tag, sigma=SCALE_FLOOR_M)
        else:
            loaded = E3.load_reported(TIER4_RESULTS, tag)
            if loaded is None:
                print(f"  {tag}: no reported checkpoint, skipped")
                continue
            model, state = loaded
            row = E3.domain_row(state, domain, tag) if state["config"].get("species_embed_dim") else None
            predictor = MovementFMPredictor(model, device=device, sigma=SCALE_FLOOR_M, species_index=row)
        results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
        results.to_parquet(out, index=False)
        written.append(out.name)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
    return written


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    attestation = assert_sealed()
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    records = read_records()
    print(f"tier 4 results   : {len(records)} of {len(T4.PLAN)} planned runs on disk")
    gap = completeness()

    manifest = {"attestation": attestation, "completeness": gap, "species": {}}
    for species, *_ in T.HELD_OUT:
        manifest["species"][species] = predict_species(species, records, device, overwrite)
    SEALED.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# Readings, sec 3
def delta(arms: dict, left: str, right: str) -> float:
    """Travel-p90 skill difference, left minus right."""
    return E3.travel_skill(arms, left) - E3.travel_skill(arms, right)


def classify_2x2(delta_z: float, delta_b: float) -> str:
    """
    The freeze sec 3.1 reading of the NES 2 x 2.

    Both deltas are `without Mirounga - with Mirounga`, so negative means removing the congener costs skill.
    """
    if not (np.isfinite(delta_z) and np.isfinite(delta_b)):
        return "not scored"
    z_resolved, b_resolved = E3.resolved(delta_z), E3.resolved(delta_b)
    if z_resolved and b_resolved and delta_z < 0 and delta_b < 0:
        return "genus"
    if not z_resolved and not b_resolved:
        return "domain"
    if z_resolved and delta_z < 0 and not b_resolved:
        return "the package and the congener interact"
    if b_resolved and delta_b < 0 and not z_resolved:
        return "the package and the congener interact"
    if z_resolved and b_resolved and delta_z > 0 and delta_b > 0:
        return "removing M. leonina helps under both trunks; unexpected, needs its own explanation"
    return "unresolved"


def classify_package(deltas: dict[str, float]) -> str:
    """The freeze sec 3.2 reading of Z26 against B26. Deltas are Z26 - B26, so positive means Z26 is better."""
    finite = {s: d for s, d in deltas.items() if np.isfinite(d)}
    if not finite:
        return "not scored"
    z_wins = [s for s, d in finite.items() if E3.resolved(d) and d > 0]
    b_wins = [s for s, d in finite.items() if E3.resolved(d) and d < 0]
    if b_wins and not z_wins:
        return "the package does not pay at 26 sources"
    if z_wins and not b_wins:
        return "the package needs pool diversity, not pool size"
    if z_wins and b_wins:
        return "mixed and species-specific; the package verdict is unresolved, not split"
    return "unresolved: the package is neutral at this scale"


def classify_scaling(deltas: dict[str, float]) -> str:
    """The freeze sec 3.3 reading of Z26 against Z9. Deltas are Z26 - Z9."""
    finite = {s: d for s, d in deltas.items() if np.isfinite(d)}
    if not finite:
        return "not scored"
    better = [s for s, d in finite.items() if E3.resolved(d) and d > 0]
    worse = [s for s, d in finite.items() if E3.resolved(d) and d < 0]
    if len(better) >= 2:
        return "scaling works"
    if not better and not worse:
        return "going from 3 to 9 to 26 sources changes nothing"
    if worse and not better:
        return "more sources hurt"
    return "unresolved"


def nes_two_by_two(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    """The package x Mirounga 2 x 2 on NES: the removal effect under each trunk, then the trunks within each pool."""
    delta_z = delta(arms, "ZdMnoMir", "ZdM")
    delta_b = delta(arms, "BdMnoMir", "BdM")
    return {
        "cells": {arm: E3.travel_skill(arms, arm) for arm in ("ZdM", "ZdMnoMir", "BdM", "BdMnoMir")},
        "delta_z": {
            "label": "ZdMnoMir - ZdM",
            "value": delta_z,
            "resolved_against_noise_scale": E3.resolved(delta_z),
            "bootstrap": E3.compare_arms("ZdMnoMir", "ZdM", arms, frames, persistence, boundary).get("bootstrap"),
        },
        "delta_b": {
            "label": "BdMnoMir - BdM",
            "value": delta_b,
            "resolved_against_noise_scale": E3.resolved(delta_b),
            "bootstrap": E3.compare_arms("BdMnoMir", "BdM", arms, frames, persistence, boundary).get("bootstrap"),
        },
        "reading": classify_2x2(delta_z, delta_b),
        "package_within_pool": {
            "with_mirounga": E3.compare_arms("BdM", "ZdM", arms, frames, persistence, boundary),
            "without_mirounga": E3.compare_arms("BdMnoMir", "ZdMnoMir", arms, frames, persistence, boundary),
        },
        "noise_scale": NOISE_SCALE,
    }


def four_way(arms: dict, species: str) -> dict:
    """The pool-composition table: the full pool, the other domains, the same domain, and one same-domain source."""
    same_domain = {"larus_fuscus": "ZdF", "canis_lupus": "ZdT", NES: "ZdM"}[species]
    rows = {
        "Z26": "26 sources, all domains",
        "Za26": "the other domains only",
        same_domain: "the same domain only",
        "Zd1": "Tier 3 carried: one source, same domain",
        "Z9": "Tier 3 carried: nine sources",
        "Zd3": "Tier 3 carried: three sources, same domain",
    }
    return {
        "read_as": rows,
        "travel_p90_skill": {arm: E3.travel_skill(arms, arm) for arm in rows if arm in arms},
        "same_domain_arm": same_domain,
    }


# Dilution diagnostic, sec 4
def loss_shares(record: dict) -> dict[str, float]:
    """Mean per-species share of the batch loss over the last half of a run's logged steps."""
    history = [entry for entry in record.get("loss_shares") or [] if isinstance(entry, dict)]
    if not history:
        return {}
    tail = history[len(history) // 2:]
    names = sorted({name for entry in tail for name in entry})
    return {
        name: float(np.nanmean([float(entry[name]) for entry in tail if name in entry]))
        for name in names
    }


def dilution_table() -> dict:
    """
    Per-species loss shares against A8 draw shares, from the Tier 4 records alone.

    Reads no prediction file and no test data, so it runs before the seal and before any partition is opened.
    """
    payload = json.loads(T4.REGISTERED_POOL.read_text())
    records = [record for record in read_records() if record["arm"] in DILUTION_ARMS]
    out: dict = {"statistic": "median across seeds of the mean share over the last half of the logged steps",
                 "arms": {}}
    for arm in DILUTION_ARMS:
        mine = [record for record in records if record["arm"] == arm]
        if not mine:
            out["arms"][arm] = {"error": "no records on disk"}
            continue
        run = next(r for r in T4.PLAN if r.arm == arm)
        node = T4.pool_node(run)
        weights = payload
        for part in node.split("."):
            weights = weights[part]
        per_seed = [loss_shares(record) for record in mine]
        per_seed = [shares for shares in per_seed if shares]
        if not per_seed:
            out["arms"][arm] = {"error": "records carry no loss_shares", "pool": node}
            continue
        species = sorted({name for shares in per_seed for name in shares})
        out["arms"][arm] = {
            "pool": node,
            "n_seeds": len(per_seed),
            "rows": [
                {
                    "species": name,
                    "draw_share": float(weights.get(name, float("nan"))),
                    "loss_share": float(
                        np.nanmedian([shares[name] for shares in per_seed if name in shares])
                    ),
                }
                for name in species
            ],
        }
        for row in out["arms"][arm]["rows"]:
            row["loss_minus_draw"] = row["loss_share"] - row["draw_share"]
        out["arms"][arm]["rows"].sort(key=lambda row: -row["loss_share"])
    return out


def print_dilution(table: dict) -> None:
    for arm, block in table["arms"].items():
        if "error" in block:
            print(f"\n{arm}: {block['error']}")
            continue
        print(f"\n{arm}  [{block['pool']}]  {block['n_seeds']} seeds")
        print(f"  {'species':<26} {'draw':>8} {'loss':>8} {'loss-draw':>10}")
        for row in block["rows"]:
            print(
                f"  {row['species']:<26} {row['draw_share'] * 100:7.3f}% {row['loss_share'] * 100:7.3f}% "
                f"{row['loss_minus_draw'] * 100:9.3f}pp"
            )


# Report
def ladder_table(report: dict) -> dict:
    """The descriptive ladder, with the Tier 4 rungs added beside the Tier 3 and Tier 2 ones."""
    blocks = {
        "training_free": ("P", "V", "ARIMA", "AVG3", "AVG10"),
        "specialist": ("XGBS", "LSTMS", "SB", "S"),
        "pooled": ("XGBB3", "XGBB9", "XGBB26", "LSTMB3", "LSTMB9", "LSTMB26", "B", "B9", "B26"),
        "foundation_model": ("Z", "Z9", "Z26", "Za9", "Za26", "Zd3", "Zd1", "ZdF", "ZdT", "ZdM", "ZdMnoMir"),
        "domain_label": ("B9dom", "B26dom", "E1embed_dom"),
    }
    out: dict = {"descriptive": True, "statistic": "travel_p90_skill", "no_pass_fail": True, "blocks": {}}
    for block, rungs in blocks.items():
        out["blocks"][block] = {
            rung: {
                species: E3.travel_skill(entry["arms"], rung)
                for species, entry in report["species"].items()
                if rung in entry["arms"]
            }
            for rung in rungs
        }
    return out


def floor_sensitivity(arms: dict, frames: dict, persistence: pd.DataFrame, boundary: float) -> dict:
    """A5 floor sensitivity for the Tier 4 arms. Descriptive; the freeze does not register it."""
    out: dict = {}
    for arm in FLOOR_SENSITIVITY_ARMS:
        frame = E3.headline_frame(arms, frames, arm)
        if frame is None:
            continue
        out[arm] = {
            str(floor): S2.summarise(frame, persistence, boundary, floor)["travel"].get("p90_m")
            for floor in E3.FLOOR_SENSITIVITY
        }
    return out


def across_species(report: dict) -> dict:
    """Collect the per-species readings of sec 3.2 and 3.3 into their freeze classifications."""
    package = {
        species: entry["readings"]["2_package"].get("skill_difference", float("nan"))
        for species, entry in report["species"].items()
    }
    scaling = {
        species: entry["readings"]["3_scaling"].get("skill_difference", float("nan"))
        for species, entry in report["species"].items()
    }
    return {
        "2_package": {
            "statement": "Z26 against B26, travel p90 skill, Z26 minus B26",
            "per_species": package,
            "reading": classify_package(package),
        },
        "3_scaling": {
            "statement": "Z26 against Tier 3's Z9, travel p90 skill, Z26 minus Z9",
            "per_species": scaling,
            "reading": classify_scaling(scaling),
        },
    }


def build_report() -> dict:
    if not PREDICTIONS.exists():
        raise SystemExit(f"FAIL: {PREDICTIONS} does not exist; run --predict first")
    report: dict = {
        "tier": "4",
        "exploratory": True,
        "banner": BANNER,
        "registered_by": "tier4_exploratory_freeze.md",
        "evaluation_floor": FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "success_bands": None,
        "success_bands_note": "Tier 4 is exploratory; the freeze registers readings, not bands, so none are applied.",
        "comparators_read_from": [
            str(TIER2_PREDICTIONS.relative_to(REPO)),
            str(TIER3_PREDICTIONS.relative_to(REPO)),
        ],
        "completeness": completeness(),
        "species": {},
    }
    if MANIFEST.exists():
        report["attestation"] = json.loads(MANIFEST.read_text()).get("attestation")

    for species, *_ in T.HELD_OUT:
        _, _, profile_key, domain, matching_source = T.held_out_entry(species)
        boundary = T.boundary_m(profile_key)
        persistence, frames, records = species_frames(species)
        assert_no_collisions(frames, records)
        summaries = E3.summarise_all(frames, persistence, boundary, records, parse=parse_tag)
        missing = missing_carried(summaries, species)
        if missing:
            raise SystemExit(
                "FAIL: carried comparators the freeze lists (sec 2) have no summary, so readings would be scored "
                "without them:\n  " + "\n  ".join(f"{arm} / {sp}" for arm, sp in missing)
            )
        assert_draws_complete(summaries, species)
        print("\n".join(placement_lines(summaries, species)), flush=True)
        if "P" not in summaries:
            raise SystemExit(f"FAIL: {species} has no persistence summary; P is the skill denominator")
        persistence_headline = {
            stratum: {name: S2.headline(summaries["P"]["by_seed"], stratum, name)[0] for name in STATISTICS}
            for stratum in (*STRATA, "all")
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

        entry = {
            "domain": domain,
            "matching_tier2_source": matching_source,
            "regime_boundary_m": boundary,
            "n_test_individuals": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "scored_arms": sorted(arms),
            "arms": arms,
            "floor_sensitivity": floor_sensitivity(arms, frames, persistence, boundary),
        }
        readings = {
            "2_package": E3.compare_arms("Z26", "B26", arms, frames, persistence, boundary),
            "3_scaling": E3.compare_arms("Z26", "Z9", arms, frames, persistence, boundary),
            "4_four_way_pool": four_way(arms, species),
            "5_few_shot_vs_scratch_1pc": E3.compare_draws("F26", "Sk", 1, arms, frames, persistence, boundary),
            "5_few_shot_vs_scratch_10pc": E3.compare_draws("F26", "Sk", 10, arms, frames, persistence, boundary),
            "5_pool_scaling_few_shot_1pc": E3.compare_draws("F26", "F9", 1, arms, frames, persistence, boundary),
            "5_pool_scaling_few_shot_10pc": E3.compare_draws("F26", "F9", 10, arms, frames, persistence, boundary),
            # k = 100% is F26 against F9 only; Sk has no 100% arm. Descriptive.
            "5_pool_scaling_few_shot_100pc": E3.compare_draws("F26", "F9", 100, arms, frames, persistence, boundary),
        }
        if species == NES:
            readings["1_nes_two_by_two"] = nes_two_by_two(arms, frames, persistence, boundary)
        entry["readings"] = readings
        report["species"][species] = entry

    report["across_species"] = across_species(report)
    report["ladder"] = ladder_table(report)
    report["dilution"] = dilution_table()
    return report


# Markdown
def _s(value) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:+.4f}"


def _interval(block) -> str:
    if not block:
        return "-"
    series = (block.get("series") or {}).get("travel_p90_m")
    if not series or "difference_m" not in series:
        return "-"
    difference = series["difference_m"]
    star = " *" if difference.get("excludes_zero") else ""
    return f"[{difference['lo']:,.0f}, {difference['hi']:,.0f}] m{star}"


def write_markdown(report: dict, path: Path) -> None:
    lines = [
        BANNER,
        "",
        "# Tier 4 sealed evaluation — the 26-source pool",
        "",
        f"Travel-regime p90 skill at floor F = {report['evaluation_floor']}, median across training seeds, paired "
        f"bootstrap of {S2.N_BOOTSTRAP:,} test-individual resamples at seed {S2.BOOTSTRAP_SEED}. A gap smaller than "
        f"the {report['noise_scale']} single-seed noise scale is reported as unresolved. Intervals marked * exclude "
        "zero. No success bands are applied: the freeze registers readings, not bands.",
        "",
        f"Comparators are read from {', '.join(f'`{p}`' for p in report['comparators_read_from'])} and are never "
        "recomputed.",
        "",
    ]

    gap = report["completeness"]
    lines += ["## Completeness", ""]
    if gap["complete"]:
        lines += [f"All {gap['n_planned']} planned runs have a record and a checkpoint.", ""]
    else:
        lines += [
            f"**Reporting-completeness gap.** {gap['n_records']} of {gap['n_planned']} planned runs have a record.",
            "",
            f"- no record: {', '.join(gap['missing_records']) or 'none'}",
            f"- no checkpoint: {', '.join(gap['missing_checkpoints']) or 'none'}",
            "",
        ]

    nes = report["species"].get(NES, {}).get("readings", {}).get("1_nes_two_by_two")
    if nes:
        lines += [
            "## 1. The NES 2 x 2 — package x *Mirounga*",
            "",
            "| cell | travel p90 skill |",
            "|---|---:|",
        ]
        for arm, value in nes["cells"].items():
            lines.append(f"| {arm} | {_s(value)} |")
        lines += [
            "",
            "| delta | value | resolved | interval |",
            "|---|---:|---|---|",
            f"| {nes['delta_z']['label']} | {_s(nes['delta_z']['value'])} | "
            f"{nes['delta_z']['resolved_against_noise_scale']} | {_interval(nes['delta_z']['bootstrap'])} |",
            f"| {nes['delta_b']['label']} | {_s(nes['delta_b']['value'])} | "
            f"{nes['delta_b']['resolved_against_noise_scale']} | {_interval(nes['delta_b']['bootstrap'])} |",
            "",
            f"**Reading: {nes['reading']}.**",
            "",
            "| package on against off | skill difference | resolved | interval |",
            "|---|---:|---|---|",
        ]
        for label, block in nes["package_within_pool"].items():
            lines.append(
                f"| {label} ({block.get('left')} - {block.get('right')}) | "
                f"{_s(block.get('skill_difference'))} | {block.get('resolved_against_noise_scale')} | "
                f"{_interval(block.get('bootstrap'))} |"
            )
        lines.append("")

    for key, title in (
        ("2_package", "2. Z26 against B26 — the package question"),
        ("3_scaling", "3. Z26 against Z9 — scaling"),
    ):
        block = report["across_species"][key]
        lines += [
            f"## {title}",
            "",
            "| species | skill difference | resolved | interval |",
            "|---|---:|---|---|",
        ]
        for species, entry in report["species"].items():
            comparison = entry["readings"][key]
            lines.append(
                f"| {species} | {_s(comparison.get('skill_difference'))} | "
                f"{comparison.get('resolved_against_noise_scale')} | {_interval(comparison.get('bootstrap'))} |"
            )
        lines += ["", f"**Reading: {block['reading']}.**", ""]

    lines += ["## 4. The four-way pool table", "", "| species | arm | pool | travel p90 skill |", "|---|---|---|---:|"]
    for species, entry in report["species"].items():
        table = entry["readings"]["4_four_way_pool"]
        for arm, description in table["read_as"].items():
            if arm in table["travel_p90_skill"]:
                lines.append(f"| {species} | {arm} | {description} | {_s(table['travel_p90_skill'][arm])} |")
    lines.append("")

    lines += [
        "## 5. Few-shot — F26 against Sk and F9, per matching draw",
        "",
        "Each draw index is paired on both sides and carries its own interval; means across draws are descriptive. "
        "k = 100% is F26 against F9 only, because Sk has no 100% arm.",
        "",
        "| species | comparison | k | mean difference (m) | F26 better in |",
        "|---|---|---:|---:|---:|",
    ]
    for species, entry in report["species"].items():
        for key, comparison in entry["readings"].items():
            if not key.startswith("5_"):
                continue
            if "error" in comparison:
                lines.append(f"| {species} | {comparison.get('left')} vs {comparison.get('right')} | "
                             f"{comparison.get('k_percent')} | - | {comparison['error']} |")
                continue
            draws = comparison.get("draws", {})
            values = [d.get("difference_m") for d in draws.values() if d.get("difference_m") is not None]
            better = sum(1 for value in values if value < 0)
            mean = float(np.mean(values)) if values else float("nan")
            lines.append(
                f"| {species} | {comparison['left']} vs {comparison['right']} | {comparison['k_percent']} | "
                f"{'-' if not values else f'{mean:+,.0f}'} | {better} of {len(values)} |"
            )
    lines.append("")

    lines += ["## 6. The ladder (descriptive, no pass/fail)", ""]
    species_names = list(report["species"])
    lines += ["| block | rung | " + " | ".join(species_names) + " |", "|---|---|" + "---:|" * len(species_names)]
    for block, rungs in report["ladder"]["blocks"].items():
        for rung, values in rungs.items():
            if not values:
                continue
            cells = " | ".join(_s(values.get(species)) for species in species_names)
            lines.append(f"| {block} | {rung} | {cells} |")
    lines.append("")

    lines += [
        "## 7. Dilution diagnostic",
        "",
        "Per-species share of the training loss against its A8 draw share, from the Tier 4 records alone. No test "
        "data enters this table.",
        "",
    ]
    for arm, block in report["dilution"]["arms"].items():
        if "error" in block:
            lines += [f"**{arm}**: {block['error']}", ""]
            continue
        lines += [
            f"**{arm}** — `{block['pool']}`, {block['n_seeds']} seeds",
            "",
            "| species | draw share | loss share | loss - draw |",
            "|---|---:|---:|---:|",
        ]
        for row in block["rows"]:
            lines.append(
                f"| {row['species']} | {row['draw_share'] * 100:.3f}% | {row['loss_share'] * 100:.3f}% | "
                f"{row['loss_minus_draw'] * 100:+.3f} pp |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="score the Tier 4 arms on test and save predictions")
    parser.add_argument("--report", action="store_true", help="build the report from the saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score arms that already have prediction files")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="print the dilution loss-share table from the Tier 4 records alone; opens nothing",
    )
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --predict, --report, --diagnostic, or a combination")

    if args.diagnostic:
        completeness()
        print_dilution(dilution_table())
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
        for key, block in report["across_species"].items():
            print(f"{key:<14} {block['reading']}")
        nes = report["species"].get(NES, {}).get("readings", {}).get("1_nes_two_by_two")
        if nes:
            print(f"{'1_nes_2x2':<14} {nes['reading']}")


if __name__ == "__main__":
    main()
