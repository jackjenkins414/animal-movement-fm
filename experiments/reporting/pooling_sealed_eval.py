# GenAI reference: An experiment script, so we used GenAI to generate this file from our outline which was specifiying the seal evaluator;
# thin wrapper over tier4_sealed_eval.py that scores only the 15 pooling runs on test, carries B26 and every comparator from our previous
# Tier 2-4 sealed prediction files, each arm against P and B26, the main effects and their interaction on the same resamples, NES against 
# the named comparators etc. GenAI wrote this reusing our Tier 3, Tier 4 and rotation evaluators.
"""
Sealed evaluation of the pooling study on the three held-out species (`pooling_exploratory_freeze.md` sec 4-5).

Exploratory; no number here is a tiered result. Only the 15 pooling runs are scored on test. B26 and every comparator
are read from the Tier 2, 3 and 4 sealed prediction files and never re-predicted, and nothing opens a test partition
until the freeze is committed, unchanged, with sec 6 and sec 7 filled.
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
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_sealed_eval as RE  # noqa: E402
import pooling_train as PT  # noqa: E402
import tier4_sealed_eval as E4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402
from animal_movement_fm.models.moe import N_EXPERTS  # noqa: E402

T, S2, E3, T4 = E4.T, E4.S2, E4.E3, E4.T4

FREEZE = REPO / "preregistration" / "pooling_exploratory_freeze.md"
SEALED = REPO / "results" / "pooling_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "pooling_results.json"
REPORT_MD = SEALED / "pooling_results.md"
MANIFEST = SEALED / "pooling_sealed_manifest.json"

NES = E4.NES
NEW_ARMS = tuple(PT.ARMS)  # B26bal, B26moe, B26balmoe
MOE_ARMS = tuple(arm for arm, spec in PT.ARMS.items() if spec["trunk"] == "moe")
CARRIED_TIER4 = ("B26", "BdM", "LSTMB26", "XGBB26")  # read from Tier 4's sealed predictions
CARRIED_TIER3 = ("B9",)
# freeze sec 5, as printed there (4 dp); the carried NES skills must reproduce them
NES_COMPARATORS = {"BdM": 0.1457, "B9": 0.1272, "LSTMB26": 0.1228, "XGBB26": 0.1728}
N_PARAMETERS = {"transformer": 2_112_258, "moe": 5_279_490}  # freeze sec 3
POOL_WINDOWS = 2_570_302  # freeze sec 2
CONTROL_THRESHOLD = 0.05  # freeze sec 5: B26wide if B26moe - B26 > +0.05 on any species (point estimate)
# (left minus right) of (B26balmoe - B26bal) - (B26moe - B26)
INTERACTION = (("B26balmoe", 1), ("B26bal", -1), ("B26moe", -1), ("B26", 1))

# From the PBS job logs, as reported on 01-10-2026; none of these is in the run records.
AS_TRAINED_FROM_PBS = {
    "exit status": "0 for every run",
    "queue and precision": "gpuhopper, fp32",
    "compute": "257.18 SU against the ~210–350 SU estimated in sec 3; longest run 00:13:59 (index 5); index 5 alone "
               "13m59s, 20.98 SU",
}

BANNER = "**Exploratory — not a tiered result.** No number in this report may be reported as a Tier 2 or Tier 3 result."


# Seal. 
def section_body(text: str, number: int) -> str:
    """The body of freeze section `number`, found by its number prefix, up to the next `## ` heading."""
    found = re.search(rf"^##\s+{number}\.[^\n]*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    if found is None:
        raise SystemExit(f"FAIL: {FREEZE.name} has no section {number}")
    return found.group(1).strip()


def check_filled(number: int, body: str) -> None:
    if not body:
        raise SystemExit(f"FAIL: {FREEZE.name} sec {number} is empty in the committed version.")
    found = RE.PLACEHOLDER.search(body)
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
            raise SystemExit(f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but "
                             f"{FREEZE.name} is now blob {attestation['freeze_blob'][:12]}.")
    return attestation


# Records. 
def read_records() -> dict[str, dict]:
    found = {}
    for path in sorted(PT.RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("tier") != "pooling":
            print(f"  skipped {path.name}: not a pooling record (tier {record.get('tier')!r})")
            continue
        if record.get("arm") not in PT.ARMS:
            raise SystemExit(f"FAIL: {path.name} names arm {record.get('arm')}, which is not a pooling arm")
        found[record["run_id"]] = record
    return found


def routing_problems(routing: dict | None, n_layers: int, counts: dict[str, int]) -> list[str]:
    """Every source, every layer, every expert, and the registered number of windows per source."""
    if not routing:
        return ["no routing record"]
    problems = []
    if set(routing) != set(counts):
        problems.append(f"routing covers {len(set(routing) & set(counts))} of {len(counts)} sources")
    for name, entry in routing.items():
        want = min(PT.ROUTING_WINDOWS, counts.get(name, 0))
        if entry.get("windows") != want:
            problems.append(f"{name} routed {entry.get('windows')} windows, expected {want}")
        layers = entry.get("layers") or []
        if len(layers) != n_layers:
            problems.append(f"{name} has {len(layers)} routing layers, expected {n_layers}")
        for i, layer in enumerate(layers):
            for key in ("mean_prob", "top_k_share"):
                values = layer.get(key) or []
                if len(values) != N_EXPERTS or abs(sum(values) - 1.0) > 1e-4:
                    problems.append(f"{name} layer {i} {key} is not {N_EXPERTS} shares summing to 1")
    return problems


def run_problems(run, record: dict | None, payload: dict) -> list[str]:
    """What stops one planned run counting as finished cleanly; loads its checkpoint to check the weights."""
    if record is None:
        return ["no record"]
    spec, config = PT.ARMS[run.arm], record.get("config") or {}
    problems = []
    if record.get("applied_updates") != PT.REGISTERED_STEPS or config.get("steps") != PT.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {config.get('steps')}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not np.isfinite(record.get("final_loss", float("nan"))):
        problems.append(f"final loss {record.get('final_loss')}")
    if config.get("trunk") != spec["trunk"]:
        problems.append(f"trunk {config.get('trunk')}, expected {spec['trunk']}")
    if record.get("n_parameters") != N_PARAMETERS[spec["trunk"]]:
        problems.append(f"{record.get('n_parameters')} parameters, expected {N_PARAMETERS[spec['trunk']]:,}")
    if record.get("pool_windows") != POOL_WINDOWS:
        problems.append(f"pool of {record.get('pool_windows')} windows, expected {POOL_WINDOWS:,}")
    weights = PT.balanced_weights(payload) if spec["balanced"] else payload["source_pool"]
    drawn = record.get("draw_probabilities") or {}
    if set(drawn) != set(weights) or any(abs(drawn[n] - weights[n]) > 1e-12 for n in weights):
        problems.append("draw probabilities differ from sec 2")
    try:
        if E3.load_reported(PT.RESULTS, run.run_id) is None:
            problems.append("no reported checkpoint")
    except SystemExit as failure:
        problems.append(str(failure).removeprefix("FAIL: "))
    if spec["trunk"] == "moe":
        problems += routing_problems(record.get("routing"), config.get("num_layers", 0), payload["train_window_counts"])
    return problems


def completeness(records: dict[str, dict], payload: dict) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in PT.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        row["records"] += run.run_id in records
        issues = run_problems(run, records.get(run.run_id), payload)
        row["clean"] += not issues
        problems += [f"{run.run_id}: {'; '.join(issues)}"] if issues else []
    return {
        "n_planned": len(PT.PLAN),
        "n_records": sum(row["records"] for row in by_arm.values()),
        "n_clean": sum(row["clean"] for row in by_arm.values()),
        "by_arm": by_arm,
        "problems": problems,
        "complete": not problems,
    }


def tier4_carried_records() -> dict[str, dict]:
    with contextlib.redirect_stdout(io.StringIO()):
        return {rid: record for rid, record in E4.record_map().items() if record["arm"] in CARRIED_TIER4}


def carried_paths(species: str, tier4: dict[str, dict]) -> dict[str, Path]:
    """Tag -> sealed prediction file for P and every carried comparator of one species. Paths only; nothing is read."""
    paths = {"tier2:P": E4.TIER2_PREDICTIONS / f"{species}__P.parquet"}
    for rid, record in tier4.items():
        if record.get("species") in (None, species):
            paths[rid] = E4.prediction_path(species, rid)
    for path in sorted(E4.TIER3_PREDICTIONS.glob(f"{species}__*.parquet")):
        tag = path.stem.split("__", 1)[1]
        split = E4.split_carried_tag(tag)
        if split and split[0] in CARRIED_TIER3:
            paths[f"tier3:{tag}"] = path
    return paths


# Prediction. 
def predict_species(species: str, device: torch.device, overwrite: bool) -> list[str]:
    """Score the 15 pooling runs on one held-out species' test partition. Nothing carried is re-predicted."""
    frame = S2.test_frame(species)
    print(f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
    written = []
    for run in PT.PLAN:
        out = PREDICTIONS / f"{species}__{run.run_id}.parquet"
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            written.append(out.name)
            continue
        started = time.time()
        loaded = E3.load_reported(PT.RESULTS, run.run_id)
        if loaded is None:
            raise SystemExit(f"FAIL: {run.run_id} has no reported checkpoint")
        predictor = MovementFMPredictor(loaded[0], device=device, sigma=SCALE_FLOOR_M)
        results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
        results.to_parquet(out, index=False)
        written.append(out.name)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
    return written


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    payload = PT.load_registered_pool()
    gap = completeness(read_records(), payload)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = {"attestation": attestation, "completeness": gap, "species": {}}
    for species, *_ in T.HELD_OUT:
        manifest["species"][species] = predict_species(species, device, overwrite)
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames. 
def species_frames(species: str, records: dict[str, dict]):
    """P, the carried comparators and the pooling predictions, all aligned to P, with the records that decode them."""
    tier4 = tier4_carried_records()
    paths = carried_paths(species, tier4) | {
        run.run_id: PREDICTIONS / f"{species}__{run.run_id}.parquet" for run in PT.PLAN
    }
    if not paths["tier2:P"].exists():
        raise SystemExit(f"FAIL: {paths['tier2:P']} is missing; P defines the strata and the skill denominator")
    frames = {tag: E3.load_frame(path) for tag, path in paths.items() if path.exists()}
    persistence = frames["tier2:P"]
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{species} {tag}")
    return persistence, frames, tier4 | records


def check_seeds(species: str, summaries: dict) -> None:
    """Every scored arm holds exactly seeds 0-4; BdM exists on NES only."""
    expected = (*NEW_ARMS, *CARRIED_TIER4, *CARRIED_TIER3)
    problems = []
    for arm in expected:
        if arm == "BdM" and species != NES:
            continue
        seeds = {key for key in summaries.get(arm, {}).get("by_seed", {}) if key >= 0}
        if seeds != set(PT.SEEDS):
            problems.append(f"{arm} has seeds {sorted(seeds)}, expected {list(PT.SEEDS)}")
    if "P" not in summaries:
        problems.append("no P")
    if problems:
        raise SystemExit(f"FAIL: {species}: " + "; ".join(problems))


def carried_check(species: str, arms: dict, tier4_report: dict) -> None:
    """The carried arms reproduce Tier 4's sealed report exactly, and the NES comparators match freeze sec 5."""
    theirs = tier4_report["species"][species]["arms"]
    problems = []
    for arm in (*CARRIED_TIER4, *CARRIED_TIER3):
        if arm not in arms:
            continue
        mine, want = E3.travel_skill(arms, arm), E3.travel_skill(theirs, arm)
        if not np.isclose(mine, want, rtol=1e-12, atol=0.0):
            problems.append(f"{arm} {mine!r}, Tier 4 report {want!r}")
    if species == NES:
        problems += [f"{arm} {E3.travel_skill(arms, arm):+.4f}, freeze {value:+.4f}"
                     for arm, value in NES_COMPARATORS.items() if round(E3.travel_skill(arms, arm), 4) != value]
    if problems:
        raise SystemExit(f"FAIL: {species}: carried skills do not reproduce the sealed values: " + "; ".join(problems))


# Readings, freeze sec 5. 
def _interval(values: np.ndarray) -> dict:
    lo, hi = float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))
    return {"lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)}


def interaction(arms: dict, frames: dict, resampler) -> dict:
    """(B26balmoe - B26bal) - (B26moe - B26), point and intervals on the same resamples as every other comparison."""
    point = sum(sign * E3.travel_skill(arms, arm) for arm, sign in INTERACTION)
    out = {"label": "(B26balmoe - B26bal) - (B26moe - B26)", "skill_difference": point,
           "resolved_against_noise_scale": E3.resolved(point)}
    if resampler.draws is None:
        return out | {"error": "fewer than two eligible individuals in travel"}
    p90 = {arm: resampler.p90(arm, E3.headline_frame(arms, frames, arm)) for arm, _ in INTERACTION}
    with np.errstate(divide="ignore", invalid="ignore"):
        skill = sum(sign * (1.0 - p90[arm] / resampler.reference) for arm, sign in INTERACTION)
    metres = sum(sign * p90[arm] for arm, sign in INTERACTION)
    out["skill_bootstrap"] = {"n_individuals": resampler.n_individuals, "n_resamples": S2.N_BOOTSTRAP,
                              "seed": S2.BOOTSTRAP_SEED, "skill_difference": _interval(skill),
                              "difference_m": _interval(metres)}
    return out


def species_readings(species: str, arms: dict, frames: dict, persistence, boundary: float) -> dict:
    resampler = RE.Resampler(persistence, boundary)

    def compare(left: str, right: str) -> dict:
        return RE.compare(left, right, arms, frames, persistence, boundary, resampler)

    readings = {
        "against_p": {arm: compare(arm, "P") for arm in ("B26", *NEW_ARMS)},
        "against_b26": {arm: compare(arm, "B26") for arm in NEW_ARMS},
        "interaction": interaction(arms, frames, resampler),
    }
    if species == NES:
        readings["nes_comparators"] = {arm: {other: compare(arm, other) for other in NES_COMPARATORS}
                                       for arm in NEW_ARMS}
    return readings


def conditional_control(report: dict) -> dict:
    gaps = {species: entry["readings"]["against_b26"]["B26moe"].get("skill_difference", float("nan"))
            for species, entry in report["species"].items()}
    fires = [species for species, gap in gaps.items() if np.isfinite(gap) and gap > CONTROL_THRESHOLD]
    return {"rule": f"train B26wide if B26moe - B26 > +{CONTROL_THRESHOLD} on any species (point estimate)",
            "per_species": gaps, "fires_on": fires, "fires": bool(fires)}


def mean_pairwise_tv(shares: dict[str, np.ndarray]) -> float:
    """Mean total-variation distance over every pair of domains' expert-share vectors: 0 same routing, 1 disjoint."""
    return float(np.mean([0.5 * np.abs(shares[a] - shares[b]).sum() for a, b in combinations(sorted(shares), 2)]))


def expert_use(records: dict[str, dict]) -> dict:
    """
    Top-k share per expert, per layer, per domain (each domain's sources weighted equally), per seed, from the training
    windows in the routing record. Expert indices mean nothing across seeds, so only the TV summary is aggregated.
    """
    out = {}
    for arm in MOE_ARMS:
        seeds = {}
        for run in (run for run in PT.PLAN if run.arm == arm):
            routing = records[run.run_id]["routing"]
            layers = []
            for i in range(records[run.run_id]["config"]["num_layers"]):
                by_domain = {domain: [entry["layers"][i] for entry in routing.values() if entry["domain"] == domain]
                             for domain in T4.DOMAINS}
                shares = {d: np.mean([layer["top_k_share"] for layer in rows], axis=0) for d, rows in by_domain.items()}
                layers.append({
                    "top_k_share": {d: values.tolist() for d, values in shares.items()},
                    "window_agreement": {d: float(np.mean([layer["window_agreement"] for layer in rows]))
                                         for d, rows in by_domain.items()},
                    "tv_distance": mean_pairwise_tv(shares),
                })
            seeds[run.seed] = layers
        n_layers = len(next(iter(seeds.values())))
        out[arm] = {"seeds": seeds,
                    "median_tv_distance": [float(np.median([seeds[s][i]["tv_distance"] for s in seeds]))
                                           for i in range(n_layers)]}
    return out


def dilution_block(records: list[dict], draw_share: dict[str, float], domains: dict[str, str]) -> dict:
    """Tier 4 sec 4's statistic: per source, the median across seeds of the mean loss share over the last half."""
    per_seed = [shares for shares in (E4.loss_shares(record) for record in records) if shares]
    rows = [{"species": name, "domain": domains[name], "draw_share": float(draw_share[name]),
             "loss_share": float(np.nanmedian([shares[name] for shares in per_seed if name in shares]))}
            for name in sorted(draw_share)]
    raw = {d: sum(r["loss_share"] for r in rows if r["domain"] == d) for d in T4.DOMAINS}
    total = sum(raw.values())
    return {
        "n_seeds": len(per_seed),
        "rows": sorted(rows, key=lambda row: -row["loss_share"]),
        "domains": {d: {"draw_share": sum(r["draw_share"] for r in rows if r["domain"] == d),
                        "loss_share_raw": raw[d], "loss_share_rescaled": raw[d] / total}
                    for d in T4.DOMAINS},
    }


def dilution(records: dict[str, dict], payload: dict) -> dict:
    out = {}
    for arm in NEW_ARMS:
        mine = [records[run.run_id] for run in PT.PLAN if run.arm == arm]
        out[arm] = dilution_block(mine, mine[0]["draw_probabilities"], payload["domains"])
    b26 = [record for record in tier4_carried_records().values() if record["arm"] == "B26"]
    out["B26 (Tier 4 records, reference)"] = dilution_block(b26, payload["source_pool"], payload["domains"])
    return out


# Report. 
def build_report() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    payload = PT.load_registered_pool()
    records = read_records()
    tier4_report = json.loads(E4.REPORT_JSON.read_text())
    report: dict = {
        "study": "pooling",
        "exploratory": True,
        "banner": BANNER,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": json.loads(MANIFEST.read_text()).get("attestation"),
        "evaluation_floor": E3.FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": E3.NOISE_SCALE,
        "persistence_from": str(E4.TIER2_PREDICTIONS.relative_to(REPO)),
        "completeness": completeness(records, payload),
        "species": {},
    }
    for species, *_ in T.HELD_OUT:
        boundary = T.boundary_m(T.held_out_entry(species)[2])
        persistence, frames, rmap = species_frames(species, records)
        E4.assert_no_collisions(frames, rmap)
        with contextlib.redirect_stdout(io.StringIO()):
            summaries = E3.summarise_all(frames, persistence, boundary, rmap, parse=E4.parse_tag)
        check_seeds(species, summaries)
        arms = RE.arm_blocks(summaries)
        carried_check(species, arms, tier4_report)
        report["species"][species] = {
            "regime_boundary_m": boundary,
            "n_test_individuals": int(persistence["individual_id"].nunique()),
            "n_test_windows": int(len(persistence)),
            "arms": arms,
            "readings": species_readings(species, arms, frames, persistence, boundary),
        }
    report["conditional_control"] = conditional_control(report)
    report["expert_use"] = expert_use(records)
    report["dilution"] = dilution(records, payload)
    return report


def _s(value) -> str:
    return f"{value:+.4f}" if value is not None and np.isfinite(value) else "—"


def _row(block: dict) -> str:
    """skill difference | resolved | skill interval | metre interval, from a comparison block."""
    boot = block.get("skill_bootstrap") or {}
    skill, metres = boot.get("skill_difference"), boot.get("difference_m")
    skill_cell = (f"[{skill['lo']:+.4f}, {skill['hi']:+.4f}]{' *' if skill['excludes_zero'] else ''}"
                  if skill else "—")
    metre_cell = f"[{metres['lo']:,.0f}, {metres['hi']:,.0f}]" if metres else "—"
    return (f"{_s(block.get('skill_difference'))} | {block.get('resolved_against_noise_scale')} | {skill_cell} | "
            f"{metre_cell}")


def write_markdown(report: dict, path: Path) -> None:
    head = "| species | comparison | skill difference | resolved | skill interval | metre interval |"
    rule = "|---|---|---:|---|---|---|"
    lines = [BANNER, "", "# Pooling study: sealed evaluation (exploratory)", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    lines += [
        f"Travel p90 skill against persistence at F = {report['evaluation_floor']}, median of five seeds; paired "
        f"bootstrap of {S2.N_BOOTSTRAP:,} resamples at seed {S2.BOOTSTRAP_SEED}, the same resamples for every "
        f"comparison. Differences are left minus right; resolved means |difference| > {report['noise_scale']}. "
        "* marks a skill interval that excludes zero. P is read from Tier 2's sealed prediction files.", "",
    ]
    gap = report["completeness"]
    lines += ["## Completeness", "", f"{gap['n_clean']} of {gap['n_planned']} runs clean."
              + ("" if gap["complete"] else " Problems: " + "; ".join(gap["problems"])), ""]

    lines += ["## 1. Against persistence", "", head, rule]
    for species, entry in report["species"].items():
        lines += [f"| {species} | {arm} − P | {_row(block)} |" for arm, block in entry["readings"]["against_p"].items()]
    lines += ["", "## 2. Against B26", "", head, rule]
    for species, entry in report["species"].items():
        lines += [f"| {species} | {arm} − B26 | {_row(block)} |"
                  for arm, block in entry["readings"]["against_b26"].items()]
    lines += ["", "## 3. Main effects and interaction", "", head, rule]
    for species, entry in report["species"].items():
        readings = entry["readings"]
        lines += [f"| {species} | sampling: B26bal − B26 | {_row(readings['against_b26']['B26bal'])} |",
                  f"| {species} | MoE: B26moe − B26 | {_row(readings['against_b26']['B26moe'])} |",
                  f"| {species} | interaction: {readings['interaction']['label']} | {_row(readings['interaction'])} |"]
    nes = report["species"][NES]["readings"]["nes_comparators"]
    lines += ["", "## 4. NES against the named comparators", "", head, rule]
    lines += [f"| {NES} | {arm} − {other} | {_row(block)} |" for arm, row in nes.items() for other, block in row.items()]

    control = report["conditional_control"]
    lines += ["", "## 5. Conditional control", "", f"Rule: {control['rule']}.", "",
              "| species | B26moe − B26 |", "|---|---:|"]
    lines += [f"| {species} | {_s(gap)} |" for species, gap in control["per_species"].items()]
    lines += ["", f"**Fires: {'yes, on ' + ', '.join(control['fires_on']) if control['fires'] else 'no'}.**", ""]

    lines += ["## 6. Expert use by domain (training windows)", "",
              "Top-k share per expert in the last layer, per seed; each domain's sources weighted equally. Expert "
              "indices are not comparable across seeds. TV is the mean pairwise total-variation distance between the "
              "domains' share vectors (0 same routing, 1 disjoint).", ""]
    for arm, block in report["expert_use"].items():
        lines += [f"**{arm}**", "", "| seed | domain | " + " | ".join(f"e{e}" for e in range(N_EXPERTS)) + " | TV |",
                  "|---|---|" + "---:|" * (N_EXPERTS + 1)]
        for seed, layers in block["seeds"].items():
            last = layers[-1]
            for domain, shares in last["top_k_share"].items():
                lines.append(f"| {seed} | {domain} | " + " | ".join(f"{v:.3f}" for v in shares)
                             + f" | {last['tv_distance']:.3f} |")
        lines += ["", "| layer | " + " | ".join(f"seed {s}" for s in block["seeds"]) + " | median |",
                  "|---|" + "---:|" * (len(block["seeds"]) + 1)]
        for i, median in enumerate(block["median_tv_distance"]):
            cells = " | ".join(f"{layers[i]['tv_distance']:.3f}" for layers in block["seeds"].values())
            lines.append(f"| {i} | {cells} | {median:.3f} |")
        lines.append("")

    lines += ["## 7. Dilution (training records only)", "",
              "Median across seeds of each source's mean loss share over the last half of logged steps, against its "
              "draw share. Domain loss shares are shown raw and rescaled to 100%.", ""]
    for arm, block in report["dilution"].items():
        lines += [f"**{arm}**, {block['n_seeds']} seeds", "", "| domain | draw | loss (raw) | loss (rescaled) |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {d} | {v['draw_share']:.1%} | {v['loss_share_raw']:.1%} | {v['loss_share_rescaled']:.1%} |"
                  for d, v in block["domains"].items()]
        lines += ["", "| source | domain | draw | loss | loss − draw |", "|---|---|---:|---:|---:|"]
        lines += [f"| {r['species']} | {r['domain']} | {r['draw_share']:.3%} | {r['loss_share']:.3%} | "
                  f"{100 * (r['loss_share'] - r['draw_share']):+.3f} pp |" for r in block["rows"]]
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: training records, registered files, and the existence of sealed prediction files. No test data.
def carried_file_counts(tier4: dict[str, dict]) -> dict[str, dict[str, int]]:
    out = {}
    for species, *_ in T.HELD_OUT:
        counts: dict[str, int] = {}
        for tag, path in carried_paths(species, tier4).items():
            if path.exists():
                arm = tier4[tag]["arm"] if tag in tier4 else E4.split_carried_tag(tag.split(":", 1)[1])[0]
                counts[arm] = counts.get(arm, 0) + 1
        out[species] = counts
    return out


def expected_carried(species: str) -> dict[str, int]:
    arms = [arm for arm in (*CARRIED_TIER4, *CARRIED_TIER3) if arm != "BdM" or species == NES]
    return {"P": 1} | {arm: len(PT.SEEDS) for arm in arms}


def section7_markdown(gap: dict) -> str:
    moe_runs = sum(1 for run in PT.PLAN if run.arm in MOE_ARMS)
    deferred = "none" if gap["complete"] else "; ".join(gap["problems"])
    return "\n".join([
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']}; exit status {AS_TRAINED_FROM_PBS['exit status']}; "
        f"{AS_TRAINED_FROM_PBS['queue and precision']} |",
        f"| neural ({gap['n_planned']}) | {gap['n_clean']} clean: {PT.REGISTERED_STEPS:,} of {PT.REGISTERED_STEPS:,} "
        "updates applied, none withheld, no non-finite steps, finite final loss and weights |",
        f"| parameters | B26bal {N_PARAMETERS['transformer']:,}; B26moe and B26balmoe {N_PARAMETERS['moe']:,}, as sec 3 |",
        f"| pool | {POOL_WINDOWS:,} training windows in every run; draw probabilities A8 (B26moe) or domain-balanced "
        "(B26bal, B26balmoe), as sec 2 |",
        f"| routing record | complete for the {moe_runs} MoE runs: 26 sources x 4 layers x {N_EXPERTS} experts, "
        f"{PT.ROUTING_WINDOWS:,} training windows per source or all of a smaller one |",
        f"| compute | {AS_TRAINED_FROM_PBS['compute']} |",
        "| persistence | P is read from Tier 2's sealed prediction files, as Tier 4 read it |",
        f"| deferred or missing | {deferred} |",
    ])


def diagnostic() -> None:
    """Seal state, completeness, routing, carried prediction files present, and the sec 7 block. Scores nothing."""
    try:
        freeze_committed()
        print("freeze           : committed, unchanged against HEAD")
        text = committed_freeze()
        for number in (6, 7):
            try:
                check_filled(number, section_body(text, number))
                print(f"sec {number}            : filled")
            except SystemExit as reason:
                print(f"sec {number}            : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    except SystemExit as reason:
        print(f"freeze           : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    payload = PT.load_registered_pool()
    gap = completeness(read_records(), payload)
    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<10} {row['records']} of {row['planned']}, {row['clean']} clean")
    for problem in gap["problems"]:
        print(f"  {problem}")
    counts = carried_file_counts(tier4_carried_records())
    missing = {species: {arm: (found.get(arm, 0), n) for arm, n in expected_carried(species).items()
                         if found.get(arm, 0) != n}
               for species, found in counts.items()}
    print("carried files    : " + ("all present" if not any(missing.values()) else f"MISSING {missing}"))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(gap))
    if not gap["complete"] or any(missing.values()):
        raise SystemExit("FAIL: the diagnostic found a problem; sec 7 must not be filled until it passes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predict", action="store_true", help="check the seal, then score the 15 pooling runs")
    parser.add_argument("--report", action="store_true", help="build the report from saved predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score runs that already have prediction files")
    parser.add_argument("--diagnostic", action="store_true", help="runs, routing, carried files and the sec 7 block")
    args = parser.parse_args()
    if not (args.predict or args.report or args.diagnostic):
        raise SystemExit("pass --diagnostic, --predict, --report, or a combination")

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
        control = report["conditional_control"]
        print(f"conditional control: {'fires on ' + ', '.join(control['fires_on']) if control['fires'] else 'does not fire'}")


if __name__ == "__main__":
    main()
