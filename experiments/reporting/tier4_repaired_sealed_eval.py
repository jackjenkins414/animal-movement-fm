# GenAI reference: I gave an outline for this sealed evaluator for the repaired Tier 2-4 transformers; score only the new runs, 
# reuse and check the old sealed comparators, apply the existing Tier 4 scoring, and report the preregistered expectations with 
# intervals; GenAI wrote it and I checked it.
"""
Sealed evaluation of the repaired encoder reruns of the Tier 2-4 pooled plain trunk
(`preregistration/tier4_repaired_exploratory_freeze.md` secs 3, 4 and 6). Exploratory: nothing this produces is a tiered
result.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO / "experiments" / "diagnostics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_sealed_eval as RS  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402
import tier3_sealed_eval as E3  # noqa: E402
import tier4_repaired_diagnostic as DG  # noqa: E402
import tier4_sealed_eval as E4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402

R, T, env = DG.R, DG.T, DG.env
T3, T4 = R.T3, R.T4

FREEZE = REPO / "preregistration" / R.FREEZE
SEALED = REPO / "results" / "tier4_repaired_sealed"
PREDICTIONS = SEALED / "predictions"
REPORT_JSON = SEALED / "tier4_repaired_results.json"
REPORT_MD = SEALED / "tier4_repaired_results.md"
MANIFEST = SEALED / "tier4_repaired_sealed_manifest.json"
SOURCE_PREDICTIONS = R.SEALED_PREDICTIONS  # tier -> results/tier{2,3,4}_sealed/predictions
SOURCE_REPORTS = {"2": S2.SEALED / "tier2_verdict.json", "3": E3.REPORT_JSON, "4": E4.REPORT_JSON}

FLOOR, NOISE_SCALE = E3.FLOOR, E3.NOISE_SCALE
HELD_OUT = R.HELD_OUT
GULL, WOLF, NES = HELD_OUT
SHORT = DG.SHORT

NEW = tuple(R.ARMS)  # B3F, B9F, B26F, BdMF, BdTF: predicted here
# Read from their own sealed prediction files: label -> (tier, the plan that trained it). Each is read on the species
# its plan scores it on (BdM on NES, Zd-T on the wolf, LSTM-S on its own species, the rest on all three).
READ = {"B": "2", "B9": "3", "B26": "4", "BdM": "4", "LSTMB26": "4", "XGBB26": "4", "ZdT": "4", "LSTMS": "3"}
PLANS = {"2": T.PLAN, "3": T3.PLAN, "4": T4.PLAN}
LABELS = ("P", *NEW, *READ)

# Freeze sec 6, in its order. Each comparison is (left, right, species), right "P" meaning skill against P; "above" is
# met by a difference > NOISE_SCALE, "within" by |difference| <= NOISE_SCALE. Point estimates (median of seeds) decide;
# intervals are shown beside them. "alongside" rows are reported with no expectation.
EXPECTATIONS: tuple[dict, ...] = (
    {"rule": "above", "pairs": [(new, twin, s) for new, twin in (("B3F", "B"), ("B9F", "B9"), ("B26F", "B26"))
                                for s in (GULL, NES)] + [("BdMF", "BdM", NES)], "alongside": []},
    {"rule": "within", "pairs": [("B26F", "LSTMB26", s) for s in (GULL, NES)],
     "alongside": [("B26F", "XGBB26", s) for s in (GULL, NES)]},
    {"rule": "within", "pairs": [(a, b, s) for a, b in (("B9F", "B3F"), ("B26F", "B9F"), ("B26F", "B3F"))
                                 for s in HELD_OUT], "alongside": []},
    {"rule": "above", "pairs": [("BdMF", "B26F", NES)], "alongside": []},
    {"rule": "within", "pairs": [(arm, "P", WOLF) for arm in ("B3F", "B9F", "B26F", "BdTF")],
     "alongside": [("BdTF", "B26F", WOLF), ("BdTF", "ZdT", WOLF)]},
)
_NAMED = {name for e in EXPECTATIONS for left, right, _ in e["pairs"] + e["alongside"] for name in (left, right)}
if set(READ) & set(NEW) or not _NAMED <= set(LABELS):
    raise RuntimeError(f"sec 6 names arms outside the evaluator's: {sorted(_NAMED - set(LABELS))}")


# Seal
@contextlib.contextmanager
def bound():
    """Point the Tier 2 freeze check and the Tier 4 "As trained" gate at this freeze while the block runs."""
    saved = S2.FREEZE, E4.FREEZE
    S2.FREEZE, E4.FREEZE = FREEZE, FREEZE
    try:
        yield
    finally:
        S2.FREEZE, E4.FREEZE = saved


def assert_sealed() -> dict:
    """The freeze committed and unchanged, its "As trained" section filled in the committed version, and any earlier
    manifest attested at the same blob. Run by --predict before any test partition is opened."""
    with bound():
        attestation = S2.assert_freeze_committed()
        attestation["as_trained_lines"] = len(E4.assert_as_trained_filled().splitlines())
    if MANIFEST.exists():
        earlier = json.loads(MANIFEST.read_text()).get("attestation") or {}
        if earlier.get("freeze_blob") and earlier["freeze_blob"] != attestation["freeze_blob"]:
            raise SystemExit(
                f"FAIL: {MANIFEST.name} was attested at blob {earlier['freeze_blob'][:12]}, but {FREEZE.name} is now "
                f"blob {attestation['freeze_blob'][:12]}. Rebuild the report with --report instead."
            )
    return attestation


def checked_runs() -> tuple[dict, dict, int, dict]:
    """(records, completeness, grid index, payloads), refusing unless all 25 runs are clean (the diagnostic's check)."""
    payloads = R.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    records = DG.read_records()
    gap = DG.completeness(records, grid_index, payloads)
    if not gap["complete"]:
        raise SystemExit("FAIL: runs are missing or unclean\n  " + "\n  ".join(gap["problems"]))
    return records, gap, grid_index, payloads


# Files
def prediction_path(species: str, tag: str) -> Path:
    return PREDICTIONS / f"{species}__{tag}.parquet"


def new_runs(species: str) -> list:
    return [run for run in R.PLAN if species in run.scored_on]


def read_runs(species: str) -> list[tuple[str, str, str, int]]:
    """(label, tier, run id, seed) of every read arm's runs on this species, from the plan that trained them."""
    return [(label, tier, run.run_id, run.seed) for label, tier in READ.items()
            for run in PLANS[tier] if run.arm == label and run.species in ("", species)]


def reused_files(species: str) -> dict[str, Path]:
    """Tag -> sealed prediction file for P and every read arm on this species. Tags are run ids, distinct by the plans."""
    out = {"P": SOURCE_PREDICTIONS["2"] / f"{species}__P.parquet"}
    out |= {run_id: SOURCE_PREDICTIONS[tier] / f"{species}__{run_id}.parquet" for _, tier, run_id, _ in read_runs(species)}
    return out


def missing_reused() -> list[str]:
    return [str(path) for species in HELD_OUT for path in reused_files(species).values() if not path.exists()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(frame: pd.DataFrame, out: Path) -> None:
    """Write beside the target, then rename, so a run killed mid-write leaves no file that reads as finished."""
    partial = out.with_name(out.name + ".partial")
    try:
        frame.to_parquet(partial, index=False)
        os.replace(partial, out)
    finally:
        partial.unlink(missing_ok=True)


def source_reports() -> dict[str, dict]:
    """The Tier 2, 3 and 4 sealed reports the read arms are tied back to; refused if missing or unattested."""
    out = {}
    for tier, path in SOURCE_REPORTS.items():
        if not path.exists():
            raise SystemExit(f"FAIL: {path} does not exist; the read arms are tied back to their sealed report")
        report = json.loads(path.read_text())
        if not (report.get("attestation") or {}).get("freeze_commit"):
            raise SystemExit(f"FAIL: {path.name} carries no freeze attestation")
        out[tier] = report
    return out


# Prediction
def predict_species(species: str, device, overwrite: bool, grid_index: int, payloads: dict) -> list[str]:
    """Score this species' share of the 25 runs on its test partition, one cached file per run."""
    frame = S2.test_frame(species)
    print(f"\n{species}: {len(frame):,} test fixes, {frame['individual_id'].nunique()} individuals", flush=True)
    written = []
    for run in new_runs(species):
        out = prediction_path(species, run.run_id)
        written.append(out.name)
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            continue
        started = time.time()
        entry = next(e for e in DG.entries(run.arm) if e["run_id"] == run.run_id)
        model, record = DG.load_model(entry, grid_index, payloads)
        if species in record["pool_species"]:
            raise SystemExit(f"FAIL: {run.run_id}'s pool held {species}")
        predictor = MovementFMPredictor(model, device=device, sigma=SCALE_FLOOR_M)
        results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
        write_atomic(results, out)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)
    return written


def run_predictions(device, overwrite: bool = False, tf32: dict | None = None) -> dict:
    attestation = assert_sealed()  # nothing before this line touches a test partition
    _, gap, grid_index, payloads = checked_runs()
    source_reports()  # refuse now rather than after a GPU job
    missing = missing_reused()
    if missing:
        raise SystemExit(f"FAIL: {len(missing)} comparator prediction files are missing\n  " + "\n  ".join(missing[:20]))
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    manifest = {"attestation": attestation, "tf32": tf32 if tf32 is not None else env.tf32_flags(),
                "completeness": gap, "species": {}, "reused": {}}
    for species in HELD_OUT:
        manifest["species"][species] = predict_species(species, device, overwrite, grid_index, payloads)
        manifest["reused"][species] = {tag: sha256(path) for tag, path in reused_files(species).items()}
    MANIFEST.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


# Frames
def load_frames(species: str, manifest: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict]]:
    """
    (P, frames by tag, parsed tag -> arm and seed) for one species: the read files checked against the sha256 recorded
    at --predict, this study's predictions with nothing else allowed in its directory, and every frame aligned to P.
    """
    recorded = (manifest.get("reused") or {}).get(species) or {}
    frames, parsed = {}, {}
    for tag, path in reused_files(species).items():
        if not path.exists() or recorded.get(tag) != sha256(path):
            raise SystemExit(f"FAIL: {path} is missing or differs from the sha256 recorded at --predict")
        frames[tag] = E3.load_frame(path)
    parsed["P"] = {"arm": "P", "seed": None, "k": None, "draw": None}
    parsed |= {run_id: {"arm": label, "seed": seed, "k": None, "draw": None}
               for label, _, run_id, seed in read_runs(species)}
    own = {run.run_id: run for run in new_runs(species)}
    present = {path.stem.split("__", 1)[1] for path in PREDICTIONS.glob(f"{species}__*.parquet")}
    if present - set(own):
        raise SystemExit(f"FAIL: {species} holds predictions this study does not score: {sorted(present - set(own))}")
    if set(own) - present:
        raise SystemExit(f"FAIL: {species} has no predictions for {sorted(set(own) - present)}; run --predict")
    for run_id, run in own.items():
        frames[run_id] = E3.load_frame(prediction_path(species, run_id))
        parsed[run_id] = {"arm": run.arm, "seed": run.seed, "k": None, "draw": None}
    persistence = frames["P"]
    for tag, frame in frames.items():
        E3.assert_aligned(frame, persistence, f"{species} {tag}")
    return persistence, frames, parsed


def lookup(tag: str, parsed: dict) -> dict | None:
    return parsed.get(tag)


def labels_on(species: str) -> list[str]:
    """P, the new arms scored on this species and the read arms with runs on it, in report order."""
    scored = {run.arm for run in new_runs(species)} | {label for label, *_ in read_runs(species)}
    return ["P"] + [label for label in LABELS[1:] if label in scored]


# Statistic
def pairs_on(species: str) -> list[tuple[str, str]]:
    """The (left, right) comparisons sec 6 reads on this species, against-P ones excluded (they are in against_p)."""
    out = []
    for expectation in EXPECTATIONS:
        for left, right, s in expectation["pairs"] + expectation["alongside"]:
            if s == species and right != "P" and (left, right) not in out:
                out.append((left, right))
    return out


def species_block(species: str, persistence: pd.DataFrame, frames: dict, parsed: dict) -> dict:
    """One species: arm blocks (median of seeds), skill against P for every arm and the sec 6 comparisons, with skill
    and metre intervals from one set of resamples."""
    _, _, profile_key, domain, _ = T.held_out_entry(species)
    boundary = T.boundary_m(profile_key)
    labels = labels_on(species)
    E4.assert_no_collisions(frames, parsed, parse=lookup)
    summaries = E3.summarise_all(frames, persistence, boundary, parsed, parse=lookup)
    problems = [label for label in labels if label not in summaries]
    for label in labels[1:]:
        seeds = {key for key in summaries.get(label, {}).get("by_seed", {}) if key >= 0}
        if label in summaries and seeds != set(R.SEEDS):
            problems.append(f"{label} has seeds {sorted(seeds)}, expected {list(R.SEEDS)}")
    if problems:
        raise SystemExit(f"FAIL: {species}: " + "; ".join(problems))
    arms = RS.arm_blocks(summaries)
    resampler = RS.Resampler(persistence, boundary)
    keep = S2.eligible_mask(persistence, FLOOR)
    return {
        "domain": domain,
        "regime_boundary_m": boundary,
        "n_test_individuals": int(persistence["individual_id"].nunique()),
        "n_scored_individuals": int(persistence.loc[keep, "individual_id"].nunique()),
        "n_test_windows": int(len(persistence)),
        "share_in_travel": float(S2.stratum_masks(persistence, boundary)["travel"][keep].mean()),
        "labels": labels,
        "arms": arms,
        "against_p": {label: RS.compare(label, "P", arms, frames, persistence, boundary, resampler)
                      for label in labels[1:]},
        "comparisons": {f"{left} - {right}": RS.compare(left, right, arms, frames, persistence, boundary, resampler)
                        for left, right in pairs_on(species)},
    }


def reading_of(report: dict, left: str, right: str, species: str) -> dict:
    block = report["species"].get(species) or {}
    if right == "P":
        return (block.get("against_p") or {}).get(left, {})
    return (block.get("comparisons") or {}).get(f"{left} - {right}", {})


def met(rule: str, difference) -> bool | None:
    if difference is None or not np.isfinite(difference):
        return None
    return bool(difference > NOISE_SCALE) if rule == "above" else bool(abs(difference) <= NOISE_SCALE)


def verdict(flags: list) -> str:
    if any(flag is None for flag in flags):
        return "not scored"
    return "met" if all(flags) else f"not met ({sum(flags)} of {len(flags)} met)"


def sec6_bullets(path: Path = FREEZE) -> list[str]:
    """The freeze's sec 6 expectations, quoted rather than restated; exactly one per reading."""
    bullets = [line[2:].strip() for line in RS.section_body(path.read_text(), 6).splitlines() if line.startswith("- ")]
    if len(bullets) != len(EXPECTATIONS):
        raise SystemExit(f"FAIL: {path.name} sec 6 has {len(bullets)} expectations; the evaluator reads "
                         f"{len(EXPECTATIONS)}")
    return bullets


def expectations(report: dict) -> list[dict]:
    """Each sec 6 expectation: every comparison's point difference, intervals and met flag, and the overall verdict."""
    out = []
    for text, expectation in zip(sec6_bullets(), EXPECTATIONS):
        rows = []
        for kind in ("pairs", "alongside"):
            for left, right, species in expectation[kind]:
                reading = reading_of(report, left, right, species)
                difference = reading.get("skill_difference")
                skill = reading.get("skill_bootstrap") or {}
                rows.append({
                    "comparison": f"{left} - {right}", "species": species, "alongside": kind == "alongside",
                    "skill_difference": difference,
                    "skill_interval": skill.get("skill_difference"), "metre_interval": skill.get("difference_m"),
                    "met": None if kind == "alongside" else met(expectation["rule"], difference),
                })
        out.append({"expectation": text, "rule": expectation["rule"], "noise_scale": NOISE_SCALE, "rows": rows,
                    "verdict": verdict([row["met"] for row in rows if not row["alongside"]])})
    return out


def source_skill(reports: dict, tier: str, species: str, label: str) -> float:
    entry = (reports[tier].get("species") or {}).get(species) or {}
    if tier == "2":
        skill = (((entry.get("comparators") or {}).get(label) or {}).get("persistence_skill") or {})
        return float(skill.get("travel_p90_m", float("nan")))
    return E3.travel_skill(entry.get("arms") or {}, label)


def tie_back(report: dict, reports: dict) -> dict:
    """Each read arm's travel p90 skill here minus its own sealed report's, per species; max |diff| and median. No pass/fail."""
    out: dict = {"sources": {tier: str(path.relative_to(REPO)) for tier, path in SOURCE_REPORTS.items()},
                 "attestations": {tier: r.get("attestation") for tier, r in reports.items()}, "arms": {}}
    for label, tier in READ.items():
        cells = {species: E3.travel_skill(block["arms"], label) - source_skill(reports, tier, species, label)
                 for species, block in report["species"].items() if label in block["labels"]}
        values = np.array(list(cells.values()), dtype=float)
        finite = values[np.isfinite(values)]
        out["arms"][label] = {"tier": tier, "per_species": cells, "n": int(finite.size),
                              "max_abs_diff": float(np.abs(finite).max()) if finite.size else None,
                              "median_diff": float(np.median(finite)) if finite.size else None}
    return out


# Report
def build_report(tf32: dict | None = None) -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"FAIL: {MANIFEST} does not exist; run --predict first")
    manifest = json.loads(MANIFEST.read_text())
    reports = source_reports()
    _, gap, _, _ = checked_runs()
    report: dict = {
        "study": "tier4_repaired",
        "exploratory": True,
        "registered_by": f"preregistration/{FREEZE.name}",
        "attestation": manifest.get("attestation"),
        "switches": R.SWITCHES,
        "evaluation_floor": FLOOR,
        "seed_aggregation": "median_across_training_seeds",
        "noise_scale": NOISE_SCALE,
        "n_bootstrap": S2.N_BOOTSTRAP,
        "bootstrap_seed": S2.BOOTSTRAP_SEED,
        "tf32": {"predict": manifest.get("tf32"), "report": tf32 if tf32 is not None else env.tf32_flags()},
        "completeness": gap,
        "comparators_read_from": {label: str(SOURCE_PREDICTIONS[tier].relative_to(REPO)) for label, tier in READ.items()},
        "species": {},
    }
    for species in HELD_OUT:
        started = time.time()
        persistence, frames, parsed = load_frames(species, manifest)
        report["species"][species] = species_block(species, persistence, frames, parsed)
        print(f"{species}: {len(report['species'][species]['labels'])} arms in {time.time() - started:.0f}s", flush=True)
    report["expectations"] = expectations(report)
    report["tie_back"] = tie_back(report, reports)
    return report


def _skill_cell(reading: dict, key: str) -> str:
    block = (reading.get("skill_bootstrap") or {}).get("skill", {}).get(key)
    return f"{RS._s(reading.get(f'{key}_travel_p90_skill'))} {RS._interval(block)}"


def write_markdown(report: dict, path: Path) -> None:
    lines = ["**Exploratory — not a tiered result.**", "",
             "# Repaired transformer on the Tier 2–4 pools: sealed evaluation", ""]
    attestation = report.get("attestation") or {}
    if attestation.get("freeze_commit"):
        lines += [f"`{report['registered_by']}` committed at `{attestation['freeze_commit'][:12]}` "
                  f"(blob `{attestation['freeze_blob'][:12]}`).", ""]
    gap = report["completeness"]
    lines += [f"The repaired encoder is {report['switches']}. Runs scored: {gap['n_clean']} of {gap['n_planned']} clean. "
              f"Travel p90 skill against P at F = {report['evaluation_floor']}, median of seeds; paired bootstrap of "
              f"{report['n_bootstrap']:,} test-individual resamples at seed {report['bootstrap_seed']}, the same draws "
              f"for every row of a species, with 95% intervals in skill and in metres (p90 difference). ±"
              f"{report['noise_scale']} counts as no difference; each expectation is read on the point estimate. "
              f"TF32 at prediction: {report['tf32']['predict']}.", ""]

    lines += ["## Expectations (freeze sec 6)", ""]
    for number, block in enumerate(report["expectations"], 1):
        lines += [f"### {number}. {block['verdict']}", "", f"> {block['expectation']}", "",
                  "| comparison | species | skill difference | skill interval | p90 difference interval (m) | met |",
                  "|---|---|---:|---|---|---|"]
        for row in block["rows"]:
            flag = "alongside" if row["alongside"] else {True: "yes", False: "no", None: "not scored"}[row["met"]]
            lines.append(f"| {row['comparison']} | {SHORT[row['species']]} | {RS._s(row['skill_difference'])} | "
                         f"{RS._interval(row['skill_interval'])} | {RS._interval(row['metre_interval'], 'm')} | {flag} |")
        lines.append("")

    names = list(report["species"])
    lines += ["## Travel p90 skill against P (median of seeds, 95% interval)", "",
              "| arm | " + " | ".join(SHORT[s] for s in names) + " |", "|---|" + "---|" * len(names)]
    for label in LABELS[1:]:
        cells = [_skill_cell(report["species"][s]["against_p"][label], label)
                 if label in report["species"][s]["against_p"] else "—" for s in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "| species | test individuals | at the floor | test windows | travel share | boundary (m) |",
              "|---|---:|---:|---:|---:|---:|"]
    for species, block in report["species"].items():
        lines.append(f"| {SHORT[species]} | {block['n_test_individuals']} | {block['n_scored_individuals']} | "
                     f"{block['n_test_windows']:,} | {block['share_in_travel']:.3f} | {block['regime_boundary_m']:,.1f} |")

    tie = report["tie_back"]
    lines += ["", "## Read arms tied back to their own sealed reports (skill here − there; no pass/fail)", "",
              "| arm | report | species | max \\|diff\\| | median diff |", "|---|---|---:|---:|---:|"]
    for label, entry in tie["arms"].items():
        top = f"{entry['max_abs_diff']:.2e}" if entry["max_abs_diff"] is not None else "—"
        median = f"{entry['median_diff']:+.2e}" if entry["median_diff"] is not None else "—"
        lines.append(f"| {label} | `{tie['sources'][entry['tier']]}` | {entry['n']} | {top} | {median} |")
    path.write_text("\n".join(lines) + "\n")


# Diagnostic: records, the twins' records and checkpoints, and the comparator files; no test partition is opened
def committed_text() -> str | None:
    shown = E4.git("show", f"HEAD:{FREEZE.relative_to(REPO).as_posix()}")
    return shown.stdout if shown.returncode == 0 else None


def diagnostic() -> None:
    with bound():
        try:
            S2.assert_freeze_committed()
        except SystemExit as reason:
            print(f"freeze           : {str(reason).removeprefix('FAIL: ')} (not enforced here)")
    text = committed_text()
    for number in (6, 7):
        body = RS.section_body(text, number) if text else ""
        filled = bool(body) and not RS.PLACEHOLDER.search(body)
        print(f"sec {number}            : {'filled' if filled else 'not filled'} in the committed version "
              "(not enforced here)")
    payloads = R.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    records = DG.read_records()
    gap = DG.completeness(records, grid_index, payloads)
    twins = DG.twin_check(grid_index, payloads)
    missing = missing_reused()
    try:
        reports, report_problem = source_reports(), None
    except SystemExit as reason:
        reports, report_problem = {}, str(reason)
    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print("twins            : " + ("as registered" if not twins else "\n  " + "\n  ".join(twins[:20])))
    n_files = sum(len(reused_files(species)) for species in HELD_OUT)
    print(f"comparator files : {n_files - len(missing)} of {n_files} present"
          + ("" if not missing else "\n  " + "\n  ".join(missing[:20])))
    print("source reports   : " + (report_problem or ", ".join(
        f"tier {tier} at {r['attestation']['freeze_commit'][:12]}" for tier, r in reports.items())))
    if not gap["complete"] or twins or missing or report_problem:
        raise SystemExit("FAIL: the diagnostic found a problem; fix it before --predict")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diagnostic", action="store_true", help="records and comparator files only; no test")
    parser.add_argument("--predict", action="store_true", help="check the seal, then score the 25 runs on test")
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
        print(f"\nwrote {REPORT_JSON}\nwrote {REPORT_MD}\n")
        for number, block in enumerate(report["expectations"], 1):
            print(f"sec 6.{number}  {block['verdict']}")


if __name__ == "__main__":
    main()
