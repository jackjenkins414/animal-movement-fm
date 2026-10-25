# GenAI reference: I outlined this validation only check for our repaired transformer across the bird rotation; compare all 60 repaired runs 
# with their 60 originals on shared train/val samples using the same loss etc; GenAI wrote it and I checked it.
"""
Validation diagnostic for the repaired transformer on the rotating held-out birds (exploratory; no training; training and
validation windows only). Freeze: `preregistration/birds_rotation_repaired_exploratory_freeze.md` sec 5.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))
sys.path.insert(0, str(REPO))

import bird_rotation_repaired_train as D  # noqa: E402
import tier4_repaired_diagnostic as DG  # noqa: E402

E3, TC, R, RT, BT, T, env = DG.E3, DG.TC, D.R, D.RT, D.BT, D.T, D.env

OUT = REPO / "results" / "birds_rotation_repaired_diagnostic"
REPORT_JSON = OUT / "birds_rotation_repaired_diagnostic.json"
REPORT_MD = OUT / "birds_rotation_repaired_diagnostic.md"
FREEZE = REPO / "preregistration" / D.FREEZE

N_TRAIN = DG.N_TRAIN  # 20,000 A8 draws from the fold's training pool, shared by its twelve models
MAX_VAL = DG.MAX_VAL  # 3,000 validation windows per held-out bird, uniform, the same for every model
SAMPLE_SEED = DG.SAMPLE_SEED
LABELS = ("BbF6", "Bb", "ZbF6", "Zb")  # each repaired arm, then its original
DEFAULTS = DG.DEFAULTS  # the switches as every original was trained
PLAN_INDEX = {run.run_id: i for i, run in enumerate(D.PLAN)}


class Forecast(torch.nn.Module):
    """The model's forecast alone: Zb and ZbF6 also return their reconstruction, which this diagnostic does not read."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.model(features)
        return out[0] if isinstance(out, tuple) else out


# Runs.
def pool_names(fold: int, payloads: dict) -> list[str]:
    """The fold's registered pool species, read from the registered files without loading any window."""
    if fold == 0:
        return sorted(payloads["birds"]["probabilities"]["main"])
    return sorted(payloads["rotation"]["probabilities"][str(fold)])


def entries(fold: int) -> list[dict]:
    """The fold's six repaired runs and, at the same seeds, their six originals, with where each record lives."""
    if fold not in D.FOLDS:
        raise SystemExit(f"FAIL: fold {fold} is not one of {list(D.FOLDS)}")
    out = []
    for run in (r for r in D.PLAN if r.fold == fold):
        out.append({"label": run.arm, "role": "repaired", "seed": run.seed, "run_id": run.run_id,
                    "results": D.RESULTS, "run": run})
        out.append({"label": run.twin.arm, "role": "original", "seed": run.seed, "run_id": run.twin.run_id,
                    "results": run.twin_results, "run": run})
    return out


def run_problems(run, record: dict | None, grid_index: int, payloads: dict) -> list[str]:
    """What stops one planned run counting as finished cleanly: its record against the plan, the switches and budget."""
    if record is None:
        return ["no record"]
    problems = [
        f"{key} {record.get(key)!r}, expected {want!r}"
        for key, want in (
            ("plan_index", PLAN_INDEX[run.run_id]),
            ("arm", run.arm),
            ("fold", run.fold),
            ("seed", run.seed),
            ("tier", D.TAG),
            ("twin", run.twin.run_id),
            ("twin_study", run.twin_study),
            ("switches", D.SWITCHES),
            ("queue", D.QUEUE),
            ("exploratory", True),
            ("opened_test_partition", False),
        )
        if record.get(key) != want
    ]
    if not re.fullmatch(rf"{D.QUEUE}(-exec)?", str(record.get("pbs_queue"))):
        problems.append(f"pbs_queue {record.get('pbs_queue')!r}")
    names = pool_names(run.fold, payloads)
    if sorted(record.get("pool_species") or []) != names:
        problems.append("pool_species differ from the registered pool")
    if R._json(record.get("excluded_individuals")) != R._json(D.excluded_individuals(run, names, payloads)):
        problems.append("excluded_individuals differ from the registered exclusions")
    expected = R._json(asdict(D.build_config(run, grid_index, D.REGISTERED_STEPS, names, payloads)))
    if R.with_defaults(record.get("config") or {}) != expected:
        problems.append("config differs from its original's with the switches")
    if any((record.get("tf32") or {"missing": True}).values()):
        problems.append(f"tf32 {record.get('tf32')}")
    if record.get("applied_updates") != D.REGISTERED_STEPS or (record.get("config") or {}).get("steps") != D.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {(record.get('config') or {}).get('steps')}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (D.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


def switch_problems(state: dict, repaired: bool) -> list[str]:
    """A checkpoint's switches against its role: the repaired encoder, or every field at the originals' default."""
    switches = {name: state["config"].get(name, DEFAULTS[name]) for name in D.SWITCHES}
    problems = [] if switches == (D.SWITCHES if repaired else DEFAULTS) else [f"checkpoint switches {switches}"]
    if repaired and state.get("switches") != D.SWITCHES:
        problems.append(f"checkpoint records switches {state.get('switches')}")
    return problems


def load_model(entry: dict, grid_index: int, payloads: dict):
    """(forecast-only model in eval mode, record), refused on any problem with the record or the checkpoint."""
    run, repaired = entry["run"], entry["role"] == "repaired"
    path = entry["results"] / f"{entry['run_id']}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    names = pool_names(run.fold, payloads)
    problems = run_problems(run, record, grid_index, payloads) if repaired else D.twin_problems(
        run, grid_index, names, payloads)
    loaded = None if problems else E3.load_reported(entry["results"], entry["run_id"])
    if not problems and loaded is None:
        problems.append("no reported checkpoint")
    if loaded is not None:
        problems += switch_problems(loaded[1], repaired)
    if problems:
        raise SystemExit(f"FAIL: {entry['run_id']}: " + "; ".join(problems))
    return Forecast(loaded[0]).eval(), record


# Windows and samples.
def training_sample(fold: int, payloads: dict, n: int = N_TRAIN, seed: int = SAMPLE_SEED) -> dict:
    """n draws under training's A8 rule (SourcePool.sample) from the fold's registered pool, at a fixed seed."""
    run = D.Run("BbF6", fold, D.SEEDS[0])
    source_pool, node = D.training_source(run, payloads)
    D.check_window_counts(run, source_pool, payloads)
    if list(source_pool.names) != pool_names(fold, payloads):
        raise SystemExit(f"FAIL: fold {fold}'s pool {source_pool.names} is not its registered node's")
    features, targets, scale = source_pool.sample(np.random.default_rng(seed), n)
    return {"node": node, "species": list(source_pool.names), "seed": seed,
            "features": features.astype(np.float32), "targets": targets, "scale": scale}


def held_out_entry(fold: int, species: str) -> tuple[str, str, float]:
    """(segments stem, split stem, pinned A4 boundary) of a held-out bird, from the driver that pinned it."""
    if fold == 0:
        stem, split_stem, key = BT.bird_attached_entry(species)
        return stem, split_stem, BT.bird_boundary_m(key)
    stem, split_stem, key = RT.rotation_attached_entry(species)
    if RT.FOLD_OF[species] != fold:
        raise SystemExit(f"FAIL: {species} is held out in fold {RT.FOLD_OF[species]}, not {fold}")
    return stem, split_stem, RT.rotation_boundary_m(key)


def validation_windows(fold: int, species: str, cap: int = MAX_VAL) -> dict:
    """Up to `cap` of one held-out bird's validation windows, the same for every model, and its A4 boundary."""
    stem, split_stem, boundary = held_out_entry(fold, species)
    windows = T._load_species_windows(stem, split_stem, "val")
    ids = TC.validation_sample(windows.n_windows, cap, stem)
    features, targets, scale = windows.gather(ids)
    return {"features": features.astype(np.float32), "targets": targets, "scale": scale,
            "available": int(windows.n_windows), "boundary_m": float(boundary)}


# One fold.
def run_fold(fold: int, *, n_train: int = N_TRAIN, max_val: int = MAX_VAL, dry_run: bool = False) -> dict:
    """The fold's twelve models on the same samples; with `dry_run`, the samples and guards without any model."""
    started = time.time()
    group = entries(fold)
    payloads = D.load_payloads()
    RT.check_boundaries()
    BT.check_boundaries()
    grid_index, _ = T.selected_grid_index(None)
    loaded = {} if dry_run else {e["run_id"]: load_model(e, grid_index, payloads) for e in group}  # fail before building
    held_out = D.Run("BbF6", fold, D.SEEDS[0]).scored_on

    loads: list[dict] = []
    with DG.audited_loader(loads):
        train = training_sample(fold, payloads, n_train)
        val = {species: validation_windows(fold, species, max_val) for species in held_out}
    print(f"fold {fold}: {len(train['targets']):,} training draws over {len(train['species'])} species "
          f"[{train['node']}]; " + "; ".join(f"{s} {len(b['targets']):,} of {b['available']:,} val windows"
                                            for s, b in val.items())
          + f"; {len(loads)} loads, none from a test partition; built in {time.time() - started:.0f}s", flush=True)

    out = {
        "fold": fold, "scored_on": list(val), "exploratory": True, "opened_test_partition": False, "dry_run": dry_run,
        "samples": {"n_train": int(len(train["targets"])), "train_seed": train["seed"], "train_node": train["node"],
                    "train_species": train["species"], "max_val": max_val, "val_seed": SAMPLE_SEED,
                    "val": {s: {"windows": int(len(b["targets"])), "available": b["available"],
                                "boundary_m": b["boundary_m"]} for s, b in val.items()}},
        "loads": loads,
        "zero_loss": {"train": DG.loss_over_zero(np.zeros_like(train["targets"]), train["targets"])["zero"],
                      "val": {s: DG.loss_over_zero(np.zeros_like(b["targets"]), b["targets"])["zero"]
                              for s, b in val.items()}},
        "models": {},
    }
    for e in (e for e in group if e["run_id"] in loaded):
        model, record = loaded[e["run_id"]]
        t0 = time.time()
        out["models"][e["run_id"]] = {"label": e["label"], "role": e["role"], "seed": e["seed"]} | DG.score(
            model, record, train, val)
        print(f"  {e['run_id']}: done in {time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - started, 1)
    return out


# Report.
def summarise(folds: list[dict]) -> dict:
    """
    Per fold and label, every measure's median over the label's seeds; across folds, per label, the median of the fold
    medians (training sample) and of the per-bird medians (validation), and how many folds sit within 1% of zero.
    """
    by_fold, birds = {}, {label: [] for label in LABELS}
    for fold in folds:
        rows = {}
        for label in LABELS:
            models = [m for m in fold["models"].values() if m["label"] == label]

            def med(*keys, models=models):
                return TC._median([TC._get(m, *keys) for m in models])

            rows[label] = {
                "seeds": sorted(m["seed"] for m in models),
                "train_loss_over_zero": med("train", "ratio"),
                "logged_final_over_zero": med("logged", "final_over_zero"),
                "logged_last_over_zero": med("logged", "last_over_zero"),
                "val": {species: {name: med("val", species, *keys) for name, keys in DG.VAL_METRICS.items()}
                        for species in fold["scored_on"]},
            }
            birds[label] += [v | {"species": s, "fold": fold["fold"]} for s, v in rows[label]["val"].items()]
        by_fold[fold["fold"]] = rows
    overall = {}
    for label in LABELS:
        losses = [by_fold[k][label]["train_loss_over_zero"] for k in by_fold]
        overall[label] = {
            "folds": len(losses),
            "birds": len(birds[label]),
            "train_loss_over_zero": TC._median(losses),
            "folds_within_1pct_of_zero": sum(v is not None and v >= 0.99 for v in losses),
            **{name: TC._median([b[name] for b in birds[label] if b[name] is not None])
               for name in ("loss_over_zero", "last_swapped_over_output", "last_plus_1_over_output", "ratio")},
        }
    return {"by_fold": by_fold, "overall": overall}


def write_markdown(folds: list[dict], summary: dict, path: Path) -> None:
    m = TC._m
    lines = [
        "# Repaired transformer on the rotating held-out birds: validation diagnostic (exploratory)", "",
        f"The 60 runs of `bird_rotation_repaired_train.py` and their 60 originals (Bb and Zb; fold 0 from the bird study), "
        f"from their reported checkpoints; no training, and no test partition read. Per fold: {N_TRAIN:,} A8 draws from "
        f"the fold's training pool (seed {SAMPLE_SEED}, shared by its twelve models) and up to {MAX_VAL:,} validation "
        "windows per held-out bird (the same for every model). Zb and ZbF6 are read on their forecast output only. Every "
        "value is a median over seeds; model units are metres / window scale. No pass/fail.", "",
        "Freeze sec 5, applied by hand:", "",
        "> " + DG.reading_rule(FREEZE).replace("\n", "\n> "), "",
        "## 1. Across folds", "",
        "Training sample loss / zero: median over the ten folds, and folds at 0.99 or above. Validation: medians over the "
        "30 held-out birds.", "",
        "| model | folds | birds | training loss / zero | folds ≥ 0.99 | val loss / zero | swap / \\|out\\| | "
        "+1 / \\|out\\| | \\|pred\\| / \\|actual\\|, travel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary["overall"].items():
        lines.append(f"| {label} | {row['folds']} | {row['birds']} | {m([row['train_loss_over_zero']])} | "
                     f"{row['folds_within_1pct_of_zero']} | {m([row['loss_over_zero']])} | "
                     f"{m([row['last_swapped_over_output']])} | {m([row['last_plus_1_over_output']])} | "
                     f"{m([row['ratio']])} |")
    lines += ["", "## 2. Per fold: loss against the zero predictor (model / zero)", "",
              "| fold | model | seeds | training sample | logged, final | logged, last 5 | val, per bird |",
              "|---:|---|---:|---:|---:|---:|---|"]
    for fold, rows in summary["by_fold"].items():
        for label, row in rows.items():
            val = "; ".join(f"{s} {m([v['loss_over_zero']])}" for s, v in row["val"].items())
            lines.append(f"| {fold} | {label} | {len(row['seeds'])} | {m([row['train_loss_over_zero']])} | "
                         f"{m([row['logged_final_over_zero']])} | {m([row['logged_last_over_zero']])} | {val} |")
    lines += ["", "The logged losses are single training batches, each divided by the zero predictor's loss on the "
              "fold's training sample; for Zb and ZbF6 they include the reconstruction term.", "",
              "## 3. Per bird: input sensitivity and magnitudes on validation", "",
              "| fold | bird | model | median \\|out\\| | swap / \\|out\\| | +1 / \\|out\\| | travel windows | "
              "\\|pred\\| / \\|actual\\|, travel | median \\|pred\\| m, travel | median \\|actual\\| m, travel |",
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fold, rows in summary["by_fold"].items():
        for species in next(iter(rows.values()))["val"]:
            for label, row in rows.items():
                v = row["val"][species]
                lines.append(f"| {fold} | {species} | {label} | {m([v['median_abs_output']])} | "
                             f"{m([v['last_swapped_over_output']])} | {m([v['last_plus_1_over_output']])} | "
                             f"{m([v['n_travel']])} | {m([v['ratio']])} | {m([v['median_pred_m']])} | "
                             f"{m([v['median_actual_m']])} |")
    lines += ["", "## Samples", "", "| fold | training pool | species | draws | validation windows (of available) | "
              "loads | from test |", "|---:|---|---:|---:|---|---:|---:|"]
    for fold in folds:
        s = fold["samples"]
        val = "; ".join(f"{k} {v['windows']:,} of {v['available']:,}" for k, v in s["val"].items())
        lines.append(f"| {fold['fold']} | {s['train_node']} | {len(s['train_species'])} | {s['n_train']:,} | {val} | "
                     f"{len(fold['loads'])} | {sum(load['test_individuals'] for load in fold['loads'])} |")
    path.write_text("\n".join(lines) + "\n")


def report() -> None:
    files = [OUT / f"fold{k}.json" for k in D.FOLDS]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing {', '.join(missing)}; run every fold first")
    folds = [json.loads(p.read_text()) for p in files]
    if any(fold.get("dry_run") for fold in folds):
        raise SystemExit("FAIL: a fold file was written by a dry run")
    if any(len(fold["models"]) != 12 for fold in folds):
        raise SystemExit("FAIL: a fold file does not hold its six repaired runs and six originals")
    summary = summarise(folds)
    REPORT_JSON.write_text(json.dumps({"exploratory": True, "opened_test_partition": False, "summary": summary,
                                       "folds": folds}, indent=1, default=str) + "\n")
    write_markdown(folds, summary, REPORT_MD)
    print(f"wrote {REPORT_JSON}\nwrote {REPORT_MD}")


# Sec 7: run records, the originals' records, checkpoints and sealed files only; no window is read.
def read_records() -> dict[str, dict]:
    found = {}
    for path in sorted(D.RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("tier") != D.TAG or "run_id" not in record:
            print(f"  skipped {path.name}: not a {D.TAG} run record (tier {record.get('tier')!r})")
            continue
        found[record["run_id"]] = record
    return found


def completeness(records: dict[str, dict], grid_index: int, payloads: dict) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in D.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record, grid_index, payloads)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (index {PLAN_INDEX[run.run_id]}): {'; '.join(issues)}")
    return {"n_planned": len(D.PLAN), "n_records": sum(r["records"] for r in by_arm.values()),
            "n_clean": sum(r["clean"] for r in by_arm.values()), "by_arm": by_arm, "problems": problems,
            "complete": not problems}


def original_check(grid_index: int, payloads: dict) -> list[str]:
    """Every original's record, pool, checkpoint and sealed prediction files, through the driver's own check."""
    return [f"{run.run_id}: {p}" for run in D.PLAN
            for p in D.twin_problems(run, grid_index, pool_names(run.fold, payloads), payloads)]


def section7_markdown(records: dict, gap: dict, originals: list[str]) -> str:
    present = [records[r.run_id] for r in D.PLAN if r.run_id in records]
    devices = ", ".join(f"{d} {n}" for d, n in sorted(Counter(r.get("device") for r in present).items())) or "none"
    queues = ", ".join(f"{q} {n}" for q, n in sorted(Counter(r.get("pbs_queue") for r in present).items())) or "none"
    tf32_off = sum(not any((r.get("tf32") or {"missing": True}).values()) for r in present)
    by_arm = "; ".join(f"{arm} {row['clean']} of {row['planned']}" for arm, row in gap["by_arm"].items())
    diagnostic = (f"`{REPORT_MD.relative_to(REPO)}`, read by hand against sec 5" if REPORT_MD.exists()
                  else "not yet written")
    lines = [
        f"- **Runs.** {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean ({by_arm}). Every clean "
        f"run applied {D.REGISTERED_STEPS:,} updates with none withheld and no non-finite steps; TF32 off on "
        f"{tf32_off}; devices {devices}; PBS queues {queues}.",
        f"- **Switches.** {D.SWITCHES} on every clean run, in the record and its config; each config equals its "
        "original's but for the switches, and fold 1–9 exclusions are as registered.",
        f"- **Originals.** The 60 originals (Bb and Zb; fold 0 from the bird study): records, pools, checkpoints and "
        "sealed prediction files " + ("as registered." if not originals else "differ: " + "; ".join(originals[:5]) + "."),
        f"- **Validation diagnostic.** {diagnostic}.",
        "- **Test-partition reads.** None: every record has opened_test_partition false, and the diagnostic reads "
        "training and validation windows only.",
        f"- **Deferred or missing.** {'None.' if gap['complete'] else '; '.join(gap['problems'])}",
        "- **Known before the seal.** This diagnostic, including validation losses on the held-out birds.",
    ]
    return "\n".join(lines)


def sec7() -> None:
    """Completeness, the switches, the originals and the sec 7 block, from records alone. Opens no window."""
    payloads = D.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    records = read_records()
    gap = completeness(records, grid_index, payloads)
    originals = original_check(grid_index, payloads)
    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<6} {row['records']:>2} of {row['planned']:>2}, {row['clean']:>2} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print("originals        : " + ("as registered" if not originals else "\n  " + "\n  ".join(originals[:20])))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(records, gap, originals))
    if not gap["complete"] or originals:
        raise SystemExit("FAIL: the record check found a problem; sec 7 must not be filled until it passes")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int, choices=list(D.FOLDS), help="one fold's twelve models to OUT/fold{k}.json")
    parser.add_argument("--dry-run", action="store_true", help="with --fold: samples and guards, no model; writes nothing")
    parser.add_argument("--report", action="store_true", help="merge the ten fold files into the report")
    parser.add_argument("--sec7", action="store_true", help="the record check and the sec 7 block for the freeze")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--max-val", type=int, default=MAX_VAL)
    args = parser.parse_args()
    if args.sec7:
        return sec7()
    if args.report:
        return report()
    if args.fold is None:
        parser.error("--fold, --report or --sec7")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    result = run_fold(args.fold, n_train=args.n_train, max_val=args.max_val, dry_run=args.dry_run) | {"tf32": flags}
    if args.dry_run:
        print(json.dumps({k: result[k] for k in ("samples", "zero_loss")}, indent=1))
        print(f"dry run done in {result['seconds']:.0f}s; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"fold{args.fold}.json"
    path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {path} ({result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
