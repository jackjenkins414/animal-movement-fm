# GenAI reference: I outlined a validation only check of our repaired Tier 2-4 transformers before sealing; compare them with their unrepaired twins 
# on fixed train/val samples using loss vs zero, last token sensitivity and |pred| / |actual|, with no test data or pass/fail; GenAI wrote it and I 
# checked it.
"""
Validation diagnostic for the repaired encoder reruns of the Tier 2-4 pooled plain trunk (exploratory; no training; training and validation windows 
only). Freeze: `preregistration/tier4_repaired_exploratory_freeze.md` sec 5.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
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

import tier3_sealed_eval as E3  # noqa: E402
import tier4_repaired_train as R  # noqa: E402
import transformer_collapse as TC  # noqa: E402

from animal_movement_fm.data.enu import FEATURES  # noqa: E402
from animal_movement_fm.data.split import load_split  # noqa: E402
from animal_movement_fm.train.loss import normalised_huber  # noqa: E402

T, env = R.T, R.env

OUT = REPO / "results" / "tier4_repaired_diagnostic"
REPORT_JSON = OUT / "tier4_repaired_diagnostic.json"
REPORT_MD = OUT / "tier4_repaired_diagnostic.md"
FREEZE = REPO / "preregistration" / R.FREEZE

PARTITIONS = ("train", "val")  # never test
N_TRAIN = 20_000  # A8 draws from the arm's training pool, shared by its runs and its twin's
MAX_VAL = 3_000  # validation windows per held-out species, uniform, the same for every arm
SAMPLE_SEED = TC.SAMPLE_SEED  # 2026; validation windows are seeded by species through TC.validation_sample
CHUNK = 4_096
LAST_LOGGED = 5  # logged points (one batch every 500 updates) in the end-of-training median
D_EAST = FEATURES.index("d_east_scaled")
SHORT = {"larus_fuscus": "gull", "canis_lupus": "wolf", "mirounga_angustirostris": "NES"}
DEFAULTS = {name: R.TrainConfig.__dataclass_fields__[name].default for name in R.SWITCHES}
PLAN_INDEX = {run.run_id: i for i, run in enumerate(R.PLAN)}


# Guards.
@contextlib.contextmanager
def audited_loader(log: list[dict]):
    """
    T._load_species_windows wrapped for the block: train or val only, and refused before anything is read if any
    individual it would build is in that species' test partition. SpeciesWindows keeps an index per window, not an
    individual id, and windows are built only from the ids passed in, so this is the window-level check. Each call is
    appended to `log`.
    """
    original = T._load_species_windows

    def load(stem, split_stem, partition, individuals=None):
        if partition not in PARTITIONS:
            raise SystemExit(
                f"FAIL: this diagnostic reads train and val only, never {partition!r}; "
                "test is read by the sealed evaluator alone"
            )
        split = load_split(T.SPLITS / f"{split_stem}_split_seed42.json")
        wanted = getattr(split, partition) if individuals is None else individuals
        TC.check_ids(stem, wanted, split.test)
        windows = original(stem, split_stem, partition, individuals)
        log.append({"species": stem, "partition": partition, "individuals": int(len(wanted)),
                    "windows": int(windows.n_windows), "test_individuals": 0})
        return windows

    T._load_species_windows = load
    try:
        yield log
    finally:
        T._load_species_windows = original


def pool_names(run, payloads: dict) -> list[str]:
    """The arm's registered pool species, read from the registered files without loading any window."""
    if run.tier == "2":
        return sorted(stem for stem, *_ in T.SOURCES)
    node = "source_pool" if run.tier == "3" else R.T4.pool_node(run.pool_run)
    weights = payloads[run.tier]
    for part in node.split("."):
        weights = weights[part]
    return sorted(weights)


def run_problems(run, record: dict | None, grid_index: int, payloads: dict) -> list[str]:
    """What stops one planned run counting as finished cleanly: its record against the plan, the switches and budget."""
    if record is None:
        return ["no record"]
    problems = [
        f"{key} {record.get(key)!r}, expected {want!r}"
        for key, want in (
            ("plan_index", PLAN_INDEX[run.run_id]),
            ("arm", run.arm),
            ("seed", run.seed),
            ("tier", R.TAG),
            ("twin", run.twin.run_id if run.twin else None),
            ("switches", R.SWITCHES),
            ("queue", R.QUEUE),
            ("exploratory", True),
            ("opened_test_partition", False),
        )
        if record.get(key) != want
    ]
    if not re.fullmatch(rf"{R.QUEUE}(-exec)?", str(record.get("pbs_queue"))):
        problems.append(f"pbs_queue {record.get('pbs_queue')!r}")
    names = pool_names(run, payloads)
    if sorted(record.get("pool_species") or []) != names:
        problems.append("pool_species differ from the registered pool")
    expected = R._json(asdict(R.build_config(run, grid_index, R.REGISTERED_STEPS, names, payloads)))
    if R.with_defaults(record.get("config") or {}) != expected:
        problems.append("config differs from its recipe's with the switches")
    if any((record.get("tf32") or {"missing": True}).values()):
        problems.append(f"tf32 {record.get('tf32')}")
    if record.get("applied_updates") != R.REGISTERED_STEPS or (record.get("config") or {}).get("steps") != R.REGISTERED_STEPS:
        problems.append(f"applied {record.get('applied_updates')} of {(record.get('config') or {}).get('steps')}")
    if record.get("skipped_steps") != 0 or record.get("withheld_fraction") != 0:
        problems.append(f"withheld {record.get('skipped_steps')}")
    if record.get("nonfinite_steps"):
        problems.append(f"{len(record['nonfinite_steps'])} non-finite steps")
    if not (R.RESULTS / str(record.get("reported_checkpoint"))).exists():
        problems.append("no reported checkpoint")
    return problems


# Runs.
def entries(arm: str) -> list[dict]:
    """The arm's five runs and, at the same seeds, its twin's five, with where each record and checkpoint live."""
    if arm not in R.ARMS:
        raise SystemExit(f"FAIL: {arm} is not an arm of this study; it has {', '.join(R.ARMS)}")
    out = []
    for run in (r for r in R.PLAN if r.arm == arm):
        out.append({"label": arm, "role": "repaired", "seed": run.seed, "run_id": run.run_id, "results": R.RESULTS,
                    "run": run})
        if run.twin is not None:
            out.append({"label": run.twin.arm, "role": "twin", "seed": run.seed, "run_id": run.twin.run_id,
                        "results": R.TWIN_RESULTS[run.tier], "run": run})
    return out


def load_model(entry: dict, grid_index: int, payloads: dict):
    """(model in eval mode, record), refused on any problem with the record or the checkpoint's switches."""
    run, repaired = entry["run"], entry["role"] == "repaired"
    path = entry["results"] / f"{entry['run_id']}.json"
    record = json.loads(path.read_text()) if path.exists() else None
    if repaired:
        problems = run_problems(run, record, grid_index, payloads)
    else:
        problems = R.twin_problems(run, grid_index, pool_names(run, payloads), payloads)
    loaded = None if problems else E3.load_reported(entry["results"], entry["run_id"])
    if not problems and loaded is None:
        problems.append("no reported checkpoint")
    if loaded is not None:
        state = loaded[1]
        switches = {name: state["config"].get(name, DEFAULTS[name]) for name in R.SWITCHES}
        if switches != (R.SWITCHES if repaired else DEFAULTS):
            problems.append(f"checkpoint switches {switches}")
        if repaired and state.get("switches") != R.SWITCHES:
            problems.append(f"checkpoint records switches {state.get('switches')}")
    if problems:
        raise SystemExit(f"FAIL: {entry['run_id']}: " + "; ".join(problems))
    return loaded[0], record


# Windows and samples.
def training_sample(arm: str, payloads: dict, n: int = N_TRAIN, seed: int = SAMPLE_SEED) -> dict:
    """n draws under training's A8 rule (SourcePool.sample) from the arm's registered pool, at a fixed seed."""
    run = R.Run(arm, R.SEEDS[0])
    source_pool, node = R.training_source(run, payloads)
    R.check_window_counts(source_pool, payloads)
    if list(source_pool.names) != pool_names(run, payloads):
        raise SystemExit(f"FAIL: {arm}'s pool {source_pool.names} is not its registered node's")
    features, targets, scale = source_pool.sample(np.random.default_rng(seed), n)
    return {"node": node, "species": list(source_pool.names), "seed": seed,
            "features": features.astype(np.float32), "targets": targets, "scale": scale}


def validation_windows(species: str, cap: int = MAX_VAL) -> dict:
    """Up to `cap` of one held-out species' validation windows, the same for every arm, and its A4 boundary."""
    stem, split_stem, profile_key, *_ = T.held_out_entry(species)
    windows = T._load_species_windows(stem, split_stem, "val")
    ids = TC.validation_sample(windows.n_windows, cap, stem)
    features, targets, scale = windows.gather(ids)
    return {"features": features.astype(np.float32), "targets": targets, "scale": scale,
            "available": int(windows.n_windows), "boundary_m": float(T.boundary_m(profile_key))}


# Measures.
def predict(model, features: np.ndarray, chunk: int = CHUNK) -> np.ndarray:
    """(n, 2) output in model units (metres / window scale)."""
    out = []
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            out.append(model(torch.as_tensor(features[start : start + chunk], dtype=torch.float32)).double().numpy())
    return np.concatenate(out) if out else np.empty((0, 2))


def loss_over_zero(pred: np.ndarray, targets: np.ndarray) -> dict:
    """normalised_huber of the prediction and of the zero predictor against the scaled targets, and their ratio."""
    y = torch.as_tensor(targets, dtype=torch.float64)
    loss = float(normalised_huber(torch.as_tensor(pred, dtype=torch.float64), y))
    zero = float(normalised_huber(torch.zeros_like(y), y))
    return {"loss": loss, "zero": zero, "ratio": loss / zero}


def logged_loss(record: dict, zero: float) -> dict:
    """The record's final logged loss and the median of its last LAST_LOGGED logged points, each against `zero`."""
    history = list(record.get("loss_history") or [])
    final, last = record.get("final_loss"), TC._median(history[-LAST_LOGGED:])
    return {"points": len(history), "final": final, f"last_{LAST_LOGGED}_median": last,
            "final_over_zero": None if final is None else final / zero,
            "last_over_zero": None if last is None else last / zero}


def _ratio(a, b):
    return None if a is None or not b else a / b


def sensitivity(model, features: np.ndarray, base: np.ndarray) -> dict:
    """
    Median |change in output| (model units) when the last token is swapped for the previous window's (np.roll by one)
    and when its d_east_scaled gets +1 on the features as they arrive; median |output| beside them.
    """
    swapped = features.copy()
    swapped[:, -1] = np.roll(features[:, -1], 1, axis=0)
    bumped = features.copy()
    bumped[:, -1, D_EAST] += 1.0
    out = {"median_abs_output": TC._median(np.linalg.norm(base, axis=-1))}
    for name, perturbed in (("last_swapped", swapped), ("last_plus_1", bumped)):
        out[name] = TC._median(np.linalg.norm(predict(model, perturbed) - base, axis=-1))
        out[f"{name}_over_output"] = _ratio(out[name], out["median_abs_output"])
    return out


def magnitudes(pred: np.ndarray, targets: np.ndarray, scale: np.ndarray, boundary_m: float) -> dict:
    """
    |pred| and |actual| in metres (model units x window scale). On travel windows (|actual| >= the species' A4
    boundary, as the scoring) the median |pred| / |actual| and medians of each; over every window, the medians alone.
    """
    pred_m = np.linalg.norm(pred, axis=-1) * scale
    actual_m = np.linalg.norm(targets, axis=-1) * scale
    travel = actual_m >= boundary_m
    out = {"n_windows": int(len(pred_m)), "n_travel": int(travel.sum()),
           "median_pred_m_all": TC._median(pred_m), "median_actual_m_all": TC._median(actual_m),
           "ratio": None, "median_pred_m": None, "median_actual_m": None}
    if travel.any():
        out |= {"ratio": TC._median(pred_m[travel] / actual_m[travel]), "median_pred_m": TC._median(pred_m[travel]),
                "median_actual_m": TC._median(actual_m[travel])}
    return out


def score(model, record: dict, train: dict, val: dict[str, dict]) -> dict:
    out = {"train": loss_over_zero(predict(model, train["features"]), train["targets"]), "val": {}}
    out["logged"] = logged_loss(record, out["train"]["zero"])
    for species, batch in val.items():
        pred = predict(model, batch["features"])
        out["val"][species] = {
            "loss": loss_over_zero(pred, batch["targets"]),
            "sensitivity": sensitivity(model, batch["features"], pred),
            "magnitudes": magnitudes(pred, batch["targets"], batch["scale"], batch["boundary_m"]),
        }
    return out


# One arm.
def run_arm(arm: str, *, n_train: int = N_TRAIN, max_val: int = MAX_VAL, dry_run: bool = False) -> dict:
    """One arm's runs and its twin's on the same samples; with `dry_run`, the samples and guards without any model."""
    started = time.time()
    group = entries(arm)
    payloads = R.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    loaded = {} if dry_run else {e["run_id"]: load_model(e, grid_index, payloads) for e in group}  # fail before building

    loads: list[dict] = []
    with audited_loader(loads):
        train = training_sample(arm, payloads, n_train)
        val = {species: validation_windows(species, max_val) for species in R.Run(arm, R.SEEDS[0]).scored_on}
    print(f"{arm}: {len(train['targets']):,} training draws over {len(train['species'])} species [{train['node']}]; "
          + "; ".join(f"{SHORT[s]} {len(b['targets']):,} of {b['available']:,} val windows" for s, b in val.items())
          + f"; {len(loads)} loads, none from a test partition; built in {time.time() - started:.0f}s", flush=True)

    out = {
        "arm": arm, "twin": next((e["label"] for e in group if e["role"] == "twin"), None),
        "scored_on": list(val), "exploratory": True, "opened_test_partition": False, "dry_run": dry_run,
        "samples": {"n_train": int(len(train["targets"])), "train_seed": train["seed"], "train_node": train["node"],
                    "train_species": train["species"], "max_val": max_val, "val_seed": SAMPLE_SEED,
                    "val": {s: {"windows": int(len(b["targets"])), "available": b["available"],
                                "boundary_m": b["boundary_m"]} for s, b in val.items()}},
        "loads": loads,
        "zero_loss": {"train": loss_over_zero(np.zeros_like(train["targets"]), train["targets"])["zero"],
                      "val": {s: loss_over_zero(np.zeros_like(b["targets"]), b["targets"])["zero"] for s, b in val.items()}},
        "models": {},
    }
    for e in (e for e in group if e["run_id"] in loaded):
        model, record = loaded[e["run_id"]]
        t0 = time.time()
        out["models"][e["run_id"]] = {"label": e["label"], "role": e["role"], "seed": e["seed"]} | score(
            model, record, train, val)
        print(f"  {e['run_id']}: done in {time.time() - t0:.0f}s", flush=True)
    out["seconds"] = round(time.time() - started, 1)
    return out


# Report.
VAL_METRICS = {
    "loss_over_zero": ("loss", "ratio"),
    "median_abs_output": ("sensitivity", "median_abs_output"),
    "last_swapped": ("sensitivity", "last_swapped"),
    "last_swapped_over_output": ("sensitivity", "last_swapped_over_output"),
    "last_plus_1": ("sensitivity", "last_plus_1"),
    "last_plus_1_over_output": ("sensitivity", "last_plus_1_over_output"),
    "ratio": ("magnitudes", "ratio"),
    "median_pred_m": ("magnitudes", "median_pred_m"),
    "median_actual_m": ("magnitudes", "median_actual_m"),
    "median_pred_m_all": ("magnitudes", "median_pred_m_all"),
    "median_actual_m_all": ("magnitudes", "median_actual_m_all"),
    "n_travel": ("magnitudes", "n_travel"),
}


def summarise(arms: list[dict]) -> dict:
    """Per arm and per twin, in plan order (each arm, then its twin): every measure's median over its seeds."""
    rows = {}
    for arm in arms:
        by_label: dict[str, list[dict]] = {}
        for model in arm["models"].values():
            by_label.setdefault(model["label"], []).append(model)
        for label, models in by_label.items():
            def med(*keys, models=models):
                return TC._median([TC._get(m, *keys) for m in models])
            rows[label] = {
                "arm": arm["arm"], "role": models[0]["role"], "seeds": sorted(m["seed"] for m in models),
                "train_loss_over_zero": med("train", "ratio"),
                "logged_final_over_zero": med("logged", "final_over_zero"),
                "logged_last_over_zero": med("logged", "last_over_zero"),
                "val": {species: {name: med("val", species, *keys) for name, keys in VAL_METRICS.items()}
                        for species in arm["scored_on"]},
            }
    return rows


def reading_rule(path: Path = FREEZE) -> str:
    """The freeze's sec 5 as committed, quoted rather than restated."""
    match = re.search(r"^## 5\.[^\n]*\n(.*?)(?=^## )", path.read_text(), flags=re.S | re.M)
    return match.group(1).strip() if match else "(sec 5 not found in the freeze)"


def write_markdown(arms: list[dict], summary: dict, path: Path) -> None:
    m = TC._m
    held_out = list(R.HELD_OUT)
    lines = [
        "# Repaired transformer on the Tier 2–4 pools: validation diagnostic (exploratory)", "",
        f"The 25 runs of `tier4_repaired_train.py` and their 20 unrepaired twins (BdTF has none), from their reported "
        f"checkpoints; no training, and no test partition read. Samples: {N_TRAIN:,} A8 draws from each arm's training "
        f"pool (seed {SAMPLE_SEED}, shared by the arm and its twin) and up to {MAX_VAL:,} validation windows per "
        "held-out species (the same for every arm). Every value is a median over seeds; model units are metres / "
        "window scale. No pass/fail.", "",
        "Freeze sec 5, applied by hand:", "",
        "> " + reading_rule().replace("\n", "\n> "), "",
        "## 1. Loss against the zero predictor (model / zero)", "",
        "| run | role | seeds | training sample | logged, final | logged, last 5 | "
        + " | ".join(f"val {SHORT[s]}" for s in held_out) + " |",
        "|---|---|---:|---:|---:|---:|" + "---:|" * len(held_out),
    ]
    for label, row in summary.items():
        val = " | ".join(m([TC._get(row, "val", s, "loss_over_zero")]) for s in held_out)
        lines.append(f"| {label} | {row['role']} | {len(row['seeds'])} | {m([row['train_loss_over_zero']])} | "
                     f"{m([row['logged_final_over_zero']])} | {m([row['logged_last_over_zero']])} | {val} |")
    lines += ["", "The logged losses are single training batches; each is divided by the zero predictor's loss on the "
              "training sample.", "",
              "## 2. Input sensitivity on validation (model units)", "",
              "| run | species | median \\|out\\| | last swapped | swapped / \\|out\\| | last +1 | +1 / \\|out\\| |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for label, row in summary.items():
        for species, v in row["val"].items():
            lines.append(f"| {label} | {SHORT[species]} | {m([v['median_abs_output']])} | {m([v['last_swapped']])} | "
                         f"{m([v['last_swapped_over_output']])} | {m([v['last_plus_1']])} | "
                         f"{m([v['last_plus_1_over_output']])} |")
    lines += ["", "## 3. |pred| / |actual| on validation (metres)", "",
              "| run | species | travel windows | \\|pred\\| / \\|actual\\|, travel | median \\|pred\\|, travel | "
              "median \\|actual\\|, travel | median \\|pred\\|, all | median \\|actual\\|, all |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for label, row in summary.items():
        for species, v in row["val"].items():
            lines.append(f"| {label} | {SHORT[species]} | {m([v['n_travel']])} | {m([v['ratio']])} | "
                         f"{m([v['median_pred_m']])} | {m([v['median_actual_m']])} | {m([v['median_pred_m_all']])} | "
                         f"{m([v['median_actual_m_all']])} |")
    lines += ["", "## Samples", "", "| arm | training pool | species | draws | validation windows (of available) | "
              "loads | from test |", "|---|---|---:|---:|---|---:|---:|"]
    for arm in arms:
        s = arm["samples"]
        val = "; ".join(f"{SHORT[k]} {v['windows']:,} of {v['available']:,}" for k, v in s["val"].items())
        lines.append(f"| {arm['arm']} | {s['train_node']} | {len(s['train_species'])} | {s['n_train']:,} | {val} | "
                     f"{len(arm['loads'])} | {sum(load['test_individuals'] for load in arm['loads'])} |")
    path.write_text("\n".join(lines) + "\n")


def report() -> None:
    files = [OUT / f"{arm}.json" for arm in R.ARMS]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing {', '.join(missing)}; run every arm first")
    arms = [json.loads(p.read_text()) for p in files]
    if any(arm.get("dry_run") for arm in arms):
        raise SystemExit("FAIL: an arm file was written by a dry run")
    summary = summarise(arms)
    REPORT_JSON.write_text(json.dumps({"exploratory": True, "opened_test_partition": False, "summary": summary,
                                       "arms": arms}, indent=1, default=str) + "\n")
    write_markdown(arms, summary, REPORT_MD)
    print(f"wrote {REPORT_JSON}\nwrote {REPORT_MD}")


# Sec 7: run records, the twins' records and checkpoints only; no window is read.
def read_records() -> dict[str, dict]:
    found = {}
    for path in sorted(R.RESULTS.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("tier") != R.TAG or "run_id" not in record:
            print(f"  skipped {path.name}: not a {R.TAG} run record (tier {record.get('tier')!r})")
            continue
        found[record["run_id"]] = record
    return found


def completeness(records: dict[str, dict], grid_index: int, payloads: dict) -> dict:
    by_arm: dict[str, dict] = {}
    problems = []
    for run in R.PLAN:
        row = by_arm.setdefault(run.arm, {"planned": 0, "records": 0, "clean": 0})
        row["planned"] += 1
        record = records.get(run.run_id)
        row["records"] += record is not None
        issues = run_problems(run, record, grid_index, payloads)
        row["clean"] += not issues
        if issues:
            problems.append(f"{run.run_id} (index {PLAN_INDEX[run.run_id]}): {'; '.join(issues)}")
    return {"n_planned": len(R.PLAN), "n_records": sum(r["records"] for r in by_arm.values()),
            "n_clean": sum(r["clean"] for r in by_arm.values()), "by_arm": by_arm, "problems": problems,
            "complete": not problems}


def twin_check(grid_index: int, payloads: dict) -> list[str]:
    """Every twin's record, checkpoint and comparator files, through the driver's own check."""
    return [f"{run.run_id}: {p}" for run in R.PLAN if run.twin is not None
            for p in R.twin_problems(run, grid_index, pool_names(run, payloads), payloads)]


def section7_markdown(records: dict, gap: dict, twins: list[str]) -> str:
    present = [records[r.run_id] for r in R.PLAN if r.run_id in records]
    devices = ", ".join(f"{d} {n}" for d, n in sorted(Counter(r.get("device") for r in present).items())) or "none"
    queues = ", ".join(f"{q} {n}" for q, n in sorted(Counter(r.get("pbs_queue") for r in present).items())) or "none"
    tf32_off = sum(not any((r.get("tf32") or {"missing": True}).values()) for r in present)
    by_arm = "; ".join(f"{arm} {row['clean']} of {row['planned']}" for arm, row in gap["by_arm"].items())
    twin_runs = sorted({f"tier {run.tier} {run.twin.arm}" for run in R.PLAN if run.twin is not None})
    n_twins = sum(run.twin is not None for run in R.PLAN)
    diagnostic = (f"`{REPORT_MD.relative_to(REPO)}`, read by hand against sec 5" if REPORT_MD.exists()
                  else "not yet written")
    lines = [
        "| | |", "|---|---|",
        f"| runs | {gap['n_records']} of {gap['n_planned']} recorded, {gap['n_clean']} clean ({by_arm}) |",
        f"| training | {R.REGISTERED_STEPS:,} updates applied, none withheld, no non-finite steps on every clean run; "
        f"TF32 off on {tf32_off}; devices {devices} |",
        f"| switches | {R.SWITCHES} on every clean run, in the record and its config; each config equals its recipe's "
        "but for the switches |",
        f"| PBS queues | {queues} |",
        f"| twins | {n_twins} runs ({', '.join(twin_runs)}): records, checkpoints and comparator files "
        + ("as registered" if not twins else "differ: " + "; ".join(twins[:5])) + " |",
        f"| validation diagnostic | {diagnostic} |",
        "| test-partition reads | none: every record has opened_test_partition false, and the diagnostic reads training "
        "and validation windows only |",
        f"| deferred or missing | {'none' if gap['complete'] else '; '.join(gap['problems'])} |",
    ]
    return "\n".join(lines)


def sec7() -> None:
    """Completeness, the switches, the twins and the sec 7 block, from records alone. Opens no window."""
    payloads = R.load_payloads()
    grid_index, _ = T.selected_grid_index(None)
    records = read_records()
    gap = completeness(records, grid_index, payloads)
    twins = twin_check(grid_index, payloads)
    print(f"completeness     : {gap['n_records']} of {gap['n_planned']} records, {gap['n_clean']} clean")
    for arm, row in gap["by_arm"].items():
        print(f"  {arm:<6} {row['records']:>2} of {row['planned']:>2}, {row['clean']:>2} clean")
    for problem in gap["problems"][:20]:
        print(f"  {problem}")
    print("twins            : " + ("as registered" if not twins else "\n  " + "\n  ".join(twins[:20])))
    print("\n--- paste into the freeze sec 7 ---\n")
    print(section7_markdown(records, gap, twins))
    if not gap["complete"] or twins:
        raise SystemExit("FAIL: the record check found a problem; sec 7 must not be filled until it passes")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=list(R.ARMS), help="one arm and its twin to OUT/{arm}.json")
    parser.add_argument("--dry-run", action="store_true", help="with --arm: samples and guards, no model; writes nothing")
    parser.add_argument("--report", action="store_true", help="merge the five arm files into the report")
    parser.add_argument("--sec7", action="store_true", help="the record check and the draft sec 7 block")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--max-val", type=int, default=MAX_VAL)
    args = parser.parse_args()
    if args.sec7:
        return sec7()
    if args.report:
        return report()
    if args.arm is None:
        parser.error("--arm, --report or --sec7")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    result = run_arm(args.arm, n_train=args.n_train, max_val=args.max_val, dry_run=args.dry_run) | {"tf32": flags}
    if args.dry_run:
        print(json.dumps({k: result[k] for k in ("samples", "zero_loss")}, indent=1))
        print(f"dry run done in {result['seconds']:.0f}s; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.arm}.json"
    path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {path} ({result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
