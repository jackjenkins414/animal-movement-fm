# GenAI reference: I outline this repaired transformer rotation driver; rerun Bb and Zb with the repair switches across all 10 folds/seeds, keep 
# the same pools/configs/initial weights as their originals, and include dry run/calibration guards with FP32 etc; GenAI wrote it and I checked it.
"""
Train the repaired transformer on the rotating held-out birds, one model per PBS subjob (exploratory).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import bird_rotation_train as RT  # noqa: E402
import tier4_repaired_train as R  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.loss import normalised_huber  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402

BT, T, env = RT.BT, RT.T, R.env

RESULTS = REPO / "results" / "birds_rotation_repaired"
CHECKPOINTS = REPO / "checkpoints" / "birds_rotation_repaired"
CALIBRATION = REPO / "results" / "birds_rotation_repaired_calibration"
FREEZE = "birds_rotation_repaired_exploratory_freeze.md"
TAG = "birds_rotation_repaired"
QUEUE = R.QUEUE  # gpuhopper

REGISTERED_STEPS = RT.REGISTERED_STEPS  # 20,000 applied updates at batch 256, the originals' budget
SEEDS = RT.SEEDS  # 0-2, as the rotation's neural arms
FOLDS = tuple(range(10))  # fold 0 included; its originals are the bird study's
EXPECTED_EXTRA = R.EXPECTED_EXTRA  # the Pre-LN stack's final LayerNorm

# The repaired encoder, pinned here and checked against the Tier 2-4 rerun's pin (itself checked against BbF and the
# repair probe's V4), so "repaired" means the same thing in every study.
SWITCHES = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
if R.SWITCHES != SWITCHES:
    raise RuntimeError(f"tier4_repaired_train pins {R.SWITCHES}; this study froze {SWITCHES}")

# arm -> the original it repeats with the switches on (the rotation's arm name, which is also the bird study's)
ARMS: dict[str, str] = {"BbF6": "Bb", "ZbF6": "Zb"}
if any(BT.ARMS[twin]["pool"] != "main" for twin in ARMS.values()):
    raise RuntimeError("both originals are pooled arms on the bird study's main pool in fold 0")

# Sealed prediction files the evaluator reads; the driver checks them for existence only and never reads them.
ROTATION_PREDICTIONS = REPO / "results" / "birds_rotation_sealed" / "predictions"
BIRD_PREDICTIONS = REPO / "results" / "birds_sealed" / "predictions"
TIER2_PREDICTIONS = R.SEALED_PREDICTIONS["2"]  # the gull's P, carried by the bird study


@dataclass(frozen=True)
class Run:
    arm: str
    fold: int
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.arm}_f{self.fold}_s{self.seed}"

    @property
    def twin(self):
        """The original this run repeats with the switches on: the bird study's run in fold 0, the rotation's after."""
        if self.fold == 0:
            return BT.Run(ARMS[self.arm], "", self.seed)
        return RT.Run(ARMS[self.arm], self.fold, "", self.seed)

    @property
    def twin_study(self) -> str:
        return "birds" if self.fold == 0 else "birds_rotation"

    @property
    def twin_results(self) -> Path:
        return BT.RESULTS if self.fold == 0 else RT.RESULTS

    @property
    def scored_on(self) -> tuple[str, ...]:
        return tuple(BT.HELD_OUT) if self.fold == 0 else RT.FOLDS[self.fold]


def build_plan() -> list[Run]:
    """The plan. Position is the PBS array index: append, never insert or reorder."""
    return [Run(arm, k, s) for arm in ARMS for k in FOLDS for s in SEEDS]


PLAN = build_plan()
# submission blocks of at most ten; every block is on gpuhopper
BLOCKS: tuple[tuple[int, int, str], ...] = tuple((first, first + 9, PLAN[first].arm) for first in range(0, 60, 10))
CALIBRATION_INDICES = tuple(PLAN.index(Run(arm, 9, 0)) for arm in ARMS)  # fold 9, the largest pool
if len(PLAN) != 60 or len({run.run_id for run in PLAN}) != 60:
    raise RuntimeError(f"the plan has {len(PLAN)} runs; 60 are planned, two arms by ten folds by three seeds")
_OTHER_IDS = {r.run_id for plan in (RT.PLAN, BT.PLAN, R.PLAN, R.bbf.PLAN, R.env.PLAN, R.TR.PLAN) for r in plan}
if {run.run_id for run in PLAN} & _OTHER_IDS:
    raise RuntimeError("run ids must be distinct from every rotation, bird, Tier 2-4 rerun, 1 h and repair-probe run id")
if any(run.twin not in (BT.PLAN if run.fold == 0 else RT.PLAN) for run in PLAN):
    raise RuntimeError("every twin must be a run of the bird study's or the rotation's registered plan")
if any(BT.counterpart(run.twin).uses_validation_selection for run in PLAN):
    raise RuntimeError("every original reported its final checkpoint; so does every run here")


# Guards.
def freeze_guard() -> dict:
    """This study's freeze exists, is tracked and is unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    previous = R.S2.FREEZE
    R.S2.FREEZE = REPO / "preregistration" / FREEZE
    try:
        return R.S2.assert_freeze_committed()
    finally:
        R.S2.FREEZE = previous


def mode_guards(run: Run, *, calibrate: bool, steps: int | None) -> dict | None:
    """Queue always; the committed freeze and the registered budget except in --calibrate. Returns the attestation."""
    R.queue_guard(run)
    if calibrate:
        return None
    attestation = freeze_guard()
    if steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {steps}")
    return attestation


# Pools and configurations.
def load_payloads() -> dict:
    """The bird study's registered pools (fold 0) and the registered rotation folds (1-9), each checked on load."""
    return {"birds": BT.load_registered_pool(), "rotation": RT.load_registered_folds()}


def payload_of(run: Run, payloads: dict) -> dict:
    return payloads["birds" if run.fold == 0 else "rotation"]


def training_source(run: Run, payloads: dict) -> tuple[SourcePool, str]:
    """The original's own pool, through the loader that built it. Training individuals only."""
    if run.fold == 0:
        return BT.pooled_source(run.twin, payloads["birds"])
    return RT.pooled_source(run.twin, payloads["rotation"])


def check_window_counts(run: Run, source_pool: SourcePool, payloads: dict) -> None:
    """Every species' built windows equal its registered count (the rotation's are counted after its exclusions)."""
    registered = payload_of(run, payloads)["train_window_counts"]
    drift = [
        f"{name}: built {source_pool.species[name].n_windows:,}, registered {registered[name]:,}"
        for name in source_pool.names
        if source_pool.species[name].n_windows != registered[name]
    ]
    if drift:
        raise SystemExit(f"FAIL: fold {run.fold} training windows differ from the registered counts\n  "
                         + "\n  ".join(drift))


def excluded_individuals(run: Run, pool_names, payloads: dict) -> dict[str, list]:
    """The rotation's registered exclusions in the pool; fold 0's main pool held none (the goose's six are the goose's)."""
    if run.fold == 0:
        return {}
    return RT.excluded_in(pool_names, payloads["rotation"])


def recipe_config(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> TrainConfig:
    """The original's configuration, made by the bird study's builder exactly as its driver made it."""
    return BT.build_config(run.twin, grid_index, steps, pool_names, payload_of(run, payloads))


def build_config(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> TrainConfig:
    """The original's configuration with the repaired encoder's switches; nothing else."""
    return replace(recipe_config(run, grid_index, steps, pool_names, payloads), **SWITCHES)


def config_differences(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> dict:
    """The fields in which the run's configuration differs from its original's."""
    recipe = asdict(recipe_config(run, grid_index, steps, pool_names, payloads))
    got = asdict(build_config(run, grid_index, steps, pool_names, payloads))
    return {name: got[name] for name in got if got[name] != recipe[name]}


def initial_state_differences(run: Run, grid_index: int, pool_names, payloads: dict) -> dict:
    """
    The repaired model and its original's, each built after torch.manual_seed(seed) as train() builds them. The switches
    add no randomly initialised parameter, so every shared tensor should be equal and the only extra ones the final
    LayerNorm; for ZbF6 that covers the conditioning path, the mask token and the reconstruction head too.
    """
    built = []
    for config in (recipe_config(run, grid_index, REGISTERED_STEPS, pool_names, payloads),
                   build_config(run, grid_index, REGISTERED_STEPS, pool_names, payloads)):
        torch.manual_seed(config.seed)
        built.append(config.build().state_dict())
    recipe, repaired = built
    return {
        "extra": sorted(set(repaired) - set(recipe)),
        "missing": sorted(set(recipe) - set(repaired)),
        "different": sorted(name for name in recipe if name in repaired and not torch.equal(recipe[name], repaired[name])),
    }


def prediction_files(run: Run) -> list[Path]:
    """P and the original's sealed predictions on each of the fold's held-out birds."""
    if run.fold:
        return [ROTATION_PREDICTIONS / f"{bird}__{tag}.parquet" for bird in run.scored_on for tag in ("P", run.twin.run_id)]
    files = []
    for bird in run.scored_on:
        home = TIER2_PREDICTIONS if bird == BT.HELD_OUT[0] else BIRD_PREDICTIONS  # the gull's P is Tier 2's
        files += [home / f"{bird}__P.parquet", BIRD_PREDICTIONS / f"{bird}__{run.twin.run_id}.parquet"]
    return files


def twin_problems(run: Run, grid_index: int, pool_names, payloads: dict, pool_windows: int | None = None) -> list[str]:
    """
    The original's record and checkpoint on disk against what this run assumes: its configuration equals the recipe's
    (fields absent from it at their defaults), it applied the registered budget on the same pool, and its sealed
    predictions exist. Empty if all hold. Prediction files are checked for existence only and never read.
    """
    problems = []
    twin_id = run.twin.run_id
    path = run.twin_results / f"{twin_id}.json"
    if not path.exists():
        problems.append(f"{twin_id}'s record {path} is missing")
    else:
        record = json.loads(path.read_text())
        expected = R._json(asdict(recipe_config(run, grid_index, REGISTERED_STEPS, pool_names, payloads)))
        if R.with_defaults(record["config"]) != expected:
            problems.append(f"{twin_id}'s recorded config differs from its builder's")
        if record.get("applied_updates") != REGISTERED_STEPS:
            problems.append(f"{twin_id} applied {record.get('applied_updates')} updates")
        if sorted(record.get("pool_species") or []) != sorted(pool_names):
            problems.append(f"{twin_id}'s pool species differ from this run's")
        if pool_windows is not None and record.get("pool_windows") != pool_windows:
            problems.append(f"{twin_id} trained on {record.get('pool_windows')} windows, this run on {pool_windows:,}")
    if not (run.twin_results / f"{twin_id}_reported.pt").exists():
        problems.append(f"{twin_id}_reported.pt is missing")
    problems += [f"{p.name} is missing from {p.parent}" for p in prediction_files(run) if not p.exists()]
    return problems


# Runners.
def run_torch(run: Run, index: int, grid_index: int, source: str, payloads: dict, attestation: dict | None) -> None:
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payloads)
    check_window_counts(run, source_pool, payloads)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], "
          f"built in {time.time() - started:.0f}s", flush=True)
    config = build_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads)
    differences = config_differences(run, grid_index, REGISTERED_STEPS, pool.names, payloads)
    if differences != SWITCHES:
        raise SystemExit(f"FAIL: {run.run_id} differs from its original in {differences}, not the switches alone")
    print(f"switches         : {SWITCHES}; twin {run.twin_study}:{run.twin.run_id}", flush=True)

    result = train(
        config, pool, device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
    )
    print(f"applied updates  : {result.applied_updates:,} of {config.steps:,}; "
          f"withheld {result.skipped_steps:,} ({result.withheld_fraction:.2%})", flush=True)
    selection = {"rule": "final_checkpoint", "step": config.steps}  # every original reported its final checkpoint
    reported = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {"run_id": run.run_id, "config": asdict(config), "grid_index": grid_index, "pool_species": list(pool.names),
         "switches": dict(SWITCHES), "selection": selection, "model": result.model.state_dict()},
        reported,
    )
    record = base_record(run, index, grid_index, source, node, attestation) | {
        "config": asdict(config),
        "excluded_individuals": excluded_individuals(run, pool.names, payloads),
        "n_parameters": result.model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "loss_shares": result.loss_shares,
        "skipped_steps": result.skipped_steps,
        "attempted_batches": result.attempted_batches,
        "applied_updates": result.applied_updates,
        "withheld_fraction": result.withheld_fraction,
        "nonfinite_steps": result.nonfinite_steps,
        "checkpoint_selection": selection,
        "reported_checkpoint": reported.name,
    }
    write_record(run, record)


def probe_losses(model, probe) -> dict:
    """R.probe_losses on the forecast output; ZbF6 and Zb also return their reconstruction, which is not scored here."""
    features, targets = probe
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(features)
        prediction = out[0] if isinstance(out, tuple) else out
        loss = float(normalised_huber(prediction, targets))
        zero = float(normalised_huber(torch.zeros_like(targets), targets))
    model.train(was_training)
    return {"loss": loss, "zero": zero, "ratio": loss / zero}


@contextlib.contextmanager
def forecast_probe():
    """R.calibrate_config with this module's probe_losses, so the full model's tuple output is read as its forecast."""
    previous = R.probe_losses
    R.probe_losses = probe_losses
    try:
        yield
    finally:
        R.probe_losses = previous


def calibrate(run: Run, grid_index: int, payloads: dict) -> None:
    """
    The run and its original at the same seed, each for R.CALIBRATION_STEPS updates on the same pool through
    R.calibrate_config. The original's logged losses are compared step by step with its record, which was written on
    gpuvolta (V100); this calibration runs on an H200, so the comparison is a closeness check, not a bitwise one.
    No record, no checkpoint for either.
    """
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payloads)
    check_window_counts(run, source_pool, payloads)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    built = time.time() - started
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], built in {built:.0f}s",
          flush=True)
    probe = pool.sample(torch.Generator(device=device).manual_seed(R.PROBE_SEED), R.PROBE_WINDOWS)[:2]
    payload = {"run_id": run.run_id, "node": node, "pool_windows": pool.n_windows, "build_seconds": built,
               "switches": dict(SWITCHES), "tf32": env.tf32_flags(), "pbs_queue": env.pbs_queue(),
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    with forecast_probe():
        payload["repaired"] = R.calibrate_config(build_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads),
                                                 pool, device, probe)
        twin = R.calibrate_config(recipe_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads), pool, device,
                                  probe)
    path = run.twin_results / f"{run.twin.run_id}.json"
    twin["record"] = str(path.relative_to(REPO))
    twin["match"] = (R.loss_match(twin["logged_losses"], json.loads(path.read_text())) if path.exists()
                     else {"compared": 0, "note": f"{path} is missing"})
    payload["twin"] = {"run_id": run.twin.run_id, "study": run.twin_study, "trained_on": "gpuvolta (V100)"} | twin
    payload["peak_rss_gb"] = env.peak_rss_gb()
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    out = CALIBRATION / f"{run.run_id}_calibration.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("repaired", "twin")}, indent=2, default=str))
    for label in ("repaired", "twin"):
        block = payload[label]
        print(f"{label:<9}: {block['seconds_per_update']:.4f} s/update, {block['extrapolated_train_hours_20000']:.2f} h "
              f"for 20,000; probe loss / zero {block['probe']['start']['ratio']:.3f} -> {block['probe']['end']['ratio']:.3f}")
    match = payload["twin"]["match"]
    print(f"twin match: {match.get('compared')} logged steps, max |diff| {match.get('max_abs_difference')}, "
          f"exact share {match.get('exact_share')} ({match.get('note', 'V100 record against H200 rerun')})")
    print(f"wrote {out}; calibration only, no run record or checkpoint", flush=True)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str, attestation: dict | None) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "fold": run.fold,
        "species": None,
        "scored_on": list(run.scored_on),
        "seed": run.seed,
        "twin": run.twin.run_id,
        "twin_study": run.twin_study,
        "recipe": f"{run.twin_study}:{run.twin.arm}",
        "switches": dict(SWITCHES),
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "queue": QUEUE,
        "pbs_queue": env.pbs_queue(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32": env.tf32_flags(),
        "tier": TAG,
        "registered_by": FREEZE,
        "freeze": attestation,
        # exploratory throughout: no number from it may be reported as a tiered result
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


# Dry run.
def dry_run(payloads: dict) -> None:
    """
    Every fold's pool through its original's loader, with the registered window counts and its step magnitudes before
    and after signed_log1p; every run's configuration against its original's (the switches alone may differ), the
    same-seed initial weights, and the original's record, pool, checkpoint and sealed prediction files. Nothing is
    trained and no test partition is opened.
    """
    grid_index, source = T.selected_grid_index(None)
    print(f"configuration    : grid {grid_index} from {source}; switches {SWITCHES}")
    failures, names, windows = [], {}, {}
    caches = {"birds": {}, "birds_rotation": {}}  # fold 0's pool has no rotation exclusions, so it is scanned apart
    print(f"\n{'fold':>4} {'node':<20} {'species':>7} {'windows':>11}   max |step| before / after   "
          f"share above {' / '.join(f'{v:g}' for v in R.STEP_LEVELS)}, before | after   held out")
    for fold in FOLDS:
        run = Run("BbF6", fold, SEEDS[0])
        source_pool, node = training_source(run, payloads)
        check_window_counts(run, source_pool, payloads)
        names[fold] = list(source_pool.names)
        windows[fold] = sum(source_pool.species[n].n_windows for n in source_pool.names)
        steps = R.step_magnitudes(source_pool, caches[run.twin_study])
        print(f"{fold:>4} {node:<20} {len(source_pool.names):>7} {windows[fold]:>11,}   "
              f"{steps['before']['max']:>10.4g} ({steps['before']['max_species']}) / {steps['after']['max']:.4g}   "
              f"{R._share(steps['before']['share_above'])} | {R._share(steps['after']['share_above'])}   "
              f"{', '.join(run.scored_on)}")

    print(f"\n{'idx':>3}  {'run':<14} {'twin':<28} initial weights")
    for index, run in enumerate(PLAN):
        differences = config_differences(run, grid_index, REGISTERED_STEPS, names[run.fold], payloads)
        if differences != SWITCHES:
            failures.append(f"{run.run_id}: differs from its original in {differences}")
        initial = initial_state_differences(run, grid_index, names[run.fold], payloads)
        if initial != {"extra": EXPECTED_EXTRA, "missing": [], "different": []}:
            failures.append(f"{run.run_id}: initial weights differ from its original's: {initial}")
        failures += [f"{run.run_id}: {p}"
                     for p in twin_problems(run, grid_index, names[run.fold], payloads, windows[run.fold])]
        print(f"{index:>3}  {run.run_id:<14} {run.twin_study + ':' + run.twin.run_id:<28} "
              f"{'equal, plus encoder.norm' if not initial['different'] else 'DIFFERENT'}")
    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    print(f"\n{len(PLAN)} runs: pools and counts as registered; configs differ from their originals in the switches "
          "alone; initial weights equal but for encoder.norm; originals' records, pools, checkpoints and sealed "
          "predictions present. Nothing trained; no test partition opened.")


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'run':<14} {'twin':<28} scored on")
    for i, run in enumerate(plan):
        print(f"{i:>4}  {run.run_id:<14} {run.twin_study + ':' + run.twin.run_id:<28} {', '.join(run.scored_on)}")
    print("\nblocks: " + "; ".join(f"{first}-{last}" for first, last, _ in BLOCKS))
    print(f"calibration: indices {', '.join(map(str, CALIBRATION_INDICES))} (fold 9, seed 0)")
    print(f"{len(plan)} exploratory runs (0-{len(plan) - 1}), all on {QUEUE}.")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="check every pool, config, initial weights and twin, and exit")
    parser.add_argument("--calibrate", action="store_true",
                        help=f"{R.CALIBRATION_STEPS:,} updates of the run and its original to {CALIBRATION.name}/; no record")
    args = parser.parse_args()

    if args.list:
        return print_plan(PLAN)
    if args.dry_run:
        return dry_run(load_payloads())
    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    index = int(raw)
    if not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")

    run = PLAN[index]
    attestation = mode_guards(run, calibrate=args.calibrate, steps=args.steps)
    payloads = load_payloads()
    grid_index, source = T.selected_grid_index(None)
    for directory in ((CALIBRATION,) if args.calibrate else (RESULTS, CHECKPOINTS)):
        directory.mkdir(parents=True, exist_ok=True)
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1}"
          f"{', CALIBRATION' if args.calibrate else ''})", flush=True)
    print(f"configuration    : grid {grid_index} from {source}; TF32 {flags}; PBS_QUEUE {env.pbs_queue()!r}", flush=True)
    if args.calibrate:
        return calibrate(run, grid_index, payloads)
    run_torch(run, index, grid_index, source, payloads, attestation)


if __name__ == "__main__":
    main()
