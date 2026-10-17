# GenAI reference: I outlined the transformer repair probe, but a lot of the details were the idea of GenAI which I have agreed to; these being
# the fold 6, seed 0, 20,000 updates, the 1 h main studys grid, recipe and k_h, final checkpoint, gpuhopper FP32 etc. GenAI wrote this driver. 
"""
Train the transformer repair probe's runs on the 1 h fold 6 pool, one model per PBS subjob (exploratory).

Each run is a 1 h main-study Bb or LSTM-Bb run (fold, seed, C, grid, recipe, k_h) with one or more of the repair
switches set: V1 input_scale "none", V2 norm_first, V3 step_transform "signed_log1p", V4 all three; V0 sets none and
is the gpuhopper reference. Validation only: nothing here or in its evaluation reads a test partition.
Plan: --list. Config checks: --dry-run. Throughput: --calibrate.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))
sys.path.insert(0, str(REPO))

import bird_1h_env_train as env  # noqa: E402
import bird_1h_train as b1h  # noqa: E402

from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import horizon_k  # noqa: E402

RESULTS = REPO / "results" / "transformer_repair"
CHECKPOINTS = REPO / "checkpoints" / "transformer_repair"
CALIBRATION = REPO / "results" / "transformer_repair_calibration"
TAG = "transformer_repair"  # validation-only model development (notes.md 17-10); no freeze, no test partition

CORE, HORIZONS = b1h.CORE, b1h.HORIZONS
REGISTERED_STEPS, CALIBRATION_STEPS = b1h.REGISTERED_STEPS, b1h.CALIBRATION_STEPS
NEURAL_QUEUE = env.NEURAL_QUEUE  # gpuhopper
PROBE_FOLDS = (6,)  # fold 3 is appended only if approved after fold 6
SEED = 0
CONTEXTS = (CORE, 96)

# variant -> the TrainConfig fields it sets; everything else is the 1 h main-study run's
VARIANTS: dict[str, dict] = {
    "V0": {},
    "V1": {"input_scale": "none"},
    "V2": {"norm_first": True},
    "V3": {"step_transform": "signed_log1p"},
    "V4": {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"},
}
VARIANTS_OF = {"Bb": ("V0", "V1", "V2", "V3", "V4"), "LSTMBb": ("V0", "V3")}


@dataclass(frozen=True)
class RepairRun:
    base: str  # the 1 h main-study arm: Bb or LSTMBb
    variant: str
    fold: int
    seed: int
    context: int

    @property
    def run_id(self) -> str:
        return f"{self.base}{self.variant}_f{self.fold}_s{self.seed}" + (f"_C{self.context}" if self.context != CORE else "")

    @property
    def main_run(self) -> b1h.Run:
        """The 1 h study's run with this run's arm, fold, seed and C: its pool, recipe, k_h and config base."""
        return b1h.Run(self.base, self.fold, "", self.seed, 0, self.context)

    @property
    def switches(self) -> dict:
        return dict(VARIANTS[self.variant])

    @property
    def compared_with(self) -> str:
        """V1-V4 against their V0; V0 against the main study's run (hardware check)."""
        return replace(self, variant="V0").run_id if self.variant != "V0" else self.main_run.run_id

    @property
    def env_rerun(self) -> str | None:
        """At C = 12 a V0 rerun repeats the env study's no-env rerun of the same run (a determinism check)."""
        if self.variant == "V0" and self.context == CORE:
            return f"{self.base}R_f{self.fold}_s{self.seed}"
        return None


def build_plan(folds=PROBE_FOLDS) -> list[RepairRun]:
    """The repair plan. Position is the PBS array index: append, never insert or reorder."""
    plan = []
    for fold in folds:
        for context in CONTEXTS:
            plan += [RepairRun(base, v, fold, SEED, context) for base in ("Bb", "LSTMBb") for v in VARIANTS_OF[base]]
    return plan


PLAN = build_plan()
BLOCKS: tuple[tuple[int, int, str, str], ...] = (
    (0, 9, NEURAL_QUEUE, "fold 6: C = 12 Bb V0-V4, LSTM-Bb V0, V3; C = 96 Bb V0-V2"),
    (10, 13, NEURAL_QUEUE, "fold 6: C = 96 Bb V3, V4, LSTM-Bb V0, V3"),
)
if len(PLAN) != 14 or [b[0] for b in BLOCKS] != [0, 10] or BLOCKS[-1][1] != len(PLAN) - 1:
    raise RuntimeError(f"the repair plan has {len(PLAN)} runs in blocks {BLOCKS}; it is 14 runs in 0-9, 10-13")
if any(run.main_run not in b1h.PLAN for run in PLAN):
    raise RuntimeError("every repair run must sit on a 1 h main-study or ablation run")


def queue_of(index: int) -> str:
    return next(queue for first, last, queue, _ in BLOCKS if first <= index <= last)


# Guards.
def queue_guard(run: RepairRun) -> None:
    cuda = torch.cuda.is_available()
    if not (cuda and env.on_queue(NEURAL_QUEUE)):
        raise SystemExit(
            f"FAIL: {run.run_id} trains on {NEURAL_QUEUE} (transformer_repair_train.pbs); "
            f"PBS_QUEUE is {env.pbs_queue()!r}, CUDA {cuda}"
        )


def mode_guards(run: RepairRun, *, calibrate: bool, steps: int | None) -> None:
    """Queue always; the 20,000-update budget except in --calibrate. TF32 is checked in run_torch (env.require_fp32)."""
    queue_guard(run)
    if not calibrate and steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {steps}")


# Pools and configs.
def training_source(run: RepairRun, payload: dict):
    """The main-study run's training pool at the run's C (training individuals after exclusions; never val or test)."""
    return b1h.training_source(run.main_run, payload, run.context)


def build_config(run: RepairRun, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    """The main-study run's config with the variant's switches; nothing else differs."""
    return replace(b1h.build_config(run.main_run, grid_index, steps, pool_names, payload), **run.switches)


def config_differences(run: RepairRun, grid_index: int, steps: int, pool_names, payload: dict) -> dict:
    base = asdict(b1h.build_config(run.main_run, grid_index, steps, pool_names, payload))
    got = asdict(build_config(run, grid_index, steps, pool_names, payload))
    return {name: got[name] for name in got if got[name] != base[name]}


# Runners.
def run_torch(run: RepairRun, index: int, steps: int, grid_index: int, source: str, payload: dict,
              calibrate: bool = False) -> None:
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payload)
    b1h.check_window_counts(source_pool)
    k = horizon_k(source_pool)
    shares = b1h.clip_shares(source_pool)
    pool = b1h.prepared_pool(source_pool, k, run.context, device)
    built = time.time() - started
    print(
        f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], C = {run.context}, "
        f"{type(pool).__name__}, {pool.bytes_on_device() / 1e9:.2f} GB on device, built in {built:.0f}s",
        flush=True,
    )
    print(f"k_h              : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))}", flush=True)
    config = build_config(run, grid_index, CALIBRATION_STEPS if calibrate else steps, pool.names, payload)
    print(f"switches         : {run.switches or 'none (V0)'}", flush=True)
    if calibrate:
        return calibrate_torch(run, config, pool, device, k, built)

    result = train(
        config,
        pool,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
    )
    print(
        f"applied updates  : {result.applied_updates:,} of {config.steps:,}; "
        f"withheld {result.skipped_steps:,} ({result.withheld_fraction:.2%})",
        flush=True,
    )
    selection = {"rule": "final_checkpoint", "step": config.steps}
    reported = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {
            "run_id": run.run_id,
            "config": asdict(config),
            "grid_index": grid_index,
            "pool_species": list(pool.names),
            "horizon_k": list(k),
            "context": run.context,
            "switches": run.switches,
            "selection": selection,
            "model": result.model.state_dict(),
        },
        reported,
    )
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares) | {
        "config": asdict(config),
        "n_parameters": result.model.n_parameters(),
        "pool_windows": pool.n_windows,
        "pool_kind": type(pool).__name__,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "head_loss_history": result.head_loss_history,
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


def calibrate_torch(run: RepairRun, config: TrainConfig, pool, device, k, built: float) -> None:
    """1,000 updates, no checkpoint, no record, no validation or test; throughput and memory."""
    padded = run.context > CORE
    probe = pool.sample_horizon(torch.Generator(device=device).manual_seed(b1h.PROBE_SEED), b1h.LOSS_PROBE)
    torch.manual_seed(config.seed)
    start_losses = b1h.probe_losses(config.build().to(device), probe, padded)
    timed, ticks, every = b1h.TimedPool(pool, device), [], 50
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with b1h.GpuPoller() as poller:
        began = time.time()
        result = train(config, timed, device=device, checkpoint_path=None, checkpoint_every=0, log_every=every,
                       progress=lambda step, loss: ticks.append((step, time.perf_counter())))
        seconds = time.time() - began
    step_times = np.diff([t for _, t in ticks]) / every
    per_step = float(np.median(step_times)) if len(step_times) else seconds / config.steps
    payload = {
        "run_id": run.run_id,
        "switches": run.switches,
        "context": run.context,
        "pool_windows": pool.n_windows,
        "pool_kind": type(pool).__name__,
        "build_seconds": built,
        "updates": result.applied_updates,
        "train_seconds": seconds,
        "seconds_per_update_median": per_step,
        "sample_seconds_per_batch_median": float(np.median(timed.sample_seconds)),
        "extrapolated_total_hours_20000": (built + per_step * REGISTERED_STEPS) / 3600,
        "gpu": poller.summary(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "peak_rss_gb": env.peak_rss_gb(),
        "tf32": env.tf32_flags(),
        "pbs_queue": env.pbs_queue(),
        "horizon_k": list(k),
        "probe": {"windows": b1h.LOSS_PROBE, "seed": b1h.PROBE_SEED, "start": start_losses,
                  "end": b1h.probe_losses(result.model, probe, padded)},
    }
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION / f"{run.run_id}_calibration.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "probe"}, indent=2, default=str))
    print(f"wrote {path}; calibration only, no run record", flush=True)


def base_record(run: RepairRun, index: int, grid_index: int, source: str, node: str, pool_names, payload: dict,
                k: tuple, shares: dict) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": f"{run.base}{run.variant}",
        "base_arm": run.base,
        "variant": run.variant,
        "switches": run.switches,
        "main_study_run": run.main_run.run_id,
        "compared_with": run.compared_with,
        "env_rerun": run.env_rerun,
        "recipe": b1h.BT.counterpart(run.main_run).arm,
        "fold": run.fold,
        "seed": run.seed,
        "horizons": list(HORIZONS),
        "context": run.context,
        "clip": b1h.CLIP,
        "clip_shares": shares,
        "horizon_k": list(k),
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "pool_species": list(pool_names),
        "excluded_individuals": b1h.RT.excluded_in(pool_names, payload),
        "queue": queue_of(index),
        "pbs_queue": env.pbs_queue(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32": env.tf32_flags(),
        "tier": TAG,
        "preregistered": False,  # validation-only model development (notes.md 17-10); no freeze
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: RepairRun, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def dry_run(payload: dict) -> None:
    """Every run's config against its main-study run's (only the variant's fields may differ), and each model builds."""
    grid_index, source = b1h.T.selected_grid_index(None)
    print(f"configuration    : grid {grid_index} from {source}")
    for index, run in enumerate(PLAN):
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        differences = config_differences(run, grid_index, REGISTERED_STEPS, names, payload)
        if differences != run.switches:
            raise SystemExit(f"FAIL: {run.run_id} differs from {run.main_run.run_id} in {differences}")
        model = build_config(run, grid_index, REGISTERED_STEPS, names, payload).build()
        print(f"{index:>3}  {run.run_id:<22} {str(run.switches or 'V0'):<72} {model.n_parameters():>9,} parameters")


def print_plan(plan: list[RepairRun], *, runs: bool = False) -> None:
    print(f"{'indices':<10} {'n':>4}  {'queue':<10} what")
    for first, last, queue, what in BLOCKS:
        print(f"{first:>3}-{last:<6} {last - first + 1:>4}  {queue:<10} {what}")
    print(f"\n{len(plan)} runs (0-{len(plan) - 1}).")
    if runs:
        print(f"\n{'idx':>4}  {'run':<22} {'compared with':<22} switches")
        for i, run in enumerate(plan):
            print(f"{i:>4}  {run.run_id:<22} {run.compared_with:<22} {run.switches or '-'}")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan's blocks and exit")
    parser.add_argument("--list-runs", action="store_true", help="print every index and exit")
    parser.add_argument("--dry-run", action="store_true", help="check every run's config and model, and exit")
    parser.add_argument("--calibrate", action="store_true",
                        help=f"{CALIBRATION_STEPS:,} updates to {CALIBRATION.name}/; no record")
    args = parser.parse_args()

    if args.list or args.list_runs:
        return print_plan(PLAN, runs=args.list_runs)
    if args.dry_run:
        return dry_run(b1h.load_registered())
    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    index = int(raw)
    if not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")

    run = PLAN[index]
    mode_guards(run, calibrate=args.calibrate, steps=args.steps)
    payload = b1h.load_registered()
    grid_index, source = b1h.T.selected_grid_index(None)
    for directory in ((CALIBRATION,) if args.calibrate else (RESULTS, CHECKPOINTS)):
        directory.mkdir(parents=True, exist_ok=True)
    print(
        f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1}, compared with "
        f"{run.compared_with}{', CALIBRATION' if args.calibrate else ''})",
        flush=True,
    )
    print(f"configuration    : grid {grid_index} from {source}; TF32 {flags}; PBS_QUEUE {env.pbs_queue()!r}", flush=True)
    run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload, calibrate=args.calibrate)


if __name__ == "__main__":
    main()
