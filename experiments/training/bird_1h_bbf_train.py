# GenAI reference: repaired transformer study; BbF = Bb with the repaired encoder (the repair probe's V4 switches) at C = 12 and 96, BbFE = BbF + 
# the env study's 15 ERA5 channels at C = 12, and LSTMBbR96 (LSTM-Bb at C = 96 rerun on gpuhopper), 10 folds x seeds 0-2, everything else the 1 h 
# main study's.
# Extension (26-10-2026, freeze sec 10): I specified BbF at C = 24, 48 and 72 appended as indices 120-209 with indices 0-119 unchanged; GenAI wrote 
# the append, the repeats guard and the checks on it which I checked. 
"""
Train the repaired-transformer study's runs on the 1 h rotation folds, one model per PBS subjob (exploratory).

BbF is Bb with the repaired encoder (the repair probe's V4 switches: input_scale "none", norm_first, step_transform
"signed_log1p"); BbFE adds the env study's 15 ERA5 channels at C = 12; LSTMBbR is LSTM-Bb at C = 96 rerun unchanged on
gpuhopper. The extension (freeze sec 10) appends BbF at C = 24, 48 and 72. Plan: --list. Config and reference checks:
--dry-run. Throughput: --calibrate.
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
import transformer_repair_train as TR  # noqa: E402

from animal_movement_fm.data import era5  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import horizon_k  # noqa: E402

b1h = env.b1h

RESULTS = REPO / "results" / "birds_1h_bbf"
CHECKPOINTS = REPO / "checkpoints" / "birds_1h_bbf"
CALIBRATION = REPO / "results" / "birds_1h_bbf_calibration"
FREEZE = "birds_1h_bbf_exploratory_freeze.md"
TAG = "birds_1h_bbf"

CORE, HORIZONS, FOLDS, SEEDS = b1h.CORE, b1h.HORIZONS, b1h.FOLDS, b1h.SEEDS
REGISTERED_STEPS, CALIBRATION_STEPS = b1h.REGISTERED_STEPS, b1h.CALIBRATION_STEPS
NEURAL_QUEUE = env.NEURAL_QUEUE  # gpuhopper
C96 = 96
EXTENSION_CONTEXTS = (24, 48, 72)  # freeze sec 10: BbF at the 1 h ablation's other contexts
ENV_CHANNELS = env.ENV_CHANNELS  # 15

# The repaired encoder: repair probe's V4, pinned here so a change to the probe's table cannot move this study. 
SWITCHES = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
if TR.VARIANTS["V4"] != SWITCHES:
    raise RuntimeError(f"the repair probe's V4 is {TR.VARIANTS['V4']}; this study froze {SWITCHES}")

# Arm -> the 1 h main-study arm it is built on, whether it carries the repaired encoder, and whether it reads ERA5. 
ARMS: dict[str, dict] = {
    "BbF": {"base": "Bb", "repaired": True, "env": False},
    "BbFE": {"base": "Bb", "repaired": True, "env": True},
    "LSTMBbR": {"base": "LSTMBb", "repaired": False, "env": False},  # LSTMBbR96: LSTM-Bb at C = 96 on gpuhopper
}


@dataclass(frozen=True)
class BbfRun:
    arm: str
    fold: int
    seed: int
    context: int = CORE

    @property
    def run_id(self) -> str:
        return f"{self.arm}_f{self.fold}_s{self.seed}" + (f"_C{self.context}" if self.context != CORE else "")

    @property
    def base(self) -> b1h.Run:
        """The 1 h main-study or ablation run with this run's base arm, fold, seed and C: its pool, recipe and k_h."""
        return b1h.Run(ARMS[self.arm]["base"], self.fold, "", self.seed, 0, self.context)

    @property
    def env(self) -> bool:
        return ARMS[self.arm]["env"]

    @property
    def switches(self) -> dict:
        return dict(SWITCHES) if ARMS[self.arm]["repaired"] else {}

    @property
    def env_reference(self) -> str | None:
        """BbFE's env-study twin, BbE of the same fold and seed: the same pool, env files and env_stats."""
        return env.EnvRun("BbE", self.fold, self.seed).run_id if self.env else None

    @property
    def compared_with(self) -> str:
        """BbF at C = 12 against the env study's BbR (same hardware); BbFE against BbF; at C = 96 the 1 h ablation's run."""
        if self.arm == "BbFE":
            return replace(self, arm="BbF").run_id
        if self.arm == "BbF" and self.context == CORE:
            return env.EnvRun("BbR", self.fold, self.seed).run_id
        return self.base.run_id

    @property
    def repeats(self) -> str | None:
        """The repair probe run with exactly this config and seed (fold 6, seed 0), for the determinism check."""
        if self.env or (self.fold, self.seed) != (TR.PROBE_FOLDS[0], TR.SEED) or self.context not in TR.CONTEXTS:
            return None
        if self.arm == "BbF":
            return TR.RepairRun("Bb", "V4", self.fold, self.seed, self.context).run_id
        if self.arm == "LSTMBbR":
            return TR.RepairRun("LSTMBb", "V0", self.fold, self.seed, self.context).run_id
        return None


def build_plan() -> list[BbfRun]:
    """
    The plan. Position is the PBS array index: append, never insert or reorder. LSTMBbR96 closes the original 120;
    the extension (BbF at C = 24, 48, 72, each C x fold x seed) follows.
    """
    plan = []
    for arm, context in (("BbF", CORE), ("BbF", C96), ("BbFE", CORE), ("LSTMBbR", C96)):
        plan += [BbfRun(arm, k, s, context) for k in sorted(FOLDS) for s in SEEDS]
    for context in EXTENSION_CONTEXTS:
        plan += [BbfRun("BbF", k, s, context) for k in sorted(FOLDS) for s in SEEDS]
    return plan


PLAN = build_plan()
BLOCKS: tuple[tuple[int, int, str, str], ...] = (
    (0, 29, NEURAL_QUEUE, "BbF, C = 12, 10 folds x 3 seeds"),
    (30, 59, NEURAL_QUEUE, "BbF, C = 96, 10 folds x 3 seeds"),
    (60, 89, NEURAL_QUEUE, "BbFE, C = 12, 10 folds x 3 seeds"),
    (90, 119, NEURAL_QUEUE, "LSTMBbR96 (LSTM-Bb, C = 96, gpuhopper rerun), 10 folds x 3 seeds"),
    (120, 149, NEURAL_QUEUE, "extension: BbF, C = 24, 10 folds x 3 seeds"),
    (150, 179, NEURAL_QUEUE, "extension: BbF, C = 48, 10 folds x 3 seeds"),
    (180, 209, NEURAL_QUEUE, "extension: BbF, C = 72, 10 folds x 3 seeds"),
)
ORIGINAL, EXTENSION = PLAN[:120], PLAN[120:]  # freeze secs 1-9; freeze sec 10
LSTMBBR96 = [run for run in ORIGINAL if run.arm == "LSTMBbR"]
if (
    len(PLAN) != 210
    or [b[0] for b in BLOCKS] != [0] + [b[1] + 1 for b in BLOCKS[:-1]]
    or BLOCKS[-1][1] != len(PLAN) - 1
    or ORIGINAL[90:] != LSTMBBR96
    or [(r.arm, r.context) for r in EXTENSION] != [("BbF", c) for c in EXTENSION_CONTEXTS for _ in range(30)]
):
    raise RuntimeError(f"the plan has {len(PLAN)} runs; 120 are planned in four contiguous blocks, LSTMBbR96 last, "
                       f"then the 90-run extension")
_OTHER_IDS = {r.run_id for r in b1h.PLAN} | {r.run_id for r in env.PLAN} | {r.run_id for r in TR.PLAN}
if len({run.run_id for run in PLAN}) != len(PLAN) or {run.run_id for run in PLAN} & _OTHER_IDS:
    raise RuntimeError("run ids must be distinct, and distinct from every 1 h, env and repair probe run id")
if any(run.base not in b1h.PLAN for run in PLAN):
    raise RuntimeError("every run must sit on a 1 h main-study or ablation run")
if any(run.env_reference not in {r.run_id for r in env.PLAN} for run in PLAN if run.env):
    raise RuntimeError("every BbFE run must have its BbE twin in the env plan")
if any(run.repeats not in {r.run_id for r in TR.PLAN} for run in PLAN if run.repeats):
    raise RuntimeError("every repeat must name a repair probe run")


def queue_of(index: int) -> str:
    return next(queue for first, last, queue, _ in BLOCKS if first <= index <= last)


# Guards.
def queue_guard(run: BbfRun) -> None:
    cuda = torch.cuda.is_available()
    if not (cuda and env.on_queue(NEURAL_QUEUE)):
        raise SystemExit(
            f"FAIL: {run.run_id} trains on {NEURAL_QUEUE} (bird_1h_bbf_train.pbs); "
            f"PBS_QUEUE is {env.pbs_queue()!r}, CUDA {cuda}"
        )


def freeze_guard() -> dict:
    """This study's freeze exists, is tracked and is unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    S2 = b1h.S2
    previous = S2.FREEZE
    S2.FREEZE = REPO / "preregistration" / FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def mode_guards(run: BbfRun, *, calibrate: bool, steps: int | None) -> dict | None:
    """Queue always; the committed freeze and the registered budget except in --calibrate. Returns the attestation."""
    queue_guard(run)
    if calibrate:
        return None
    attestation = freeze_guard()
    if steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {steps}")
    return attestation


def env_reference_problems(run: BbfRun, stats_record: dict | None, files: dict | None) -> list[str]:
    """BbFE's env_stats and env file sha256 against its BbE twin's record (results/birds_1h_env/); empty if equal."""
    path = env.RESULTS / f"{run.env_reference}.json"
    if not path.exists():
        return [f"{run.env_reference}'s record {path} is missing"]
    twin = json.loads(path.read_text())
    problems = []
    if twin.get("env_clip_sd") != era5.CLIP_SD or twin.get("env_channels") != ENV_CHANNELS:
        problems.append(f"{run.env_reference} has clip {twin.get('env_clip_sd')}, {twin.get('env_channels')} channels")
    if stats_record is not None and twin.get("env_stats") != stats_record:
        problems.append(f"env_stats differ from {run.env_reference}'s")
    if files is not None and twin.get("env_files") != files:
        problems.append(f"env file sha256 differ from {run.env_reference}'s")
    return problems


# Pools and configs.
def training_source(run: BbfRun, payload: dict):
    """
    (pool, node, env_stats). BbFE: the env study's pool for its fold, env_stats from that training pool alone. Otherwise
    the 1 h base run's pool at the run's C. Training individuals after exclusions; never val or test.
    """
    if run.env:
        return env.training_source(env.EnvRun("BbE", run.fold, run.seed), payload)
    pool, node = b1h.training_source(run.base, payload)
    return pool, node, None


def build_config(run: BbfRun, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    """The base run's config (BbFE: the env study's BbE config) with the repaired encoder's switches; nothing else."""
    if run.env:
        config = env.build_config(env.EnvRun("BbE", run.fold, run.seed), grid_index, steps, pool_names, payload)
    else:
        config = b1h.build_config(run.base, grid_index, steps, pool_names, payload)
    return replace(config, **run.switches)


def config_differences(run: BbfRun, grid_index: int, steps: int, pool_names, payload: dict) -> dict:
    """The fields in which the run's config differs from its 1 h base run's."""
    base = asdict(b1h.build_config(run.base, grid_index, steps, pool_names, payload))
    got = asdict(build_config(run, grid_index, steps, pool_names, payload))
    return {name: got[name] for name in got if got[name] != base[name]}


def expected_differences(run: BbfRun) -> dict:
    return run.switches | ({"env_channels": ENV_CHANNELS} if run.env else {})


# Runners.
def run_torch(run: BbfRun, index: int, steps: int, grid_index: int, source: str, payload: dict,
              attestation: dict | None, calibrate: bool = False) -> None:
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node, stats = training_source(run, payload)
    b1h.check_window_counts(source_pool)
    files = env.env_files(source_pool.names) if run.env else None
    if run.env and not calibrate:
        problems = env_reference_problems(run, env.stats_payload(stats), files)
        if problems:
            raise SystemExit(f"FAIL: {run.run_id} is not on its env-study twin's pool: " + "; ".join(problems))
    k = horizon_k(source_pool)
    shares = b1h.clip_shares(source_pool)
    coverage = {n: env.env_coverage(source_pool.species[n]) for n in source_pool.names} if run.env else None
    pool = b1h.prepared_pool(source_pool, k, run.context, device)
    built = time.time() - started
    print(
        f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], C = {run.context}, "
        f"{type(pool).__name__}, {pool.bytes_on_device() / 1e9:.2f} GB on device, built in {built:.0f}s",
        flush=True,
    )
    print(f"k_h              : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))}", flush=True)
    config = build_config(run, grid_index, CALIBRATION_STEPS if calibrate else steps, pool.names, payload)
    print(f"switches         : {run.switches or 'none'}; env_channels {config.env_channels}", flush=True)
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
    selection = {"rule": "final_checkpoint", "step": config.steps}  # pooled arms report their final checkpoint
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
            "env_channels": config.env_channels,
            "env_stats": env.stats_payload(stats),
            "env_clip_sd": era5.CLIP_SD if run.env else None,
            "selection": selection,
            "model": result.model.state_dict(),
        },
        reported,
    )
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares, stats, files,
                         coverage, attestation) | {
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


def calibrate_torch(run: BbfRun, config: TrainConfig, pool, device, k, built: float) -> None:
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
        "env_channels": config.env_channels,
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
    print(json.dumps({key: v for key, v in payload.items() if key != "probe"}, indent=2, default=str))
    print(f"wrote {path}; calibration only, no run record", flush=True)


def base_record(run: BbfRun, index: int, grid_index: int, source: str, node: str, pool_names, payload: dict,
                k: tuple, shares: dict, stats, files: dict | None, coverage: dict | None,
                attestation: dict | None) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "base_arm": run.base.arm,
        "main_study_run": run.base.run_id,
        "compared_with": run.compared_with,
        "repeats": run.repeats,
        "switches": run.switches,
        "recipe": b1h.BT.counterpart(run.base).arm,
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
        "env": run.env,
        "env_channels": ENV_CHANNELS if run.env else 0,
        "env_channel_names": list(era5.ENV_CHANNELS) if run.env else [],
        "env_stats": env.stats_payload(stats),
        "env_stats_source": "the run's training pool, window weight p_s / n_s" if run.env else None,
        "env_clip_sd": era5.CLIP_SD if run.env else None,
        "env_coverage": coverage,
        "env_files": files,
        "env_reference": run.env_reference,
        "queue": queue_of(index),
        "pbs_queue": env.pbs_queue(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32": env.tf32_flags(),
        "tier": TAG,
        "registered_by": FREEZE,
        "freeze": attestation,
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: BbfRun, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def dry_run(payload: dict) -> None:
    """
    Every run's config against its 1 h base run's (only the switches, and BbFE's env channels, may differ), BbFE against
    its BbE twin's config (only the switches), each model builds, and the env twins and probe repeats are on disk.
    """
    grid_index, source = b1h.T.selected_grid_index(None)
    print(f"configuration    : grid {grid_index} from {source}")
    failures = []
    for index, run in enumerate(PLAN):
        names = sorted(b1h.fold_probabilities(run.fold, payload))
        differences = config_differences(run, grid_index, REGISTERED_STEPS, names, payload)
        if differences != expected_differences(run):
            raise SystemExit(f"FAIL: {run.run_id} differs from {run.base.run_id} in {differences}")
        config = build_config(run, grid_index, REGISTERED_STEPS, names, payload)
        if run.env:
            twin = asdict(env.build_config(env.EnvRun("BbE", run.fold, run.seed), grid_index, REGISTERED_STEPS,
                                           names, payload))
            if {n: v for n, v in asdict(config).items() if v != twin[n]} != run.switches:
                raise SystemExit(f"FAIL: {run.run_id} differs from {run.env_reference} in more than the switches")
            failures += [f"{run.run_id}: {p}" for p in env_reference_problems(run, None, None)]
        if run.repeats and not (TR.RESULTS / f"{run.repeats}.json").exists():
            failures.append(f"{run.run_id}: the probe record {run.repeats}.json is missing")
        model = config.build()
        print(f"{index:>3}  {run.run_id:<22} {str(expected_differences(run) or 'none'):<92} "
              f"{model.n_parameters():>9,} parameters")
    if failures:
        raise SystemExit("FAIL: references missing or different\n  " + "\n  ".join(failures))
    print(f"\n{len(PLAN)} runs: configs as planned; env twins and probe repeats present. Nothing trained.")


def print_plan(plan: list[BbfRun], *, runs: bool = False) -> None:
    print(f"{'indices':<10} {'n':>4}  {'queue':<10} what")
    for first, last, queue, what in BLOCKS:
        print(f"{first:>3}-{last:<6} {last - first + 1:>4}  {queue:<10} {what}")
    print(f"\n{len(plan)} runs (0-{len(plan) - 1}).")
    if runs:
        print(f"\n{'idx':>4}  {'run':<22} {'compared with':<22} repeats")
        for i, run in enumerate(plan):
            print(f"{i:>4}  {run.run_id:<22} {run.compared_with:<22} {run.repeats or '-'}")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan's blocks and exit")
    parser.add_argument("--list-runs", action="store_true", help="print every index and exit")
    parser.add_argument("--dry-run", action="store_true", help="check every run's config and references, and exit")
    parser.add_argument("--calibrate", action="store_true",
                        help=f"{CALIBRATION_STEPS:,} updates to {CALIBRATION.name}/; no record")
    args = parser.parse_args()

    if args.list or args.list_runs:
        return print_plan(PLAN, runs=args.list_runs)
    env.check_main_study()
    if args.dry_run:
        return dry_run(b1h.load_registered())
    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    index = int(raw)
    if not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")

    run = PLAN[index]
    attestation = mode_guards(run, calibrate=args.calibrate, steps=args.steps)
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
    run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload, attestation, calibrate=args.calibrate)


if __name__ == "__main__":
    main()
