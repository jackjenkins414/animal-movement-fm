# GenAI reference: I asked GenAI to write this driver from things I outlined which where, ERA5 covariates for Bb, LSTM-Bb and XGB-Bb on our 1 h rotation 
# folds at C = 12, no env Bb and LSTM-Bb reruns on the same GPUs, full FP32 (TF32 off, recorded), env statistics from each folds training pool only etc. 
# I checked this code. 
"""
Train the env study's arms on the 1 h rotation folds, one model per PBS subjob (exploratory).

"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import re
import resource
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

import bird_1h_train as b1h  # noqa: E402

from animal_movement_fm.data import era5  # noqa: E402
from animal_movement_fm.data.enu import N_FEATURES  # noqa: E402
from animal_movement_fm.data.multispecies import HorizonWindows, SourcePool, history_positions  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool, env_stats, horizon_k  # noqa: E402

ENV_DIR = REPO / "data" / "processed_1h_env"
BUILD = REPO / "results" / "birds_1h_env_build"
RESULTS = REPO / "results" / "birds_1h_env"
CHECKPOINTS = REPO / "checkpoints" / "birds_1h_env"
CALIBRATION = REPO / "results" / "birds_1h_env_calibration"
FREEZE = "birds_1h_env_exploratory_freeze.md"
TAG = "birds_1h_env"

ENV_CHANNELS = era5.N_ENV_CHANNELS  # 15: 8 hourly, 2 static, 4 derived winds, the observed flag
TAIL_SD = 10.0  # the dry run reports the share of standardised values beyond this
CORE, HORIZONS, FOLDS, BIRDS = b1h.CORE, b1h.HORIZONS, b1h.FOLDS, b1h.BIRDS
SEEDS, XGB_SEED = b1h.SEEDS, b1h.XGB_SEED
REGISTERED_STEPS, CALIBRATION_STEPS = b1h.REGISTERED_STEPS, b1h.CALIBRATION_STEPS
NEURAL_QUEUE, CPU_QUEUE = "gpuhopper", "normal"

# the 1 h study this one builds on: its plan's run ids and the evaluator's parts, as committed with the 1 h freeze
MAIN_PLAN_RUNS = 630
MAIN_PLAN_SHA256 = "050c688b1289bc4d0a98f03d4a9068bc3a0c0e95ee4b7188e1c0246b8ae85d28"
MAIN_PARTS = {"main": (0, 269), "ablation": (270, 629)}

# arm -> the main-study arm whose recipe, pool and XGB setup it takes, and whether it reads ERA5
ARMS: dict[str, dict] = {
    "BbE": {"base": "Bb", "env": True, "xgb": False},
    "LSTMBbE": {"base": "LSTMBb", "env": True, "xgb": False},
    "BbR": {"base": "Bb", "env": False, "xgb": False},  # no-env rerun on gpuhopper
    "LSTMBbR": {"base": "LSTMBb", "env": False, "xgb": False},  # no-env rerun on gpuhopper
    "XGBBbE": {"base": "XGBBb", "env": True, "xgb": True},
}


@dataclass(frozen=True)
class EnvRun:
    arm: str
    fold: int
    seed: int
    horizon: int = 0  # XGB only

    @property
    def run_id(self) -> str:
        return f"{self.arm}_f{self.fold}_" + (f"h{self.horizon}" if self.horizon else f"s{self.seed}")

    @property
    def base(self) -> b1h.Run:
        """The main-study run at C = 12 with this arm's recipe, fold, seed and horizon."""
        return b1h.Run(ARMS[self.arm]["base"], self.fold, "", self.seed, self.horizon)

    @property
    def env(self) -> bool:
        return ARMS[self.arm]["env"]

    @property
    def xgb(self) -> bool:
        return ARMS[self.arm]["xgb"]

    @property
    def compared_with(self) -> str:
        """BbE and LSTMBbE against their gpuhopper reruns; XGBBbE against the main study's XGB-Bb; a rerun against the
        main study's gpuvolta run (the hardware check)."""
        if self.arm in ("BbE", "LSTMBbE"):
            return replace(self, arm=self.arm[:-1] + "R").run_id
        return self.base.run_id


def build_plan() -> list[EnvRun]:
    """The env plan. Position is the PBS array index: append, never insert or reorder."""
    plan = []
    for arm in ("BbE", "LSTMBbE", "BbR", "LSTMBbR"):
        plan += [EnvRun(arm, k, s) for k in sorted(FOLDS) for s in SEEDS]
    plan += [EnvRun("XGBBbE", k, XGB_SEED, h) for k in sorted(FOLDS) for h in HORIZONS]
    return plan


PLAN = build_plan()
BLOCKS: tuple[tuple[int, int, str, str], ...] = (
    (0, 29, NEURAL_QUEUE, "BbE, 10 folds x 3 seeds"),
    (30, 59, NEURAL_QUEUE, "LSTMBbE, 10 folds x 3 seeds"),
    (60, 89, NEURAL_QUEUE, "BbR (Bb, no env), 10 folds x 3 seeds"),
    (90, 119, NEURAL_QUEUE, "LSTMBbR (LSTM-Bb, no env), 10 folds x 3 seeds"),
    (120, 149, CPU_QUEUE, "XGBBbE, 10 folds x 3 horizons"),
)
if (
    len(PLAN) != 150
    or [b[0] for b in BLOCKS] != [0] + [b[1] + 1 for b in BLOCKS[:-1]]
    or BLOCKS[-1][1] != len(PLAN) - 1
):
    raise RuntimeError(f"the env plan has {len(PLAN)} runs; 150 are planned in contiguous queue blocks")


def queue_of(index: int) -> str:
    return next(queue for first, last, queue, _ in BLOCKS if first <= index <= last)


if any(queue_of(i) != (CPU_QUEUE if run.xgb else NEURAL_QUEUE) for i, run in enumerate(PLAN)):
    raise RuntimeError("a block's queue does not match its runs: neural on gpuhopper, XGBoost on normal")
if len({run.run_id for run in PLAN}) != len(PLAN) or {r.run_id for r in PLAN} & {r.run_id for r in b1h.PLAN}:
    raise RuntimeError("env run ids must be distinct, and distinct from every 1 h run id")
if any(run.base not in b1h.PLAN[: MAIN_PARTS["main"][1] + 1] for run in PLAN):
    raise RuntimeError("every env run must take the recipe of a main-study 1 h run at C = 12")


# Guards.
def check_main_study(plan=None, parts=None) -> None:
    """The 1 h plan and the 1 h evaluator's parts load exactly as committed."""
    plan = b1h.PLAN if plan is None else plan
    if parts is None:
        import bird_1h_sealed_eval as E

        parts = E.PARTS
    digest = hashlib.sha256("\n".join(run.run_id for run in plan).encode()).hexdigest()
    if (len(plan), digest) != (MAIN_PLAN_RUNS, MAIN_PLAN_SHA256) or dict(parts) != MAIN_PARTS:
        raise SystemExit(
            f"FAIL: the 1 h plan ({len(plan)} runs, sha256 {digest[:12]}) or the 1 h evaluator's parts {dict(parts)} "
            f"differ from what this study builds on ({MAIN_PLAN_RUNS} runs, {MAIN_PLAN_SHA256[:12]}, {MAIN_PARTS})"
        )


def disable_tf32() -> dict:
    """Full FP32 on every neural run: no TF32 in cuDNN or in matmuls. Called first thing in main()."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return tf32_flags()


def tf32_flags() -> dict:
    return {
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda.matmul.allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def pbs_queue() -> str:
    return os.environ.get("PBS_QUEUE", "")


def on_queue(name: str) -> bool:
    """PBS reports the routing queue or its execution queue (`<name>-exec`)."""
    return re.fullmatch(rf"{name}(-exec)?", pbs_queue()) is not None


def queue_guard(run: EnvRun) -> None:
    """Neural runs on gpuhopper with a GPU; XGBoost on normal, with no GPU."""
    cuda = torch.cuda.is_available()
    if not run.xgb and not (cuda and on_queue(NEURAL_QUEUE)):
        raise SystemExit(
            f"FAIL: {run.arm} trains on {NEURAL_QUEUE} (bird_1h_env_train.pbs); PBS_QUEUE is {pbs_queue()!r}, "
            f"CUDA {cuda}"
        )
    if run.xgb and (cuda or not on_queue(CPU_QUEUE)):
        raise SystemExit(
            f"FAIL: {run.arm} is a CPU run on {CPU_QUEUE} (bird_1h_env_train_cpu.pbs); PBS_QUEUE is {pbs_queue()!r}, "
            f"CUDA {cuda}"
        )


def freeze_guard() -> dict:
    """The env freeze exists, is tracked and is unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    S2 = b1h.S2
    previous = S2.FREEZE
    S2.FREEZE = REPO / "preregistration" / FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def mode_guards(run: EnvRun, *, calibrate: bool, steps: int | None) -> dict | None:
    """Queue always; the committed freeze and the registered budget except in --calibrate. Returns the attestation."""
    queue_guard(run)
    if calibrate:
        return None
    attestation = freeze_guard()
    if not run.xgb and steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {steps}")
    return attestation


# Training data.
def env_path(stem: str) -> Path:
    return ENV_DIR / f"{stem}_env.parquet"


def check_env_files(stems) -> None:
    missing = [str(env_path(s)) for s in stems if not env_path(s).exists()]
    if missing:
        raise SystemExit("FAIL: missing env files (run era5_extract.py --assemble):\n  " + "\n  ".join(missing))


def env_files(stems) -> dict[str, str]:
    """sha256 of each env parquet."""
    digests = {}
    for stem in stems:
        digest = hashlib.sha256()
        with open(env_path(stem), "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digests[stem] = digest.hexdigest()
    return digests


def env_windows(stem: str, ids, stats=None) -> HorizonWindows:
    """b1h.horizon_windows at C = 12 with the bird's ERA5 joined; `stats` (the training pool's) standardise any split."""
    windows = HorizonWindows.build(
        b1h.SEGMENTS / f"{stem}_segments.parquet",
        ids,
        context=CORE,
        core=CORE,
        horizons=HORIZONS,
        bin_seconds=b1h.BIN_SECONDS,
        clip=b1h.CLIP,
        name=stem,
        env_parquet=env_path(stem),
    )
    return windows if stats is None else replace(windows, env_stats=stats)


def training_source(run: EnvRun, payload: dict) -> tuple[SourcePool, str, tuple | None]:
    """
    The run's fold pool and, for an env arm, the statistics that standardise it: computed from this training pool alone,
    then set on every species. A rerun's pool is the main study's, unchanged.
    """
    if not run.env:
        pool, node = b1h.training_source(run.base, payload)
        return pool, node, None
    weights = b1h.fold_probabilities(run.fold, payload)
    b1h.assert_fold_sources(run.fold, weights)
    check_env_files(sorted(weights))
    species = {stem: env_windows(stem, b1h.split_ids(stem, "train", payload)) for stem in sorted(weights)}
    stats = env_stats(SourcePool(species, weights))
    return SourcePool({n: replace(w, env_stats=stats) for n, w in species.items()}, weights), f"fold {run.fold}", stats


def build_config(run: EnvRun, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    """The main-study configuration of the run's base arm; an env arm adds its 15 channels and nothing else."""
    config = b1h.build_config(run.base, grid_index, steps, pool_names, payload)
    return replace(config, env_channels=ENV_CHANNELS) if run.env else config


def stats_payload(stats) -> dict | None:
    if stats is None:
        return None
    mean, sd = stats
    return {"channels": list(era5.ENV_CHANNELS[: era5.N_STANDARDISED]), "mean": [float(v) for v in mean],
            "sd": [float(v) for v in sd]}


def stats_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """A record's env_stats back as (mean, sd), for standardising validation or test windows."""
    if payload["channels"] != list(era5.ENV_CHANNELS[: era5.N_STANDARDISED]):
        raise SystemExit(f"FAIL: env_stats channels {payload['channels']} are not era5.ENV_CHANNELS")
    return np.asarray(payload["mean"], dtype=float), np.asarray(payload["sd"], dtype=float)


def env_coverage(windows: HorizonWindows, chunk: int = b1h.GATHER_CHUNK) -> dict:
    """ERA5 coverage of a species' training windows: tokens, whole windows and segment rows with every covariate."""
    observed = np.isfinite(windows.env).all(axis=1)
    tokens = complete = 0
    for start in range(0, windows.n_windows, chunk):
        rows = history_positions(windows.keys, windows.anchors[start : start + chunk], windows.context)[:, 1:]
        seen = observed[rows]
        tokens += int(seen.sum())
        complete += int(seen.all(axis=1).sum())
    n = windows.n_windows
    return {
        "windows": n,
        "token_share": tokens / (n * (CORE - 1)) if n else None,
        "window_share_every_token": complete / n if n else None,
        "row_share": float(observed.mean()) if len(observed) else None,
    }


def bytes_on_device(source_pool: SourcePool, channels: int) -> int:
    """PreparedPool's float32 feature tensor at C = 12 with `channels` per token."""
    return sum(source_pool.species[n].n_windows for n in source_pool.names) * (CORE - 1) * channels * 4


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)


# Runners.
def require_fp32() -> dict:
    flags = tf32_flags()
    if any(flags.values()):
        raise SystemExit(f"FAIL: TF32 is on ({flags}); every neural run here is full FP32. Call disable_tf32 first")
    return flags


def run_torch(
    run: EnvRun, index: int, steps: int, grid_index: int, source: str, payload: dict, attestation: dict | None,
    calibrate: bool = False,
) -> None:
    require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node, stats = training_source(run, payload)
    b1h.check_window_counts(source_pool)
    k = horizon_k(source_pool)
    shares = b1h.clip_shares(source_pool)
    coverage = {n: env_coverage(source_pool.species[n]) for n in source_pool.names} if run.env else None
    pool = PreparedPool.from_source_pool(source_pool, device=device, horizon_k=k)
    built = time.time() - started
    print(
        f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], C = {CORE}, "
        f"{pool.n_features} channels, {pool.bytes_on_device() / 1e9:.2f} GB on device, built in {built:.0f}s",
        flush=True,
    )
    print(f"k_h              : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))}", flush=True)
    config = build_config(run, grid_index, CALIBRATION_STEPS if calibrate else steps, pool.names, payload)
    if calibrate:
        return calibrate_torch(run, config, pool, device, k, shares, built, stats)

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
            "context": CORE,
            "env_channels": config.env_channels,
            "env_stats": stats_payload(stats),
            "selection": selection,
            "model": result.model.state_dict(),
        },
        reported,
    )
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares, stats,
                         coverage, attestation) | {
        "config": asdict(config),
        "n_parameters": result.model.n_parameters(),
        "pool_windows": pool.n_windows,
        "pool_kind": type(pool).__name__,
        "bytes_on_device": pool.bytes_on_device(),
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


def calibrate_torch(run, config, pool, device, k, shares, built, stats) -> None:
    """1,000 updates, no checkpoint, no record, no validation or test; throughput, GPU and host memory, head losses."""
    probe = pool.sample_horizon(torch.Generator(device=device).manual_seed(b1h.PROBE_SEED), b1h.LOSS_PROBE)
    torch.manual_seed(config.seed)
    start_losses = b1h.probe_losses(config.build().to(device), probe, False)
    timed, ticks, every = b1h.TimedPool(pool, device), [], 50
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with b1h.GpuPoller() as poller:
        began = time.time()
        result = train(
            config,
            timed,
            device=device,
            checkpoint_path=None,
            checkpoint_every=0,
            log_every=every,
            progress=lambda step, loss: ticks.append((step, time.perf_counter())),
        )
        seconds = time.time() - began
    end_losses = b1h.probe_losses(result.model, probe, False)
    step_times = np.diff([t for _, t in ticks]) / every
    sample_times = np.array(timed.sample_seconds)
    per_step = float(np.median(step_times)) if len(step_times) else seconds / config.steps
    write_calibration(
        run,
        {
            "run_id": run.run_id,
            "env_channels": config.env_channels,
            "pool_windows": pool.n_windows,
            "pool_channels": pool.n_features,
            "bytes_on_device": pool.bytes_on_device(),
            "build_seconds": built,
            "updates": result.applied_updates,
            "train_seconds": seconds,
            "seconds_per_update_median": per_step,
            "sample_seconds_per_batch_median": float(np.median(sample_times)),
            "extrapolated_train_hours_20000": per_step * REGISTERED_STEPS / 3600,
            "extrapolated_total_hours_20000": (built + per_step * REGISTERED_STEPS) / 3600,
            "gpu": poller.summary(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "peak_gpu_memory_allocated_gb": (
                torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
            ),
            "peak_rss_gb": peak_rss_gb(),
            "tf32": tf32_flags(),
            "pbs_queue": pbs_queue(),
            "horizon_k": list(k),
            "env_stats": stats_payload(stats),
            "clip_shares": shares,
            "probe": {"windows": b1h.LOSS_PROBE, "seed": b1h.PROBE_SEED, "start": start_losses, "end": end_losses},
            "head_loss_history": result.head_loss_history,
            "skipped_steps": result.skipped_steps,
        },
    )


def run_xgb(
    run: EnvRun, index: int, grid_index: int, source: str, payload: dict, attestation: dict | None,
    calibrate: bool = False,
) -> None:
    from animal_movement_fm.models.xgboost import WindowXGBoost

    started = time.time()
    source_pool, node, stats = training_source(run, payload)
    b1h.check_window_counts(source_pool)
    k = horizon_k(source_pool)  # the pool's k_h, as the main study's XGB-Bb
    shares = b1h.clip_shares(source_pool)
    coverage = {n: env_coverage(source_pool.species[n]) for n in source_pool.names}
    features, targets, sampled = b1h.xgb_horizon_set(source_pool, run.base, k)  # the main study's draws
    if np.isnan(features).any():
        raise SystemExit("FAIL: NaN features at C = 12; missing ERA5 is zero with env_observed 0, never NaN")
    built = time.time() - started
    print(
        f"xgboost data     : {len(features):,} windows ({sampled}) [{node}], {features.shape[-1]} channels, "
        f"{features.nbytes / 1e9:.2f} GB, built in {built:.0f}s",
        flush=True,
    )
    model = WindowXGBoost(random_state=run.seed, n_jobs=int(os.environ.get("PBS_NCPUS", "0")) or None)
    began = time.time()
    model.fit(features, targets)
    fit_seconds = time.time() - began
    if calibrate:
        return write_calibration(
            run,
            {
                "run_id": run.run_id,
                "horizon": run.horizon,
                "training_windows": len(features),
                "channels": int(features.shape[-1]),
                "feature_gb": features.nbytes / 1e9,
                "build_seconds": built,
                "fit_seconds": fit_seconds,
                "peak_rss_gb": peak_rss_gb(),
                "pbs_queue": pbs_queue(),
                "horizon_k": list(k),
                "env_stats": stats_payload(stats),
                "clip_shares": shares,
                "sampling": sampled,
            },
        )
    stem_path = RESULTS / f"{run.run_id}_xgb"
    model.save(stem_path)
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares, stats,
                         coverage, attestation) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "allow_missing": model.allow_missing,
        "training_windows": int(len(features)),
        "channels": int(features.shape[-1]),
        "sampling": sampled,
        "target_space": f"scaled displacement / k_h at {run.horizon} h",
        "fit_seconds": fit_seconds,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(
    run: EnvRun, index: int, grid_index: int, source: str, node: str, pool_names, payload: dict, k: tuple,
    shares: dict, stats, coverage: dict | None, attestation: dict | None,
) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "base_arm": run.base.arm,
        "main_study_run": run.base.run_id,
        "compared_with": run.compared_with,
        "recipe": b1h.BT.counterpart(run.base).arm,
        "fold": run.fold,
        "seed": run.seed,
        "horizon": run.horizon or None,
        "horizons": list(HORIZONS),
        "context": CORE,
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
        "env_stats": stats_payload(stats),
        "env_stats_source": "the run's training pool, window weight p_s / n_s" if run.env else None,
        "env_coverage": coverage,
        "env_files": env_files(pool_names) if run.env else None,
        "queue": queue_of(index),
        "pbs_queue": pbs_queue(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "tf32": tf32_flags(),
        "tier": TAG,
        "registered_by": FREEZE,
        "freeze": attestation,
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: EnvRun, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def write_calibration(run: EnvRun, payload: dict) -> None:
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION / f"{run.run_id}_calibration.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "head_loss_history"}, indent=2, default=str))
    print(f"wrote {path}; calibration only, no run record", flush=True)


# Dry run.
def env_matches(main: SourcePool, env: SourcePool, chunk: int = b1h.GATHER_CHUNK) -> list[str]:
    """
    Failures (empty if none) of the env pool against the main study's: equal anchors, targets and keys per species; on
    every window, the first 12 channels, token mask, targets, target mask and scale bitwise equal; then k_h bitwise.
    """
    if tuple(main.names) != tuple(env.names) or not np.array_equal(main.probabilities, env.probabilities):
        return ["species or probabilities differ"]
    failures = []
    for name in main.names:
        m, e = main.species[name], env.species[name]
        if not all(np.array_equal(getattr(m, a), getattr(e, a)) for a in ("anchors", "targets", "keys")):
            failures.append(f"{name}: anchors, targets or keys differ")
            continue
        for start in range(0, m.n_windows, chunk):
            ids = np.arange(start, min(start + chunk, m.n_windows))
            plain, covariates = m.gather(ids), e.gather(ids)
            same = (
                covariates[0].shape[-1] == N_FEATURES + ENV_CHANNELS
                and np.array_equal(covariates[0][..., :N_FEATURES], plain[0])
                and all(np.array_equal(a, b) for a, b in zip(plain[1:], covariates[1:]))
            )
            if not same:
                failures.append(f"{name}: windows {start}-{ids[-1]} differ in the first 12 channels, masks, targets "
                                "or scale")
                break
    if not failures and horizon_k(main) != horizon_k(env):
        failures.append("k_h differ")
    return failures


def tail_shares(pool: SourcePool, chunk: int = b1h.GATHER_CHUNK) -> dict:
    """
    Per standardised channel, the share of observed training tokens beyond +-TAIL_SD and the largest |value| (token
    counts, unweighted), with total precipitation's share per species.
    """
    beyond, largest, total = np.zeros(era5.N_STANDARDISED), np.zeros(era5.N_STANDARDISED), 0
    tp, tp_by_species = era5.ENV_CHANNELS.index("tp"), {}
    for name in pool.names:
        windows, hits, count = pool.species[name], 0, 0
        for start in range(0, windows.n_windows, chunk):
            z = windows.gather(np.arange(start, min(start + chunk, windows.n_windows)))[0][..., N_FEATURES:]
            values = np.abs(z[z[..., -1] > 0][:, :-1])
            beyond += (values > TAIL_SD).sum(axis=0)
            if len(values):
                largest = np.maximum(largest, values.max(axis=0))
            total += len(values)
            hits, count = hits + int((values[:, tp] > TAIL_SD).sum()), count + len(values)
        tp_by_species[name] = hits / count if count else None
    return {
        "tail_sd": TAIL_SD,
        "observed_tokens": total,
        "channels": {
            c: {"share_beyond": float(beyond[i] / total) if total else None, "max_abs": float(largest[i])}
            for i, c in enumerate(era5.ENV_CHANNELS[: era5.N_STANDARDISED])
        },
        "tp_share_beyond_by_species": tp_by_species,
    }


def check_all_birds(payload: dict) -> None:
    """Every bird's env file, its sha256 and the ERA5 coverage of its training windows."""
    check_env_files(BIRDS)
    digests, rows = env_files(BIRDS), {}
    for bird in BIRDS:
        windows = env_windows(bird, b1h.split_ids(bird, "train", payload))
        b1h.check_window_counts(SourcePool({bird: windows}, {bird: 1.0}))
        rows[bird] = env_coverage(windows)
        c = rows[bird]
        print(f"{bird:<26} {c['windows']:>10,} windows; tokens {c['token_share']:.4f}, windows with every token "
              f"{c['window_share_every_token']:.4f}, rows {c['row_share']:.4f}; sha256 {digests[bird][:12]}")
    BUILD.mkdir(parents=True, exist_ok=True)
    path = BUILD / "dry_run_birds.json"
    path.write_text(json.dumps({"coverage": rows, "env_files": digests}, indent=2) + "\n")
    print(f"\n{len(BIRDS)} birds: env files present, training counts match the pins. wrote {path}")
    print("dry run only; nothing was trained and no test partition was opened.")


def dry_run(payload: dict, index: int | None) -> None:
    """Every bird's env coverage, or one index's fold: the env pool against the main study's, statistics and tails."""
    print("pins: PASS (bird_1h_train.load_registered); 1 h plan and evaluator parts unchanged")
    if index is None:
        return check_all_birds(payload)
    run = PLAN[index]
    started = time.time()
    print(f"index {index:>3}  {run.run_id}  (fold {run.fold}, queue {queue_of(index)}); the checks are the fold's")
    env_run = EnvRun("BbE", run.fold, SEEDS[0])
    main_pool, node = b1h.training_source(env_run.base, payload)
    b1h.check_window_counts(main_pool)
    env_pool, _, stats = training_source(env_run, payload)
    b1h.check_window_counts(env_pool)
    windows = sum(env_pool.species[n].n_windows for n in env_pool.names)
    print(f"  pool           : {node}, {len(env_pool.names)} species, {windows:,} windows; counts match the pins")
    failures = env_matches(main_pool, env_pool)
    print("  main study     : " + ("PASS" if not failures else "FAIL " + "; ".join(failures[:5]))
          + " (anchors, targets, keys; every window's first 12 channels, masks, targets and scale; k_h)")
    if failures:
        raise SystemExit("FAIL: the env pool's movement inputs differ from the main study's")
    k = horizon_k(env_pool)
    print(f"  k_h            : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))} (equal to the main study's)")
    mean, sd = stats
    print("  env_stats      : channel, mean, sd (training pool, window weight p_s / n_s)")
    for c, m, s in zip(era5.ENV_CHANNELS, mean, sd):
        print(f"    {c:<10} {m:>14.6g} {s:>14.6g}")
    coverage = {n: env_coverage(env_pool.species[n]) for n in env_pool.names}
    token_share = sum(c["token_share"] * c["windows"] for c in coverage.values()) / windows
    print(f"  ERA5 coverage  : {token_share:.4f} of training tokens; lowest "
          + ", ".join(f"{n} {c['token_share']:.4f}" for n, c in sorted(coverage.items(),
                                                                       key=lambda kv: kv[1]["token_share"])[:3]))
    tails = tail_shares(env_pool)
    print(f"  beyond +-{TAIL_SD:g} SD : share of {tails['observed_tokens']:,} observed training tokens, and max |z|")
    for c, t in tails["channels"].items():
        print(f"    {c:<10} {t['share_beyond']:>10.2e} {t['max_abs']:>10.1f}")
    tp_top = sorted(((v, n) for n, v in tails["tp_share_beyond_by_species"].items() if v), reverse=True)[:5]
    print("    tp by species: " + (", ".join(f"{n} {v:.2e}" for v, n in tp_top) or "none beyond"))
    plain_bytes, env_bytes = bytes_on_device(env_pool, N_FEATURES), bytes_on_device(env_pool, N_FEATURES + ENV_CHANNELS)
    print(f"  bytes_on_device: {env_bytes / 1e9:.2f} GB at {N_FEATURES + ENV_CHANNELS} channels "
          f"({plain_bytes / 1e9:.2f} GB at {N_FEATURES}); XGBBbE features "
          f"{b1h.XGB_POOLED_DRAWS * (CORE - 1) * (N_FEATURES + ENV_CHANNELS) * 4 / 1e9:.2f} GB per horizon")
    grid_index, _ = b1h.T.selected_grid_index(None)
    for arm in ("BbE", "LSTMBbE"):
        config = build_config(EnvRun(arm, run.fold, SEEDS[0]), grid_index, REGISTERED_STEPS, env_pool.names, payload)
        print(f"  config {arm:<8}: trunk {config.trunk}, d_model {config.d_model}, layers {config.num_layers}, "
              f"lr {config.lr:g}, horizons {config.horizons}, env_channels {config.env_channels}")
    BUILD.mkdir(parents=True, exist_ok=True)
    path = BUILD / f"dry_run_f{run.fold}.json"
    path.write_text(json.dumps({
        "fold": run.fold, "pool_species": list(env_pool.names), "windows": windows, "horizon_k": list(k),
        "env_stats": stats_payload(stats), "coverage": coverage, "tails": tails,
        "bytes_on_device": {str(N_FEATURES): plain_bytes, str(N_FEATURES + ENV_CHANNELS): env_bytes},
        "env_files": env_files(env_pool.names),
    }, indent=2) + "\n")
    print(f"  wrote {path}")
    print(f"  dry run only; nothing trained, no test partition opened ({time.time() - started:.0f}s).")


def print_plan(plan: list[EnvRun], *, runs: bool = False) -> None:
    print(f"{'indices':<10} {'n':>4}  {'queue':<10} what")
    for first, last, queue, what in BLOCKS:
        print(f"{first:>3}-{last:<6} {last - first + 1:>4}  {queue:<10} {what}")
    print(f"\n{len(plan)} runs (0-{len(plan) - 1}).")
    if runs:
        print(f"\n{'idx':>4}  {'run':<18} {'compared with':<16} queue")
        for i, run in enumerate(plan):
            print(f"{i:>4}  {run.run_id:<18} {run.compared_with:<16} {queue_of(i)}")


def main() -> None:
    flags = disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan's blocks and exit")
    parser.add_argument("--list-runs", action="store_true", help="print every index and exit")
    parser.add_argument("--dry-run", action="store_true", help="check every bird's env data, or --index's fold, and exit")
    parser.add_argument(
        "--calibrate", action="store_true",
        help=f"{CALIBRATION_STEPS:,} updates (XGB: one fit) to {CALIBRATION.name}/; no record",
    )
    args = parser.parse_args()

    if args.list or args.list_runs:
        return print_plan(PLAN, runs=args.list_runs)
    check_main_study()
    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    index = int(raw) if raw is not None else None
    if index is not None and not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")
    if args.dry_run:
        return dry_run(b1h.load_registered(), index)
    if index is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")

    run = PLAN[index]
    attestation = mode_guards(run, calibrate=args.calibrate, steps=args.steps)
    payload = b1h.load_registered()
    grid_index, source = b1h.T.selected_grid_index(None)
    for directory in ((CALIBRATION,) if args.calibrate else (RESULTS, CHECKPOINTS)):
        directory.mkdir(parents=True, exist_ok=True)
    print(
        f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1}, fold {run.fold}, "
        f"{'env' if run.env else 'no env'}, compared with {run.compared_with}"
        f"{', CALIBRATION' if args.calibrate else ''})",
        flush=True,
    )
    print(f"configuration    : grid {grid_index} from {source}; TF32 {flags}; PBS_QUEUE {pbs_queue()!r}", flush=True)
    if run.xgb:
        run_xgb(run, index, grid_index, source, payload, attestation, calibrate=args.calibrate)
    else:
        run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload, attestation, calibrate=args.calibrate)


if __name__ == "__main__":
    main()
