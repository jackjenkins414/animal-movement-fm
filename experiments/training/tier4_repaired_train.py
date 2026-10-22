# GenAI reference: Like the other scripts, I gave GenAI the five arms, their registered pools/twins, the repaired encoder settings and the main 
# guards/checks; GenAI wrote the file, reusing our existing Tier 2-4 training code instead of duplicating it.
"""
Train the repaired encoder reruns of the Tier 2-4 pooled plain-trunk transformer, one model per PBS subjob (exploratory).

Each arm is a registered plain trunk (Tier 2's B, Tier 3's B9, Tier 4's B26 and BdM) or Tier 4's Zd-T pool as a plain trunk (BdTF), retrained with 
the repaired encoder: input_scale "none", norm_first, step_transform "signed_log1p". Everything else is the twin's own recipe at 6 h. Plan: --list. 
Pools, twins and step magnitudes: --dry-run. Throughput and the twin reproduction check: --calibrate.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from dataclasses import MISSING, asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))
sys.path.insert(0, str(REPO))

import bird_1h_bbf_train as bbf  # noqa: E402
import bird_1h_env_train as env  # noqa: E402
import tier2_final_train as T  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402
import tier3_train as T3  # noqa: E402
import tier4_train as T4  # noqa: E402
import transformer_repair_train as TR  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.models.fm import STEP_CHANNELS, signed_log1p  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.loss import normalised_huber  # noqa: E402
from animal_movement_fm.train.pool import ENCODE_CHUNK, PreparedPool  # noqa: E402

RESULTS = REPO / "results" / "tier4_repaired"
CHECKPOINTS = REPO / "checkpoints" / "tier4_repaired"
CALIBRATION = REPO / "results" / "tier4_repaired_calibration"
FREEZE = "tier4_repaired_exploratory_freeze.md"
TAG = "tier4_repaired"
QUEUE = env.NEURAL_QUEUE  # gpuhopper

REGISTERED_STEPS = T4.REGISTERED_STEPS  # 20,000 applied updates at batch 256, the twins' budget
CALIBRATION_STEPS = 1_000
LOG_EVERY = 500  # train()'s default, at which every twin's loss_history was logged
SEEDS = T.SEEDS
NES, WOLF = "mirounga_angustirostris", "canis_lupus"
HELD_OUT = T3.HELD_OUT  # gull, wolf, NES

STEP_LEVELS = (10.0, 100.0, 1_000.0)  # the dry run's |scaled step| shares
PROBE_WINDOWS = 4_096  # calibration's fixed training windows for loss against the zero predictor
PROBE_SEED = 123

# The repaired encoder: the repair probe's V4, pinned here so a change to the probe's table cannot move this study, and
# checked against the BbF study's pin so "repaired" means what it meant there.
SWITCHES = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
if TR.VARIANTS["V4"] != SWITCHES or bbf.SWITCHES != SWITCHES:
    raise RuntimeError(f"the repair probe's V4 is {TR.VARIANTS['V4']} and BbF's is {bbf.SWITCHES}; this study froze {SWITCHES}")

# Sealed comparator predictions, read by the evaluator and only checked for existence here.
SEALED_PREDICTIONS = {tier: REPO / "results" / f"tier{tier}_sealed" / "predictions" for tier in ("2", "3", "4")}
TWIN_RESULTS = {"2": T.TIER2_FINAL_RESULTS, "3": T3.RESULTS, "4": T4.RESULTS}

# arm -> the tier whose registered pool and recipe it reruns, its unrepaired twin there, and the species it is scored
# on. `pool` and `recipe` default to the twin; BdTF has no plain twin, so it reads Zd-T's pool with B26's recipe.
ARMS: dict[str, dict] = {
    "B3F": {"tier": "2", "twin": "B"},
    "B9F": {"tier": "3", "twin": "B9"},
    "B26F": {"tier": "4", "twin": "B26"},
    "BdMF": {"tier": "4", "twin": "BdM", "species": NES},
    "BdTF": {"tier": "4", "twin": None, "pool": "ZdT", "recipe": "B26", "species": WOLF},
}
TIER_RUN = {"2": T.Run, "3": T3.Run, "4": T4.Run}


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int

    @property
    def species(self) -> str:
        """The one held-out species a domain arm is scored on; empty for the arms scored on all three."""
        return ARMS[self.arm].get("species", "")

    @property
    def run_id(self) -> str:
        return f"{self.arm}_{self.species}_s{self.seed}" if self.species else f"{self.arm}_s{self.seed}"

    @property
    def tier(self) -> str:
        return ARMS[self.arm]["tier"]

    @property
    def scored_on(self) -> tuple[str, ...]:
        return (self.species,) if self.species else HELD_OUT

    def _tier_run(self, arm: str | None):
        if arm is None:
            return None
        species = T4.ARMS[arm].get("species", "") if self.tier == "4" else ""
        return TIER_RUN[self.tier](arm, species, self.seed)

    @property
    def twin(self):
        """The unrepaired run this one repeats with the switches on (same tier, pool, recipe and seed), or None."""
        return self._tier_run(ARMS[self.arm]["twin"])

    @property
    def recipe(self):
        """The registered run whose builder makes this run's configuration before the switches."""
        spec = ARMS[self.arm]
        return self._tier_run(spec.get("recipe", spec["twin"]))

    @property
    def pool_run(self):
        """The registered run whose pool this run reads."""
        spec = ARMS[self.arm]
        return self._tier_run(spec.get("pool", spec["twin"]))

    @property
    def comparators(self) -> tuple[str, ...]:
        """Run ids the evaluator reads beside P: the twin, or for BdTF B26F and Zd-T at the same seed."""
        if self.twin is not None:
            return (self.twin.run_id,)
        return (Run("B26F", self.seed).run_id, T4.Run("ZdT", WOLF, self.seed).run_id)


def build_plan() -> list[Run]:
    """The plan. Position is the PBS array index: append, never insert or reorder."""
    return [Run(arm, s) for arm in ARMS for s in SEEDS]


PLAN = build_plan()
# submission blocks of at most ten; every block is on gpuhopper
BLOCKS: tuple[tuple[int, int, str], ...] = ((0, 9, "B3F, B9F"), (10, 19, "B26F, BdMF"), (20, 24, "BdTF"))
if len(PLAN) != 25 or [b[0] for b in BLOCKS] != [0, 10, 20] or BLOCKS[-1][1] != len(PLAN) - 1:
    raise RuntimeError(f"the plan has {len(PLAN)} runs; 25 are planned, five arms of five seeds, in three blocks")
_OTHER_IDS = {r.run_id for plan in (T.PLAN, T.TIER_E_PLAN, T3.PLAN, T3.TIER_E_PLAN, T4.PLAN) for r in plan}
if len({run.run_id for run in PLAN}) != len(PLAN) or {run.run_id for run in PLAN} & _OTHER_IDS:
    raise RuntimeError("run ids must be distinct, and distinct from every Tier 2, Tier 3, Tier 4 and Tier E run id")
if any(run.twin is not None and run.twin not in {"2": T.PLAN, "3": T3.PLAN, "4": T4.PLAN}[run.tier] for run in PLAN):
    raise RuntimeError("every twin must be a run of its tier's registered plan")


# Guards.
def queue_guard(run: Run) -> None:
    cuda = torch.cuda.is_available()
    if not (cuda and env.on_queue(QUEUE)):
        raise SystemExit(
            f"FAIL: {run.run_id} trains on {QUEUE} (tier4_repaired_train.pbs); PBS_QUEUE is {env.pbs_queue()!r}, "
            f"CUDA {cuda}"
        )


def freeze_guard() -> dict:
    """This study's freeze exists, is tracked and is unchanged against HEAD (the Tier 2 check, pointed at this freeze)."""
    previous = S2.FREEZE
    S2.FREEZE = REPO / "preregistration" / FREEZE
    try:
        return S2.assert_freeze_committed()
    finally:
        S2.FREEZE = previous


def mode_guards(run: Run, *, calibrate: bool, steps: int | None) -> dict | None:
    """Queue always; the committed freeze and the registered budget except in --calibrate. Returns the attestation."""
    queue_guard(run)
    if calibrate:
        return None
    attestation = freeze_guard()
    if steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {steps}")
    return attestation


# Pools and configurations.
def load_payloads() -> dict:
    """The registered Tier 3 and Tier 4 pools, each checked against its constants (Tier 2's is checked on load)."""
    return {"3": T3.load_registered_pool(), "4": T4.load_registered_pool()}


def training_source(run: Run, payloads: dict) -> tuple[SourcePool, str]:
    """
    The registered training pool, through the loader of the tier that registered it. Tier 2's B pool is read as
    tier3_train reads it for LSTM-B3 and XGB-B3: the three sources under tier2_source_pool_weights.json, which is what
    tier2_final_train.training_pool built for B. Training individuals only.
    """
    if run.tier == "2":
        return T3.pooled_source(T3.Run("LSTMB3", "", run.seed), payloads["3"])
    if run.tier == "3":
        return T3.pooled_source(run.pool_run, payloads["3"])
    return T4.pooled_source(run.pool_run, payloads["4"])


def check_window_counts(source_pool: SourcePool, payloads: dict) -> None:
    """Every source's training windows equal the registered count (Tier 4's file holds all 26)."""
    registered = payloads["4"]["train_window_counts"]
    drift = [
        f"{name}: built {source_pool.species[name].n_windows:,}, registered {registered[name]:,}"
        for name in source_pool.names
        if source_pool.species[name].n_windows != registered[name]
    ]
    if drift:
        raise SystemExit("FAIL: training windows differ from the registered counts\n  " + "\n  ".join(drift))


def recipe_config(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> TrainConfig:
    """The unrepaired configuration, made by the registering tier's own builder."""
    if run.tier == "2":
        return T.build_config(run.recipe, grid_index, steps)
    module = {"3": T3, "4": T4}[run.tier]
    return module.build_config(run.recipe, grid_index, steps, pool_names, payloads[run.tier])


def build_config(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> TrainConfig:
    """The recipe's configuration with the repaired encoder's switches; nothing else."""
    return replace(recipe_config(run, grid_index, steps, pool_names, payloads), **SWITCHES)


def config_differences(run: Run, grid_index: int, steps: int, pool_names, payloads: dict) -> dict:
    """The fields in which the run's configuration differs from its recipe's."""
    recipe = asdict(recipe_config(run, grid_index, steps, pool_names, payloads))
    got = asdict(build_config(run, grid_index, steps, pool_names, payloads))
    return {name: got[name] for name in got if got[name] != recipe[name]}


def _json(value):
    return json.loads(json.dumps(value))


def with_defaults(stored: dict) -> dict:
    """A saved configuration with every field written after it filled at its TrainConfig default."""
    defaults = {f.name: f.default for f in fields(TrainConfig) if f.default is not MISSING}
    return _json({name: stored.get(name, default) for name, default in defaults.items()} | stored)


def twin_problems(run: Run, grid_index: int, pool_names, payloads: dict) -> list[str]:
    """
    The twin's record and checkpoint on disk against what this run assumes: its configuration equals the recipe's
    (fields absent from it at their defaults), it applied the registered budget, and its comparator predictions exist.
    Empty if all hold. Prediction files are checked for existence only and never read.
    """
    problems = []
    if run.twin is not None:
        directory = TWIN_RESULTS[run.tier]
        path = directory / f"{run.twin.run_id}.json"
        if not path.exists():
            problems.append(f"{run.twin.run_id}'s record {path} is missing")
        else:
            record = json.loads(path.read_text())
            expected = _json(asdict(recipe_config(run, grid_index, REGISTERED_STEPS, pool_names, payloads)))
            if with_defaults(record["config"]) != expected:
                problems.append(f"{run.twin.run_id}'s recorded config differs from its builder's")
            if record.get("applied_updates") != REGISTERED_STEPS:
                problems.append(f"{run.twin.run_id} applied {record.get('applied_updates')} updates")
        if not (directory / f"{run.twin.run_id}_reported.pt").exists():
            problems.append(f"{run.twin.run_id}_reported.pt is missing")
    for species in run.scored_on:
        wanted = [SEALED_PREDICTIONS["2"] / f"{species}__P.parquet"]
        if run.twin is not None:
            wanted.append(SEALED_PREDICTIONS[run.tier] / f"{species}__{run.twin.run_id}.parquet")
        else:
            wanted.append(SEALED_PREDICTIONS["4"] / f"{species}__{run.comparators[1]}.parquet")
        problems += [f"{path.name} is missing from {path.parent}" for path in wanted if not path.exists()]
    return problems


def initial_state_differences(run: Run, grid_index: int, pool_names, payloads: dict) -> dict:
    """
    The repaired model and its recipe's model, each built after torch.manual_seed(seed) as train() builds them. The
    switches add no randomly initialised parameter, so every shared tensor should be equal and the only extra ones the
    Pre-LN stack's final LayerNorm.
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


EXPECTED_EXTRA = ["encoder.norm.bias", "encoder.norm.weight"]


def step_magnitudes(source_pool: SourcePool, cache: dict | None = None) -> dict:
    """
    Max |scaled step| and the share of window tokens with |d_east_scaled| or |d_north_scaled| above each level, before
    and after signed_log1p, over every training window of the pool (unweighted). 6 h windows carry no clip, so these
    are the raw values the transform receives. `cache` holds per-species counts so a species is scanned once.
    """
    cache = {} if cache is None else cache
    for name in source_pool.names:
        if name in cache:
            continue
        windows = source_pool.species[name]
        tokens, maxima = 0, {"before": 0.0, "after": 0.0}
        above = {stage: [0] * len(STEP_LEVELS) for stage in maxima}
        for start in range(0, windows.n_windows, ENCODE_CHUNK):
            features, _, _ = windows.gather(np.arange(start, min(start + ENCODE_CHUNK, windows.n_windows)),
                                            sigma=SCALE_FLOOR_M)
            after = signed_log1p(torch.as_tensor(features, dtype=torch.float32)).numpy()
            for stage, values in (("before", features), ("after", after)):
                largest = np.abs(values[..., list(STEP_CHANNELS)]).max(axis=-1)
                maxima[stage] = max(maxima[stage], float(largest.max()))
                for j, level in enumerate(STEP_LEVELS):
                    above[stage][j] += int((largest > level).sum())
            tokens += features.shape[0] * features.shape[1]
        cache[name] = {"tokens": tokens, "max": maxima, "above": above}
    entries = [cache[name] for name in source_pool.names]
    total = sum(e["tokens"] for e in entries)
    out = {"tokens": total, "levels": list(STEP_LEVELS)}
    for stage in ("before", "after"):
        largest = max(source_pool.names, key=lambda n: cache[n]["max"][stage])
        out[stage] = {
            "max": cache[largest]["max"][stage],
            "max_species": largest,
            "share_above": [sum(e["above"][stage][j] for e in entries) / total for j in range(len(STEP_LEVELS))],
        }
    return out


# Runners.
def run_torch(run: Run, index: int, grid_index: int, source: str, payloads: dict, attestation: dict | None) -> None:
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payloads)
    check_window_counts(source_pool, payloads)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], "
          f"built in {time.time() - started:.0f}s", flush=True)
    config = build_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads)
    differences = config_differences(run, grid_index, REGISTERED_STEPS, pool.names, payloads)
    if differences != SWITCHES:
        raise SystemExit(f"FAIL: {run.run_id} differs from its recipe in {differences}, not the switches alone")
    print(f"switches         : {SWITCHES}; twin {run.twin.run_id if run.twin else 'none'}", flush=True)

    result = train(
        config, pool, device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
    )
    print(f"applied updates  : {result.applied_updates:,} of {config.steps:,}; "
          f"withheld {result.skipped_steps:,} ({result.withheld_fraction:.2%})", flush=True)
    selection = {"rule": "final_checkpoint", "step": config.steps}  # every twin reported its final checkpoint
    reported = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {"run_id": run.run_id, "config": asdict(config), "grid_index": grid_index, "pool_species": list(pool.names),
         "switches": dict(SWITCHES), "selection": selection, "model": result.model.state_dict()},
        reported,
    )
    record = base_record(run, index, grid_index, source, node, attestation) | {
        "config": asdict(config),
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


class CalibrationStop(Exception):
    """Raised by the calibration's monitor after CALIBRATION_STEPS applied updates, so train() runs unchanged until then."""


def probe_losses(model, probe) -> dict:
    """normalised_huber of the model and of the zero predictor on the fixed probe windows, and their ratio."""
    features, targets = probe
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = float(normalised_huber(model(features), targets))
        zero = float(normalised_huber(torch.zeros_like(targets), targets))
    model.train(was_training)
    return {"loss": loss, "zero": zero, "ratio": loss / zero}


def calibrate_config(config: TrainConfig, pool, device, probe) -> dict:
    """
    CALIBRATION_STEPS applied updates of `config` exactly as registered (its 20,000-update schedule, train()'s logging
    cadence), then stopped by the monitor; no checkpoint, no record. The monitor fires only at the stop, after that
    update's loss is logged, so everything logged before it is what a full run logs.
    """
    logged, stamps, end = [], [], {}

    def progress(step, loss):
        logged.append(loss)
        stamps.append(time.perf_counter())

    def stop(step, model):
        end.update(probe_losses(model, probe))
        raise CalibrationStop

    torch.manual_seed(config.seed)
    start = probe_losses(config.build(input_dim=pool.n_features).to(device), probe)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.time()
    try:
        train(config, pool, device=device, checkpoint_path=None, checkpoint_every=0, log_every=LOG_EVERY,
              progress=progress, monitor=stop, monitor_every=CALIBRATION_STEPS)
    except CalibrationStop:
        pass
    else:
        raise RuntimeError(f"the calibration ran past {CALIBRATION_STEPS:,} updates; its monitor never stopped it")
    seconds = time.time() - began
    per_update = (stamps[-1] - stamps[0]) / (LOG_EVERY * (len(stamps) - 1)) if len(stamps) > 1 else seconds / CALIBRATION_STEPS
    return {
        "updates": CALIBRATION_STEPS,
        "train_seconds": seconds,
        "seconds_per_update": per_update,
        "extrapolated_train_hours_20000": per_update * REGISTERED_STEPS / 3600,
        "peak_gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
        "logged_steps": [LOG_EVERY * (i + 1) for i in range(len(logged))],
        "logged_losses": logged,
        "probe": {"windows": len(probe[1]), "seed": PROBE_SEED, "start": start, "end": end},
    }


def loss_match(logged: list[float], record: dict) -> dict:
    """The calibration's logged losses against the record's at the same applied updates: max |difference|, exact share."""
    reference = list(record.get("loss_history", []))[: len(logged)]
    if len(reference) != len(logged):
        return {"compared": 0, "note": f"the record logs {len(reference)} of {len(logged)} steps"}
    differences = [abs(a - b) for a, b in zip(logged, reference)]
    return {
        "compared": len(logged),
        "record_losses": reference,
        "max_abs_difference": max(differences),
        "exact_share": sum(a == b for a, b in zip(logged, reference)) / len(logged),
    }


def calibrate(run: Run, grid_index: int, payloads: dict) -> None:
    """
    The run and, where it has one, its unrepaired twin at the same seed, each for CALIBRATION_STEPS updates on the same
    pool. The twin's logged losses are compared step by step with its record. No record, no checkpoint for either.
    """
    env.require_fp32()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payloads)
    check_window_counts(source_pool, payloads)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    built = time.time() - started
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], built in {built:.0f}s",
          flush=True)
    probe = pool.sample(torch.Generator(device=device).manual_seed(PROBE_SEED), PROBE_WINDOWS)[:2]
    payload = {"run_id": run.run_id, "node": node, "pool_windows": pool.n_windows, "build_seconds": built,
               "switches": dict(SWITCHES), "tf32": env.tf32_flags(), "pbs_queue": env.pbs_queue(),
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    payload["repaired"] = calibrate_config(build_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads),
                                           pool, device, probe)
    if run.twin is not None:
        twin = calibrate_config(recipe_config(run, grid_index, REGISTERED_STEPS, pool.names, payloads), pool, device,
                                probe)
        path = TWIN_RESULTS[run.tier] / f"{run.twin.run_id}.json"
        twin["record"] = str(path.relative_to(REPO))
        twin["match"] = (loss_match(twin["logged_losses"], json.loads(path.read_text())) if path.exists()
                         else {"compared": 0, "note": f"{path} is missing"})
        payload["twin"] = {"run_id": run.twin.run_id} | twin
    payload["peak_rss_gb"] = env.peak_rss_gb()
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    out = CALIBRATION / f"{run.run_id}_calibration.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out}; calibration only, no run record or checkpoint", flush=True)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str, attestation: dict | None) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "species": run.species or None,
        "scored_on": list(run.scored_on),
        "seed": run.seed,
        "twin": run.twin.run_id if run.twin else None,
        "twin_tier": run.tier if run.twin else None,
        "comparators": list(run.comparators),
        "recipe": f"tier{run.tier}:{run.recipe.arm}",
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
def _share(values) -> str:
    return " / ".join(f"{v:.2e}" for v in values)


def dry_run(payloads: dict) -> None:
    """
    Every arm's pool through its registering tier's loader, with the registered window counts and its step magnitudes
    before and after signed_log1p; every run's configuration against its recipe (the switches alone may differ), its
    twin's record and checkpoint, the comparator prediction files and the same-seed initial weights. Nothing is trained
    and no test partition is opened.
    """
    grid_index, source = T.selected_grid_index(None)
    print(f"configuration    : grid {grid_index} from {source}; switches {SWITCHES}")
    failures, cache, names = [], {}, {}
    print(f"\n{'arm':<6} {'node':<32} {'species':>7} {'windows':>11}   max |step| before / after   "
          f"share above {' / '.join(f'{v:g}' for v in STEP_LEVELS)}, before | after")
    for arm in ARMS:
        run = Run(arm, SEEDS[0])
        source_pool, node = training_source(run, payloads)
        check_window_counts(source_pool, payloads)
        names[arm] = source_pool.names
        steps = step_magnitudes(source_pool, cache)
        windows = sum(source_pool.species[n].n_windows for n in source_pool.names)
        print(f"{arm:<6} {node:<32} {len(source_pool.names):>7} {windows:>11,}   "
              f"{steps['before']['max']:>10.4g} ({steps['before']['max_species']}) / {steps['after']['max']:.4g}   "
              f"{_share(steps['before']['share_above'])} | {_share(steps['after']['share_above'])}")

    print(f"\n{'idx':>3}  {'run':<34} {'twin':<36} initial weights")
    for index, run in enumerate(PLAN):
        differences = config_differences(run, grid_index, REGISTERED_STEPS, names[run.arm], payloads)
        if differences != SWITCHES:
            failures.append(f"{run.run_id}: differs from its recipe in {differences}")
        initial = initial_state_differences(run, grid_index, names[run.arm], payloads)
        if initial != {"extra": EXPECTED_EXTRA, "missing": [], "different": []}:
            failures.append(f"{run.run_id}: initial weights differ from its recipe's: {initial}")
        failures += [f"{run.run_id}: {p}" for p in twin_problems(run, grid_index, names[run.arm], payloads)]
        twin = f"tier{run.tier}:{run.twin.run_id}" if run.twin else f"none; read against {', '.join(run.comparators)}"
        print(f"{index:>3}  {run.run_id:<34} {twin:<36} "
              f"{'equal, plus encoder.norm' if not initial['different'] else 'DIFFERENT'}")
    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    print(f"\n{len(PLAN)} runs: pools and counts as registered; configs differ from their recipes in the switches "
          "alone; twins, checkpoints and comparator files present. Nothing trained; no test partition opened.")


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'run':<34} {'tier':>4}  {'twin':<36} scored on")
    for i, run in enumerate(plan):
        twin = run.twin.run_id if run.twin else "-"
        print(f"{i:>4}  {run.run_id:<34} {run.tier:>4}  {twin:<36} {', '.join(run.scored_on)}")
    print("\nblocks: " + "; ".join(f"{first}-{last} {what}" for first, last, what in BLOCKS))
    print(f"{len(plan)} exploratory runs (0-{len(plan) - 1}), all on {QUEUE}.")


def main() -> None:
    flags = env.disable_tf32()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="check every pool, config, twin and comparator, and exit")
    parser.add_argument("--calibrate", action="store_true",
                        help=f"{CALIBRATION_STEPS:,} updates of the run and its twin to {CALIBRATION.name}/; no record")
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
