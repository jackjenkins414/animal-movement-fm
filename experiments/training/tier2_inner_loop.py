# Note, I defined the experiment protocol used here, including the source roles, grid order, pseudo target folds, data access 
# rules, convergence probe, and selection procedure; GenAI assisted with implementing the surrounding run orchestration from that 
# specification, especially PBS/CLI handling, defensive checks, verbose progress/debug output, result file handling, and selection 
# report formatting. I have intentionally kept much of that defensive/verbose code because the failure messages and logs are useful 
# when running on Gadi.
"""
Run the Tier 2 inner loop: one configuration and pseudo target fold per PBS subjob.

Each run trains on the other two source species under the registered sampling rule and scores only the pseudo-target validation 
individuals. Test partitions and held-out species are never opened here; configuration selection is a separate pass over 
completed result files.

`--probe` compares independent, complete training budgets to determine the fixed optimiser-step budget before the full grid is
evaluated. Each budget starts from the same seed but has its own warmup/cosine schedule and checkpoint; the probe points are not
successive checkpoints from one model.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import pandas as pd
import torch

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import load_split
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.selection import combine_folds, score_fold, select_configuration
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.train.loop import TrainConfig, train
from animal_movement_fm.train.pool import PreparedPool

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
RESULTS = REPO / "results" / "tier2_inner_loop"
CHECKPOINTS = REPO / "checkpoints" / "tier2_inner_loop"

# Diagnostic runs are written outside RESULTS on purpose. `report_selection` globs RESULTS for *.json, so a diagnostic
# file placed there could be picked up as a scored run; a separate directory makes that structurally impossible rather
# than relying on a name filter.
DIAGNOSTICS = REPO / "results" / "tier2_diagnostics"

# (segments stem, split manifest stem, profile row key)
SOURCES: tuple[tuple[str, str, str], ...] = (
    ("ciconia_ciconia", "stork", "white_stork"),
    ("cervus_elaphus", "cervus_elaphus", "cervus_elaphus"),
    ("mirounga_leonina", "mirounga_leonina", "mirounga_leonina"),
)

# registered grid order; the final tie break uses this exact sequence
GRID: tuple[tuple[int, int, float], ...] = tuple(
    (d_model, layers, lr)
    for d_model in (64, 128, 256)
    for layers in (2, 4)
    for lr in (3e-4, 1e-3)
)
N_FOLDS = len(SOURCES)
PROBE_GRID_INDEX = 6
PROBE_SEED = 0
REGISTERED_PROBE_BUDGETS: tuple[int, ...] = (5_000, 10_000, 20_000, 30_000, 40_000)
PROBE_DESIGN = "independent_complete_schedules"


def decode_index(index: int) -> tuple[int, int]:
    """
    Map a flat PBS array index to `(grid_index, fold_index)`.

    Folds are grouped contiguously, so each block of twelve indices covers the full grid for one pseudo target.
    """
    if not 0 <= index < len(GRID) * N_FOLDS:
        raise SystemExit(f"index {index} is outside 0..{len(GRID) * N_FOLDS - 1}")
    return index % len(GRID), index // len(GRID)


def run_id(grid_index: int, fold_index: int, seed: int) -> str:
    d_model, layers, lr = GRID[grid_index]
    return f"fold{fold_index}_g{grid_index:02d}_d{d_model}_L{layers}_lr{lr:g}_s{seed}"


def build_inner_loop_config(grid_index: int, steps: int, seed: int) -> TrainConfig:
    """
    Build the registered full FM configuration used by every probe and selection run.

    The reconstruction flag is deliberately explicit. Its dataclass default is false because the same training loop also builds
    baseline B, but the inner loop selects the full FM package that is later retrained as Z.
    """
    d_model, layers, lr = GRID[grid_index]
    config = TrainConfig(
        d_model=d_model,
        num_layers=layers,
        lr=lr,
        steps=steps,
        seed=seed,
        use_fm_package=True,
        use_reconstruction_head=True,
    )
    if not config.use_fm_package or not config.use_reconstruction_head:
        raise RuntimeError("the Tier 2 inner loop must train the complete registered FM package")
    return config


def build_diagnostic_config(
    grid_index: int, steps: int, seed: int, *, reconstruction: bool
) -> TrainConfig:
    """
    Build a training only configuration for a numerical diagnostic.

    Identical to `build_inner_loop_config` except that the auxiliary reconstruction head can be switched off, which is the
    point this exercise; it isolates whether a divergence follows the masked objective or the trunk. A configuration
    built here is never scored and never selected from, so it does not carry the full FM guard.
    """
    d_model, layers, lr = GRID[grid_index]
    return TrainConfig(
        d_model=d_model,
        num_layers=layers,
        lr=lr,
        steps=steps,
        seed=seed,
        use_fm_package=True,
        use_reconstruction_head=bool(reconstruction),
    )


def run_diagnostic(
    grid_index: int,
    fold_index: int,
    steps: int,
    seed: int,
    *,
    reconstruction: bool,
    skip_nonfinite: bool = True,
    anomaly: bool = False,
    device: torch.device,
    pool: PreparedPool,
    identifier: str,
    started: float,
) -> None:
    """
    Train one configuration on one fold with gradient norm instrumentation and no scoring.

    Nothing here selects, ranks or compares anything: no pseudo target validation individuals are read, and the output is
    written outside the results directory the selection pass globs. The run exists to name the step at which the pre clip
    gradient norm first becomes non-finite, and the parameter the overflow originated in.
    """
    config = build_diagnostic_config(grid_index, steps, seed, reconstruction=reconstruction)
    tag = "reconON" if reconstruction else "reconOFF"
    # T3 is the default, so only the deviation is named in the filename
    if not skip_nonfinite:
        tag += "_noSkipNF"
    if anomaly:
        tag += "_anomaly"
    print(
        f"diagnostic       : {tag}, {steps:,} steps, no pseudo-target scoring"
        + (", withholding non-finite steps (T3)" if skip_nonfinite else ", T3 disabled")
        + (", autograd anomaly detection on" if anomaly else ""),
        flush=True,
    )

    if anomaly:
        # names the operation that produced a NaN in the backward pass, at the cost of running every backward twice.
        # Diagnostic only: it changes nothing that is computed, only how much bookkeeping surrounds it.
        torch.autograd.set_detect_anomaly(True)

    result = train(
        config,
        pool,
        device=device,
        checkpoint_path=checkpoint_path(
            f"{identifier}_{tag}", steps, probe=False, diagnose=True
        ),
        diagnose=True,
        skip_nonfinite=skip_nonfinite,
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
    )
    # GenAI wrote the rest of the below from its suggestions on 21-08-26. I have checked it below 
    norms = [row["grad_norm"] for row in result.grad_norms if math.isfinite(row["grad_norm"])]
    first = result.first_nonfinite_step
    print(f"\nfinal loss       : {result.final_loss:.5f}")
    print(
        "pre-clip norm    : "
        + (
            f"max {max(norms):.4g} over {len(norms):,} finite steps"
            if norms
            else "no finite step recorded"
        )
    )
    n_nonfinite = sum(1 for row in result.grad_norms if not math.isfinite(row["grad_norm"]))
    print(
        f"non-finite steps : {n_nonfinite:,} of {len(result.grad_norms):,} "
        f"({n_nonfinite / max(1, len(result.grad_norms)):.2%})"
    )
    if skip_nonfinite:
        print(
            f"skipped updates  : {result.skipped_steps:,} "
            f"({result.skipped_steps / max(1, steps):.2%} of the budget)"
        )

    if first is None:
        print("gradient norm    : finite at every step")
    else:
        record = result.nonfinite_steps[0]
        print(f"gradient norm    : first non-finite at step {first:,} of {steps:,}")
        print(
            f"  kind           : {record['kind']}  "
            + (
                "(the norm's sum of squares overflowed; the clip zeroes the step)"
                if record["kind"] == "inf"
                else "(a NaN was already in the gradients; the clip can neither cause nor contain it)"
            )
        )
        print(f"  loss at step   : {record['loss']}  (finite: {record['loss_finite']})")
        print(f"  max |feature|  : {record['max_abs_feature']:.6g}")
        print(f"  max |target|   : {record['max_abs_target']:.6g}")
        print(f"  batch species  : {record['batch_species']}")
        # a short list points at the tensor that overflowed; a list covering the whole model means the forward pass was
        # already poisoned, so the full set is kept in the JSON and only the head of it is printed
        origin = record["origin_parameters"]
        shown = ", ".join(origin[:8]) + (f", ... (+{len(origin) - 8} more)" if len(origin) > 8 else "")
        print(f"  origin tensors : {len(origin)} of {sum(1 for _ in result.model.parameters())}: {shown}")

    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnostic": "tier2_nonfinite_gradient",
        "scored": False,
        "run_id": identifier,
        "grid_index": grid_index,
        "fold_index": fold_index,
        "fold": SOURCES[fold_index][0],
        "reconstruction_head": bool(reconstruction),
        "skip_nonfinite": bool(skip_nonfinite),
        "anomaly_detection": bool(anomaly),
        "skipped_steps": result.skipped_steps,
        "n_nonfinite_steps": n_nonfinite,
        "config": asdict(config),
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "loss_shares": result.loss_shares,
        "grad_norms": result.grad_norms,
        "nonfinite_steps": result.nonfinite_steps,
        "first_nonfinite_step": first,
        "elapsed_s": time.time() - started,
    }
    out = DIAGNOSTICS / f"{identifier}_{tag}_{steps}_diagnostic.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  ({payload['elapsed_s'] / 60:.1f} min)")


def validate_probe_request(grid_index: int, steps: int, seed: int) -> None:
    """
    Refuse any step-budget probe outside the registered Option A design."""
    problems = []
    if grid_index != PROBE_GRID_INDEX:
        problems.append(f"grid index must be {PROBE_GRID_INDEX}, got {grid_index}")
    if steps != REGISTERED_PROBE_BUDGETS[-1]:
        problems.append(f"STEPS must be {REGISTERED_PROBE_BUDGETS[-1]}, got {steps}")
    if seed != PROBE_SEED:
        problems.append(f"seed must be {PROBE_SEED}, got {seed}")
    if problems:
        raise SystemExit("FAIL: unregistered step-budget probe: " + "; ".join(problems))


def checkpoint_path(
    identifier: str, target_steps: int, *, probe: bool, diagnose: bool = False
) -> Path:
    """
    Keep corrected full FM probe, grid, diagnostic and invalid legacy checkpoints disjoint."""
    mode = "diagnose" if diagnose else ("probe" if probe else "grid")
    return CHECKPOINTS / f"{identifier}_registered_fullfm_{mode}_{target_steps}.pt"


def boundary_m(profile_key: str) -> float:
    table = pd.read_csv(REPO / "results" / "regime_profiles.csv").set_index("species")
    if profile_key not in table.index:
        raise SystemExit(f"{profile_key} has no row in results/regime_profiles.csv")
    return float(table.loc[profile_key, "selected_boundary_m"])


def training_pool(fold_index: int, device: torch.device) -> PreparedPool:
    """
    Build the two species training pool for one pseudo target fold.

    The pseudo target is excluded, and the remaining source species use their registered subpool weights.
    """
    held_out = SOURCES[fold_index][0]
    species = {}
    for segments_stem, split_stem, _ in SOURCES:
        if segments_stem == held_out:
            continue
        split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
        species[segments_stem] = SpeciesWindows.build(
            PROCESSED / f"{segments_stem}_segments.parquet",
            split.train,
            context_length=CONTEXT_LENGTH,
            name=segments_stem,
        )
    pool = SourcePool.from_registered_weights(
        species,
        REPO / "results" / "source_pool_weights.json",
        pool=f"sub_pools.without_{held_out}",
    )
    return PreparedPool.from_source_pool(pool, device=device)


def validation_frame(fold_index: int) -> pd.DataFrame:
    """
    Load only the pseudo target's validation individuals. Its training and test partitions are not read here.
    """
    segments_stem, split_stem, _ = SOURCES[fold_index]
    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    frame = load_split_frame(
        PROCESSED / f"{segments_stem}_segments.parquet",
        split.val,
    )
    return frame[["individual_id", "bin_index", "timestamp", "lon", "lat"]].reset_index(drop=True)


def score(model, frame: pd.DataFrame, fold_index: int, device: torch.device) -> dict:
    """
    Score one trained model by travel regime p90 skill against persistence.
    """
    predictor = MovementFMPredictor(model, device=device)
    model_results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
    persistence_results = evaluate(
        Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False
    )
    fold_score = score_fold(
        SOURCES[fold_index][0],
        model_results,
        persistence_results,
        boundary_m=boundary_m(SOURCES[fold_index][2]),
    )
    return asdict(fold_score)

# the CLI guards, run status prints, probe loop and result writing scaffolding below were implemented from my 
# experiment specification by GenAI
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="0..35; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, required=False, help="registered optimiser steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="score at a ladder of step counts to fix the registered step budget",
    )
    parser.add_argument(
        "--select",
        action="store_true",
        help="apply §4 to the 36 result files already on disk and print the winner",
    )
    parser.add_argument(
        "--select-probe",
        action="store_true",
        help="apply amendment T2 to the three corrected probe files and fix the step budget",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="train only, with per-step gradient-norm instrumentation and no pseudo-target scoring",
    )
    parser.add_argument(
        "--no-reconstruction",
        action="store_true",
        help="diagnostic only: disable the auxiliary reconstruction head to isolate the divergence",
    )
    parser.add_argument(
        "--no-skip-nonfinite",
        action="store_true",
        help="diagnostic only: apply every optimiser step, reproducing the pre-T3 divergence",
    )
    parser.add_argument(
        "--anomaly",
        action="store_true",
        help="diagnostic only: enable autograd anomaly detection to name the op producing a NaN",
    )
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.select and args.select_probe:
        raise SystemExit("FAIL: --select and --select-probe are mutually exclusive")
    if args.diagnose and (args.probe or args.select or args.select_probe):
        raise SystemExit(
            "FAIL: --diagnose is a training-only instrument and cannot be combined with --probe, "
            "--select or --select-probe. It produces no skill and selects nothing."
        )
    if args.no_reconstruction and not args.diagnose:
        raise SystemExit(
            "FAIL: --no-reconstruction is only available under --diagnose. The inner loop trains the "
            "complete registered FM package; dropping the auxiliary objective from a scored run would "
            "make it an unregistered configuration."
        )
    if (args.no_skip_nonfinite or args.anomaly) and not args.diagnose:
        raise SystemExit(
            "FAIL: --no-skip-nonfinite and --anomaly are only available under --diagnose. T3 registers that "
            "a non-finite gradient step is withheld, so a scored run may not switch that off."
        )
    if args.select:
        return report_selection()
    if args.select_probe:
        return report_probe_selection()

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX in the environment")
    if args.steps is None:
        raise SystemExit(
            "FAIL: --steps is required. The optimiser step budget is a registered constant "
            "(preregistration_tier2_tuning.md §5) and must be passed explicitly, not defaulted here."
        )

    grid_index, fold_index = decode_index(int(raw))
    if args.probe:
        validate_probe_request(grid_index, args.steps, args.seed)
    identifier = run_id(grid_index, fold_index, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run              : {identifier}")
    print(f"pseudo target    : {SOURCES[fold_index][0]}  (held out of this fold's training pool)")
    print(f"device           : {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu              : {torch.cuda.get_device_name(0)}", flush=True)

    started = time.time()
    pool = training_pool(fold_index, device)
    print(
        f"pool             : {pool.n_windows:,} windows over {list(pool.names)}, "
        f"{pool.bytes_on_device() / 1e6:.0f} MB on device, built in {time.time() - started:.0f}s",
        flush=True,
    )
    print(f"realised shares  : {pool.realised_species_shares(torch.Generator(device=device).manual_seed(0))}")

    if args.diagnose:
        # returns before any validation individual is touched
        return run_diagnostic(
            grid_index,
            fold_index,
            args.steps,
            args.seed,
            reconstruction=not args.no_reconstruction,
            skip_nonfinite=not args.no_skip_nonfinite,
            anomaly=args.anomaly,
            device=device,
            pool=pool,
            identifier=identifier,
            started=started,
        )

    frame = validation_frame(fold_index)
    print(f"validation frame : {len(frame):,} fixes, {frame['individual_id'].nunique()} individuals", flush=True)

    config = build_inner_loop_config(grid_index, args.steps, args.seed)
    budgets = probe_budgets(args.steps) if args.probe else [args.steps]
    if args.probe:
        print(
            "probe budgets    : "
            + ", ".join(f"{budget:,}" for budget in budgets)
            + " independent complete schedules",
            flush=True,
        )

    scores = []
    for target in budgets:
        # a fresh config and target specific checkpoint make each point an independent complete training budget, so requeued jobs may
        # resume that point, but no point inherits model or optimiser state from an earlier budget.
        stage = build_inner_loop_config(grid_index, target, args.seed)
        result = train(
            stage,
            pool,
            device=device,
            checkpoint_path=checkpoint_path(identifier, target, probe=args.probe),
            progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
        )
        # T3 requires the withheld-step count to travel with the result rather than only the log
        entry = {
            "steps": target,
            "final_loss": result.final_loss,
            "skipped_steps": result.skipped_steps,
            "nonfinite_steps": result.nonfinite_steps,
            **score(result.model, frame, fold_index, device),
        }
        scores.append(entry)
        print(
            f"  scored at {target:,} steps: skill {entry['skill']:+.4f}, "
            f"{result.skipped_steps:,} step(s) withheld ({result.skipped_steps / target:.2%})",
            flush=True,
        )

    payload = {
        "run_id": identifier,
        "grid_index": grid_index,
        "fold_index": fold_index,
        "fold": SOURCES[fold_index][0],
        "config": asdict(config),
        "n_parameters": config.build().n_parameters(),
        "scores": scores,
        "elapsed_s": time.time() - started,
        "probe": bool(args.probe),
        "probe_design": PROBE_DESIGN if args.probe else None,
        "probe_budgets": budgets if args.probe else None,
    }
    suffix = "_registered_fullfm_probe" if args.probe else ""
    out = RESULTS / f"{identifier}{suffix}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  ({payload['elapsed_s'] / 60:.1f} min)")


def probe_budgets(max_steps: int) -> list[int]:
    """
    Return the five exact independent budgets in registered amendment T2.
    """
    if max_steps != REGISTERED_PROBE_BUDGETS[-1]:
        raise ValueError(
            f"registered probe maximum is {REGISTERED_PROBE_BUDGETS[-1]}, got {max_steps}"
        )
    return list(REGISTERED_PROBE_BUDGETS)


def rank_probe_budgets(skills_by_budget: dict[int, list[float]]) -> list[dict]:
    """
    Rank budgets by T2; highest mean, highest minimum, then fewer steps.
    """
    if set(skills_by_budget) != set(REGISTERED_PROBE_BUDGETS):
        raise ValueError(
            "probe skill table must contain exactly "
            + ", ".join(str(budget) for budget in REGISTERED_PROBE_BUDGETS)
        )

    candidates = []
    for budget in REGISTERED_PROBE_BUDGETS:
        skills = skills_by_budget[budget]
        if len(skills) != N_FOLDS:
            raise ValueError(f"budget {budget} has {len(skills)}/{N_FOLDS} fold scores")
        if not all(math.isfinite(skill) for skill in skills):
            raise ValueError(f"budget {budget} contains a non-finite skill")
        candidates.append(
            {
                "steps": budget,
                "mean_skill": fmean(skills),
                "min_skill": min(skills),
                "fold_skills": list(skills),
            }
        )

    candidates.sort(key=lambda row: (-row["mean_skill"], -row["min_skill"], row["steps"]))
    return candidates


def _validate_registered_probe_run(run: dict, fold_index: int, path: Path) -> None:
    """
    Fail closed if a probe file is not one of the three corrected full FM runs.
    """
    expected_id = run_id(PROBE_GRID_INDEX, fold_index, PROBE_SEED)
    config = run.get("config", {})
    expected_config = asdict(
        build_inner_loop_config(PROBE_GRID_INDEX, REGISTERED_PROBE_BUDGETS[-1], PROBE_SEED)
    )
    score_steps = [score.get("steps") for score in run.get("scores", [])]
    checks = {
        "run_id": run.get("run_id") == expected_id,
        "grid_index": run.get("grid_index") == PROBE_GRID_INDEX,
        "fold_index": run.get("fold_index") == fold_index,
        "fold": run.get("fold") == SOURCES[fold_index][0],
        "seed": config.get("seed") == PROBE_SEED,
        "maximum steps": config.get("steps") == REGISTERED_PROBE_BUDGETS[-1],
        "FM package": config.get("use_fm_package") is True,
        "reconstruction head": config.get("use_reconstruction_head") is True,
        "complete registered configuration": config == expected_config,
        "probe marker": run.get("probe") is True,
        "probe design": run.get("probe_design") == PROBE_DESIGN,
        "registered budgets": run.get("probe_budgets") == list(REGISTERED_PROBE_BUDGETS),
        "one score per registered budget": score_steps == list(REGISTERED_PROBE_BUDGETS),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"FAIL: {path} is not a corrected registered probe: {', '.join(failed)}")


def report_probe_selection() -> None:
    """
    Apply T2 mechanically once all three corrected probe folds are on disk.
    """
    runs = []
    paths = []
    for fold_index in range(N_FOLDS):
        path = RESULTS / (
            f"{run_id(PROBE_GRID_INDEX, fold_index, PROBE_SEED)}_registered_fullfm_probe.json"
        )
        if not path.exists():
            raise SystemExit(f"FAIL: missing corrected probe fold {fold_index}: {path}")
        run = json.loads(path.read_text())
        _validate_registered_probe_run(run, fold_index, path)
        runs.append(run)
        paths.append(path)

    skills_by_budget = {
        budget: [run["scores"][budget_index]["skill"] for run in runs]
        for budget_index, budget in enumerate(REGISTERED_PROBE_BUDGETS)
    }
    ranked = rank_probe_budgets(skills_by_budget)
    winner = ranked[0]
    rank_by_steps = {candidate["steps"]: rank for rank, candidate in enumerate(ranked, start=1)}

    print(f"\n{'steps':>8} {'stork':>9} {'Cervus':>9} {'SOES':>9} {'mean':>9} {'min':>9} {'rank':>5}")
    for budget in REGISTERED_PROBE_BUDGETS:
        candidate = next(row for row in ranked if row["steps"] == budget)
        print(
            f"{budget:>8,} "
            + " ".join(f"{skill:>+9.4f}" for skill in candidate["fold_skills"])
            + f" {candidate['mean_skill']:>+9.4f} {candidate['min_skill']:>+9.4f} "
            + f"{rank_by_steps[budget]:>5}"
        )

    decision = {
        "decision": "tier2_optimizer_step_budget",
        "rule": ["highest_mean_skill", "highest_minimum_skill", "fewer_steps"],
        "grid_index": PROBE_GRID_INDEX,
        "config": {
            "d_model": GRID[PROBE_GRID_INDEX][0],
            "num_layers": GRID[PROBE_GRID_INDEX][1],
            "lr": GRID[PROBE_GRID_INDEX][2],
            "seed": PROBE_SEED,
        },
        "folds": [source[0] for source in SOURCES],
        "candidates_by_rank": ranked,
        "selected_steps": winner["steps"],
        "source_files": [path.name for path in paths],
    }
    out = RESULTS / "tier2_probe_selection.json"
    out.write_text(json.dumps(decision, indent=2) + "\n")
    print(
        f"\nselected optimiser-step budget: {winner['steps']:,} "
        f"(mean {winner['mean_skill']:+.4f}, minimum {winner['min_skill']:+.4f})"
    )
    print(f"wrote {out}")


def report_selection() -> None:
    """
    Apply the registered selection rule to complete result sets on disk.

    Incomplete configurations are reported and excluded rather than filled or inferred.
    """
    # GenAI also assisted with this aggregation/reporting block 
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json")) if "_probe" not in p.name]
    by_config: dict[int, list] = {}
    for run in runs:
        by_config.setdefault(run["grid_index"], []).append(run)

    complete, incomplete = [], []
    for grid_index, group in sorted(by_config.items()):
        if len(group) != N_FOLDS:
            incomplete.append((grid_index, len(group)))
            continue
        complete.append(
            {
                "grid_index": grid_index,
                "n_parameters": group[0]["n_parameters"],
                "scores": [_as_fold_score(r) for r in group],
            }
        )

    for grid_index, n in incomplete:
        print(f"grid index {grid_index} has {n}/{N_FOLDS} folds and is excluded from selection")
    if not complete:
        raise SystemExit("no configuration has all three folds; nothing to select")

    print(f"\n{'grid':>4} {'d_model':>7} {'L':>2} {'lr':>7} {'params':>10} {'mean':>8} {'min':>8}")
    for candidate in sorted(complete, key=lambda c: -combine_folds(c["scores"])):
        d_model, layers, lr = GRID[candidate["grid_index"]]
        print(
            f"{candidate['grid_index']:>4} {d_model:>7} {layers:>2} {lr:>7.0e} "
            f"{candidate['n_parameters']:>10,} {combine_folds(candidate['scores']):>+8.4f} "
            f"{min(s.skill for s in candidate['scores']):>+8.4f}"
        )

    winner = select_configuration(complete)
    d_model, layers, lr = GRID[winner["grid_index"]]
    print(
        f"\nselected: grid index {winner['grid_index']}, d_model={d_model}, layers={layers}, lr={lr:g}, "
        f"mean skill {winner['mean_skill']:+.4f}, min {winner['min_skill']:+.4f}"
    )
    if winner["grid_index"] == len(GRID) - 1:
        print(
            "\nNOTE: this is the maximum of the capacity axis. Report it as a boundary selection -- "
            "the budget did not bracket capacity from above, so 'best available' and 'best' are not "
            "distinguishable here."
        )


def _as_fold_score(run: dict):
    from animal_movement_fm.eval.selection import FoldScore

    final = run["scores"][-1]
    return FoldScore(
        fold=run["fold"],
        skill=final["skill"],
        model_p90_m=final["model_p90_m"],
        persistence_p90_m=final["persistence_p90_m"],
        n_travel_windows=final["n_travel_windows"],
        n_windows=final["n_windows"],
    )


if __name__ == "__main__":
    main()
