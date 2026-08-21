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

`--probe` scores one run at several step counts to determine the fixed optimiser-step budget before the full grid is evaluated.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import id_key, load_split
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
    wanted = {id_key(i) for i in split.val}

    import duckdb

    frame = duckdb.connect().execute(
        f"""
        SELECT individual_id, bin_index, timestamp, lon, lat
        FROM '{PROCESSED / f"{segments_stem}_segments.parquet"}'
        ORDER BY individual_id, bin_index
        """
    ).df()
    keep = np.fromiter((id_key(v) in wanted for v in frame["individual_id"]), bool, len(frame))
    return frame.loc[keep].reset_index(drop=True)


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
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.select:
        return report_selection()

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX in the environment")
    if args.steps is None:
        raise SystemExit(
            "FAIL: --steps is required. The optimiser step budget is a registered constant "
            "(preregistration_tier2_tuning.md §5) and must be passed explicitly, not defaulted here."
        )

    grid_index, fold_index = decode_index(int(raw))
    d_model, layers, lr = GRID[grid_index]
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

    frame = validation_frame(fold_index)
    print(f"validation frame : {len(frame):,} fixes, {frame['individual_id'].nunique()} individuals", flush=True)

    config = TrainConfig(
        d_model=d_model, num_layers=layers, lr=lr, steps=args.steps, seed=args.seed
    )
    ladder = _probe_ladder(args.steps) if args.probe else []
    scores, reached = [], 0

    for target in [*ladder, args.steps]:
        if target <= reached:
            continue
        stage = TrainConfig(**{**asdict(config), "steps": target})
        result = train(
            stage,
            pool,
            device=device,
            checkpoint_path=CHECKPOINTS / f"{identifier}_{target}.pt",
            progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
        )
        reached = target
        entry = {"steps": target, "final_loss": result.final_loss, **score(result.model, frame, fold_index, device)}
        scores.append(entry)
        print(f"  scored at {target:,} steps: skill {entry['skill']:+.4f}", flush=True)

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
    }
    out = RESULTS / f"{identifier}{'_probe' if args.probe else ''}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  ({payload['elapsed_s'] / 60:.1f} min)")


def _probe_ladder(steps: int) -> list[int]:
    """Return the intermediate probe points used to inspect convergence over the step budget."""
    return [int(steps * f) for f in (0.125, 0.25, 0.5, 0.75)]


def report_selection() -> None:
    """Apply the registered selection rule to complete result sets on disk.

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
