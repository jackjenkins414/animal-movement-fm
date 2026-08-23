# GenAI reference: assisted with some of the surrounding PBS/CLI orchestration, defensive checks progress output where 
# I outlined the arm definitions, training pools and S/F selection rules etc. 
"""
Train the registered Tier 2 comparator arms, one model per PBS subjob.

Z and B train on the full three species source pool; S trains on the held out species' own training individuals; Za trains 
on the source sub pool with the matching domain removed. P and V require no training and are handled by sealed evaluation.

This script never opens a test partition. Z, B and Za are also prevented from opening held-out species data. S may use 
held-out validation data only for its registered checkpoint-selection rule; the other trained arms report their final 
checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tier2_inner_loop import GRID, SOURCES  # single source of truth for the registered grid order

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import load_split
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.selection import score_fold
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.train.loop import TrainConfig, train
from animal_movement_fm.train.pool import PreparedPool

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
WEIGHTS = REPO / "results" / "source_pool_weights.json"
INNER_LOOP_RESULTS = REPO / "results" / "tier2_inner_loop"
RESULTS = REPO / "results" / "tier2_final"
CHECKPOINTS = REPO / "checkpoints" / "tier2_final"

# Tier 1 multi-seed sweep  reused unchanged across all trained arms
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# A5 minimum valid windows per individual for evaluation.
EVALUATION_FLOOR = 25

# S (and later F) selects the checkpoint with minimum validation travel regime p90; Z, B and Za have no validation privilege
# and report the final registered checkpoint
SELECTION_RULE = "argmin_validation_travel_p90"
SELECTION_EVERY = 1_000

# (segments stem, split manifest stem, regime_profiles.csv row key, domain, matching source)
HELD_OUT: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "larus_fuscus",
        "larus_fuscus",
        "larus_fuscus",
        "flying",
        "ciconia_ciconia",
    ),
    (
        "canis_lupus",
        "canis_lupus",
        "canis_lupus",
        "terrestrial",
        "cervus_elaphus",
    ),
    (
        "mirounga_angustirostris",
        "mirounga_angustirostris",
        "mirounga_angustirostris",
        "marine",
        "mirounga_leonina",
    ),
)

SOURCE_STEMS = frozenset(source[0] for source in SOURCES)
HELD_OUT_STEMS = frozenset(species[0] for species in HELD_OUT)


@dataclass(frozen=True)
class Run:
    """
    Describe one trained arm/species/seed combination.

    `species` is empty for Z and B. For S and Za it identifies the held-out species the run is attached to; Za still trains 
    only on source species.
    """

    arm: str
    species: str
    seed: int
    deferred: bool = False

    @property
    def run_id(self) -> str:
        return (
            f"{self.arm}_{self.species}_s{self.seed}"
            if self.species
            else f"{self.arm}_s{self.seed}"
        )

    @property
    def uses_validation_selection(self) -> bool:
        """
        Return whether this arm may use validation data for checkpoint selection.
        """
        return self.arm == "S"


def build_plan() -> list[Run]:
    """
    Return the fixed PBS run plan.

    List position is the array index, so the order is load-bearing. Z, B and S form the critical path; deferred Za runs 
    follow them.
    """
    plan = [Run("Z", "", seed) for seed in SEEDS]
    plan += [Run("B", "", seed) for seed in SEEDS]
    for species, *_ in HELD_OUT:
        plan += [Run("S", species, seed) for seed in SEEDS]
    for species, *_ in HELD_OUT:
        plan += [Run("Za", species, seed, deferred=True) for seed in SEEDS]
    return plan


PLAN = build_plan()
N_CRITICAL_PATH = sum(1 for run in PLAN if not run.deferred)


def held_out_entry(species: str) -> tuple[str, str, str, str, str]:
    for entry in HELD_OUT:
        if entry[0] == species:
            return entry
    raise SystemExit(f"{species} is not a registered held-out species")


def _assert_source_only(stems) -> None:
    """
    Fail if a Z, B or Za training pool contains anything outside the registered sources.

    This guard enforces the zero-shot data boundary directly at pool construction rather than relying on later review.
    """
    intruders = sorted(set(stems) & HELD_OUT_STEMS)
    if intruders:
        raise SystemExit(
            f"FAIL: {', '.join(intruders)} is a held-out species and cannot enter a Z, B or Za "
            "training pool. Z is zero-shot by construction; this is not a recoverable condition."
        )
    unknown = sorted(set(stems) - SOURCE_STEMS)
    if unknown:
        raise SystemExit(f"FAIL: {', '.join(unknown)} is not a registered source species")


def _load_species_windows(
    stem: str,
    split_stem: str,
    partition: str,
) -> SpeciesWindows:
    """
    Build one species' train or validation window index.

    Test partitions are refused unconditionally so sealed evaluation cannot be bypassed through this training script.
    """
    if partition == "test":
        raise SystemExit(
            "FAIL: this script never opens a test partition. Sealed evaluation happens in a "
            "separate script, after the freeze is committed."
        )
    if partition not in {"train", "val"}:
        raise SystemExit(f"FAIL: unknown partition {partition!r}")

    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    return SpeciesWindows.build(
        PROCESSED / f"{stem}_segments.parquet",
        getattr(split, partition),
        context_length=CONTEXT_LENGTH,
        name=stem,
    )


def training_pool(run: Run, device: torch.device) -> tuple[PreparedPool, str]:
    """
    Build runs training pool and return its registered weight node name.

    Z/B use the full source pool, Za uses the registered "leave-one-domain-source-out" sub-pool, and S uses only its held-out species' 
    training individuals with weight 1.0.
    """
    if run.arm == "S":
        stem, split_stem, _, _, _ = held_out_entry(run.species)
        windows = _load_species_windows(stem, split_stem, "train")
        pool = SourcePool({stem: windows}, {stem: 1.0})
        return PreparedPool.from_source_pool(pool, device=device), f"specialist:{stem}"

    if run.arm == "Za":
        _, _, _, _, matching_source = held_out_entry(run.species)
        node = f"sub_pools.without_{matching_source}"
        wanted = [s for s in SOURCES if s[0] != matching_source]
    else:
        node = "source_pool"
        wanted = list(SOURCES)

    _assert_source_only([s[0] for s in wanted])
    species = {
        stem: _load_species_windows(stem, split_stem, "train")
        for stem, split_stem, *_ in wanted
    }
    pool = SourcePool.from_registered_weights(species, WEIGHTS, pool=node)
    return PreparedPool.from_source_pool(pool, device=device), node


def build_config(run: Run, grid_index: int, steps: int) -> TrainConfig:
    """
    Build one run's configuration at the selected grid cell.

    Z, S and Za use the full FM package. B uses the same trunk with conditioning and reconstruction disabled; all other 
    training settings remain shared.
    """
    d_model, layers, lr = GRID[grid_index]
    is_baseline = run.arm == "B"
    config = TrainConfig(
        d_model=d_model,
        num_layers=layers,
        lr=lr,
        steps=steps,
        seed=run.seed,
        use_fm_package=not is_baseline,
        use_reconstruction_head=not is_baseline,
    )
    if is_baseline:
        if config.use_fm_package or config.use_reconstruction_head:
            raise RuntimeError("B must be the trunk with the FM package stripped")
    elif not (config.use_fm_package and config.use_reconstruction_head):
        raise RuntimeError(f"{run.arm} must train the complete registered FM package")
    return config


# GenAI reference: assisted with the decision file/CLI override plumbing and defensive checks in this function below
def selected_grid_index(explicit: int | None) -> tuple[int, str]:
    """
    Return the selected grid index and record where that choice came from.

    The inner loop decision file is authoritative. An explicit CLI override is allowed for deliberate reruns and is labelled 
    as such in the result record.
    """
    if explicit is not None:
        if not 0 <= explicit < len(GRID):
            raise SystemExit(f"grid index {explicit} is outside 0..{len(GRID) - 1}")
        print(
            f"WARNING: grid index {explicit} was passed on the command line rather than read from "
            "the selection decision file. It is recorded as `explicit_cli` in every result written "
            "by this run.",
            flush=True,
        )
        return explicit, "explicit_cli"

    path = INNER_LOOP_RESULTS / "tier2_selection.json"
    if not path.exists():
        raise SystemExit(
            f"FAIL: {path} does not exist, so no configuration has been selected by the registered "
            "rule. Run `tier2_inner_loop.py --select` over the 36 grid results first, or pass "
            "--grid-index explicitly if you are deliberately reproducing one cell."
        )
    decision = json.loads(path.read_text())
    index = int(decision["selected_grid_index"])
    if not 0 <= index < len(GRID):
        raise SystemExit(f"{path} names grid index {index}, which is outside the registered grid")
    return index, path.name


# validation scoring used only by arms with registered validation access;
def validation_frame(species: str) -> pd.DataFrame:
    """
    Load one held-out species' validation individuals without touching test data.
    """
    stem, split_stem, _, _, _ = held_out_entry(species)
    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    frame = load_split_frame(
        PROCESSED / f"{stem}_segments.parquet",
        split.val,
    )
    return frame[
        ["individual_id", "bin_index", "timestamp", "lon", "lat"]
    ].reset_index(drop=True)


def boundary_m(profile_key: str) -> float:
    table = pd.read_csv(REPO / "results" / "regime_profiles.csv").set_index("species")
    if profile_key not in table.index:
        raise SystemExit(f"{profile_key} has no row in results/regime_profiles.csv")
    return float(table.loc[profile_key, "selected_boundary_m"])


def apply_evaluation_floor(
    model_results: pd.DataFrame,
    persistence_results: pd.DataFrame,
    floor: int = EVALUATION_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Apply the A5 evaluation floor to model and persistence results identically.

    The same individual mask is applied to both aligned frames so the row correspondence required by `score_fold` is 
    preserved.
    """
    counts = persistence_results.groupby("individual_id").size()
    keep = set(counts[counts >= floor].index)
    mask = persistence_results["individual_id"].isin(keep).to_numpy()
    return (
        model_results.loc[mask].reset_index(drop=True),
        persistence_results.loc[mask].reset_index(drop=True),
        int(counts.size),
        len(keep),
    )


def score_against_persistence(
    model,
    frame: pd.DataFrame,
    persistence_results: pd.DataFrame,
    profile_key: str,
    label: str,
    device: torch.device,
) -> dict:
    """
    Score one model by individual weighted travel regime p90 skill against persistence.
    """
    predictor = MovementFMPredictor(model, device=device)
    model_results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
    model_results, persistence_floored, n_all, n_kept = apply_evaluation_floor(
        model_results,
        persistence_results,
    )
    fold_score = score_fold(
        label,
        model_results,
        persistence_floored,
        boundary_m=boundary_m(profile_key),
    )
    payload = asdict(fold_score)
    payload["n_individuals_scored"] = n_kept
    payload["n_individuals_before_floor"] = n_all
    payload["evaluation_floor"] = EVALUATION_FLOOR
    return payload

# GenAI reference: helped with the CLI/PBS orchestration, status output, checkpoint/result plumbing and defensive guards below; 
# the arm definitions and experiment rules were my own specification; whole file I have read over and also had GenAI check
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="run-plan index; defaults to $PBS_ARRAY_INDEX",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="registered optimiser steps",
    )
    parser.add_argument(
        "--grid-index",
        type=int,
        default=None,
        help="override the selected configuration; recorded as unregistered in the result file",
    )
    parser.add_argument("--list", action="store_true", help="print the run plan and exit")
    args = parser.parse_args()

    if args.list:
        return print_plan()

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX in the environment")
    index = int(raw)
    if not 0 <= index < len(PLAN):
        raise SystemExit(f"index {index} is outside 0..{len(PLAN) - 1}; see --list")
    if args.steps is None:
        raise SystemExit(
            "FAIL: --steps is required. The optimiser step budget is a registered constant "
            "(preregistration_tier2_tuning.md §5) and must be passed explicitly, not defaulted here."
        )

    run = PLAN[index]
    grid_index, configuration_source = selected_grid_index(args.grid_index)
    d_model, layers, lr = GRID[grid_index]

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1})")
    print(
        f"arm              : {run.arm}"
        + (f", attached to {run.species}" if run.species else "")
    )
    if run.deferred:
        print("                   registered but deferred; this arm is not on the pre-meeting path")
    print(
        f"configuration    : grid index {grid_index} "
        f"(d_model={d_model}, layers={layers}, lr={lr:g})"
    )
    print(f"                   from {configuration_source}")
    print(f"device           : {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu              : {torch.cuda.get_device_name(0)}", flush=True)

    started = time.time()
    pool, pool_node = training_pool(run, device)
    print(
        f"pool             : {pool.n_windows:,} windows over {list(pool.names)} [{pool_node}], "
        f"{pool.bytes_on_device() / 1e6:.0f} MB on device, built in {time.time() - started:.0f}s",
        flush=True,
    )

    config = build_config(run, grid_index, args.steps)
    checkpoint = CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{args.steps}.pt"

    monitor, selection = _make_monitor(run, device, grid_index, args.steps)
    result = train(
        config,
        pool,
        device=device,
        checkpoint_path=checkpoint,
        progress=lambda step, loss: print(
            f"  step {step:>7,}  loss {loss:.5f}",
            flush=True,
        ),
        **monitor,
    )

    print(f"\nfinal loss       : {result.final_loss:.5f}")
    # T3: keep the withheld-step count in both the log and result record.
    print(
        f"withheld steps   : {result.skipped_steps:,} of {args.steps:,} "
        f"({result.skipped_steps / args.steps:.2%}) -- T3"
    )

    reported_model, selection_payload = _resolve_reported_model(run, result, selection)

    # Sealed evaluation reads this explicit reported checkpoint rather than reconstructing a filename.
    reported_checkpoint = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {
            "run_id": run.run_id,
            "config": asdict(config),
            "grid_index": grid_index,
            "selection": {k: v for k, v in selection_payload.items() if k != "history"},
            "model": reported_model.state_dict(),
        },
        reported_checkpoint,
    )

    payload = {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "species": run.species or None,
        "seed": run.seed,
        "deferred_arm": run.deferred,
        "grid_index": grid_index,
        "configuration_source": configuration_source,
        "config": asdict(config),
        "n_parameters": reported_model.n_parameters(),
        "training_pool": pool_node,
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "loss_shares": result.loss_shares,
        # T3 diagnostics carried into the freeze.
        "skipped_steps": result.skipped_steps,
        "nonfinite_steps": result.nonfinite_steps,
        "checkpoint_selection": selection_payload,
        "reported_checkpoint": reported_checkpoint.name,
        "opened_test_partition": False,
        "elapsed_s": time.time() - started,
    }
    out = RESULTS / f"{run.run_id}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  ({payload['elapsed_s'] / 60:.1f} min)")

# GenAI reference: assisted with the resumable monitor/history/checkpoint plumbing implementing that rule and I did the S 
# selects by validation travel regime p90 at the fixed; and also added the 2 functions below 
def _make_monitor(
    run: Run,
    device: torch.device,
    grid_index: int,
    steps: int,
) -> tuple[dict, dict]:
    """
    Build the periodic validation hook for arms allowed to select checkpoints.

    Z, B and Za receive no hook and never open validation data. For S, validation history is persisted after each evaluation 
    so a requeued PBS job can recover the same checkpoint selection state.
    """
    selection: dict = {"rule": None, "history": [], "best": None}
    if not run.uses_validation_selection:
        selection["rule"] = "final_checkpoint"
        return {}, selection

    stem, _, profile_key, _, _ = held_out_entry(run.species)
    identity = {
        "run_id": run.run_id,
        "grid_index": grid_index,
        "steps": steps,
        "rule": SELECTION_RULE,
        "every": SELECTION_EVERY,
        "floor": EVALUATION_FLOOR,
    }
    history_path = CHECKPOINTS / f"{run.run_id}_selection.json"
    if history_path.exists():
        stored = json.loads(history_path.read_text())
        if stored.get("identity") == identity:
            selection["history"] = stored["history"]
            if selection["history"]:
                selection["best"] = min(
                    selection["history"],
                    key=lambda row: row["model_p90_m"],
                )
                print(
                    f"resumed          : {len(selection['history'])} validation evaluations "
                    f"recovered, best so far step {selection['best']['step']:,}",
                    flush=True,
                )
        else:
            print(
                f"discarding {history_path.name}: it belongs to a different configuration",
                flush=True,
            )

    frame = validation_frame(run.species)
    print(
        f"validation frame : {len(frame):,} fixes, {frame['individual_id'].nunique()} individuals "
        f"({stem}); used for checkpoint selection only",
        flush=True,
    )
    # Persistence is deterministic; score it once and reuse it for both comparison and travel labels.
    persistence_results = evaluate(
        Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False
    )
    selection["rule"] = SELECTION_RULE
    selection["every"] = SELECTION_EVERY
    best = CHECKPOINTS / f"{run.run_id}_best.pt"

    def monitor(step: int, model) -> None:
        entry = score_against_persistence(
            model,
            frame,
            persistence_results,
            profile_key,
            stem,
            device,
        )
        entry["step"] = int(step)
        selection["history"].append(entry)
        incumbent = selection["best"]
        # Persistence is fixed across the run, so ranking by model p90 and skill is equivalent.
        if incumbent is None or entry["model_p90_m"] < incumbent["model_p90_m"]:
            selection["best"] = entry
            best.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": int(step),
                    "model": copy.deepcopy(model.state_dict()),
                },
                best,
            )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps({"identity": identity, "history": selection["history"]}, indent=2) + "\n"
        )
        print(
            f"  step {step:>7,}  val travel p90 {entry['model_p90_m']:>10,.0f} m  "
            f"skill {entry['skill']:+.4f}"
            + ("  <- best" if selection["best"] is entry else ""),
            flush=True,
        )

    return {"monitor": monitor, "monitor_every": SELECTION_EVERY}, selection


def _resolve_reported_model(run: Run, result, selection: dict):
    """
    Return the model reported by this run and the checkpoint-selection record.

    Z, B and Za report the final checkpoint. S reports the checkpoint with minimum validation travel regime p90.
    """
    if not run.uses_validation_selection:
        return result.model, {"rule": "final_checkpoint", "step": result.config.steps}

    best = selection.get("best")
    if best is None:
        raise SystemExit(
            "FAIL: S completed without a single validation evaluation, so no checkpoint could be "
            "selected. Check that the step budget is at least SELECTION_EVERY."
        )
    state = torch.load(
        CHECKPOINTS / f"{run.run_id}_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    result.model.load_state_dict(state["model"])
    print(
        f"\nselected         : step {best['step']:,} of {result.config.steps:,}, "
        f"validation travel p90 {best['model_p90_m']:,.0f} m"
    )
    return result.model, {
        "rule": SELECTION_RULE,
        "every": SELECTION_EVERY,
        "evaluation_floor": EVALUATION_FLOOR,
        "selected_step": best["step"],
        "selected_validation_travel_p90_m": best["model_p90_m"],
        "history": selection["history"],
    }


def print_plan() -> None:
    # GenAI reference: did all of this; I checked it 
    print(f"{'idx':>4}  {'arm':<3} {'species':<24} {'seed':>4}  path")
    for index, run in enumerate(PLAN):
        print(
            f"{index:>4}  {run.arm:<3} {run.species or '-':<24} {run.seed:>4}  "
            + ("deferred" if run.deferred else "critical")
        )
    print(
        f"\n{N_CRITICAL_PATH} critical-path runs (0-{N_CRITICAL_PATH - 1}), "
        f"{len(PLAN) - N_CRITICAL_PATH} deferred "
        f"({N_CRITICAL_PATH}-{len(PLAN) - 1})."
    )
    print("P and V are untrained and are produced by the sealed-evaluation script.")


if __name__ == "__main__":
    main()
