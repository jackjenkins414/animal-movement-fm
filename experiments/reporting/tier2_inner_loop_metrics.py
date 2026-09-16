# GenAI reference: GenAI implemented this reporting script from my spec; it re scores saved grid checkpoints on
# the same pseudo target validation individuals, reports all metrics/strata, reuses the sealed evaluation summary,
# and writes outside the report_selection glob; descriptive only and reopens nothing. I reviewed the code, and 
# checked the numbers against the inner loop result files.
"""
Report the full metric set for inner-loop grid checkpoints, not just the selection metric.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tier2_inner_loop import ( # type: ignore
    GRID,
    REGISTERED_SELECTION,
    SOURCES,
    boundary_m,
    checkpoint_path,
    run_id,
    validation_frame,
)
from tier2_sealed_eval import COMPARATOR_ORDER, STATISTICS, _m, summarise

from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.models.averaging import DisplacementAveraging
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.train.loop import TrainConfig

REPO = Path(__file__).resolve().parents[2]
DIAGNOSTICS = REPO / "results" / "tier2_diagnostics"


def load_grid_checkpoint(grid_index: int, fold_index: int, steps: int, seed: int):
    """
    Rebuild one trained grid model from its saved checkpoint.

    The configuration travels with the checkpoint, so the model is reconstructed from what was
    trained rather than from what this script assumes the grid contains.
    """
    path = checkpoint_path(run_id(grid_index, fold_index, seed), steps, probe=False)
    if not path.exists():
        return None, path
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainConfig(**state["config"]).build()
    model.load_state_dict(state["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit(f"FAIL: {path} holds non-finite weights; that run diverged")
    return model.eval(), path


def score_fold_fully(grid_index: int, fold_index: int, steps: int, seed: int, device) -> dict | None:
    """
    Score the FM, persistence and constant velocity on one pseudo-target's validation individuals.

    V is included because A11 registers it as a reported comparator; it costs one extra pass and
    gives the FM a second reference point that is not "predict no movement".
    """
    model, path = load_grid_checkpoint(grid_index, fold_index, steps, seed)
    if model is None:
        print(f"  fold {fold_index}: no checkpoint at {path.name}, skipped", flush=True)
        return None

    frame = validation_frame(fold_index)
    boundary = boundary_m(SOURCES[fold_index][2])
    persistence = evaluate(Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False)
    velocity = evaluate(
        DisplacementAveraging(k=1), frame, context_length=CONTEXT_LENGTH, show_progress=False
    )
    forecast = evaluate_batched(
        MovementFMPredictor(model, device=device), frame, context_length=CONTEXT_LENGTH
    )

    return {
        "fold_index": fold_index,
        "pseudo_target": SOURCES[fold_index][0],
        "regime_boundary_m": boundary,
        "n_individuals": int(frame["individual_id"].nunique()),
        "checkpoint": path.name,
        "comparators": {
            name: summarise(results, persistence, boundary)
            for name, results in (("Z", forecast), ("P", persistence), ("V", velocity))
        },
    }


def write_markdown(payload: dict, path: Path) -> None:
    d_model, layers, lr = GRID[payload["grid_index"]]
    lines = [
        "# Inner-loop grid metrics beyond the selection statistic",
        "",
        f"Grid index {payload['grid_index']} — d_model {d_model}, {layers} layers, lr {lr:g}, "
        f"{payload['steps']:,} steps, seed {payload['seed']}.",
        "",
        "Source-species **validation** individuals, one seed per fold. This is the inner loop, not "
        "Tier 2's claim: two sources predicting a third *source* species. Nothing sealed is involved "
        "and nothing here re-opens the selection, which the registered rule already decided.",
        "",
        "Mean, p50, p90 and p99 are individual-weighted (A5) at the evaluation floor F = 25. "
        "`Z` is the FM, `P` persistence, `V` constant velocity.",
        "",
    ]
    for fold in payload["folds"]:
        lines += [
            f"## {fold['pseudo_target']}",
            "",
            f"Regime boundary {fold['regime_boundary_m']:.1f} m; "
            f"{fold['n_individuals']} validation individuals.",
            "",
        ]
        for stratum, title in (
            ("all", "All windows"),
            ("travel", "Travel regime"),
            ("low_displacement", "Low displacement"),
        ):
            n = fold["comparators"]["P"][stratum].get("n_windows")
            lines += [
                f"**{title}**" + (f" — {n:,} windows" if n else ""),
                "",
                "| model | mean | p50 | p90 | p99 | skill (p90) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            reference = fold["comparators"]["P"][stratum].get("p90_m")
            for name in COMPARATOR_ORDER:
                if name not in fold["comparators"]:
                    continue
                values = fold["comparators"][name][stratum]
                skill = (
                    None
                    if name == "P" or not reference or not values.get("p90_m")
                    else 1.0 - values["p90_m"] / reference
                )
                lines.append(
                    f"| {name} | {_m(values.get('mean_m_individual_weighted'))} | "
                    f"{_m(values.get('p50_m'))} | {_m(values.get('p90_m'))} | "
                    f"{_m(values.get('p99_m_individual_weighted'))} | "
                    + (f"{skill:+.4f} |" if skill is not None else "— |")
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-index", type=int, default=None, help="default: the selected cell")
    parser.add_argument("--all", action="store_true", help="score all twelve configurations")
    parser.add_argument("--steps", type=int, default=20_000, help="must match the trained budget")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.all:
        indices = list(range(len(GRID)))
    elif args.grid_index is not None:
        indices = [args.grid_index]
    else:
        decision = REGISTERED_SELECTION
        if not decision.exists():
            raise SystemExit(f"FAIL: {decision} does not exist; pass --grid-index")
        indices = [int(json.loads(decision.read_text())["selected_grid_index"])]

    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device           : {device}", flush=True)

    for grid_index in indices:
        d_model, layers, lr = GRID[grid_index]
        print(f"\ngrid index {grid_index}: d_model {d_model}, {layers} layers, lr {lr:g}", flush=True)
        started = time.time()
        folds = [
            scored
            for fold_index in range(len(SOURCES))
            if (scored := score_fold_fully(grid_index, fold_index, args.steps, args.seed, device))
        ]
        if not folds:
            print("  no checkpoints for this configuration, nothing written")
            continue

        payload = {
            "diagnostic": "tier2_inner_loop_full_metrics",
            "scored_partition": "source_species_validation",
            "opened_test_partition": False,
            "opened_held_out_species": False,
            "grid_index": grid_index,
            "config": {"d_model": d_model, "num_layers": layers, "lr": lr},
            "steps": args.steps,
            "seed": args.seed,
            "statistics": list(STATISTICS),
            "folds": folds,
            "elapsed_s": time.time() - started,
        }
        out = DIAGNOSTICS / f"inner_loop_metrics_g{grid_index:02d}.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        write_markdown(payload, DIAGNOSTICS / f"inner_loop_metrics_g{grid_index:02d}.md")
        print(f"  wrote {out.name} and {out.stem}.md  ({payload['elapsed_s'] / 60:.1f} min)")

        for fold in folds:
            travel = fold["comparators"]["Z"]["travel"]
            allw = fold["comparators"]["Z"]["all"]
            p_travel = fold["comparators"]["P"]["travel"]
            p_all = fold["comparators"]["P"]["all"]
            print(
                f"  {fold['pseudo_target']:<20} "
                f"all p50 {allw.get('p50_m', float('nan')):>9,.0f} vs P {p_all.get('p50_m', float('nan')):>9,.0f}   "
                f"travel p90 {travel.get('p90_m', float('nan')):>10,.0f} vs P {p_travel.get('p90_m', float('nan')):>10,.0f}"
            )


if __name__ == "__main__":
    main()
