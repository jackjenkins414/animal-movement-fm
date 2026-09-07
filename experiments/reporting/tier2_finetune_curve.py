# GenAI reference: I asked GenAI to implement reporting script from my outline; being to build the travel p90 curve over 
# k, uses Z as k = 0 and S, report the median and range across the low k draws, reuse the sealed evaluation's `summarise` 
# function and prediction files; it also added the F100 vs S comparison and caveat section. I have checked the full file.
"""
Report the registered fine-tuning curve (F) on the sealed test partition.

Under our tier 2 prereg sec 3, we report F as travel-regime p90 against k rather than as a single comparator. Z is the k = 
0 point, S is the asymptote, and k = 1% and 10% are summarised by the median and range across three independent draws.

We handle F  separately from `tier2_sealed_eval.py` because its runs are indexed by `(k, draw)` rather than `(arm, species, 
seed)`. But the registered statistic is still computed with that script's `summarise` function so the reporting logic cant 
drift.

Note; `--predict` opens the sealed test partitions after the freeze check and writes one prediction file per F run. `--report` 
builds the curve from those files and the existing P, Z, B and S predictions. F itself was trained only on held-out training 
individuals.
Note2; this main docstring and GenAI comment below are mine; the below is GenAI generated but validated.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))

import tier2_final_train as T  # noqa: E402
import tier2_sealed_eval as SE  # noqa: E402
from tier2_persistence_distance import geodesic_m  # noqa: E402

from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402
from animal_movement_fm.train.finetune import REGISTERED_FRACTIONS_PERCENT  # noqa: E402

OUT = SE.SEALED

# Registered reporting statistic for the F curve.
STRATUM = "travel"
STATISTIC = "p90_m"


def finetune_runs() -> list[dict]:
    """
    Return completed F run payloads ordered by species, k and draw.
    """
    runs = []
    for path in sorted(SE.FINAL.glob("F_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("finetune") is None:
            continue
        runs.append(payload)
    order = {species: index for index, (species, *_) in enumerate(T.HELD_OUT)}
    return sorted(
        runs,
        key=lambda p: (order[p["species"]], p["finetune"]["k_percent"], p["finetune"]["draw"]),
    )


def prediction_path(species: str, payload: dict) -> Path:
    """
    Return the prediction path for one F run.
    """
    finetune = payload["finetune"]
    return SE.PREDICTIONS / (
        f"{species}__F_k{finetune['k_percent']:03d}_d{finetune['draw']}.parquet"
    )


def run_predictions(device: torch.device, overwrite: bool = False) -> dict:
    """
    Score each F run on its species' sealed test partition.
    """
    attestation = SE.assert_freeze_committed()
    SE.PREDICTIONS.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[str]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for payload in finetune_runs():
        species = payload["species"]
        if species not in frames:
            frames[species] = SE.test_frame(species)
            print(
                f"\n{species}: {len(frames[species]):,} test fixes, "
                f"{frames[species]['individual_id'].nunique()} individuals",
                flush=True,
            )
        out = prediction_path(species, payload)
        written.setdefault(species, []).append(out.name)
        if out.exists() and not overwrite:
            print(f"  {out.name}  (exists, skipped)")
            continue

        started = time.time()
        model = SE.load_trained(payload["run_id"])
        if model is None:
            print(f"  {payload['run_id']}: no reported checkpoint, skipped")
            written[species].remove(out.name)
            continue
        results = evaluate_batched(
            MovementFMPredictor(model, device=device),
            frames[species],
            context_length=CONTEXT_LENGTH,
        )
        results.to_parquet(out, index=False)
        print(f"  {out.name}  {len(results):,} windows  ({time.time() - started:.0f}s)", flush=True)

    manifest = {
        "attestation": attestation,
        "arm": "F",
        "reported_by": "tier2_finetune_curve.py",
        "species": written,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier2_finetune_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_predictions(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def travel_p90(results: pd.DataFrame, persistence: pd.DataFrame, boundary: float) -> float:
    """Return the registered travel-regime p90 using the sealed evaluator's summary function."""
    return SE.summarise(results, persistence, boundary)[STRATUM].get(STATISTIC, float("nan"))


def movement(results: pd.DataFrame, persistence: pd.DataFrame) -> float:
    """
    Median distance from persistence's prediction as a share of the animal's realised step.

    Persistence predicts the last observed fix, so this is how far the model chose to move, and
    persistence's own error is the step the animal actually took.
    """
    moved = geodesic_m(
        persistence["pred_lon"].to_numpy(float),
        persistence["pred_lat"].to_numpy(float),
        results["pred_lon"].to_numpy(float),
        results["pred_lat"].to_numpy(float),
    )
    step = persistence["error_m"].to_numpy(float)
    return float(np.median(moved / np.maximum(step, 1e-9)))


def seeded_arm_point(species: str, arm: str, persistence, boundary: float) -> dict | None:
    """
    Return a seeded arm's median travel p90 from its sealed predictions.
    """
    values, moves = [], []
    for seed in T.SEEDS:
        results = load_predictions(SE.prediction_path(species, arm, seed))
        if results is not None:
            values.append(travel_p90(results, persistence, boundary))
            moves.append(movement(results, persistence))
    if not values:
        return None
    return {
        "n_seeds": len(values),
        "p90_m": float(median(values)),
        "per_seed": values,
        "moves_share_of_step": float(median(moves)),
    }


def build_curve(species: str, profile_key: str) -> dict:
    """Assemble one species' P, Z, F and S fine-tuning curve."""
    boundary = T.boundary_m(profile_key)
    persistence = load_predictions(SE.prediction_path(species, "P", None))
    if persistence is None:
        raise SystemExit(
            f"FAIL: no persistence predictions for {species}. Run tier2_sealed_eval.py --predict "
            "first; this script reports against the sealed run rather than replacing it."
        )
    reference = travel_p90(persistence, persistence, boundary)

    def skill(value: float) -> float:
        return 1.0 - value / reference

    curve: dict = {
        "species": species,
        "regime_boundary_m": boundary,
        "persistence_travel_p90_m": reference,
        "references": {},
        "points": [],
    }
    for arm in ("Z", "B", "S"):
        point = seeded_arm_point(species, arm, persistence, boundary)
        if point is not None:
            point["skill"] = skill(point["p90_m"])
            curve["references"][arm] = point

    for k_percent in sorted(REGISTERED_FRACTIONS_PERCENT):
        draws = []
        for payload in finetune_runs():
            if payload["species"] != species or payload["finetune"]["k_percent"] != k_percent:
                continue
            results = load_predictions(prediction_path(species, payload))
            if results is None:
                continue
            finetune = payload["finetune"]
            draws.append(
                {
                    "draw": finetune["draw"],
                    "p90_m": travel_p90(results, persistence, boundary),
                    "moves_share_of_step": movement(results, persistence),
                    "n_individuals": finetune["n_individuals"],
                    "n_individuals_available": finetune["n_individuals_available"],
                    "n_windows": finetune["n_windows"],
                    "realised_fraction": finetune["realised_fraction"],
                    "raised_to_floor": finetune["raised_to_floor"],
                    "selected_step": payload["checkpoint_selection"].get("selected_step"),
                }
            )
        if not draws:
            continue
        values = [row["p90_m"] for row in draws]
        moves = [row["moves_share_of_step"] for row in draws]
        curve["points"].append(
            {
                "k_percent": k_percent,
                "n_draws": len(draws),
                # Registered low-k summary: median with the range across draws.
                "p90_m_median": float(median(values)),
                "p90_m_min": float(min(values)),
                "p90_m_max": float(max(values)),
                "skill_median": skill(float(median(values))),
                "moves_share_of_step": float(median(moves)),
                "skill_min": skill(float(max(values))),
                "skill_max": skill(float(min(values))),
                "n_individuals": draws[0]["n_individuals"],
                "n_individuals_available": draws[0]["n_individuals_available"],
                "realised_fraction": draws[0]["realised_fraction"],
                "raised_to_floor": draws[0]["raised_to_floor"],
                "n_windows_range": [
                    min(row["n_windows"] for row in draws),
                    max(row["n_windows"] for row in draws),
                ],
                "draws": draws,
            }
        )
    return curve


def write_markdown(payload: dict, path: Path) -> None:
    lines = [
        "# Tier 2 fine-tuning curve (F)",
        "",
        "Registered by `preregistration_tier2.md` sec 3 and budgeted by "
        "`preregistration_tier2_freeze.md` sec 3. Travel-regime p90 on the **sealed test partition**, "
        "individual-weighted at floor F = 25. Z is the k = 0 point and S the asymptote; at k = 1% and "
        "k = 10% the median across three independent draws is reported with the range beside it.",
        "",
        "**F does not move the goalposts.** The sec 4 success bands are defined on Z alone. A good "
        "fine-tuning curve does not rescue a WEAK zero-shot result and is not reported as though it "
        "does.",
        "",
    ]
    for curve in payload["species"]:
        reference = curve["persistence_travel_p90_m"]
        lines += [
            f"## {curve['species']}",
            "",
            f"Regime boundary {curve['regime_boundary_m']:,.1f} m; persistence travel p90 "
            f"{reference:,.0f} m.",
            "",
            "| point | individuals | windows | travel p90 | skill vs P | moves % of step | "
            "range across draws |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        z = curve["references"].get("Z")
        if z:
            lines.append(
                f"| **k = 0 (Z)** | 0 | 0 | {z['p90_m']:,.0f} m | {z['skill']:+.4f} | "
                f"{z['moves_share_of_step'] * 100:.1f}% | median of {z['n_seeds']} seeds |"
            )
        for point in curve["points"]:
            label = f"k = {point['k_percent']}%"
            if point["raised_to_floor"]:
                label += f" (floored, {point['realised_fraction']:.1%})"
            windows = point["n_windows_range"]
            window_cell = (
                f"{windows[0]:,}" if windows[0] == windows[1] else f"{windows[0]:,}-{windows[1]:,}"
            )
            spread = (
                "—"
                if point["n_draws"] == 1
                else f"{point['p90_m_min']:,.0f} - {point['p90_m_max']:,.0f} m "
                f"({point['n_draws']} draws)"
            )
            lines.append(
                f"| {label} | {point['n_individuals']} | {window_cell} | "
                f"{point['p90_m_median']:,.0f} m | {point['skill_median']:+.4f} | "
                f"{point['moves_share_of_step'] * 100:.1f}% | {spread} |"
            )
        for arm, title in (("S", "S (asymptote)"), ("B", "B (no FM package)")):
            entry = curve["references"].get(arm)
            if entry:
                lines.append(
                    f"| **{title}** | all | all | {entry['p90_m']:,.0f} m | "
                    f"{entry['skill']:+.4f} | {entry['moves_share_of_step'] * 100:.1f}% | "
                    f"median of {entry['n_seeds']} seeds |"
                )
        lines.append("")

        # Compare F100 with S when both use all target-species training individuals.
        full = next((p for p in curve["points"] if p["k_percent"] == 100), None)
        specialist = curve["references"].get("S")
        if full and specialist:
            gap = full["p90_m_median"] - specialist["p90_m"]
            lines += [
                f"**F₁₀₀ against S: {full['p90_m_median']:,.0f} m against "
                f"{specialist['p90_m']:,.0f} m ({gap:+,.0f} m).** Both see all of this species' "
                "training data and differ only in whether the weights were pretrained on the other "
                "species first, so this isolates the value of pretraining with training data held "
                "constant.",
                "",
            ]

    lines += [
        "## Caveats carried by this table",
        "",
        "- **F is one seed.** Every F run initialises from Z seed 0 (`FINETUNE_INIT_SEED`), because "
        "the registered count of 21 runs leaves no seed dimension. F is the only trained arm with no "
        "seed-variance estimate; the ranges above are draw spread, not seed spread, and the two are "
        "different quantities.",
        "- **The k = 1% point is five individuals on every species.** One percent of each held-out "
        "species' training pool falls below the registered floor of five, so all three were raised. "
        "The realised fractions are in the table; the nominal 1% is not what was run.",
        "- **F's early stopping is a choice between two checkpoints.** The selection cadence is every "
        "1,000 steps and the budget is 2,000, so each run evaluates twice. Both are registered "
        "constants, but this is not a resolved early-stopping curve.",
        "- **Ten test individuals per species.** The same power limit as every other sealed number; "
        "see `results.md` section 9.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", action="store_true", help="score F on the sealed partitions")
    parser.add_argument("--report", action="store_true", help="build the curve from predictions")
    parser.add_argument("--overwrite", action="store_true", help="re-score existing predictions")
    args = parser.parse_args()
    if not (args.predict or args.report):
        raise SystemExit("pass --predict, --report, or both")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device           : {device}")

    runs = finetune_runs()
    print(f"F runs found     : {len(runs)} of a registered 21")
    if not runs:
        raise SystemExit("FAIL: no F result files in results/tier2_final; train F first")

    if args.predict:
        run_predictions(device, overwrite=args.overwrite)

    if not args.report:
        return

    payload = {
        "arm": "F",
        "registered_by": "preregistration_tier2.md sec 3",
        "budget": "preregistration_tier2_freeze.md sec 3",
        "partition": "sealed_test",
        "stratum": STRATUM,
        "statistic": STATISTIC,
        "evaluation_floor": SE.EVALUATION_FLOOR,
        "initialised_from": f"{T.FINETUNE_INIT_ARM}_s{T.FINETUNE_INIT_SEED}",
        "species": [build_curve(species, key) for species, _, key, _, _ in T.HELD_OUT],
    }

    for curve in payload["species"]:
        print(f"\n=== {curve['species']}   persistence travel p90 "
              f"{curve['persistence_travel_p90_m']:,.0f} m ===")
        header = (
            f"{'point':<26} {'inds':>5} {'travel p90':>13} {'skill':>9} {'moves':>8}"
            "  range across draws"
        )
        print(header)
        print("-" * (len(header) + 12))
        z = curve["references"].get("Z")
        if z:
            print(
                f"{'k = 0 (Z)':<26} {0:>5} {z['p90_m']:>11,.0f} m {z['skill']:>+9.4f} "
                f"{z['moves_share_of_step'] * 100:>7.1f}%"
            )
        for point in curve["points"]:
            label = f"k = {point['k_percent']}%" + (" [floored]" if point["raised_to_floor"] else "")
            spread = (
                ""
                if point["n_draws"] == 1
                else f"  {point['p90_m_min']:,.0f} - {point['p90_m_max']:,.0f} m"
            )
            print(
                f"{label:<26} {point['n_individuals']:>5} "
                f"{point['p90_m_median']:>11,.0f} m {point['skill_median']:>+9.4f} "
                f"{point['moves_share_of_step'] * 100:>7.1f}%{spread}"
            )
        for arm, title in (("S", "S (asymptote)"), ("B", "B (package stripped)")):
            entry = curve["references"].get(arm)
            if entry:
                print(
                    f"{title:<26} {'all':>5} {entry['p90_m']:>11,.0f} m {entry['skill']:>+9.4f} "
                    f"{entry['moves_share_of_step'] * 100:>7.1f}%"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tier2_finetune_curve.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(payload, OUT / "tier2_finetune_curve.md")
    print(f"\nwrote {OUT / 'tier2_finetune_curve.json'} and tier2_finetune_curve.md")
    print(
        "\nF is a registered secondary comparator. The sec 4 success bands are defined on Z alone "
        "and no band reads F."
    )


if __name__ == "__main__":
    main()
