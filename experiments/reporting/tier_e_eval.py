"""
Tier E evaluation of exploratory variants on held-out species validation individuals.

Designed from our earlier `tierE_exploratory_ablations.md`, not the Tier 2 preregistration. As such nothing here is treated as
a Tier 2 result, no success band is applied, and no test partition is opened.

For each held-out species and arm, we reports the median across five training seeds for travel-regime p90 skill, low displacement 
p90 relative to persistence, and how far the model prediction moves from persistence as a share of the animal's step. The last 
measure is also reported for low displacement windows, where the sealed evaluation showed the clearest FM failure.

Za removes conditioning while keeping reconstruction; `Zrecon` removes reconstruction while keeping conditioning. Comparing them 
aims to help us figure out which part of the FM package drives that behaviour.

The arms are rescored on validation only so the comparison is like for like. Their sealed test results are not re read or changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))

import tier2_final_train as T  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402
from animal_movement_fm.eval.harness import evaluate  # noqa: E402
from animal_movement_fm.models.persistence import Persistence  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig  # noqa: E402

OUT = REPO / "results" / "tier_e"
EARTH_RADIUS_M = 6_371_008.8

# tier 2 arms are rescored from their original results directory; Tier E variants read from results/tier_e
ARMS = (
    ("Z", T.RESULTS),
    ("B", T.RESULTS),
    ("Za", T.RESULTS),
    ("Zrecon", T.TIER_E_RESULTS),
)


def geodesic_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = phi2 - phi1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

# GenAI added 
def load_reported_model(results_dir: Path, arm: str, species: str, seed: int):
    """
    Rebuild the model an arm reports for one species and seed, or None if it was not trained.

    The `_reported.pt` file holds a state dict, not a model object, so the architecture is rebuilt
    from the saved configuration exactly as the sealed evaluation does. Non-finite weights are
    refused rather than silently scored.

    Z, B and the Tier E variants are species independent; Za is trained separately for each held-out
    species.
    """
    run_id = f"{arm}_{species}_s{seed}" if arm == "Za" else f"{arm}_s{seed}"
    path = results_dir / f"{run_id}_reported.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainConfig(**state["config"]).build()
    model.load_state_dict(state["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit(f"FAIL: {path} holds non-finite weights; that run diverged")
    return model.eval()



def quantile_by_individual(frame: pd.DataFrame, mask: np.ndarray, q: float) -> float:
    """
    Return the A5 individual weighted quantile.

    Calculated within each individual first, then averaged across individuals.
    """
    rows = frame.loc[mask]
    if rows.empty:
        return float("nan")
    per_individual = rows.groupby("individual_id")["error_m"].quantile(q)
    return float(per_individual.mean())


# GenAI reference: I asked GenAI to write this main orchestration from my Tier E evaluation specification like the 
# CLI, scoring loop, reporting and JSON output; I reviewed the implementation afterwards
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000, help="must match the trained budget")
    parser.add_argument(
        "--species",
        default=None,
        help="evaluate one held-out species; default is all three",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device           : {device}")
    print("partition        : validation individuals only; no test partition is opened\n")

    wanted = [entry for entry in T.HELD_OUT if args.species in (None, entry[0])]
    if not wanted:
        raise SystemExit(f"FAIL: {args.species!r} is not a held-out species")

    payload: dict = {"tier": "E", "partition": "validation", "species": {}}

    for species, _, profile_key, _, _ in wanted:
        # HELD_OUT is (stem, split_stem, profile_key, domain, matching_source). Taking the last
        # element here instead of the third silently scored canis_lupus against cervus_elaphus's
        # 49.4 m boundary rather than its own 131.3 m, which changes the strata and every number
        # in the table. The guard makes that failure loud rather than plausible.
        if profile_key in T.SOURCE_STEMS:
            raise SystemExit(
                f"FAIL: {species} resolved to regime profile {profile_key!r}, which is a source "
                "species. HELD_OUT was unpacked in the wrong order."
            )
        frame = T.validation_frame(species)
        boundary = T.boundary_m(profile_key)
        persistence = evaluate(
            Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False
        ).sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

        step = persistence["error_m"].to_numpy(float)
        low = step <= boundary
        travel = ~low

        p_travel_p90 = quantile_by_individual(persistence, travel, 0.90)
        p_low_p90 = quantile_by_individual(persistence, low, 0.90)

        print(
            f"=== {species}   {len(persistence):,} validation windows"
            f"   travel {int(travel.sum()):,}   low-displacement {int(low.sum()):,}"
            f"   (boundary {boundary:,.1f} m)"
        )
        header = (
            f"{'arm':8} {'travel p90 skill':>17} {'low-disp p90 / P':>17}"
            f" {'moves % of step':>16} {'low-disp % step':>16}"
        )
        print(header)
        print("-" * len(header))
        print(f"{'P':8} {'—':>17} {'1.00x':>17} {'0.0%':>16} {'0.0%':>16}")

        species_rows: dict = {}
        for arm, results_dir in ARMS:
            per_seed = []
            for seed in T.SEEDS:
                model = load_reported_model(results_dir, arm, species, seed)
                if model is None:
                    continue
                results = evaluate_batched(
                    MovementFMPredictor(model, device=device),
                    frame,
                    context_length=CONTEXT_LENGTH,
                ).sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

                moved = geodesic_m(
                    persistence["pred_lon"].to_numpy(float),
                    persistence["pred_lat"].to_numpy(float),
                    results["pred_lon"].to_numpy(float),
                    results["pred_lat"].to_numpy(float),
                )
                share = moved / np.maximum(step, 1e-9)
                per_seed.append(
                    (
                        1.0 - quantile_by_individual(results, travel, 0.90) / p_travel_p90,
                        quantile_by_individual(results, low, 0.90) / p_low_p90,
                        float(np.median(share)),
                        float(np.median(share[low])) if low.any() else float("nan"),
                    )
                )

            if not per_seed:
                print(f"{arm:8} {'not trained':>17}")
                continue

            median = np.median(np.asarray(per_seed), axis=0)
            print(
                f"{arm:8} {median[0]:+17.4f} {median[1]:16.2f}x"
                f" {median[2] * 100:15.1f}% {median[3] * 100:15.1f}%"
            )
            species_rows[arm] = {
                "n_seeds": len(per_seed),
                "travel_p90_persistence_skill": median[0],
                "low_displacement_p90_over_persistence": median[1],
                "median_move_as_share_of_step": median[2],
                "low_displacement_move_as_share_of_step": median[3],
            }

        print()
        payload["species"][species] = {
            "n_validation_windows": int(len(persistence)),
            "regime_boundary_m": boundary,
            "arms": species_rows,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tier_e_validation_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")
    print(
        "\nTier E is exploratory. No success band is applied, no test partition was opened, and "
        "nothing here may be reported as a Tier 2 result."
    )


if __name__ == "__main__":
    main()