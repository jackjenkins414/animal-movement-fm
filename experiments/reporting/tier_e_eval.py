"""
Tier E evaluation of exploratory variants on held-out species validation individuals.

Designed from our earlier `tierE_exploratory_ablations.md`, not the Tier 2 preregistration. As such nothing here is treated as
a Tier 2 result, no success band is applied, and no test partition is opened.

For each held-out species and arm, we reports the median across five training seeds for travel-regime p90 skill, low displacement 
p90 relative to persistence, and how far the model prediction moves from persistence as a share of the animal's step. The last 
measure is also reported for low displacement windows, where the sealed evaluation showed the clearest FM failure.

`Zrecon` removes the masked reconstruction objective and changes nothing else, so Z vs Zrecon isolates that objective.
Za is not its complement; Za carries the full package and ablates the training pool (leave one domain source out), so we hope it
answers whether transfer depends on a same domain source rather than which half of the architecture is responsible; `Zfloor300` keeps 
the whole package and raises the scale floor instead, testing whether the objective is harmful only because sigma = 30m makes a 
stationary window encode as a large normalised displacement.

The arms are rescored on validation only so the comparison is like for like. Their sealed test results are not re read or changed.

Two scored partitions; by default, the script scores held-out species on validation data. `--sources` instead scores source species v
alidation data, which is where learned species embeddings such as E1embed can actually be used because they only contain rows for source 
species.

Source validation is in distribution, so these results show whether species information is useful when the species is known, not whether 
that information transfers to an unseen species. The two questions are therefore reported separately.
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
import tier2_inner_loop as IL  # noqa: E402
from animal_movement_fm.data.resample import CONTEXT_LENGTH  # noqa: E402
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched  # noqa: E402
from animal_movement_fm.eval.harness import evaluate  # noqa: E402
from animal_movement_fm.models.persistence import Persistence  # noqa: E402
from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig  # noqa: E402

OUT = REPO / "results" / "tier_e"
EARTH_RADIUS_M = 6_371_008.8

# tier 2 arms are rescored from their original results directory; Tier E variants read from results/tier_e
ARMS = (
    ("Z", T.RESULTS),
    ("B", T.RESULTS),
    ("Za", T.RESULTS),
    ("Zrecon", T.TIER_E_RESULTS),
    ("Zfloor300", T.TIER_E_RESULTS),
    ("Brecon", T.TIER_E_RESULTS),
    ("Zfloor1000", T.TIER_E_RESULTS),
    # S is the oracle reference E2 is read against
    ("S", T.RESULTS),
    ("E2embed", T.TIER_E_RESULTS),
)

# arms scored on source species validation individuals, where a species lookup has a row to read
SOURCE_ARMS = (
    ("Z", T.RESULTS),
    ("B", T.RESULTS),
    ("E1embed", T.TIER_E_RESULTS),
    ("E1shuffle", T.TIER_E_RESULTS),
)

# arms trained once per attached species rather than once overall
SPECIES_ATTACHED_ARMS = frozenset({"Za", "S"}) | frozenset(
    arm for arm, spec in T.TIER_E_ARMS.items() if spec.get("per_species")
)

# S is scored on the individuals it selected its own checkpoint on
OPTIMISTIC_ON_VALIDATION = frozenset({"S"})


def geodesic_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = phi2 - phi1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

# GenAI added 
def load_reported_model(results_dir: Path, arm: str, species: str, seed: int):
    """
    Rebuild an arm's reported model and the scale floor it trained under, or None if untrained.

    The `_reported.pt` file holds a state dict, not a model object, so the architecture is rebuilt from the saved configuration exactly as 
    the sealed evaluation does. Non-finite weights are refused rather than silently scored.

    Z, B and the Tier E variants are species independent; Za is trained separately for each held-out species.

    Returns `(model, sigma, pool_species)`. `pool_species` is the training pool's species order, which is what any learned species lookup 
    is indexed by; it is read from the checkpoint rather than reconstructed here, because it is `sorted(weights)` and not the order `SOURCES` 
    is written in, and getting it wrong would score one species under anothers identity without erroring.
    """
    run_id = f"{arm}_{species}_s{seed}" if arm in SPECIES_ATTACHED_ARMS else f"{arm}_s{seed}"
    path = results_dir / f"{run_id}_reported.pt"
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TrainConfig(**state["config"]).build()
    model.load_state_dict(state["model"])
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise SystemExit(f"FAIL: {path} holds non-finite weights; that run diverged")

    # tier E variant couldve trained under different scale floor; so scoring it at the registered 30 m would feed it a 
    # representation it never saw, so the floor is read back from the runs own result file; runs written before that 
    # field existed used the registered value by construction
    record = results_dir / f"{run_id}.json"
    sigma = SCALE_FLOOR_M
    if record.exists():
        sigma = float(json.loads(record.read_text()).get("scale_floor_m", SCALE_FLOOR_M))

    pool_species = list(state.get("pool_species") or [])
    if model.species_embed is not None and not pool_species:
        raise SystemExit(
            f"FAIL: {path.name} carries a species lookup but records no pool_species, so there is no way to know which row belongs to "
            "which species. Retrain that arm."
        )
    return model.eval(), sigma, pool_species


def species_row(model, pool_species: list[str], stem: str, arm: str) -> int | None:
    """
    Return the lookup row this species should be scored under, or None for an arm with no lookup.

    A species absent from our trained table has no row and cannot be scored by that arm at all. That is the finding E1 exists to make 
    concrete, so it is reported as a skip with a reason rather than quietly substituted with row 0.
    """
    if model.species_embed is None:
        return None
    if stem in pool_species:
        return pool_species.index(stem)
    if len(pool_species) == 1:
        # per species arm trains one row on its own attached species; the pool names it
        return 0
    raise LookupError(
        f"{arm} has no embedding row for {stem} (table covers {pool_species}); this arm is not "
        "zero-shot on that species"
    )


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


def score_species(stem: str, frame: pd.DataFrame, boundary: float, arms, device) -> dict:
    """
    Score every arm on one species' validation individuals against persistence.

    Shared by both modes; only the frame, the boundary and the arm list differ between scoring a held-out species and scoring a source 
    species.
    """
    persistence = evaluate(
        Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False
    ).sort_values(["individual_id", "timestamp"]).reset_index(drop=True)

    step = persistence["error_m"].to_numpy(float)
    low = step <= boundary
    travel = ~low

    p_travel_p90 = quantile_by_individual(persistence, travel, 0.90)
    p_low_p90 = quantile_by_individual(persistence, low, 0.90)

    print(
        f"=== {stem}   {len(persistence):,} validation windows"
        f"   travel {int(travel.sum()):,}   low-displacement {int(low.sum()):,}"
        f"   (boundary {boundary:,.1f} m)"
    )
    header = (
        f"{'arm':10} {'travel p90 skill':>17} {'low-disp p90 / P':>17}"
        f" {'moves % of step':>16} {'low-disp % step':>16}"
    )
    print(header)
    print("-" * len(header))
    print(f"{'P':10} {'—':>17} {'1.00x':>17} {'0.0%':>16} {'0.0%':>16}")

    rows: dict = {}
    for arm, results_dir in arms:
        per_seed = []
        skipped = None
        for seed in T.SEEDS:
            loaded = load_reported_model(results_dir, arm, stem, seed)
            if loaded is None:
                continue
            model, sigma, pool_species = loaded
            try:
                row = species_row(model, pool_species, stem, arm)
            except LookupError as error:
                skipped = str(error)
                break

            results = evaluate_batched(
                MovementFMPredictor(model, device=device, sigma=sigma, species_index=row),
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

        if skipped:
            print(f"{arm:10} {'no embedding row':>17}")
            rows[arm] = {"scored": False, "reason": skipped}
            continue
        if not per_seed:
            print(f"{arm:10} {'not trained':>17}")
            continue

        median = np.median(np.asarray(per_seed), axis=0)
        flag = "  (validation-selected; optimistic)" if arm in OPTIMISTIC_ON_VALIDATION else ""
        print(
            f"{arm:10} {median[0]:+17.4f} {median[1]:16.2f}x"
            f" {median[2] * 100:15.1f}% {median[3] * 100:15.1f}%{flag}"
        )
        rows[arm] = {
            "scored": True,
            "n_seeds": len(per_seed),
            "travel_p90_persistence_skill": median[0],
            "low_displacement_p90_over_persistence": median[1],
            "median_move_as_share_of_step": median[2],
            "low_displacement_move_as_share_of_step": median[3],
            "selected_checkpoint_on_this_partition": arm in OPTIMISTIC_ON_VALIDATION,
        }

    print()
    return {
        "n_validation_windows": int(len(persistence)),
        "n_individuals": int(persistence["individual_id"].nunique()),
        "regime_boundary_m": boundary,
        "arms": rows,
    }


# GenAI reference: I asked GenAI to write this main orchestration from my Tier E evaluation specification like the  CLI, scoring loop, 
# reporting and JSON output; I reviewed the implementation afterwards
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20_000, help="must match the trained budget")
    parser.add_argument(
        "--species",
        default=None,
        help="evaluate one species; default is all three in the selected mode",
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help=(
            "score source-species validation individuals instead of held-out ones. This is where "
            "an arm carrying a learned species lookup can be read; it answers whether species "
            "information is usable, not whether it transfers."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device           : {device}")
    print("partition        : validation individuals only; no test partition is opened\n")

    if args.sources:
        wanted = [
            (stem, profile_key)
            for stem, _, profile_key in IL.SOURCES
            if args.species in (None, stem)
        ]
        fold_of = {stem: index for index, (stem, *_) in enumerate(IL.SOURCES)}
        arms, mode = SOURCE_ARMS, "source_validation"
    else:
        wanted = [
            # HELD_OUT is (stem, split_stem, profile_key, domain, matching_source). Taking the last
            # element here instead of the third silently scored canis_lupus against cervus_elaphus's
            # 49.4 m boundary rather than its own 131.3 m, which changes the strata and every number
            # in the table. The guard makes that failure loud rather than plausible.
            (stem, profile_key)
            for stem, _, profile_key, _, _ in T.HELD_OUT
            if args.species in (None, stem)
        ]
        for stem, profile_key in wanted:
            if profile_key in T.SOURCE_STEMS:
                raise SystemExit(
                    f"FAIL: {stem} resolved to regime profile {profile_key!r}, which is a source "
                    "species. HELD_OUT was unpacked in the wrong order."
                )
        fold_of = None
        arms, mode = ARMS, "held_out_validation"

    if not wanted:
        raise SystemExit(f"FAIL: {args.species!r} is not a species in this mode")

    payload: dict = {
        "tier": "E",
        "exploratory": True,
        "partition": mode,
        "opened_test_partition": False,
        "arms": [arm for arm, _ in arms],
        "species": {},
    }

    for stem, profile_key in wanted:
        frame = (
            IL.validation_frame(fold_of[stem])
            if fold_of is not None
            else T.validation_frame(stem)
        )
        payload["species"][stem] = score_species(
            stem, frame, T.boundary_m(profile_key), arms, device
        )

    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_sources" if args.sources else ""
    path = OUT / f"tier_e_validation_metrics{suffix}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")
    if args.sources:
        print(
            "\nSource-species validation is IN DISTRIBUTION: every arm here trained on these "
            "species' training individuals. A gain says species information is usable, not that it "
            "transfers to a species with no row in the table."
        )
    print(
        "\nTier E is exploratory. No success band is applied, no test partition was opened, and "
        "nothing here may be reported as a Tier 2 result."
    )


if __name__ == "__main__":
    main()
