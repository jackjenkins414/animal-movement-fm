# GenAI reference: I wrote the docstring, the CLI design (training individuals only profiling, shared cadence flag) and
# the call into profile_species(); GenAI helped with the CSV append/replace logic and the terminal print summaries.
# It was verified by rerunning the stork profile and checking the printed p50/boundary matched profile_species() directly,
# and that a repeat run replaced the species row rather than duplicating it. We later modified this ourselves so the bird 
# study's rows go to their own table; it refuses --decide-threshold and --reselect, which only run on the shared one
"""
Create a movement regime profile for one species.

Our profile summarises movement characteristics used to compare candidate species and support selection for cross-domain
pretraining. All calculations, including the low displacement boundary, are implemented in
`animal_movement_fm.analysis.regime_profile`.

Why this exists
---------------
Cross-domain pretraining benefits come from covering distinct MOVEMENT REGIMES, not from taxonomic count: adding a second
soaring migrant teaches a model much less than adding a benthic diver or a resident grazer. Without measurement, "we
chose species that span the movement  space" is an assertion. This script turns it into a table.

What it does NOT do
-------------------
It never fits a forecasting model or touches a prediction target. The two component lognormal fit is only a distributional
diagnostic on training individual step lengths. Under the exploration/modelling line drawn by Hofman et al. (2023), this is
dataset property exploration.

It also does not decide eligibility: that mechanical gate is applied first, and only survivors are profiled. And it does not
choose a boundary formula opportunistically. With `--decide-threshold` it applies the decision tree registered in our
preregistration amendment A4.

Usage
-----
    # a species with a frozen split: the boundary must come from TRAINING individuals only
    uv run python experiments/data/regime_profile.py \\
        --segments data/processed/stork_segments.parquet --name white_stork \\
        --domain flying --split splits/stork_split_seed42.json

    # a candidate with no split yet, profiled before committing to a full ingest.
    # Raw fixes are resampled internally with the pipeline's own resampler.
    uv run python experiments/data/regime_profile.py \\
        --segments data/raw/elephant_seal_sample.parquet --name southern_elephant_seal \\
        --domain marine

Appends to `results/regime_profiles.csv` and prints the accumulated comparison. Once every eligible candidate has a training only
row, pass `--decide-threshold` to apply the registered tree and write `results/regime_threshold_decision.json` plus the frozen
descriptor standardisation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import pandas as pd

from animal_movement_fm.analysis.regime_profile import (
    DESCRIPTORS,
    ThresholdDecision,
    apply_threshold_decision,
    coverage_distances,
    descriptor_standardisation,
    profile_species,
    select_threshold_rule,
)
from animal_movement_fm.data.split import load_split

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / "results" / "regime_profiles.csv"
OUT_DECISION = REPO / "results" / "regime_threshold_decision.json"
# Tracked, unlike everything under results/. The A4 decision is made once and carried forward; see the file.
FROZEN_DECISION = REPO / "registered" / "tier2_regime_decision.json"
OUT_STANDARDISATION = REPO / "results" / "regime_descriptor_standardisation.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--segments",
        required=True,
        help="parquet of resampled segments, or of raw fixes (resampled internally)",
    )
    p.add_argument("--name", required=True, help="species name for the profile row")
    p.add_argument(
        "--domain",
        default="unknown",
        choices=("terrestrial", "marine", "flying", "unknown"),
        help="taxon group; a constraint on selection, never on measurement",
    )
    p.add_argument(
        "--split",
        default=None,
        help="frozen split manifest. REQUIRED for any species whose boundary will be used in "
        "evaluation: the registered definition computes the anchor on training individuals only",
    )
    p.add_argument(
        "--split-part",
        default="train",
        choices=("train", "val", "test"),
        help="which part of the manifest to profile; leave as train unless diagnosing",
    )
    p.add_argument(
        "--cadence-hours",
        type=float,
        default=6.0,
        help="modelling cadence; rho is NOT cadence invariant, so every species must share this",
    )
    p.add_argument("--group-col", default="segment_id")
    p.add_argument(
        "--reselect",
        action="store_true",
        help="re-run the A4 tree instead of applying the frozen decision; changes the boundary for every "
             "species at once, including species whose evaluation is already sealed",
    )
    p.add_argument(
        "--decide-threshold",
        action="store_true",
        help="assert that the accumulated CSV holds every mechanically eligible candidate's "
        "training-only profile, then apply the tree registered in amendment A4",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=OUT_CSV,
        help="profile table to add this species' row to; anything but the default refuses --decide-threshold "
             "and --reselect",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # the A4 decision and the screening standardisation are defined over the shared table, never a separate one
    if args.out_csv.resolve() != OUT_CSV.resolve() and (args.decide_threshold or args.reselect):
        raise SystemExit(
            f"FAIL: --decide-threshold and --reselect run only on {OUT_CSV}, not on {args.out_csv}. Nothing was written."
        )

    path = Path(args.segments)
    if not path.exists():
        raise SystemExit(f"Segments file not found: {path}")

    frame = pd.read_parquet(path)

    individual_ids = None
    if args.split:
        split = load_split(Path(args.split))
        individual_ids = getattr(split, args.split_part)
        print(
            f"restricting to {len(individual_ids)} {args.split_part} individuals from {args.split}"
        )
    else:
        print(
            "WARNING: no --split given, so the boundary is computed over ALL individuals. That is "
            "fine for a first look at a candidate species, but the registered definition requires "
            "training individuals only, so re-run with --split before this species' boundary is "
            "used in any evaluation."
        )

    decision_ready = bool(args.split and args.split_part == "train")
    print(
        f"profiling {args.name} at {args.cadence_hours}h cadence (steps spanning exactly one bin)..."
    )
    row = {
        "species": args.name,
        "domain": args.domain,
        "cadence_hours": args.cadence_hours,
        "split_manifest": args.split or "",
        "split_part": args.split_part if args.split else "",
        "threshold_decision_ready": decision_ready,
    }
    row.update(
        profile_species(
            frame,
            individual_ids=individual_ids,
            group_col=args.group_col,
            bin_seconds=int(args.cadence_hours * 3600),
        )
    )

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        table = pd.read_csv(out_csv)
        table = table[table["species"] != args.name]  # Replace any earlier profile for this species
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(out_csv, index=False)

    print(
        f"\n  p50 / p75 / p90: {row['step_p50_m']:,.1f} / "
        f"{row['step_p75_m']:,.1f} / {row['step_p90_m']:,.1f} m"
    )
    print(
        f"  median anchor: 0.10 x p50 = {row['median_anchor_boundary_m']:,.1f} m "
        f"({row['median_anchor_low_fraction']:.1%} low displacement)"
    )
    if row["mixture_supported"]:
        print(
            f"  mixture crossing: {row['mixture_crossing_m']:,.1f} m "
            f"({row['mixture_crossing_fraction']:.1%} low displacement; "
            f"dBIC={row['mixture_bic_improvement']:,.1f})"
        )
        print(
            "  anchors inside low component: "
            f"p50={row['p50_inside_low_component']}, "
            f"p75={row['p75_inside_low_component']}, "
            f"p90={row['p90_inside_low_component']}"
        )
    else:
        print(f"  mixture crossing: unavailable ({row['mixture_failure_reason']})")

    if row["n_turns_used"] and pd.isna(row["rho"]):
        print(
            f"  NOTE: rho withheld, only {row['n_turns_used']:,} qualifying turn pairs. This "
            "species does not clear the descriptor eligibility gate and cannot enter the coverage "
            "comparison until it does."
        )

    print("\naccumulated regime profiles:")
    show = [
        "species",
        "domain",
        "n_individuals",
        "n_steps_at_cadence",
        "step_p50_m",
        "step_p75_m",
        "step_p90_m",
        "median_anchor_boundary_m",
        "median_anchor_low_fraction",
        "mixture_supported",
        "mixture_crossing_m",
        "mixture_crossing_fraction",
    ] + DESCRIPTORS
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    if len(table) > 1:
        try:
            distances = coverage_distances(table)
        except ValueError as exc:
            print(f"\ncoverage distances unavailable: {exc}")
        else:
            print(
                "\npairwise distance in standardised descriptor space (larger = more complementary):"
            )
            print(distances.to_string(float_format=lambda v: f"{v:6.2f}"))
            print(
                "\nPick candidates FAR from the species already in the pool, and far from each "
                "other. Within a domain, the source/held-out pair's distance is MEASURED, not "
                "optimised: secondary prediction 2 uses it as the explanatory variable for uneven "
                "transfer. NOTE these distances are standardised over the species currently in the "
                "table and will shift as candidates are added; --decide-threshold freezes them."
            )
    else:
        print("\n(only one profile so far; add candidates to see coverage distances)")

    if args.decide_threshold:
        if "threshold_decision_ready" not in table.columns:
            raise SystemExit("No training-only decision rows found in the accumulated CSV")
        ready_mask = table["threshold_decision_ready"].map(
            lambda value: str(value).strip().lower() == "true"
        )
        ready = table[ready_mask].copy()
        if ready.empty:
            raise SystemExit("No training-only decision rows found in the accumulated CSV")

        # A4's tree is re-run only on demand. sec 1 holds the metrics and protocol constant from Tier 2 into
        # Tier 3, and the boundary formula is part of that, so once decided it is APPLIED to new species rather
        # than re-selected over a table that keeps growing. Re-selecting on 15-09-2026 moved the universal rule
        # from 0.1 x p50 to 0.00603816 x p90 because two bird candidates have a median step inside their own low
        # displacement component, and that silently rewrote the strata a sealed evaluation had already used.
        # GenAI catch: a missing frozen file used to fall through to re selection silently, which is how a
        # doubled path in FROZEN_DECISION switched A16 off without anyone noticing; GenAI changed to be a hard 
        # failure.
        if not args.reselect and not FROZEN_DECISION.exists():
            raise SystemExit(
                f"FAIL: {FROZEN_DECISION} is missing. The A4 decision is frozen and only --reselect may re-run "
                "the tree; restore the file from git rather than letting the tree re-select."
            )
        if not args.reselect:
            frozen = json.loads(FROZEN_DECISION.read_text())
            decision = ThresholdDecision(
                anchor_quantile=int(frozen["anchor_quantile"]),
                multiplier=float(frozen["multiplier"]),
                reference_species=str(frozen["reference_species"]),
                reference_boundary_m=float(frozen["reference_boundary_m"]),
                pooled_only_species=tuple(frozen["pooled_only_species"]),
                reasons=tuple(frozen["reasons"]),
            )
            print(f"\napplying the frozen decision from {FROZEN_DECISION.name}; pass --reselect to re-run the tree")
        else:
            decision = select_threshold_rule(ready)
        decided = apply_threshold_decision(ready, decision)
        if not args.reselect:
            # the frozen rule must also reproduce the frozen Tier 2 numbers before anything is written
            applied = decided.set_index("species")["selected_boundary_m"]
            for species, value in frozen["tier2_boundaries_m"].items():
                if species in applied.index and not math.isclose(
                    float(applied[species]), float(value), rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise SystemExit(
                        f"FAIL: the frozen rule now gives {species} {float(applied[species])!r} m against the "
                        f"registered {float(value)!r} m, so its profile row has changed. Nothing was written."
                    )
        selected = decided.set_index("species")[
            [
                "selected_anchor_quantile",
                "selected_multiplier",
                "selected_boundary_m",
                "pooled_only",
            ]
        ]
        for column in selected.columns:
            table[column] = table["species"].map(selected[column])
        table.to_csv(OUT_CSV, index=False)

        payload = asdict(decision)
        payload["rule_name"] = decision.rule_name
        OUT_DECISION.write_text(json.dumps(payload, indent=2) + "\n")

        # A16; screening standardisation only, is recomputed over the current decision-ready species, so 
        # distances can move as the candidate pool changes. Sealed Tier 2 results instead use the frozen
        # `registered/tier2_descriptor_standardisation.json`.
        screening = descriptor_standardisation(ready)
        screening = {
            "role": "screening",
            "note": (
                "Recomputed over the current decision-ready species. Sealed Tier 2 results use "
                "registered/tier2_descriptor_standardisation.json instead (A16)."
            ),
            "n_species": len(screening["species"]),
            **screening,
        }
        OUT_STANDARDISATION.write_text(json.dumps(screening, indent=2) + "\n")

        print(f"\nREGISTERED DECISION (amendment A4): {decision.rule_name}")
        for reason in decision.reasons:
            print(f"  - {reason}")
        if decision.pooled_only_species:
            print(f"  pooled only: {', '.join(decision.pooled_only_species)}")
        print(f"decision -> {OUT_DECISION}")
        print(
            f"screening descriptor standardisation ({len(screening['species'])} species) "
            f"-> {OUT_STANDARDISATION}"
        )
        print(
            "  this file moves with the candidate pool; sealed Tier 2 distances read "
            "registered/tier2_descriptor_standardisation.json (A16)"
        )

    print(f"\nprofiles -> {out_csv}")


if __name__ == "__main__":
    main()
