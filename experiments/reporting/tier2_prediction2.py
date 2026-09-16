# GenAI reference: I used GenAI write the main method and parts of the other functions from my spec; that being the reading definitions,
# reporting every one instead of picking after seeing results, and the original drift guard are mine. I reviewed the code, added the
# docstrings/comments, and checked the distances and headline numbers against the sealed verdict. The drift guard was later replaced
# under A16 (15-09-2026); mine compared two files that are written by the same branch, so it could not fail; GenAI wrote the
# replacement, which checks the role species' raw rows against constants frozen at the Tier 2 freeze.
"""
Secondary preregistered prediction 2: does transfer track regime distance from the source pool?

In our `preregistration_tier2.md` sec 5 we predicted that transfer varies across the three held-out species and tracks their regime 
distance from the source pool. Since that wording allows somewhat several reasonable distance definitions, this script reports all 
of them rather than choosing one after seeing the results.

Note that only committed/frozen artefacts are read. The test partition is not reopened: transfer results and paired bootstrap intervals 
come from the existing sealed verdict, so this is more of a rereport rather than a new evaluation.

Inputs:
    results/regime_profiles.csv
    registered/tier2_descriptor_standardisation.json  (frozen Tier 2 standardisation; A16)
    results/tier2_sealed/tier2_verdict.json

Writes `results/tier2_sealed/tier2_prediction2.json` and prints the reporting tables.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

# A16; the frozen Tier 2 standardisation used by sealed analyses; the results/copy is a moving screening artefact and must 
# not be used here
FROZEN_STANDARDISATION = REPO / "registered" / "tier2_descriptor_standardisation.json"

# profile keys only; the segment/split identifiers used elsewhere are not needed here
SOURCES = ("white_stork", "cervus_elaphus", "mirounga_leonina")
HELD_OUT = ("larus_fuscus", "canis_lupus", "mirounga_angustirostris")
MATCHING_DOMAIN_SOURCE = {
    "larus_fuscus": "white_stork",
    "canis_lupus": "cervus_elaphus",
    "mirounga_angustirostris": "mirounga_leonina",
}

# report every defensible reading of "distance from the source pool"; `matching_domain` is the pre-training interpretation from 
# tier2_species_roles.md; the others are sensitivity checks
READINGS = ("matching_domain", "nearest", "mean_to_pool", "centroid", "max_to_pool")


def standardised_descriptors() -> pd.DataFrame:
    """Apply the frozen A10 transforms and standardisation to the descriptor table."""
    if not FROZEN_STANDARDISATION.exists():
        raise SystemExit(
            f"FAIL: {FROZEN_STANDARDISATION.name} is missing. The sealed distances are only "
            "reproducible against the frozen Tier 2 standardisation; the screening file in "
            "results/ is not a substitute for it."
        )
    std = json.loads(FROZEN_STANDARDISATION.read_text())
    profiles = pd.read_csv(RESULTS / "regime_profiles.csv").set_index("species")

    frame = pd.DataFrame(index=profiles.index)
    for name in std["descriptors"]:
        values = profiles[name].to_numpy(dtype=float)
        transform = std["transforms"][name]
        if transform == "identity":
            frame[name] = values
        elif transform == "log10":
            frame[name] = np.log10(values)
        elif transform == "logit":
            frame[name] = np.log(values / (1.0 - values))  # natural log; reproduces the frozen centres
        else:
            raise ValueError(f"unknown transform {transform!r} for {name}")

    centre = pd.Series(std["centre"])
    scale = pd.Series(std["scale"])

    # A16: verify that the six frozen Tier 2 role species descriptor rows have not drifted; centres/scales are already frozen, 
    # so the remaining risk is a changed profile row underneath them
    recorded = std["role_species_transformed"]

    missing = sorted(set(recorded) - set(frame.index))
    if missing:
        raise SystemExit(
            f"FAIL: regime_profiles.csv has no row for {', '.join(missing)}; the sealed distances "
            "cannot be recomputed without every role species."
        )

    drift = {
        species: float(np.abs(frame.loc[species] - pd.Series(values)).max())
        for species, values in recorded.items()
    }

    worst, value = max(drift.items(), key=lambda item: item[1])
    if value > 1e-12:
        raise SystemExit(
            f"FAIL: {worst} no longer matches the descriptors frozen at the Tier 2 freeze "
            f"(max drift {value:.3e}). A2 forbids rebuilding the role species, so either that "
            "happened or the A10 transforms changed. The registered distances cannot be "
            "reproduced from this table; do not report these numbers."
        )
    return (frame - centre) / scale


def distance_table(standardised: pd.DataFrame) -> pd.DataFrame:
    """Return held-out-to-source Euclidean distances under each reading."""

    def distance(a: str, b: str) -> float:
        return float(np.linalg.norm(standardised.loc[a].to_numpy() - standardised.loc[b].to_numpy()))

    centroid = standardised.loc[list(SOURCES)].to_numpy().mean(axis=0)
    rows = []
    for species in HELD_OUT:
        per_source = {source: distance(species, source) for source in SOURCES}
        rows.append(
            {
                "species": species,
                "matching_domain": per_source[MATCHING_DOMAIN_SOURCE[species]],
                "nearest": min(per_source.values()),
                "nearest_source": min(per_source, key=per_source.get),
                "mean_to_pool": float(np.mean(list(per_source.values()))),
                "max_to_pool": max(per_source.values()),
                "centroid": float(np.linalg.norm(standardised.loc[species].to_numpy() - centroid)),
                **{f"d_{source}": per_source[source] for source in SOURCES},
            }
        )
    return pd.DataFrame(rows).set_index("species")


def sealed_transfer() -> pd.DataFrame:
    """Read headline travel-p90 values and paired-bootstrap results from the sealed verdict."""
    verdict = json.loads((RESULTS / "tier2_sealed" / "tier2_verdict.json").read_text())

    def headline(species: str, arm: str) -> float:
        # A3 uses the median across training seeds; untrained arms have one headline block.
        comparator = verdict["species"][species]["comparators"][arm]
        per_seed = comparator.get("per_seed") or {}
        if per_seed:
            return float(np.median(sorted(run["travel"]["p90_m"] for run in per_seed.values())))
        return float(comparator["headline"]["travel"]["p90_m"])

    rows = []
    for species in HELD_OUT:
        p, z, b, s = (headline(species, arm) for arm in ("P", "Z", "B", "S"))
        series = verdict["species"][species]["bootstrap"]["series"]["travel_p90_m"]
        paired = series["paired_difference_vs_Z_m"]  # Stored as Z - arm; positive means Z is worse.
        rows.append(
            {
                "species": species,
                "n_test_individuals": series["n_individuals"],
                "P_p90_m": p,
                "Z_p90_m": z,
                "skill": 1.0 - z / p,
                # skill = -(Z - P) / P, so the interval flips and swaps ends.
                "skill_lo": -paired["P"]["hi"] / p,
                "skill_hi": -paired["P"]["lo"] / p,
                "Z_minus_B_m": z - b,
                "Z_minus_B_excludes_zero": paired["B"]["excludes_zero"],
                "Z_minus_S_m": z - s,
                "Z_minus_S_excludes_zero": paired["S"]["excludes_zero"],
            }
        )
    return pd.DataFrame(rows).set_index("species")


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Return Spearman rank correlation; with n = 3 only a few values are possible."""
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def main() -> None:
    standardised = standardised_descriptors()
    distances = distance_table(standardised)
    transfer = sealed_transfer()
    pd.set_option("display.width", 200)

    print("== distance from the source pool, 8 descriptors, A10 transforms, frozen standardisation ==")
    print(distances.round(3).to_string())

    print("\n== transfer: Z travel-regime p90, median across seeds, registered paired bootstrap ==")
    print(transfer[["n_test_individuals", "P_p90_m", "Z_p90_m", "skill", "skill_lo", "skill_hi"]].round(4).to_string())

    spread = float(transfer["skill"].max() - transfer["skill"].min())
    contains_zero = bool(((transfer["skill_lo"] < 0) & (transfer["skill_hi"] > 0)).all())
    shared_lo = float(transfer["skill_lo"].max())
    shared_hi = float(transfer["skill_hi"].min())
    print(f"\nclause (a): spread in Z skill across the three species = {spread:.4f}")
    print(f"            every interval contains zero: {contains_zero}")
    print(f"            all three intervals share [{shared_lo:.4f}, {shared_hi:.4f}]: {shared_lo < shared_hi}")

    # The prediction implies rho = -1: smaller distance should mean larger transfer skill.
    # Report three transfer measures because the registration does not name one uniquely.
    measures = {
        "Z_skill": transfer["skill"],
        "neg_Z_minus_B": -transfer["Z_minus_B_m"],
        "neg_Z_minus_S": -transfer["Z_minus_S_m"],
    }
    rho = {
        measure: {reading: round(spearman(distances[reading], values), 2) for reading in READINGS}
        for measure, values in measures.items()
    }
    print("\n== clause (b): Spearman rho, prediction implies -1 ==")
    print(pd.DataFrame(rho).T.to_string())

    print("\n== orderings, nearest first ==")
    for reading in READINGS:
        order = distances[reading].sort_values().index
        print(f"  {reading:<16} " + "  <  ".join(f"{s} {distances.loc[s, reading]:.2f}" for s in order))
    order = transfer["skill"].sort_values(ascending=False).index
    print(f"  {'observed skill':<16} " + "  >  ".join(f"{s} {transfer.loc[s, 'skill']:+.4f}" for s in order))

    n = len(HELD_OUT)
    permutations = list(itertools.permutations(range(n)))
    attainable = sorted({round(spearman(pd.Series(range(n)), pd.Series(p)), 2) for p in permutations})
    print(f"\n== power at n = {n} ==")
    print(f"  attainable rho: {attainable}")
    print(f"  one-tailed p for a perfect ordering: 1/{len(permutations)} = {1 / len(permutations):.3f}")
    print("  The held-out set is frozen across tiers (sec 1), so Tier 3 does not add observations here.")

    payload = {
        "prediction": "preregistration_tier2.md section 5, secondary prediction 2",
        "statistic": "travel-regime p90 persistence skill, individual-weighted, floor F = 25",
        "seed_aggregation": "median across training seeds (A3)",
        "descriptor_standardisation": "registered/tier2_descriptor_standardisation.json (A10, frozen under A16)",
        "opened_test_partition": False,
        "distances": json.loads(distances.to_json(orient="index")),
        "transfer": json.loads(transfer.to_json(orient="index")),
        "clause_a": {
            "skill_spread": spread,
            "all_intervals_contain_zero": contains_zero,
            "shared_interval": [shared_lo, shared_hi],
            "intervals_mutually_overlap": shared_lo < shared_hi,
        },
        "clause_b_spearman": rho,
        "power": {
            "n_held_out": n,
            "attainable_rho": attainable,
            "min_one_tailed_p": 1 / len(permutations),
        },
    }
    out = RESULTS / "tier2_sealed" / "tier2_prediction2.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
