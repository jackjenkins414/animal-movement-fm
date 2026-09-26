"""
Derive the Tier 2 pretraining sampling weights from A8.

Species are sampled proportional to n^alpha, where n is training window count at the registered context length. We rederive counts  
from frozen manifests using the production windowing code and checked against registered guard values.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import id_key, window_counts_per_individual

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
RESULTS = REPO / "results"

ALPHA = 0.5  # fixed by A8

# segments stem, split stem, domain, role
ROLES: tuple[tuple[str, str, str, str], ...] = (
    ("ciconia_ciconia", "stork", "flying", "source"),
    ("larus_fuscus", "larus_fuscus", "flying", "held-out"),
    ("cervus_elaphus", "cervus_elaphus", "terrestrial", "source"),
    ("canis_lupus", "canis_lupus", "terrestrial", "held-out"),
    ("mirounga_leonina", "mirounga_leonina", "marine", "source"),
    ("mirounga_angustirostris", "mirounga_angustirostris", "marine", "held-out"),
)

# registered window count checks.
GUARDS: dict[str, dict[str, int]] = {
    "ciconia_ciconia": {"train": 806595, "all": 1064095},
    "larus_fuscus": {"train": 597407},
    "cervus_elaphus": {"train": 556579, "all": 593874},
    "canis_lupus": {"train": 47136, "all": 62013},
    "mirounga_leonina": {"train": 53076, "all": 55822},
    "mirounga_angustirostris": {"train": 43815, "all": 48587},
}


def species_window_counts(segments_stem: str, split_stem: str) -> dict[str, int]:
    """Return window counts for each frozen split of one species."""
    counts = window_counts_per_individual(
        str(PROCESSED / f"{segments_stem}_segments.parquet"),
        context_length=CONTEXT_LENGTH,
    )
    # GenAI reference: Match manifest ids to parquet ids through id_key, which is what SpeciesWindows.build uses when it 
    # assembles the training pool. Plain string equality silently scores a miss as zero windows, and the two spellings do 
    # differ: canonical_manifest_ids writes an id through a float parse, and Python reads underscores inside digits as
    # separators, so the RAATD id "146_2160" is written to the manifest as "1462160". Counted strictly, every such
    # individual disappeared from n_s. Corrected 20-09-2026; see preregistration_tier3.md sec 1.
    counts["individual_id"] = counts["individual_id"].map(id_key)
    per_individual = counts.set_index("individual_id")["n_windows"]

    manifest_path = SPLITS / f"{split_stem}_split_seed42.json"
    manifest = json.loads(manifest_path.read_text())

    out: dict[str, int] = {}

    for split in ("train", "val", "test"):
        ids = [id_key(i) for i in manifest[split]]
        windows = per_individual.reindex(ids).fillna(0)

        out[split] = int(windows.sum())
        out[f"n_{split}"] = len(ids)

    out["all"] = out["train"] + out["val"] + out["test"]
    return out


def draw_probabilities(
    counts: dict[str, int],
    alpha: float = ALPHA,
) -> dict[str, float]:
    "Add after"
    weights = {
    species: n_windows**alpha
    for species, n_windows in counts.items()}
    
    total = sum(weights.values())

    return {
        species: weight / total
        for species, weight in weights.items()}


def check_guards(table: pd.DataFrame) -> list[str]:
    """Return count mismatches."""
    failures = []
    indexed = table.set_index("species")

    for species, expected in GUARDS.items():
        if species not in indexed.index:
            continue

        for key, want in expected.items():
            got = int(indexed.loc[species, key])
            if got != want:
                failures.append(f"{species} {key} = {got}, expected {want}")

    return failures


# GenAI (16-09-2026): the Tier 3 mode below was written by GenAI to apply preregistration_tier3.md sec 1 mechanically
TIER3_FIXED = {"ciconia_ciconia": "flying", "cervus_elaphus": "terrestrial", "mirounga_leonina": "marine"}
TIER3_PROFILE_KEY = {"ciconia_ciconia": "white_stork"}  # the stork's profile row predates the binomial stems
TIER3_MIN_TRAIN_WINDOWS = 2_000  # preregistration_tier3.md sec 1, candidate condition 4
TIER3_OUT = REPO / "registered" / "tier3_source_pool.json"


def sub_pools_for(domains: dict[str, str], counts: dict[str, int]) -> dict:
    """
    A8 draw probabilities for the `domain_<domain>` and `without_<domain>` sub pools of one pool.

    `domains` maps every pool member to its domain; `counts` holds their training window counts. A single species
    sub pool (Tier 3's Zd1, Tier 4's carried Zd1) needs no entry, because its draw probability is 1.0.
    """
    pool = list(domains)
    pools = {}
    for domain in sorted(set(domains.values())):
        outside = [n for n in pool if domains[n] != domain]
        inside = [n for n in pool if domains[n] == domain]
        pools[f"without_{domain}"] = draw_probabilities({n: counts[n] for n in outside})
        pools[f"domain_{domain}"] = draw_probabilities({n: counts[n] for n in inside})
    return pools


def tier3_sub_pools(payload: dict) -> dict:
    """
    A8 draw probabilities for the Tier 3 sub pools, which we recompute from the registered counts alone.

    `without_<domain>` is Za9s pool (the 6 sources outside that domain); `domain_<domain>` is Zd3's (the three inside it). Zd1 is a single
    species and needs no entry.
    """
    domain_of = dict(payload["fixed"])
    for domain, added in payload["added"].items():
        domain_of.update({name: domain for name in added})
    return sub_pools_for({n: domain_of[n] for n in payload["pool"]}, payload["train_window_counts"])


def tier3(args: argparse.Namespace) -> None:
    """Select the nine Tier 3 sources under preregistration_tier3.md sec 1 and weight them under A8."""
    from animal_movement_fm.analysis.regime_profile import DESCRIPTORS, coverage_distances, select_source_pool

    role_stems = {stem for stem, *_ in ROLES}
    eligibility = pd.read_csv(RESULTS / "species_eligibility.csv")
    passing = set(eligibility.loc[eligibility["clears_gate1"].astype(bool), "species"])
    profiles = pd.read_csv(RESULTS / "regime_profiles.csv")
    complete = profiles[profiles[DESCRIPTORS].notna().all(axis=1)].set_index("species")

    counts: dict[str, int] = {}
    for stem in TIER3_FIXED:
        split_stem = next(split for seg, split, *_ in ROLES if seg == stem)
        counts[stem] = species_window_counts(stem, split_stem)["train"]
        if counts[stem] != GUARDS[stem]["train"]:
            raise SystemExit(f"FAIL: {stem} has {counts[stem]} training windows, registered {GUARDS[stem]['train']}")

    table = []
    candidates: dict[str, str] = {}
    for stem in sorted(complete.index):
        if stem in role_stems or stem in TIER3_PROFILE_KEY.values() or stem == "stork":
            continue
        manifest = SPLITS / f"{stem}_split_seed42.json"
        if stem not in passing or not manifest.exists():
            table.append({"species": stem, "domain": complete.loc[stem, "domain"], "reason": "fails A7 gate 1"})
            continue
        counts[stem] = species_window_counts(stem, stem)["train"]
        ok = counts[stem] >= TIER3_MIN_TRAIN_WINDOWS
        table.append({"species": stem, "domain": complete.loc[stem, "domain"], "train_windows": counts[stem],
                      "reason": "candidate" if ok else f"below {TIER3_MIN_TRAIN_WINDOWS} training windows"})
        if ok:
            candidates[stem] = str(complete.loc[stem, "domain"])

    keyed = complete.rename(index={v: k for k, v in TIER3_PROFILE_KEY.items()})
    rows = keyed.loc[list(TIER3_FIXED) + sorted(candidates)].reset_index()
    frozen = json.loads((REPO / "registered" / "tier2_descriptor_standardisation.json").read_text())
    distances = coverage_distances(rows, standardisation=frozen)

    result = select_source_pool(distances, TIER3_FIXED, candidates, counts)
    pool = result["pool"]
    weights = draw_probabilities({name: counts[name] for name in pool})

    print("candidates: " + ", ".join(f"{d} {sum(1 for v in candidates.values() if v == d)}"
                                      for d in ("flying", "terrestrial", "marine")))
    print(f"pools scored: {result['n_pools_scored']:,}; tied on mean: {result['n_tied_on_mean']}")
    print(f"selected, mean {result['mean_distance']:.4f}, min {result['min_distance']:.4f}:")
    domain_of = {**TIER3_FIXED, **candidates}
    for name in sorted(pool, key=lambda n: (domain_of[n], n)):
        print(f"  {domain_of[name]:12s} {name:26s} n={counts[name]:9,d}  p={weights[name] * 100:7.3f}%")
    print("runners up:")
    for item in result["runners_up"]:
        print(f"  mean {item['mean_distance']:.4f} min {item['min_distance']:.4f}  added {item['added']}")

    payload = {
        "_comment": [
            "Tier 3 source pool selected mechanically under preregistration_tier3.md sec 1, with A8 draw",
            "probabilities (alpha 0.5) over the selected nine. Written by source_pool_weights.py --tier3.",
            "Sub-pools are included: without_<domain> is Za9's six-source pool, domain_<domain> is Zd3's three,",
            "each weighted under A8 from the same counts. Written once, then verified on every later run.",
        ],
        "rule": "preregistration_tier3.md sec 1",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "min_train_windows": TIER3_MIN_TRAIN_WINDOWS,
        "standardisation": "registered/tier2_descriptor_standardisation.json",
        "fixed": TIER3_FIXED,
        "candidate_table": table,
        **{k: result[k] for k in ("pool", "added", "mean_distance", "min_distance", "train_windows",
                                  "n_pools_scored", "n_tied_on_mean", "candidates_per_domain",
                                  "distance_matrix", "runners_up")},
        "train_window_counts": {name: counts[name] for name in pool},
        "source_pool": weights,
    }

    if TIER3_OUT.exists():
        previous = json.loads(TIER3_OUT.read_text())
        same = previous["pool"] == pool and all(
            abs(previous["source_pool"][k] - weights[k]) <= 1e-12 for k in pool
        )
        if not same and not args.regenerate:
            raise SystemExit(
                f"FAIL: {TIER3_OUT.name} already holds a different selection; the rule is run once. Pass "
                "--regenerate only to correct a defect in the rule's inputs, and record it in the preregistration."
            )
        if not same:
            payload["regenerated"] = {
                "date": time.strftime("%Y-%m-%d"),
                "reason": args.regenerate,
                "supersedes": {
                    "pool": previous["pool"],
                    "source_pool": previous["source_pool"],
                    "train_window_counts": previous.get("train_window_counts"),
                    "sub_pools": previous.get("sub_pools"),
                },
            }
            payload["sub_pools"] = tier3_sub_pools(payload)
            if args.no_write:
                print("would regenerate; rerun without --no-write to write it")
                return
            TIER3_OUT.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"REGENERATED {TIER3_OUT.relative_to(REPO)}; the superseded selection is recorded inside it")
            for name, probs in payload["sub_pools"].items():
                print(f"  {name:22s} " + ", ".join(f"{n} {v * 100:.3f}%" for n, v in probs.items()))
            return
        print(f"reproduced {TIER3_OUT.relative_to(REPO)}")
        # GenAI reference: I wrote this originally, GenAI cleaned it; sub pools for Za9 and Zd3, added once to the registered file and verified thereafter
        sub_pools = tier3_sub_pools(previous)
        if "sub_pools" not in previous:
            if args.no_write:
                print("sub-pools not yet registered; rerun without --no-write to add them")
                return
            previous["sub_pools"] = sub_pools
            TIER3_OUT.write_text(json.dumps(previous, indent=2) + "\n")
            print(f"added {len(sub_pools)} sub-pools to {TIER3_OUT.relative_to(REPO)}")
        else:
            drift = [
                f"{pool}.{name}"
                for pool, probs in sub_pools.items()
                for name, value in probs.items()
                if abs(previous["sub_pools"].get(pool, {}).get(name, float("nan")) - value) > 1e-12
            ]
            if drift or set(previous["sub_pools"]) != set(sub_pools):
                raise SystemExit(f"FAIL: registered sub-pools no longer reproduce: {drift or 'pool names differ'}")
            print("sub-pools reproduced")
        for pool, probs in sub_pools.items():
            print(f"  {pool:22s} " + ", ".join(f"{n} {v * 100:.3f}%" for n, v in probs.items()))
        return
    if args.no_write:
        return
    TIER3_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {TIER3_OUT.relative_to(REPO)}")


# GenAI reference: based off my specification, the Tier 4 mode below, the pool derived from the eligibility gate and then asserted
# against the 26 in tier4_exploratory_freeze.md sec 1, the stork dedupe, A8 over the corrected id_key counts, the seven sub pools, 
# the pinned check percentages, and the refusal of held-out and reserved species. I used GenAI to draft this; copied it into this file;
# i reviewed and checked the code as well. 
TIER4_OUT = REPO / "registered" / "tier4_source_pool.json"

# tier4_exploratory_freeze.md sec 1: the pool is the eligible species minus these two groups
TIER4_HELD_OUT = frozenset(stem for stem, _, _, role in ROLES if role == "held-out")
TIER4_RESERVED = frozenset({"anser_albifrons", "ovis_dalli", "pygoscelis_adeliae"}) 

# `stork` and `ciconia_ciconia` are two extracts of one species and both clear gate 1, so the eligibility table holds
# two rows for it. The pool keys the species by its segments stem, `ciconia_ciconia`, which is what every registered
# artefact and tier3_train.py already use; the older `stork` row collapses onto it before the pool is derived.
TIER4_ELIGIBILITY_ALIAS = {"stork": "ciconia_ciconia"}
TIER4_PROFILE_KEY = dict(TIER3_PROFILE_KEY)  # the stork's profile row is still keyed `white_stork`
SPLIT_STEM = {"ciconia_ciconia": "stork"}  # the one source whose manifest stem differs from its segments stem

# the 26 sources of tier4_exploratory_freeze.md sec 1. The pool is derived from the eligibility gate, not read from
# here; this is the assertion the derivation has to match
TIER4_POOL: dict[str, str] = {
    "anser_caerulescens": "flying",
    "branta_canadensis": "flying",
    "cathartes_aura": "flying",
    "ciconia_ciconia": "flying",
    "corvus_corone": "flying",
    "diomedea_exulans": "flying",
    "falco_naumanni": "flying",
    "gyps_himalayensis": "flying",
    "haematopus_ostralegus": "flying",
    "milvus_milvus": "flying",
    "mycteria_americana": "flying",
    "pernis_apivorus": "flying",
    "thalassarche_chrysostoma": "flying",
    "thalassoica_antarctica": "flying",
    "bison_bison": "terrestrial",
    "canis_latrans": "terrestrial",
    "cervus_elaphus": "terrestrial",
    "connochaetes_taurinus": "terrestrial",
    "lynx_rufus": "terrestrial",
    "aptenodytes_patagonicus": "marine",
    "arctocephalus_gazella": "marine",
    "eudyptes_chrysolophus": "marine",
    "leptonychotes_weddellii": "marine",
    "lobodon_carcinophaga": "marine",
    "mirounga_leonina": "marine",
    "odobenus_rosmarus": "marine",
}

# ZdM-noMir and BdM-noMir (tier4_exploratory_freeze.md sec 2) draw from the marine sources without the held-out species' congener
TIER4_NO_MIROUNGA = "marine_no_mirounga"
TIER4_NO_MIROUNGA_EXCLUDES = "mirounga_leonina"

# registered A8 percentages, to 3 dp; the freeze prints the pool but not the probabilities, so these constants are the record, in the 
# same sense as tier3_train.REGISTERED_SOURCE_PERCENT. A mismatch is a failure, not a warning; an idea from GenAI that we adopt
TIER4_SOURCE_PERCENT: dict[str, float] = {
    "ciconia_ciconia": 14.949, "cervus_elaphus": 12.418, "cathartes_aura": 7.645, "anser_caerulescens": 7.439,
    "mycteria_americana": 6.182, "haematopus_ostralegus": 5.940, "corvus_corone": 5.568, "branta_canadensis": 5.256,
    "mirounga_leonina": 3.835, "connochaetes_taurinus": 3.774, "bison_bison": 3.233, "canis_latrans": 3.213,
    "falco_naumanni": 2.977, "milvus_milvus": 2.644, "pernis_apivorus": 2.497, "lynx_rufus": 2.366,
    "gyps_himalayensis": 1.633, "leptonychotes_weddellii": 1.462, "arctocephalus_gazella": 1.225,
    "diomedea_exulans": 0.970, "eudyptes_chrysolophus": 0.884, "thalassarche_chrysostoma": 0.856,
    "thalassoica_antarctica": 0.843, "odobenus_rosmarus": 0.826, "lobodon_carcinophaga": 0.796,
    "aptenodytes_patagonicus": 0.569,
}

TIER4_SUB_POOL_PERCENT: dict[str, dict[str, float]] = {
    "domain_flying": {
        "ciconia_ciconia": 22.858, "cathartes_aura": 11.689, "anser_caerulescens": 11.375,
        "mycteria_americana": 9.453, "haematopus_ostralegus": 9.083, "corvus_corone": 8.514,
        "branta_canadensis": 8.037, "falco_naumanni": 4.553, "milvus_milvus": 4.043, "pernis_apivorus": 3.818,
        "gyps_himalayensis": 2.496, "diomedea_exulans": 1.483, "thalassarche_chrysostoma": 1.309,
        "thalassoica_antarctica": 1.289,
    },
    "domain_terrestrial": {
        "cervus_elaphus": 49.664, "connochaetes_taurinus": 15.095, "bison_bison": 12.930,
        "canis_latrans": 12.849, "lynx_rufus": 9.463,
    },
    "domain_marine": {
        "mirounga_leonina": 39.961, "leptonychotes_weddellii": 15.238, "arctocephalus_gazella": 12.763,
        "eudyptes_chrysolophus": 9.208, "odobenus_rosmarus": 8.603, "lobodon_carcinophaga": 8.293,
        "aptenodytes_patagonicus": 5.933,
    },
    "marine_no_mirounga": {
        "leptonychotes_weddellii": 25.381, "arctocephalus_gazella": 21.258, "eudyptes_chrysolophus": 15.337,
        "odobenus_rosmarus": 14.329, "lobodon_carcinophaga": 13.813, "aptenodytes_patagonicus": 9.882,
    },
    "without_flying": {
        "cervus_elaphus": 35.890, "mirounga_leonina": 11.083, "connochaetes_taurinus": 10.908,
        "bison_bison": 9.344, "canis_latrans": 9.285, "lynx_rufus": 6.838, "leptonychotes_weddellii": 4.226,
        "arctocephalus_gazella": 3.540, "eudyptes_chrysolophus": 2.554, "odobenus_rosmarus": 2.386,
        "lobodon_carcinophaga": 2.300, "aptenodytes_patagonicus": 1.646,
    },
    "without_terrestrial": {
        "ciconia_ciconia": 19.933, "cathartes_aura": 10.194, "anser_caerulescens": 9.920,
        "mycteria_americana": 8.244, "haematopus_ostralegus": 7.921, "corvus_corone": 7.424,
        "branta_canadensis": 7.009, "mirounga_leonina": 5.113, "falco_naumanni": 3.970, "milvus_milvus": 3.526,
        "pernis_apivorus": 3.330, "gyps_himalayensis": 2.177, "leptonychotes_weddellii": 1.950,
        "arctocephalus_gazella": 1.633, "diomedea_exulans": 1.293, "eudyptes_chrysolophus": 1.178,
        "thalassarche_chrysostoma": 1.141, "thalassoica_antarctica": 1.124, "odobenus_rosmarus": 1.101,
        "lobodon_carcinophaga": 1.061, "aptenodytes_patagonicus": 0.759,
    },
    "without_marine": {
        "ciconia_ciconia": 16.536, "cervus_elaphus": 13.736, "cathartes_aura": 8.456, "anser_caerulescens": 8.229,
        "mycteria_americana": 6.839, "haematopus_ostralegus": 6.571, "corvus_corone": 6.159,
        "branta_canadensis": 5.814, "connochaetes_taurinus": 4.175, "bison_bison": 3.576, "canis_latrans": 3.554,
        "falco_naumanni": 3.293, "milvus_milvus": 2.925, "pernis_apivorus": 2.762, "lynx_rufus": 2.617,
        "gyps_himalayensis": 1.806, "diomedea_exulans": 1.072, "thalassarche_chrysostoma": 0.947,
        "thalassoica_antarctica": 0.932,
    },
}

PERCENT_TOLERANCE = 1e-3  # the constants are 3 dp, so this is agreement to the last recorded digit


def tier4_pool(passing: set[str], profile_domains: dict[str, str]) -> dict[str, str]:
    """
    Derive the Tier 4 pool from the eligibility gate and assert it against TIER4_POOL.

    `passing` is the set of species clearing A7 gate 1; `profile_domains` maps species to the domain column of the
    profile table. The derivation is eligibility minus the three held-out and three reserved species, with the two
    stork rows collapsed first. Nothing here applies a window floor: tier4_exploratory_freeze.md sec 1 takes the
    pool as preregistration_tier3.md sec 2 registers it, and the 2,000 window floor is a Tier 3 candidate condition.
    """
    eligible = {TIER4_ELIGIBILITY_ALIAS.get(name, name) for name in passing}
    if len(eligible) != 32:
        raise SystemExit(f"FAIL: {len(eligible)} distinct eligible species after the stork alias; sec 1 expects 32")

    derived = eligible - TIER4_HELD_OUT - TIER4_RESERVED
    expected = set(TIER4_POOL)
    if derived != expected:
        missing = sorted(expected - derived)
        extra = sorted(derived - expected)
        raise SystemExit(
            "FAIL: the derived Tier 4 pool does not match tier4_exploratory_freeze.md sec 1.\n"
            f"  in the freeze but not derived: {missing}\n"
            f"  derived but not in the freeze: {extra}"
        )

    # A domain re-label (as on 15-09-2026, when the seabirds moved) would silently reshape every sub-pool, so the
    # profile table has to agree with the freeze rather than override it.
    drift = []
    for name, domain in sorted(TIER4_POOL.items()):
        key = TIER4_PROFILE_KEY.get(name, name)
        if key not in profile_domains:
            raise SystemExit(f"FAIL: {name} has no profile row (looked for {key!r})")
        if profile_domains[key] != domain:
            drift.append(f"{name}: profile says {profile_domains[key]}, the freeze says {domain}")
    if drift:
        raise SystemExit("FAIL: domain drift against tier4_exploratory_freeze.md sec 1\n  " + "\n  ".join(drift))

    return dict(TIER4_POOL)


def tier4_sub_pools(domains: dict[str, str], counts: dict[str, int]) -> dict:
    """The seven Tier 4 sub pools: `domain_*` and `without_*` for the three domains, plus `marine_no_mirounga`."""
    pools = sub_pools_for(domains, counts)
    marine = [n for n in domains if domains[n] == "marine" and n != TIER4_NO_MIROUNGA_EXCLUDES]
    pools[TIER4_NO_MIROUNGA] = draw_probabilities({n: counts[n] for n in marine})
    return pools


def check_percentages(label: str, weights: dict[str, float], registered: dict[str, float]) -> list[str]:
    """Return every disagreement between computed A8 probabilities and the registered percentages."""
    failures = []
    if set(weights) != set(registered):
        missing = sorted(set(registered) - set(weights))
        extra = sorted(set(weights) - set(registered))
        failures.append(f"{label} membership differs; missing {missing}, unexpected {extra}")
        return failures
    for name, value in sorted(weights.items()):
        got = value * 100
        if abs(got - registered[name]) > PERCENT_TOLERANCE:
            failures.append(f"{label}.{name} = {got:.3f}%, registered {registered[name]:.3f}%")
    return failures


def refuse_excluded_species(pools: dict[str, dict[str, float]]) -> None:
    """Fail if any pool or sub pool holds a held-out or reserved species. Checked, not assumed from construction."""
    banned = TIER4_HELD_OUT | TIER4_RESERVED
    hits = sorted(f"{label}.{name}" for label, probs in pools.items() for name in probs if name in banned)
    if hits:
        raise SystemExit(f"FAIL: held-out or reserved species in a Tier 4 pool: {hits}")


def tier4(args: argparse.Namespace) -> None:
    """Build the 26 source Tier 4 pool under tier4_exploratory_freeze.md sec 1 and weight it under A8."""
    eligibility = pd.read_csv(RESULTS / "species_eligibility.csv")
    passing = set(eligibility.loc[eligibility["clears_gate1"].astype(bool), "species"])
    profiles = pd.read_csv(RESULTS / "regime_profiles.csv").set_index("species")
    domains = tier4_pool(passing, profiles["domain"].to_dict())

    counts: dict[str, int] = {}
    for stem in sorted(domains):
        counts[stem] = species_window_counts(stem, SPLIT_STEM.get(stem, stem))["train"]
        if stem in GUARDS and counts[stem] != GUARDS[stem]["train"]:
            raise SystemExit(f"FAIL: {stem} has {counts[stem]} training windows, registered {GUARDS[stem]['train']}")

    weights = draw_probabilities(counts)
    sub_pools = tier4_sub_pools(domains, counts)

    refuse_excluded_species({"pool": weights, **sub_pools})

    failures = check_percentages("source_pool", weights, TIER4_SOURCE_PERCENT)
    for label, registered in sorted(TIER4_SUB_POOL_PERCENT.items()):
        failures += check_percentages(label, sub_pools[label], registered)
    if failures:
        raise SystemExit(
            "FAIL: A8 probabilities do not match the registered Tier 4 percentages\n  " + "\n  ".join(failures)
        )

    print(f"C={CONTEXT_LENGTH}; alpha={ALPHA}; {len(domains)} sources, {sum(counts.values()):,} training windows")
    for domain in ("flying", "terrestrial", "marine"):
        members = [n for n in domains if domains[n] == domain]
        share = sum(weights[n] for n in members)
        print(f"\n{domain} ({len(members)}), {share * 100:.3f}% of draws")
        for name in sorted(members, key=lambda n: -counts[n]):
            print(f"  {name:26s} n={counts[name]:9,d}  p={weights[name] * 100:7.3f}%")
    for label in sorted(sub_pools):
        probs = sub_pools[label]
        print(f"\n{label} ({len(probs)})")
        for name in sorted(probs, key=lambda n: -probs[n]):
            print(f"  {name:26s} p={probs[name] * 100:7.3f}%")

    payload = {
        "_comment": [
            "Tier 4 source pool: every eligible species except the three held-out and the three reserved, as",
            "preregistration_tier3.md sec 2 registers it and tier4_exploratory_freeze.md sec 1 records it. No window",
            "floor. A8 draw probabilities (alpha 0.5) over the 26, with the seven sub-pools each renormalised over",
            "their own members. Written by source_pool_weights.py --tier4, once, then verified on every later run.",
        ],
        "rule": "tier4_exploratory_freeze.md sec 1",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "min_train_windows": None,
        "pool": sorted(domains),
        "domains": domains,
        "train_window_counts": counts,
        "source_pool": weights,
        "sub_pools": sub_pools,
    }

    if TIER4_OUT.exists():
        previous = json.loads(TIER4_OUT.read_text())
        same = previous["pool"] == payload["pool"] and all(
            abs(previous["source_pool"][k] - weights[k]) <= 1e-12 for k in weights
        )
        if not same and not args.regenerate:
            raise SystemExit(
                f"FAIL: {TIER4_OUT.name} already holds a different pool; the rule is run once. Pass --regenerate "
                "only to correct a defect in its inputs, and record it in tier4_exploratory_freeze.md."
            )
        if not same:
            payload["regenerated"] = {
                "date": time.strftime("%Y-%m-%d"),
                "reason": args.regenerate,
                "supersedes": {
                    "pool": previous["pool"],
                    "source_pool": previous["source_pool"],
                    "train_window_counts": previous.get("train_window_counts"),
                    "sub_pools": previous.get("sub_pools"),
                },
            }
            if args.no_write:
                print("\nwould regenerate; rerun without --no-write to write it")
                return
            TIER4_OUT.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"\nREGENERATED {TIER4_OUT.relative_to(REPO)}; the superseded pool is recorded inside it")
            return
        drift = [
            f"sub_pools.{label}.{name}"
            for label, probs in sub_pools.items()
            for name, value in probs.items()
            if abs(previous.get("sub_pools", {}).get(label, {}).get(name, float("nan")) - value) > 1e-12
        ]
        if drift or set(previous.get("sub_pools", {})) != set(sub_pools):
            raise SystemExit(f"FAIL: registered Tier 4 sub-pools no longer reproduce: {drift or 'pool names differ'}")
        print(f"\nreproduced {TIER4_OUT.relative_to(REPO)}")
        return

    if args.no_write:
        print(f"\n{TIER4_OUT.relative_to(REPO)} does not exist; rerun without --no-write to write it")
        return
    TIER4_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {TIER4_OUT.relative_to(REPO)}")
# GenAI code ends here. 

# GenAI reference: GenAI wrote part of this below (the main framework I copied myself from the above and modified for the bird fm;
# GenAI cleaned and then added 2 of the functions and fixed spelling in my latin names)
BIRDS_OUT = REPO / "registered" / "birds_source_pool.json"
BIRDS_RESULTS = RESULTS / "birds"  # gate and profile tables for the 14 species first extracted for this study

BIRDS_HELD_OUT = frozenset({"larus_fuscus", "anser_albifrons", "streptopelia_turtur"})
# preregistration_tier3.md sec 2 reserves three species; the goose is released for this study only
BIRDS_RESERVED = TIER4_RESERVED - {"anser_albifrons"}
BIRDS_ELIGIBLE_FLYING = 30

# the pools are derived from the gate, not read from here; these are the assertions the derivation has to match
BIRDS_CONGENERS = frozenset(
    {"larus_argentatus", "anser_caerulescens", "anser_fabalis", "anser_anser", "anser_brachyrhynchus"}
)
BIRDS_MAIN_POOL = frozenset({
    "ciconia_ciconia", "cathartes_aura", "haematopus_ostralegus", "mycteria_americana", "corvus_corone",
    "branta_canadensis", "falco_naumanni", "pernis_apivorus", "milvus_milvus", "gyps_himalayensis",
    "diomedea_exulans", "thalassarche_chrysostoma", "thalassoica_antarctica", "anas_platyrhynchos",
    "branta_leucopsis", "hydroprogne_caspia", "pelecanus_occidentalis", "falco_tinnunculus", "phoenicopterus_roseus",
    "platalea_leucorodia", "morus_bassanus", "ardea_alba",
})

# sec 3: A. albifrons training birds with >= 100 fixes duplicated in a val or test bird, left out of every goose training
# set. Recorded in the registered file so the driver reads its exclusions from there
BIRDS_TRAINING_EXCLUSIONS: dict[str, tuple[int, ...]] = {
    "anser_albifrons": (490431939, 662388807, 1056306620, 662388728, 133992693, 133992680),
}

# registered A8 percentages, to 3 dp. Pinned 26-09-2026 from the reviewed --no-write dry run; while either is empty,
# --birds refuses to write the registered file
BIRDS_MAIN_PERCENT: dict[str, float] = {
    "ciconia_ciconia": 17.150, "cathartes_aura": 8.770, "mycteria_americana": 7.093, "haematopus_ostralegus": 6.815,
    "corvus_corone": 6.388, "branta_canadensis": 6.030, "hydroprogne_caspia": 5.849, "platalea_leucorodia": 5.570,
    "phoenicopterus_roseus": 5.174, "anas_platyrhynchos": 4.847, "branta_leucopsis": 4.476, "falco_naumanni": 3.416,
    "milvus_milvus": 3.033, "pernis_apivorus": 2.865, "ardea_alba": 2.782, "falco_tinnunculus": 2.377,
    "gyps_himalayensis": 1.873, "pelecanus_occidentalis": 1.602, "diomedea_exulans": 1.112,
    "thalassarche_chrysostoma": 0.982, "thalassoica_antarctica": 0.967, "morus_bassanus": 0.829,
}
BIRDS_CONGENER_PERCENT: dict[str, float] = {
    "ciconia_ciconia": 13.324, "larus_argentatus": 8.402, "cathartes_aura": 6.814, "anser_caerulescens": 6.631,
    "mycteria_americana": 5.510, "haematopus_ostralegus": 5.294, "corvus_corone": 4.963, "branta_canadensis": 4.685,
    "hydroprogne_caspia": 4.544, "anser_anser": 4.474, "platalea_leucorodia": 4.328, "phoenicopterus_roseus": 4.020,
    "anas_platyrhynchos": 3.765, "branta_leucopsis": 3.477, "falco_naumanni": 2.654, "milvus_milvus": 2.357,
    "pernis_apivorus": 2.226, "ardea_alba": 2.161, "anser_fabalis": 2.040, "falco_tinnunculus": 1.847,
    "gyps_himalayensis": 1.455, "pelecanus_occidentalis": 1.245, "diomedea_exulans": 0.864,
    "thalassarche_chrysostoma": 0.763, "anser_brachyrhynchus": 0.763, "thalassoica_antarctica": 0.751,
    "morus_bassanus": 0.644,
}


def genus(stem: str) -> str:
    """The genus of a binomial segments stem, e.g. `anser` for `anser_fabalis`."""
    return stem.split("_", 1)[0]


def refuse_birds_excluded(pools: dict) -> None:
    """Fail if any pool holds a held-out bird or a still-reserved species. Checked, not assumed from construction."""
    banned = BIRDS_HELD_OUT | BIRDS_RESERVED
    hits = sorted(f"{label}.{name}" for label, members in pools.items() for name in members if name in banned)
    if hits:
        raise SystemExit(f"FAIL: held-out or reserved species in a bird pool: {hits}")


def birds_pools(passing_tables: list[set[str]], profile_domains: dict[str, str]) -> dict[str, list[str]]:
    """
    Derive the main (22) and congener (27) bird pools and assert them against the registered lists.

    `passing_tables` holds the gate-1 species of each eligibility table; `profile_domains` maps profile keys to the
    domain column. Eligible flying birds, minus the held-out three, minus every other member of their genera, is the
    main pool; the congener pool adds those congeners back.
    """
    eligible: set[str] = set()
    for table in passing_tables:
        aliased = {TIER4_ELIGIBILITY_ALIAS.get(name, name) for name in table}
        twice = sorted(eligible & aliased)
        if twice:
            raise SystemExit(f"FAIL: {twice} pass the gate in more than one eligibility table")
        eligible |= aliased

    missing = sorted(n for n in eligible if TIER4_PROFILE_KEY.get(n, n) not in profile_domains)
    if missing:
        raise SystemExit(f"FAIL: no profile row for gate-1 species {missing}")
    flying = {n for n in eligible if profile_domains[TIER4_PROFILE_KEY.get(n, n)] == "flying"}
    if len(flying) != BIRDS_ELIGIBLE_FLYING:
        raise SystemExit(
            f"FAIL: {len(flying)} eligible flying birds after the stork alias; preregistration_birds.md expects "
            f"{BIRDS_ELIGIBLE_FLYING}"
        )

    held_out_genera = {genus(n) for n in BIRDS_HELD_OUT}
    congeners = {n for n in flying - BIRDS_HELD_OUT if genus(n) in held_out_genera}
    main = flying - BIRDS_HELD_OUT - congeners
    for label, derived, expected in (("congeners", congeners, BIRDS_CONGENERS), ("main pool", main, BIRDS_MAIN_POOL)):
        if derived != expected:
            raise SystemExit(
                f"FAIL: the derived {label} does not match preregistration_birds.md sec 3.\n"
                f"  registered but not derived: {sorted(expected - derived)}\n"
                f"  derived but not registered: {sorted(derived - expected)}"
            )

    pools = {"main": sorted(main), "congener": sorted(main | congeners)}
    if set(pools["congener"]) - set(pools["main"]) != BIRDS_CONGENERS or len(pools["congener"]) != 27:
        raise SystemExit("FAIL: the congener pool is not the main pool plus exactly the five congeners")
    refuse_birds_excluded(pools)
    return pools


def check_counts_against_tier4(counts: dict[str, int], registered: dict[str, int]) -> int:
    """Every source shared with Tier 4 must reproduce its registered training-window count. Returns how many matched."""
    shared = sorted(set(counts) & set(registered))
    drift = [f"{n}: {counts[n]} here, {registered[n]} in {TIER4_OUT.name}" for n in shared if counts[n] != registered[n]]
    if drift:
        raise SystemExit("FAIL: training-window counts differ from the registered Tier 4 pool\n  " + "\n  ".join(drift))
    return len(shared)


def register_birds(payload: dict, args: argparse.Namespace) -> None:
    """Write `registered/birds_source_pool.json` once, and verify it on every later run."""
    if BIRDS_OUT.exists():
        previous = json.loads(BIRDS_OUT.read_text())
        same = (
            previous["pools"] == payload["pools"]
            and previous.get("training_exclusions") == payload["training_exclusions"]
            and all(
                abs(previous["probabilities"][label][name] - value) <= 1e-12
                for label, probs in payload["probabilities"].items()
                for name, value in probs.items()
            )
        )
        if same:
            print(f"\nreproduced {BIRDS_OUT.relative_to(REPO)}")
            return
        if not args.regenerate:
            raise SystemExit(
                f"FAIL: {BIRDS_OUT.name} already holds a different pool; the rule is run once. Pass --regenerate "
                "only to correct a defect in its inputs, and record it in preregistration_birds.md."
            )
        payload = payload | {
            "regenerated": {
                "date": time.strftime("%Y-%m-%d"),
                "reason": args.regenerate,
                "supersedes": {
                    key: previous.get(key)
                    for key in ("pools", "probabilities", "train_window_counts", "training_exclusions")
                },
            }
        }

    if args.no_write:
        print(f"\n--no-write: {BIRDS_OUT.relative_to(REPO)} not written")
        return
    if not (BIRDS_MAIN_PERCENT and BIRDS_CONGENER_PERCENT):
        raise SystemExit(
            "FAIL: BIRDS_MAIN_PERCENT and BIRDS_CONGENER_PERCENT are not pinned. Review the --no-write dry run and pin "
            "them before the registered file is written."
        )
    BIRDS_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {BIRDS_OUT.relative_to(REPO)}")


def birds(args: argparse.Namespace) -> None:
    """Build the two bird pools under preregistration_birds.md sec 3 and weight each under A8."""
    passing_tables = []
    for path in (RESULTS / "species_eligibility.csv", BIRDS_RESULTS / "species_eligibility.csv"):
        table = pd.read_csv(path)
        passing_tables.append(set(table.loc[table["clears_gate1"].astype(bool), "species"]))
    profile_domains: dict[str, str] = {}
    for path in (RESULTS / "regime_profiles.csv", BIRDS_RESULTS / "regime_profiles.csv"):
        profiles = pd.read_csv(path)
        for name, domain in zip(profiles["species"], profiles["domain"]):
            if profile_domains.get(name, domain) != domain:
                raise SystemExit(f"FAIL: {name} has conflicting domains across the two profile tables")
            profile_domains[name] = domain
    pools = birds_pools(passing_tables, profile_domains)

    counts: dict[str, int] = {}
    for stem in pools["congener"]:
        counts[stem] = species_window_counts(stem, SPLIT_STEM.get(stem, stem))["train"]
        if stem in GUARDS and counts[stem] != GUARDS[stem]["train"]:
            raise SystemExit(f"FAIL: {stem} has {counts[stem]} training windows, registered {GUARDS[stem]['train']}")
    matched = check_counts_against_tier4(counts, json.loads(TIER4_OUT.read_text())["train_window_counts"])

    weights = {label: draw_probabilities({n: counts[n] for n in members}) for label, members in pools.items()}
    refuse_birds_excluded(weights)
    failures = []
    for label, registered in (("main", BIRDS_MAIN_PERCENT), ("congener", BIRDS_CONGENER_PERCENT)):
        if registered:
            failures += check_percentages(label, weights[label], registered)
    if failures:
        raise SystemExit("FAIL: A8 probabilities do not match the registered bird percentages\n  " + "\n  ".join(failures))

    print(f"C={CONTEXT_LENGTH}; alpha={ALPHA}; {matched} sources reproduce their Tier 4 training-window counts")
    for label, probs in weights.items():
        print(f"\n{label} ({len(probs)}), {sum(counts[n] for n in probs):,} training windows")
        for name in sorted(probs, key=lambda n: -counts[n]):
            marker = "  (congener)" if name in BIRDS_CONGENERS else ""
            print(f"  {name:26s} n={counts[name]:9,d}  p={probs[name] * 100:7.3f}%{marker}")
    for species, ids in BIRDS_TRAINING_EXCLUSIONS.items():
        print(f"\ntraining exclusions, {species}: {', '.join(map(str, ids))}")

    payload = {
        "_comment": [
            "Bird-only study pools (preregistration_birds.md sec 3): every eligible flying bird except the three",
            "held out and their congeners (main, 22), and the same plus the five congeners (congener, 27). No window",
            "floor. A8 draw probabilities (alpha 0.5), renormalised within each pool. training_exclusions lists the",
            "individuals left out of every training set for that species. Written by source_pool_weights.py --birds,",
            "once, then verified on every later run.",
        ],
        "rule": "preregistration_birds.md sec 3",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "min_train_windows": None,
        "held_out": sorted(BIRDS_HELD_OUT),
        "congeners": sorted(BIRDS_CONGENERS),
        "pools": pools,
        "train_window_counts": counts,
        "probabilities": weights,
        "training_exclusions": {species: list(ids) for species, ids in BIRDS_TRAINING_EXCLUSIONS.items()},
    }
    register_birds(payload, args)
# GenAI code ends here.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate",
        metavar="REASON",
        default=None,
        help=(
            "rewrite an existing registered selection because its inputs were wrong. The reason is stored in the "
            "file beside the selection it supersedes; the preregistration must record the correction too."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print only; do not write results",
    )
    parser.add_argument(
        "--tier3",
        action="store_true",
        help="select the nine Tier 3 sources (preregistration_tier3.md sec 1) and write registered/tier3_source_pool.json",
    )
    parser.add_argument(
        "--tier4",
        action="store_true",
        help="build the 26 source Tier 4 pool (tier4_exploratory_freeze.md sec 1) and write registered/tier4_source_pool.json",
    )
    parser.add_argument(
        "--birds",
        action="store_true",
        help="build the bird-only study's two pools (preregistration_birds.md sec 3) and write "
             "registered/birds_source_pool.json",
    )
    args = parser.parse_args()
    if args.tier3 + args.tier4 + args.birds > 1:
        raise SystemExit("FAIL: pass only one of --tier3, --tier4 or --birds")
    if args.birds:
        birds(args)
        return
    if args.tier3:
        tier3(args)
        return
    if args.tier4:
        tier4(args)
        return

    rows = []
    for segments_stem, split_stem, domain, role in ROLES:
        counts = species_window_counts(segments_stem, split_stem)
        rows.append(
            {
                "species": segments_stem,
                "domain": domain,
                "role": role,
                "n_train": counts["n_train"],
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
                "all": counts["all"],
            }
        )

    # GenAI reference: used to polish reporting/output code below
    table = pd.DataFrame(rows)

    failures = check_guards(table)
    if failures:
        raise SystemExit("FAIL: guard mismatch\n  " + "\n  ".join(failures))

    sources = table[table["role"] == "source"]
    source_counts = dict(zip(sources["species"], sources["train"].astype(int)))

    full = draw_probabilities(source_counts)
    sub_pools = {
        f"without_{held}": draw_probabilities(
            {s: n for s, n in source_counts.items() if s != held}
        )
        for held in source_counts
    }

    pd.set_option("display.width", 200)

    print(f"C={CONTEXT_LENGTH}; alpha={ALPHA}")
    print()
    print(table.to_string(index=False))

    print("\nsource pool:")
    for species, p in full.items():
        print(f"  {species:24s} n={source_counts[species]:8d}  p={p * 100:7.3f}%")

    print("\nsub-pools:")
    for name, probs in sub_pools.items():
        inner = ", ".join(f"{s} {p * 100:.3f}%" for s, p in probs.items())
        print(f"  {name:32s} {inner}")

    print(f"\nguards: {sum(len(v) for v in GUARDS.values())} passed")

    # GenAI catch: the training scripts now read the tracked registered/tier2_source_pool_weights.json, not
    # the file this script writes; the recomputed weights must still reproduce it exactly
    registered = json.loads((REPO / "registered" / "tier2_source_pool_weights.json").read_text())
    drift = [
        f"{pool}.{species}: {value!r} vs registered {registered_pool[species]!r}"
        for pool, computed_pool, registered_pool in [("source_pool", full, registered["source_pool"])]
        + [(f"sub_pools.{k}", v, registered["sub_pools"][k]) for k, v in sub_pools.items()]
        for species, value in computed_pool.items()
        if abs(value - registered_pool[species]) > 1e-12
    ]
    if drift:
        raise SystemExit("FAIL: recomputed weights differ from the registered file\n  " + "\n  ".join(drift))
    print("matches registered/tier2_source_pool_weights.json")

    if args.no_write:
        return

    RESULTS.mkdir(exist_ok=True)

    payload = {
        "registered_by": "preregistration_tier2.md amendment A8",
        "context_length": CONTEXT_LENGTH,
        "alpha": ALPHA,
        "tunable": False,
        "counts": table.to_dict(orient="records"),
        "source_pool": full,
        "sub_pools": sub_pools,
    }

    out = RESULTS / "source_pool_weights.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()