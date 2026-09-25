# GenAI reference: I specified what the Tier 4 pool builder has to guarantee; that being membership per domain, the stork
# dedupe, refusal of held-out and reserved species, marine_no_mirounga holding no Mirounga, tier3_sub_pools still reproducing 
# the registered Tier 3 file from JSON alone; it also suggested each without_* excluding exactly its own domain and the pinned 
# percentages. GenAI wrote the tests from that list and I checked them.

from __future__ import annotations

import json

import pytest

import experiments.data.source_pool_weights as spw

TIER3_PAYLOAD = json.loads(spw.TIER3_OUT.read_text())

# A stand-in eligibility gate: the 26 pool members plus the held-out, the reserved and the duplicate stork row.
PASSING = set(spw.TIER4_POOL) | set(spw.TIER4_HELD_OUT) | set(spw.TIER4_RESERVED) | {"stork"}

# Profile domains for everything the derivation looks up. The stork's row is keyed `white_stork`.
PROFILE_DOMAINS = {
    spw.TIER4_PROFILE_KEY.get(name, name): domain for name, domain in spw.TIER4_POOL.items()
}

# Counts are arbitrary but distinct, so a sub-pool that silently dropped or duplicated a member changes its weights.
COUNTS = {name: 1_000 * (i + 1) for i, name in enumerate(sorted(spw.TIER4_POOL))}

DOMAIN_SIZES = {"flying": 14, "terrestrial": 5, "marine": 7}


def test_pool_membership_and_domain_sizes():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    assert len(domains) == 26
    sizes = {d: sum(1 for v in domains.values() if v == d) for d in DOMAIN_SIZES}
    assert sizes == DOMAIN_SIZES
    assert not set(domains) & (spw.TIER4_HELD_OUT | spw.TIER4_RESERVED)


def test_stork_is_one_species_keyed_by_its_segments_stem():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    assert "ciconia_ciconia" in domains
    assert "stork" not in domains
    assert spw.TIER4_ELIGIBILITY_ALIAS["stork"] == "ciconia_ciconia"
    # the manifest and the profile row keep their own spellings
    assert spw.SPLIT_STEM["ciconia_ciconia"] == "stork"
    assert spw.TIER4_PROFILE_KEY["ciconia_ciconia"] == "white_stork"


def test_pool_without_the_duplicate_stork_row_is_still_26():
    domains = spw.tier4_pool(PASSING - {"stork"}, PROFILE_DOMAINS)
    assert set(domains) == set(spw.TIER4_POOL)


@pytest.mark.parametrize("dropped", sorted(spw.TIER4_POOL)[:3])
def test_losing_an_eligible_species_trips_the_count_guard(dropped):
    with pytest.raises(SystemExit, match="distinct eligible species"):
        spw.tier4_pool(PASSING - {dropped}, PROFILE_DOMAINS)


@pytest.mark.parametrize("dropped", sorted(spw.TIER4_POOL)[:3])
def test_a_swapped_species_fails_against_the_freeze(dropped):
    # Count still 32, so this reaches the membership assertion rather than the count guard. 
    swapped = (PASSING - {dropped}) | {"larus_argentatus"}
    domains = {**PROFILE_DOMAINS, "larus_argentatus": "flying"}
    with pytest.raises(SystemExit, match="does not match"):
        spw.tier4_pool(swapped, domains)


@pytest.mark.parametrize("name", sorted(spw.TIER4_HELD_OUT | spw.TIER4_RESERVED))
def test_held_out_and_reserved_species_are_refused_in_any_pool(name):
    with pytest.raises(SystemExit, match="held-out or reserved"):
        spw.refuse_excluded_species({"pool": {**dict.fromkeys(spw.TIER4_POOL, 0.0), name: 0.1}})
    with pytest.raises(SystemExit, match="held-out or reserved"):
        spw.refuse_excluded_species({"domain_marine": {name: 1.0}})


def test_the_derived_pool_holds_no_held_out_or_reserved_species():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    sub_pools = spw.tier4_sub_pools(domains, COUNTS)
    spw.refuse_excluded_species({"pool": dict.fromkeys(domains, 0.0), **sub_pools})


def test_domain_drift_against_the_freeze_fails():
    moved = dict(PROFILE_DOMAINS)
    moved["thalassoica_antarctica"] = "marine"  
    with pytest.raises(SystemExit, match="domain drift"):
        spw.tier4_pool(PASSING, moved)


def test_marine_no_mirounga_holds_no_mirounga():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    sub_pools = spw.tier4_sub_pools(domains, COUNTS)
    members = set(sub_pools[spw.TIER4_NO_MIROUNGA])
    assert not any(name.startswith("mirounga_") for name in members)
    assert members == set(sub_pools["domain_marine"]) - {spw.TIER4_NO_MIROUNGA_EXCLUDES}
    assert len(members) == 6


@pytest.mark.parametrize("domain", sorted(DOMAIN_SIZES))
def test_without_sub_pool_excludes_exactly_its_own_domain(domain):
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    sub_pools = spw.tier4_sub_pools(domains, COUNTS)
    without = set(sub_pools[f"without_{domain}"])
    inside = {name for name, value in domains.items() if value == domain}
    assert not without & inside
    assert without == set(domains) - inside
    assert without | set(sub_pools[f"domain_{domain}"]) == set(domains)


def test_every_pool_and_sub_pool_sums_to_one():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    sub_pools = spw.tier4_sub_pools(domains, COUNTS)
    assert sum(spw.draw_probabilities(COUNTS).values()) == pytest.approx(1.0)
    for label, probs in sub_pools.items():
        assert sum(probs.values()) == pytest.approx(1.0), label


def test_the_seven_sub_pools_are_the_registered_ones():
    domains = spw.tier4_pool(PASSING, PROFILE_DOMAINS)
    sub_pools = spw.tier4_sub_pools(domains, COUNTS)
    assert set(sub_pools) == set(spw.TIER4_SUB_POOL_PERCENT)
    assert len(sub_pools) == 7


def test_check_percentages_accepts_the_registered_values_and_rejects_a_drifted_one():
    exact = {name: percent / 100 for name, percent in spw.TIER4_SOURCE_PERCENT.items()}
    assert spw.check_percentages("source_pool", exact, spw.TIER4_SOURCE_PERCENT) == []

    drifted = dict(exact)
    drifted["ciconia_ciconia"] += 0.01  # one percentage point
    failures = spw.check_percentages("source_pool", drifted, spw.TIER4_SOURCE_PERCENT)
    assert len(failures) == 1 and "ciconia_ciconia" in failures[0]

    missing = {k: v for k, v in exact.items() if k != "lynx_rufus"}
    assert "membership differs" in spw.check_percentages("source_pool", missing, spw.TIER4_SOURCE_PERCENT)[0]


def test_registered_percentages_cover_the_pool_and_each_sub_pool():
    assert set(spw.TIER4_SOURCE_PERCENT) == set(spw.TIER4_POOL)
    for label, percents in spw.TIER4_SUB_POOL_PERCENT.items():
        assert set(percents) <= set(spw.TIER4_POOL), label
        assert sum(percents.values()) == pytest.approx(100.0, abs=0.01), label


def test_reserved_species_agree_with_the_tier3_driver():
    # The two constants restate preregistration_tier3.md sec 2; this fails if one is edited alone. 
    pytest.importorskip("torch", reason="tier3_train imports the training stack")
    import experiments.training.tier3_train as t3

    assert spw.TIER4_RESERVED == t3.RESERVED


def test_tier3_sub_pools_still_reproduce_the_registered_file():
    # JSON only: no parquet is read, so this runs anywhere the repo is checked out. 
    rebuilt = spw.tier3_sub_pools(TIER3_PAYLOAD)
    registered = TIER3_PAYLOAD["sub_pools"]
    assert set(rebuilt) == set(registered)
    for label, probs in rebuilt.items():
        assert probs == pytest.approx(registered[label], abs=1e-12), label


def test_tier3_sub_pools_membership_is_unchanged_by_the_shared_helper():
    rebuilt = spw.tier3_sub_pools(TIER3_PAYLOAD)
    assert set(rebuilt) == set(TIER3_PAYLOAD["sub_pools"])
    for label, probs in rebuilt.items():
        assert set(probs) == set(TIER3_PAYLOAD["sub_pools"][label]), label
