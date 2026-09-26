# GenAI reference: will add later but the list was pool sizes, congener membership, the stork thing, refusal of the held-out birds
# and the two still-reserved species, and the 27 pool being the 22 pool plus exactly the five congeners; FINISH LATER

from __future__ import annotations

import argparse
import json
import sys

import pytest

import experiments.data.source_pool_weights as spw

BIRDS = spw.BIRDS_MAIN_POOL | spw.BIRDS_CONGENERS | spw.BIRDS_HELD_OUT

# the 14 species first extracted for the bird study; the rest of BIRDS sit in the original eligibility table
NEW = {
    "larus_argentatus", "anas_platyrhynchos", "branta_leucopsis", "streptopelia_turtur", "hydroprogne_caspia",
    "anser_fabalis", "anser_anser", "pelecanus_occidentalis", "falco_tinnunculus", "phoenicopterus_roseus",
    "platalea_leucorodia", "morus_bassanus", "ardea_alba", "anser_brachyrhynchus",
}
NON_BIRDS = {"cervus_elaphus": "terrestrial", "ovis_dalli": "terrestrial", "canis_lupus": "terrestrial",
             "mirounga_leonina": "marine", "pygoscelis_adeliae": "marine"}

# stand-in gate tables: the original one also holds the duplicate stork row and non-flying species, penguin included
OLD = (BIRDS - NEW) | {"stork"} | set(NON_BIRDS)
PROFILE_DOMAINS = {spw.TIER4_PROFILE_KEY.get(n, n): "flying" for n in BIRDS} | NON_BIRDS


def pools(old=OLD, new=NEW, profiles=PROFILE_DOMAINS):
    return spw.birds_pools([set(old), set(new)], dict(profiles))


# --------------------------------------------------------------------------- membership

def test_pool_sizes():
    derived = pools()
    assert len(derived["main"]) == 22
    assert len(derived["congener"]) == 27
    assert set(derived["main"]) == spw.BIRDS_MAIN_POOL


def test_the_congener_pool_is_the_main_pool_plus_exactly_the_five():
    derived = pools()
    assert set(derived["main"]) < set(derived["congener"])
    assert set(derived["congener"]) - set(derived["main"]) == spw.BIRDS_CONGENERS
    assert spw.BIRDS_CONGENERS == {
        "larus_argentatus", "anser_caerulescens", "anser_fabalis", "anser_anser", "anser_brachyrhynchus"
    }


def test_congeners_share_a_held_out_genus_and_no_main_source_does():
    held_out_genera = {spw.genus(n) for n in spw.BIRDS_HELD_OUT}
    assert {spw.genus(n) for n in spw.BIRDS_CONGENERS} <= held_out_genera
    assert not {spw.genus(n) for n in spw.BIRDS_MAIN_POOL} & held_out_genera


@pytest.mark.parametrize("old", [OLD, OLD - {"stork"}], ids=["with the stork row", "without it"])
def test_stork_is_one_species_keyed_by_its_segments_stem(old):
    derived = pools(old=old)
    for members in derived.values():
        assert members.count("ciconia_ciconia") == 1
        assert "stork" not in members


def test_a_penguin_that_clears_the_gate_is_not_a_bird_source():
    derived = pools()
    assert "pygoscelis_adeliae" in OLD
    assert all("pygoscelis_adeliae" not in members for members in derived.values())


# --------------------------------------------------------------------------- failures

def test_a_flying_bird_without_a_profile_row_fails():
    profiles = dict(PROFILE_DOMAINS)
    del profiles["ardea_alba"]
    with pytest.raises(SystemExit, match="no profile row"):
        pools(profiles=profiles)


def test_losing_an_eligible_bird_trips_the_count():
    with pytest.raises(SystemExit, match="eligible flying birds"):
        pools(new=NEW - {"ardea_alba"})


def test_an_extra_eligible_bird_trips_the_count():
    profiles = PROFILE_DOMAINS | {"grus_grus": "flying"}
    with pytest.raises(SystemExit, match="eligible flying birds"):
        pools(new=NEW | {"grus_grus"}, profiles=profiles)


def test_a_swapped_bird_fails_against_the_registered_list():
    profiles = PROFILE_DOMAINS | {"grus_grus": "flying"}
    with pytest.raises(SystemExit, match="main pool does not match"):
        pools(new=(NEW - {"ardea_alba"}) | {"grus_grus"}, profiles=profiles)


def test_a_new_congener_fails_against_the_registered_list():
    # a sixth Anser displacing a main-pool bird keeps the count at 30 but changes the congener set
    profiles = PROFILE_DOMAINS | {"anser_cygnoides": "flying"}
    with pytest.raises(SystemExit, match="congeners does not match"):
        pools(new=(NEW - {"ardea_alba"}) | {"anser_cygnoides"}, profiles=profiles)


def test_a_species_passing_in_both_tables_fails():
    with pytest.raises(SystemExit, match="more than one eligibility table"):
        pools(old=OLD | {"ardea_alba"})


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize(
    "name", ["larus_fuscus", "anser_albifrons", "streptopelia_turtur", "ovis_dalli", "pygoscelis_adeliae"]
)
@pytest.mark.parametrize("label", ["main", "congener"])
def test_held_out_and_still_reserved_species_are_refused_in_either_pool(name, label):
    members = {"main": sorted(spw.BIRDS_MAIN_POOL), "congener": sorted(spw.BIRDS_MAIN_POOL | spw.BIRDS_CONGENERS)}
    members[label] = members[label] + [name]
    with pytest.raises(SystemExit, match="held-out or reserved"):
        spw.refuse_birds_excluded(members)


def test_the_refusal_also_reads_probability_dicts():
    with pytest.raises(SystemExit, match="congener.larus_fuscus"):
        spw.refuse_birds_excluded({"congener": {"ciconia_ciconia": 0.9, "larus_fuscus": 0.1}})


def test_the_goose_is_released_and_only_the_goose():
    assert spw.BIRDS_RESERVED == {"ovis_dalli", "pygoscelis_adeliae"}
    assert "anser_albifrons" in spw.TIER4_RESERVED  # still reserved for Tier 4


def test_the_six_goose_exclusions_are_recorded():
    ids = spw.BIRDS_TRAINING_EXCLUSIONS["anser_albifrons"]
    assert sorted(ids) == sorted([490431939, 662388807, 1056306620, 662388728, 133992693, 133992680])
    assert set(spw.BIRDS_TRAINING_EXCLUSIONS) == {"anser_albifrons"}


# --------------------------------------------------------------------------- counts and weights

def test_counts_must_reproduce_the_tier4_counts_for_shared_sources():
    registered = {"ciconia_ciconia": 806_595, "cathartes_aura": 210_939, "cervus_elaphus": 556_579}
    assert spw.check_counts_against_tier4({"ciconia_ciconia": 806_595, "ardea_alba": 10}, registered) == 1
    with pytest.raises(SystemExit, match="cathartes_aura"):
        spw.check_counts_against_tier4({"cathartes_aura": 210_938}, registered)


def test_each_pool_sums_to_one():
    derived = pools()
    counts = {name: 1_000 * (i + 1) for i, name in enumerate(sorted(derived["congener"]))}
    for members in derived.values():
        assert sum(spw.draw_probabilities({n: counts[n] for n in members}).values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- write once

def _payload(main=("ciconia_ciconia", "ardea_alba")):
    return {
        "pools": {"main": list(main), "congener": [*main, "anser_fabalis"]},
        "probabilities": {"main": {n: 1 / len(main) for n in main},
                          "congener": {n: 1 / (len(main) + 1) for n in [*main, "anser_fabalis"]}},
        "training_exclusions": {"anser_albifrons": [490431939]},
    }


def _args(no_write=False, regenerate=None):
    return argparse.Namespace(no_write=no_write, regenerate=regenerate)


@pytest.fixture
def out(tmp_path, monkeypatch):
    path = tmp_path / "birds_source_pool.json"
    monkeypatch.setattr(spw, "BIRDS_OUT", path)
    monkeypatch.setattr(spw, "REPO", tmp_path)
    return path


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(spw, "BIRDS_MAIN_PERCENT", {"ciconia_ciconia": 50.0})
    monkeypatch.setattr(spw, "BIRDS_CONGENER_PERCENT", {"ciconia_ciconia": 33.333})


def test_the_pinned_percentages_cover_exactly_each_pool_and_sum_to_100():
    assert set(spw.BIRDS_MAIN_PERCENT) == spw.BIRDS_MAIN_POOL
    assert set(spw.BIRDS_CONGENER_PERCENT) == spw.BIRDS_MAIN_POOL | spw.BIRDS_CONGENERS
    for registered in (spw.BIRDS_MAIN_PERCENT, spw.BIRDS_CONGENER_PERCENT):
        assert sum(registered.values()) == pytest.approx(100.0, abs=0.01)  # 3 dp rounding only


def test_writing_is_refused_while_the_percentages_are_unpinned(out, monkeypatch):
    monkeypatch.setattr(spw, "BIRDS_MAIN_PERCENT", {})
    monkeypatch.setattr(spw, "BIRDS_CONGENER_PERCENT", {})
    with pytest.raises(SystemExit, match="not pinned"):
        spw.register_birds(_payload(), _args())
    assert not out.exists()


def test_no_write_never_writes(out, pinned):
    spw.register_birds(_payload(), _args(no_write=True))
    assert not out.exists()


def test_the_file_is_written_once_then_reproduced(out, pinned, capsys):
    spw.register_birds(_payload(), _args())
    written = out.read_text()
    spw.register_birds(_payload(), _args())
    assert out.read_text() == written
    assert "reproduced" in capsys.readouterr().out


def test_a_different_pool_is_refused_without_regenerate(out, pinned):
    spw.register_birds(_payload(), _args())
    with pytest.raises(SystemExit, match="already holds a different pool"):
        spw.register_birds(_payload(main=("ciconia_ciconia", "morus_bassanus")), _args())


def test_changed_exclusions_are_a_different_pool(out, pinned):
    spw.register_birds(_payload(), _args())
    changed = _payload() | {"training_exclusions": {"anser_albifrons": []}}
    with pytest.raises(SystemExit, match="already holds a different pool"):
        spw.register_birds(changed, _args())


def test_regenerate_records_what_it_supersedes(out, pinned):
    spw.register_birds(_payload(), _args())
    spw.register_birds(_payload(main=("ciconia_ciconia", "morus_bassanus")), _args(regenerate="test"))
    record = json.loads(out.read_text())
    assert record["regenerated"]["reason"] == "test"
    assert record["regenerated"]["supersedes"]["pools"]["main"] == ["ciconia_ciconia", "ardea_alba"]


def test_check_percentages_catches_a_drifted_probability():
    assert spw.check_percentages("main", {"a": 0.5, "b": 0.5}, {"a": 50.0, "b": 50.0}) == []
    assert spw.check_percentages("main", {"a": 0.5, "b": 0.5}, {"a": 50.01, "b": 49.99})


# --------------------------------------------------------------------------- CLI

@pytest.mark.parametrize("other", ["--tier3", "--tier4"])
def test_birds_is_refused_alongside_another_mode(monkeypatch, other):
    monkeypatch.setattr(sys, "argv", ["source_pool_weights.py", "--birds", other, "--no-write"])
    with pytest.raises(SystemExit, match="only one of"):
        spw.main()
