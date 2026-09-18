# GenAI reference: I got GenAI to check the Tier 3 driver against preregistration_tier3.md (plan layout, pool guards, the B9dom domain 
# lookup) and the TrainConfig species_label_map validation

import json
from collections import Counter

import pytest

import experiments.training.tier3_train as t3
from animal_movement_fm.train.loop import TrainConfig

REGISTERED_COUNTS = {
    "Z9": 5, "B9": 5, "Za9": 15, "Zd3": 15, "Zd1": 15, "B9dom": 5, "LSTMS": 15, "LSTMB3": 5, "LSTMB9": 5,
    "Sk": 18, "F3c": 21, "F9": 21, "XGBS": 3, "XGBB3": 5, "XGBB9": 5,
}
PAYLOAD = json.loads(t3.REGISTERED_POOL.read_text())


def test_plan_matches_the_registered_arm_counts_and_order():
    assert len(t3.PLAN) == 158
    assert Counter(run.arm for run in t3.PLAN) == REGISTERED_COUNTS
    assert len({run.run_id for run in t3.PLAN}) == 158
    # contiguous blocks, in this order; the PBS header documents these indices
    order = list(dict.fromkeys(run.arm for run in t3.PLAN))
    assert order == list(REGISTERED_COUNTS)
    starts = {arm: next(i for i, run in enumerate(t3.PLAN) if run.arm == arm) for arm in order}
    assert starts == {
        "Z9": 0, "B9": 5, "Za9": 10, "Zd3": 25, "Zd1": 40, "B9dom": 55, "LSTMS": 60, "LSTMB3": 75,
        "LSTMB9": 80, "Sk": 85, "F3c": 103, "F9": 124, "XGBS": 145, "XGBB3": 148, "XGBB9": 153,
    }


def test_fine_tuned_arms_use_the_tier2_subsets_and_seed_zero():
    for arm in ("F3c", "F9"):
        runs = [run for run in t3.PLAN if run.arm == arm]
        assert {run.seed for run in runs} == {0}
        assert Counter(run.k_percent for run in runs) == {1: 9, 10: 9, 100: 3}
    sk = [run for run in t3.PLAN if run.arm == "Sk"]
    assert Counter(run.k_percent for run in sk) == {1: 9, 10: 9}
    # Sk trains on exactly the subsets F uses
    f9 = {(r.species, r.k_percent, r.draw) for r in t3.PLAN if r.arm == "F9" and r.k_percent != 100}
    assert {(r.species, r.k_percent, r.draw) for r in sk} == f9


def test_only_specialist_and_fine_tuned_arms_select_on_validation():
    selecting = {run.arm for run in t3.PLAN if run.uses_validation_selection}
    assert selecting == {"LSTMS", "Sk", "F3c", "F9"}
    assert not any(run.uses_validation_selection for run in t3.TIER_E_PLAN)


def test_registered_pool_matches_the_preregistered_percentages():
    got = {name: round(100 * p, 3) for name, p in PAYLOAD["source_pool"].items()}
    assert got == t3.REGISTERED_SOURCE_PERCENT
    assert set(t3.domain_of(PAYLOAD)) == set(t3.REGISTERED_SOURCE_PERCENT)


@pytest.mark.parametrize("stems", [["larus_fuscus"], ["canis_lupus"], ["mirounga_angustirostris"],
                                   ["anser_albifrons"], ["ovis_dalli"], ["pygoscelis_adeliae"],
                                   ["cervus_elaphus", "lynx_rufus"]])
def test_pool_guard_refuses_held_out_reserved_and_unregistered_species(stems):
    with pytest.raises(SystemExit):
        t3.assert_tier3_sources(stems)


def test_pool_guard_accepts_the_nine_sources():
    t3.assert_tier3_sources(t3.REGISTERED_SOURCE_PERCENT)


@pytest.mark.skipif("sub_pools" not in PAYLOAD, reason="sub_pools not registered yet")
def test_sub_pools_split_by_domain():
    domains = t3.domain_of(PAYLOAD)
    for domain in t3.DOMAINS:
        inside = PAYLOAD["sub_pools"][f"domain_{domain}"]
        outside = PAYLOAD["sub_pools"][f"without_{domain}"]
        assert {domains[name] for name in inside} == {domain} and len(inside) == 3
        assert domain not in {domains[name] for name in outside} and len(outside) == 6
        assert sum(inside.values()) == pytest.approx(1.0)
        assert sum(outside.values()) == pytest.approx(1.0)


def test_b9dom_routes_each_source_to_its_domain_row():
    names = sorted(t3.REGISTERED_SOURCE_PERCENT)
    config = t3.build_config(t3.Run("B9dom", "", 0), 11, 20_000, names, PAYLOAD)
    domains = t3.domain_of(PAYLOAD)
    assert config.n_species == 3 and config.species_embed_dim == 16
    assert not config.use_context_conditioning and not config.use_reconstruction_head
    assert [t3.DOMAINS[row] for row in config.species_label_map] == [domains[name] for name in names]
    config.build()


def test_registered_arm_configs():
    names = sorted(t3.REGISTERED_SOURCE_PERCENT)
    z9 = t3.build_config(t3.Run("Z9", "", 0), 11, 20_000, names, PAYLOAD)
    assert z9.use_fm_package and z9.use_reconstruction_head and z9.use_context_conditioning
    assert (z9.d_model, z9.num_layers, z9.lr, z9.steps) == (256, 4, 1e-3, 20_000)
    b9 = t3.build_config(t3.Run("B9", "", 0), 11, 20_000, names, PAYLOAD)
    assert not b9.use_fm_package and not b9.use_reconstruction_head
    lstm = t3.build_config(t3.Run("LSTMS", "canis_lupus", 0), 11, 20_000, ["canis_lupus"], PAYLOAD)
    assert lstm.trunk == "lstm" and (lstm.d_model, lstm.num_layers) == (64, 1)
    f9 = t3.build_config(t3.Run("F9", "canis_lupus", 0, 1, 0), 11, 20_000, ["canis_lupus"], PAYLOAD)
    assert (f9.lr, f9.steps) == (1e-3, 20_000) and f9.use_reconstruction_head


def test_species_label_map_is_validated():
    base = dict(d_model=64, num_layers=2, lr=1e-3, steps=10)
    # the registered arms carry no map, and a frozen dataclass must still construct
    assert TrainConfig(**base).species_label_map == ()
    with pytest.raises(ValueError):
        TrainConfig(**base, species_label_map=(0, 1))  # no embedding
    with pytest.raises(ValueError):
        TrainConfig(**base, species_embed_dim=16, n_species=3, species_label_map=(0, 3))
    # a list, as a saved record gives it back, is normalised to the tuple it was written from
    config = TrainConfig(**base, species_embed_dim=16, n_species=3, species_label_map=[2, 0])
    assert config.species_label_map == (2, 0)
