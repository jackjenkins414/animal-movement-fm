# GenAI reference: will finish comment later. 

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import fields as dataclass_fields

import pytest

import experiments.training.tier3_train as t3
import experiments.training.tier4_train as t4
from animal_movement_fm.train.loop import TrainConfig

REGISTERED_COUNTS = {
    "Z26": 5, "B26": 5, "Za26": 15, "ZdF": 5, "ZdT": 5, "ZdM": 5, "BdM": 5, "ZdMnoMir": 5, "BdMnoMir": 5,
    "B26dom": 5, "LSTMB26": 5, "XGBB26": 5, "F26": 21,
}
START_INDICES = {
    "Z26": 0, "B26": 5, "Za26": 10, "ZdF": 25, "ZdT": 30, "ZdM": 35, "BdM": 40, "ZdMnoMir": 45,
    "BdMnoMir": 50, "B26dom": 55, "LSTMB26": 60, "XGBB26": 65, "F26": 70,
}
PAYLOAD = json.loads(t4.REGISTERED_POOL.read_text())
NES = "mirounga_angustirostris"


def test_plan_matches_the_freeze_arm_counts_and_order():
    assert len(t4.PLAN) == 91
    assert Counter(run.arm for run in t4.PLAN) == REGISTERED_COUNTS
    assert len({run.run_id for run in t4.PLAN}) == 91
    order = list(dict.fromkeys(run.arm for run in t4.PLAN))
    assert order == list(REGISTERED_COUNTS)
    starts = {arm: next(i for i, run in enumerate(t4.PLAN) if run.arm == arm) for arm in order}
    assert starts == START_INDICES


def test_z26_is_first_and_f26_is_last():
    assert t4.PLAN[0].arm == "Z26" and t4.PLAN[0].seed == 0
    assert t4.PLAN[-1].arm == "F26"
    assert [run.arm for run in t4.PLAN[70:]] == ["F26"] * 21


def test_per_species_arms_run_in_gull_wolf_nes_order():
    for arm, start in (("Za26", 10), ("F26", 70)):
        block = [run for run in t4.PLAN if run.arm == arm]
        species = list(dict.fromkeys(run.species for run in block))
        assert species == list(t4.HELD_OUT)
        assert t4.PLAN[start].species == "larus_fuscus"


def test_domain_arms_are_pinned_to_one_species_each():
    expected = {"ZdF": "larus_fuscus", "ZdT": "canis_lupus", "ZdM": NES, "BdM": NES, "ZdMnoMir": NES, "BdMnoMir": NES}
    for arm, species in expected.items():
        runs = [run for run in t4.PLAN if run.arm == arm]
        assert {run.species for run in runs} == {species}, arm
        assert [run.seed for run in runs] == list(t4.SEEDS), arm


def test_f26_draws_follow_the_registered_fractions():
    block = [run for run in t4.PLAN if run.arm == "F26" and run.species == NES]
    assert [(run.k_percent, run.draw) for run in block] == [
        (1, 0), (1, 1), (1, 2), (10, 0), (10, 1), (10, 2), (100, 0)
    ]
    assert {run.seed for run in block} == {t4.T.FINETUNE_INIT_SEED}


# Pools and guards.
def test_every_arm_resolves_to_a_node_that_exists_in_the_registered_file():
    for arm in t4.ARMS:
        if t4.ARMS[arm]["pool"].startswith("held_out"):
            continue
        run = next(r for r in t4.PLAN if r.arm == arm)
        node = t4.pool_node(run)
        weights = PAYLOAD
        for part in node.split("."):
            assert part in weights, f"{arm}: {node}"
            weights = weights[part]
        assert weights, node


def test_arms_draw_from_the_nodes_the_freeze_names():
    expected = {
        "Z26": "source_pool", "B26": "source_pool", "B26dom": "source_pool",
        "LSTMB26": "source_pool", "XGBB26": "source_pool",
        "ZdF": "sub_pools.domain_flying", "ZdT": "sub_pools.domain_terrestrial",
        "ZdM": "sub_pools.domain_marine", "BdM": "sub_pools.domain_marine",
        "ZdMnoMir": "sub_pools.marine_no_mirounga", "BdMnoMir": "sub_pools.marine_no_mirounga",
    }
    for arm, node in expected.items():
        assert t4.pool_node(next(r for r in t4.PLAN if r.arm == arm)) == node, arm
    za = {sp: t4.pool_node(next(r for r in t4.PLAN if r.arm == "Za26" and r.species == sp)) for sp in t4.HELD_OUT}
    assert za == {
        "larus_fuscus": "sub_pools.without_flying",
        "canis_lupus": "sub_pools.without_terrestrial",
        NES: "sub_pools.without_marine",
    }


def test_the_marine_sub_pools_differ_by_mirounga_alone():
    marine = set(PAYLOAD["sub_pools"]["domain_marine"])
    no_mirounga = set(PAYLOAD["sub_pools"][t4.NO_MIROUNGA])
    assert marine - no_mirounga == {"mirounga_leonina"}
    assert not any(name.startswith("mirounga_") for name in no_mirounga)


@pytest.mark.parametrize("name", sorted(set(t4.HELD_OUT) | set(t4.RESERVED)))
def test_held_out_and_reserved_species_cannot_enter_a_pool(name):
    with pytest.raises(SystemExit, match="cannot enter a source pool"):
        t4.assert_tier4_sources(set(t4.REGISTERED_SOURCE_PERCENT) | {name})


def test_an_unregistered_species_cannot_enter_a_pool():
    with pytest.raises(SystemExit, match="not a registered Tier 4 source"):
        t4.assert_tier4_sources({"larus_argentatus"})


def test_every_registered_node_passes_the_guard():
    t4.assert_tier4_sources(PAYLOAD["source_pool"])
    for label, weights in PAYLOAD["sub_pools"].items():
        t4.assert_tier4_sources(weights), label


def test_the_registered_pool_loads_and_matches_the_constants():
    payload = t4.load_registered_pool()
    assert set(payload["source_pool"]) == set(t4.REGISTERED_SOURCE_PERCENT)
    assert set(payload["sub_pools"]) == set(t4.REGISTERED_SUB_POOL_PERCENT)
    assert set(payload["domains"]) == set(t4.REGISTERED_SOURCE_PERCENT)


def test_a_drifted_registered_percentage_is_refused(tmp_path, monkeypatch):
    payload = json.loads(t4.REGISTERED_POOL.read_text())
    payload["source_pool"]["ciconia_ciconia"] += 1e-4  # one hundredth of a percentage point
    path = tmp_path / "tier4_source_pool.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(t4, "REGISTERED_POOL", path)
    with pytest.raises(SystemExit, match="source_pool is"):
        t4.load_registered_pool()


# F26.
def test_f26_refuses_to_start_until_z26_seed_0_has_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(t4, "RESULTS", tmp_path)
    run = next(r for r in t4.PLAN if r.arm == "F26")
    with pytest.raises(SystemExit, match="fine-tunes Z26_s0"):
        t4.initial_state(run, 11)


def test_f26_uses_the_same_subsets_as_tier3_f9_and_sk():
    for species in t4.HELD_OUT:
        for k, draws in t4.T.FINETUNE_DRAWS_PER_FRACTION.items():
            for draw in range(draws):
                ours = t4.T.finetune_subset(t4.Run("F26", species, 0, k, draw))
                theirs = t4.T.finetune_subset(t3.Run("F9", species, 0, k, draw))
                assert list(ours.individuals) == list(theirs.individuals), (species, k, draw)
                assert ours.n_individuals == theirs.n_individuals


def test_f26_and_f9_configurations_are_identical_apart_from_the_donor():
    for species in t4.HELD_OUT:
        for k, draws in t4.T.FINETUNE_DRAWS_PER_FRACTION.items():
            for draw in range(draws):
                ours = t4.build_config(t4.Run("F26", species, 0, k, draw), 11, t4.REGISTERED_STEPS, [species], PAYLOAD)
                theirs = t3.build_config(
                    t3.Run("F9", species, 0, k, draw), 11, t3.REGISTERED_STEPS, [species],
                    json.loads(t3.REGISTERED_POOL.read_text()),
                )
                assert ours == theirs, (species, k, draw)
    # the donor is the only declared difference, and it is not part of TrainConfig
    assert t4.ARMS["F26"]["init"] == ("tier4", "Z26_s0")
    assert t3.ARMS["F9"]["init"] == ("tier3", "Z9_s0")


def test_f26_selects_checkpoints_on_validation_like_f9():
    assert t4.Run("F26", "canis_lupus", 0, 1, 0).uses_validation_selection
    assert t3.Run("F9", "canis_lupus", 0, 1, 0).uses_validation_selection
    # every pooled Tier 4 arm keeps the final checkpoint instead
    for arm in ("Z26", "B26", "Za26", "ZdF", "ZdM", "BdM", "ZdMnoMir", "BdMnoMir", "B26dom", "LSTMB26"):
        assert not t4.Run(arm, "", 0).uses_validation_selection, arm


# B26dom.
def test_b26dom_builds_a_26_species_domain_label_map():
    run = next(r for r in t4.PLAN if r.arm == "B26dom")
    names = sorted(PAYLOAD["source_pool"])
    config = t4.build_config(run, 11, t4.REGISTERED_STEPS, names, PAYLOAD)
    assert len(config.species_label_map) == 26
    assert config.n_species == 3 and config.species_embed_dim == 16
    assert config.use_context_conditioning is False and config.use_reconstruction_head is False
    expected = tuple(t4.DOMAINS.index(PAYLOAD["domains"][name]) for name in names)
    assert config.species_label_map == expected
    assert set(config.species_label_map) == {0, 1, 2}


def test_tier4_domain_rows_mean_what_tier3_domain_rows_mean():
    assert t4.DOMAINS == t3.DOMAINS == ("flying", "marine", "terrestrial")


def test_the_package_arms_and_trunk_arms_are_what_the_freeze_says():
    names = sorted(PAYLOAD["source_pool"])
    for arm in ("Z26", "ZdM", "ZdMnoMir"):
        config = t4.build_config(next(r for r in t4.PLAN if r.arm == arm), 11, t4.REGISTERED_STEPS, names, PAYLOAD)
        assert config.use_fm_package and config.use_reconstruction_head and config.use_context_conditioning, arm
    for arm in ("B26", "BdM", "BdMnoMir"):
        config = t4.build_config(next(r for r in t4.PLAN if r.arm == arm), 11, t4.REGISTERED_STEPS, names, PAYLOAD)
        assert not config.use_fm_package and not config.use_reconstruction_head, arm


# Carried arms, test partitions, frozen configuration.
def test_carried_arms_are_not_in_the_plan():
    for arm in ("S", "SB", "Sk", "Zd1", "LSTMS", "XGBS", "F9", "F3c", "P", "V", "ARIMA", "AVG3", "AVG10"):
        assert arm not in t4.ARMS, arm
        assert not any(run.arm == arm for run in t4.PLAN), arm
    assert set(t4.ARMS) & t4.CARRIED_ARMS == set()


def test_no_arm_opens_a_validation_or_test_partition_for_training():
    for arm, spec in t4.ARMS.items():
        assert spec["pool"] in {"source_pool", "without_domain", "domain", "no_mirounga", "held_out_subset"}, arm
    record = t4.base_record(t4.PLAN[0], 0, 11, "tier2_selection.json", "source_pool")
    assert record["opened_test_partition"] is False


def test_records_are_tagged_tier_4_and_exploratory():
    for index in (0, 45, 70):
        record = t4.base_record(t4.PLAN[index], index, 11, "tier2_selection.json", "source_pool")
        assert record["tier"] == "4"
        assert record["exploratory"] is True
        assert record["registered_by"] == "tier4_exploratory_freeze.md"


def test_trainconfig_fields_are_frozen_while_tier4_runs_are_in_flight():
    got = tuple(sorted(field.name for field in dataclass_fields(TrainConfig)))
    assert got == tuple(sorted(t4.FROZEN_TRAIN_CONFIG_FIELDS)), (
        "TrainConfig's fields have changed. They are frozen while Tier 4 runs are in flight: a new field changes "
        "asdict(config), and the resume guard at src/animal_movement_fm/train/loop.py:434 then refuses every "
        "checkpoint written before it, so any requeued subjob dies instead of resuming. Update "
        "tier4_train.FROZEN_TRAIN_CONFIG_FIELDS only after the last Tier 4 run has reported."
    )


# Dry run, with the window loader stubbed so no parquet is read.
class _StubWindows:
    def __init__(self, n_windows=1_000):
        self.n_windows = n_windows


@pytest.fixture
def stub_loader(monkeypatch):
    monkeypatch.setattr(t4.T, "_load_species_windows", lambda *a, **k: _StubWindows())
    return _StubWindows


@pytest.mark.parametrize("arm,index", [(arm, i) for arm, i in START_INDICES.items() if arm != "F26"])
def test_dry_run_completes_for_the_first_index_of_every_arm(arm, index, stub_loader, capsys):
    t4.dry_run(PAYLOAD, index)
    printed = capsys.readouterr().out
    assert t4.PLAN[index].run_id in printed
    assert "dry run only" in printed
    assert "pool           :" in printed
    if t4.ARMS[arm].get("xgb"):
        # the XGBoost rungs build no TrainConfig, so the dry run must not try to
        assert "WindowXGBoost" in printed and "config         :" not in printed
    else:
        assert "config         :" in printed


def test_dry_run_for_f26_refuses_without_its_donor(stub_loader, tmp_path, monkeypatch):
    monkeypatch.setattr(t4, "RESULTS", tmp_path)
    with pytest.raises(SystemExit, match="fine-tunes Z26_s0"):
        t4.dry_run(PAYLOAD, START_INDICES["F26"])


def test_dry_run_prints_the_lstm_geometry_not_the_grid_cell(stub_loader, capsys):
    t4.dry_run(PAYLOAD, START_INDICES["LSTMB26"])
    printed = capsys.readouterr().out
    assert f"d_model {t4.LSTM_D_MODEL}, layers {t4.LSTM_LAYERS}" in printed
    assert "trunk lstm" in printed


def test_dry_run_over_every_node_builds_each_registered_pool(stub_loader, capsys):
    t4.dry_run(PAYLOAD, None)
    printed = capsys.readouterr().out
    for node in ["source_pool"] + [f"sub_pools.{name}" for name in PAYLOAD["sub_pools"]]:
        assert node in printed
    assert "dry run only" in printed


# Configuration equality against the Tier 3 arms these mirror.
TIER3_PAYLOAD = json.loads(t3.REGISTERED_POOL.read_text())
T4_NAMES = sorted(PAYLOAD["source_pool"])
T3_NAMES = sorted(TIER3_PAYLOAD["source_pool"])


def _t4_config(arm, species=""):
    run = next(r for r in t4.PLAN if r.arm == arm)
    return t4.build_config(run, 11, t4.REGISTERED_STEPS, T4_NAMES, PAYLOAD)


def _t3_config(arm, species=""):
    return t3.build_config(t3.Run(arm, species, 0), 11, t3.REGISTERED_STEPS, T3_NAMES, TIER3_PAYLOAD)


def test_b26dom_matches_b9dom_apart_from_the_label_map():
    ours, theirs = _t4_config("B26dom"), _t3_config("B9dom")
    assert asdict(ours) | {"species_label_map": ()} == asdict(theirs) | {"species_label_map": ()}
    assert len(ours.species_label_map) == 26 and len(theirs.species_label_map) == 9


def test_lstmb26_matches_lstmb9_exactly():
    assert _t4_config("LSTMB26") == _t3_config("LSTMB9")


def test_zdm_matches_zd3_exactly():
    assert _t4_config("ZdM") == _t3_config("Zd3", "mirounga_angustirostris")


def test_bdm_matches_b9_exactly():
    assert _t4_config("BdM") == _t3_config("B9")


def test_z26_matches_z9_exactly():
    # the pool changes, the configuration does not: Tier 4 varies the sources and nothing else
    assert _t4_config("Z26") == _t3_config("Z9")
