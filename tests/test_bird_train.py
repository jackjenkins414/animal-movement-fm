# GenAI reference: I specified what the bird driver has to guarantee: the 95-run plan and its index ranges, the pools
# matching the pinned constants, the held-out and reserved refusals, the goose only ever a held-out target, etc etc; 
# GenAI wrote these tests from that list, mirroring my test_tier4_train.py.
import json
from collections import Counter
from dataclasses import asdict
from dataclasses import fields as dataclass_fields

import pandas as pd
import pytest

import experiments.training.bird_train as bt
import experiments.training.tier3_train as t3
import experiments.training.tier4_train as t4
from animal_movement_fm.train.loop import TrainConfig

GULL, GOOSE, DOVE = "larus_fuscus", "anser_albifrons", "streptopelia_turtur"
SIX = {490431939, 662388807, 1056306620, 662388728, 133992693, 133992680}

REGISTERED_COUNTS = {
    "Zb": 5, "Bb": 5, "Zbc": 5, "Bbc": 5, "LSTMBb": 5, "XGBBb": 5,
    "S": 10, "SB": 10, "Sk": 12, "LSTMS": 10, "XGBS": 2, "Fb": 21,
}
START_INDICES = {
    "Zb": 0, "Bb": 5, "Zbc": 10, "Bbc": 15, "LSTMBb": 20, "XGBBb": 25,
    "S": 30, "SB": 40, "Sk": 50, "LSTMS": 62, "XGBS": 72, "Fb": 74,
}
# the first index of every (arm, species) block, as the dry-run loop visits them
FIRST_INDICES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 56, 62, 67, 72, 73, 74, 81, 88]
FB_FIRST = [74, 81, 88]
PAYLOAD = json.loads(bt.REGISTERED_POOL.read_text())
GRID = 11


@pytest.fixture(autouse=True)
def restore_tier2(monkeypatch):
    """dry_run() rebinds these on the shared Tier 2 module; registering them here restores them after each test."""
    for name in ("attached_entry", "boundary_m", "RESULTS", "CHECKPOINTS"):
        monkeypatch.setattr(bt.T, name, getattr(bt.T, name))


class _StubWindows:
    def __init__(self, n_windows=1_000):
        self.n_windows = n_windows


@pytest.fixture
def loader(monkeypatch):
    """Stub the window loader, recording which individuals each call asks for, so no parquet is read."""
    calls = []

    def fake(stem, split_stem, partition, individuals=None):
        calls.append((stem, partition, None if individuals is None else {str(i) for i in individuals}))
        return _StubWindows()

    monkeypatch.setattr(bt.T, "_load_species_windows", fake)
    return calls


# Plan.
def test_plan_matches_the_registered_arm_counts_and_order():
    assert len(bt.PLAN) == 95
    assert Counter(run.arm for run in bt.PLAN) == REGISTERED_COUNTS
    assert len({run.run_id for run in bt.PLAN}) == 95
    assert list(dict.fromkeys(run.arm for run in bt.PLAN)) == list(REGISTERED_COUNTS)
    starts = {arm: next(i for i, run in enumerate(bt.PLAN) if run.arm == arm) for arm in REGISTERED_COUNTS}
    assert starts == START_INDICES


def test_the_first_index_of_every_block_is_the_dry_run_list():
    firsts, seen = [], set()
    for i, run in enumerate(bt.PLAN):
        if (run.arm, run.species) not in seen:
            seen.add((run.arm, run.species))
            firsts.append(i)
    assert firsts == FIRST_INDICES


def test_specialists_run_arm_by_arm_goose_then_dove():
    for arm in ("S", "SB", "Sk", "LSTMS", "XGBS"):
        species = [run.species for run in bt.PLAN if run.arm == arm]
        half = len(species) // 2
        assert species == [GOOSE] * half + [DOVE] * half, arm
    assert (bt.PLAN[72].species, bt.PLAN[73].species) == (GOOSE, DOVE)


def test_fb_is_last_in_gull_goose_dove_order():
    assert [run.arm for run in bt.PLAN[74:]] == ["Fb"] * 21
    assert [bt.PLAN[i].species for i in FB_FIRST] == [GULL, GOOSE, DOVE]
    for start, species in zip(FB_FIRST, (GULL, GOOSE, DOVE)):
        block = bt.PLAN[start:start + 7]
        assert {run.species for run in block} == {species}
        assert [(run.k_percent, run.draw) for run in block] == [
            (1, 0), (1, 1), (1, 2), (10, 0), (10, 1), (10, 2), (100, 0)
        ]
        assert {run.seed for run in block} == {bt.T.FINETUNE_INIT_SEED}


def test_sk_takes_the_one_and_ten_percent_draws_at_seed_0():
    block = [run for run in bt.PLAN if run.arm == "Sk" and run.species == DOVE]
    assert [(run.k_percent, run.draw) for run in block] == [(1, 0), (1, 1), (1, 2), (10, 0), (10, 1), (10, 2)]
    assert {run.seed for run in block} == {0}


def test_the_gull_appears_only_in_fb():
    assert {run.arm for run in bt.PLAN if run.species == GULL} == {"Fb"}
    assert {"S", "SB", "Sk", "LSTMS", "XGBS"} <= bt.CARRIED_GULL_ARMS


# Pools and guards.
def test_pooled_arms_draw_from_the_registered_nodes(loader):
    expected = {"Zb": "main", "Bb": "main", "LSTMBb": "main", "XGBBb": "main", "Zbc": "congener", "Bbc": "congener"}
    for arm, label in expected.items():
        pool, node = bt.pooled_source(bt.Run(arm, "", 0), PAYLOAD)
        assert node == f"probabilities.{label}", arm
        assert set(pool.names) == set(PAYLOAD["probabilities"][label]), arm
    assert len(PAYLOAD["probabilities"]["main"]) == 22 and len(PAYLOAD["probabilities"]["congener"]) == 27


def test_the_registered_pool_loads_and_matches_the_pinned_constants():
    payload = bt.load_registered_pool()
    assert set(payload["probabilities"]["main"]) == set(bt.SPW.BIRDS_MAIN_PERCENT)
    assert set(payload["probabilities"]["congener"]) == set(bt.SPW.BIRDS_CONGENER_PERCENT)
    assert {int(i) for i in payload["training_exclusions"][GOOSE]} == SIX


def _write(tmp_path, monkeypatch, payload):
    path = tmp_path / "birds_source_pool.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(bt, "REGISTERED_POOL", path)


def test_a_drifted_registered_percentage_is_refused(tmp_path, monkeypatch):
    payload = json.loads(bt.REGISTERED_POOL.read_text())
    payload["probabilities"]["main"]["ciconia_ciconia"] += 1e-4
    _write(tmp_path, monkeypatch, payload)
    with pytest.raises(SystemExit, match="main is"):
        bt.load_registered_pool()


def test_changed_exclusions_are_refused(tmp_path, monkeypatch):
    payload = json.loads(bt.REGISTERED_POOL.read_text())
    payload["training_exclusions"][GOOSE] = payload["training_exclusions"][GOOSE][:5]
    _write(tmp_path, monkeypatch, payload)
    with pytest.raises(SystemExit, match="training_exclusions"):
        bt.load_registered_pool()


@pytest.mark.parametrize("name", sorted(set(bt.HELD_OUT) | set(bt.SPW.BIRDS_RESERVED)))
def test_held_out_and_reserved_species_cannot_enter_a_pool(name):
    with pytest.raises(SystemExit, match="cannot enter a source pool"):
        bt.assert_bird_sources(set(PAYLOAD["probabilities"]["main"]) | {name})


def test_an_unregistered_species_cannot_enter_a_pool():
    with pytest.raises(SystemExit, match="not a registered bird source"):
        bt.assert_bird_sources({"grus_grus"})


def test_both_registered_nodes_pass_the_guard():
    for label in ("main", "congener"):
        bt.assert_bird_sources(PAYLOAD["probabilities"][label])


def test_a_pooled_arm_cannot_be_attached_to_the_goose(loader):
    with pytest.raises(SystemExit, match="pooled arm"):
        bt.pooled_source(bt.Run("Zb", GOOSE, 0), PAYLOAD)


@pytest.mark.parametrize("arm", ["S", "SB", "Sk", "LSTMS", "XGBS"])
@pytest.mark.parametrize("species", [GULL, "ciconia_ciconia", "anser_caerulescens"])
def test_specialists_train_only_on_the_goose_or_the_dove(arm, species, loader):
    k, d = (1, 0) if arm == "Sk" else (None, None)
    with pytest.raises(SystemExit, match="cannot train on"):
        bt.held_out_source(bt.Run(arm, species, 0, k, d), PAYLOAD)
    assert loader == []


def test_fb_trains_only_on_a_held_out_bird(loader):
    with pytest.raises(SystemExit, match="cannot train on"):
        bt.held_out_source(bt.Run("Fb", "ciconia_ciconia", 0, 1, 0), PAYLOAD)


# Goose exclusions and the shared subsets.
def test_the_goose_training_list_is_its_split_minus_the_six():
    ids = bt.held_out_train_ids(GOOSE, PAYLOAD)
    train = list(bt.load_split(bt.T.SPLITS / f"{GOOSE}_split_seed42.json").train)
    assert len(ids) == len(train) - 6 == 293
    assert not {int(i) for i in ids} & SIX
    assert SIX <= {int(i) for i in train}


@pytest.mark.parametrize("species", [GULL, DOVE])
def test_the_gull_and_dove_training_lists_are_their_whole_split(species):
    train = list(bt.load_split(bt.T.SPLITS / f"{species}_split_seed42.json").train)
    assert bt.held_out_train_ids(species, PAYLOAD) == train


def test_none_of_the_six_appears_in_any_goose_training_list(loader):
    goose_runs = [run for run in bt.PLAN if run.species == GOOSE]
    assert {run.arm for run in goose_runs} == {"S", "SB", "Sk", "LSTMS", "XGBS", "Fb"}
    for run in goose_runs:
        bt.held_out_source(run, PAYLOAD)
    assert len(loader) == len(goose_runs)
    for stem, partition, ids in loader:
        assert (stem, partition) == (GOOSE, "train")
        assert ids and not ids & {str(i) for i in SIX}


def test_an_exclusion_outside_the_training_partition_is_refused():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["training_exclusions"][GOOSE].append(123)
    with pytest.raises(SystemExit, match="not in anser_albifrons's training partition"):
        bt.held_out_train_ids(GOOSE, payload)


@pytest.mark.parametrize("species", [GOOSE, DOVE])
def test_fb_and_sk_train_on_identical_individuals_at_each_k_and_draw(species, loader):
    for k in bt.T3.SK_FRACTIONS:
        for d in range(bt.T.FINETUNE_DRAWS_PER_FRACTION[k]):
            fb = bt.finetune_subset(bt.Run("Fb", species, 0, k, d), PAYLOAD)
            sk = bt.finetune_subset(bt.Run("Sk", species, 0, k, d), PAYLOAD)
            assert list(fb.individuals) == list(sk.individuals), (k, d)
            loader.clear()
            bt.held_out_source(bt.Run("Fb", species, 0, k, d), PAYLOAD)
            bt.held_out_source(bt.Run("Sk", species, 0, k, d), PAYLOAD)
            assert loader[0] == loader[1], (k, d)


def test_the_gull_fb_subsets_are_tier3s_f9_and_sk_subsets():
    # the gull's Sk is carried from Tier 3, so a paired Fb-Sk comparison needs the same individuals on both sides
    for k, draws in bt.T.FINETUNE_DRAWS_PER_FRACTION.items():
        for d in range(draws):
            ours = bt.finetune_subset(bt.Run("Fb", GULL, 0, k, d), PAYLOAD)
            theirs = bt.T.finetune_subset(t3.Run("F9", GULL, 0, k, d))
            assert list(ours.individuals) == list(theirs.individuals), (k, d)


# Recipes against their counterparts.
T4_PAYLOAD = json.loads(t4.REGISTERED_POOL.read_text())
T3_PAYLOAD = json.loads(t3.REGISTERED_POOL.read_text())
MAIN = sorted(PAYLOAD["probabilities"]["main"])


def _bird(arm, species="", seed=0, k=None, d=None, names=MAIN):
    return bt.build_config(bt.Run(arm, species, seed, k, d), GRID, bt.REGISTERED_STEPS, names, PAYLOAD)


@pytest.mark.parametrize("ours,theirs", [("Zb", "Z26"), ("Zbc", "Z26"), ("Bb", "B26"), ("Bbc", "B26"),
                                         ("LSTMBb", "LSTMB26")])
def test_pooled_arms_match_their_tier4_counterparts(ours, theirs):
    for seed in bt.SEEDS:
        expected = t4.build_config(t4.Run(theirs, "", seed), GRID, t4.REGISTERED_STEPS, [], T4_PAYLOAD)
        assert _bird(ours, seed=seed) == expected


def test_zb_carries_the_package_and_bb_is_the_plain_trunk():
    zb, bb = _bird("Zb"), _bird("Bb")
    assert zb.use_fm_package and zb.use_reconstruction_head and zb.use_context_conditioning
    assert not bb.use_fm_package and not bb.use_reconstruction_head


@pytest.mark.parametrize("arm", ["S", "SB"])
def test_s_and_sb_match_their_tier2_recipes(arm):
    for species in (GOOSE, DOVE):
        for seed in bt.SEEDS:
            expected = bt.T.build_config(bt.T.Run(arm, GULL, seed), GRID, bt.REGISTERED_STEPS)
            assert _bird(arm, species, seed, names=[species]) == expected
    s, sb = _bird("S", GOOSE, names=[GOOSE]), _bird("SB", GOOSE, names=[GOOSE])
    assert s.use_fm_package and s.use_reconstruction_head and s.steps == 20_000
    assert not sb.use_fm_package and not sb.use_reconstruction_head and sb.steps == 20_000


@pytest.mark.parametrize("ours,theirs", [("Sk", "Sk"), ("Fb", "F9")])
def test_sk_and_fb_match_their_tier3_recipes(ours, theirs):
    for species in (GOOSE, DOVE):
        for k in bt.T3.SK_FRACTIONS:
            for d in range(3):
                expected = t3.build_config(t3.Run(theirs, GULL, 0, k, d), GRID, t3.REGISTERED_STEPS, [GULL], T3_PAYLOAD)
                assert _bird(ours, species, 0, k, d, names=[species]) == expected


def test_lstms_matches_tier3_lstms():
    for seed in bt.SEEDS:
        expected = t3.build_config(t3.Run("LSTMS", GULL, seed), GRID, t3.REGISTERED_STEPS, [GULL], T3_PAYLOAD)
        assert _bird("LSTMS", DOVE, seed, names=[DOVE]) == expected


def test_the_fb_donor_is_the_only_declared_difference_from_f9():
    assert bt.ARMS["Fb"]["init"] == "Zb_s0"
    assert t3.ARMS["F9"]["init"] == ("tier3", "Z9_s0")


def test_checkpoint_selection_follows_the_counterparts():
    selecting = {"S", "SB", "Sk", "LSTMS", "Fb"}
    for run in bt.PLAN:
        assert run.uses_validation_selection == (run.arm in selecting), run.run_id
        assert run.uses_validation_selection == bt.counterpart(run).uses_validation_selection


def test_xgb_rungs_sample_as_their_counterparts():
    assert bt.XGB_POOLED_DRAWS == t4.XGB_POOLED_DRAWS == t3.XGB_POOLED_DRAWS
    assert bt.ARMS["XGBBb"]["pool"] == "main" and bt.ARMS["XGBS"]["pool"] == "held_out_train"


# Fb's donor.
def test_fb_refuses_to_start_until_zb_seed_0_has_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "RESULTS", tmp_path)
    with pytest.raises(SystemExit, match="fine-tunes Zb_s0"):
        bt.initial_state(bt.PLAN[74], GRID)


# Boundaries and the rebound Tier 2 helpers.
def test_the_pinned_boundaries_reproduce_from_the_profile_tables():
    assert bt.check_boundaries() == bt.HELD_OUT_BOUNDARY_M
    assert set(bt.HELD_OUT_BOUNDARY_M) == set(bt.HELD_OUT)


def test_the_goose_boundary_reads_the_snapshot_not_the_shared_profile_table():
    # Gadi's results/regime_profiles.csv is older and has no goose row; tier2_sealed_eval.py reads it, so it stays as is
    assert bt.PROFILE_TABLES[GOOSE] == bt.SPW.BIRDS_RESULTS / "regime_profiles_main_snapshot.csv"
    assert bt.PROFILE_TABLES[GULL] == bt.SPW.BIRDS_RESULTS / "regime_profiles_main_snapshot.csv" 
    assert bt.IL.PROFILES_CSV not in bt.PROFILE_TABLES.values()
    assert bt.PROFILE_TABLES[DOVE] == bt.SPW.BIRDS_RESULTS / "regime_profiles.csv"


def test_the_boundary_check_passes_when_the_shared_table_has_no_goose_row(tmp_path, monkeypatch):
    # the Gadi situation: a shared table without the goose, and the goose's row only in the snapshot
    shared = pd.read_csv(bt.IL.PROFILES_CSV)
    without_goose, snapshot = tmp_path / "regime_profiles.csv", tmp_path / "regime_profiles_main_snapshot.csv"
    shared[shared["species"] != GOOSE].to_csv(without_goose, index=False)
    shared.to_csv(snapshot, index=False)
    monkeypatch.setitem(bt.PROFILE_TABLES, GULL, without_goose)
    monkeypatch.setitem(bt.PROFILE_TABLES, GOOSE, snapshot)
    assert bt.check_boundaries() == bt.HELD_OUT_BOUNDARY_M


def test_a_drifted_pinned_boundary_is_refused(monkeypatch):
    monkeypatch.setitem(bt.HELD_OUT_BOUNDARY_M, DOVE, 25.9)
    with pytest.raises(SystemExit, match="streptopelia_turtur boundary is pinned"):
        bt.check_boundaries()


def test_the_rebound_helpers_know_only_the_held_out_birds():
    assert bt.bird_attached_entry(DOVE) == (DOVE, DOVE, DOVE)
    assert bt.bird_boundary_m(GOOSE) == bt.HELD_OUT_BOUNDARY_M[GOOSE]
    with pytest.raises(SystemExit):
        bt.bird_attached_entry("canis_lupus")
    with pytest.raises(SystemExit):
        bt.bird_boundary_m("canis_lupus")


def test_binding_points_the_tier2_monitor_at_the_birds():
    bt.bind_tier2_helpers()
    assert bt.T.attached_entry is bt.bird_attached_entry
    assert bt.T.boundary_m is bt.bird_boundary_m
    assert (bt.T.RESULTS, bt.T.CHECKPOINTS) == (bt.RESULTS, bt.CHECKPOINTS)


# Partitions, records, frozen configuration.
def test_the_window_loader_refuses_a_test_partition():
    with pytest.raises(SystemExit, match="never opens a test partition"):
        bt.T._load_species_windows(DOVE, DOVE, "test")


def test_no_arm_trains_on_anything_but_training_individuals():
    assert {spec["pool"] for spec in bt.ARMS.values()} == {"main", "congener", "held_out_train", "held_out_subset"}


def test_records_are_tagged_birds_and_carry_the_exclusions():
    for index in (0, 30, 81):
        record = bt.base_record(bt.PLAN[index], index, GRID, "tier2_selection.json", "node", PAYLOAD)
        assert record["tier"] == "birds"
        assert record["registered_by"] == "preregistration_birds.md"
        assert record["exploratory"] is False
        assert record["opened_test_partition"] is False
    goose = bt.base_record(bt.PLAN[30], 30, GRID, "tier2_selection.json", "node", PAYLOAD)
    assert {int(i) for i in goose["excluded_individuals"]} == SIX
    assert bt.base_record(bt.PLAN[35], 35, GRID, "tier2_selection.json", "node", PAYLOAD)["excluded_individuals"] == []
    assert bt.RESULTS.name == bt.CHECKPOINTS.name == "birds"


def test_no_bird_config_adds_a_trainconfig_field():
    frozen = set(t4.FROZEN_TRAIN_CONFIG_FIELDS)
    assert {field.name for field in dataclass_fields(TrainConfig)} == frozen
    for run in bt.PLAN:
        if bt.ARMS[run.arm].get("xgb"):
            continue
        names = MAIN if not run.species else [run.species]
        assert set(asdict(bt.build_config(run, GRID, bt.REGISTERED_STEPS, names, PAYLOAD))) == frozen


# Dry run, with the window loader stubbed.
@pytest.mark.parametrize("index", [i for i in FIRST_INDICES if i not in FB_FIRST])
def test_dry_run_completes_for_the_first_index_of_every_block(index, loader, capsys):
    bt.dry_run(PAYLOAD, index)
    printed = capsys.readouterr().out
    run = bt.PLAN[index]
    assert run.run_id in printed and "dry run only" in printed and "pool           :" in printed
    if bt.ARMS[run.arm].get("xgb"):
        assert "WindowXGBoost" in printed and "config         :" not in printed
    else:
        assert "config         :" in printed
    assert ("excluded       :" in printed) == (run.species == GOOSE)


@pytest.mark.parametrize("index", FB_FIRST)
def test_dry_run_for_fb_refuses_without_its_donor(index, loader, tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "RESULTS", tmp_path)
    with pytest.raises(SystemExit, match="fine-tunes Zb_s0"):
        bt.dry_run(PAYLOAD, index)


def test_dry_run_over_every_pool_and_held_out_set(loader, capsys):
    bt.dry_run(PAYLOAD, None)
    printed = capsys.readouterr().out
    assert "probabilities.main" in printed and "probabilities.congener" in printed
    for species in bt.HELD_OUT:
        assert f"held_out:{species}" in printed
    goose_line = next(line for line in printed.splitlines() if line.startswith("held_out:anser_albifrons"))
    assert "293 individuals" in goose_line
