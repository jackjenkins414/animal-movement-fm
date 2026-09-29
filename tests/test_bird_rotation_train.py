# GenAI reference: I listed the plan layout and its indices, fold pools read from the registered file, no held-out bird 
# or genus mate in its own pool, exclusions applied to pooled and specialist data, and the pinned boundaries matching 
# the profile tables.

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest

import experiments.training.bird_rotation_train as rot
from animal_movement_fm.data.split import id_key

PAYLOAD = json.loads(rot.REGISTERED_FOLDS.read_text())
GRID = 11


@pytest.fixture(autouse=True)
def restore_tier2(monkeypatch):
    """bind_tier2_helpers() rebinds these on the shared Tier 2 module; registering them here restores them after each test."""
    for name in ("attached_entry", "boundary_m", "RESULTS", "CHECKPOINTS"):
        monkeypatch.setattr(rot.T, name, getattr(rot.T, name))


@pytest.fixture
def built(monkeypatch):
    """Stub the window builder: record which individuals each species was built from, and report registered counts."""
    calls: dict[str, list] = {}

    def fake(stem, split_stem, partition, individuals=None):
        assert partition == "train"
        calls[stem] = list(individuals)
        return SimpleNamespace(n_windows=PAYLOAD["train_window_counts"][stem])

    monkeypatch.setattr(rot.T, "_load_species_windows", fake)
    return calls


# --------------------------------------------------------------------------- plan

RANGES = {"Bb": (0, 26), "LSTMBb": (27, 53), "XGBBb": (54, 62), "LSTMS": (63, 143), "XGBS": (144, 170),
          "Zb": (171, 197)}


def test_the_plan_has_198_runs_and_unique_ids():
    assert len(rot.PLAN) == 198
    assert len({run.run_id for run in rot.PLAN}) == 198


@pytest.mark.parametrize("arm", list(RANGES))
def test_each_arm_holds_its_registered_index_range(arm):
    indices = [i for i, run in enumerate(rot.PLAN) if run.arm == arm]
    assert (indices[0], indices[-1]) == RANGES[arm]
    assert indices == list(range(RANGES[arm][0], RANGES[arm][1] + 1))


def test_zb_is_last():
    assert all(run.arm == "Zb" for run in rot.PLAN[171:])


def test_seeds_are_0_to_2_and_xgboost_is_seed_0():
    for run in rot.PLAN:
        assert run.seed in ((0,) if rot.ARMS[run.arm].get("xgb") else (0, 1, 2))
    counts = Counter((run.arm, run.fold, run.species) for run in rot.PLAN)
    assert set(counts.values()) <= {1, 3}


def test_pooled_runs_cover_folds_1_to_9_and_never_fold_0():
    for arm in ("Bb", "LSTMBb", "XGBBb", "Zb"):
        assert sorted({run.fold for run in rot.PLAN if run.arm == arm}) == list(range(1, 10))
        assert all(not run.species for run in rot.PLAN if run.arm == arm)


def test_specialists_cover_the_27_in_fold_order():
    lstms = [run for run in rot.PLAN if run.arm == "LSTMS"]
    assert [run.species for run in lstms[::3]] == list(rot.HELD_OUT)
    assert [run.fold for run in lstms] == sorted(run.fold for run in lstms)
    assert [run.species for run in rot.PLAN if run.arm == "XGBS"] == list(rot.HELD_OUT)
    assert not set(rot.HELD_OUT) & {"larus_fuscus", "anser_albifrons", "streptopelia_turtur"}


def test_run_ids():
    assert rot.PLAN[0].run_id == "Bb_f1_s0"
    assert rot.PLAN[75].run_id == "LSTMS_ciconia_ciconia_s0"
    assert rot.PLAN[197].run_id == "Zb_f9_s2"


def test_recipes_and_selection_come_from_the_bird_study():
    recipe = {arm: rot.BT.counterpart(next(r for r in rot.PLAN if r.arm == arm)).arm for arm in RANGES}
    assert recipe == {"Bb": "B26", "LSTMBb": "LSTMB26", "XGBBb": "XGBB26", "LSTMS": "LSTMS", "XGBS": "XGBS",
                      "Zb": "Z26"}
    assert {run.arm for run in rot.PLAN if run.uses_validation_selection} == {"LSTMS"}


@pytest.mark.parametrize("index", [0, 27, 63, 171])
def test_configs_build(index):
    run = rot.PLAN[index]
    config = rot.BT.build_config(run, GRID, rot.REGISTERED_STEPS, ["ardea_alba"], PAYLOAD)
    assert config.steps == 20_000 and config.seed == run.seed


# --------------------------------------------------------------------------- registered folds

def test_the_registered_file_loads_and_matches_the_pinned_constants():
    payload = rot.load_registered_folds()
    assert payload["train_window_counts"] == rot.SPW.BIRDS_ROTATION_TRAIN_WINDOWS


@pytest.fixture
def tampered(tmp_path, monkeypatch):
    def write(change):
        payload = json.loads(json.dumps(PAYLOAD))
        change(payload)
        path = tmp_path / "birds_rotation_folds.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(rot, "REGISTERED_FOLDS", path)
    return write


@pytest.mark.parametrize("change, message", [
    (lambda p: p["folds"]["5"].reverse() or p["folds"]["5"].append("x"), "folds"),
    (lambda p: p["train_window_counts"].update(ardea_alba=1), "counts differ"),
    (lambda p: p["training_exclusions"].pop("mycteria_americana"), "training_exclusions"),
    (lambda p: p["pools"]["3"].append("larus_fuscus"), "genus-mates-out"),
    (lambda p: p["probabilities"]["4"].update(ardea_alba=0.5), "no longer follow"),
])
def test_a_tampered_registered_file_is_refused(tampered, change, message):
    tampered(change)
    with pytest.raises(SystemExit, match=message):
        rot.load_registered_folds()


def test_pooled_runs_draw_their_fold_node_from_the_registered_file(built):
    for k in range(1, 10):
        pool, node = rot.pooled_source(rot.Run("Bb", k, "", 0), PAYLOAD)
        assert node == f"probabilities.{k}"
        assert pool.names == sorted(PAYLOAD["pools"][str(k)])
        weights = PAYLOAD["probabilities"][str(k)]
        assert dict(zip(pool.names, pool.probabilities)) == pytest.approx(weights)


def test_no_held_out_bird_or_genus_mate_in_its_own_pool(built):
    for k, members in rot.FOLDS.items():
        pool, _ = rot.pooled_source(rot.Run("Bb", k, "", 0), PAYLOAD)
        genera = {rot.SPW.genus(n) for n in members}
        assert not set(pool.names) & set(members)
        assert not {rot.SPW.genus(n) for n in pool.names} & genera
    assert "anser_albifrons" not in PAYLOAD["pools"]["1"] and "larus_fuscus" not in PAYLOAD["pools"]["3"]


@pytest.mark.parametrize("fold, intruder", [(2, "ciconia_ciconia"), (1, "anser_albifrons"), (3, "larus_fuscus"),
                                            (4, "ovis_dalli"), (5, "grus_grus")])
def test_the_source_guard_refuses_a_fold_member_genus_mate_reserved_or_unknown_bird(fold, intruder):
    with pytest.raises(SystemExit):
        rot.assert_fold_sources(fold, ["ardea_alba", intruder])


def test_pooled_runs_refuse_a_species_and_fold_0():
    with pytest.raises(SystemExit, match="pooled arm"):
        rot.pooled_source(rot.Run("Bb", 1, "ardea_alba", 0), PAYLOAD)
    with pytest.raises(SystemExit, match="fold 0"):
        rot.pooled_source(rot.Run("Bb", 0, "", 0), PAYLOAD)


# --------------------------------------------------------------------------- exclusions

EXCLUDED = {s: {id_key(i) for i in ids} for s, ids in PAYLOAD["training_exclusions"].items()}


@pytest.mark.parametrize("stem", sorted(EXCLUDED))
def test_training_ids_drop_the_registered_exclusions(stem):
    kept = {id_key(i) for i in rot.training_ids(stem, PAYLOAD)}
    manifest = rot.load_split(rot.T.SPLITS / f"{rot.SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json")
    assert not kept & EXCLUDED[stem]
    assert kept | EXCLUDED[stem] == {id_key(i) for i in manifest.train}


def test_an_exclusion_outside_the_training_partition_fails():
    payload = json.loads(json.dumps(PAYLOAD))
    payload["training_exclusions"]["ardea_alba"] = [123456789]
    with pytest.raises(SystemExit, match="not in ardea_alba's training partition"):
        rot.training_ids("ardea_alba", payload)


def test_pooled_data_is_built_without_the_exclusions(built):
    rot.pooled_source(rot.Run("Bb", 2, "", 0), PAYLOAD)  # fold 2 holds the goose, vulture, griffon and wood stork
    for stem, ids in EXCLUDED.items():
        if stem == "ciconia_ciconia":
            continue  # held out in fold 2
        assert stem in built and not {id_key(i) for i in built[stem]} & ids, stem


def test_specialist_data_is_built_without_the_exclusions(built):
    for species in ("ciconia_ciconia", "cathartes_aura", "gyps_himalayensis", "mycteria_americana"):
        pool, node = rot.held_out_source(rot.Run("LSTMS", rot.FOLD_OF[species], species, 0), PAYLOAD)
        assert node == f"specialist:{species}"
        assert not {id_key(i) for i in built[species]} & EXCLUDED[species]


def test_specialists_refuse_a_fold_0_bird_or_the_wrong_fold(built):
    with pytest.raises(SystemExit, match="not a rotation held-out bird"):
        rot.held_out_source(rot.Run("LSTMS", 0, "streptopelia_turtur", 0), PAYLOAD)
    with pytest.raises(SystemExit, match="held out in fold 2"):
        rot.held_out_source(rot.Run("LSTMS", 3, "ciconia_ciconia", 0), PAYLOAD)


def test_window_counts_are_checked_against_the_registered_file():
    good = SimpleNamespace(names=["ardea_alba"], species={"ardea_alba": SimpleNamespace(n_windows=21_218)})
    rot.check_window_counts(good, PAYLOAD)
    bad = SimpleNamespace(names=["ciconia_ciconia"], species={"ciconia_ciconia": SimpleNamespace(n_windows=806_595)})
    with pytest.raises(SystemExit, match="ciconia_ciconia"):
        rot.check_window_counts(bad, PAYLOAD)  # the count with the 44 still in


def test_records_carry_the_exclusions_and_the_exploratory_tag():
    record = rot.base_record(rot.PLAN[3], 3, GRID, "registered", "probabilities.2",
                             PAYLOAD["pools"]["2"], PAYLOAD)
    assert record["tier"] == "birds_rotation" and record["exploratory"] is True
    assert record["opened_test_partition"] is False
    assert set(record["excluded_individuals"]) == {"anser_albifrons", "cathartes_aura", "gyps_himalayensis",
                                                   "mycteria_americana"}


# --------------------------------------------------------------------------- boundaries

def test_the_pinned_boundaries_match_the_profile_tables():
    assert rot.check_boundaries() == rot.HELD_OUT_BOUNDARY_M


def test_stork_uses_its_registered_tier2_boundary():
    frozen = json.loads(rot.IL.REGISTERED_DECISION.read_text())
    assert rot.HELD_OUT_BOUNDARY_M["ciconia_ciconia"] == frozen["tier2_boundaries_m"]["white_stork"]
    assert rot.rotation_attached_entry("ciconia_ciconia") == ("ciconia_ciconia", "stork", "white_stork")


def test_a_drifted_boundary_is_refused(monkeypatch):
    monkeypatch.setitem(rot.HELD_OUT_BOUNDARY_M, "morus_bassanus", 1361.0)
    with pytest.raises(SystemExit, match="morus_bassanus"):
        rot.check_boundaries()


def test_binding_points_the_tier2_monitor_at_the_rotation():
    rot.bind_tier2_helpers()
    assert rot.T.attached_entry("ardea_alba") == ("ardea_alba", "ardea_alba", "ardea_alba")
    assert rot.T.boundary_m("white_stork") == rot.HELD_OUT_BOUNDARY_M["ciconia_ciconia"]
    assert rot.T.CHECKPOINTS == rot.CHECKPOINTS
    with pytest.raises(SystemExit):
        rot.T.attached_entry("larus_fuscus")
