# GenAI reference: I listed what these must cover and GenAI wrote tests from that; largely I said that the fold rule reproduces 
# the agreed table, genus blocks must stay intact, no bird or genus mate sits in its own fold's pool, and exclusions are applied 
# before windows are counted. 

from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd
import pytest

import experiments.data.source_pool_weights as spw

REGISTERED = json.loads(spw.BIRDS_OUT.read_text())
BIRDS = spw.rotation_birds(REGISTERED)
FOLDS = spw.rotation_folds(REGISTERED["train_window_counts"])
POOLS = {k: spw.rotation_pool(members, BIRDS) for k, members in FOLDS.items()}


# --------------------------------------------------------------------------- the rule

def test_the_rule_reproduces_the_pinned_folds():
    assert FOLDS == {k: sorted(members) for k, members in spw.BIRDS_ROTATION_FOLDS.items()}


def test_fold_zero_is_the_bird_study_held_out_three():
    assert set(FOLDS[0]) == spw.BIRDS_HELD_OUT


def test_every_bird_is_in_exactly_one_fold():
    placed = [n for members in FOLDS.values() for n in members]
    assert sorted(placed) == BIRDS
    assert len(placed) == 30


def test_fold_sizes():
    assert {k: len(m) for k, m in FOLDS.items()} == {0: 3, 1: 4, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3}


def test_genus_blocks_are_intact_outside_fold_zero():
    fold_of = {n: k for k, members in FOLDS.items() for n in members}
    for genus in {spw.genus(n) for n in BIRDS}:
        folds = {fold_of[n] for n in BIRDS if spw.genus(n) == genus} - {0}
        assert len(folds) <= 1, f"{genus} is split across folds {folds}"


def test_the_oversized_genus_is_its_own_fold():
    assert FOLDS[1] == ["anser_anser", "anser_brachyrhynchus", "anser_caerulescens", "anser_fabalis"]


def test_placement_is_largest_mass_first_into_the_lightest_fold_with_room():
    counts = {"a_x": 100, "b_x": 81, "c_x": 64, "d_x": 49, "e_x": 36, "f_x": 25}
    # masses 10, 9, 8, 7, 6, 5 into two folds of three: 10 | 9; 8 to the lighter (10 | 17); 7 (17 | 17); 6 on the tie to
    # the lower fold, which fills it (23 | 17); 5 to the other (23 | 22)
    folds = spw.rotation_folds(counts, fold0=())
    assert folds == {0: [], 1: ["a_x", "d_x", "e_x"], 2: ["b_x", "c_x", "f_x"]}


def test_a_block_larger_than_capacity_is_a_fold_of_its_own():
    counts = {f"big_{i}": 10 for i in range(5)} | {"x_a": 1, "y_a": 1}
    folds = spw.rotation_folds(counts, fold0=())
    assert folds[1] == [f"big_{i}" for i in range(5)]
    assert folds[2] == ["x_a", "y_a"]


def test_a_two_bird_block_is_never_split():
    counts = {"p_a": 100, "p_b": 100, "q_a": 90, "r_a": 80, "s_a": 70}
    folds = spw.rotation_folds(counts, fold0=())
    assert any({"p_a", "p_b"} <= set(members) for members in folds.values())


# --------------------------------------------------------------------------- pools

def test_no_bird_or_genus_mate_is_in_its_own_fold_pool():
    for k, members in FOLDS.items():
        genera = {spw.genus(n) for n in members}
        assert not set(members) & set(POOLS[k])
        assert not {spw.genus(n) for n in POOLS[k]} & genera


def test_the_genus_mates_left_out():
    assert "anser_albifrons" not in POOLS[1]
    assert "larus_fuscus" not in POOLS[3]
    assert "anser_albifrons" in POOLS[2] and "larus_fuscus" in POOLS[2]


def test_pool_sizes():
    assert {k: len(p) for k, p in POOLS.items()} == {0: 22, 1: 25, 2: 28, 3: 26, 4: 27, 5: 27, 6: 27, 7: 27, 8: 27,
                                                      9: 27}


def test_fold_zero_pool_is_the_registered_main_pool():
    assert POOLS[0] == sorted(REGISTERED["pools"]["main"])


def test_the_leak_check_fails_on_an_own_bird_a_genus_mate_or_a_missing_bird():
    pools = dict(POOLS)
    pools[2] = POOLS[2] + ["morus_bassanus"]
    with pytest.raises(SystemExit, match="fold 2"):
        spw.refuse_rotation_leaks(FOLDS, pools, BIRDS)
    pools = dict(POOLS)
    pools[3] = POOLS[3] + ["larus_fuscus"]
    with pytest.raises(SystemExit, match="fold 3"):
        spw.refuse_rotation_leaks(FOLDS, pools, BIRDS)
    with pytest.raises(SystemExit, match="every bird exactly once"):
        spw.refuse_rotation_leaks(FOLDS, POOLS, BIRDS + ["grus_grus"])


# --------------------------------------------------------------------------- exclusions

def test_the_goose_six_are_carried():
    assert set(spw.BIRDS_ROTATION_TRAINING_EXCLUSIONS["anser_albifrons"]) == set(
        spw.BIRDS_TRAINING_EXCLUSIONS["anser_albifrons"]
    )


def test_every_exclusion_is_a_training_individual_of_its_species():
    for species, ids in spw.BIRDS_ROTATION_TRAINING_EXCLUSIONS.items():
        manifest = json.loads((spw.SPLITS / f"{spw.SPLIT_STEM.get(species, species)}_split_seed42.json").read_text())
        train = {spw.id_key(i) for i in manifest["train"]}
        assert {spw.id_key(i) for i in ids} <= train, species
        assert len(set(ids)) == len(ids), species


@pytest.fixture
def toy_species(tmp_path, monkeypatch):
    """Three training birds of 10, 20 and 40 windows, one val and one test bird."""
    (tmp_path / "toy_split_seed42.json").write_text(json.dumps({"train": [1, 2, 3], "val": [4], "test": [5]}))
    monkeypatch.setattr(spw, "SPLITS", tmp_path)
    counts = pd.DataFrame({"individual_id": ["1", "2", "3", "4", "5"], "n_windows": [10, 20, 40, 80, 160]})
    monkeypatch.setattr(spw, "window_counts_per_individual", lambda *a, **k: counts)
    return "toy"


def test_windows_are_counted_without_the_excluded_individuals(toy_species):
    assert spw.train_windows_without(toy_species, toy_species) == 70
    assert spw.train_windows_without(toy_species, toy_species, (2,)) == 50
    assert spw.train_windows_without(toy_species, toy_species, ("1", 3.0)) == 20


def test_an_exclusion_outside_the_training_partition_fails(toy_species):
    with pytest.raises(SystemExit, match="not in toy's training partition"):
        spw.train_windows_without(toy_species, toy_species, (5,))


def test_the_pinned_counts_cover_the_30_and_carry_the_exclusions():
    assert sorted(spw.BIRDS_ROTATION_TRAIN_WINDOWS) == BIRDS
    registered = REGISTERED["train_window_counts"]
    for name, n in spw.BIRDS_ROTATION_TRAIN_WINDOWS.items():
        if name in spw.BIRDS_ROTATION_TRAINING_EXCLUSIONS and name in registered:
            assert n < registered[name], name  # excluded individuals carry windows
        elif name in registered:
            assert n == registered[name], name
    assert spw.BIRDS_ROTATION_TRAIN_WINDOWS["anser_albifrons"] == 77_550  # preregistration_birds.md sec 7


def test_pooled_probabilities_use_the_counts_after_exclusions():
    counts = spw.BIRDS_ROTATION_TRAIN_WINDOWS
    probabilities = spw.rotation_probabilities(POOLS, counts)
    assert "0" not in probabilities  # carried with the registered main-pool probabilities
    for k, probs in probabilities.items():
        assert set(probs) == set(POOLS[int(k)])
        assert sum(probs.values()) == pytest.approx(1.0)
    total = sum(math.sqrt(counts[n]) for n in POOLS[1])
    assert probabilities["1"]["ciconia_ciconia"] == pytest.approx(math.sqrt(779_723) / total)


# --------------------------------------------------------------------------- write once

def _payload(counts=None):
    counts = dict(spw.BIRDS_ROTATION_TRAIN_WINDOWS if counts is None else counts)
    return {
        "folds": {str(k): m for k, m in FOLDS.items()},
        "pools": {str(k): p for k, p in POOLS.items()},
        "train_window_counts": counts,
        "probabilities": spw.rotation_probabilities(POOLS, counts),
        "training_exclusions": {s: list(ids) for s, ids in spw.BIRDS_ROTATION_TRAINING_EXCLUSIONS.items()},
    }


def _args(no_write=False, regenerate=None):
    return argparse.Namespace(no_write=no_write, regenerate=regenerate)


@pytest.fixture
def out(tmp_path, monkeypatch):
    path = tmp_path / "birds_rotation_folds.json"
    monkeypatch.setattr(spw, "BIRDS_ROTATION_OUT", path)
    monkeypatch.setattr(spw, "REPO", tmp_path)
    return path


def test_writing_is_refused_while_the_counts_are_unpinned(out, monkeypatch):
    payload = _payload()
    monkeypatch.setattr(spw, "BIRDS_ROTATION_TRAIN_WINDOWS", {})
    with pytest.raises(SystemExit, match="not pinned"):
        spw.register_birds_rotation(payload, _args())
    assert not out.exists()


def test_no_write_never_writes(out):
    spw.register_birds_rotation(_payload(), _args(no_write=True))
    assert not out.exists()


def test_the_file_is_written_once_then_reproduced(out, capsys):
    spw.register_birds_rotation(_payload(), _args())
    written = out.read_text()
    spw.register_birds_rotation(_payload(), _args())
    assert out.read_text() == written
    assert "reproduced" in capsys.readouterr().out


def test_different_counts_are_refused_without_regenerate(out):
    spw.register_birds_rotation(_payload(), _args())
    changed = dict(spw.BIRDS_ROTATION_TRAIN_WINDOWS) | {"ardea_alba": 1}
    with pytest.raises(SystemExit, match="already holds different folds"):
        spw.register_birds_rotation(_payload(changed), _args())


def test_changed_exclusions_are_refused_without_regenerate(out):
    spw.register_birds_rotation(_payload(), _args())
    changed = _payload() | {"training_exclusions": {"anser_albifrons": []}}
    with pytest.raises(SystemExit, match="already holds different folds"):
        spw.register_birds_rotation(changed, _args())


def test_regenerate_records_what_it_supersedes(out):
    spw.register_birds_rotation(_payload(), _args())
    changed = dict(spw.BIRDS_ROTATION_TRAIN_WINDOWS) | {"ardea_alba": 1}
    spw.register_birds_rotation(_payload(changed), _args(regenerate="test"))
    record = json.loads(out.read_text())
    assert record["regenerated"]["reason"] == "test"
    assert record["regenerated"]["supersedes"]["train_window_counts"]["ardea_alba"] == 21_218


def test_stored_probabilities_must_follow_from_stored_counts(out):
    spw.register_birds_rotation(_payload(), _args())
    record = json.loads(out.read_text())
    record["probabilities"]["1"]["ciconia_ciconia"] += 1e-6
    out.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="no longer follow from its counts"):
        spw.register_birds_rotation(_payload(), _args())


# --------------------------------------------------------------------------- CLI

@pytest.mark.parametrize("other", ["--tier3", "--tier4", "--birds"])
def test_birds_rotation_is_refused_alongside_another_mode(monkeypatch, other):
    monkeypatch.setattr(sys, "argv", ["source_pool_weights.py", "--birds-rotation", other, "--no-write"])
    with pytest.raises(SystemExit, match="only one of"):
        spw.main()
