# GenAI reference: to add. 

import json
from collections import Counter
from dataclasses import asdict
from dataclasses import fields as dataclass_fields

import numpy as np
import pytest

import experiments.training.pooling_train as pt
import experiments.training.tier4_train as t4
from animal_movement_fm.train.loop import TrainConfig
from tests.test_b26_golden import golden_pool

PAYLOAD = json.loads(t4.REGISTERED_POOL.read_text())
GRID = 11
START_INDICES = {"B26bal": 0, "B26moe": 5, "B26balmoe": 10}
B26 = t4.PLAN[5]


def test_plan_is_three_arms_at_b26s_seeds_in_the_registered_order():
    assert len(pt.PLAN) == 15 and len({run.run_id for run in pt.PLAN}) == 15
    assert Counter(run.arm for run in pt.PLAN) == {"B26bal": 5, "B26moe": 5, "B26balmoe": 5}
    assert {arm: next(i for i, r in enumerate(pt.PLAN) if r.arm == arm) for arm in pt.ARMS} == START_INDICES
    assert [run.seed for run in pt.PLAN] == [0, 1, 2, 3, 4] * 3
    assert B26.arm == "B26" and "B26" not in pt.ARMS  # carried, never retrained


def test_the_arms_are_the_2x2_less_the_carried_cell():
    cells = {(spec["balanced"], spec["trunk"]) for spec in pt.ARMS.values()}
    assert cells == {(True, "transformer"), (False, "moe"), (True, "moe")}


def test_balanced_weights_give_each_domain_a_third_and_keep_a8_within_it():
    weights = pt.balanced_weights(PAYLOAD)
    a8 = PAYLOAD["source_pool"]
    assert set(weights) == set(a8) == set(t4.REGISTERED_SOURCE_PERCENT)
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-12)
    for domain in t4.DOMAINS:
        members = [n for n in a8 if PAYLOAD["domains"][n] == domain]
        assert sum(weights[n] for n in members) == pytest.approx(1 / 3, abs=1e-12)
        a8_total = sum(a8[n] for n in members)
        for n in members:
            assert weights[n] == pytest.approx(a8[n] / a8_total / 3, rel=1e-12)
    assert not set(weights) & (set(t4.HELD_OUT) | set(t4.RESERVED))


def test_the_pinned_balanced_constants_load_and_a_drift_is_refused(monkeypatch):
    pt.load_registered_pool()
    drifted = dict(pt.BALANCED_SOURCE_PERCENT, ciconia_ciconia=7.62)
    monkeypatch.setattr(pt, "BALANCED_SOURCE_PERCENT", drifted)
    with pytest.raises(SystemExit, match="balanced weights"):
        pt.load_registered_pool()


# Configuration.
def test_every_config_is_b26s_but_for_the_trunk_and_adds_no_field():
    b26 = asdict(t4.build_config(B26, GRID, t4.REGISTERED_STEPS, sorted(PAYLOAD["source_pool"]), PAYLOAD))
    frozen = set(t4.FROZEN_TRAIN_CONFIG_FIELDS)
    assert {field.name for field in dataclass_fields(TrainConfig)} == frozen
    for run in pt.PLAN:
        got = asdict(pt.build_config(run, GRID, pt.REGISTERED_STEPS))
        assert set(got) == frozen
        assert {k for k in got if got[k] != b26[k]} <= {"trunk", "seed"}
        assert (got["trunk"], got["seed"]) == (pt.ARMS[run.arm]["trunk"], run.seed)


def test_records_are_tagged_exploratory_under_the_pooling_freeze():
    for index in START_INDICES.values():
        record = pt.base_record(pt.PLAN[index], index, GRID, "tier2_selection.json", "source_pool")
        assert record["tier"] == "pooling"
        assert record["exploratory"] is True and record["opened_test_partition"] is False
        assert record["registered_by"] == "pooling_exploratory_freeze.md"
    assert pt.RESULTS.name == pt.CHECKPOINTS.name == "pooling"


# Pools and the dry run, with the window loader stubbed so no parquet is read.
class _StubWindows:
    def __init__(self, n_windows=1_000):
        self.n_windows = n_windows


@pytest.fixture
def stub_loader(monkeypatch):
    monkeypatch.setattr(t4.T, "_load_species_windows", lambda *a, **k: _StubWindows())


def test_the_a8_arm_draws_exactly_as_b26_drew(stub_loader):
    ours, node = pt.training_source(pt.PLAN[START_INDICES["B26moe"]], PAYLOAD)
    theirs, their_node = t4.pooled_source(B26, PAYLOAD)
    assert node == their_node == "source_pool"
    assert ours.names == theirs.names
    assert np.array_equal(ours.probabilities, theirs.probabilities)


def test_the_balanced_arms_draw_a_third_per_domain(stub_loader):
    pool, node = pt.training_source(pt.PLAN[START_INDICES["B26bal"]], PAYLOAD)
    assert node == "source_pool:domain_balanced"
    for domain in t4.DOMAINS:
        share = sum(p for n, p in zip(pool.names, pool.probabilities) if PAYLOAD["domains"][n] == domain)
        assert share == pytest.approx(1 / 3, abs=1e-12)


@pytest.mark.parametrize("arm,index", START_INDICES.items())
def test_dry_run_completes_for_the_first_index_of_every_arm(arm, index, stub_loader, capsys):
    pt.dry_run(PAYLOAD, index)
    printed = capsys.readouterr().out
    assert pt.PLAN[index].run_id in printed and "dry run only" in printed
    assert f"trunk {pt.ARMS[arm]['trunk']}" in printed


def test_dry_run_over_every_source_prints_both_weightings(stub_loader, capsys):
    pt.dry_run(PAYLOAD, None)
    printed = capsys.readouterr().out
    assert "flying  65.4%" in printed and "flying  33.3%" in printed
    assert "26 sources built" in printed


# Routing.
def test_routing_summary_covers_every_source_and_layer(monkeypatch):
    monkeypatch.setattr(pt, "ROUTING_WINDOWS", 50)
    config = TrainConfig(d_model=32, num_layers=2, lr=1e-3, steps=1, use_fm_package=False,
                         use_reconstruction_head=False, trunk="moe")
    summary = pt.routing_summary(config.build(), golden_pool(), {"a": "flying", "b": "marine", "c": "terrestrial"})
    assert set(summary) == {"a", "b", "c"}
    for entry in summary.values():
        assert entry["windows"] == 50 and len(entry["layers"]) == 2
        for layer in entry["layers"]:
            assert sum(layer["mean_prob"]) == pytest.approx(1.0, abs=1e-5)
            assert sum(layer["top_k_share"]) == pytest.approx(1.0, abs=1e-5)
            assert 0.0 <= layer["window_agreement"] <= 1.0
