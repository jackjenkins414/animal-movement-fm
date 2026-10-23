# GenAI reference: tests for tier4_repaired_diagnostic.py, written by GenAI from the outline I gave for the diagnostic
# (that being runs and twins, the test partition guard, the samples, each measure, the record check and the sec 7 block).
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch import nn

import experiments.diagnostics.tier4_repaired_diagnostic as D
from animal_movement_fm.data.split import load_split
from animal_movement_fm.models.fm import STEP_CHANNELS

R, T = D.R, D.T  # the modules the diagnostic imported
PAYLOADS = R.load_payloads()
GRID = 11
NES, WOLF = "mirounga_angustirostris", "canis_lupus"


class _StubWindows:
    """A species' windows whose features carry the window id in one cell, so no parquet is read."""

    def __init__(self, n_windows):
        self.n_windows = n_windows

    def gather(self, ids, sigma=None):
        ids = np.asarray(ids)
        features = np.zeros((len(ids), 12, 12))
        features[:, 0, 1] = ids
        return features, np.ones((len(ids), 2)), np.ones(len(ids))


@pytest.fixture
def stub_loader(monkeypatch):
    seen = []

    def load(stem, split_stem, partition, individuals=None):
        seen.append((stem, partition))
        n = PAYLOADS["4"]["train_window_counts"][stem] if partition == "train" else 10_000
        return _StubWindows(n)

    monkeypatch.setattr(T, "_load_species_windows", load)
    return seen


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------------------------------- runs
def test_each_arm_is_its_five_runs_and_its_twins_five_at_the_same_seeds():
    groups = {arm: D.entries(arm) for arm in R.ARMS}
    assert sum(map(len, groups.values())) == 45
    twins = {arm: sorted({e["label"] for e in g if e["role"] == "twin"}) for arm, g in groups.items()}
    assert twins == {"B3F": ["B"], "B9F": ["B9"], "B26F": ["B26"], "BdMF": ["BdM"], "BdTF": []}
    assert [e["run_id"] for e in groups["B3F"] if e["role"] == "twin"] == [f"B_s{s}" for s in R.SEEDS]
    assert [e["run_id"] for e in groups["BdMF"] if e["role"] == "twin"] == [f"BdM_{NES}_s{s}" for s in R.SEEDS]
    for group in groups.values():
        assert [e["seed"] for e in group if e["role"] == "repaired"] == list(R.SEEDS)
        assert [e["seed"] for e in group if e["role"] == "twin"] in ([], list(R.SEEDS))
    assert {e["results"] for e in groups["B9F"] if e["role"] == "twin"} == {R.T3.RESULTS}
    assert {e["results"] for e in groups["B26F"] if e["role"] == "repaired"} == {R.RESULTS}
    with pytest.raises(SystemExit, match="not an arm"):
        D.entries("B26")


@pytest.mark.parametrize("arm", list(R.ARMS))
def test_the_registered_pool_names_are_the_pool_the_driver_builds(arm, stub_loader):
    run = R.Run(arm, 0)
    source_pool, _ = R.training_source(run, PAYLOADS)
    assert D.pool_names(run, PAYLOADS) == list(source_pool.names)


# ---------------------------------------------------------------------------------------------------- guards
def test_the_audited_loader_refuses_test_and_any_test_individual_before_loading(stub_loader):
    stub = T._load_species_windows
    split = load_split(T.SPLITS / f"{WOLF}_split_seed42.json")
    log = []
    with D.audited_loader(log):
        with pytest.raises(SystemExit, match="never 'test'"):
            T._load_species_windows(WOLF, WOLF, "test")
        with pytest.raises(SystemExit, match="test individuals"):
            T._load_species_windows(WOLF, WOLF, "train", [split.train[0], split.test[0]])
        T._load_species_windows(WOLF, WOLF, "val")
    assert T._load_species_windows is stub
    assert stub_loader == [(WOLF, "val")]  # nothing refused reached the loader
    assert log == [{"species": WOLF, "partition": "val", "individuals": len(split.val), "windows": 10_000,
                    "test_individuals": 0}]


def test_the_audited_loader_is_restored_when_the_block_fails(stub_loader):
    stub = T._load_species_windows
    with pytest.raises(RuntimeError):
        with D.audited_loader([]):
            raise RuntimeError
    assert T._load_species_windows is stub


# ---------------------------------------------------------------------------------------------------- samples
def test_the_training_sample_is_trainings_a8_draw_at_a_fixed_seed(stub_loader):
    first = D.training_sample("BdTF", PAYLOADS, n=2_000)
    second = D.training_sample("BdTF", PAYLOADS, n=2_000)
    np.testing.assert_array_equal(first["features"], second["features"])
    source_pool, _ = R.training_source(R.Run("BdTF", 0), PAYLOADS)
    features, _, _ = source_pool.sample(np.random.default_rng(D.SAMPLE_SEED), 2_000)
    np.testing.assert_array_equal(first["features"], features.astype(np.float32))
    assert first["node"] == "sub_pools.domain_terrestrial" and len(first["targets"]) == 2_000
    assert {partition for _, partition in stub_loader} == {"train"}


@pytest.mark.parametrize("available, expected", [(10_000, 3_000), (100, 100)])
def test_validation_windows_are_capped_uniform_without_replacement_and_fixed(monkeypatch, available, expected):
    monkeypatch.setattr(T, "_load_species_windows", lambda stem, split_stem, partition, individuals=None:
                        _StubWindows(available))
    first, second = D.validation_windows(WOLF), D.validation_windows(WOLF)
    ids = first["features"][:, 0, 1]
    assert len(ids) == len(set(ids)) == expected and np.all(np.diff(ids) > 0)
    np.testing.assert_array_equal(ids, second["features"][:, 0, 1])
    assert first["available"] == available and first["boundary_m"] == T.boundary_m(WOLF)


# ---------------------------------------------------------------------------------------------------- measures
def test_loss_over_zero_is_one_for_the_zero_predictor_and_zero_for_an_exact_one():
    targets = np.random.default_rng(0).normal(size=(50, 2)) * 3
    assert D.loss_over_zero(np.zeros_like(targets), targets)["ratio"] == 1.0
    assert D.loss_over_zero(targets, targets)["ratio"] == 0.0


def test_the_logged_loss_is_the_final_and_the_median_of_the_last_five_against_zero():
    out = D.logged_loss({"loss_history": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0], "final_loss": 3.0}, 2.0)
    assert out == {"points": 7, "final": 3.0, "last_5_median": 5.0, "final_over_zero": 1.5, "last_over_zero": 2.5}
    assert D.logged_loss({}, 2.0)["last_over_zero"] is None


class _LastStep(nn.Module):
    def forward(self, x):
        return x[:, -1, list(STEP_CHANNELS)]


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.clone())
        return torch.ones(len(x), 2)


def test_sensitivity_is_zero_for_a_model_that_ignores_its_input_and_one_for_the_last_step():
    features = np.random.default_rng(1).normal(size=(64, 12, 12)).astype(np.float32)
    flat = _Recorder()
    out = D.sensitivity(flat, features, D.predict(flat, features))
    assert out["last_swapped"] == 0.0 and out["last_plus_1"] == 0.0
    last = _LastStep()
    base = D.predict(last, features)
    out = D.sensitivity(last, features, base)
    assert out["last_plus_1"] == pytest.approx(1.0)
    rolled = np.roll(features[:, -1], 1, axis=0)[:, list(STEP_CHANNELS)]
    assert out["last_swapped"] == pytest.approx(np.median(np.linalg.norm(rolled - base, axis=-1)))
    assert out["last_plus_1_over_output"] == pytest.approx(1.0 / np.median(np.linalg.norm(base, axis=-1)))


def test_the_perturbations_reach_the_model_as_the_data_arrives():
    """The +1 is on the raw feature; the repaired encoder's signed_log1p is inside the model and runs after it."""
    features = np.random.default_rng(2).normal(size=(8, 12, 12)).astype(np.float32)
    recorder = _Recorder()
    D.sensitivity(recorder, features, np.ones((8, 2)))
    swapped, bumped = (x.numpy() for x in recorder.inputs)
    np.testing.assert_array_equal(swapped[:, :-1], features[:, :-1])
    np.testing.assert_array_equal(swapped[:, -1], np.roll(features[:, -1], 1, axis=0))
    difference = bumped - features
    assert np.allclose(difference[:, -1, D.D_EAST], 1.0)
    difference[:, -1, D.D_EAST] = 0.0
    assert not difference.any()


def test_magnitudes_are_in_metres_with_the_ratio_on_travel_windows_only():
    pred = np.array([[3.0, 4.0], [0.0, 0.0], [6.0, 8.0]])
    targets = np.array([[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]])
    scale = np.array([10.0, 10.0, 1.0])  # |pred| 50, 0, 10 m; |actual| 10, 5, 2 m
    out = D.magnitudes(pred, targets, scale, boundary_m=6.0)
    assert out == {"n_windows": 3, "n_travel": 1, "median_pred_m_all": 10.0, "median_actual_m_all": 5.0,
                   "ratio": 5.0, "median_pred_m": 50.0, "median_actual_m": 10.0}
    assert D.magnitudes(pred, targets, scale, boundary_m=100.0)["ratio"] is None


# ---------------------------------------------------------------------------------------------------- summary
def _val(value):
    out = {}
    for first, second in D.VAL_METRICS.values():
        out.setdefault(first, {})[second] = value
    return out


def _model(label, role, seed, value):
    return {"label": label, "role": role, "seed": seed, "train": {"ratio": value},
            "logged": {"final_over_zero": value, "last_over_zero": value}, "val": {NES: _val(value)}}


def _arm():
    models = {}
    for seed in R.SEEDS:
        models[f"BdMF_s{seed}"] = _model("BdMF", "repaired", seed, float(seed))
        models[f"BdM_s{seed}"] = _model("BdM", "twin", seed, None if seed == 4 else 10.0 + seed)
    return {"arm": "BdMF", "scored_on": [NES], "models": models, "loads": [], "dry_run": False,
            "samples": {"train_node": "sub_pools.domain_marine", "train_species": ["a"], "n_train": 20_000,
                        "val": {NES: {"windows": 3_000, "available": 9_000}}}}


def test_the_summary_is_the_median_over_seeds_each_arm_before_its_twin():
    summary = D.summarise([_arm()])
    assert list(summary) == ["BdMF", "BdM"]
    assert summary["BdMF"]["seeds"] == list(R.SEEDS) and summary["BdMF"]["role"] == "repaired"
    assert summary["BdMF"]["train_loss_over_zero"] == 2.0
    assert summary["BdM"]["val"][NES]["ratio"] == 11.5  # seed 4's missing value is left out
    assert summary["BdM"]["logged_last_over_zero"] == 11.5


def test_the_reading_rule_is_quoted_from_the_freeze():
    assert "within 1% of the zero predictor" in D.reading_rule()


def test_the_report_needs_every_arm_and_refuses_a_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "OUT", tmp_path)
    monkeypatch.setattr(D, "REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr(D, "REPORT_MD", tmp_path / "report.md")
    with pytest.raises(SystemExit, match="run every arm"):
        D.report()
    for arm in R.ARMS:
        (tmp_path / f"{arm}.json").write_text(json.dumps(_arm() | {"dry_run": arm == "BdTF"}))
    with pytest.raises(SystemExit, match="dry run"):
        D.report()
    (tmp_path / "BdTF.json").write_text(json.dumps(_arm()))
    D.report()
    text = (tmp_path / "report.md").read_text()
    assert "> " in text and "| BdMF | repaired | 5 |" in text and "| BdM | twin | 5 |" in text


# ---------------------------------------------------------------------------------------------------- records
def _record(run, directory):
    names = D.pool_names(run, PAYLOADS)
    checkpoint = f"{run.run_id}_reported.pt"
    (directory / checkpoint).write_bytes(b"")
    return {
        "run_id": run.run_id, "plan_index": D.PLAN_INDEX[run.run_id], "arm": run.arm, "seed": run.seed,
        "tier": R.TAG, "twin": run.twin.run_id if run.twin else None, "switches": dict(R.SWITCHES),
        "queue": R.QUEUE, "pbs_queue": "gpuhopper", "device": "NVIDIA H200", "exploratory": True,
        "opened_test_partition": False, "pool_species": names,
        "config": asdict(R.build_config(run, GRID, R.REGISTERED_STEPS, names, PAYLOADS)),
        "tf32": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False},
        "applied_updates": R.REGISTERED_STEPS, "skipped_steps": 0, "withheld_fraction": 0.0, "nonfinite_steps": [],
        "reported_checkpoint": checkpoint,
    }


FAULTS = {
    "tf32": lambda r: r["tf32"].update({"cuda.matmul.allow_tf32": True}),
    "non-finite": lambda r: r.update(nonfinite_steps=[1_200]),
    "config differs": lambda r: r["config"].update(norm_first=False),
    "pbs_queue": lambda r: r.update(pbs_queue="gpuvolta"),
    "applied": lambda r: r.update(applied_updates=19_999),
    "opened_test_partition": lambda r: r.update(opened_test_partition=True),
    "no reported checkpoint": lambda r: r.update(reported_checkpoint="missing.pt"),
    "pool_species": lambda r: r.update(pool_species=r["pool_species"][:-1]),
}


@pytest.mark.parametrize("fault", list(FAULTS))
def test_the_record_check_passes_a_clean_run_and_names_each_fault(fault, results):
    run = R.PLAN[12]  # B26F seed 2
    record = _record(run, results)
    assert D.run_problems(run, record, GRID, PAYLOADS) == []
    assert D.run_problems(run, record | {"pbs_queue": "gpuhopper-exec"}, GRID, PAYLOADS) == []
    FAULTS[fault](record)
    assert any(fault in problem for problem in D.run_problems(run, record, GRID, PAYLOADS))
    assert D.run_problems(run, None, GRID, PAYLOADS) == ["no record"]


def test_sec7_counts_clean_runs_by_arm_and_names_what_is_missing(results):
    records = {run.run_id: _record(run, results) for run in R.PLAN[:-1]}
    gap = D.completeness(records, GRID, PAYLOADS)
    assert (gap["n_records"], gap["n_clean"], gap["complete"]) == (24, 24, False)
    assert gap["by_arm"]["BdTF"] == {"planned": 5, "records": 4, "clean": 4}
    text = D.section7_markdown(records, gap, [])
    for part in ("24 of 25 recorded, 24 clean", "BdTF_canis_lupus_s4 (index 24): no record", "gpuhopper 24",
                 "TF32 off on 24", "NVIDIA H200 24", "20 runs", "as registered"):
        assert part in text
    assert "differ: B_s0: x" in D.section7_markdown(records, gap, ["B_s0: x"])


def _checkpoint(directory, run_id, config, switches):
    model = R.TrainConfig(**config).build()
    torch.save({"run_id": run_id, "config": config, "switches": switches, "model": model.state_dict()},
               directory / f"{run_id}_reported.pt")


def test_load_model_refuses_a_checkpoint_whose_switches_are_not_its_roles(results, monkeypatch):
    run = R.PLAN[0]  # B3F seed 0
    record = _record(run, results)
    (results / f"{run.run_id}.json").write_text(json.dumps(record))
    entry = next(e for e in D.entries("B3F") if e["run_id"] == run.run_id) | {"results": results}
    _checkpoint(results, run.run_id, record["config"], dict(R.SWITCHES))
    model, loaded = D.load_model(entry, GRID, PAYLOADS)
    assert loaded["run_id"] == run.run_id and not model.training
    unrepaired = record["config"] | D.DEFAULTS
    _checkpoint(results, run.run_id, unrepaired, None)
    with pytest.raises(SystemExit, match="checkpoint switches"):
        D.load_model(entry, GRID, PAYLOADS)

    monkeypatch.setattr(R, "twin_problems", lambda *args: [])
    twin = next(e for e in D.entries("B3F") if e["run_id"] == "B_s0") | {"results": results}
    _checkpoint(results, "B_s0", record["config"], None)
    with pytest.raises(SystemExit, match="checkpoint switches"):
        D.load_model(twin, GRID, PAYLOADS)
    _checkpoint(results, "B_s0", unrepaired, None)
    assert D.load_model(twin, GRID, PAYLOADS)[1] is None  # no twin record in tmp; twin_problems was stubbed


def test_the_dry_run_builds_samples_through_the_audited_loader_and_loads_no_model(stub_loader, monkeypatch):
    monkeypatch.setattr(D, "load_model", lambda *args: pytest.fail("a dry run loads no model"))
    out = D.run_arm("BdMF", n_train=500, dry_run=True)
    assert out["models"] == {} and out["scored_on"] == [NES] and out["twin"] == "BdM"
    assert out["samples"]["val"][NES]["windows"] == D.MAX_VAL
    assert {load["partition"] for load in out["loads"]} == {"train", "val"}
    assert [load["species"] for load in out["loads"] if load["partition"] == "val"] == [NES]
    assert out["zero_loss"]["train"] > 0 and out["opened_test_partition"] is False
