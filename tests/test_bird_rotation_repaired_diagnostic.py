# GenAI reference: tests for bird_rotation_repaired_diagnostic.py, written by GenAI from my specification
from __future__ import annotations

import json
import re
from dataclasses import asdict

import numpy as np
import pytest
import torch

import experiments.diagnostics.bird_rotation_repaired_diagnostic as G

D, DG, RT, BT, T = G.D, G.DG, G.RT, G.BT, G.T  # the modules the diagnostic imported
PAYLOADS = D.load_payloads()
GRID = 11
PLACEHOLDER = re.compile(r"\b(?:to|will)\s+fill\b|\bdraft\b", re.IGNORECASE)  # the evaluators' refusal


class _StubWindows:
    """A species' windows with fixed features, so no parquet is read."""

    def __init__(self, n_windows):
        self.n_windows = n_windows

    def gather(self, ids, sigma=None):
        ids = np.asarray(ids)
        features = np.zeros((len(ids), 11, 12))
        features[:, 0, 1] = ids
        return features, np.ones((len(ids), 2)), np.ones(len(ids))


@pytest.fixture
def stub_loader(monkeypatch):
    """Training loads at the registered counts (the bird study's for fold 0, the rotation's after exclusions for 1-9)."""
    seen = []

    def load(stem, split_stem, partition, individuals=None):
        seen.append((stem, partition))
        if partition != "train":
            return _StubWindows(10_000)
        return _StubWindows(PAYLOADS["birds" if individuals is None else "rotation"]["train_window_counts"][stem])

    monkeypatch.setattr(T, "_load_species_windows", load)
    return seen


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "RESULTS", tmp_path)
    return tmp_path


def _tiny(arm):
    config = D.recipe_config(D.Run(arm, 1, 0), GRID, 10, [], PAYLOADS)
    return D.replace(config, d_model=16, num_layers=1, **D.SWITCHES)


# ---------------------------------------------------------------------------------------------------- runs
def test_each_fold_is_its_six_repaired_runs_and_their_six_originals():
    groups = {k: G.entries(k) for k in D.FOLDS}
    assert sum(map(len, groups.values())) == 120 and all(len(g) == 12 for g in groups.values())
    fold0 = {e["run_id"]: e for e in groups[0]}
    assert fold0["Bb_s0"]["results"] == BT.RESULTS and fold0["Zb_s2"]["role"] == "original"
    assert fold0["ZbF6_f0_s1"]["results"] == D.RESULTS and fold0["ZbF6_f0_s1"]["label"] == "ZbF6"
    fold3 = {e["run_id"] for e in groups[3]}
    assert fold3 == {f"{a}_f3_s{s}" for a in ("BbF6", "ZbF6", "Bb", "Zb") for s in range(3)}
    assert all(e["results"] == RT.RESULTS for e in groups[3] if e["role"] == "original")
    with pytest.raises(SystemExit, match="fold 10"):
        G.entries(10)


@pytest.mark.parametrize("fold", [0, 5])
def test_the_registered_pool_names_are_the_pool_the_driver_builds(fold, stub_loader):
    pool, _ = D.training_source(D.Run("BbF6", fold, 0), PAYLOADS)
    assert G.pool_names(fold, PAYLOADS) == list(pool.names)


def test_held_out_entries_and_boundaries_come_from_the_driver_that_pinned_them():
    assert G.held_out_entry(0, "larus_fuscus") == ("larus_fuscus", "larus_fuscus", BT.HELD_OUT_BOUNDARY_M["larus_fuscus"])
    stem, split_stem, boundary = G.held_out_entry(2, "ciconia_ciconia")
    assert (stem, split_stem) == RT.rotation_attached_entry("ciconia_ciconia")[:2]
    assert boundary == RT.HELD_OUT_BOUNDARY_M["ciconia_ciconia"]
    with pytest.raises(SystemExit, match="held out in fold 2, not 3"):
        G.held_out_entry(3, "ciconia_ciconia")


# ---------------------------------------------------------------------------------------------------- measures
def test_the_full_model_is_read_on_its_forecast_and_the_plain_model_unchanged():
    x = torch.randn(5, 11, 12)
    torch.manual_seed(0)
    full = _tiny("ZbF6").build().eval()
    plain = _tiny("BbF6").build().eval()
    with torch.no_grad():
        assert isinstance(full(x), tuple)
        torch.testing.assert_close(G.Forecast(full)(x), full(x)[0])
        torch.testing.assert_close(G.Forecast(plain)(x), plain(x))


def test_the_tier4_measures_run_on_a_forecast_wrapped_full_model():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 11, 12)).astype(np.float32)
    targets, scale = rng.normal(size=(40, 2)), np.full(40, 100.0)
    torch.manual_seed(0)
    model = G.Forecast(_tiny("ZbF6").build()).eval()
    train = {"features": features, "targets": targets}
    val = {"bird": {"features": features, "targets": targets, "scale": scale, "boundary_m": 10.0}}
    scored = DG.score(model, {"loss_history": [1.0, 0.9], "final_loss": 0.9}, train, val)
    assert scored["train"]["ratio"] > 0 and scored["val"]["bird"]["sensitivity"]["median_abs_output"] > 0
    assert scored["val"]["bird"]["magnitudes"]["n_travel"] > 0


def test_the_dry_run_builds_samples_through_the_audited_loader_and_loads_no_model(stub_loader, monkeypatch):
    monkeypatch.setattr(G, "load_model", lambda *args: pytest.fail("a dry run loads no model"))
    for fold in (0, 2):
        out = G.run_fold(fold, n_train=500, dry_run=True)
        held_out = list(D.Run("BbF6", fold, 0).scored_on)
        assert out["models"] == {} and out["scored_on"] == held_out and out["opened_test_partition"] is False
        assert {load["partition"] for load in out["loads"]} == {"train", "val"}
        assert [load["species"] for load in out["loads"] if load["partition"] == "val"] == held_out
        assert all(load["test_individuals"] == 0 for load in out["loads"])
        assert out["samples"]["n_train"] == 500 and out["zero_loss"]["train"] > 0


def test_the_audited_loader_refuses_a_test_partition(stub_loader):
    with DG.audited_loader([]):
        with pytest.raises(SystemExit, match="never 'test'"):
            T._load_species_windows("larus_fuscus", "larus_fuscus", "test")


# ---------------------------------------------------------------------------------------------------- records
def _record(run, directory):
    names = G.pool_names(run.fold, PAYLOADS)
    checkpoint = f"{run.run_id}_reported.pt"
    (directory / checkpoint).write_bytes(b"")
    return json.loads(json.dumps({
        "run_id": run.run_id, "plan_index": G.PLAN_INDEX[run.run_id], "arm": run.arm, "fold": run.fold,
        "seed": run.seed, "tier": D.TAG, "twin": run.twin.run_id, "twin_study": run.twin_study,
        "switches": dict(D.SWITCHES), "queue": D.QUEUE, "pbs_queue": "gpuhopper", "device": "NVIDIA H200",
        "exploratory": True, "opened_test_partition": False, "pool_species": names,
        "excluded_individuals": D.excluded_individuals(run, names, PAYLOADS),
        "config": asdict(D.build_config(run, GRID, D.REGISTERED_STEPS, names, PAYLOADS)),
        "tf32": {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False},
        "applied_updates": D.REGISTERED_STEPS, "skipped_steps": 0, "withheld_fraction": 0.0, "nonfinite_steps": [],
        "reported_checkpoint": checkpoint,
    }))


FAULTS = {
    "tf32": lambda r: r["tf32"].update({"cuda.matmul.allow_tf32": True}),
    "non-finite": lambda r: r.update(nonfinite_steps=[1_200]),
    "config differs": lambda r: r["config"].update(step_transform="none"),
    "pbs_queue": lambda r: r.update(pbs_queue="gpuvolta"),
    "applied": lambda r: r.update(applied_updates=19_999),
    "opened_test_partition": lambda r: r.update(opened_test_partition=True),
    "no reported checkpoint": lambda r: r.update(reported_checkpoint="missing.pt"),
    "pool_species": lambda r: r.update(pool_species=r["pool_species"][:-1]),
    "excluded_individuals": lambda r: r.update(excluded_individuals={}),
    "twin_study": lambda r: r.update(twin_study="birds"),
    "fold": lambda r: r.update(fold=4),
}


@pytest.mark.parametrize("fault", list(FAULTS))
def test_the_record_check_passes_a_clean_run_and_names_each_fault(fault, results):
    run = D.PLAN[38]  # ZbF6_f2_s2: a fold with the rotation's stork exclusions
    record = _record(run, results)
    assert G.run_problems(run, record, GRID, PAYLOADS) == []
    assert G.run_problems(run, record | {"pbs_queue": "gpuhopper-exec"}, GRID, PAYLOADS) == []
    FAULTS[fault](record)
    assert any(fault in problem for problem in G.run_problems(run, record, GRID, PAYLOADS))
    assert G.run_problems(run, None, GRID, PAYLOADS) == ["no record"]


def test_a_fold_0_record_carries_no_exclusions(results):
    run = D.PLAN[1]  # BbF6_f0_s1
    record = _record(run, results)
    assert record["excluded_individuals"] == {} and G.run_problems(run, record, GRID, PAYLOADS) == []


def test_switch_problems_hold_each_role_to_its_switches():
    repaired = {"config": dict(D.SWITCHES), "switches": dict(D.SWITCHES)}
    assert G.switch_problems(repaired, True) == []
    assert G.switch_problems({"config": {}}, False) == []  # an original written before the fields existed
    assert G.switch_problems(repaired, False) and G.switch_problems({"config": {}, "switches": None}, True)


def _checkpoint(directory, run_id, config, switches):
    model = D.TrainConfig(**config).build()
    torch.save({"run_id": run_id, "config": config, "switches": switches, "model": model.state_dict()},
               directory / f"{run_id}_reported.pt")


def test_load_model_refuses_a_checkpoint_whose_switches_are_not_its_roles(results, monkeypatch):
    run = D.PLAN[30]  # ZbF6_f0_s0
    record = _record(run, results)
    (results / f"{run.run_id}.json").write_text(json.dumps(record))
    entry = next(e for e in G.entries(0) if e["run_id"] == run.run_id) | {"results": results}
    _checkpoint(results, run.run_id, record["config"], dict(D.SWITCHES))
    model, loaded = G.load_model(entry, GRID, PAYLOADS)
    assert loaded["run_id"] == run.run_id and not model.training and isinstance(model, G.Forecast)
    unrepaired = record["config"] | G.DEFAULTS
    _checkpoint(results, run.run_id, unrepaired, None)
    with pytest.raises(SystemExit, match="checkpoint switches"):
        G.load_model(entry, GRID, PAYLOADS)

    monkeypatch.setattr(D, "twin_problems", lambda *args, **kwargs: [])
    original = next(e for e in G.entries(0) if e["run_id"] == "Zb_s0") | {"results": results}
    _checkpoint(results, "Zb_s0", record["config"], None)
    with pytest.raises(SystemExit, match="checkpoint switches"):
        G.load_model(original, GRID, PAYLOADS)
    _checkpoint(results, "Zb_s0", unrepaired, None)
    assert G.load_model(original, GRID, PAYLOADS)[1] is None  # no original record in tmp; twin_problems was stubbed


# ---------------------------------------------------------------------------------------------------- report
def _fold(fold, swap):
    held_out = list(D.Run("BbF6", fold, 0).scored_on)
    val = {s: {"loss": {"ratio": 0.9}, "sensitivity": {"median_abs_output": 1.0, "last_swapped_over_output": swap,
                                                        "last_plus_1_over_output": swap},
               "magnitudes": {"ratio": 0.3, "n_travel": 10}} for s in held_out}
    models = {f"{label}_{fold}_{s}": {"label": label, "role": "x", "seed": s, "train": {"ratio": 0.995 if label in ("Bb", "Zb") else 0.9},
                                      "logged": {}, "val": val} for label in G.LABELS for s in range(3)}
    return {"fold": fold, "scored_on": held_out, "models": models}


def test_the_summary_keeps_each_fold_apart_and_medians_across_folds():
    summary = G.summarise([_fold(0, 0.5), _fold(2, 0.1)])
    assert set(summary["by_fold"]) == {0, 2} and list(summary["by_fold"][0]) == list(G.LABELS)
    assert summary["by_fold"][2]["BbF6"]["seeds"] == [0, 1, 2]
    overall = summary["overall"]
    assert overall["BbF6"]["folds"] == 2 and overall["BbF6"]["birds"] == 5
    assert overall["Bb"]["folds_within_1pct_of_zero"] == 2 and overall["BbF6"]["folds_within_1pct_of_zero"] == 0
    assert overall["ZbF6"]["last_swapped_over_output"] == pytest.approx(0.5)  # 3 fold-0 birds at 0.5, 2 at 0.1


def test_the_report_needs_every_fold_and_refuses_a_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(G, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="missing fold0.json"):
        G.report()
    for k in D.FOLDS:
        (tmp_path / f"fold{k}.json").write_text(json.dumps(_fold(k, 0.5) | {"dry_run": k == 4}))
    with pytest.raises(SystemExit, match="dry run"):
        G.report()


def test_the_reading_rule_is_quoted_from_this_freeze():
    rule = DG.reading_rule(G.FREEZE)
    assert "within 1% of the zero predictor" in rule and "amendment" in rule


# ---------------------------------------------------------------------------------------------------- sec 7
def test_sec7_counts_clean_runs_by_arm_names_what_is_missing_and_carries_no_placeholder(results):
    records = {run.run_id: _record(run, results) for run in D.PLAN[:-1]}
    gap = G.completeness(records, GRID, PAYLOADS)
    assert (gap["n_records"], gap["n_clean"], gap["complete"]) == (59, 59, False)
    assert gap["by_arm"]["ZbF6"] == {"planned": 30, "records": 29, "clean": 29}
    text = G.section7_markdown(records, gap, [])
    for part in ("59 of 60 recorded, 59 clean", "ZbF6_f9_s2 (index 59): no record", "gpuhopper 59",
                 "TF32 off on 59", "NVIDIA H200 59", "as registered"):
        assert part in text
    assert not PLACEHOLDER.search(text)
    assert "differ: Bb_s0: x" in G.section7_markdown(records, gap, ["Bb_s0: x"])
