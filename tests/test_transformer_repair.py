# GenAI reference: will come back to
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.diagnostics.transformer_collapse as TC
import experiments.training.transformer_repair_train as TR
from animal_movement_fm.data.multispecies import HorizonWindows
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.train.loop import TrainConfig

b1h = TR.b1h
HOUR = 3_600
PAYLOAD = b1h.load_registered()
GRID = 11


# ---------------------------------------------------------------------------------------------------- the plan
def test_the_plan_is_14_fold_6_runs_in_two_gpuhopper_blocks():
    ids = [run.run_id for run in TR.PLAN]
    assert ids == [
        "BbV0_f6_s0", "BbV1_f6_s0", "BbV2_f6_s0", "BbV3_f6_s0", "BbV4_f6_s0", "LSTMBbV0_f6_s0", "LSTMBbV3_f6_s0",
        "BbV0_f6_s0_C96", "BbV1_f6_s0_C96", "BbV2_f6_s0_C96", "BbV3_f6_s0_C96", "BbV4_f6_s0_C96",
        "LSTMBbV0_f6_s0_C96", "LSTMBbV3_f6_s0_C96",
    ]
    assert [(f, l) for f, l, *_ in TR.BLOCKS] == [(0, 9), (10, 13)]
    assert {TR.queue_of(i) for i in range(14)} == {"gpuhopper"}
    assert all(run.main_run in b1h.PLAN and run.fold == 6 and run.seed == 0 for run in TR.PLAN)
    assert TR.PLAN[11].run_id == "BbV4_f6_s0_C96"  # the calibration index in the PBS header


def test_variants_compare_with_v0_and_v0_with_the_main_study_and_the_env_rerun():
    by_id = {run.run_id: run for run in TR.PLAN}
    assert by_id["BbV3_f6_s0_C96"].compared_with == "BbV0_f6_s0_C96"
    assert by_id["BbV0_f6_s0_C96"].compared_with == "Bb_f6_s0_C96"
    assert by_id["LSTMBbV0_f6_s0"].compared_with == "LSTMBb_f6_s0"
    assert by_id["BbV0_f6_s0"].env_rerun == "BbR_f6_s0" and by_id["LSTMBbV0_f6_s0"].env_rerun == "LSTMBbR_f6_s0"
    assert all(run.env_rerun is None for run in TR.PLAN if run.variant != "V0" or run.context != 12)
    assert TR.VARIANTS["V4"] == TR.VARIANTS["V1"] | TR.VARIANTS["V2"] | TR.VARIANTS["V3"]


@pytest.mark.parametrize("index", range(14))
def test_each_config_is_its_main_study_runs_but_for_its_switches(index):
    run = TR.PLAN[index]
    names = sorted(b1h.fold_probabilities(run.fold, PAYLOAD))
    assert TR.config_differences(run, GRID, TR.REGISTERED_STEPS, names, PAYLOAD) == run.switches
    config = TR.build_config(run, GRID, TR.REGISTERED_STEPS, names, PAYLOAD)
    assert config.steps == 20_000 and config.seed == 0 and config.history == (run.context if run.context > 12 else 0)
    assert config.trunk == ("lstm" if run.base == "LSTMBb" else "transformer")
    model = config.build()
    assert (model.input_scale, model.norm_first, model.step_transform) == (
        config.input_scale, config.norm_first, config.step_transform)


# ---------------------------------------------------------------------------------------------------- guards
def test_training_reads_the_train_partition_only_at_the_runs_context(monkeypatch, tmp_path):
    seen = []
    toy = _windows(_track(tmp_path, "toy", np.arange(80)), "toy")
    monkeypatch.setattr(b1h, "split_ids", lambda stem, partition, payload: seen.append(partition) or ["toy"])
    monkeypatch.setattr(b1h, "horizon_windows", lambda stem, ids, context: seen.append(context) or toy)
    TR.training_source(TR.PLAN[7], PAYLOAD)
    assert set(seen) == {"train", 96}


def test_the_1h_split_reader_refuses_test():
    with pytest.raises(SystemExit, match="never"):
        b1h.split_ids("corvus_corone", "test", PAYLOAD)


def test_the_queue_guard_refuses_anything_but_gpuhopper_with_a_gpu(monkeypatch):
    monkeypatch.setenv("PBS_QUEUE", "gpuhopper-exec")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="trains on gpuhopper"):
        TR.queue_guard(TR.PLAN[0])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    TR.queue_guard(TR.PLAN[0])
    monkeypatch.setenv("PBS_QUEUE", "gpuvolta")
    with pytest.raises(SystemExit, match="trains on gpuhopper"):
        TR.queue_guard(TR.PLAN[0])


def test_mode_guards_hold_the_queue_and_the_budget_and_need_no_freeze(monkeypatch):
    calls = []
    monkeypatch.setattr(TR, "queue_guard", lambda run: calls.append(run.run_id))
    assert TR.mode_guards(TR.PLAN[11], calibrate=True, steps=0) is None
    with pytest.raises(SystemExit, match="20,000"):
        TR.mode_guards(TR.PLAN[0], calibrate=False, steps=40_000)
    assert TR.mode_guards(TR.PLAN[0], calibrate=False, steps=20_000) is None
    assert calls == ["BbV4_f6_s0_C96", "BbV0_f6_s0", "BbV0_f6_s0"]  # the queue is checked in every mode
    assert not any(hasattr(TR, name) for name in ("freeze_guard", "FREEZE", "PLACEHOLDER"))


def test_training_refuses_tf32(monkeypatch):
    monkeypatch.setattr(TR.env, "tf32_flags", lambda: {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False})
    monkeypatch.setattr(TR, "training_source", lambda run, payload: pytest.fail("a pool was built with TF32 on"))
    with pytest.raises(SystemExit, match="TF32 is on"):
        TR.run_torch(TR.PLAN[0], 0, 20_000, GRID, "tier2_selection.json", PAYLOAD)


def test_records_are_tagged_exploratory_and_closed_to_test():
    run = TR.PLAN[10]
    names = sorted(b1h.fold_probabilities(6, PAYLOAD))
    record = TR.base_record(run, 10, GRID, "tier2_selection.json", "fold 6", names, PAYLOAD, (1.0, 6.0, 5.0), {})
    assert record["tier"] == TR.TAG == "transformer_repair" and record["exploratory"] is True
    assert record["opened_test_partition"] is False and record["preregistered"] is False
    assert "freeze" not in record and "registered_by" not in record
    assert record["switches"] == {"step_transform": "signed_log1p"} and record["compared_with"] == "BbV0_f6_s0_C96"
    assert record["main_study_run"] == "Bb_f6_s0_C96" and record["context"] == 96 and record["queue"] == "gpuhopper"


# ---------------------------------------------------------------------------------------------------- evaluation
def test_load_run_checks_the_records_tier(tmp_path):
    run = TR.PLAN[0]
    config = TrainConfig(d_model=16, num_layers=1, nhead=2, lr=1e-3, steps=3, use_fm_package=False, horizons=(1, 6, 24))
    torch.save({"config": asdict(config), "horizon_k": [1.0, 6.0, 5.0], "context": 12,
                "model": config.build().state_dict()}, tmp_path / f"{run.run_id}_reported.pt")
    record = {"opened_test_partition": False, "context": 12, "applied_updates": 20_000, "horizon_k": [1.0, 6.0, 5.0],
              "tier": "birds_1h"}
    (tmp_path / f"{run.run_id}.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="tier"):
        TC.load_run(run, tmp_path, TR.TAG)
    (tmp_path / f"{run.run_id}.json").write_text(json.dumps(record | {"tier": TR.TAG}))
    model, k, _ = TC.load_run(run, tmp_path, TR.TAG)
    assert not model.training and k.tolist() == [1.0, 6.0, 5.0]


def test_paired_loss_is_variant_minus_v0_point_by_point():
    v0, variant = [3.0, 2.0, 2.0, 1.0], [2.0, 2.0, 3.0, 0.5]
    paired = TC.paired_loss(variant, v0, last=2)
    assert paired == {"points": 4, "median": pytest.approx(-0.25), "median_last_2": pytest.approx(0.25),
                      "share_below": 0.5}
    assert TC.paired_loss(variant, v0[:3]) is None and TC.paired_loss(None, v0) is None


def _track(tmp_path, name, bins, *, step_deg=0.01, noise=2e-4, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"segment_id": f"{name}_s0", "individual_id": name, "bin_index": int(b),
             "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(seconds=HOUR * int(b)
                                                                               + int(rng.integers(0, 3000))),
             "lon": 10.0 + step_deg * b + rng.normal(0, noise), "lat": 50.0 + rng.normal(0, noise)} for b in bins]
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _windows(path, name, context=12):
    return HorizonWindows.build(path, [name], context=context, core=12, horizons=(1, 6, 24), bin_seconds=HOUR,
                                clip=1000.0)


def _toy_model(run):
    torch.manual_seed(len(run.run_id))
    padded = run.context > 12
    lstm = run.base == "LSTMBb"
    return MovementFM(input_dim=12 + (lstm and padded), d_model=16, nhead=4, num_layers=2, trunk="lstm" if lstm else
                      "transformer", n_horizons=3, observed_flag=lstm and padded, use_fm_package=False,
                      **run.switches).eval()


def test_a_repair_fold_and_its_report_run_end_to_end_on_toy_data(tmp_path, monkeypatch):
    bins = np.concatenate(([0, 3, 9], np.arange(20, 260)))
    paths = {n: _track(tmp_path, n, bins, step_deg=0.002 * (i + 1), seed=i) for i, n in enumerate("abc")}
    rng = np.random.default_rng(0)
    histories = {run.run_id: (np.linspace(1.0, 0.5, 40) + rng.normal(0, 0.01, 40)).tolist() for run in TR.PLAN}

    def load(run, results_dir=None, tier=None):
        assert results_dir == TR.RESULTS and tier == TR.TAG
        return _toy_model(run), np.array([0.7, 6.0, 5.0]), {"loss_history": histories[run.run_id],
                                                             "head_loss_history": [[1.0, 0.5, 0.2]] * 40}

    monkeypatch.setattr(b1h, "load_registered", lambda: {})
    monkeypatch.setattr(b1h, "fold_probabilities", lambda fold, payload: {"a": 0.6, "b": 0.4})
    monkeypatch.setattr(b1h, "assert_fold_sources", lambda fold, stems: None)
    monkeypatch.setattr(b1h, "check_window_counts", lambda pool: None)
    monkeypatch.setattr(TC, "FOLDS", {6: ("c",)})
    monkeypatch.setattr(TC, "BOUNDARY_M", defaultdict(lambda: 30.0))
    monkeypatch.setattr(TC, "build", lambda stem, partition, payload, context=12: _windows(paths[stem], stem, context))
    monkeypatch.setattr(TC, "load_run", load)
    monkeypatch.setattr(TC, "REPAIR_OUT", tmp_path / "out")
    monkeypatch.setattr(TC, "REPAIR_JSON", tmp_path / "out" / "transformer_repair.json")
    monkeypatch.setattr(TC, "REPAIR_MD", tmp_path / "out" / "transformer_repair.md")

    with pytest.raises(SystemExit, match="repair probe covers"):
        TC.run_fold(3, repair=True)
    result = TC.run_fold(6, n_train=300, max_val=150, n_check=60, repair=True)
    labels = [f"{r.base}{r.variant}_C{r.context}" for r in TR.PLAN]
    assert result["mode"] == "repair" and result["labels"] == labels and result["contexts"] == [12, 96]
    assert set(result["checks"]) == set(labels) and set(result["loss_histories"]) == set(labels)
    assert {k for k, v in result["checks"].items() if "fast_path" in v} == {l for l in labels if l.startswith("Bb")}
    for label in ("BbV1_C96", "BbV2_C96", "BbV4_C96"):
        assert result["checks"][label]["masked_noise"]["max_abs_diff"] < 1e-5
        assert result["checks"][label]["fast_path"]["max_abs_diff"] < 1e-5
    assert set(result["core_only"]["BbV4_C96"]) == {"full_history", "core_only", "c12_model"}
    assert set(result["skill"]["c"]["6"]) >= {"last_step", *labels}

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "fold6.json").write_text(json.dumps(result, default=str))
    TC.repair_report()
    summary = json.loads((tmp_path / "out" / "transformer_repair.json").read_text())["summary"]
    assert summary["BbV0_C12"]["paired_loss"] is None
    expected = TC.paired_loss(histories["BbV3_f6_s0_C96"], histories["BbV0_f6_s0_C96"])
    assert summary["BbV3_C96"]["paired_loss"] == [expected]
    markdown = (tmp_path / "out" / "transformer_repair.md").read_text()
    for heading in ("## 1. Loss, input sensitivity", "## 2. Validation behaviour", "## 3. Logged training loss"):
        assert heading in markdown
    assert "| BbV4_C12 | 24 |" in markdown  # sec 2: every run x horizon
    assert "| BbV3_C96 | 6 | 40 |" in markdown and "| BbV0_C96 | 6 | 40 |" not in markdown  # sec 3: variants only


def test_the_diagnosis_mode_is_unchanged_by_the_repair_entries():
    entries = TC.diagnosis_entries(6)
    assert [e["label"] for e in entries] == [f"{a}_C{c}" for a in TC.ARMS for c in TC.CONTEXTS]
    assert [e["run_id"] for e in entries][:4] == ["Bb_f6_s0", "Bb_f6_s0_C24", "Bb_f6_s0_C48", "Bb_f6_s0_C96"]
    assert {e["label"] for e in entries if e["transformer"]} == {f"Bb_C{c}" for c in TC.CONTEXTS}
