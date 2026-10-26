# GenAI reference: will write my prompt in short
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch

import experiments.training.bird_1h_bbf_train as F
from animal_movement_fm.data import era5
from animal_movement_fm.models.fm import signed_log1p

b1h, env, TR = F.b1h, F.env, F.TR  # the modules the driver imported, so monkeypatching reaches them
PAYLOAD = b1h.load_registered()
GRID = 11
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}


def _names(run):
    return sorted(b1h.fold_probabilities(run.fold, PAYLOAD))


# ---------------------------------------------------------------------------------------------------- the plan
# sha256 of "run_id|compared_with|repeats" over indices 0-119, taken from the plan before the extension was appended
ORIGINAL_SHA256 = "4874a24d6324f9a2b646449d481797e20fd9c98941f957c5a7a9e5c3f5d43f9b"


def test_the_plan_is_210_gpuhopper_runs_with_lstmbbr96_closing_the_original_120():
    plan = F.PLAN
    assert len(plan) == 210 and len({r.run_id for r in plan}) == 210
    for (arm, context), first in ((("BbF", 12), 0), (("BbF", 96), 30), (("BbFE", 12), 60), (("LSTMBbR", 96), 90),
                                  (("BbF", 24), 120), (("BbF", 48), 150), (("BbF", 72), 180)):
        block = plan[first:first + 30]
        assert {(r.arm, r.context) for r in block} == {(arm, context)}
        assert [(r.fold, r.seed) for r in block] == [(k, s) for k in range(10) for s in (0, 1, 2)]
    assert [(f, l) for f, l, *_ in F.BLOCKS] == [(0, 29), (30, 59), (60, 89), (90, 119), (120, 149), (150, 179),
                                                 (180, 209)]
    assert {F.queue_of(i) for i in range(210)} == {"gpuhopper"}
    assert F.ORIGINAL == plan[:120] and F.EXTENSION == plan[120:]
    assert plan[90:120] == F.LSTMBBR96 and len(F.LSTMBBR96) == 30
    ids = {r.run_id for r in plan}
    assert not ids & ({r.run_id for r in b1h.PLAN} | {r.run_id for r in env.PLAN} | {r.run_id for r in TR.PLAN})
    assert [plan[i].run_id for i in (18, 48, 78, 108, 138, 168, 198)] == [
        "BbF_f6_s0", "BbF_f6_s0_C96", "BbFE_f6_s0", "LSTMBbR_f6_s0_C96", "BbF_f6_s0_C24", "BbF_f6_s0_C48",
        "BbF_f6_s0_C72"]


def test_indices_0_to_119_are_unchanged_by_the_extension():
    text = "\n".join(f"{r.run_id}|{r.compared_with}|{r.repeats}" for r in F.PLAN[:120])
    assert hashlib.sha256(text.encode()).hexdigest() == ORIGINAL_SHA256


def test_the_extension_is_bbf_on_the_ablations_runs_at_c_24_48_72():
    assert F.EXTENSION_CONTEXTS == (24, 48, 72) and set(F.EXTENSION_CONTEXTS) < set(b1h.ABLATION_CONTEXTS)
    for run in F.EXTENSION:
        assert run.switches == V4 and not run.env and run.env_reference is None
        assert run.base == b1h.Run("Bb", run.fold, "", run.seed, 0, run.context) and run.base in b1h.PLAN
        assert run.compared_with == run.base.run_id == f"Bb_f{run.fold}_s{run.seed}_C{run.context}"
        assert run.repeats is None  # the probe ran C = 12 and 96 only
        assert F.queue_of(F.PLAN.index(run)) == "gpuhopper"


def test_comparators_twins_and_repeats():
    by_id = {r.run_id: r for r in F.PLAN}
    assert by_id["BbF_f3_s1"].compared_with == "BbR_f3_s1"  # the env study's gpuhopper rerun
    assert by_id["BbF_f3_s1_C96"].compared_with == "Bb_f3_s1_C96"  # the 1 h ablation's run
    assert by_id["BbFE_f3_s1"].compared_with == "BbF_f3_s1"
    assert by_id["LSTMBbR_f3_s1_C96"].compared_with == "LSTMBb_f3_s1_C96"
    assert by_id["BbFE_f3_s1"].env_reference == "BbE_f3_s1"
    assert all(r.env_reference is None for r in F.PLAN if r.arm != "BbFE")
    repeats = {r.run_id: r.repeats for r in F.PLAN if r.repeats}
    assert repeats == {"BbF_f6_s0": "BbV4_f6_s0", "BbF_f6_s0_C96": "BbV4_f6_s0_C96",
                       "LSTMBbR_f6_s0_C96": "LSTMBbV0_f6_s0_C96"}
    assert all(r.base in b1h.PLAN for r in F.PLAN)


def test_the_repaired_encoder_is_the_probes_v4_pinned():
    assert F.SWITCHES == V4 == TR.VARIANTS["V4"]
    assert all(r.switches == V4 for r in F.PLAN if r.arm in ("BbF", "BbFE"))
    assert all(r.switches == {} for r in F.LSTMBBR96)


# ---------------------------------------------------------------------------------------------------- configs
def test_each_config_differs_from_its_1h_base_only_in_the_switches_and_bbfes_env_channels():
    for index, run in enumerate(F.PLAN):
        differences = F.config_differences(run, GRID, F.REGISTERED_STEPS, _names(run), PAYLOAD)
        assert differences == F.expected_differences(run), index
    assert F.expected_differences(F.PLAN[60]) == V4 | {"env_channels": era5.N_ENV_CHANNELS}
    assert F.expected_differences(F.PLAN[90]) == {}


@pytest.mark.parametrize("fold, seed", [(0, 0), (6, 2), (9, 1)])
def test_bbfe_is_its_bbe_twin_with_the_switches_and_bbf_with_env_channels(fold, seed):
    bbfe, bbf = F.BbfRun("BbFE", fold, seed), F.BbfRun("BbF", fold, seed)
    names = _names(bbfe)
    got = asdict(F.build_config(bbfe, GRID, F.REGISTERED_STEPS, names, PAYLOAD))
    twin = asdict(env.build_config(env.EnvRun("BbE", fold, seed), GRID, F.REGISTERED_STEPS, names, PAYLOAD))
    plain = asdict(F.build_config(bbf, GRID, F.REGISTERED_STEPS, names, PAYLOAD))
    assert {k: v for k, v in got.items() if v != twin[k]} == V4
    assert {k: v for k, v in got.items() if v != plain[k]} == {"env_channels": era5.N_ENV_CHANNELS}


@pytest.mark.parametrize("index", [0, 48, 78, 108, 120, 165, 198])
def test_the_models_build_with_the_switches(index):
    run = F.PLAN[index]
    config = F.build_config(run, GRID, F.REGISTERED_STEPS, _names(run), PAYLOAD)
    assert config.steps == 20_000 and config.seed == run.seed and config.horizons == (1, 6, 24)
    assert config.history == (run.context if run.context > 12 else 0)
    assert config.observed_flag == (run.context > 12 and run.arm == "LSTMBbR")
    assert config.trunk == ("lstm" if run.arm == "LSTMBbR" else "transformer")
    model = config.build()
    assert (model.input_scale, model.norm_first, model.step_transform) == (
        config.input_scale, config.norm_first, config.step_transform)
    assert model.input_proj.in_features == 12 + (15 if run.env else 0) + (run.arm == "LSTMBbR")


def test_the_step_transform_touches_the_two_step_channels_only_env_channels_included():
    x = torch.randn(4, 11, 27) * 50
    y = signed_log1p(x)
    torch.testing.assert_close(y[..., :2], torch.sign(x[..., :2]) * torch.log1p(x[..., :2].abs()))
    assert torch.equal(y[..., 2:], x[..., 2:])
    run = F.PLAN[60]
    model = F.build_config(run, GRID, F.REGISTERED_STEPS, _names(run), PAYLOAD).build().eval()
    with torch.no_grad():
        assert model(x).shape == (4, 3, 2)


def test_the_fold_6_seed_0_runs_have_the_probe_runs_configs():
    for run in F.PLAN:
        if not run.repeats:
            continue
        path = TR.RESULTS / f"{run.repeats}.json"
        if not path.exists():
            pytest.skip("the repair probe's records are not on this machine")
        mine = json.loads(json.dumps(asdict(F.build_config(run, GRID, F.REGISTERED_STEPS, _names(run), PAYLOAD))))
        assert mine == json.loads(path.read_text())["config"], run.run_id


# ---------------------------------------------------------------------------------------------------- guards
def test_the_queue_guard_refuses_anything_but_gpuhopper_with_a_gpu(monkeypatch):
    for cuda, queue, ok in [(True, "gpuhopper", True), (True, "gpuhopper-exec", True), (True, "gpuvolta", False),
                            (False, "gpuhopper", False), (False, "normal", False), (True, "", False)]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda cuda=cuda: cuda)
        monkeypatch.setenv("PBS_QUEUE", queue)
        if ok:
            F.queue_guard(F.PLAN[0])
        else:
            with pytest.raises(SystemExit, match="trains on gpuhopper"):
                F.queue_guard(F.PLAN[0])


def test_mode_guards_need_the_freeze_and_the_budget_except_in_calibration(monkeypatch):
    queued, frozen = [], []
    monkeypatch.setattr(F, "queue_guard", lambda run: queued.append(run.run_id))
    monkeypatch.setattr(F, "freeze_guard", lambda: frozen.append(1) or {"freeze_commit": "abc"})
    assert F.mode_guards(F.PLAN[48], calibrate=True, steps=0) is None and not frozen
    assert F.mode_guards(F.PLAN[0], calibrate=False, steps=20_000) == {"freeze_commit": "abc"} and frozen
    with pytest.raises(SystemExit, match="20,000"):
        F.mode_guards(F.PLAN[0], calibrate=False, steps=1_000)
    assert queued == ["BbF_f6_s0_C96", "BbF_f0_s0", "BbF_f0_s0"]  # the queue is checked in every mode


def test_the_freeze_must_exist_and_the_tier2_freeze_path_is_restored(monkeypatch):
    monkeypatch.setattr(F, "FREEZE", "no_such_freeze.md")
    before = b1h.S2.FREEZE
    with pytest.raises(SystemExit, match="does not exist"):
        F.freeze_guard()
    assert b1h.S2.FREEZE == before
    assert F.FREEZE != env.FREEZE


def test_the_driver_never_reads_a_test_partition(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="never 'test'"):
        b1h.split_ids("anas_platyrhynchos", "test", PAYLOAD)
    seen = []
    monkeypatch.setattr(b1h, "split_ids", lambda stem, partition, payload: seen.append(partition) or ["x"])
    monkeypatch.setattr(b1h, "horizon_windows", lambda stem, ids, context: seen.append(context) or None)
    monkeypatch.setattr(b1h, "SourcePool", lambda species, weights: "pool")
    assert F.training_source(F.PLAN[48], PAYLOAD) == ("pool", "fold 6", None)
    assert set(seen) == {"train", 96}


def test_bbfe_takes_the_env_studys_pool_for_its_twin(monkeypatch):
    calls = []
    monkeypatch.setattr(env, "training_source", lambda run, payload: calls.append(run) or ("pool", "fold 4", "stats"))
    assert F.training_source(F.BbfRun("BbFE", 4, 2), PAYLOAD) == ("pool", "fold 4", "stats")
    assert calls == [env.EnvRun("BbE", 4, 2)]


def test_training_refuses_tf32_before_building_a_pool(monkeypatch):
    monkeypatch.setattr(env, "tf32_flags", lambda: {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False})
    monkeypatch.setattr(F, "training_source", lambda run, payload: pytest.fail("a pool was built with TF32 on"))
    with pytest.raises(SystemExit, match="TF32 is on"):
        F.run_torch(F.PLAN[0], 0, 20_000, GRID, "tier2_selection.json", PAYLOAD, None)


def _twin(run, stats, files, **changes):
    record = {"run_id": run.env_reference, "env_clip_sd": era5.CLIP_SD, "env_channels": era5.N_ENV_CHANNELS,
              "env_stats": stats, "env_files": files} | changes
    (env.RESULTS / f"{run.env_reference}.json").write_text(json.dumps(record))


def test_bbfe_is_refused_unless_its_env_inputs_equal_its_twins(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "RESULTS", tmp_path)
    run = F.PLAN[70]
    stats = {"channels": list(era5.ENV_CHANNELS[:14]), "mean": [0.5] * 14, "sd": [2.0] * 14}
    files = {"a": "1" * 64, "b": "2" * 64}
    assert F.env_reference_problems(run, stats, files) == [f"{run.env_reference}'s record "
                                                           f"{tmp_path / (run.env_reference + '.json')} is missing"]
    _twin(run, stats, files)
    assert F.env_reference_problems(run, stats, files) == []
    assert F.env_reference_problems(run, stats | {"sd": [2.0] * 13 + [2.1]}, files) == [
        f"env_stats differ from {run.env_reference}'s"]
    assert F.env_reference_problems(run, stats, files | {"b": "3" * 64}) == [
        f"env file sha256 differ from {run.env_reference}'s"]
    _twin(run, stats, files, env_clip_sd=None)
    assert "clip None" in F.env_reference_problems(run, stats, files)[0]


def test_records_are_tagged_exploratory_registered_and_closed_to_test():
    run = F.PLAN[78]
    stats = (np.zeros(14), np.ones(14))
    record = F.base_record(run, 78, GRID, "tier2_selection.json", "fold 6", _names(run), PAYLOAD, (1.0, 6.0, 5.0), {},
                           stats, {"a": "1" * 64}, None, {"freeze_commit": "abc"})
    assert record["tier"] == F.TAG == "birds_1h_bbf" and record["exploratory"] is True
    assert record["opened_test_partition"] is False and record["registered_by"] == F.FREEZE
    assert record["freeze"] == {"freeze_commit": "abc"} and record["queue"] == "gpuhopper"
    assert record["switches"] == V4 and record["env"] is True and record["env_channels"] == 15
    assert record["env_clip_sd"] == era5.CLIP_SD and record["env_reference"] == "BbE_f6_s0"
    assert record["main_study_run"] == "Bb_f6_s0" and record["compared_with"] == "BbF_f6_s0"
    assert record["env_stats"]["channels"] == list(era5.ENV_CHANNELS[:14])
    plain = F.base_record(F.PLAN[108], 108, GRID, "s", "fold 6", _names(run), PAYLOAD, (1.0, 6.0, 5.0), {}, None,
                          None, None, None)
    assert plain["switches"] == {} and plain["env_clip_sd"] is None and plain["repeats"] == "LSTMBbV0_f6_s0_C96"


def test_the_dry_run_passes_with_every_twin_and_probe_record_and_fails_without(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(env, "RESULTS", tmp_path / "env")
    monkeypatch.setattr(TR, "RESULTS", tmp_path / "probe")
    env.RESULTS.mkdir()
    TR.RESULTS.mkdir()
    with pytest.raises(SystemExit, match="(?s)references missing.*BbV4_f6_s0.json is missing.*BbE_f0_s0"):
        F.dry_run(PAYLOAD)
    for run in F.PLAN:
        if run.env:
            _twin(run, None, None)
        if run.repeats:
            (TR.RESULTS / f"{run.repeats}.json").write_text("{}")
    F.dry_run(PAYLOAD)
    assert "210 runs: configs as planned" in capsys.readouterr().out
