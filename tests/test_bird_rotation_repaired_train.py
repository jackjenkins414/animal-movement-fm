# GenAI reference: tests for bird_rotation_repaired_train.py, written by GenAI from my specification; checked by me too. 
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

import experiments.training.bird_rotation_repaired_train as D
from animal_movement_fm.models.fm import signed_log1p
from animal_movement_fm.train.pool import PreparedPool

R, RT, BT, T, env = D.R, D.RT, D.BT, D.T, D.env  # the modules the driver imported
PAYLOADS = D.load_payloads()
GRID = 11
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
MAIN = PAYLOADS["birds"]["probabilities"]["main"]
ROTATION = PAYLOADS["rotation"]["probabilities"]


def _weights(fold):
    return MAIN if fold == 0 else ROTATION[str(fold)]


def _names(fold):
    return sorted(_weights(fold))


class _StubWindows:
    """A species' windows with fixed features, so no parquet is read."""

    def __init__(self, n_windows=1_000, features=None):
        self.n_windows = n_windows
        self.features = features

    def gather(self, ids, sigma=None):
        features = self.features[ids]
        return features, np.zeros((len(ids), 2)), np.ones(len(ids))


@pytest.fixture
def stub_loader(monkeypatch):
    """T._load_species_windows at the registered counts: the bird study's when it builds every training individual
    (fold 0), the rotation's when it builds them after the exclusions (folds 1-9)."""
    seen = []

    def load(stem, split_stem, partition, individuals=None):
        seen.append(partition)
        counts = PAYLOADS["birds" if individuals is None else "rotation"]["train_window_counts"]
        return _StubWindows(counts[stem])

    monkeypatch.setattr(T, "_load_species_windows", load)
    return seen


# ---------------------------------------------------------------------------------------------------- the plan
def test_the_plan_is_60_runs_of_two_arms_by_ten_folds_by_three_seeds_in_six_blocks():
    assert len(D.PLAN) == 60 and len({r.run_id for r in D.PLAN}) == 60
    for offset, arm in ((0, "BbF6"), (30, "ZbF6")):
        assert [(r.arm, r.fold, r.seed) for r in D.PLAN[offset:offset + 30]] == [
            (arm, k, s) for k in range(10) for s in range(3)]
    assert [D.PLAN[i].run_id for i in (0, 29, 30, 59)] == ["BbF6_f0_s0", "BbF6_f9_s2", "ZbF6_f0_s0", "ZbF6_f9_s2"]
    assert [(first, last) for first, last, _ in D.BLOCKS] == [(i, i + 9) for i in range(0, 60, 10)]
    assert all(last - first + 1 <= 10 for first, last, _ in D.BLOCKS)
    assert D.CALIBRATION_INDICES == (27, 57) and {D.PLAN[i].fold for i in D.CALIBRATION_INDICES} == {9}


def test_run_ids_are_new_to_every_earlier_study():
    others = {r.run_id for plan in (RT.PLAN, BT.PLAN, R.PLAN, R.bbf.PLAN, R.env.PLAN, R.TR.PLAN) for r in plan}
    assert not {r.run_id for r in D.PLAN} & others
    assert "BbF_f6_s0" in others  # the 1 h BbF id, which is why the arm here is BbF6


def test_the_repaired_encoder_is_the_pin_every_repaired_study_uses():
    assert D.SWITCHES == V4 == R.SWITCHES == R.bbf.SWITCHES == R.TR.VARIANTS["V4"]


def test_each_twin_is_the_original_of_its_fold_and_seed():
    for arm, original in (("BbF6", "Bb"), ("ZbF6", "Zb")):
        fold0 = D.Run(arm, 0, 2)
        assert fold0.twin == BT.Run(original, "", 2) and fold0.twin in BT.PLAN
        assert fold0.twin.run_id == f"{original}_s2" and fold0.twin_results == BT.RESULTS
        assert fold0.scored_on == ("larus_fuscus", "anser_albifrons", "streptopelia_turtur")
        later = D.Run(arm, 4, 1)
        assert later.twin == RT.Run(original, 4, "", 1) and later.twin in RT.PLAN
        assert later.twin.run_id == f"{original}_f4_s1" and later.twin_results == RT.RESULTS
        assert later.scored_on == RT.FOLDS[4]
    assert {run.twin.run_id for run in D.PLAN if run.fold} == {
        r.run_id for r in RT.PLAN if r.arm in ("Bb", "Zb") and r.seed in (0, 1, 2)}


# ---------------------------------------------------------------------------------------------------- configurations
@pytest.mark.parametrize("index", range(60))
def test_each_config_differs_from_its_original_in_the_switches_alone(index):
    run = D.PLAN[index]
    assert D.config_differences(run, GRID, D.REGISTERED_STEPS, _names(run.fold), PAYLOADS) == V4


def test_each_recipe_is_the_original_drivers_own_builder():
    steps = D.REGISTERED_STEPS
    for arm, original in (("BbF6", "Bb"), ("ZbF6", "Zb")):
        assert D.recipe_config(D.Run(arm, 0, 1), GRID, steps, _names(0), PAYLOADS) == BT.build_config(
            BT.Run(original, "", 1), GRID, steps, _names(0), PAYLOADS["birds"])
        assert D.recipe_config(D.Run(arm, 5, 1), GRID, steps, _names(5), PAYLOADS) == BT.build_config(
            RT.Run(original, 5, "", 1), GRID, steps, _names(5), PAYLOADS["rotation"])
    plain = D.recipe_config(D.Run("BbF6", 3, 2), GRID, steps, _names(3), PAYLOADS)
    full = D.recipe_config(D.Run("ZbF6", 3, 2), GRID, steps, _names(3), PAYLOADS)
    for config in (plain, full):
        assert (config.d_model, config.num_layers, config.lr, config.steps, config.batch_size, config.seed) == (
            256, 4, 1e-3, 20_000, 256, 2)
        assert config.trunk == "transformer"
        assert {n: getattr(config, n) for n in V4} == {"input_scale": "sqrt_d", "norm_first": False, "step_transform": "none"}
    assert not plain.use_fm_package and not plain.use_reconstruction_head
    assert full.use_fm_package and full.use_reconstruction_head and full.use_context_conditioning
    assert (full.mask_ratio, full.reconstruction_weight) == (0.15, 0.5)


@pytest.mark.parametrize("arm", list(D.ARMS))
@pytest.mark.parametrize("fold", [0, 7])
def test_same_seed_initial_weights_equal_the_originals_apart_from_the_final_layernorm(arm, fold):
    assert D.initial_state_differences(D.Run(arm, fold, 1), GRID, _names(fold), PAYLOADS) == {
        "extra": D.EXPECTED_EXTRA, "missing": [], "different": []}


def test_the_repaired_full_model_transforms_its_inputs_only_on_unclipped_6h_windows():
    config = D.build_config(D.Run("ZbF6", 2, 0), GRID, D.REGISTERED_STEPS, _names(2), PAYLOADS)
    original = D.recipe_config(D.Run("ZbF6", 2, 0), GRID, D.REGISTERED_STEPS, _names(2), PAYLOADS)
    torch.manual_seed(0)
    model = config.build().eval()
    torch.manual_seed(0)
    reference = D.replace(original, input_scale="none", norm_first=True).build().eval()
    features = torch.randn(4, 11, 12)
    features[:, -1, :2] = torch.tensor([[2_500.0, -40_000.0]])  # 6 h windows carry no +-1,000 clip
    mask = torch.zeros(4, 11, dtype=torch.bool)
    mask[:, 3] = True
    with torch.no_grad():
        for got, want in zip(model(features, token_mask=mask), reference(signed_log1p(features), token_mask=mask)):
            torch.testing.assert_close(got, want)


# ---------------------------------------------------------------------------------------------------- frozen pools
@pytest.mark.parametrize("fold", range(10))
def test_each_fold_reads_its_registered_pool_and_probabilities_from_training_individuals_only(fold, stub_loader):
    pool, node = D.training_source(D.Run("ZbF6", fold, 0), PAYLOADS)
    assert node == ("probabilities.main" if fold == 0 else f"probabilities.{fold}")
    weights = _weights(fold)
    assert pool.names == sorted(weights)
    np.testing.assert_allclose(pool.probabilities, [weights[n] for n in pool.names], rtol=0, atol=1e-12)
    assert set(stub_loader) == {"train"}
    D.check_window_counts(D.Run("ZbF6", fold, 0), pool, PAYLOADS)
    held_out = set(D.Run("ZbF6", fold, 0).scored_on)
    assert not set(pool.names) & (held_out | set(RT.SPW.BIRDS_RESERVED))
    assert not {RT.SPW.genus(n) for n in pool.names} & {RT.SPW.genus(n) for n in held_out}


def test_fold_0_is_the_bird_studys_main_pool_as_trained(stub_loader):
    pool, _ = D.training_source(D.Run("BbF6", 0, 0), PAYLOADS)
    assert len(pool.names) == 22
    assert sum(pool.species[n].n_windows for n in pool.names) == 2_009_726  # rotation freeze sec 2, "as trained"
    assert D.excluded_individuals(D.Run("BbF6", 0, 0), pool.names, PAYLOADS) == {}


def test_folds_1_to_9_carry_the_rotations_exclusions(stub_loader):
    pool, _ = D.training_source(D.Run("BbF6", 9, 0), PAYLOADS)
    assert sum(pool.species[n].n_windows for n in pool.names) == 3_156_752  # rotation freeze sec 2, fold 9
    excluded = D.excluded_individuals(D.Run("BbF6", 9, 0), pool.names, PAYLOADS)
    assert len(excluded["ciconia_ciconia"]) == 44 and len(excluded["gyps_himalayensis"]) == 10
    assert excluded == RT.excluded_in(pool.names, PAYLOADS["rotation"])


@pytest.mark.parametrize("fold", [0, 6])
def test_a_drifted_window_count_is_refused(fold, stub_loader):
    run = D.Run("BbF6", fold, 0)
    pool, _ = D.training_source(run, PAYLOADS)
    pool.species[pool.names[0]].n_windows += 1
    with pytest.raises(SystemExit, match=f"fold {fold} training windows differ"):
        D.check_window_counts(run, pool, PAYLOADS)


def test_the_loader_never_opens_a_test_partition():
    with pytest.raises(SystemExit, match="never opens a test partition"):
        T._load_species_windows("larus_fuscus", "larus_fuscus", "test")


# ---------------------------------------------------------------------------------------------------- guards
def test_mode_guards_need_the_freeze_and_the_budget_except_in_calibration(monkeypatch):
    queued, frozen = [], []
    monkeypatch.setattr(R, "queue_guard", lambda run: queued.append(run.run_id))
    monkeypatch.setattr(D, "freeze_guard", lambda: frozen.append(1) or {"freeze_commit": "abc"})
    assert D.mode_guards(D.PLAN[27], calibrate=True, steps=0) is None and not frozen
    assert D.mode_guards(D.PLAN[0], calibrate=False, steps=20_000) == {"freeze_commit": "abc"} and frozen
    with pytest.raises(SystemExit, match="20,000"):
        D.mode_guards(D.PLAN[0], calibrate=False, steps=1_000)
    assert queued == ["BbF6_f9_s0", "BbF6_f0_s0", "BbF6_f0_s0"]  # the queue is checked in every mode


def test_the_queue_guard_refuses_anything_but_gpuhopper_with_a_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("PBS_QUEUE", "gpuvolta")
    with pytest.raises(SystemExit, match="trains on gpuhopper"):
        D.mode_guards(D.PLAN[0], calibrate=True, steps=0)


def test_the_freeze_guard_points_at_this_freeze_and_restores_the_tier2_path(monkeypatch):
    seen = []
    monkeypatch.setattr(R.S2, "assert_freeze_committed", lambda: seen.append(R.S2.FREEZE) or {"freeze_commit": "x"})
    before = R.S2.FREEZE
    assert D.freeze_guard() == {"freeze_commit": "x"}
    assert seen == [D.REPO / "preregistration" / "birds_rotation_repaired_exploratory_freeze.md"]
    assert R.S2.FREEZE == before


def test_training_and_calibration_refuse_tf32_before_building_a_pool(monkeypatch):
    monkeypatch.setattr(env, "tf32_flags", lambda: {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False})
    monkeypatch.setattr(D, "training_source", lambda run, payloads: pytest.fail("a pool was built with TF32 on"))
    with pytest.raises(SystemExit, match="TF32 is on"):
        D.run_torch(D.PLAN[0], 0, GRID, "tier2_selection.json", PAYLOADS, None)
    with pytest.raises(SystemExit, match="TF32 is on"):
        D.calibrate(D.PLAN[57], GRID, PAYLOADS)


# ---------------------------------------------------------------------------------------------------- records
def test_records_are_tagged_exploratory_closed_to_test_and_name_their_original():
    record = D.base_record(D.PLAN[33], 33, GRID, "tier2_selection.json", "probabilities.1", {"freeze_commit": "a"})
    assert record["tier"] == D.TAG == "birds_rotation_repaired" and record["exploratory"] is True
    assert record["opened_test_partition"] is False and record["registered_by"] == D.FREEZE
    assert record["switches"] == V4 and record["twin"] == "Zb_f1_s0" and record["twin_study"] == "birds_rotation"
    assert record["fold"] == 1 and record["species"] is None and record["scored_on"] == list(RT.FOLDS[1])
    assert record["queue"] == "gpuhopper" and record["freeze"] == {"freeze_commit": "a"}
    fold0 = D.base_record(D.PLAN[0], 0, GRID, "s", "probabilities.main", None)
    assert fold0["twin"] == "Bb_s0" and fold0["twin_study"] == "birds" and fold0["recipe"] == "birds:Bb"


def _twin_files(tmp_path, monkeypatch, run, config, *, pool_species, pool_windows=100):
    monkeypatch.setattr(BT, "RESULTS", tmp_path)
    monkeypatch.setattr(RT, "RESULTS", tmp_path)
    for name in ("ROTATION_PREDICTIONS", "BIRD_PREDICTIONS", "TIER2_PREDICTIONS"):
        monkeypatch.setattr(D, name, tmp_path)
    record = {"config": config, "applied_updates": 20_000, "pool_species": pool_species, "pool_windows": pool_windows}
    (tmp_path / f"{run.twin.run_id}.json").write_text(json.dumps(record))
    (tmp_path / f"{run.twin.run_id}_reported.pt").write_bytes(b"")
    for path in D.prediction_files(run):
        path.write_bytes(b"")


@pytest.mark.parametrize("index", [31, 44])
def test_the_twin_check_fills_later_fields_and_names_what_differs(tmp_path, monkeypatch, index):
    run = D.PLAN[index]
    names = _names(run.fold)
    config = json.loads(json.dumps(D.asdict(D.recipe_config(run, GRID, D.REGISTERED_STEPS, names, PAYLOADS))))
    for later in ("horizons", "history", "observed_flag", "env_channels", "input_scale", "norm_first", "step_transform"):
        config.pop(later)  # written before these fields existed
    _twin_files(tmp_path, monkeypatch, run, config, pool_species=names)
    assert D.twin_problems(run, GRID, names, PAYLOADS, 100) == []
    assert D.twin_problems(run, GRID, names, PAYLOADS, 101) == [
        f"{run.twin.run_id} trained on 100 windows, this run on 101"]
    _twin_files(tmp_path, monkeypatch, run, config | {"lr": 3e-4}, pool_species=names[1:])
    assert D.twin_problems(run, GRID, names, PAYLOADS) == [
        f"{run.twin.run_id}'s recorded config differs from its builder's",
        f"{run.twin.run_id}'s pool species differ from this run's"]
    missing = D.prediction_files(run)[1]
    missing.unlink()
    assert D.twin_problems(run, GRID, names, PAYLOADS)[-1] == f"{missing.name} is missing from {tmp_path}"


def test_fold_0_reads_the_gulls_p_from_tier2_and_the_rest_from_the_bird_study():
    files = D.prediction_files(D.PLAN[0])
    assert files[0] == D.TIER2_PREDICTIONS / "larus_fuscus__P.parquet"
    assert files[1] == D.BIRD_PREDICTIONS / "larus_fuscus__Bb_s0.parquet"
    assert files[2:4] == [D.BIRD_PREDICTIONS / "anser_albifrons__P.parquet",
                          D.BIRD_PREDICTIONS / "anser_albifrons__Bb_s0.parquet"]
    later = D.prediction_files(D.PLAN[40])  # ZbF6_f3_s1
    assert later[:2] == [D.ROTATION_PREDICTIONS / "ardea_alba__P.parquet",
                         D.ROTATION_PREDICTIONS / "ardea_alba__Zb_f3_s1.parquet"]


def test_originals_on_this_machine_pass_the_twin_check():
    if not (RT.RESULTS / "Bb_f1_s0.json").exists():
        pytest.skip("the rotation records are not on this machine")
    for run in (r for r in D.PLAN if r.fold == 1):
        assert D.twin_problems(run, GRID, _names(1), PAYLOADS) == [], run.run_id


# ---------------------------------------------------------------------------------------------------- calibration
def _tiny_pool(n=512):
    generator = torch.Generator().manual_seed(0)
    return PreparedPool(
        names=("a", "b"),
        features=torch.randn(n, 11, 12, generator=generator),
        targets=torch.randn(n, 2, generator=generator),
        species_bounds=torch.tensor([[0, n // 2], [n // 2, n]]),
        probabilities=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_the_probe_scores_the_full_models_forecast_and_is_restored(monkeypatch):
    before = R.probe_losses
    with D.forecast_probe():
        assert R.probe_losses is D.probe_losses
    assert R.probe_losses is before
    config = D.replace(D.recipe_config(D.Run("ZbF6", 1, 0), GRID, 10, [], PAYLOADS), d_model=16, num_layers=1)
    model = config.build()
    features, targets = torch.randn(8, 11, 12), torch.randn(8, 2)
    got = D.probe_losses(model, (features, targets))
    assert model.training  # left as it was found
    with torch.no_grad():
        forecast = model.eval()(features)[0]
    assert got["loss"] == pytest.approx(float(R.normalised_huber(forecast, targets)))


def test_the_calibration_runs_the_full_model_and_stops_without_changing_what_a_full_run_logs(monkeypatch):
    monkeypatch.setattr(R, "CALIBRATION_STEPS", 20)
    monkeypatch.setattr(R, "LOG_EVERY", 10)
    pool = _tiny_pool()
    config = D.replace(D.build_config(D.Run("ZbF6", 1, 0), GRID, 60, [], PAYLOADS), d_model=16, num_layers=1,
                       batch_size=32)
    probe = pool.sample(torch.Generator().manual_seed(R.PROBE_SEED), 64)[:2]
    with D.forecast_probe():
        got = R.calibrate_config(config, pool, torch.device("cpu"), probe)
    full = R.train(config, pool, device="cpu", checkpoint_path=None, checkpoint_every=0, log_every=10)
    assert got["logged_steps"] == [10, 20] and got["logged_losses"] == full.loss_history[:2]
    assert got["probe"]["end"]["ratio"] > 0


# ---------------------------------------------------------------------------------------------------- dry run
def test_the_dry_run_passes_with_every_original_and_fails_without(monkeypatch, capsys):
    features = np.zeros((4, 11, 12))
    features[0, 0, 0] = 1_500.0

    def source(run, payloads):
        weights = _weights(run.fold)
        return D.SourcePool({n: _StubWindows(4, features) for n in weights}, dict(weights)), f"node{run.fold}"

    monkeypatch.setattr(D, "training_source", source)
    monkeypatch.setattr(D, "check_window_counts", lambda run, pool, payloads: None)
    monkeypatch.setattr(D, "twin_problems", lambda run, grid_index, names, payloads, windows=None: [])
    D.dry_run(PAYLOADS)
    printed = capsys.readouterr().out
    assert "60 runs" in printed and "Nothing trained; no test partition opened." in printed
    assert "birds:Zb_s2" in printed and "birds_rotation:Bb_f9_s2" in printed and "DIFFERENT" not in printed
    monkeypatch.setattr(D, "twin_problems",
                        lambda run, grid_index, names, payloads, windows=None: ["missing"] if run.fold == 0 else [])
    with pytest.raises(SystemExit, match="BbF6_f0_s0: missing"):
        D.dry_run(PAYLOADS)
