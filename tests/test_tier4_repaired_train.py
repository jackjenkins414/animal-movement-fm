# GenAI reference: tests for tier4_repaired_train.py, written by GenAI from our specification; which is... add. 
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

import experiments.training.tier4_repaired_train as R
from animal_movement_fm.models.fm import signed_log1p
from animal_movement_fm.train.pool import PreparedPool

T, T3, T4, TR, bbf, env = R.T, R.T3, R.T4, R.TR, R.bbf, R.env  # the modules the driver imported
PAYLOADS = R.load_payloads()
GRID = 11
V4 = {"input_scale": "none", "norm_first": True, "step_transform": "signed_log1p"}
NES, WOLF = "mirounga_angustirostris", "canis_lupus"
NODES = {
    "B3F": ("tier2:source_pool", {stem for stem, *_ in T.SOURCES}),
    "B9F": ("source_pool", set(PAYLOADS["3"]["source_pool"])),
    "B26F": ("source_pool", set(PAYLOADS["4"]["source_pool"])),
    "BdMF": ("sub_pools.domain_marine", set(PAYLOADS["4"]["sub_pools"]["domain_marine"])),
    "BdTF": ("sub_pools.domain_terrestrial", set(PAYLOADS["4"]["sub_pools"]["domain_terrestrial"])),
}


def _names(arm):
    return sorted(NODES[arm][1])


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
    seen = []

    def load(stem, split_stem, partition, individuals=None):
        seen.append(partition)
        return _StubWindows(PAYLOADS["4"]["train_window_counts"][stem])

    monkeypatch.setattr(T, "_load_species_windows", load)
    return seen


# ---------------------------------------------------------------------------------------------------- the plan
def test_the_plan_is_25_runs_of_five_arms_by_five_seeds_in_three_blocks():
    assert len(R.PLAN) == 25 and len({r.run_id for r in R.PLAN}) == 25
    for first, arm in enumerate(("B3F", "B9F", "B26F", "BdMF", "BdTF")):
        block = R.PLAN[5 * first:5 * first + 5]
        assert [(r.arm, r.seed) for r in block] == [(arm, s) for s in range(5)]
    assert [(first, last) for first, last, _ in R.BLOCKS] == [(0, 9), (10, 19), (20, 24)]
    assert [R.PLAN[i].run_id for i in (0, 5, 10, 15, 20)] == [
        "B3F_s0", "B9F_s0", "B26F_s0", "BdMF_mirounga_angustirostris_s0", "BdTF_canis_lupus_s0"]
    others = {r.run_id for plan in (T.PLAN, T.TIER_E_PLAN, T3.PLAN, T3.TIER_E_PLAN, T4.PLAN) for r in plan}
    assert not {r.run_id for r in R.PLAN} & others


def test_the_repaired_encoder_is_the_probes_v4_and_bbfs_pin():
    assert R.SWITCHES == V4 == TR.VARIANTS["V4"] == bbf.SWITCHES


def test_twins_recipes_pools_and_scored_species():
    by_arm = {arm: R.Run(arm, 3) for arm in R.ARMS}
    assert by_arm["B3F"].twin == T.Run("B", "", 3) and by_arm["B3F"].twin in T.PLAN
    assert by_arm["B9F"].twin == T3.Run("B9", "", 3) and by_arm["B9F"].twin in T3.PLAN
    assert by_arm["B26F"].twin == T4.Run("B26", "", 3) and by_arm["B26F"].twin in T4.PLAN
    assert by_arm["BdMF"].twin == T4.Run("BdM", NES, 3) and by_arm["BdMF"].twin in T4.PLAN
    bdtf = by_arm["BdTF"]
    assert bdtf.twin is None and bdtf.pool_run == T4.Run("ZdT", WOLF, 3) and bdtf.recipe == T4.Run("B26", "", 3)
    assert bdtf.comparators == ("B26F_s3", "ZdT_canis_lupus_s3")
    assert all(by_arm[a].comparators == (by_arm[a].twin.run_id,) for a in ("B3F", "B9F", "B26F", "BdMF"))
    assert by_arm["BdMF"].scored_on == (NES,) and bdtf.scored_on == (WOLF,)
    assert by_arm["B26F"].scored_on == ("larus_fuscus", WOLF, NES)


# ---------------------------------------------------------------------------------------------------- configurations
@pytest.mark.parametrize("index", range(25))
def test_each_config_differs_from_its_recipe_in_the_switches_alone(index):
    run = R.PLAN[index]
    assert R.config_differences(run, GRID, R.REGISTERED_STEPS, _names(run.arm), PAYLOADS) == V4


def test_each_recipe_is_its_registering_tiers_own_builder():
    steps = R.REGISTERED_STEPS
    assert R.recipe_config(R.Run("B3F", 2), GRID, steps, [], PAYLOADS) == T.build_config(T.Run("B", "", 2), GRID, steps)
    assert R.recipe_config(R.Run("B9F", 2), GRID, steps, _names("B9F"), PAYLOADS) == T3.build_config(
        T3.Run("B9", "", 2), GRID, steps, _names("B9F"), PAYLOADS["3"])
    for arm, twin in (("B26F", T4.Run("B26", "", 2)), ("BdMF", T4.Run("BdM", NES, 2)), ("BdTF", T4.Run("B26", "", 2))):
        assert R.recipe_config(R.Run(arm, 2), GRID, steps, _names(arm), PAYLOADS) == T4.build_config(
            twin, GRID, steps, _names(arm), PAYLOADS["4"])
    recipes = {R.recipe_config(R.Run(a, 2), GRID, steps, _names(a), PAYLOADS) for a in R.ARMS}
    assert len(recipes) == 1  # every arm is the same plain trunk before the switches; only the pool differs
    config = recipes.pop()
    assert (config.d_model, config.num_layers, config.lr, config.steps, config.seed) == (256, 4, 1e-3, 20_000, 2)
    assert not config.use_fm_package and not config.use_reconstruction_head and config.trunk == "transformer"


@pytest.mark.parametrize("arm", list(R.ARMS))
def test_same_seed_initial_weights_equal_the_recipes_apart_from_the_final_layernorm(arm):
    assert R.initial_state_differences(R.Run(arm, 1), GRID, _names(arm), PAYLOADS) == {
        "extra": R.EXPECTED_EXTRA, "missing": [], "different": []}


def test_the_model_applies_signed_log1p_to_the_two_step_channels_of_unclipped_6h_windows():
    config = R.build_config(R.Run("BdMF", 0), GRID, R.REGISTERED_STEPS, _names("BdMF"), PAYLOADS)
    torch.manual_seed(0)
    model = config.build().eval()
    plain = copy.deepcopy(model)
    plain.step_transform = "none"
    features = torch.randn(4, 11, 12)
    features[:, -1, :2] = torch.tensor([[2_500.0, -40_000.0]])  # 6 h windows carry no +-1,000 clip
    with torch.no_grad():
        assert torch.allclose(model(features), plain(signed_log1p(features)))
    transformed = signed_log1p(features)
    assert torch.equal(transformed[..., 2:], features[..., 2:])
    assert torch.allclose(transformed[0, -1, :2], torch.tensor([np.log1p(2_500.0), -np.log1p(40_000.0)]).float())


# ---------------------------------------------------------------------------------------------------- pools
@pytest.mark.parametrize("arm", list(R.ARMS))
def test_each_arm_reads_its_registered_node_from_training_individuals_only(arm, stub_loader):
    pool, node = R.training_source(R.Run(arm, 0), PAYLOADS)
    assert (node, set(pool.names)) == NODES[arm]
    assert set(stub_loader) == {"train"}
    R.check_window_counts(pool, PAYLOADS)
    assert not set(pool.names) & (set(R.HELD_OUT) | set(T3.RESERVED))


def test_the_loader_never_opens_a_test_partition():
    with pytest.raises(SystemExit, match="never opens a test partition"):
        T._load_species_windows("canis_lupus", "canis_lupus", "test")


def test_a_drifted_window_count_is_refused(stub_loader):
    pool, _ = R.training_source(R.Run("BdMF", 0), PAYLOADS)
    pool.species["mirounga_leonina"].n_windows += 1
    with pytest.raises(SystemExit, match="mirounga_leonina: built"):
        R.check_window_counts(pool, PAYLOADS)


def test_step_magnitudes_before_and_after_the_transform():
    big = np.zeros((3, 11, 12))
    big[0, 0, 0], big[1, 5, 1], big[2, 3, 0] = 5_000.0, -150.0, 20.0
    small = np.zeros((2, 11, 12))
    small[0, 0, 1] = 3.0
    pool = R.SourcePool({"a": _StubWindows(3, big), "b": _StubWindows(2, small)}, {"a": 0.5, "b": 0.5})
    cache = {}
    steps = R.step_magnitudes(pool, cache)
    assert steps["tokens"] == 55 and steps["levels"] == [10.0, 100.0, 1_000.0]
    assert steps["before"]["max"] == 5_000.0 and steps["before"]["max_species"] == "a"
    assert steps["before"]["share_above"] == [3 / 55, 2 / 55, 1 / 55]
    assert steps["after"]["max"] == pytest.approx(np.log1p(5_000.0))
    assert steps["after"]["share_above"] == [0.0, 0.0, 0.0]
    assert set(cache) == {"a", "b"}
    pool.species["a"].features = None  # a cached species is not gathered again
    assert R.step_magnitudes(pool, cache) == steps


# ---------------------------------------------------------------------------------------------------- guards
def test_the_queue_guard_refuses_anything_but_gpuhopper_with_a_gpu(monkeypatch):
    for cuda, queue, ok in [(True, "gpuhopper", True), (True, "gpuhopper-exec", True), (True, "gpuvolta", False),
                            (False, "gpuhopper", False), (True, "normal", False), (True, "", False)]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda cuda=cuda: cuda)
        monkeypatch.setenv("PBS_QUEUE", queue)
        if ok:
            R.queue_guard(R.PLAN[0])
        else:
            with pytest.raises(SystemExit, match="trains on gpuhopper"):
                R.queue_guard(R.PLAN[0])


def test_mode_guards_need_the_freeze_and_the_budget_except_in_calibration(monkeypatch):
    queued, frozen = [], []
    monkeypatch.setattr(R, "queue_guard", lambda run: queued.append(run.run_id))
    monkeypatch.setattr(R, "freeze_guard", lambda: frozen.append(1) or {"freeze_commit": "abc"})
    assert R.mode_guards(R.PLAN[10], calibrate=True, steps=0) is None and not frozen
    assert R.mode_guards(R.PLAN[0], calibrate=False, steps=20_000) == {"freeze_commit": "abc"} and frozen
    with pytest.raises(SystemExit, match="20,000"):
        R.mode_guards(R.PLAN[0], calibrate=False, steps=1_000)
    assert queued == ["B26F_s0", "B3F_s0", "B3F_s0"]  # the queue is checked in every mode


def test_the_freeze_must_exist_and_the_tier2_freeze_path_is_restored(monkeypatch):
    monkeypatch.setattr(R, "FREEZE", "no_such_freeze.md")
    before = R.S2.FREEZE
    with pytest.raises(SystemExit, match="does not exist"):
        R.freeze_guard()
    assert R.S2.FREEZE == before


def test_training_and_calibration_refuse_tf32_before_building_a_pool(monkeypatch):
    monkeypatch.setattr(env, "tf32_flags", lambda: {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False})
    monkeypatch.setattr(R, "training_source", lambda run, payloads: pytest.fail("a pool was built with TF32 on"))
    with pytest.raises(SystemExit, match="TF32 is on"):
        R.run_torch(R.PLAN[0], 0, GRID, "tier2_selection.json", PAYLOADS, None)
    with pytest.raises(SystemExit, match="TF32 is on"):
        R.calibrate(R.PLAN[10], GRID, PAYLOADS)


# ---------------------------------------------------------------------------------------------------- records
def test_records_are_tagged_exploratory_and_closed_to_test_and_never_read_as_tier4():
    import experiments.reporting.tier4_sealed_eval as T4E

    record = R.base_record(R.PLAN[15], 15, GRID, "tier2_selection.json", "sub_pools.domain_marine", {"freeze_commit": "a"})
    assert record["tier"] == R.TAG == "tier4_repaired" and record["exploratory"] is True
    assert record["opened_test_partition"] is False and record["registered_by"] == R.FREEZE
    assert record["switches"] == V4 and record["twin"] == "BdM_mirounga_angustirostris_s0"
    assert record["twin_tier"] == "4" and record["recipe"] == "tier4:BdM" and record["scored_on"] == [NES]
    assert record["queue"] == "gpuhopper" and record["freeze"] == {"freeze_commit": "a"}
    assert not T4E.is_tier4(record)
    bdtf = R.base_record(R.PLAN[20], 20, GRID, "s", "sub_pools.domain_terrestrial", None)
    assert bdtf["twin"] is None and bdtf["comparators"] == ["B26F_s0", "ZdT_canis_lupus_s0"]


def _twin_files(tmp_path, monkeypatch, run, config):
    monkeypatch.setattr(R, "TWIN_RESULTS", {"2": tmp_path, "3": tmp_path, "4": tmp_path})
    monkeypatch.setattr(R, "SEALED_PREDICTIONS", {"2": tmp_path, "3": tmp_path, "4": tmp_path})
    (tmp_path / f"{run.twin.run_id}.json").write_text(json.dumps({"config": config, "applied_updates": 20_000}))
    (tmp_path / f"{run.twin.run_id}_reported.pt").write_bytes(b"")
    for species in run.scored_on:
        for tag in ("P", run.twin.run_id):
            (tmp_path / f"{species}__{tag}.parquet").write_bytes(b"")


def test_the_twin_check_fills_later_fields_at_their_defaults_and_names_what_differs(tmp_path, monkeypatch):
    run = R.PLAN[10]
    names = _names("B26F")
    config = json.loads(json.dumps(R.asdict(R.recipe_config(run, GRID, R.REGISTERED_STEPS, names, PAYLOADS))))
    for later in ("horizons", "history", "observed_flag", "env_channels", "input_scale", "norm_first", "step_transform"):
        config.pop(later)  # written before these fields existed
    _twin_files(tmp_path, monkeypatch, run, config)
    assert R.twin_problems(run, GRID, names, PAYLOADS) == []
    _twin_files(tmp_path, monkeypatch, run, config | {"lr": 3e-4})
    assert R.twin_problems(run, GRID, names, PAYLOADS) == ["B26_s0's recorded config differs from its builder's"]
    (tmp_path / "canis_lupus__B26_s0.parquet").unlink()
    (tmp_path / "B26_s0_reported.pt").unlink()
    assert R.twin_problems(run, GRID, names, PAYLOADS)[1:] == [
        "B26_s0_reported.pt is missing", f"canis_lupus__B26_s0.parquet is missing from {tmp_path}"]


def test_bdtf_checks_p_and_zdt_predictions_only(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "SEALED_PREDICTIONS", {"2": tmp_path, "3": tmp_path, "4": tmp_path})
    assert R.twin_problems(R.PLAN[22], GRID, _names("BdTF"), PAYLOADS) == [
        f"canis_lupus__P.parquet is missing from {tmp_path}",
        f"canis_lupus__ZdT_canis_lupus_s2.parquet is missing from {tmp_path}"]


def test_tier2_twins_on_this_machine_pass_the_check():
    if not (T.TIER2_FINAL_RESULTS / "B_s0.json").exists():
        pytest.skip("the Tier 2 records are not on this machine")
    for run in R.PLAN[:5]:
        assert R.twin_problems(run, GRID, _names("B3F"), PAYLOADS) == [], run.run_id


# ---------------------------------------------------------------------------------------------------- calibration
def test_loss_match_reports_max_difference_and_the_exact_share():
    record = {"loss_history": [0.5, 0.25, 0.125]}
    assert R.loss_match([0.5, 0.25], record) == {
        "compared": 2, "record_losses": [0.5, 0.25], "max_abs_difference": 0.0, "exact_share": 1.0}
    got = R.loss_match([0.5, 0.26], record)
    assert got["max_abs_difference"] == pytest.approx(0.01) and got["exact_share"] == 0.5
    assert R.loss_match([0.5, 0.25], {"loss_history": [0.5]})["compared"] == 0


def _tiny_pool(n=512):
    generator = torch.Generator().manual_seed(0)
    return PreparedPool(
        names=("a", "b"),
        features=torch.randn(n, 11, 12, generator=generator),
        targets=torch.randn(n, 2, generator=generator),
        species_bounds=torch.tensor([[0, n // 2], [n // 2, n]]),
        probabilities=torch.tensor([0.5, 0.5], dtype=torch.float64),
    )


def test_the_calibration_stops_early_without_changing_what_a_full_run_logs(monkeypatch):
    monkeypatch.setattr(R, "CALIBRATION_STEPS", 20)
    monkeypatch.setattr(R, "LOG_EVERY", 10)
    pool = _tiny_pool()
    config = R.replace(R.recipe_config(R.Run("B26F", 0), GRID, 60, [], PAYLOADS), d_model=16, num_layers=1, batch_size=32)
    probe = pool.sample(torch.Generator().manual_seed(R.PROBE_SEED), 64)[:2]
    got = R.calibrate_config(R.replace(config, **V4), pool, torch.device("cpu"), probe)
    full = R.train(R.replace(config, **V4), pool, device="cpu", checkpoint_path=None, checkpoint_every=0, log_every=10)
    assert got["logged_steps"] == [10, 20] and got["logged_losses"] == full.loss_history[:2]
    assert got["probe"]["start"]["zero"] == got["probe"]["end"]["zero"] and got["probe"]["end"]["ratio"] > 0
    assert R.loss_match(got["logged_losses"], {"loss_history": full.loss_history})["exact_share"] == 1.0


# ---------------------------------------------------------------------------------------------------- dry run
def test_the_dry_run_passes_with_every_twin_and_fails_without(monkeypatch, capsys):
    features = np.zeros((4, 11, 12))
    features[0, 0, 0] = 1_500.0
    monkeypatch.setattr(R, "training_source", lambda run, payloads: (
        R.SourcePool({n: _StubWindows(4, features) for n in NODES[run.arm][1]}, {n: 1.0 for n in NODES[run.arm][1]}),
        NODES[run.arm][0]))
    monkeypatch.setattr(R, "check_window_counts", lambda pool, payloads: None)
    monkeypatch.setattr(R, "twin_problems", lambda run, grid_index, names, payloads: [])
    R.dry_run(PAYLOADS)
    printed = capsys.readouterr().out
    assert "25 runs" in printed and "Nothing trained; no test partition opened." in printed
    assert "sub_pools.domain_terrestrial" in printed and "none; read against B26F_s0, ZdT_canis_lupus_s0" in printed
    monkeypatch.setattr(R, "twin_problems", lambda run, grid_index, names, payloads: ["missing"] if run.twin else [])
    with pytest.raises(SystemExit, match="B3F_s0: missing"):
        R.dry_run(PAYLOADS)
