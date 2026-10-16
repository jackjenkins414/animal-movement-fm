# GenAI reference: I specified these tests for the env study's driver from what I outlined, that being the queue, freeze, and guards, env stats, training
# only etc, and I briefly skimmed over these.
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.training.bird_1h_env_train as env
from animal_movement_fm.data import era5
from animal_movement_fm.data.multispecies import SourcePool
from animal_movement_fm.train.loop import TrainConfig
from animal_movement_fm.train.pool import env_stats, horizon_k

b1h = env.b1h  # the module the driver imported, so monkeypatching reaches it
PAYLOAD = json.loads(b1h.RT.REGISTERED_FOLDS.read_text())
GRID = 11
HOUR = 3600
WEIGHTS = {"x": 0.6, "y": 0.4}


# ----------------------------------------------------------------------------- plan and configuration
def test_plan_layout_queues_and_ids():
    plan = env.PLAN
    assert len(plan) == 150 and len({r.run_id for r in plan}) == 150
    assert not {r.run_id for r in plan} & {r.run_id for r in b1h.PLAN}
    expected = [("BbE", 0, 29), ("LSTMBbE", 30, 59), ("BbR", 60, 89), ("LSTMBbR", 90, 119), ("XGBBbE", 120, 149)]
    for arm, first, last in expected:
        block = plan[first:last + 1]
        assert {r.arm for r in block} == {arm}
        assert {env.queue_of(i) for i in range(first, last + 1)} == {"normal" if arm == "XGBBbE" else "gpuhopper"}
    neural = [r for r in plan if not r.xgb]
    assert [(r.fold, r.seed) for r in neural[:30]] == [(k, s) for k in range(10) for s in (0, 1, 2)]
    assert [(r.fold, r.horizon) for r in plan[120:]] == [(k, h) for k in range(10) for h in (1, 6, 24)]
    assert plan[18].run_id == "BbE_f6_s0" and plan[48].run_id == "LSTMBbE_f6_s0" and plan[140].run_id == "XGBBbE_f6_h24"
    assert plan[18].compared_with == "BbR_f6_s0" and plan[48].compared_with == "LSTMBbR_f6_s0"
    assert plan[78].compared_with == "Bb_f6_s0" and plan[140].compared_with == "XGBBb_f6_h24"
    main = b1h.PLAN[:270]
    assert all(r.base in main and r.base.context == 12 for r in plan)
    assert [r.env for r in plan].count(True) == 90


def test_rerun_configs_are_the_main_studys_and_env_configs_add_only_channels():
    for index, run in enumerate(env.PLAN):
        if run.xgb:
            continue
        names = sorted(b1h.fold_probabilities(run.fold, PAYLOAD))
        config = env.build_config(run, GRID, env.REGISTERED_STEPS, names, PAYLOAD)
        main = b1h.build_config(b1h.PLAN[b1h.PLAN.index(run.base)], GRID, b1h.REGISTERED_STEPS, names, PAYLOAD)
        assert config == (replace(main, env_channels=era5.N_ENV_CHANNELS) if run.env else main), index
        assert config.history == 0 and not config.use_fm_package and config.horizons == (1, 6, 24)


# ----------------------------------------------------------------------------- guards
def test_queue_guard(monkeypatch):
    neural, xgb = env.PLAN[0], env.PLAN[120]
    for cuda, queue, ok_neural, ok_xgb in [
        (True, "gpuhopper", True, False), (True, "gpuhopper-exec", True, False), (True, "gpuvolta", False, False),
        (False, "gpuhopper", False, False), (False, "normal", False, True), (False, "normal-exec", False, True),
        (False, "normalsr", False, False), (False, "", False, False),
    ]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda cuda=cuda: cuda)
        monkeypatch.setenv("PBS_QUEUE", queue)
        for run, ok in ((neural, ok_neural), (xgb, ok_xgb)):
            if ok:
                env.queue_guard(run)
            else:
                with pytest.raises(SystemExit, match="FAIL"):
                    env.queue_guard(run)


def test_freeze_and_budget_guards(monkeypatch):
    monkeypatch.setattr(env, "queue_guard", lambda run: None)
    called = []
    monkeypatch.setattr(env, "freeze_guard", lambda: called.append(1) or {"freeze_commit": "abc"})
    assert env.mode_guards(env.PLAN[0], calibrate=True, steps=None) is None and not called
    assert env.mode_guards(env.PLAN[0], calibrate=False, steps=20_000) == {"freeze_commit": "abc"} and called
    assert env.mode_guards(env.PLAN[120], calibrate=False, steps=None) == {"freeze_commit": "abc"}
    with pytest.raises(SystemExit, match="--steps"):
        env.mode_guards(env.PLAN[0], calibrate=False, steps=1_000)


def test_the_freeze_must_exist_and_the_tier2_freeze_path_is_restored(monkeypatch):
    monkeypatch.setattr(env, "FREEZE", "no_such_freeze.md")
    before = b1h.S2.FREEZE
    with pytest.raises(SystemExit, match="does not exist"):
        env.freeze_guard()
    assert b1h.S2.FREEZE == before


def test_the_driver_never_reads_a_test_partition():
    with pytest.raises(SystemExit, match="never 'test'"):
        b1h.split_ids("anas_platyrhynchos", "test", PAYLOAD)


def test_tf32_is_switched_off_and_required(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    with pytest.raises(SystemExit, match="TF32"):
        env.require_fp32()
    assert env.disable_tf32() == {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert env.require_fp32() == env.tf32_flags()


def test_the_1h_plan_and_evaluator_parts_are_unchanged():
    env.check_main_study()
    with pytest.raises(SystemExit, match="differ"):
        env.check_main_study(plan=b1h.PLAN[:-1])
    with pytest.raises(SystemExit, match="differ"):
        env.check_main_study(plan=list(reversed(b1h.PLAN)))
    with pytest.raises(SystemExit, match="differ"):
        env.check_main_study(parts={"main": (0, 269), "ablation": (270, 628)})


# ----------------------------------------------------------------------------- synthetic birds
def _bird(root, stem, seed, n_bins=70, drop=(30,), nan=(20,)):
    """Train (a_), validation (b_) and test (c_) individuals, one segment each; ERA5 for every row but `drop`, with tp
    missing at `nan`, both on the training individual."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, who in enumerate(("a", "b", "c")):
        for b in range(n_bins):
            rows.append({"segment_id": f"{who}_{stem}_s0", "individual_id": f"{who}_{stem}", "bin_index": b,
                         "timestamp": pd.Timestamp("2020-03-01", tz="UTC") + pd.Timedelta(seconds=HOUR * b + int(rng.integers(0, 3000))),
                         "lon": 10.0 + k + 0.004 * b + rng.normal(0, 3e-4), "lat": 50.0 + 0.002 * b + rng.normal(0, 3e-4)})
    segments = pd.DataFrame(rows)
    (root / "segments").mkdir(exist_ok=True)
    (root / "env").mkdir(exist_ok=True)
    segments.to_parquet(root / "segments" / f"{stem}_segments.parquet")
    values = rng.normal(50.0, 10.0, size=(len(segments), len(era5.RAW))).astype(np.float32)
    frame = segments[["segment_id", "bin_index"]].assign(**{v: values[:, i] for i, v in enumerate(era5.RAW)})
    train = frame["segment_id"] == f"a_{stem}_s0"
    frame.loc[train & frame["bin_index"].isin(nan), "tp"] = np.nan
    frame = frame[~(train & frame["bin_index"].isin(drop))]
    frame.to_parquet(root / "env" / f"{stem}_env.parquet", index=False)
    return frame


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    frames = {stem: _bird(tmp_path, stem, seed) for stem, seed in (("x", 1), ("y", 2))}
    monkeypatch.setattr(b1h, "SEGMENTS", tmp_path / "segments")
    monkeypatch.setattr(env, "ENV_DIR", tmp_path / "env")
    for name in ("RESULTS", "CHECKPOINTS", "CALIBRATION", "BUILD"):
        monkeypatch.setattr(env, name, tmp_path / name.lower())
    env.RESULTS.mkdir()
    env.CHECKPOINTS.mkdir()
    monkeypatch.setattr(b1h, "fold_probabilities", lambda fold, payload: dict(WEIGHTS))
    monkeypatch.setattr(b1h, "assert_fold_sources", lambda fold, stems: None)
    monkeypatch.setattr(b1h, "check_window_counts", lambda pool: None)

    def split_ids(stem, partition, payload):
        return {"train": [f"a_{stem}"], "val": [f"b_{stem}"]}[partition]  # KeyError on anything else

    monkeypatch.setattr(b1h, "split_ids", split_ids)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # as disable_tf32, restored after the test
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return tmp_path, frames


def test_env_stats_come_from_the_training_pool_only(synthetic):
    root, frames = synthetic
    run = env.PLAN[0]
    pool, node, stats = env.training_source(run, PAYLOAD)
    assert node == "fold 0" and all(pool.species[n].env_stats is stats for n in pool.names)
    unstandardised = SourcePool({n: replace(w, env_stats=None) for n, w in pool.species.items()}, WEIGHTS)
    expected = env_stats(unstandardised)
    assert all(np.array_equal(a, b) for a, b in zip(stats, expected))
    # perturb every validation and test row of both birds: the statistics and the training windows do not move
    for stem, frame in frames.items():
        frame = frame.copy()
        held = ~frame["segment_id"].str.startswith("a_")
        frame.loc[held, list(era5.RAW)] *= 7.0
        frame.to_parquet(root / "env" / f"{stem}_env.parquet", index=False)
    again, _, stats_again = env.training_source(run, PAYLOAD)
    assert all(np.array_equal(a, b) for a, b in zip(stats, stats_again))
    for name in pool.names:
        ids = np.arange(pool.species[name].n_windows)
        assert np.array_equal(pool.species[name].gather(ids)[0], again.species[name].gather(ids)[0])
    # a validation split standardises through the same gather, with the training pool's statistics
    val = env.env_windows("x", ["b_x"], stats)
    features = val.gather(np.arange(val.n_windows))[0]
    raw = val.env_tokens(np.arange(val.n_windows))
    z = np.clip((raw[..., :-1] - stats[0]) / stats[1], -era5.CLIP_SD, era5.CLIP_SD)
    np.testing.assert_allclose(features[..., 12:26], z, rtol=1e-12)


def test_a_missing_env_file_is_refused(synthetic):
    root, _ = synthetic
    (root / "env" / "y_env.parquet").unlink()
    with pytest.raises(SystemExit, match="y_env.parquet"):
        env.training_source(env.PLAN[0], PAYLOAD)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("index", [0, 30, 60, 90])
def test_neural_records_and_reported_checkpoints(synthetic, monkeypatch, index):
    import experiments.reporting.tier3_sealed_eval as E3

    root, _ = synthetic
    run = env.PLAN[index]
    env.run_torch(run, index, 2, GRID, "test", PAYLOAD, {"freeze_commit": "abc"})
    record = json.loads((env.RESULTS / f"{run.run_id}.json").read_text())
    assert record["tier"] == "birds_1h_env" and record["exploratory"] and not record["opened_test_partition"]
    assert record["queue"] == "gpuhopper" and record["freeze"] == {"freeze_commit": "abc"}
    assert record["tf32"] == {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}
    assert record["main_study_run"] == run.base.run_id and record["compared_with"] == run.compared_with
    assert record["applied_updates"] == 2 and record["checkpoint_selection"]["rule"] == "final_checkpoint"
    config = TrainConfig(**record["config"])
    assert config == env.build_config(run, GRID, 2, ["x", "y"], PAYLOAD)
    model, state = E3.load_reported(env.RESULTS, run.run_id)
    if run.env:
        assert record["env_channels"] == 15 == state["env_channels"] and config.env_channels == 15
        assert record["env_channel_names"] == list(era5.ENV_CHANNELS)
        assert record["env_stats"] == state["env_stats"] and len(record["env_stats"]["mean"]) == 14
        assert record["env_clip_sd"] == state["env_clip_sd"] == era5.CLIP_SD == 10.0
        assert record["env_files"] == {s: _sha(root / "env" / f"{s}_env.parquet") for s in ("x", "y")}
        coverage = record["env_coverage"]["x"]
        assert coverage["windows"] > 0 and 0 < coverage["token_share"] < 1 and coverage["row_share"] == pytest.approx(1 - 2 / 70)
        assert model.input_proj.in_features == 27
    else:
        assert record["env_channels"] == 0 and record["env_stats"] is None and record["env_files"] is None
        assert record["env_coverage"] is None and config.env_channels == 0 and model.input_proj.in_features == 12
        assert record["env_clip_sd"] is None and state["env_clip_sd"] is None


def test_the_xgb_record_uses_the_pools_statistics_and_27_channels(synthetic, monkeypatch):
    monkeypatch.setattr(b1h, "XGB_POOLED_DRAWS", 300)
    run = env.PLAN[122]  # XGBBbE fold 0, 24 h
    env.run_xgb(run, 122, GRID, "test", PAYLOAD, {"freeze_commit": "abc"})
    record = json.loads((env.RESULTS / f"{run.run_id}.json").read_text())
    assert record["queue"] == "normal" and record["channels"] == 27 and record["training_windows"] == 300
    assert record["compared_with"] == "XGBBb_f0_h24" and not record["allow_missing"]
    assert record["env_clip_sd"] == era5.CLIP_SD
    pool, _, stats = env.training_source(run, PAYLOAD)
    assert record["env_stats"] == env.stats_payload(stats)
    assert all(np.array_equal(a, b) for a, b in zip(env.stats_from_payload(record["env_stats"]), stats))
    assert record["horizon_k"] == list(horizon_k(pool))


# ----------------------------------------------------------------------------- dry run
def test_env_matches_passes_on_the_main_pool_and_catches_a_perturbed_core(synthetic):
    run = env.PLAN[0]
    main, _ = b1h.training_source(run.base, PAYLOAD)
    pool, _, _ = env.training_source(run, PAYLOAD)
    assert env.env_matches(main, pool) == [] and env.env_matches(main, pool, chunk=5) == []
    x = pool.species["x"]
    lon = x.lon.copy()
    lon[15] += 1e-3
    bad = SourcePool({"x": replace(x, lon=lon), "y": pool.species["y"]}, WEIGHTS)
    assert any("first 12 channels" in f for f in env.env_matches(main, bad))


def test_tail_shares_and_coverage_count_what_gather_gives(synthetic):
    pool, _, (mean, sd) = env.training_source(env.PLAN[0], PAYLOAD)
    pool = SourcePool({n: replace(w, env_stats=(mean, sd / 8)) for n, w in pool.species.items()}, WEIGHTS)  # fatter tails
    tails = env.tail_shares(pool, chunk=7)
    assert tails == env.tail_shares(pool) and tails["tail_sd"] == tails["clip_sd"] == era5.CLIP_SD == 10
    species = [pool.species[n] for n in pool.names]
    after = np.concatenate([w.gather(np.arange(w.n_windows))[0][..., 12:] for w in species])
    before = np.concatenate([era5.standardise(w.env_tokens(np.arange(w.n_windows)), mean, sd / 8, clip=None)
                             for w in species])
    assert np.array_equal(np.clip(before[..., :-1], -10, 10), after[..., :-1])
    for stage, z in (("before", before), ("after", after)):
        values = np.abs(z[z[..., -1] > 0][:, :-1])
        assert tails["observed_tokens"] == len(values)
        for i, c in enumerate(era5.ENV_CHANNELS[:14]):
            assert tails["channels"][c][stage]["share_beyond"] == float((values[:, i] > 10).mean())
            assert tails["channels"][c][stage]["max_abs"] == float(values[:, i].max())
    assert any(t["before"]["share_beyond"] > 0 for t in tails["channels"].values())
    assert all(t["after"]["share_beyond"] == 0 and t["after"]["max_abs"] <= 10 for t in tails["channels"].values())
    x = pool.species["x"]
    rows = x.anchors[:, None] + np.arange(-10, 1)
    missing = np.isin(x.keys[rows] % (1 << 32), [20, 30])  # the training individual's bins 20 and 30
    assert env.env_coverage(x, chunk=4)["token_share"] == pytest.approx(1 - missing.mean(), rel=1e-12)
    assert env.bytes_on_device(pool, 27) == sum(pool.species[n].n_windows for n in pool.names) * 11 * 27 * 4


def test_calibration_writes_timing_and_memory_and_no_record(synthetic, monkeypatch):
    monkeypatch.setattr(env, "CALIBRATION_STEPS", 100)
    monkeypatch.setattr(b1h, "XGB_POOLED_DRAWS", 300)
    env.run_torch(env.PLAN[48], 48, env.REGISTERED_STEPS, GRID, "test", PAYLOAD, None, calibrate=True)
    env.run_xgb(env.PLAN[140], 140, GRID, "test", PAYLOAD, None, calibrate=True)
    neural = json.loads((env.CALIBRATION / "LSTMBbE_f6_s0_calibration.json").read_text())
    assert neural["updates"] == 100 and neural["env_channels"] == 15 and neural["pool_channels"] == 27
    assert neural["seconds_per_update_median"] > 0 and neural["peak_rss_gb"] > 0
    assert neural["tf32"] == {"cudnn.allow_tf32": False, "cuda.matmul.allow_tf32": False}
    xgb = json.loads((env.CALIBRATION / "XGBBbE_f6_h24_calibration.json").read_text())
    assert xgb["channels"] == 27 and xgb["fit_seconds"] > 0 and xgb["peak_rss_gb"] > 0
    assert not list(env.RESULTS.iterdir()) and not list(env.CHECKPOINTS.iterdir())
