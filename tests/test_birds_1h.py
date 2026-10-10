# GenAI reference: tests for the 1 h rebuild's driver from the behaviours I specified, which was roughtly a 630 run plan and its 
# queue blocks, the pins against their files, fold 0 present, the test partition refused, the queue guard etc. I checked them. 
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

import experiments.training.bird_1h_train as b1h
from animal_movement_fm.data.multispecies import HorizonWindows, SourcePool
from animal_movement_fm.eval.metrics import haversine_distance
from animal_movement_fm.train.pool import horizon_k

PAYLOAD = json.loads(b1h.RT.REGISTERED_FOLDS.read_text())
GRID = 11


# ----------------------------------------------------------------------------- the plan


def test_the_plan_has_630_runs_in_its_registered_blocks():
    plan = b1h.PLAN
    assert len(plan) == 630 and len({r.run_id for r in plan}) == 630
    expected = {
        (0, 29): ("Bb", 12), (30, 59): ("LSTMBb", 12), (60, 149): ("LSTMS", 12), (150, 179): ("XGBBb", 12),
        (180, 269): ("XGBS", 12), (270, 389): ("Bb", None), (390, 509): ("LSTMBb", None), (510, 629): ("XGBBb", None),
    }
    for (first, last), (arm, context) in expected.items():
        block = plan[first:last + 1]
        assert {r.arm for r in block} == {arm}
        assert all(r.context == context for r in block) if context else all(r.context in (24, 48, 72, 96)
                                                                              for r in block)
    assert Counter(b1h.queue_of(i) for i in range(630)) == {"gpuvolta": 390, "normal": 240}
    assert all(b1h.queue_of(i) == ("normal" if r.xgb else "gpuvolta") for i, r in enumerate(plan))


def test_every_fold_bird_seed_horizon_and_context_appears_exactly_once():
    plan = b1h.PLAN
    for arm in ("Bb", "LSTMBb"):
        cells = Counter((r.fold, r.seed, r.context) for r in plan if r.arm == arm)
        assert set(cells) == {(k, s, c) for k in range(10) for s in (0, 1, 2) for c in (12, 24, 48, 72, 96)}
        assert set(cells.values()) == {1}
    assert Counter((r.species, r.seed) for r in plan if r.arm == "LSTMS") == {
        (b, s): 1 for b in b1h.BIRDS for s in (0, 1, 2)}
    assert Counter((r.species, r.horizon) for r in plan if r.arm == "XGBS") == {
        (b, h): 1 for b in b1h.BIRDS for h in (1, 6, 24)}
    assert Counter((r.fold, r.horizon, r.context) for r in plan if r.arm == "XGBBb") == {
        (k, h, c): 1 for k in range(10) for h in (1, 6, 24) for c in (12, 24, 48, 72, 96)}
    assert {r.seed for r in plan if r.xgb} == {0}


def test_ablation_indices_follow_c_then_fold_then_seed_or_horizon():
    assert b1h.PLAN[270].run_id == "Bb_f0_s0_C24" and b1h.PLAN[378].run_id == "Bb_f6_s0_C96"
    assert b1h.PLAN[498].run_id == "LSTMBb_f6_s0_C96" and b1h.PLAN[620].run_id == "XGBBb_f6_h24_C96"
    assert b1h.PLAN[18].run_id == "Bb_f6_s0" and b1h.PLAN[60].run_id == f"LSTMS_{b1h.BIRDS[0]}_s0"


def test_fold_0_is_trained_with_its_three_birds():
    assert b1h.FOLDS[0] == ("anser_albifrons", "larus_fuscus", "streptopelia_turtur")
    assert len(b1h.BIRDS) == 30 and b1h.BIRDS[:3] == b1h.FOLDS[0]
    assert sum(1 for r in b1h.PLAN if r.fold == 0) == 3 * 2 + 9 + 3 + 9 + 4 * 6 + 12


# ----------------------------------------------------------------------------- pins


def test_the_pins_match_their_files():
    assert b1h.check_boundaries() == b1h.BOUNDARY_M and len(b1h.BOUNDARY_M) == 90
    assert b1h.check_fold0(PAYLOAD) == b1h.FOLD0_PROBABILITIES
    b1h.check_horizon_counts()
    assert sum(b1h.FOLD0_PROBABILITIES.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(v >= 30.0 for v in b1h.BOUNDARY_M.values())


def test_a_drifted_pin_fails(monkeypatch):
    key = ("anas_platyrhynchos", 6)
    monkeypatch.setitem(b1h.BOUNDARY_M, key, b1h.BOUNDARY_M[key] + 1e-6)
    with pytest.raises(SystemExit, match="anas_platyrhynchos 6 h boundary"):
        b1h.check_boundaries()
    monkeypatch.undo()
    monkeypatch.setitem(b1h.TRAIN_WINDOWS, "corvus_corone", (1, 1, 1, 1))
    with pytest.raises(SystemExit, match="corvus_corone"):
        b1h.check_horizon_counts()
    monkeypatch.undo()
    name = sorted(b1h.FOLD0_PROBABILITIES)[0]
    monkeypatch.setitem(b1h.FOLD0_PROBABILITIES, name, b1h.FOLD0_PROBABILITIES[name] * (1 + 1e-9))
    with pytest.raises(SystemExit, match="fold 0"):
        b1h.check_fold0(PAYLOAD)


def test_fold_0_pool_keeps_its_birds_and_genus_mates_out():
    weights = b1h.fold_probabilities(0, PAYLOAD)
    b1h.assert_fold_sources(0, weights)
    with pytest.raises(SystemExit, match="held-out bird"):
        b1h.assert_fold_sources(0, list(weights) + ["larus_argentatus"])
    assert b1h.fold_probabilities(6, PAYLOAD) == PAYLOAD["probabilities"]["6"]


# ----------------------------------------------------------------------------- guards


def test_the_test_partition_is_refused():
    for partition in ("test", "test_scored"):
        with pytest.raises(SystemExit, match="never"):
            b1h.split_ids("corvus_corone", partition, PAYLOAD)


def test_the_queue_guard(monkeypatch):
    neural, xgb = b1h.PLAN[0], b1h.PLAN[150]
    monkeypatch.setattr(b1h.torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("PBS_QUEUE", raising=False)
    with pytest.raises(SystemExit, match="GPU"):
        b1h.queue_guard(neural)
    b1h.queue_guard(xgb)
    monkeypatch.setenv("PBS_QUEUE", "gpuvolta-exec")
    with pytest.raises(SystemExit, match="CPU run"):
        b1h.queue_guard(xgb)
    monkeypatch.setattr(b1h.torch.cuda, "is_available", lambda: True)
    b1h.queue_guard(neural)
    monkeypatch.setenv("PBS_QUEUE", "normal-exec")
    with pytest.raises(SystemExit, match="CPU run"):
        b1h.queue_guard(xgb)


def test_training_is_refused_without_the_freeze(monkeypatch, tmp_path):
    monkeypatch.setattr(b1h, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="freeze"):
        b1h.freeze_guard()
    (tmp_path / "preregistration").mkdir()
    (tmp_path / "preregistration" / b1h.FREEZE).write_text("x")
    b1h.freeze_guard()


@pytest.mark.parametrize("index,trunk,history,flag", [
    (18, "transformer", 0, False), (48, "lstm", 0, False), (60, "lstm", 0, False),
    (378, "transformer", 96, False), (498, "lstm", 96, True), (270, "transformer", 24, False),
])
def test_configs_take_the_bird_recipe_with_heads_and_history(index, trunk, history, flag):
    run = b1h.PLAN[index]
    config = b1h.build_config(run, GRID, b1h.REGISTERED_STEPS, ["a", "b"], PAYLOAD)
    base = b1h.BT.build_config(run, GRID, b1h.REGISTERED_STEPS, ["a", "b"], PAYLOAD)
    assert (config.trunk, config.history, config.observed_flag) == (trunk, history, flag)
    assert config.horizons == (1, 6, 24) and not config.use_fm_package and not config.use_reconstruction_head
    assert {k: v for k, v in vars(config).items() if k not in ("horizons", "history", "observed_flag")} == \
        {k: v for k, v in vars(base).items() if k not in ("horizons", "history", "observed_flag")}


# ----------------------------------------------------------------------------- data, on a synthetic pool

HOUR = 3600


def _hourly(tmp_path, name, individual, bins, seed, step_deg=0.002):
    rng = np.random.default_rng(seed)
    rows = [{"segment_id": f"{individual}_s0", "individual_id": individual, "bin_index": int(b),
             "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(seconds=HOUR * int(b)
                                                                               + int(rng.integers(0, 3000))),
             "lon": 10.0 + step_deg * b + rng.normal(0, 2e-4), "lat": 50.0 + rng.normal(0, 2e-4)} for b in bins]
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def _pool(tmp_path, context=12):
    paths = {"a": _hourly(tmp_path, "a.parquet", "a", np.concatenate(([0, 1, 5], np.arange(10, 160))), 1),
             "b": _hourly(tmp_path, "b.parquet", "b", np.setdiff1d(np.arange(130), np.arange(40, 44)), 2, 0.01)}
    species = {n: HorizonWindows.build(p, [n], context=context, core=12, horizons=(1, 6, 24), bin_seconds=HOUR,
                                       clip=1000.0) for n, p in paths.items()}
    return SourcePool(species, {"a": 0.7, "b": 0.3})


def test_pooled_xgb_draws_follow_source_pool_sample_on_the_horizon_views(tmp_path, monkeypatch):
    monkeypatch.setattr(b1h, "XGB_POOLED_DRAWS", 500)
    pool = _pool(tmp_path, context=24)
    k = horizon_k(pool)
    run = b1h.Run("XGBBb", 6, "", 0, 6, 24)
    features, targets, _ = b1h.xgb_horizon_set(pool, run, k)
    views = SourcePool({n: pool.species[n].for_horizon(6) for n in pool.names}, {"a": 0.7, "b": 0.3})
    f, mask, t, tmask, _ = views.sample(np.random.default_rng(0), 500)
    assert tmask[:, 1].all()
    assert np.array_equal(targets, (t[:, 1] / k[1]).astype(np.float32))
    assert np.array_equal(np.isnan(features).any(-1), ~mask)
    assert np.array_equal(np.nan_to_num(features), np.where(mask[..., None], f, 0).astype(np.float32))
    assert features.dtype == np.float32 and np.isnan(features).any()


def test_specialist_xgb_takes_every_window_with_a_target_at_h(tmp_path):
    pool = _pool(tmp_path)
    single = SourcePool({"a": pool.species["a"]}, {"a": 1.0})
    k = horizon_k(single)
    features, targets, _ = b1h.xgb_horizon_set(single, b1h.Run("XGBS", 0, "a", 0, 24), k)
    view = pool.species["a"].for_horizon(24)
    assert len(targets) == view.n_windows < pool.species["a"].n_windows
    assert not np.isnan(features).any()
    assert np.array_equal(targets, (view.gather(np.arange(view.n_windows))[2][:, 2] / k[2]).astype(np.float32))


def test_error_frames_measure_great_circle_metres_and_persistence(tmp_path):
    windows = _pool(tmp_path).species["b"]
    frames = b1h.horizon_frames(windows, np.zeros((windows.n_windows, 3, 2)))
    persistence = b1h.horizon_frames(windows, None)
    for j, h in enumerate((1, 6, 24)):
        model, p = frames[h]
        rows = np.flatnonzero(windows.targets[:, j] >= 0)
        a, t = windows.anchors[rows], windows.targets[rows, j]
        expected = haversine_distance(windows.lon[a], windows.lat[a], windows.lon[t], windows.lat[t])
        assert np.allclose(model["error_m"], expected, rtol=1e-9) and np.array_equal(p["error_m"], expected)
        assert np.array_equal(persistence[h][0]["error_m"], expected)
        assert set(model["individual_id"]) == {"b"}


def test_the_selection_score_is_one_for_persistence(tmp_path, monkeypatch):
    windows = _pool(tmp_path).species["a"]
    frames = b1h.horizon_frames(windows, None)
    for h in (1, 6, 24):
        monkeypatch.setitem(b1h.BOUNDARY_M, ("a", h), 30.0)
    monkeypatch.setattr(b1h, "FLOOR", 50)
    qualifying = b1h.qualifying_horizons("a", frames)
    assert qualifying == (1, 6, 24)
    assert b1h.selection_score("a", frames, qualifying)["ratio"] == pytest.approx(1.0)
    monkeypatch.setattr(b1h, "FLOOR", 10_000)
    assert b1h.qualifying_horizons("a", frames) == ()


def test_clip_shares_count_core_and_observed_extended_tokens(tmp_path, monkeypatch):
    monkeypatch.setattr(b1h, "CLIP", 1.0)  # about the median scaled step here, so some tokens reach it
    clipped, raw = _pool(tmp_path, context=24), _pool(tmp_path, context=24)
    for name in clipped.names:
        clipped.species[name].clip, raw.species[name].clip = 1.0, None
    shares = b1h.clip_shares(clipped)  # every window: CLIP_SAMPLE exceeds both species' counts
    expected = {"core": [0, 0], "extended": [0, 0]}
    for name in raw.names:
        w = raw.species[name]
        features, mask, *_ = w.gather(np.arange(w.n_windows))
        over = (np.abs(features[..., [0, 1]]) >= 1.0).any(-1)
        core = np.zeros_like(mask)
        core[:, -11:] = True
        for part, m in (("core", core), ("extended", mask & ~core)):
            expected[part][0] += int(over[m].sum())
            expected[part][1] += int(m.sum())
    for part, (hit, total) in expected.items():
        assert shares[part] == hit / total and 0 < shares[part] < 1
    assert shares["max_core"][0] in clipped.names
    assert b1h.clip_shares(_pool(tmp_path))["extended"] is None
