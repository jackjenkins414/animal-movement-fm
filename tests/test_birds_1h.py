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


def test_the_dry_run_prints_pins_pass(monkeypatch, capsys):
    monkeypatch.setattr(b1h, "check_all_birds", lambda payload: None)
    b1h.dry_run({}, None)
    assert capsys.readouterr().out.startswith("pins: PASS")

import experiments.reporting.bird_1h_sealed_eval as E  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig  # noqa: E402

FREEZE_TEXT = (
    "# 1 h\n\n## 5. Readings\n\ntext\n\n## 6. Pre-run predictions\n\nFilled by me.\n\n"
    "## 7. As trained\n\n630 of 630 runs.\n\n## 8. Limitations\n\n- one\n"
)


class _Shown:
    def __init__(self, stdout, returncode=0):
        self.stdout, self.returncode, self.stderr = stdout, returncode, ""


def _refuse(*args, **kwargs):
    raise AssertionError("a test reached a test-partition reader")


@pytest.fixture(autouse=True)
def _no_test_partition(monkeypatch):
    """Every reader of test-bird fixes or ids raises, so no test in this file can open a test partition."""
    for owner, name in ((E, "test_windows"), (E, "scored_ids"), (E.RS, "load_corpus"), (E.RS, "test_frame")):
        monkeypatch.setattr(owner, name, _refuse)


def test_the_evaluator_refuses_before_the_freeze_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "FREEZE", tmp_path / "birds_1h_exploratory_freeze.md")
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "manifest.json")
    for call in (lambda: E.run_predictions("cpu"), E.diagnostic):
        with pytest.raises(SystemExit, match="does not exist"):
            call()


@pytest.mark.parametrize("old,new", [("Filled by me.", "Jack to write"), ("630 of 630 runs.", "")])
def test_the_seal_refuses_a_committed_freeze_with_6_or_7_unfilled(monkeypatch, tmp_path, old, new):
    monkeypatch.setattr(E.S2, "assert_freeze_committed", lambda: {"freeze_commit": "c", "freeze_blob": "b"})
    monkeypatch.setattr(E.E4, "git", lambda *args: _Shown(FREEZE_TEXT.replace(old, new)))
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(SystemExit, match="sec [67]"):
        E.run_predictions("cpu")


def test_the_seal_passes_on_6_and_7_and_restores_the_shared_globals(monkeypatch, tmp_path):
    seen = {}

    def committed():
        seen["freeze"] = E.S2.FREEZE
        return {"freeze_commit": "c", "freeze_blob": "b"}

    monkeypatch.setattr(E.S2, "assert_freeze_committed", committed)
    monkeypatch.setattr(E.E4, "git", lambda *args: _Shown(FREEZE_TEXT))
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "manifest.json")
    before = (E.RS.FREEZE, E.RS.MANIFEST, E.RS.FLOOR, E.E3.FLOOR, E.S2.FREEZE)
    assert E.assert_sealed()["sections_filled"] == [6, 7]
    assert seen["freeze"] == E.FREEZE and E.FREEZE.name == "birds_1h_exploratory_freeze.md"
    assert (E.RS.FREEZE, E.RS.MANIFEST, E.RS.FLOOR, E.E3.FLOOR, E.S2.FREEZE) == before
    with pytest.raises(RuntimeError), E.bound():
        assert (E.RS.FLOOR, E.E3.FLOOR) == (150, 150)
        raise RuntimeError
    assert (E.RS.FREEZE, E.RS.MANIFEST, E.RS.FLOOR, E.E3.FLOOR) == before[:4]


def test_predictions_check_the_seal_before_records_or_test_data(monkeypatch):
    order = []
    def seal():
        order.append("seal")
        raise SystemExit("sealed")

    monkeypatch.setattr(E, "assert_sealed", seal)
    monkeypatch.setattr(E, "record_map", lambda *a: order.append("records"))
    with pytest.raises(SystemExit, match="sealed"):
        E.run_predictions("cpu")
    assert order == ["seal"]


def test_each_bird_scores_its_own_fold_and_its_own_specialists_only():
    for bird in E.BIRDS:
        runs = E.own_runs(bird)
        assert Counter((r.arm, r.context > 12) for r in runs) == {
            ("Bb", False): 3, ("LSTMBb", False): 3, ("XGBBb", False): 3, ("LSTMS", False): 3, ("XGBS", False): 3,
            ("Bb", True): 12, ("LSTMBb", True): 12, ("XGBBb", True): 12}
        assert all(r.fold == E.FOLD_OF[bird] and r.species in ("", bird) for r in runs)
        assert all(bird not in b1h.fold_probabilities(r.fold, PAYLOAD) for r in runs if not r.species)


def test_completeness_names_missing_and_unclean_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(E.b1h, "RESULTS", tmp_path)
    run, xgb = E.b1h.PLAN[0], E.b1h.PLAN[150]
    (tmp_path / "Bb_f0_s0_reported.pt").write_text("x")
    records = {
        run.run_id: {"plan_index": 0, "context": 12, "horizon": None, "opened_test_partition": False,
                     "config": {"steps": 20_000}, "applied_updates": 19_999, "skipped_steps": 0, "withheld_fraction": 0,
                     "reported_checkpoint": "Bb_f0_s0_reported.pt"},
        xgb.run_id: {"plan_index": 150, "context": 12, "horizon": 1, "opened_test_partition": False,
                     "allow_missing": False},
    }
    gap = E.completeness(records, E.part_runs("main"))
    assert gap["n_planned"] == 270 and gap["n_records"] == 2 and gap["n_clean"] == 0 and not gap["complete"]
    assert any("applied 19999 of 20000" in p for p in gap["problems"])
    assert any(p.startswith(xgb.run_id) and "booster missing" in p for p in gap["problems"])
    assert sum("no record" in p for p in gap["problems"]) == 268
    assert E.completeness({}, E.part_runs("ablation"))["n_planned"] == 360


def _toy_windows(tmp_path, context):
    path = _hourly(tmp_path, f"toy_{context}.parquet", "a", np.concatenate(([0, 1, 5], np.arange(10, 160))), 1)
    return HorizonWindows.build(path, ["a"], context=context, core=12, horizons=(1, 6, 24), bin_seconds=HOUR,
                                clip=1000.0)


def test_persistence_is_identical_across_c_and_a_change_is_caught(tmp_path):
    core = _toy_windows(tmp_path, 12)
    reference = E.frame_table(core, None, 12, E.history_share(core, 12), persistence=True)
    for context in (24, 48, 96):
        windows = _toy_windows(tmp_path, context)
        table = E.frame_table(windows, None, context, E.history_share(windows, context), persistence=True)
        E.assert_same_persistence(reference, table, f"C = {context}")
        assert set(table["context"]) == {context} and table["history_share"].between(0, 1).all()
    shifted = reference.copy()
    shifted.loc[3, "error_m"] += 1e-9
    with pytest.raises(SystemExit, match="identical across C|differs"):
        E.assert_same_persistence(reference, shifted, "C = 24")


def test_history_share_counts_observed_extended_bins_within_the_segment(tmp_path):
    observed = set(np.concatenate(([0, 1, 5], np.arange(10, 160))).tolist())
    windows = _toy_windows(tmp_path, 24)
    bins = (windows.keys[windows.anchors] & 0xFFFFFFFF).astype(int)
    expected = [sum(b in observed for b in range(t - 23, t - 11)) / 12 for t in bins]
    assert np.allclose(E.history_share(windows, 24), expected)
    assert expected[0] == 3 / 12 and expected[-1] == 1.0
    assert np.isnan(E.history_share(windows, 12)).all()


@pytest.mark.parametrize("trunk,flag", [("transformer", False), ("lstm", True)])
def test_scoring_at_c_matches_the_drivers_predictions_and_xgb_takes_its_horizon(tmp_path, trunk, flag):
    from animal_movement_fm.models.xgboost import WindowXGBoost

    windows = _toy_windows(tmp_path, 24)
    k = np.array([0.7, 3.0, 5.0])
    config = TrainConfig(d_model=16, num_layers=1, lr=1e-3, steps=1, nhead=2, use_fm_package=False,
                         use_context_conditioning=False, trunk=trunk, horizons=(1, 6, 24), history=24,
                         observed_flag=flag)
    model = config.build().eval()
    features, mask, targets, target_mask, _ = windows.for_horizon(6).gather(np.arange(windows.for_horizon(6).n_windows))
    booster = WindowXGBoost(n_estimators=5, max_depth=2, allow_missing=True).fit(
        np.where(mask[..., None], features, np.nan), targets[:, 1])
    neural_run, xgb_run = E.b1h.Run("Bb", 6, "", 0, 0, 24), E.b1h.Run("XGBBb", 6, "", 0, 6, 24)
    # at the driver's chunk the two agree bitwise; across chunk sizes float32 batching moves neural outputs by < 1 mm
    out = E.score_context(windows, [(neural_run, model, k)], [], "cpu", chunk=8_192)
    assert np.array_equal(out[neural_run.run_id], b1h.predictions_m(model, windows, k, "cpu", observed=True))
    out = E.score_context(windows, [], [(xgb_run, booster, k)], "cpu", chunk=37)
    rows = windows.targets[:, 1] >= 0
    _, _, _, _, scale = windows.gather(np.arange(windows.n_windows))
    full_features, full_mask, *_ = windows.gather(np.flatnonzero(rows))
    expected = booster.predict(np.where(full_mask[..., None], full_features, np.nan)) * k[1] * scale[rows, None]
    assert np.allclose(out[xgb_run.run_id][rows, 1], expected)
    assert not out[xgb_run.run_id][~rows].any() and not out[xgb_run.run_id][:, [0, 2]].any()


def test_eligibility_counts_individuals_at_the_floor_and_checks_horizon_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "FLOOR", 120)
    monkeypatch.setattr(E, "MIN_SCORED", 1)
    eligible = E.eligibility(_toy_windows(tmp_path, 12))
    windows = _toy_windows(tmp_path, 12)
    for j, h in enumerate((1, 6, 24)):
        n = int((windows.targets[:, j] >= 0).sum())
        assert eligible[str(h)] == {"individuals": 1, "at_floor": int(n >= 120), "windows": n, "scored": n >= 120}
    rows = [{"species": "toy", "horizon": h, "partition": "test_scored", "ge1": e["individuals"],
             "ge_equal_time": e["at_floor"], "windows": e["windows"]} for h, e in eligible.items()]
    pd.DataFrame(rows).to_csv(tmp_path / "counts.csv", index=False)
    monkeypatch.setattr(E.b1h, "HORIZON_COUNTS", tmp_path / "counts.csv")
    assert E.eligibility_mismatches({"toy": eligible}) == []
    drifted = {"toy": {**eligible, "6": {**eligible["6"], "windows": eligible["6"]["windows"] - 1}}}
    assert E.eligibility_mismatches(drifted) == [
        f"toy 6 h: (individuals, at floor, windows) {(1, eligible['6']['at_floor'], eligible['6']['windows'] - 1)}, "
        f"horizon_counts.csv {(1, eligible['6']['at_floor'], eligible['6']['windows'])}"]


# synthetic prediction tables for one real bird: P errors drawn at random, each run's errors P's times a known factor
BIRD = "anas_platyrhynchos"
FACTOR = {"Bb": 0.8, "LSTMBb": 0.7, "XGBBb": 0.9, "LSTMS": 0.6, "XGBS": 0.95}


def _factor(run):
    return FACTOR[run.arm] - 0.01 * (run.context // 24 if run.context > 12 else 0) + 0.05 * (run.seed - 1) * (not run.xgb)


def _tables(rng):
    parts = []
    for h in (1, 6, 24):
        ids = np.repeat(["i0", "i1", "i2", "i3"], [40, 40, 40, 10])
        parts.append(pd.DataFrame({
            "individual_id": ids,
            "timestamp": pd.Timestamp("2021-01-01", tz="UTC") + pd.to_timedelta(np.arange(len(ids)) * 3600, unit="s"),
            "target_lon": rng.uniform(0, 1, len(ids)), "target_lat": rng.uniform(0, 1, len(ids)),
            "pred_lon": 0.5, "pred_lat": 0.5, "error_m": rng.uniform(1, 500, len(ids)), "horizon": h, "context": 12,
            "history_share": np.nan}))
    p = pd.concat(parts, ignore_index=True)
    tables = {"P": p}
    for c in (24, 48, 72, 96):
        tables[f"P_C{c}"] = p.assign(context=c, history_share=rng.choice([0.5, 1.0], len(p)))
    for run in E.own_runs(BIRD):
        frame = p.assign(error_m=p["error_m"] * _factor(run), context=run.context)
        tables[run.run_id] = frame[frame["horizon"] == run.horizon] if run.xgb else frame
    return tables


def _records():
    return {run.run_id: {"arm": run.arm, "seed": run.seed, "context": run.context} for run in E.own_runs(BIRD)}


def test_a_main_cell_reads_skill_gaps_and_travel_share(monkeypatch):
    monkeypatch.setattr(E, "FLOOR", 30)
    monkeypatch.setattr(E.S2, "N_BOOTSTRAP", 50)
    tables = _tables(np.random.default_rng(0))
    eligible = {"1": {"individuals": 4, "at_floor": 3, "windows": 130, "scored": True}}
    cell = E.main_cell(BIRD, 1, tables, _records(), eligible)
    for arm, factor in FACTOR.items():
        assert E.E3.travel_skill(cell["arms"], arm) == pytest.approx(1 - factor)
    assert cell["readings"]["gaps"]["LSTM"]["skill_difference"] == pytest.approx(-0.1)
    assert cell["readings"]["gaps"]["XGB"]["skill_bootstrap"]["n_individuals"] == 3
    p = E.at(tables["P"], 1)
    assert cell["share_in_travel"] == pytest.approx((p["error_m"][p["individual_id"] != "i3"] >= 30).mean())
    unscored = E.main_cell(BIRD, 1, tables, _records(), {"1": {**eligible["1"], "scored": False}})
    assert "arms" not in unscored and unscored["at_floor"] == 3


def test_ablation_strata_hold_the_floor_on_the_whole_horizon(monkeypatch):
    monkeypatch.setattr(E, "FLOOR", 30)
    tables = _tables(np.random.default_rng(1))
    cells = E.ablation_cell(BIRD, 6, tables)
    for c in (24, 48, 72, 96):
        share = E.at(tables[f"P_C{c}"], 6)["history_share"]
        for stratum in E.STRATA:
            cell = cells[str(c)][stratum]
            assert cell["n_individuals"] == 3  # i3 is below the floor; the others stay in each stratum though < 30 there
            for arm in E.ABLATION_ARMS:
                assert cell[arm]["skill"] == pytest.approx(1 - FACTOR[arm] + 0.01 * (c // 24))
                assert cell[arm]["skill_C12"] == pytest.approx(1 - FACTOR[arm])
                assert cell[arm]["difference"] == pytest.approx(0.01 * (c // 24))
        assert cells[str(c)]["full"]["n_travel_windows"] + cells[str(c)]["gappy"]["n_travel_windows"] == \
            cells[str(c)]["all"]["n_travel_windows"]
        assert cells[str(c)]["gappy"]["n_travel_windows"] > 0 and cells[str(c)]["full"]["n_travel_windows"] > 0
        assert set(share.unique()) == {0.5, 1.0}
    broken = dict(tables, P_C48=tables["P_C48"].assign(error_m=tables["P_C48"]["error_m"] * 1.01))
    with pytest.raises(SystemExit, match="persistence"):
        E.ablation_cell(BIRD, 6, broken)


def test_the_report_runs_end_to_end_on_saved_synthetic_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "FLOOR", 30)
    monkeypatch.setattr(E.S2, "N_BOOTSTRAP", 20)
    monkeypatch.setattr(E, "BIRDS", (BIRD,))
    monkeypatch.setattr(E, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(E, "record_map", lambda *a: _records())
    for tag, frame in _tables(np.random.default_rng(2)).items():
        frame.to_parquet(tmp_path / f"{BIRD}__{tag}.parquet", index=False)
    counts = {"individuals": 4, "at_floor": 3, "windows": 130}
    eligible = {"1": {**counts, "scored": True}, "6": {**counts, "scored": True}, "24": {**counts, "scored": False}}
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"scoring_rules": {BIRD: {"dropped": []}}, "eligibility": {BIRD: eligible}, "attestation": {}}))
    report = E.build_report("all")
    assert report["persistence_identical_across_C"] == {"status": "PASS", "comparisons": 4}
    assert report["across_birds"]["common_set"] == []
    assert report["across_birds"]["horizons"]["1"]["skill"]["LSTMS"]["scored"]["median"] == pytest.approx(0.4)
    assert "ablation" not in report["species"][BIRD]["horizons"]["24"]
    assert report["ablation"]["Bb"]["6"]["96"]["full"]["scored"]["difference"]["median"] == pytest.approx(0.04)
    E.write_markdown(report, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "unscored, 3 / 4" in text and "Context ablation" in text
    (tmp_path / f"{BIRD}__Bb_f1_s0.parquet").write_bytes((tmp_path / f"{BIRD}__P.parquet").read_bytes())
    with pytest.raises(SystemExit, match="outside its own fold"):
        E.build_report("all")


def test_error_frames_round_trip_a_true_target_exactly_at_high_latitude(tmp_path):
    bins = np.arange(60)
    rows = [{"segment_id": "f_s0", "individual_id": "f", "bin_index": int(b),
             "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=int(b)),
             "lon": 10.0 + 0.4 * b, "lat": 55.0 + 0.25 * b} for b in bins]  # ~30 km/h north-east, 55-70 deg N
    pd.DataFrame(rows).to_parquet(tmp_path / "fast.parquet")
    windows = HorizonWindows.build(tmp_path / "fast.parquet", ["f"], context=12, core=12, horizons=(1, 6, 24),
                                   bin_seconds=HOUR, clip=1000.0)
    _, _, targets, target_mask, scale = windows.gather(np.arange(windows.n_windows))
    frames = b1h.horizon_frames(windows, targets * scale[:, None, None])
    for j, h in enumerate((1, 6, 24)):
        model, p = frames[h]
        assert len(model) == target_mask[:, j].sum() and model["error_m"].max() < 1e-6
        assert np.allclose(model["pred_lat"], model["target_lat"], rtol=0, atol=1e-12)
        assert p["error_m"].min() > 1_000
    # the anchor-latitude inverse the 6 h evaluators used misses the same 24 h targets by kilometres
    from animal_movement_fm.data.enu import enu_to_lonlat
    rows24 = np.flatnonzero(target_mask[:, 2])
    a, t = windows.anchors[rows24], windows.targets[rows24, 2]
    step = targets[rows24, 2] * scale[rows24, None]
    lon, lat = enu_to_lonlat(step[:, 0], step[:, 1], windows.lon[a], windows.lat[a])
    assert np.median(haversine_distance(lon, lat, windows.lon[t], windows.lat[t])) > 1_000


# ----------------------------------------------------------------------------- 3.10: remaining guards


class _Git:
    """Stands in for subprocess.run inside S2.assert_freeze_committed; no real git is run."""

    def __init__(self, tracked=True, clean=True):
        self.tracked, self.clean = tracked, clean

    def __call__(self, args, **kwargs):
        verb = args[3]
        code = {"ls-files": 0 if self.tracked else 1, "diff": 0 if self.clean else 1}.get(verb, 0)
        return _Shown("abc123\n", returncode=code)


@pytest.mark.parametrize("git,message", [(_Git(tracked=False), "not tracked"), (_Git(clean=False), "uncommitted")])
def test_the_evaluator_refuses_a_freeze_that_exists_but_is_not_committed(monkeypatch, tmp_path, git, message):
    freeze = tmp_path / "preregistration" / "birds_1h_exploratory_freeze.md"
    freeze.parent.mkdir()
    freeze.write_text(FREEZE_TEXT)
    monkeypatch.setattr(E.S2, "REPO", tmp_path)
    monkeypatch.setattr(E, "FREEZE", freeze)
    monkeypatch.setattr(E, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(E.S2.subprocess, "run", git)
    for call in (lambda: E.run_predictions("cpu"), E.diagnostic):
        with pytest.raises(SystemExit, match=message):
            call()


def _pbs(name):
    text = (b1h.REPO / "experiments" / "training" / name).read_text()
    return {line.split()[1] if line.split()[1] != "-l" else line.split()[2].split("=")[0]:
            line.split()[-1].split("=")[-1] for line in text.splitlines() if line.startswith("#PBS ")}


def test_each_pbs_script_targets_its_blocks_queue_with_the_calibrated_resources():
    for name, queue, cpus in (("bird_1h_train.pbs", "gpuvolta", "12"), ("bird_1h_train_cpu.pbs", "normal", "16")):
        directives = _pbs(name)
        assert directives["-q"] == queue and directives["ncpus"] == cpus
        assert directives["walltime"] == "01:00:00" and directives["mem"] == "32GB"
    scripts = {"gpuvolta": "bird_1h_train.pbs", "normal": "bird_1h_train_cpu.pbs"}
    assert all(_pbs(scripts[queue])["-q"] == queue for _, _, queue, _ in b1h.BLOCKS)
    assert "ngpus" in _pbs("bird_1h_train.pbs") and "ngpus" not in _pbs("bird_1h_train_cpu.pbs")


def test_run_records_are_exploratory_and_never_open_a_test_partition():
    for index in (0, 60, 180, 620):
        run = b1h.PLAN[index]
        names = [run.species] if run.species else sorted(b1h.fold_probabilities(run.fold, PAYLOAD))
        record = b1h.base_record(run, index, GRID, "registered", "node", names, PAYLOAD, (1.0, 2.0, 3.0), {})
        assert record["tier"] == "birds_1h" and record["exploratory"] is True
        assert record["opened_test_partition"] is False and record["registered_by"] == b1h.FREEZE
        assert record["plan_index"] == index and record["context"] == run.context and record["queue"] == b1h.queue_of(index)


@pytest.mark.parametrize("argv,message", [
    (["--index", "0", "--steps", "1000"], "--steps must be 20,000"),
    (["--index", "630", "--steps", "20000"], "outside 0..629"),
    (["--steps", "20000"], "no --index"),
])
def test_the_driver_refuses_a_wrong_budget_or_index(monkeypatch, argv, message):
    monkeypatch.setattr(b1h, "queue_guard", lambda run: None)
    monkeypatch.setattr(b1h, "freeze_guard", lambda: None)
    monkeypatch.delenv("PBS_ARRAY_INDEX", raising=False)
    monkeypatch.setattr(b1h.sys, "argv", ["bird_1h_train.py", *argv])
    with pytest.raises(SystemExit, match=message):
        b1h.main()


def _join_report(attested=True):
    birds = ("ciconia_ciconia", "larus_fuscus", "corvus_corone", "diomedea_exulans", "anser_anser")
    skill = {b: i / 10 for i, b in enumerate(birds)}
    report = {"attestation": {"freeze_commit": "c"} if attested else {}, "species": {}}
    for i, bird in enumerate(birds):
        horizons = {}
        for h in (1, 6, 24):
            arms = {arm: {"skill": {"travel": {"p90_m": skill[bird] * (1 if arm != "XGBS" else -1)}}}
                    for arm in E.JOIN_ARMS}
            horizons[str(h)] = {"scored": not (h == 6 and bird == "anser_anser"), "arms": arms}
        report["species"][bird] = {"horizons": horizons}
    table = pd.DataFrame({"species": [E.SPW.TIER4_PROFILE_KEY.get(b, b) for b in birds],
                          "rbar_corr": [0.1, 0.3, 0.2, 0.5, 0.4]})
    return report, table


def test_the_join_correlates_skill_with_rbar_corr_per_horizon_over_scored_birds():
    from scipy.stats import spearmanr

    report, table = _join_report()
    join = E.rbar_join(report, table)
    rbar = [0.1, 0.3, 0.2, 0.5, 0.4]
    for h, n in (("1", 5), ("6", 4), ("24", 5)):
        block = join["horizons"][h]
        assert len(block["birds"]) == n and set(block["rho"]) == set(E.JOIN_ARMS)
        expected = spearmanr(rbar[:n], [i / 10 for i in range(n)]).statistic
        assert block["rho"]["LSTMS"] == {"n": n, "rho": pytest.approx(expected)}
        assert block["rho"]["XGBS"]["rho"] == pytest.approx(-expected)
    assert join["rbar_corr"]["ciconia_ciconia"] == 0.1  # stork's row is under its profile key


def test_the_join_refuses_an_unsealed_report_or_a_missing_rbar_row():
    report, table = _join_report(attested=False)
    with pytest.raises(SystemExit, match="attestation"):
        E.rbar_join(report, table)
    report, table = _join_report()
    with pytest.raises(SystemExit, match="no row"):
        E.rbar_join(report, table.iloc[1:])
