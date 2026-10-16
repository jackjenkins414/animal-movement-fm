# GenAI reference: In the same window of the transformer collapse file I asked for tests for it too, based off the original outline
# i wanted for it. GenAI wrote these tests which I read over. 
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import torch

import experiments.diagnostics.transformer_collapse as TC
from animal_movement_fm.data.multispecies import HorizonWindows
from animal_movement_fm.data.split import load_split
from animal_movement_fm.models.fm import MovementFM

b1h = TC.b1h
HOUR = 3_600
PAYLOAD = json.loads(b1h.RT.REGISTERED_FOLDS.read_text())


def _track(tmp_path, name, bins, *, step_deg=0.01, noise=0.0, seed=0, jitter=True):
    rng = np.random.default_rng(seed)
    rows = [{"segment_id": f"{name}_s0", "individual_id": name, "bin_index": int(b),
             "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
             + pd.Timedelta(seconds=HOUR * int(b) + (int(rng.integers(0, 3000)) if jitter else 0)),
             "lon": 10.0 + step_deg * b + rng.normal(0, noise), "lat": 50.0 + rng.normal(0, noise)} for b in bins]
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _windows(path, name, context=12):
    return HorizonWindows.build(path, [name], context=context, core=12, horizons=(1, 6, 24), bin_seconds=HOUR,
                                clip=1000.0)


def _toy(trunk, context, dropout=0.1, seed=0):
    torch.manual_seed(seed)
    flag = trunk == "lstm" and context > 12
    return MovementFM(input_dim=12 + flag, d_model=16, nhead=4, num_layers=2, dropout=dropout, trunk=trunk,
                      n_horizons=3, observed_flag=flag, use_fm_package=False).eval()


def _batch(n, context, seed=0):
    """Random features with the last 11 tokens observed and about half the extended tokens unobserved (zeroed)."""
    rng = np.random.default_rng(seed)
    mask = np.ones((n, context - 1), dtype=bool)
    mask[:, : context - 12] = rng.random((n, context - 12)) > 0.5
    features = np.where(mask[..., None], rng.normal(0, 1, (n, context - 1, 12)), 0.0).astype(np.float32)
    share = mask[:, : context - 12].mean(axis=1) if context > 12 else np.full(n, np.nan)
    return {"features": features, "token_mask": mask, "share": share}


# ------------------------------------------------------------------------------------------- reference predictors
def test_last_step_is_the_last_token_times_h_over_k_and_its_metres_drop_k():
    features = np.zeros((2, 11, 12), dtype=np.float32)
    features[0, -1, :2] = (2.0, -1.0)
    features[1, -1, :2] = (0.5, 0.0)
    features[:, -2, :2] = 99.0  # an earlier token is never read
    k, scale = np.array([1.0, 2.0, 4.0]), np.array([30.0, 100.0])
    pred = TC.last_step(features, k)
    assert pred.shape == (2, 3, 2)
    for i, step in enumerate(((2.0, -1.0), (0.5, 0.0))):
        for j, h in enumerate((1, 6, 24)):
            np.testing.assert_allclose(pred[i, j], np.array(step) * h / k[j])
            np.testing.assert_allclose(TC.to_metres(pred, k, scale)[i, j], np.array(step) * scale[i] * h)


def test_last_step_dead_reckons_a_straight_track_onto_its_targets(tmp_path):
    windows = _windows(_track(tmp_path, "line", np.arange(80), jitter=False), "line")
    batch = TC.gather(windows, 12)
    ones = np.ones(3)
    last_m = TC.to_metres(TC.last_step(batch["features"], ones), ones, batch["scale"])
    actual_m = batch["scaled"] * batch["scale"][:, None, None]
    has = batch["target_mask"]
    assert has.all(axis=0).any()
    np.testing.assert_allclose(last_m[has], actual_m[has], rtol=1e-6, atol=1e-3)
    # and the loss of the last-step predictor is then ~0 against a clearly positive zero predictor
    k = np.array([1.0, 6.0, 24.0])
    losses = TC.reference_losses(np.zeros((windows.n_windows, 3, 2)), batch["features"], batch["scaled"], has, k)
    assert losses["last_step"]["mean"] < 1e-8 < losses["zero"]["mean"]


def test_slope_and_r2_and_behaviour_on_hand_values():
    x = np.linspace(-3, 3, 50)
    assert TC.slope_r2(x, 0.5 * x) == pytest.approx((0.5, 1.0))
    assert TC.slope_r2(x, np.zeros_like(x)) == pytest.approx((0.0, 0.0))
    actual = np.tile(np.array([[[100.0, 0.0]] * 3]), (4, 1, 1))
    scale = np.full(4, 30.0)
    mask = np.ones((4, 3), dtype=bool)
    boundaries = {1: 30.0, 6: 30.0, 24: 30.0}
    still = TC.behaviour(np.zeros_like(actual), actual, actual, scale, mask, boundaries)["6"]
    assert still["ratio"] == 0 and still["share_pred_under_1m"] == 1 and still["median_pred_m_all"] == 0
    same = TC.behaviour(actual, actual, actual, scale, mask, boundaries)["6"]
    assert same["ratio"] == pytest.approx(1) and same["cos_actual"] == pytest.approx(1)


def test_training_sample_draws_species_by_probability_then_uniformly_and_is_seeded(tmp_path):
    windows = {n: _windows(_track(tmp_path, n, np.arange(200), noise=1e-4, seed=i), n) for i, n in enumerate("ab")}
    picks = TC.training_sample(windows, {"a": 0.7, "b": 0.3}, 20_000, 5)
    assert sum(len(v) for v in picks.values()) == 20_000
    assert len(picks["a"]) / 20_000 == pytest.approx(0.7, abs=0.02)
    assert all(v.max() < windows[n].n_windows for n, v in picks.items())
    again = TC.training_sample(windows, {"a": 0.7, "b": 0.3}, 20_000, 5)
    assert all(np.array_equal(picks[n], again[n]) for n in picks)
    assert np.array_equal(TC.validation_sample(10_000, 300, "a"), TC.validation_sample(10_000, 300, "a"))
    assert TC.validation_sample(100, 300, "a").tolist() == list(range(100))


def test_the_loss_curve_reads_the_logged_points_and_the_plateau():
    history = np.concatenate([np.linspace(2.0, 1.0, 20), np.full(20, 1.0)])
    curve = TC.loss_curve({"loss_history": history.tolist(), "head_loss_history": [[1.0, 1.0, 1.0]] * 40})
    assert curve["points"] == 40 and curve["smoothed"]["20000"] == pytest.approx(1.0)
    assert curve["smoothed"]["2000"] > 1.5
    assert 8_000 <= curve["within_5pct_of_final_from"] <= 11_000
    assert "note" in TC.loss_curve({"loss_history": [1.0] * 10})


# ---------------------------------------------------------------------------------------------------- guards
@pytest.mark.parametrize("partition", ["test", "test_scored", "all"])
def test_the_test_partition_is_refused_before_anything_is_read(partition, monkeypatch):
    monkeypatch.setattr(b1h, "split_ids", lambda *a, **k: pytest.fail("a split was read"))
    with pytest.raises(SystemExit, match="never"):
        TC.split_ids("corvus_corone", partition, PAYLOAD)


def test_a_test_individual_among_the_ids_is_refused():
    split = load_split(b1h.T.SPLITS / "corvus_corone_split_seed42.json")
    TC.assert_not_test("corvus_corone", list(split.train) + list(split.val))
    with pytest.raises(SystemExit, match="test individuals"):
        TC.assert_not_test("corvus_corone", list(split.val) + [split.test[0]])
    with pytest.raises(SystemExit, match="test individuals"):
        TC.check_ids("toy", ["a", "b"], ["b"])


def test_only_the_diagnosis_folds_and_planned_runs_are_accepted():
    with pytest.raises(SystemExit, match="covers folds"):
        TC.run_fold(1)
    assert TC.diagnosis_run("Bb", 6, 96).run_id == "Bb_f6_s0_C96"
    with pytest.raises(SystemExit, match="not in the 1 h plan"):
        TC.diagnosis_run("Bb", 6, 36)


# ---------------------------------------------------------------------------------------------------- bug checks
@pytest.mark.parametrize("context", [24, 96])
def test_masked_tokens_do_not_move_a_masked_transformer_but_do_move_the_flagged_lstm(context):
    batch = _batch(64, context)
    rng = np.random.default_rng(1)
    transformer = TC.masked_noise_check(_toy("transformer", context), batch, context, rng)
    assert transformer["n_with_unobserved"] > 0 and transformer["max_abs_diff"] < 1e-5
    lstm = TC.masked_noise_check(_toy("lstm", context), batch, context, rng)
    assert lstm["max_abs_diff"] > 1e-3  # the check has power: an unmasked reader moves


def test_a_masking_bug_is_caught():
    class Leaky(torch.nn.Module):  # reads every token, masked or not
        def forward(self, x, observed=None):
            return x.mean(dim=1)[:, :6].reshape(-1, 3, 2)

    result = TC.masked_noise_check(Leaky(), _batch(32, 48), 48, np.random.default_rng(0))
    assert result["max_abs_diff"] > 1e-2


@pytest.mark.parametrize("context", [12, 48])
def test_the_last_token_moves_the_output(context):
    sensitivity = TC.token_sensitivity(_toy("transformer", context), _batch(64, context), context)
    assert sensitivity["last_plus_1"]["h6"] > 0 and sensitivity["last_swapped"]["h6"] > 0


@pytest.mark.parametrize("context", [12, 96])
def test_eval_and_train_mode_without_dropout_agree_and_the_model_stays_in_eval(context):
    model = _toy("transformer", context, dropout=0.3)
    result = TC.fast_path_check(model, _batch(64, context), context)
    assert result["max_abs_diff"] < 1e-5
    assert not model.training and all(m.p == 0.3 for m in model.modules() if isinstance(m, torch.nn.Dropout))


def test_core_only_ignores_whatever_the_extended_tokens_held():
    model, batch = _toy("transformer", 48), _batch(32, 48)
    other = dict(batch, features=batch["features"] + np.float32(5.0))
    other["features"][:, -11:] = batch["features"][:, -11:]
    np.testing.assert_allclose(TC.core_only(model, batch, 48), TC.core_only(model, other, 48), atol=1e-5)


# ---------------------------------------------------------------------------------------------------- end to end
def test_a_fold_and_the_report_run_end_to_end_on_toy_data(tmp_path, monkeypatch):
    bins = np.concatenate(([0, 3, 9], np.arange(20, 260)))
    paths = {n: _track(tmp_path, n, bins, step_deg=0.002 * (i + 1), noise=2e-4, seed=i) for i, n in enumerate("abc")}

    def build(stem, partition, payload, context=12):
        assert partition in ("train", "val")
        return _windows(paths[stem], stem, context)

    models = {(arm, c): _toy("lstm" if arm == "LSTMBb" else "transformer", c, seed=c) for arm in TC.ARMS
              for c in TC.CONTEXTS}
    record = {"loss_history": np.linspace(1.0, 0.5, 40).tolist(), "head_loss_history": [[1.0, 0.5, 0.2]] * 40}
    monkeypatch.setattr(b1h, "load_registered", lambda: {})
    monkeypatch.setattr(b1h, "fold_probabilities", lambda fold, payload: {"a": 0.6, "b": 0.4})
    monkeypatch.setattr(b1h, "assert_fold_sources", lambda fold, stems: None)
    monkeypatch.setattr(b1h, "check_window_counts", lambda pool: None)
    monkeypatch.setattr(TC, "FOLDS", {0: ("c",)})
    monkeypatch.setattr(TC, "BOUNDARY_M", defaultdict(lambda: 30.0))
    monkeypatch.setattr(TC, "build", build)
    monkeypatch.setattr(TC, "load_run", lambda run: (models[(run.arm, run.context)], np.array([0.7, 6.0, 5.0]), record))
    monkeypatch.setattr(TC, "OUT", tmp_path / "out")
    monkeypatch.setattr(TC, "REPORT_JSON", tmp_path / "out" / "transformer_diagnosis.json")
    monkeypatch.setattr(TC, "REPORT_MD", tmp_path / "out" / "transformer_diagnosis.md")
    monkeypatch.setattr(TC, "DIAGNOSIS_FOLDS", (0,))

    result = TC.run_fold(0, n_train=300, max_val=150, n_check=60)
    assert result["opened_test_partition"] is False and result["group_of"] == {"a": "pool", "b": "pool", "c": "held_out"}
    assert set(result["checks"]) == {f"{a}_C{c}" for a in TC.ARMS for c in TC.CONTEXTS}
    assert result["checks"]["Bb_C96"]["masked_noise"]["max_abs_diff"] < 1e-5
    assert result["checks"]["Bb_C96"]["fast_path"]["max_abs_diff"] < 1e-5
    assert set(result["core_only"]) == {f"{a}_C{c}" for a in TC.ARMS for c in TC.CONTEXTS if c > 12}
    assert set(result["skill"]["c"]["6"]) >= {"P_p90_m", "last_step", "Bb_C12", "LSTMBb_C96"}
    assert result["behaviour"]["Bb_C48"]["a"].keys() == {"all", "full", "gappy"}

    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "fold0.json").write_text(json.dumps(result, default=str))
    TC.report()
    markdown = (tmp_path / "out" / "transformer_diagnosis.md").read_text()
    for heading in ("## 1. Loss", "## 1b.", "## 2. Validation behaviour, pool", "## 2b.", "## 3. Bug checks",
                    "## 3b.", "## 3c.", "## 4. Validation travel p90"):
        assert heading in markdown
    assert "| Bb_C96 |" in markdown and "| held-out | 24 |" in markdown


def test_a_dry_run_loads_no_model(tmp_path, monkeypatch):
    paths = {n: _track(tmp_path, n, np.arange(200), noise=2e-4, seed=i) for i, n in enumerate("abc")}
    monkeypatch.setattr(b1h, "load_registered", lambda: {})
    monkeypatch.setattr(b1h, "fold_probabilities", lambda fold, payload: {"a": 0.6, "b": 0.4})
    monkeypatch.setattr(b1h, "assert_fold_sources", lambda fold, stems: None)
    monkeypatch.setattr(b1h, "check_window_counts", lambda pool: None)
    monkeypatch.setattr(TC, "FOLDS", {3: ("c",)})
    monkeypatch.setattr(TC, "BOUNDARY_M", defaultdict(lambda: 30.0))
    monkeypatch.setattr(TC, "build", lambda stem, partition, payload, context=12: _windows(paths[stem], stem, context))
    monkeypatch.setattr(TC, "load_run", lambda run: pytest.fail("a dry run loaded a model"))
    result = TC.run_fold(3, n_train=200, max_val=100, n_check=50, dry_run=True)
    assert result["dry_run"] and set(result["magnitudes"]) == {"24", "48", "96"} and not result["checks"]
    assert set(result["skill"]["c"]["1"]) >= {"P_p90_m", "last_step"}
