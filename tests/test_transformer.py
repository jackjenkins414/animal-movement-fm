# GenAI reference: Test suite also made with GenAI, mirroring the structure of our previous ones like (test_lstm.py, test_arima.py etc). 
# We asked it to cover protocol conformance, the unfitted predict guard, fit/predict behaviour, the checkpoint round trip, and a small 
# learning sanity check. Note, synthetic fixtures only; all models pinned to CPU and tiny configs (with nhead dividing d_model) so the 
# suite stays fast and deterministic.

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from animal_movement_fm.eval.harness import PredictionContext, iter_windows
from animal_movement_fm.models.transformer import Transformer
from animal_movement_fm.state.dataset import build_state_windows


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

def _make_df(n_individuals=3, n_steps=80, drift=(0.0, 0.0), noise=0.01, seed=0):
    """Random-walk DataFrame in the harness schema (individual_id, timestamp, lon, lat)."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_individuals):
        steps = rng.standard_normal((n_steps, 2)) * noise + np.asarray(drift)
        coords = steps.cumsum(axis=0)
        frames.append(pd.DataFrame({
            "individual_id": i,
            "timestamp": pd.date_range("2024-01-01", periods=n_steps, freq="h"),
            "lon": coords[:, 0],
            "lat": coords[:, 1],
        }))
    return pd.concat(frames, ignore_index=True)


def _mean_error(df, context_length, predict_fn):
    """Mean Euclidean error of a predict_fn(context)->(2,) over all windows."""
    errs = []
    for _, traj in df.groupby("individual_id", sort=False):
        for history, target, _ts, _t in iter_windows(traj, context_length):
            pred = predict_fn(PredictionContext(history=history))
            errs.append(np.linalg.norm(pred - target))
    return float(np.mean(errs))


def _context_with_time(history, freq="6h", start="2024-06-01"):
    """A PredictionContext carrying per-fix timestamps row-aligned with history (what Design B needs).

    Mirrors what the harness threads through at eval time, so time-feature predicts can be exercised
    directly without going through evaluate().
    """
    ts = pd.date_range(start, periods=len(history), freq=freq).to_numpy()
    return PredictionContext(
        history=history, timestamp=pd.Timestamp(ts[-1]), history_timestamps=ts
    )


@pytest.fixture
def small_df():
    return _make_df()


@pytest.fixture(scope="module")
def fitted_model():
    # Tiny config + CPU so fitting is fast and deterministic. nhead divides d_model (16 / 2).
    df = _make_df()
    return Transformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                       context_length=12, epochs=3, batch_size=32, device="cpu").fit(df)


@pytest.fixture(scope="module")
def fitted_time_model():
    # Same tiny config but with Design B time-of-day on. _make_df carries a timestamp column; nhead divides d_model.
    df = _make_df()
    return Transformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, context_length=12,
                       epochs=3, batch_size=32, use_time_features=True, device="cpu").fit(df)


def _make_seg_df(n_segments=3, n_steps=40, noise=0.02, seed=0):
    """Random-walk frame WITH a segment_id column (what the state head needs to label states).

    One segment per individual, concatenated in sorted segment-id order ('s0','s1',...) at 6 h cadence, so the frame is
    pre-sorted by (segment_id, timestamp) and the model's per-segment window order matches build_state_windows'.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_segments):
        coords = (rng.standard_normal((n_steps, 2)) * noise).cumsum(axis=0)
        frames.append(pd.DataFrame({
            "segment_id": f"s{i}",
            "individual_id": i,
            "timestamp": pd.date_range("2024-03-01", periods=n_steps, freq="6h"),
            "lon": coords[:, 0],
            "lat": coords[:, 1],
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def fitted_state_model():
    # Tiny state-head transformer on a segment frame; exercises the joint MSE + state-CE training path.
    df = _make_seg_df()
    return Transformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, context_length=12,
                       epochs=3, batch_size=32, use_state_head=True, device="cpu").fit(df, group_col="segment_id")


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------

class TestProtocol:
    def test_has_predict_method(self):
        assert callable(getattr(Transformer(), "predict", None))

    def test_has_name_attribute(self):
        assert isinstance(Transformer().name, str)

    def test_name_reflects_architecture(self):
        assert Transformer(d_model=32, num_layers=2).name == "transformer_d32_l2"

    def test_name_reflects_time_feature(self):
        assert Transformer(d_model=32, num_layers=2, use_time_features=True).name == "transformer_d32_l2_time"


# ----------------------------------------------------------------------
# Unfitted guards
# ----------------------------------------------------------------------

class TestUnfitted:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            Transformer().predict(PredictionContext(history=np.zeros((12, 2), dtype=np.float32)))

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            Transformer().save(tmp_path / "x.pt")


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, small_df):
        model = Transformer(d_model=16, nhead=2, num_layers=1, context_length=12,
                            epochs=1, device="cpu")
        assert model.fit(small_df) is model

    def test_fit_sets_state(self, fitted_model):
        assert fitted_model.net is not None
        assert fitted_model._mean is not None
        assert fitted_model._std is not None

    def test_fit_no_windows_raises(self):
        # Every individual shorter than context_length + 1 -> no windows.
        df = _make_df(n_individuals=2, n_steps=5)
        with pytest.raises(ValueError):
            Transformer(d_model=16, nhead=2, context_length=24, epochs=1, device="cpu").fit(df)


# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_model):
        pred = fitted_model.predict(PredictionContext(history=np.zeros((12, 2), dtype=np.float32)))
        assert pred.shape == (2,)

    def test_predict_bad_shape_raises(self, fitted_model):
        with pytest.raises(ValueError):
            fitted_model.predict(PredictionContext(history=np.zeros((12, 3))))

    def test_predict_short_history_falls_back(self, fitted_model):
        pred = fitted_model.predict(PredictionContext(history=np.array([[1.0, 2.0]], dtype=np.float32)))
        np.testing.assert_allclose(pred, [1.0, 2.0])

    def test_predict_finite_and_near_last(self, fitted_model):
        rng = np.random.default_rng(1)
        history = (rng.standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        pred = fitted_model.predict(PredictionContext(history=history))
        assert np.all(np.isfinite(pred))
        # Catch catastrophic failure (NaN/inf/huge jump), not exact value.
        assert np.linalg.norm(pred - history[-1]) < 1.0


# ----------------------------------------------------------------------
# Checkpoint round-trip
# ----------------------------------------------------------------------

class TestCheckpoint:
    def test_roundtrip_predictions_match(self, fitted_model, tmp_path):
        history = (np.random.default_rng(2).standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        before = fitted_model.predict(PredictionContext(history=history))

        path = tmp_path / "ckpt.pt"
        fitted_model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        after = reloaded.predict(PredictionContext(history=history))

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_reloaded_restores_config(self, fitted_model, tmp_path):
        path = tmp_path / "ckpt.pt"
        fitted_model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.net is not None
        assert reloaded.d_model == fitted_model.d_model
        assert reloaded.nhead == fitted_model.nhead
        assert reloaded.context_length == fitted_model.context_length


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------

class TestWindows:
    def test_window_shapes(self, small_df):
        model = Transformer(d_model=16, nhead=2, context_length=12, device="cpu")
        X, Y, T = model._make_windows(small_df)
        assert X.ndim == 3
        assert X.shape[1:] == (11, 2)   # context_length - 1 displacements, 2 channels
        assert Y.shape[1:] == (2,)
        assert X.shape[0] == Y.shape[0]
        assert T is None                # time features off by default


# ----------------------------------------------------------------------
# Learning sanity
# ----------------------------------------------------------------------

class TestLearning:
    def test_beats_persistence_on_drift(self):
        # Strong constant drift + small noise: a trained model should predict
        # the next position better than persistence (predict no movement).
        cl = 12
        df = _make_df(drift=(0.02, -0.01), noise=0.004, seed=0)
        model = Transformer(d_model=16, nhead=2, num_layers=1, context_length=cl,
                            epochs=60, batch_size=32, lr=1e-2, device="cpu").fit(df)

        trained = _mean_error(df, cl, model.predict)
        persistence = _mean_error(df, cl, lambda c: c.history[-1])
        assert trained < persistence

    def test_seeded_fit_is_deterministic(self, small_df):
        torch.manual_seed(0)
        m1 = Transformer(d_model=16, nhead=2, context_length=12, epochs=3, device="cpu").fit(small_df)
        torch.manual_seed(0)
        m2 = Transformer(d_model=16, nhead=2, context_length=12, epochs=3, device="cpu").fit(small_df)
        h = np.zeros((12, 2), dtype=np.float32)
        np.testing.assert_allclose(
            m1.predict(PredictionContext(history=h)),
            m2.predict(PredictionContext(history=h)),
            rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Time-of-day feature (Design B)
# ----------------------------------------------------------------------

class TestTimeFeatures:
    def test_input_dim_toggles(self):
        assert Transformer(use_time_features=False).input_dim == 2
        assert Transformer(use_time_features=True).input_dim == 4

    def test_make_windows_emits_time_channels(self, small_df):
        model = Transformer(d_model=16, nhead=2, context_length=12, use_time_features=True, device="cpu")
        X, Y, T = model._make_windows(small_df)
        assert T is not None
        assert T.shape == (X.shape[0], 11, 2)             # (n, W-1, [sin, cos])
        r2 = T[..., 0] ** 2 + T[..., 1] ** 2              # sin^2 + cos^2 == 1
        assert torch.allclose(r2, torch.ones_like(r2), atol=1e-5)

    def test_fit_builds_four_channel_net(self, fitted_time_model):
        assert fitted_time_model.use_time_features is True
        # attention sees time: the input projection lifts 4 channels (2 displacement + 2 time) into d_model
        assert fitted_time_model.net.input_proj.in_features == 4

    def test_predict_requires_timestamps(self, fitted_time_model):
        # Fitted with time but handed a context without per-fix times -> explicit error, not silent wrong answer.
        with pytest.raises(ValueError):
            fitted_time_model.predict(PredictionContext(history=np.zeros((12, 2), dtype=np.float32)))

    def test_predict_with_time_shape(self, fitted_time_model):
        history = (np.random.default_rng(3).standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        pred = fitted_time_model.predict(_context_with_time(history))
        assert pred.shape == (2,)
        assert np.all(np.isfinite(pred))

    def test_checkpoint_roundtrip_with_time(self, fitted_time_model, tmp_path):
        history = (np.random.default_rng(4).standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        ctx = _context_with_time(history)
        before = fitted_time_model.predict(ctx)

        path = tmp_path / "ckpt_time.pt"
        fitted_time_model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_time_features is True
        assert reloaded.input_dim == 4
        after = reloaded.predict(ctx)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------
# Day-of-year (season) feature
# ----------------------------------------------------------------------

class TestSeasonFeatures:
    def _season_model(self, **kw):
        return Transformer(d_model=16, nhead=2, context_length=12, epochs=2, batch_size=32,
                           device="cpu", **kw)

    def test_input_dim_toggles(self):
        assert Transformer(use_season_features=True).input_dim == 4
        assert Transformer(use_time_features=True, use_season_features=True).input_dim == 6

    def test_name_reflects_season(self):
        assert Transformer(d_model=32, num_layers=2, use_season_features=True).name == \
            "transformer_d32_l2_season"
        assert Transformer(d_model=32, num_layers=2, use_time_features=True,
                           use_season_features=True).name == "transformer_d32_l2_time_season"

    def test_make_windows_emits_season_channels(self, small_df):
        model = self._season_model(use_season_features=True)
        X, _, T = model._make_windows(small_df)
        assert T.shape == (X.shape[0], 11, 2)
        r2 = T[..., 0] ** 2 + T[..., 1] ** 2
        assert torch.allclose(r2, torch.ones_like(r2), atol=1e-5)

    def test_channel_order_is_time_then_season(self, small_df):
        # With both flags on the layout must be [disp, time_sin, time_cos, season_sin, season_cos];
        # each half must equal what the single-feature models emit, so fit/predict/checkpoints agree.
        both = self._season_model(use_time_features=True, use_season_features=True)
        only_time = self._season_model(use_time_features=True)
        only_season = self._season_model(use_season_features=True)

        _, _, T_both = both._make_windows(small_df)
        _, _, T_time = only_time._make_windows(small_df)
        _, _, T_season = only_season._make_windows(small_df)

        assert T_both.shape[-1] == 4
        assert torch.allclose(T_both[..., :2], T_time, atol=1e-6)
        assert torch.allclose(T_both[..., 2:], T_season, atol=1e-6)

    def test_season_is_near_constant_within_a_window(self, small_df):
        # Day of year barely moves across a 6 day window: it conditions the window rather than
        # differentiating positions (unlike time of day, which swings through the diurnal cycle).
        model = self._season_model(use_time_features=True, use_season_features=True)
        _, _, T = model._make_windows(small_df)
        time_spread = (T[..., 0].max(dim=1).values - T[..., 0].min(dim=1).values).mean()
        season_spread = (T[..., 2].max(dim=1).values - T[..., 2].min(dim=1).values).mean()
        assert season_spread < time_spread

    def test_fit_builds_six_channel_net(self, small_df):
        model = self._season_model(use_time_features=True, use_season_features=True).fit(small_df)
        assert model.net.input_proj.in_features == 6

    def test_predict_requires_timestamps_with_season_only(self, small_df):
        # Season alone still needs the per-fix times; missing them must raise, not silently mispredict.
        model = self._season_model(use_season_features=True).fit(small_df)
        with pytest.raises(ValueError):
            model.predict(PredictionContext(history=np.zeros((12, 2), dtype=np.float32)))

    def test_checkpoint_roundtrip_with_season(self, small_df, tmp_path):
        model = self._season_model(use_time_features=True, use_season_features=True).fit(small_df)
        history = (np.random.default_rng(7).standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        ctx = _context_with_time(history)
        before = model.predict(ctx)

        path = tmp_path / "ckpt_season.pt"
        model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_season_features is True
        assert reloaded.input_dim == 6
        np.testing.assert_allclose(before, reloaded.predict(ctx), rtol=1e-5, atol=1e-6)

    def test_old_checkpoint_defaults_to_no_season(self, small_df, tmp_path):
        # A checkpoint written before the flag existed must still load as a raw/time model.
        model = self._season_model(use_time_features=True).fit(small_df)
        path = tmp_path / "ckpt_legacy.pt"
        model.save(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["config"]["use_season_features"]
        torch.save(ckpt, path)

        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_season_features is False
        assert reloaded.input_dim == 4


# ----------------------------------------------------------------------
# Two-factor state head (Rung 2)
# ----------------------------------------------------------------------

class TestStateHead:
    def test_name_reflects_state_head(self):
        assert Transformer(d_model=32, num_layers=2, use_state_head=True).name == "transformer_d32_l2_state"
        assert (
            Transformer(d_model=32, num_layers=2, use_time_features=True,
                        use_season_features=True, use_state_head=True).name
            == "transformer_d32_l2_time_season_state"
        )

    def test_state_head_does_not_change_input_dim(self):
        # The head is on the window summary, not the input sequence, so input_dim is unaffected.
        assert Transformer(use_state_head=True).input_dim == 2
        assert Transformer(use_time_features=True, use_season_features=True, use_state_head=True).input_dim == 6

    def test_flag_off_net_is_unconditioned(self, fitted_model):
        # Default model: no state head, position head reads the window summary directly, forward returns one tensor.
        assert fitted_model.net.use_state_head is False
        assert fitted_model.net.head.in_features == fitted_model.d_model
        assert not hasattr(fitted_model.net, "state_head")
        out = fitted_model.net(torch.zeros(1, 11, 2))
        assert isinstance(out, torch.Tensor)

    def test_head_on_wires_conditioning(self, fitted_state_model):
        net = fitted_state_model.net
        assert net.use_state_head is True
        assert net.state_head.out_features == 5
        # position head is fed [window_summary, softmax(state)] -> d_model + n_states inputs
        assert net.head.in_features == fitted_state_model.d_model + 5
        pos, logits = net(torch.zeros(1, 11, 2))
        assert pos.shape == (1, 2)
        assert logits.shape == (1, 5)

    def test_state_labels_align_with_canonical_windows(self):
        # _state_labels must reproduce the standalone classifier's labels window-for-window (same labeller, same order).
        df = _make_seg_df()
        cl = 12
        model = Transformer(d_model=16, nhead=2, context_length=cl, use_state_head=True, device="cpu")
        S = model._state_labels(df, group_col="segment_id").numpy()
        sw = build_state_windows(df, context_length=cl)
        assert S.tolist() == sw.y.tolist()

    def test_fit_requires_segment_id(self, small_df):
        # small_df has no segment_id; a state-head fit must fail loudly rather than silently mislabel.
        with pytest.raises(ValueError):
            Transformer(d_model=16, nhead=2, context_length=12, epochs=1,
                        use_state_head=True, device="cpu").fit(small_df)

    def test_predict_shape_with_state_head(self, fitted_state_model):
        history = (np.random.default_rng(5).standard_normal((12, 2)) * 0.02).cumsum(axis=0)
        pred = fitted_state_model.predict(PredictionContext(history=history))
        assert pred.shape == (2,)
        assert np.all(np.isfinite(pred))

    def test_checkpoint_roundtrip_with_state_head(self, fitted_state_model, tmp_path):
        history = (np.random.default_rng(6).standard_normal((12, 2)) * 0.02).cumsum(axis=0)
        ctx = PredictionContext(history=history)
        before = fitted_state_model.predict(ctx)

        path = tmp_path / "ckpt_state.pt"
        fitted_state_model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_state_head is True
        assert reloaded.net.head.in_features == fitted_state_model.d_model + 5
        np.testing.assert_allclose(before, reloaded.predict(ctx), rtol=1e-5, atol=1e-6)

    def test_old_checkpoint_defaults_to_no_state_head(self, small_df, tmp_path):
        # A checkpoint predating the flag must load as an unconditioned model.
        model = Transformer(d_model=16, nhead=2, context_length=12, epochs=1, device="cpu").fit(small_df)
        path = tmp_path / "ckpt_prestate.pt"
        model.save(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["config"]["use_state_head"]
        torch.save(ckpt, path)

        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_state_head is False
        assert reloaded.net.head.in_features == reloaded.d_model


# ----------------------------------------------------------------------
# Mixture of experts (Rung 3)
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_moe_model():
    # Tiny MoE transformer on a segment frame; exercises the gate + experts training path.
    df = _make_seg_df()
    return Transformer(d_model=16, nhead=2, num_layers=1, dim_feedforward=32, context_length=12,
                       epochs=3, batch_size=32, use_moe=True, device="cpu").fit(df, group_col="segment_id")


class TestMoE:
    def test_moe_and_state_head_are_mutually_exclusive(self):
        # The gate already predicts the state; asking for both is ambiguous, so it must raise.
        with pytest.raises(ValueError):
            Transformer(use_moe=True, use_state_head=True)

    def test_expert_count_must_match_state_count(self):
        # Supervised gate needs one expert per state; a mismatch would only surface on real multi-state data.
        with pytest.raises(ValueError):
            Transformer(use_moe=True, n_experts=3)

    def test_name_reflects_moe(self):
        assert Transformer(d_model=32, num_layers=2, use_moe=True).name == "transformer_d32_l2_moe5"
        assert (
            Transformer(d_model=32, num_layers=2, use_time_features=True,
                        use_season_features=True, use_moe=True).name
            == "transformer_d32_l2_time_season_moe5"
        )

    def test_moe_does_not_change_input_dim(self):
        assert Transformer(use_moe=True).input_dim == 2
        assert Transformer(use_time_features=True, use_season_features=True, use_moe=True).input_dim == 6

    def test_net_has_gate_and_k_experts(self, fitted_moe_model):
        net = fitted_moe_model.net
        assert net.use_moe is True
        assert net.gate.out_features == 5
        assert net.experts.out_features == 5 * 2
        assert not hasattr(net, "head")          # no single shared position head
        pos, gate_logits = net(torch.zeros(1, 11, 2))
        assert pos.shape == (1, 2)
        assert gate_logits.shape == (1, 5)

    def test_gate_weights_form_a_distribution(self, fitted_moe_model):
        rng = np.random.default_rng(21)
        x = torch.tensor((rng.standard_normal((4, 11, 2)) * 0.02), dtype=torch.float32)
        fitted_moe_model.net.eval()
        with torch.no_grad():
            _, gate_logits = fitted_moe_model.net(x)
        g = torch.softmax(gate_logits, dim=-1)
        assert torch.allclose(g.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert (g >= 0).all()

    def test_prediction_is_the_gated_mixture_of_experts(self, fitted_moe_model):
        # forward must equal sum_k g_k * e_k recomputed from the same weights (eval mode: dropout off).
        net = fitted_moe_model.net
        net.eval()
        rng = np.random.default_rng(22)
        x = torch.tensor((rng.standard_normal((3, 11, 2)) * 0.02), dtype=torch.float32)
        with torch.no_grad():
            pos, gate_logits = net(x)
            h = net.pos_encoder(net.input_proj(x) * math.sqrt(net.d_model))
            last = net.encoder(h)[:, -1, :]
            g = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
            e = net.experts(last).view(-1, net.n_experts, 2)
            expected = (g * e).sum(dim=1)
        torch.testing.assert_close(pos, expected, rtol=1e-4, atol=1e-6)

    def test_experts_are_not_all_identical(self, fitted_moe_model):
        w = fitted_moe_model.net.experts.weight
        assert not torch.allclose(w[0:2], w[2:4], atol=1e-6)

    def test_predict_shape_with_moe(self, fitted_moe_model):
        history = (np.random.default_rng(23).standard_normal((12, 2)) * 0.02).cumsum(axis=0)
        pred = fitted_moe_model.predict(PredictionContext(history=history))
        assert pred.shape == (2,)
        assert np.all(np.isfinite(pred))

    def test_gate_labels_align_with_canonical_windows(self):
        df = _make_seg_df()
        cl = 12
        model = Transformer(d_model=16, nhead=2, context_length=cl, use_moe=True, device="cpu")
        S = model._state_labels(df, group_col="segment_id").numpy()
        assert S.tolist() == build_state_windows(df, context_length=cl).y.tolist()

    def test_checkpoint_roundtrip_with_moe(self, fitted_moe_model, tmp_path):
        history = (np.random.default_rng(24).standard_normal((12, 2)) * 0.02).cumsum(axis=0)
        ctx = PredictionContext(history=history)
        before = fitted_moe_model.predict(ctx)

        path = tmp_path / "ckpt_moe.pt"
        fitted_moe_model.save(path)
        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_moe is True
        assert reloaded.n_experts == 5
        assert reloaded.net.gate.out_features == 5
        np.testing.assert_allclose(before, reloaded.predict(ctx), rtol=1e-5, atol=1e-6)

    def test_old_checkpoint_defaults_to_no_moe(self, small_df, tmp_path):
        model = Transformer(d_model=16, nhead=2, context_length=12, epochs=1, device="cpu").fit(small_df)
        path = tmp_path / "ckpt_premoe.pt"
        model.save(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["config"]["use_moe"]
        torch.save(ckpt, path)

        reloaded = Transformer.from_checkpoint(path, device="cpu")
        assert reloaded.use_moe is False
        assert reloaded.net.head.in_features == reloaded.d_model