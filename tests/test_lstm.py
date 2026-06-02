# GenAI reference: Test suite also made with GenAI, mirroring the structure of our previous ones
# like test_arima.py / test_averaging.py etc. We asked it to cover protocol conformance, the 
# unfitted predict guard, fit/predict behaviour etc (and it added the heckpoint round trip, and a 
# small learning sanity check). Note, synthetic fixtures only; all models pinned to CPU and tiny 
# configs so the suite stays fast and deterministic.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from animal_movement_fm.eval.harness import iter_windows
from animal_movement_fm.models.lstm import LSTM


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
    """Mean Euclidean error of a predict_fn(history)->(2,) over all windows."""
    errs = []
    for _, traj in df.groupby("individual_id", sort=False):
        for history, target, _ts, _t in iter_windows(traj, context_length):
            errs.append(np.linalg.norm(predict_fn(history) - target))
    return float(np.mean(errs))


@pytest.fixture
def small_df():
    return _make_df()


@pytest.fixture(scope="module")
def fitted_model():
    # Tiny config + CPU so fitting is fast and deterministic in tests.
    df = _make_df()
    return LSTM(hidden_dim=8, num_layers=1, context_length=12,
                epochs=3, batch_size=32, device="cpu").fit(df)


# ----------------------------------------------------------------------
# Protocol conformance
# ----------------------------------------------------------------------

class TestProtocol:
    def test_has_predict_method(self):
        assert callable(getattr(LSTM(), "predict", None))

    def test_has_name_attribute(self):
        assert isinstance(LSTM().name, str)

    def test_name_reflects_architecture(self):
        assert LSTM(hidden_dim=32, num_layers=2).name == "lstm_h32_l2"


# ----------------------------------------------------------------------
# Unfitted guards
# ----------------------------------------------------------------------

class TestUnfitted:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LSTM().predict(np.zeros((12, 2), dtype=np.float32))

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            LSTM().save(tmp_path / "x.pt")


# ----------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, small_df):
        model = LSTM(hidden_dim=8, context_length=12, epochs=1, device="cpu")
        assert model.fit(small_df) is model

    def test_fit_sets_state(self, fitted_model):
        assert fitted_model.net is not None
        assert fitted_model._mean is not None
        assert fitted_model._std is not None

    def test_fit_no_windows_raises(self):
        # Every individual shorter than context_length + 1 -> no windows.
        df = _make_df(n_individuals=2, n_steps=5)
        with pytest.raises(ValueError):
            LSTM(context_length=24, epochs=1, device="cpu").fit(df)


# ----------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------

class TestPredict:
    def test_predict_shape(self, fitted_model):
        pred = fitted_model.predict(np.zeros((12, 2), dtype=np.float32))
        assert pred.shape == (2,)

    def test_predict_bad_shape_raises(self, fitted_model):
        with pytest.raises(ValueError):
            fitted_model.predict(np.zeros((12, 3)))

    def test_predict_short_history_falls_back(self, fitted_model):
        pred = fitted_model.predict(np.array([[1.0, 2.0]], dtype=np.float32))
        np.testing.assert_allclose(pred, [1.0, 2.0])

    def test_predict_finite_and_near_last(self, fitted_model):
        rng = np.random.default_rng(1)
        history = (rng.standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        pred = fitted_model.predict(history)
        assert np.all(np.isfinite(pred))
        # Catch catastrophic failure (NaN/inf/huge jump), not exact value.
        assert np.linalg.norm(pred - history[-1]) < 1.0


# ----------------------------------------------------------------------
# Checkpoint round-trip
# ----------------------------------------------------------------------

class TestCheckpoint:
    def test_roundtrip_predictions_match(self, fitted_model, tmp_path):
        history = (np.random.default_rng(2).standard_normal((12, 2)) * 0.01).cumsum(axis=0)
        before = fitted_model.predict(history)

        path = tmp_path / "ckpt.pt"
        fitted_model.save(path)
        reloaded = LSTM.from_checkpoint(path, device="cpu")
        after = reloaded.predict(history)

        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)

    def test_reloaded_restores_config(self, fitted_model, tmp_path):
        path = tmp_path / "ckpt.pt"
        fitted_model.save(path)
        reloaded = LSTM.from_checkpoint(path, device="cpu")
        assert reloaded.net is not None
        assert reloaded.hidden_dim == fitted_model.hidden_dim
        assert reloaded.context_length == fitted_model.context_length


# ----------------------------------------------------------------------
# Windowing
# ----------------------------------------------------------------------

class TestWindows:
    def test_window_shapes(self, small_df):
        model = LSTM(context_length=12, device="cpu")
        X, Y = model._make_windows(small_df)
        assert X.ndim == 3
        assert X.shape[1:] == (11, 2)   # context_length - 1 displacements, 2 channels
        assert Y.shape[1:] == (2,)
        assert X.shape[0] == Y.shape[0]


# ----------------------------------------------------------------------
# Learning sanity
# ----------------------------------------------------------------------

class TestLearning:
    def test_beats_persistence_on_drift(self):
        # Strong constant drift + small noise: a trained model should predict
        # the next position better than persistence (predict no movement).
        cl = 12
        df = _make_df(drift=(0.02, -0.01), noise=0.004, seed=0)
        model = LSTM(hidden_dim=16, context_length=cl,
                     epochs=60, batch_size=32, lr=1e-2, device="cpu").fit(df)

        trained = _mean_error(df, cl, model.predict)
        persistence = _mean_error(df, cl, lambda h: h[-1])
        assert trained < persistence

    def test_seeded_fit_is_deterministic(self, small_df):
        torch.manual_seed(0)
        m1 = LSTM(hidden_dim=8, context_length=12, epochs=3, device="cpu").fit(small_df)
        torch.manual_seed(0)
        m2 = LSTM(hidden_dim=8, context_length=12, epochs=3, device="cpu").fit(small_df)
        h = np.zeros((12, 2), dtype=np.float32)
        np.testing.assert_allclose(m1.predict(h), m2.predict(h), rtol=1e-5, atol=1e-6)