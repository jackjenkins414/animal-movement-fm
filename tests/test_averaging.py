"""Tests for the DisplacementAveraging baseline."""

from __future__ import annotations

import numpy as np
import pytest

from animal_movement_fm.eval.harness import PredictionContext
from animal_movement_fm.models.averaging import DisplacementAveraging

# GenAI reference: These tests are GenAI generated, where whe asked to "give me 
# concise pytest unit tests for my displacement-averaging movement baseline, 
# covering constructor validation, prediction behaviour across normal and edge 
# cases, shape errors, and core math properties of the displacement rule (see paper
# attached to this chat)"".


class TestDisplacementAveragingConstruction:
    def test_default_name_includes_k(self):
        assert DisplacementAveraging(k=1).name == "averaging_k1"
        assert DisplacementAveraging(k=10).name == "averaging_k10"

    def test_custom_name(self):
        assert DisplacementAveraging(k=5, name="cvm5").name == "cvm5"

    def test_rejects_zero_k(self):
        with pytest.raises(ValueError):
            DisplacementAveraging(k=0)

    def test_rejects_negative_k(self):
        with pytest.raises(ValueError):
            DisplacementAveraging(k=-1)

    def test_rejects_non_int_k(self):
        with pytest.raises(ValueError):
            DisplacementAveraging(k=1.5)


class TestDisplacementAveragingPredict:
    def test_constant_velocity_recovered(self):
        """k=1 on a perfectly linear trajectory predicts the next linear step."""
        # 0.1 deg east per step
        history = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        pred = DisplacementAveraging(k=1).predict(PredictionContext(history=history))
        np.testing.assert_allclose(pred, np.array([0.4, 0.0]))

    def test_averaging_smooths_noise(self):
        """Mean of [+1, -1, +1, -1] displacements is 0 → predicts current."""
        history = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        pred = DisplacementAveraging(k=4).predict(PredictionContext(history=history))
        # 4 displacements: +1, -1, +1, -1 → mean 0 → predict current = [0, 0]
        np.testing.assert_allclose(pred, np.array([0.0, 0.0]))

    def test_k_larger_than_available_clipped(self):
        """k=100 on a 3-step history uses the 2 available displacements."""
        history = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
        pred = DisplacementAveraging(k=100).predict(PredictionContext(history=history))
        # Both displacements are [0.1, 0]; mean is [0.1, 0]; pred = [0.3, 0]
        np.testing.assert_allclose(pred, np.array([0.3, 0.0]))

    def test_k1_uses_only_last_displacement(self):
        """k=1 must ignore older displacements even when they differ."""
        history = np.array([[0.0, 0.0], [5.0, 5.0], [5.1, 5.0]])
        pred = DisplacementAveraging(k=1).predict(PredictionContext(history=history))
        # Last displacement is [0.1, 0]; current is [5.1, 5.0]
        np.testing.assert_allclose(pred, np.array([5.2, 5.0]))

    def test_single_row_history_falls_back_to_persistence(self):
        """No displacements available → return the last position."""
        history = np.array([[7.0, 8.0]])
        pred = DisplacementAveraging(k=5).predict(PredictionContext(history=history))
        np.testing.assert_array_equal(pred, np.array([7.0, 8.0]))

    def test_output_shape(self):
        history = np.random.default_rng(0).standard_normal((24, 2))
        pred = DisplacementAveraging(k=3).predict(PredictionContext(history=history))
        assert pred.shape == (2,)

    def test_rejects_wrong_shape(self):
        model = DisplacementAveraging(k=1)
        with pytest.raises(ValueError):
            model.predict(PredictionContext(history=np.zeros((24, 3))))
        with pytest.raises(ValueError):
            model.predict(PredictionContext(history=np.zeros(24)))

    def test_rejects_empty_history(self):
        with pytest.raises(ValueError):
            DisplacementAveraging(k=1).predict(PredictionContext(history=np.zeros((0, 2))))

    def test_no_fit_method(self):
        """Averaging is stateless; no fit method."""
        assert not hasattr(DisplacementAveraging(k=1), "fit")


class TestDisplacementAveragingProperties:
    def test_zero_displacement_input_returns_current(self):
        """If history is constant, prediction equals that constant position."""
        history = np.full((10, 2), 3.14)
        for k in [1, 3, 9, 100]:
            pred = DisplacementAveraging(k=k).predict(PredictionContext(history=history))
            np.testing.assert_allclose(pred, np.array([3.14, 3.14]))

    def test_linearity(self):
        """Scaling all positions by c scales the prediction by c."""
        rng = np.random.default_rng(42)
        history = rng.standard_normal((20, 2))
        c = 7.5
        p1 = DisplacementAveraging(k=5).predict(PredictionContext(history=history))
        p2 = DisplacementAveraging(k=5).predict(PredictionContext(history=c * history))
        np.testing.assert_allclose(p2, c * p1)

    def test_translation_invariance_of_displacement(self):
        """Adding a constant offset to every row shifts the prediction by the same offset."""
        rng = np.random.default_rng(1)
        history = rng.standard_normal((20, 2))
        offset = np.array([100.0, -50.0])
        p1 = DisplacementAveraging(k=5).predict(PredictionContext(history=history))
        p2 = DisplacementAveraging(k=5).predict(PredictionContext(history=history + offset))
        np.testing.assert_allclose(p2, p1 + offset)