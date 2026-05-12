"""Tests for the Persistence baseline."""

from __future__ import annotations

import numpy as np
import pytest

from animal_movement_fm.models.persistence import Persistence

# GenAI reference: All tests here are GenAI generated, where the prompt was "write concise 
# pytest unit tests for my Persistence baseline whose predict(history) returns a copy of the 
# last (lon, lat) row, and be sure to validate output shape, inputs, edge cases, 
# invalid shapes, etc.


class TestPersistence:
    def test_has_name(self):
        assert Persistence().name == "persistence"

    def test_predict_returns_last_row(self):
        model = Persistence()
        history = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        pred = model.predict(history)
        np.testing.assert_array_equal(pred, np.array([5.0, 6.0]))

    def test_predict_output_shape(self):
        model = Persistence()
        history = np.zeros((24, 2))
        pred = model.predict(history)
        assert pred.shape == (2,)

    def test_predict_returns_copy_not_view(self):
        """Mutating the prediction must not corrupt history."""
        model = Persistence()
        history = np.array([[1.0, 2.0], [3.0, 4.0]])
        pred = model.predict(history)
        pred[0] = 999.0
        assert history[-1, 0] == 3.0

    def test_predict_single_row_history(self):
        model = Persistence()
        history = np.array([[7.0, 8.0]])
        pred = model.predict(history)
        np.testing.assert_array_equal(pred, np.array([7.0, 8.0]))

    def test_rejects_wrong_shape(self):
        model = Persistence()
        with pytest.raises(ValueError):
            model.predict(np.zeros((24, 3)))
        with pytest.raises(ValueError):
            model.predict(np.zeros(24))

    def test_rejects_empty_history(self):
        model = Persistence()
        with pytest.raises(ValueError):
            model.predict(np.zeros((0, 2)))

    def test_no_fit_method(self):
        """Persistence is stateless; it should not advertise a fit method."""
        assert not hasattr(Persistence(), "fit")

    def test_accepts_list_input(self):
        """The Protocol says ndarray; in practice np.asarray is forgiving."""
        model = Persistence()
        pred = model.predict([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(pred, np.array([3.0, 4.0]))
