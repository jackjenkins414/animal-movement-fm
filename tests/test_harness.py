"""
Tests for the evaluation harness.

Two layers:
1. Metrics: haversine numerics tested against known great circle distances.
2. Harness: end-to-end tests using a trivial dummy predictor.

Reference distances for haversine tests come from authoritative sources
(NIST / well-known geographic constants). They are NOT computed by another
implementation of the same formula — that would be circular. Using
independent ground truth catches errors that a self-consistent
implementation would hide.
"""

# Declaration: We asked GenAI to generate a pytest suite for synthetic.py,
# harness.py and metrics.py, asking to generate tests for the evaluation harness. 
# The code returned from GenAI was checked to ensure correctness, and modified 
# with docstrings and comments. 

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.synthetic import generate_brownian_motion
from animal_movement_fm.eval.harness import (
    Predictor,
    contiguous_run_slices,
    evaluate,
    iter_windows,
    stratified_summary,
)
from animal_movement_fm.eval.metrics import (
    EARTH_RADIUS_M,
    euclidean_distance_degrees,
    haversine_distance,
    summarise_errors,
)


# ---------------------------------------------------------------------------
# Metrics: haversine
# ---------------------------------------------------------------------------

class TestHaversine:

    def test_zero_distance(self):
        """Distance from a point to itself is exactly 0."""
        assert haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0
        assert haversine_distance(150.0, -33.0, 150.0, -33.0) == 0.0

    def test_one_degree_at_equator_is_111km(self):
        """
        One degree of latitude is 111.195 km anywhere on Earth (a known
        geographic constant; comes from R * pi/180 with R=6371km).
        """
        d = haversine_distance(0.0, 0.0, 0.0, 1.0)
        assert abs(d - 111_194.93) < 5.0  # within 5 m of the textbook value

    def test_one_degree_longitude_at_equator(self):
        """One degree of longitude at the equator equals one degree of lat."""
        d = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert abs(d - 111_194.93) < 5.0

    def test_one_degree_longitude_shrinks_with_latitude(self):
        """
        At 60 degrees latitude, one degree of longitude is half the equatorial
        distance (cos(60) = 0.5). This is the property that Euclidean
        (lon, lat) distance fails to capture.
        """
        eq = haversine_distance(0.0, 0.0, 1.0, 0.0)
        high = haversine_distance(0.0, 60.0, 1.0, 60.0)
        assert abs(high / eq - 0.5) < 0.001

    def test_antipodal_distance(self):
        """
        Antipodal points (opposite sides of the globe) are pi*R apart.
        For R = 6371 km that is 20_015.087 km.
        """
        d = haversine_distance(0.0, 0.0, 180.0, 0.0)
        expected = np.pi * EARTH_RADIUS_M
        assert abs(d - expected) < 1.0

    def test_symmetric(self):
        """d(A, B) == d(B, A) always."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            lon1, lat1 = rng.uniform(-180, 180), rng.uniform(-89, 89)
            lon2, lat2 = rng.uniform(-180, 180), rng.uniform(-89, 89)
            d_ab = haversine_distance(lon1, lat1, lon2, lat2)
            d_ba = haversine_distance(lon2, lat2, lon1, lat1)
            assert abs(d_ab - d_ba) < 1e-6

    def test_vectorised(self):
        """Arrays in, arrays out, with correct shape."""
        lon1 = np.array([0.0, 1.0, 2.0])
        lat1 = np.array([0.0, 0.0, 0.0])
        lon2 = np.array([0.0, 1.0, 2.0])
        lat2 = np.array([1.0, 1.0, 1.0])
        d = haversine_distance(lon1, lat1, lon2, lat2)
        assert d.shape == (3,)
        # Each pair is a 1-degree-latitude separation.
        assert np.allclose(d, 111_194.93, atol=5.0)


class TestEuclideanDegrees:
    def test_zero_distance(self):
        assert euclidean_distance_degrees(0.0, 0.0, 0.0, 0.0) == 0.0

    def test_unit(self):
        assert euclidean_distance_degrees(0.0, 0.0, 0.0, 1.0) == pytest.approx(1.0)

    def test_pythagorean(self):
        d = euclidean_distance_degrees(0.0, 0.0, 3.0, 4.0)
        assert d == pytest.approx(5.0)


class TestSummariseErrors:
    def test_basic(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s = summarise_errors(errors)
        assert s["n"] == 5
        assert s["mean_m"] == 3.0
        assert s["median_m"] == 3.0


# ---------------------------------------------------------------------------
# Harness: end-to-end with a dummy predictor
# ---------------------------------------------------------------------------

class _ConstantOffsetPredictor:
    """
    Always predicts (current + offset). With offset=(0, 0) this is the
    persistence baseline. With other offsets we get controlled, predictable
    errors which let us verify the harness arithmetic.
    """

    def __init__(self, offset: tuple[float, float] = (0.0, 0.0)):
        self.offset = np.asarray(offset)
        self.name = f"ConstantOffset({offset})"

    def predict(self, context) -> np.ndarray:
        return context.history[-1] + self.offset


class TestHarness:

    def test_contiguous_runs_split_at_missing_bin(self):
        trajectory = pd.DataFrame(
            {
                "bin_index": [10, 11, 12, 14, 15, 16],
                "timestamp": pd.date_range("2020-01-01", periods=6, freq="6h"),
                "lon": np.arange(6, dtype=float),
                "lat": 0.0,
            }
        )

        assert contiguous_run_slices(trajectory) == [(0, 3), (3, 6)]

        windows = iter_windows(trajectory, context_length=2)
        assert [target_idx for *_rest, target_idx in windows] == [2, 5]

    def test_window_never_uses_gap_as_history_or_target(self):
        trajectory = pd.DataFrame(
            {
                "bin_index": [0, 1, 2, 4, 5, 6],
                "timestamp": pd.date_range("2020-01-01", periods=6, freq="6h"),
                "lon": np.arange(6, dtype=float),
                "lat": 0.0,
            }
        )

        windows = iter_windows(trajectory, context_length=2)
        histories = [history[:, 0].tolist() for history, *_rest in windows]
        targets = [float(target[0]) for _history, target, *_rest in windows]

        assert histories == [[0.0, 1.0], [3.0, 4.0]]
        assert targets == [2.0, 5.0]

    def test_predictor_protocol(self):
        """A class with a `predict` method satisfies the Predictor protocol."""
        p = _ConstantOffsetPredictor()
        assert isinstance(p, Predictor)

    def test_evaluate_returns_expected_columns(self):
        df = generate_brownian_motion(n_individuals=2, n_steps=50, seed=0)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=10, show_progress=False)
        expected_cols = {
            "individual_id",
            "timestamp",
            "target_lon",
            "target_lat",
            "pred_lon",
            "pred_lat",
            "error_m",
            "error_deg",
        }
        assert expected_cols.issubset(results.columns)

    def test_evaluate_row_count(self):
        """For T=50 and W=10, each individual produces 40 predictions."""
        df = generate_brownian_motion(n_individuals=3, n_steps=50, seed=0)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=10, show_progress=False)
        assert len(results) == 3 * (50 - 10)

    def test_evaluate_skips_short_trajectories(self):
        """Trajectories shorter than the context window produce no rows."""
        df = generate_brownian_motion(n_individuals=2, n_steps=5, seed=0)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=10, show_progress=False)
        assert len(results) == 0

    def test_persistence_error_is_just_the_displacement(self):
        """
        Persistence predicts current = next. So error_m must equal the
        haversine between consecutive observations. We check this directly
        because it validates that the harness is pairing predictions with
        the right targets.
        """
        df = generate_brownian_motion(n_individuals=1, n_steps=20, seed=42)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=5, show_progress=False)

        # For each result row, recompute expected error from the raw data.
        df_sorted = df.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)
        for row in results.itertuples():
            # Find the position of the target in df_sorted, then look at the row before.
            mask = (df_sorted["individual_id"] == row.individual_id) & (
                df_sorted["timestamp"] == row.timestamp
            )
            target_idx = df_sorted.index[mask][0]
            prev = df_sorted.iloc[target_idx - 1]
            expected_err = haversine_distance(
                prev["lon"], prev["lat"], row.target_lon, row.target_lat
            )
            assert abs(row.error_m - expected_err) < 1e-6

    def test_known_offset_produces_known_error(self):
        """
        A predictor that always shifts by (0, 1 degree) of latitude should
        produce errors of approximately 111 km (one degree of latitude) on
        a Brownian-motion trajectory where consecutive observations are
        very close. This is a direct test that prediction and target are
        wired up the right way.
        """
        df = generate_brownian_motion(
            n_individuals=1, n_steps=20, step_std=0.0001, seed=0  # tiny BM
        )
        results = evaluate(
            _ConstantOffsetPredictor(offset=(0.0, 1.0)),
            df,
            context_length=5,
            show_progress=False,
        )
        # Errors should be very close to 111 km because the predictor
        # systematically over-shoots latitude by 1 degree.
        assert results["error_m"].mean() == pytest.approx(111_194.93, abs=200.0)

    def test_extra_cols_carried_through(self):
        """species/class columns should be propagated into results."""
        df = generate_brownian_motion(n_individuals=2, n_steps=20, seed=0)
        df["species"] = "test_species"
        df["class"] = "test_class"
        results = evaluate(
            _ConstantOffsetPredictor(),
            df,
            context_length=5,
            show_progress=False,
            extra_cols=("species", "class"),
        )
        assert (results["species"] == "test_species").all()
        assert (results["class"] == "test_class").all()


class TestStratifiedSummary:

    def test_overall_summary(self):
        df = generate_brownian_motion(n_individuals=3, n_steps=30, seed=0)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=5, show_progress=False)
        summary = stratified_summary(results)
        assert len(summary) == 1
        assert "mean_m" in summary.columns
        assert "median_m" in summary.columns

    def test_grouped_summary(self):
        df = generate_brownian_motion(n_individuals=3, n_steps=30, seed=0)
        results = evaluate(_ConstantOffsetPredictor(), df, context_length=5, show_progress=False)
        summary = stratified_summary(results, by="individual_id")
        assert len(summary) == 3
        assert "individual_id" in summary.columns
        assert "mean_m" in summary.columns
