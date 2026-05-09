"""
Tests for synthetic data generators.

Three things per generator:
1. Schema correctness — output has the right columns, dtypes, and shape.
2. Reproducibility — same seed gives same output, different seeds differ.
3. Statistical properties — generated data matches theoretical expectations.

The third class of test is the most valuable: it catches bugs where the code
runs and produces sensible-looking output, but the output does not actually
match the model we claim to be sampling from. These bugs are invisible
without ground-truth checks.
"""

# Declaration: We asked GenAI to generate a pytest suite for synthetic.py,
# asking to generate tests for correctness (both statistical and schema); 
# it added reproducibility tests as well. The code from GenAI has been slighlty 
# modified with comments, and some docstrings. 

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.synthetic import (
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)


# ---------------------------------------------------------------------------
# Brownian motion
# ---------------------------------------------------------------------------

class TestBrownianMotion:

    def test_schema(self):
        """Output has the canonical columns and the right total length."""
        df = generate_brownian_motion(n_individuals=5, n_steps=100, seed=42)
        assert list(df.columns) == ["individual_id", "timestamp", "lon", "lat"]
        assert len(df) == 500
        assert df["individual_id"].nunique() == 5
        # Each individual has exactly n_steps observations.
        assert (df.groupby("individual_id").size() == 100).all()

    def test_reproducibility(self):
        """Same seed => identical output. Different seed => different output."""
        df1 = generate_brownian_motion(n_individuals=3, n_steps=50, seed=42)
        df2 = generate_brownian_motion(n_individuals=3, n_steps=50, seed=42)
        df3 = generate_brownian_motion(n_individuals=3, n_steps=50, seed=43)
        pd.testing.assert_frame_equal(df1, df2)
        # If seed=43 produced the same lon/lat as seed=42, our seeding is broken.
        assert not np.allclose(df1["lon"].values, df3["lon"].values)

    def test_starts_at_origin(self):
        """First observation per individual sits at the requested start point."""
        df = generate_brownian_motion(
            n_individuals=4, n_steps=10, start_lon=1.5, start_lat=-2.0, seed=0
        )
        first_obs = df.groupby("individual_id").first()
        assert np.allclose(first_obs["lon"], 1.5)
        assert np.allclose(first_obs["lat"], -2.0)

    def test_increment_distribution(self):
        """
        Per-step displacements should be approximately N(0, step_std^2).

        Brownian motion's defining property is that increments are iid Gaussian
        with the requested variance. We sample many trajectories and check
        the empirical std of dlon and dlat is close to step_std.
        """
        step_std = 0.02
        df = generate_brownian_motion(
            n_individuals=200, n_steps=500, step_std=step_std, seed=123
        )
        # Compute per-individual differences, then stack.
        df_sorted = df.sort_values(["individual_id", "timestamp"])
        dlon = df_sorted.groupby("individual_id")["lon"].diff().dropna()
        dlat = df_sorted.groupby("individual_id")["lat"].diff().dropna()

        # With ~100k samples, std should match to within ~1%.
        assert abs(dlon.std() - step_std) / step_std < 0.02
        assert abs(dlat.std() - step_std) / step_std < 0.02
        # Mean should be ~0.
        assert abs(dlon.mean()) < 0.001
        assert abs(dlat.mean()) < 0.001


# ---------------------------------------------------------------------------
# Correlated random walk
# ---------------------------------------------------------------------------

class TestCRW:

    def test_schema(self):
        df = generate_crw(n_individuals=5, n_steps=100, seed=42)
        assert list(df.columns) == ["individual_id", "timestamp", "lon", "lat"]
        assert len(df) == 500

    def test_reproducibility(self):
        df1 = generate_crw(n_individuals=3, n_steps=50, seed=42)
        df2 = generate_crw(n_individuals=3, n_steps=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_directional_persistence(self):
        """
        Consecutive headings should be correlated when kappa > 0.

        Direct test of the key property that distinguishes CRW from BM:
        the heading at step t+1 is concentrated around the heading at step t.
        We compute the angular difference between consecutive headings and
        check it has lower circular variance than a uniform distribution.
        """
        df = generate_crw(
            n_individuals=50, n_steps=500, turning_kappa=4.0, seed=0
        )
        df = df.sort_values(["individual_id", "timestamp"])
        # Heading at step t is atan2(dlat_t, dlon_t).
        df["dlon"] = df.groupby("individual_id")["lon"].diff()
        df["dlat"] = df.groupby("individual_id")["lat"].diff()
        df = df.dropna()
        df["heading"] = np.arctan2(df["dlat"], df["dlon"])

        # Differences between consecutive headings, wrapped to [-pi, pi].
        df["heading_diff"] = df.groupby("individual_id")["heading"].diff()
        diffs = df["heading_diff"].dropna()
        diffs = np.arctan2(np.sin(diffs), np.cos(diffs))

        # Mean resultant length: 1.0 = perfectly aligned, 0.0 = uniform.
        # With kappa=4, theoretical R is I_1(4)/I_0(4) ~ 0.86.
        R = np.sqrt(np.mean(np.cos(diffs)) ** 2 + np.mean(np.sin(diffs)) ** 2)
        assert R > 0.7, f"Expected strong directional persistence, got R={R:.3f}"

    def test_low_kappa_approximates_brownian(self):
        """With kappa~0, the CRW reduces to (approximately) Brownian motion."""
        df = generate_crw(
            n_individuals=20, n_steps=500, turning_kappa=0.0, seed=7
        )
        df = df.sort_values(["individual_id", "timestamp"])
        df["dlon"] = df.groupby("individual_id")["lon"].diff()
        df["dlat"] = df.groupby("individual_id")["lat"].diff()
        df = df.dropna()
        df["heading"] = np.arctan2(df["dlat"], df["dlon"])
        df["heading_diff"] = df.groupby("individual_id")["heading"].diff().dropna()
        diffs = np.arctan2(np.sin(df["heading_diff"].dropna()), np.cos(df["heading_diff"].dropna()))
        R = np.sqrt(np.mean(np.cos(diffs)) ** 2 + np.mean(np.sin(diffs)) ** 2)
        # No directional persistence => R near 0.
        assert R < 0.15


# ---------------------------------------------------------------------------
# State-switching CRW
# ---------------------------------------------------------------------------

class TestSwitchingCRW:

    def test_schema_includes_state(self):
        """The state column is present as ground truth (but not a model input)."""
        df = generate_switching_crw(n_individuals=5, n_steps=100, seed=42)
        assert "state" in df.columns
        assert set(df["state"].unique()).issubset({0, 1})

    def test_reproducibility(self):
        df1 = generate_switching_crw(n_individuals=3, n_steps=50, seed=42)
        df2 = generate_switching_crw(n_individuals=3, n_steps=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_state_step_length_relationship(self):
        """
        Foraging-state steps should be shorter than transit-state steps.

        This is the defining property of the two-state model: if our generator
        is correct, mean step length conditional on state must reflect the
        configured scale parameters.
        """
        df = generate_switching_crw(
            n_individuals=30,
            n_steps=2000,
            foraging_step_scale=0.001,
            transit_step_scale=0.01,
            seed=99,
        )
        df = df.sort_values(["individual_id", "timestamp"])
        df["dlon"] = df.groupby("individual_id")["lon"].diff()
        df["dlat"] = df.groupby("individual_id")["lat"].diff()
        df["step_length"] = np.sqrt(df["dlon"] ** 2 + df["dlat"] ** 2)

        foraging_mean = df.loc[df["state"] == 0, "step_length"].mean()
        transit_mean = df.loc[df["state"] == 1, "step_length"].mean()
        # Transit steps should be roughly 10x foraging given the configured scales.
        assert transit_mean > foraging_mean * 5

    def test_state_persistence(self):
        """
        With p_stay=0.95, mean run-length per state should be ~20 steps.

        This catches a class of bugs where transitions happen too often
        (e.g. comparing > vs <) which would silently break the HMM dynamics.
        """
        df = generate_switching_crw(
            n_individuals=20, n_steps=2000, p_stay=0.95, seed=11
        )
        # Count run lengths per individual.
        run_lengths = []
        for _, group in df.groupby("individual_id"):
            states = group["state"].values
            # Find indices where state changes.
            change_points = np.where(np.diff(states) != 0)[0] + 1
            runs = np.diff(np.concatenate([[0], change_points, [len(states)]]))
            run_lengths.extend(runs)

        mean_run = np.mean(run_lengths)
        # Theoretical mean run length is 1 / (1 - p_stay) = 20.
        # Allow a wide tolerance for sample noise.
        assert 12 < mean_run < 30, f"Expected mean run-length ~20, got {mean_run:.1f}"
