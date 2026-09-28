# GenAI reference: I asked GenAI to write these tests, as per previously. I told it the previous validation test I 
# ran was analytical (see comment above `track_randomness`), and asked for more thorough tests. Each one I validated 
# for correctness (which they are). 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.analysis.randomness import (
    directional_persistence,
    mean_resultant_length,
    mean_resultant_length_corrected,
    track_randomness,
)


class TestDirectionalPersistence:
    def test_straight_line_is_one(self):
        # Every step heads the same way -> turning angle 0 -> cos 1.
        pos = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        assert directional_persistence(pos) == pytest.approx(1.0)

    def test_right_angle_zigzag_is_zero(self):
        # Alternating 90-degree turns -> cos 0 each -> mean 0.
        pos = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
        assert directional_persistence(pos) == pytest.approx(0.0)

    def test_reversal_is_minus_one(self):
        # 180-degree turns -> cos -1 each.
        pos = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
        assert directional_persistence(pos) == pytest.approx(-1.0)

    def test_too_short_is_nan(self):
        pos = np.array([[0, 0], [1, 0]], dtype=float)  # one step, no turn defined
        assert np.isnan(directional_persistence(pos))

    def test_zero_length_steps_dropped(self):
        # A pause (repeated point) has no heading; it should be dropped, not
        # poison the result with a 0/0 turn.
        pos = np.array([[0, 0], [1, 0], [1, 0], [2, 0]], dtype=float)
        assert directional_persistence(pos) == pytest.approx(1.0)


class TestTrackRandomness:
    @pytest.fixture
    def two_tracks(self):
        straight = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        zigzag = np.array([[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
        frames = []
        for ind, pos in [("a", straight), ("b", zigzag)]:
            frames.append(
                pd.DataFrame(
                    {
                        "individual_id": ind,
                        "timestamp": np.arange(len(pos)),
                        "lon": pos[:, 0],
                        "lat": pos[:, 1],
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_one_row_per_individual(self, two_tracks):
        out = track_randomness(two_tracks)
        assert len(out) == 2
        assert set(out.columns) == {
            "individual_id",
            "n_steps",
            "persistence",
            "randomness",
        }

    def test_values_match_single_track(self, two_tracks):
        out = track_randomness(two_tracks).set_index("individual_id")
        assert out.loc["a", "persistence"] == pytest.approx(1.0)
        assert out.loc["b", "persistence"] == pytest.approx(0.0)

    def test_randomness_is_complement(self, two_tracks):
        out = track_randomness(two_tracks)
        assert np.allclose(out["randomness"], 1.0 - out["persistence"])

# GenAI reference: Rbar helpers; GenAI wrote them.
class TestMeanResultantLength:
    def test_straight_is_one(self):
        assert mean_resultant_length(np.zeros(50)) == pytest.approx(1.0)

    def test_reversal_is_one(self):
        # Rbar measures concentration, not direction: constant reversals are perfectly predictable.
        assert mean_resultant_length(np.full(50, np.pi)) == pytest.approx(1.0)

    def test_constant_right_angle_is_one(self):
        assert mean_resultant_length(np.full(50, np.pi / 2)) == pytest.approx(1.0)

    def test_alternating_right_angles_is_zero(self):
        turns = np.tile([np.pi / 2, -np.pi / 2], 25)
        assert mean_resultant_length(turns) == pytest.approx(0.0, abs=1e-12)

    def test_uniform_random_is_near_zero(self):
        turns = np.random.default_rng(42).uniform(-np.pi, np.pi, 100_000)
        assert mean_resultant_length(turns) < 0.01

    def test_bounds_rho(self):
        turns = np.random.default_rng(0).vonmises(0.5, 2.0, 500)
        assert mean_resultant_length(turns) >= abs(np.cos(turns).mean())

    def test_empty_is_nan(self):
        assert np.isnan(mean_resultant_length(np.array([])))
        assert np.isnan(mean_resultant_length(np.array([np.nan])))


class TestMeanResultantLengthCorrected:
    def test_uniform_random_is_near_zero(self):
        turns = np.random.default_rng(42).uniform(-np.pi, np.pi, 100_000)
        assert mean_resultant_length_corrected(turns) < 0.01

    def test_removes_small_sample_floor(self):
        # With few random turns the raw value sits near sqrt(pi / 4n); the corrected one is well below it on average.
        rng = np.random.default_rng(7)
        raw, corrected = [], []
        for _ in range(500):
            turns = rng.uniform(-np.pi, np.pi, 30)
            raw.append(mean_resultant_length(turns))
            corrected.append(mean_resultant_length_corrected(turns))
        assert np.mean(raw) == pytest.approx(np.sqrt(np.pi / (4 * 30)), rel=0.1)
        assert np.mean(corrected) < 0.5 * np.mean(raw)

    def test_straight_equals_raw(self):
        turns = np.zeros(40)
        assert mean_resultant_length_corrected(turns) == pytest.approx(mean_resultant_length(turns))

    def test_concentrated_converges_to_raw(self):
        turns = np.random.default_rng(1).vonmises(0.0, 20.0, 200_000)
        assert mean_resultant_length_corrected(turns) == pytest.approx(mean_resultant_length(turns), abs=1e-4)

    def test_too_few_is_nan(self):
        assert np.isnan(mean_resultant_length_corrected(np.array([0.3])))
