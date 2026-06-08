# GenAI reference: I asked GenAI to write these tests, as per previously. I told it the previous validation test I 
# ran was analytical (see comment above `track_randomness`), and asked for more thorough tests. Each one I validated 
# for correctness (which they are). 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.analysis.randomness import (
    directional_persistence,
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