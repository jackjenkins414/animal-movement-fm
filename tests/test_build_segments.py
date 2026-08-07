# GenAI reference: I asked for tests covering our shared segment building rule after we extracted it from four separate 
# copies (build_segments, xgboost_timefeature_6h, and both cadence sweeps). I specified the behaviours; that string and 
# integer ids both work, that deployment is optional and changes the segment id, that per-individual metadata is carried 
# but per fix quality is not. It also added another case too. I verified each assertion against the intended logic. 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.build_segments import (
    segments_from_fixes,
    trajectory_segment_frames,
)
from animal_movement_fm.data.resample import BIN_SECONDS

SIX_HOURS = pd.Timedelta(seconds=BIN_SECONDS)


def fixes(individual_id, n=60, start="2020-01-01", step_h=6, lat0=0.0, **extra):
    """A regular track dense enough to survive resampling at the cadence."""
    stamps = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * step_h, unit="h")
    frame = pd.DataFrame({
        "individual_id": individual_id,
        "timestamp": stamps,
        "lat": lat0 + np.arange(n) * 0.01,
        "lon": 0.0,
    })
    for key, value in extra.items():
        frame[key] = value
    return frame


class TestTrajectoryRule:
    def test_segment_id_includes_deployment_when_present(self):
        frames = trajectory_segment_frames(
            fixes(1)["timestamp"].to_numpy(),
            fixes(1)["lat"].to_numpy(),
            fixes(1)["lon"].to_numpy(),
            individual_id=7,
            deployment_id=99,
            min_segment_bins=25,
        )
        assert frames[0]["segment_id"].iloc[0] == "7_99_0"
        assert "deployment_id" in frames[0].columns

    def test_segment_id_omits_deployment_when_absent(self):
        frames = trajectory_segment_frames(
            fixes(1)["timestamp"].to_numpy(),
            fixes(1)["lat"].to_numpy(),
            fixes(1)["lon"].to_numpy(),
            individual_id="CAR2013_sel_a_m_03",
            min_segment_bins=25,
        )
        assert frames[0]["segment_id"].iloc[0] == "CAR2013_sel_a_m_03_0"
        assert "deployment_id" not in frames[0].columns

    def test_too_short_a_trajectory_yields_nothing(self):
        short = fixes(1, n=10)
        frames = trajectory_segment_frames(
            short["timestamp"].to_numpy(), short["lat"].to_numpy(), short["lon"].to_numpy(),
            individual_id=1, min_segment_bins=25,
        )
        assert frames == []

    def test_columns_match_the_pipeline_schema(self):
        frames = trajectory_segment_frames(
            fixes(1)["timestamp"].to_numpy(),
            fixes(1)["lat"].to_numpy(),
            fixes(1)["lon"].to_numpy(),
            individual_id=1, deployment_id=2, min_segment_bins=25,
        )
        assert list(frames[0].columns) == [
            "segment_id", "individual_id", "deployment_id", "bin_index",
            "timestamp", "lat", "lon", "centre_offset_s",
        ]


class TestFrameLevelRule:
    def test_string_ids_survive_end_to_end(self):
        """RAATD ids are strings; an int cast anywhere in this path would kill the marine leg."""
        frame = pd.concat([fixes("CAR2013_sel_a_m_01"), fixes("CAR2013_sel_a_m_02")])

        segments, stats = segments_from_fixes(frame, min_segment_bins=25)

        assert stats["individuals_out"] == 2
        assert set(segments["individual_id"]) == {"CAR2013_sel_a_m_01", "CAR2013_sel_a_m_02"}

    def test_deployments_are_separate_trajectories(self):
        """Concatenating two deployments of one animal would invent a gap crossing or a spurious
        boundary depending on their spacing, so they must be resampled separately."""
        a = fixes(1, start="2020-01-01", deployment_id=10)
        b = fixes(1, start="2021-01-01", deployment_id=11)

        segments, stats = segments_from_fixes(pd.concat([a, b]), min_segment_bins=25)

        assert stats["trajectories"] == 2
        assert segments["segment_id"].nunique() == 2
        assert stats["individuals_out"] == 1

    def test_individual_level_metadata_is_carried(self):
        frame = pd.concat([
            fixes("a", device_type="PTT", sex="F"),
            fixes("b", device_type="GPS", sex="M"),
        ])

        segments, _ = segments_from_fixes(
            frame, min_segment_bins=25, carry_columns=("device_type", "sex")
        )

        by_id = segments.groupby("individual_id")["device_type"].first().to_dict()
        assert by_id == {"a": "PTT", "b": "GPS"}

    def test_a_species_yielding_nothing_is_reported_not_raised(self):
        """Expected for short-deployment seabirds at a high threshold; it is an eligibility fact."""
        segments, stats = segments_from_fixes(fixes(1, n=10), min_segment_bins=120)

        assert segments is None
        assert stats["segments"] == 0
        assert stats["trajectories_no_segments"] == 1

    def test_missing_required_columns_raise(self):
        with pytest.raises(ValueError, match="missing required columns"):
            segments_from_fixes(pd.DataFrame({"individual_id": [1], "timestamp": [pd.Timestamp.now()]}))

    def test_native_cadence_is_reported_not_enforced(self):
        """A track coarser than the bin is not rejected here; it fails the window floor later."""
        coarse = fixes(1, n=60, step_h=24)

        _, stats = segments_from_fixes(coarse, min_segment_bins=1)

        assert stats["median_native_gap_h"] == pytest.approx(24.0)
        assert stats["share_gaps_within_bin"] == pytest.approx(0.0)

    def test_min_segment_bins_of_25_cannot_lose_a_valid_window(self):
        """A contiguous run of 25 bins cannot exist inside a segment shorter than 25 bins, so
        filtering segments at 25 is lossless with respect to windows."""
        frame = fixes(1, n=30)

        at_25, _ = segments_from_fixes(frame, min_segment_bins=25)
        at_1, _ = segments_from_fixes(frame, min_segment_bins=1)

        assert len(at_25) == len(at_1)
