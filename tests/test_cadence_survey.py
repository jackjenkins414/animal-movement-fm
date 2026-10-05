# GenAI reference: I asked GenAI to make a test file for the corresponding script it made; ensuring to test for
# the cadence surveys window counting, the gap rule invariance and the phase hole detector
from __future__ import annotations

import numpy as np
import pandas as pd

import experiments.eda.cadence_survey as cs
from animal_movement_fm.data.build_segments import segments_from_fixes
from animal_movement_fm.data.resample import MIN_SEGMENT_BINS, _to_bins
from animal_movement_fm.eval.harness import iter_windows

HOUR = 3600


def fixes_at(bins: list[int], grid_s: int, individual: str = "a") -> pd.DataFrame:
    """One fix per listed bin, at the bin centre, on a small eastward walk."""
    epoch = np.asarray(bins, dtype=np.int64) * grid_s + grid_s // 2
    return pd.DataFrame({
        "individual_id": individual,
        "timestamp": pd.to_datetime(epoch, unit="s"),
        "lat": np.full(len(bins), 50.0),
        "lon": 10.0 + 0.001 * np.arange(len(bins)),
    })


# runs of 20, 5 and 14 bins, a hole longer than MAX_GAP_BINS, a run of 30, then an isolated run of 5 that
# min_segment_bins drops
BINS = list(range(0, 20)) + list(range(22, 27)) + list(range(29, 43)) + list(range(60, 90)) + list(range(120, 125))


def test_closed_form_matches_iter_windows():
    frame = fixes_at(BINS, 6 * HOUR).assign(bin_index=BINS)
    lengths = cs.run_lengths(np.asarray(BINS))
    assert lengths.tolist() == [20, 5, 14, 30, 5]
    for c in (1, 4, 12, 24):
        assert cs.windows(lengths, c) == len(iter_windows(frame, context_length=c))


def test_run_position_restarts_at_each_break():
    pos = cs.run_position(np.array([3, 4, 5, 7, 8, 20]))
    assert pos.tolist() == [0, 1, 2, 0, 1, 0]


def test_gap_rule_does_not_change_windows():
    for grid_h in (1, 6):
        grid_s = grid_h * HOUR
        fixes = fixes_at(BINS, grid_s)
        segments, _ = segments_from_fixes(fixes, bin_seconds=grid_s, min_segment_bins=MIN_SEGMENT_BINS)
        # two segments: the short holes stay inside the first, the long hole splits off the 30-run, the last run is dropped
        assert segments["segment_id"].nunique() == 2
        assert len(segments) == len(BINS) - 5
        bins, *_ = _to_bins(fixes["timestamp"].to_numpy(), fixes["lat"].to_numpy(), fixes["lon"].to_numpy(),
                            bin_seconds=grid_s)
        lengths = cs.run_lengths(bins)
        from_segments = cs.segment_windows(segments, (12, 24))
        for c in (12, 24):
            assert from_segments.loc["a", c] == cs.windows(lengths, c)


def test_usable_hours_counts_only_runs_that_hold_a_window():
    assert cs.usable_hours(np.array([20, 5, 14, 30]), 12, 6) == (20 + 14 + 30) * 6


def test_on_the_hour_jitter_is_a_phase_hole_and_centring_removes_it():
    # an hourly tag fixing on the hour, a few seconds either side
    epoch = np.array([10 * HOUR + 5, 10 * HOUR + 3598, 12 * HOUR + 3, 13 * HOUR + 2, 15 * HOUR + 10], dtype=float)
    n_breaks, n_holes = cs.phase_holes(epoch, np.floor(epoch / HOUR), HOUR)
    assert (n_breaks, n_holes) == (2, 1)  # 10:59:58 -> 12:00:03 is a hole; 13:00:02 -> 15:00:10 is a real miss
    shifted = np.floor((epoch + HOUR // 2) / HOUR)
    assert cs.phase_holes(epoch, shifted, HOUR) == (1, 0)


def test_night_hours_wrap_midnight():
    hours = np.array([19.9, 20.0, 23.5, 0.5, 3.99, 4.0, 12.0])
    assert cs.in_hours(hours, cs.NIGHT_H).tolist() == [False, True, True, True, True, False, False]
    assert cs.in_hours(hours, cs.DAY_H).tolist() == [False, False, False, False, False, False, True]


def test_schedule_change_needs_two_qualifying_months():
    months = np.repeat([0, 1, 2], [60, 60, 10])
    steady = np.full(130, 3600.0)
    switched = np.concatenate((np.full(60, 900.0), np.full(70, 3600.0)))
    assert cs.schedule_changed(months, steady) == 0.0
    assert cs.schedule_changed(months, switched) == 1.0
    assert np.isnan(cs.schedule_changed(np.zeros(130, dtype=int), steady))
