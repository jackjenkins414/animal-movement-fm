#GenAI reference: These tests are GenAI generated where I gave it the logic I wanted to test which 
# was it assigns it to the correct bin, keeps the fix closest to the bin centre, handles unsorted 
# or empty input, and gets edge cases like exactly on bin boundary right. I validated the correctness
# of them all. 
"""
Tests for the 6h resampler: binning, nearest to centre selection, splitting.

Inputs are hand-constructed with computable answers — no random data — so each
assertion pins exact behaviour at the boundaries where vectorised binning hides
off-by-one bugs (bin edges, gap == threshold, segment length == floor).
"""

import numpy as np

from animal_movement_fm.data.resample import (
    BIN_SECONDS,
    MAX_GAP_BINS,
    MIN_SEGMENT_BINS,
    Segment,
    _to_bins,
    resample_trajectory,
)


def _ts(seconds: float) -> np.datetime64:
    """A datetime64[us] at `seconds` past the epoch — readable bin-relative timestamps."""
    return np.datetime64(int(round(seconds * 1_000_000)), "us")


def _centred_fix(bin_idx: int) -> float:
    """Seconds-past-epoch of a fix sitting exactly at the centre of `bin_idx`."""
    return bin_idx * BIN_SECONDS + BIN_SECONDS / 2.0


def _track(bin_indices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one centred fix per given bin index. lat/lon encode the bin for tracing."""
    timestamp = np.array([_ts(_centred_fix(b)) for b in bin_indices])
    lat = np.array([float(b) for b in bin_indices])
    lon = np.array([float(b) + 100.0 for b in bin_indices])
    return timestamp, lat, lon


# --------------------------------------------------------------------------- #
# _to_bins: selection and bin assignment
# --------------------------------------------------------------------------- #

def test_nearest_fix_to_centre_wins():
    """Within one bin, the fix closest to the bin centre is kept."""
    centre = BIN_SECONDS / 2.0
    timestamp = np.array([_ts(10.0), _ts(centre + 5.0), _ts(BIN_SECONDS - 10.0)])
    lat = np.array([1.0, 2.0, 3.0])
    lon = np.array([10.0, 20.0, 30.0])

    bin_index, ts, lat_k, lon_k, offset = _to_bins(timestamp, lat, lon)

    assert len(bin_index) == 1
    assert bin_index[0] == 0
    assert lat_k[0] == 2.0          # the centre-most fix
    assert lon_k[0] == 20.0
    assert offset[0] == 5.0


def test_unsorted_input_is_handled():
    """Input need not be time-ordered; output is ordered by bin_index."""
    # Three centred fixes in bins 2, 0, 1 — deliberately out of order.
    timestamp = np.array([_ts(_centred_fix(2)), _ts(_centred_fix(0)), _ts(_centred_fix(1))])
    lat = np.array([2.0, 0.0, 1.0])
    lon = np.array([102.0, 100.0, 101.0])

    bin_index, ts, lat_k, lon_k, offset = _to_bins(timestamp, lat, lon)

    assert list(bin_index) == [0, 1, 2]
    assert list(lat_k) == [0.0, 1.0, 2.0]   # reordered to match bins


def test_fix_on_bin_boundary_floors_into_next_bin():
    """A fix exactly on a bin edge belongs to the later bin (floor convention)."""
    # Exactly BIN_SECONDS past epoch is the first instant of bin 1, not the last of bin 0.
    timestamp = np.array([_ts(BIN_SECONDS)])
    lat = np.array([5.0])
    lon = np.array([6.0])

    bin_index, ts, lat_k, lon_k, offset = _to_bins(timestamp, lat, lon)

    assert bin_index[0] == 1
    assert offset[0] == BIN_SECONDS / 2.0   # start of a bin is half a bin from its centre


def test_centred_fix_has_zero_offset():
    """A fix at the exact bin centre has zero centre offset."""
    timestamp = np.array([_ts(_centred_fix(7))])
    bin_index, ts, lat_k, lon_k, offset = _to_bins(
        timestamp, np.array([0.0]), np.array([0.0])
    )
    assert bin_index[0] == 7
    assert offset[0] == 0.0


def test_to_bins_empty_input():
    """Empty input returns five empty arrays, not an error."""
    empty_ts = np.array([], dtype="datetime64[us]")
    empty = np.array([], dtype=np.float64)
    bin_index, ts, lat_k, lon_k, offset = _to_bins(empty_ts, empty, empty)
    assert len(bin_index) == 0
    assert len(ts) == 0
    assert len(offset) == 0


# --------------------------------------------------------------------------- #
# resample_trajectory: splitting, length floor, segment integrity
# --------------------------------------------------------------------------- #

def test_hole_free_track_returns_one_segment():
    """A contiguous track over the length floor returns exactly one segment."""
    timestamp, lat, lon = _track(range(MIN_SEGMENT_BINS))
    segments = resample_trajectory(timestamp, lat, lon)

    assert len(segments) == 1
    assert len(segments[0]) == MIN_SEGMENT_BINS
    assert int((np.diff(segments[0].bin_index) > 1).sum()) == 0   # fully contiguous


def test_gap_equal_to_threshold_does_not_split():
    """A hole of exactly MAX_GAP_BINS is tolerated — one segment, with an internal gap."""
    # Two runs of 120 bins, separated by a gap of exactly MAX_GAP_BINS.
    first = list(range(MIN_SEGMENT_BINS))
    start2 = first[-1] + MAX_GAP_BINS          # successor is MAX_GAP_BINS away
    second = list(range(start2, start2 + MIN_SEGMENT_BINS))
    timestamp, lat, lon = _track(first + second)

    segments = resample_trajectory(timestamp, lat, lon)

    assert len(segments) == 1                  # not split
    assert int((np.diff(segments[0].bin_index) > 1).sum()) == 1   # one tolerated hole


def test_gap_over_threshold_splits():
    """A hole larger than MAX_GAP_BINS splits the trajectory into two segments."""
    first = list(range(MIN_SEGMENT_BINS))
    start2 = first[-1] + MAX_GAP_BINS + 1       # one past the threshold
    second = list(range(start2, start2 + MIN_SEGMENT_BINS))
    timestamp, lat, lon = _track(first + second)

    segments = resample_trajectory(timestamp, lat, lon)

    assert len(segments) == 2
    assert len(segments[0]) == MIN_SEGMENT_BINS
    assert len(segments[1]) == MIN_SEGMENT_BINS
    # Each segment is internally contiguous — the hole fell between them.
    assert int((np.diff(segments[0].bin_index) > 1).sum()) == 0
    assert int((np.diff(segments[1].bin_index) > 1).sum()) == 0


def test_short_segment_dropped_below_floor():
    """A segment with fewer than MIN_SEGMENT_BINS bins is dropped."""
    timestamp, lat, lon = _track(range(MIN_SEGMENT_BINS - 1))   # 119 bins
    assert resample_trajectory(timestamp, lat, lon) == []


def test_segment_exactly_at_floor_is_kept():
    """A segment of exactly MIN_SEGMENT_BINS bins survives (boundary is inclusive)."""
    timestamp, lat, lon = _track(range(MIN_SEGMENT_BINS))       # 120 bins
    segments = resample_trajectory(timestamp, lat, lon)
    assert len(segments) == 1
    assert len(segments[0]) == MIN_SEGMENT_BINS


def test_split_keeps_long_drops_short():
    """When a split yields one long and one short piece, only the long one survives."""
    long_run = list(range(MIN_SEGMENT_BINS))                    # 120 bins -> kept
    start2 = long_run[-1] + MAX_GAP_BINS + 1
    short_run = list(range(start2, start2 + 10))                # 10 bins -> dropped
    timestamp, lat, lon = _track(long_run + short_run)

    segments = resample_trajectory(timestamp, lat, lon)

    assert len(segments) == 1
    assert len(segments[0]) == MIN_SEGMENT_BINS


def test_empty_trajectory_returns_no_segments():
    """Empty input yields an empty list, not an error."""
    empty_ts = np.array([], dtype="datetime64[us]")
    empty = np.array([], dtype=np.float64)
    assert resample_trajectory(empty_ts, empty, empty) == []


def test_segment_arrays_are_aligned_and_typed():
    """All Segment arrays share length and carry the documented dtypes."""
    timestamp, lat, lon = _track(range(MIN_SEGMENT_BINS))
    seg = resample_trajectory(timestamp, lat, lon)[0]

    n = len(seg)
    assert seg.bin_index.shape == (n,)
    assert seg.timestamp.shape == (n,)
    assert seg.lat.shape == (n,)
    assert seg.lon.shape == (n,)
    assert seg.centre_offset_s.shape == (n,)
    assert seg.bin_index.dtype == np.int64
    assert isinstance(seg, Segment)