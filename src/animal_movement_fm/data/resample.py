"""
Resample irregular GPS fixes onto a regular 6h grid. 

One trajectory in -> one or more regular cadence segments out. Each raw fix is assigned to a 
6h bin; the fix nearest the bin centre represents that bin (so it is a real observation, never
an interpolated point). The trajectory is split into seperate segments wherever consecutive 
occupied bins lie more than `MAX_GAP_BINS apart`, so no segment spans a multi-day transmission 
hole. 

Bins may be missing within a segment (gaps up to `MAX_GAP_BINS` are tolerated); each bins carries
its integer bin_index so downstream windowing can decide whether to cross a small internal gap
rather than assuming contiguity. 

Note: we will improve upon this method (as in go towards a cadence invariant transformer for 
example, but for now we want to have a working data pipeline). 
"""

from dataclasses import dataclass

import numpy as np 

BIN_SECONDS = 6 * 60 * 60 # 6h analysis cadence
MAX_GAP_BINS = 8 # split when a hole exceeds 2 days (8*6h)

# The context length in force. A window is CONTEXT_LENGTH history bins plus one target, so it
# spans CONTEXT_LENGTH + 1 consecutive bins. Fixed at 12 by preregistration amendment A6.
CONTEXT_LENGTH = 12

def min_segment_bins(context_length: int = CONTEXT_LENGTH) -> int:
    """
    Return the minimum segment length needed to produce one window: `context_length + 1; derived from the current context length rather 
    than hard coded so the segment floor cannot go stale when the context changes.
    """
    if context_length < 1:
        raise ValueError("context_length must be at least 1")
    return context_length + 1


# backwards compatible module constant which is now derived rather than asserted
MIN_SEGMENT_BINS = min_segment_bins()

@dataclass(frozen=True)
class Segment:
    """
    One contiguous enough run of 6h buns from a single trajectory. 

    All arrays are aligned and ordered by `bin_index`, which is the absolute 6h-in bin number 
    (epoch // 6h): a step of 1 is a normal 6h move, a step >1 (up to `MAX_GAP_BINS`) is a tolerated
    internal gap. 
    """

    bin_index: np.ndarray # int64, and shape (n,)
    timestamp: np.ndarray # datetime64[us], representative fix's real time
    lat: np.ndarray # float 64, and shape (n,)
    lon: np.ndarray # float64, and shape (n,)
    centre_offset_s: np.ndarray # float64, |fix_time - bin_centre|, seconds 

    def __len__(self) -> int:
        return len(self.bin_index)

# GenAI reference: my attempt at this function proved inefficient since it required a pass over all observations FINISH LATER 
# Modified slightly to add cadence as a parameter 
def _to_bins(
    timestamp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    bin_seconds: int = BIN_SECONDS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce raw fixes to one representative per occupied bin (nearest-to-centre).

    `bin_seconds` defaults to the module cadence so existing callers and tests are
    unaffected; the sweep passes other cadences.
    """
    if len(timestamp) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, timestamp[:0], empty_f, empty_f, empty_f

    epoch_s = timestamp.astype("datetime64[us]").astype(np.int64) / 1_000_000.0
    bin_index = np.floor(epoch_s / bin_seconds).astype(np.int64)
    centre_s = bin_index * bin_seconds + bin_seconds / 2.0
    offset_s = np.abs(epoch_s - centre_s)

    order = np.lexsort((offset_s, bin_index))
    bin_sorted = bin_index[order]
    first_in_bin = np.empty(len(bin_sorted), dtype=bool)
    first_in_bin[0] = True
    first_in_bin[1:] = bin_sorted[1:] != bin_sorted[:-1]
    keep = order[first_in_bin]

    return (
        bin_index[keep],
        timestamp[keep],
        lat[keep],
        lon[keep],
        offset_s[keep],
    )
########## GenAI contribution ends here. 
# same modification logic for the above as well
def resample_trajectory(
    timestamp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    bin_seconds: int = BIN_SECONDS,
    max_gap_bins: int = MAX_GAP_BINS,
    min_segment_bins: int = MIN_SEGMENT_BINS,
) -> list[Segment]:
    """Resample one trajectory to `bin_seconds` bins and split into clean segments.

    Keyword cadence params default to the module constants (6h build), so the existing pipeline and tests are unchanged; 
    the cadence sweep overrides them.
    """
    bin_index, ts, lat_b, lon_b, offset = _to_bins(
        timestamp, lat, lon, bin_seconds=bin_seconds
    )
    if len(bin_index) == 0:
        return []

    breaks = np.flatnonzero(np.diff(bin_index) > max_gap_bins) + 1

    segments = []
    for s_bin, s_ts, s_lat, s_lon, s_off in zip(
        np.split(bin_index, breaks),
        np.split(ts, breaks),
        np.split(lat_b, breaks),
        np.split(lon_b, breaks),
        np.split(offset, breaks),
    ):
        if len(s_bin) >= min_segment_bins:
            segments.append(
                Segment(
                    bin_index=s_bin,
                    timestamp=s_ts,
                    lat=s_lat,
                    lon=s_lon,
                    centre_offset_s=s_off,
                )
            )
    return segments
    
