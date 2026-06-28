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
MIN_SEGMENT_BINS = 120 # drop segments shorter than >=120 point floor

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
def _to_bins(
    timestamp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce raw fixes to one representative per occupied 6h bin.

    For each bin, keeps the fix nearest the bin centre. Input need not be sorted; output is ordered by 
    `bin_index`, one entry per occupied bin. Returns arrays in `Segment` field order: (bin_index, timestamp,
    lat, lon, centre_offset_s).
    """
    if len(timestamp) == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, timestamp[:0], empty_f, empty_f, empty_f

    epoch_s = timestamp.astype("datetime64[us]").astype(np.int64) / 1_000_000.0
    bin_index = np.floor(epoch_s / BIN_SECONDS).astype(np.int64)
    centre_s = bin_index * BIN_SECONDS + BIN_SECONDS / 2.0
    offset_s = np.abs(epoch_s - centre_s)

    # Sort by bin, then by closeness to centre --- first row per bin is the winner.
    order = np.lexsort((offset_s, bin_index))                   # primary key = bin_index
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

def resample_trajectory(
    timestamp: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> list[Segment]:
    """Resample one trajectory to 6h bins and split it into clean segments.

    Bins each raw fix (nearest-to-centre representative), then splits wherever a hole exceeds `MAX_GAP_BINS`.
    Segments shorter than `MIN_SEGMENT_BINS` are dropped. Returns segments in time order; may be empty if 
    nothing survives.
    """
    bin_index, ts, lat_b, lon_b, offset = _to_bins(timestamp, lat, lon)
    if len(bin_index) == 0:
        return []

    # cut after every bin whose successor is more than MAX_GAP_BINS away
    breaks = np.flatnonzero(np.diff(bin_index) > MAX_GAP_BINS) + 1

    segments = []
    for s_bin, s_ts, s_lat, s_lon, s_off in zip(
        np.split(bin_index, breaks),
        np.split(ts, breaks),
        np.split(lat_b, breaks),
        np.split(lon_b, breaks),
        np.split(offset, breaks),
    ):
        if len(s_bin) >= MIN_SEGMENT_BINS:
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
    
