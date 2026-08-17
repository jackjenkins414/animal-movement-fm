# GenAI reference: Tests generated with GenAI; I asked for it to test RAATD string ids load, int ids from Movebank still work
# float or strng manifest still matches an integer keyed segments file through cannonical key. All of these have been checked. 

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.benchmark import load_split_frame


def _write_segments(tmp_path, individual_ids, name="segments.parquet"):
    """Two occupied bins per individual, which is all the loader needs to be exercised."""
    rows = []
    for individual in individual_ids:
        for bin_index in range(2):
            rows.append(
                {
                    "segment_id": f"{individual}_s0",
                    "individual_id": individual,
                    "bin_index": bin_index,
                    "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                    + pd.Timedelta(hours=6 * bin_index),
                    "lon": 10.0 + bin_index,
                    "lat": -40.0 + bin_index,
                }
            )
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_string_ids_load(tmp_path):
    """The RAATD case. This raised on the int64 cast before the fix."""
    ids = ["CAR2013_sel_a_m_03", "CAR2013_sel_a_m_04", "WESE2011_x_01"]
    path = _write_segments(tmp_path, ids)

    frame = load_split_frame(path, np.array(ids[:2], dtype=object))

    assert set(frame["individual_id"]) == set(ids[:2])
    assert len(frame) == 4


def test_integer_ids_load(tmp_path):
    """The Movebank case, which must keep working unchanged."""
    ids = [101, 102, 103]
    path = _write_segments(tmp_path, ids)

    frame = load_split_frame(path, np.array([101, 103]))

    assert set(frame["individual_id"]) == {101, 103}


def test_float_manifest_matches_integer_file(tmp_path):
    """A parquet round trip can turn an int id into 101.0; the canonical key absorbs that."""
    path = _write_segments(tmp_path, [101, 102])

    frame = load_split_frame(path, np.array([101.0]))

    assert set(frame["individual_id"]) == {101}


def test_string_manifest_matches_integer_file(tmp_path):
    """Manifests written as JSON strings must still reach an integer-keyed segments file."""
    path = _write_segments(tmp_path, [101, 102])

    frame = load_split_frame(path, np.array(["101"], dtype=object))

    assert set(frame["individual_id"]) == {101}


def test_no_overlap_raises_rather_than_returning_empty(tmp_path):
    """The guard: a manifest/segments mismatch must not evaluate to zero windows in silence."""
    path = _write_segments(tmp_path, ["CAR2013_sel_a_m_03"])

    with pytest.raises(ValueError, match="manifest/segments mismatch"):
        load_split_frame(path, np.array(["MIR2019_not_here"], dtype=object))


def test_empty_request_returns_empty_frame(tmp_path):
    """An empty request is a caller's business, not a mismatch, so it must not raise."""
    path = _write_segments(tmp_path, ["CAR2013_sel_a_m_03"])

    frame = load_split_frame(path, np.array([], dtype=object))

    assert frame.empty


def test_absent_individuals_are_dropped_not_fatal(tmp_path):
    """The continuity filter legitimately removes individuals; a partial match is normal."""
    path = _write_segments(tmp_path, ["a", "b"])

    frame = load_split_frame(path, np.array(["a", "missing"], dtype=object))

    assert set(frame["individual_id"]) == {"a"}
