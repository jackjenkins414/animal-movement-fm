# GenAI reference: After suffering a majorly painful bug, I have added test cases for split.py. Again, to save time,
# as we have done throughout this project, I described to GenAI what I wanted down from a logic point of view, and
# importantly explained the bug, GenAI wrote these tests which I validated by reviewing each assertion against the
# intended logic, running the full test file, and confirming that the row order regression test fails with the
# original implementation but passes after the sorting fix. It does.

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.split import (
    Split,
    apply_window_eligibility,
    canonical_manifest_ids,
    id_key,
    load_split,
    make_split,
    save_split,
    split_eligibility_table,
    window_counts_per_individual,
)


def _counts(n=200, seed=0, tie_every=3):
    """Per-individual window counts with deliberate ties in n_windows (the condition that broke reproducibility)."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(10_000, 99_999), size=n, replace=False)
    # Heavy ties: many individuals share a window count, so row order among them is ambiguous.
    n_windows = 100 + (np.arange(n) // tie_every) * 10
    return pd.DataFrame({"individual_id": ids, "n_windows": n_windows})


class TestReproducibility:
    def test_split_is_invariant_to_input_row_order(self):
        # THE regression test: shuffling the rows of `counts` must not change the assignment.
        counts = _counts()
        base = make_split(counts, strata=None, min_windows=100, seed=42)
        for rs in (1, 7, 123):
            shuffled = counts.sample(frac=1.0, random_state=rs)
            alt = make_split(shuffled, strata=None, min_windows=100, seed=42)
            for a, b in ((base.train, alt.train), (base.val, alt.val), (base.test, alt.test)):
                assert set(a) == set(b)

    def test_same_seed_same_split(self):
        counts = _counts()
        a = make_split(counts, strata=None, min_windows=100, seed=42)
        b = make_split(counts, strata=None, min_windows=100, seed=42)
        np.testing.assert_array_equal(np.sort(a.test), np.sort(b.test))

    def test_different_seed_different_split(self):
        counts = _counts()
        a = make_split(counts, strata=None, min_windows=100, seed=42)
        b = make_split(counts, strata=None, min_windows=100, seed=7)
        assert set(a.test) != set(b.test)


class TestAssignment:
    def test_splits_are_disjoint(self):
        s = make_split(_counts(), strata=None, min_windows=100, seed=42)
        assert set(s.train).isdisjoint(s.val)
        assert set(s.train).isdisjoint(s.test)
        assert set(s.val).isdisjoint(s.test)

    def test_all_eligible_individuals_assigned(self):
        counts = _counts()
        s = make_split(counts, strata=None, min_windows=100, seed=42)
        assigned = set(s.train) | set(s.val) | set(s.test)
        assert assigned == set(counts["individual_id"])

    def test_min_windows_filter_drops_short_tracks(self):
        counts = _counts()
        counts.loc[counts.index[:5], "n_windows"] = 10  # below threshold
        dropped = set(counts["individual_id"].iloc[:5])
        s = make_split(counts, strata=None, min_windows=100, seed=42)
        assigned = set(s.train) | set(s.val) | set(s.test)
        assert assigned.isdisjoint(dropped)

    def test_fractions_are_respected(self):
        counts = _counts(n=200)
        s = make_split(counts, strata=None, min_windows=100, seed=42)
        assert len(s.test) == round(200 * 0.125)
        assert len(s.val) == round(200 * 0.125)
        assert len(s.train) == 200 - len(s.test) - len(s.val)

    def test_strata_are_split_independently(self):
        counts = _counts(n=200)
        strata = pd.DataFrame({
            "individual_id": counts["individual_id"],
            "stratum": ["a"] * 100 + ["b"] * 100,
        })
        s = make_split(counts, strata=strata, min_windows=100, seed=42)
        by_id = dict(zip(strata["individual_id"], strata["stratum"], strict=True))
        # Each stratum contributes its own share of the test split, so both are represented.
        test_strata = {by_id[i] for i in s.test}
        assert test_strata == {"a", "b"}


class TestFreeze:
    def test_save_load_roundtrip(self, tmp_path):
        s = make_split(_counts(), strata=None, min_windows=100, seed=42)
        path = tmp_path / "split.json"
        save_split(s, path, note="test")
        back = load_split(path)
        for a, b in ((s.train, back.train), (s.val, back.val), (s.test, back.test)):
            np.testing.assert_array_equal(np.sort(a), np.sort(b))

    def test_manifest_ids_are_sorted_for_clean_diffs(self, tmp_path):
        import json
        s = make_split(_counts(), strata=None, min_windows=100, seed=42)
        path = tmp_path / "split.json"
        save_split(s, path)
        payload = json.loads(path.read_text())
        for key in ("train", "val", "test"):
            assert payload[key] == sorted(payload[key])

    def test_overlapping_manifest_raises(self, tmp_path):
        import json
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"train": [1, 2, 3], "val": [3, 4], "test": [5]}))
        with pytest.raises(ValueError):
            load_split(path)


class TestContiguousWindowEligibility:
    def test_window_counts_split_at_internal_gap(self, tmp_path):
        # Two three-fix runs, context=2: one target from each run. A naive
        # six-row slide would incorrectly report four windows.
        frame = pd.DataFrame(
            {
                "segment_id": "s1",
                "individual_id": 7,
                "bin_index": [0, 1, 2, 4, 5, 6],
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="6h"),
                "lat": 0.0,
                "lon": np.arange(6, dtype=float),
            }
        )
        path = tmp_path / "segments.parquet"
        frame.to_parquet(path)

        counts = window_counts_per_individual(path, context_length=2)

        assert counts.to_dict("records") == [
            {"individual_id": 7, "n_windows": 2}
        ]

    def test_training_and_evaluation_use_different_floors(self):
        split = Split(
            train=np.array([1, 2, 3]),
            val=np.array([4, 5]),
            test=np.array([6, 7]),
        )
        counts = pd.DataFrame(
            {
                "individual_id": [1, 2, 3, 4, 5, 6, 7],
                "n_windows": [0, 1, 99, 99, 100, 100, 101],
            }
        )

        eligible = apply_window_eligibility(split, counts)

        assert eligible.train.tolist() == [2, 3]
        assert eligible.val.tolist() == [5]
        assert eligible.test.tolist() == [6, 7]

    def test_missing_count_is_zero_and_no_id_is_reassigned(self):
        split = Split(
            train=np.array([10, 11]),
            val=np.array([20]),
            test=np.array([30]),
        )
        counts = pd.DataFrame(
            {"individual_id": [11, 20, 30], "n_windows": [1, 100, 99]}
        )

        table = split_eligibility_table(split, counts)
        eligible = apply_window_eligibility(split, counts)

        missing = table.loc[table["individual_id"] == 10].iloc[0]
        assert missing["n_windows"] == 0
        assert not missing["eligible"]
        assert eligible.train.tolist() == [11]
        assert eligible.val.tolist() == [20]
        assert eligible.test.tolist() == []
        assert set(eligible.train) <= set(split.train)
        assert set(eligible.val) <= set(split.val)
        assert set(eligible.test) <= set(split.test)

    def test_eligibility_does_not_mutate_frozen_assignment(self):
        split = Split(
            train=np.array([1, 2]),
            val=np.array([3]),
            test=np.array([4]),
        )
        before = tuple(a.copy() for a in (split.train, split.val, split.test))
        counts = pd.DataFrame(
            {"individual_id": [1, 2, 3, 4], "n_windows": [0, 1, 99, 100]}
        )

        _ = apply_window_eligibility(split, counts)

        for original, current in zip(
            before, (split.train, split.val, split.test), strict=True
        ):
            np.testing.assert_array_equal(original, current)


class TestIdsAcrossSources:
    """Individual ids are not one type across the corpus.

    Movebank ids are integers arriving as int64, as float64 after a parquet round trip, or as text.
    RAATD ids are strings such as "CAR2013_sel_a_m_03". This module used to cast every id with
    `int()`, which raised on every RAATD id, i.e. on the whole marine leg.
    """

    def test_frozen_stork_manifest_round_trips_byte_identically(self):
        """THE acceptance criterion for making this module string-safe.

        Every Tier 1 GPU result was computed against this manifest, so if writing it back produces
        even a different byte, the change is not safe to ship.
        """
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        original = Path(__file__).resolve().parents[1] / "splits" / "stork_split_seed42.json"
        before = original.read_bytes()

        split = load_split(original)
        assert (len(split.train), len(split.val), len(split.test)) == (630, 105, 105)
        assert split.train.dtype == np.int64

        target = Path(tempfile.mkdtemp()) / "roundtrip.json"
        save_split(split, target, note=json.loads(before)["note"])

        assert hashlib.md5(target.read_bytes()).hexdigest() == hashlib.md5(before).hexdigest()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(12345, "12345"), (12345.0, "12345"), ("12345", "12345"), (" 12345 ", "12345"),
         (np.int64(12345), "12345"), ("CAR2013_sel_a_m_03", "CAR2013_sel_a_m_03")],
    )
    def test_id_key_collapses_spellings_of_the_same_id(self, value, expected):
        assert id_key(value) == expected

    def test_id_key_keeps_distinct_ids_distinct(self):
        """Truncating non-integral values would silently merge two different animals."""
        assert len({id_key(v) for v in (1.5, 1.7, 2)}) == 3

    def test_integer_ids_stay_json_numbers_and_strings_stay_strings(self):
        """Integer manifests must keep writing numbers, or the frozen manifest changes."""
        assert canonical_manifest_ids([3, 1, 2]) == [1, 2, 3]
        assert canonical_manifest_ids([3.0, 1.0, 2.0]) == [1, 2, 3]
        assert canonical_manifest_ids(["b", "a"]) == ["a", "b"]

    def test_string_ids_survive_save_and_load(self, tmp_path):
        split = Split(
            train=np.array(["CAR2013_sel_a_m_01", "CAR2013_sel_a_m_02"], dtype=object),
            val=np.array(["CAR2013_sel_a_m_03"], dtype=object),
            test=np.array(["CAR2013_sel_a_m_04"], dtype=object),
        )
        path = tmp_path / "raatd.json"
        save_split(split, path, note="RAATD-style string ids")

        loaded = load_split(path)

        assert sorted(loaded.train) == sorted(split.train)
        assert loaded.train.dtype == object

    def test_make_split_accepts_string_ids(self):
        ids = [f"CAR2013_sel_a_m_{i:02d}" for i in range(1, 25)]
        counts = pd.DataFrame({"individual_id": ids, "n_windows": [200] * 24})

        split = make_split(counts, min_windows=1, seed=42)

        assert len(split.train) + len(split.val) + len(split.test) == 24
        assert set(split.train) | set(split.val) | set(split.test) == set(ids)

    def test_eligibility_matches_ids_across_dtype_mismatch(self):
        """A manifest holding ints and a counts table holding floats or text refer to the same
        animals; a silent mismatch would report every individual as having zero windows."""
        split = Split(train=np.array([1, 2, 3]), val=np.array([4]), test=np.array([5]))

        for counts in (
            pd.DataFrame({"individual_id": [1.0, 2.0, 3.0, 4.0, 5.0], "n_windows": [200] * 5}),
            pd.DataFrame({"individual_id": ["1", "2", "3", "4", "5"], "n_windows": [200] * 5}),
        ):
            table = split_eligibility_table(split, counts)
            assert int(table["eligible"].sum()) == 5, f"failed for {counts['individual_id'].dtype}"

    def test_eligibility_preserves_string_dtype_when_nothing_survives(self):
        split = Split(
            train=np.array(["a", "b"], dtype=object),
            val=np.array(["c"], dtype=object),
            test=np.array(["d"], dtype=object),
        )
        counts = pd.DataFrame({"individual_id": ["a", "b", "c", "d"], "n_windows": [0, 0, 0, 0]})

        eligible = apply_window_eligibility(split, counts)

        assert len(eligible.val) == 0
        assert eligible.val.dtype == object
