# GenAI reference: tests for build_1h_segments.py; simple. 
import numpy as np

import experiments.data.build_1h_segments as b1
import experiments.eda.cadence_horizons as ch

# a 14-bin run, then 18, 19 and 36 occupied: anchors 11, 12, 13
BINS = np.concatenate((np.arange(14), [18, 19, 36]))
LON = 10.0 + 0.001 * np.arange(len(BINS))
LAT = np.full(len(BINS), 50.0)


def test_target_index_marks_empty_bins():
    assert b1.target_index(BINS, np.array([11, 12, 13]), 6).tolist() == [-1, 14, 15]


def test_pairs_are_every_occupied_pair_h_bins_apart():
    assert len(b1.pair_displacements(BINS, LON, LAT, 1)) == 13 + 1  # 0..12 -> 1..13, and 18 -> 19
    assert len(b1.pair_displacements(BINS, LON, LAT, 6)) == 8 + 2  # 0..7 -> 6..13, then 12 -> 18, 13 -> 19
    assert len(b1.pair_displacements(BINS, LON, LAT, 24)) == 1  # 12 -> 36


def test_scaled_targets_cover_the_direct_rule_windows():
    hit = ch.targets_hit(BINS, 12, ch.HORIZONS)
    for j, h in enumerate(ch.HORIZONS):
        r = b1.scaled_targets(BINS, LON, LAT, h)
        assert len(r) == hit[:, j].sum()
        assert np.all(r > 0)


def test_boundary_floor_binds_only_below_the_scale_floor():
    assert b1.floored(7.15) == {"a4_m": 7.15, "boundary_m": 30.0, "floor_binds": True}
    assert b1.floored(83.23) == {"a4_m": 83.23, "boundary_m": 83.23, "floor_binds": False}


def test_a_species_run_writes_under_subset_never_over_the_full_build():
    assert b1.output_dirs(False) == (b1.OUT_SEGMENTS, b1.OUT_RESULTS)
    segments, results = b1.output_dirs(True)
    assert (segments.parent, results.parent) == (b1.OUT_SEGMENTS, b1.OUT_RESULTS)
    assert segments.name == results.name == "subset"
