# GenAI reference: tests for history_occupancy.py
import numpy as np

import experiments.eda.history_occupancy as ho

# a 14-bin run 10..23 (anchors 21, 22, 23 by bin), with history bins 0, 1 and 5 observed before a hole
BINS = np.concatenate(([0, 1, 5], np.arange(10, 24)))
LON = 10.0 + 0.001 * BINS
LAT = np.full(len(BINS), 50.0)


def test_grid_and_occupancy_at_c24():
    anchors = np.flatnonzero(BINS == 23)
    grid = ho.history_grid(BINS, anchors, 24)
    assert (grid[0] >= 0).sum() == 3 + 14 - 0  # 0, 1, 5 and 10..23 all sit inside t - 23 ... t
    # extended bins 0..11: observed 0, 1, 5, 10, 11 -> 5 of 12
    assert ho.occupancy(grid)[0] == 5 / 12
    assert ho.stratum(np.array([0.0, 0.2, 0.5, 0.89, 0.9, 1.0])).tolist() == [0, 1, 2, 2, 3, 3]


def test_extended_steps_bridge_gaps_from_the_previous_observed_fix():
    anchors = np.flatnonzero(BINS == 23)
    grid = ho.history_grid(BINS, anchors, 24)
    scale = ho.core_scale(grid, LON, LAT)
    mag, span = ho.extended_steps(grid, LON, LAT, scale)
    # tokens end at bins 1..12: observed 1 (from 0), 5 (from 1, gap), 10 (from 5, gap), 11, 12
    assert len(mag) == 5
    assert span.tolist() == [1, 4, 5, 1, 1]
    one_step = mag[0]
    assert np.allclose(mag[[1, 2]], [4 * one_step, 5 * one_step], rtol=1e-3)


def test_per_bin_gap_tokens_match_one_step_on_a_steady_walk():
    anchors = np.flatnonzero(BINS == 23)
    grid = ho.history_grid(BINS, anchors, 24)
    mag, span = ho.extended_steps(grid, LON, LAT, ho.core_scale(grid, LON, LAT))
    assert np.allclose(mag / span, mag[0], rtol=1e-3)  # constant speed: every per-bin step equals a one-bin step


def test_ablation_strata_fold_below_0_9_into_gappy():
    assert ho.ablation_strata(np.array([2, 3, 5, 7])) == {"gappy": 10, "full": 7}


def test_core_scale_is_the_main_studys():
    anchors = np.flatnonzero(BINS == 23)
    for c in (24, 96):
        grid = ho.history_grid(BINS, anchors, c)
        assert np.array_equal(grid[:, -12:], ho.history_grid(BINS, anchors, 12))
