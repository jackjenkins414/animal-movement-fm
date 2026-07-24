# GenAI reference: I specified the analytically known test segment (a chosen step sequence whose per-fix states,
# window count, can_launch flags and displacement/covariate values are all known by hand) and what each case must
# assert; GenAI helped write these tests for this. Verified by construction as well. 
"""
Tests for the state-classifier windowed dataset (`state.dataset`).

The fixture segment has a hand-chosen incoming-step sequence, so every fix's refined state, every window's target
label, the can_launch flag, and the exact displacement / time / season channels are known analytically and checked
directly. A second short segment checks the min-length skip and that windows never cross a segment boundary.
"""

import numpy as np
import pandas as pd

from animal_movement_fm.analysis.regime import label_migratory_launches, label_regimes
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos
from animal_movement_fm.state.dataset import (
    STATE_TO_IDX,
    build_state_windows,
)

# KM per degree of latitude (pure-latitude steps at lon=0 convert a km step length into a coordinate delta), matching
# the helper in test_regime so step lengths land cleanly inside / outside the hysteresis band.
KM_PER_DEG_LAT = 6_371_000.0 * np.pi / 180.0 / 1000.0


def _segment(seg_id, individual_id, step_km_list, *, start_lat=0.0, lon=0.0):
    """One segment whose consecutive fixes are `step_km_list` km apart in latitude (incoming steps [NaN, *list])."""
    lats = [start_lat]
    for s in step_km_list:
        lats.append(lats[-1] + s / KM_PER_DEG_LAT)
    ts = pd.date_range("2020-03-01", periods=len(lats), freq="6h")
    return pd.DataFrame(
        {
            "segment_id": seg_id,
            "individual_id": individual_id,
            "timestamp": ts,
            "lat": lats,
            "lon": lon,
        }
    )


# Incoming steps: forage, forage, forage, moving(->launch), moving, forage(->settle).
# Fix index:        0      1        2        3        4(L)     5        6(settle)
STEPS = [0.01, 0.01, 0.01, 0.20, 0.20, 0.01]
CL = 3  # small context so the whole thing is checkable by hand: windows have CL-1 = 2 steps.


def _fixture():
    return _segment("segA", 7, STEPS)


def test_expected_labels_and_can_launch():
    frame = _fixture()
    sw = build_state_windows(frame, context_length=CL)

    # T = 7 fixes -> targets t = 3,4,5,6 -> 4 windows.
    assert len(sw.y) == 4
    assert sw.X.shape == (4, CL - 1, 2)
    assert sw.channels == ("dlon", "dlat")
    assert sw.context_length == CL

    # Target fix states: fix3 foraging, fix4 local_launch, fix5 moving, fix6 settle.
    expected = ["foraging", "local_launch", "moving", "settle"]
    assert list(sw.y) == [STATE_TO_IDX[s] for s in expected]

    # can_launch = last in-window fix (t-1) is foraging: fixes 2,3 foraging -> True; fixes 4,5 moving -> False.
    assert list(sw.meta["can_launch"]) == [True, True, False, False]
    assert list(sw.meta["prev_base_regime"]) == ["foraging", "foraging", "moving", "moving"]
    assert list(sw.meta["target_idx"]) == [3, 4, 5, 6]
    assert list(sw.meta["segment_id"]) == ["segA"] * 4
    assert list(sw.meta["individual_id"]) == [7] * 4


def test_displacements_match_raw_diff():
    frame = _fixture()
    sw = build_state_windows(frame, context_length=CL)

    coords = frame[["lon", "lat"]].to_numpy()
    for row, t in enumerate(range(CL, len(frame))):
        expected = np.diff(coords[t - CL : t], axis=0)  # (CL-1, 2)
        np.testing.assert_allclose(sw.X[row], expected)

    # Pure-latitude motion: dlon channel is all zero.
    np.testing.assert_allclose(sw.X[:, :, 0], 0.0)


def test_time_and_season_channels_on_landed_fixes():
    frame = _fixture()
    sw = build_state_windows(
        frame, context_length=CL, use_time_features=True, use_season_features=True
    )

    assert sw.channels == ("dlon", "dlat", "tod_sin", "tod_cos", "doy_sin", "doy_cos")
    assert sw.X.shape == (4, CL - 1, 6)

    ts = frame["timestamp"].to_numpy()
    lon = frame["lon"].to_numpy()
    for row, t in enumerate(range(CL, len(frame))):
        dest = slice(t - CL + 1, t)  # fixes the CL-1 steps land on
        tod_sin, tod_cos = local_solar_sincos(ts[dest], lon[dest])
        doy_sin, doy_cos = day_of_year_sincos(ts[dest])
        np.testing.assert_allclose(sw.X[row, :, 2], tod_sin)
        np.testing.assert_allclose(sw.X[row, :, 3], tod_cos)
        np.testing.assert_allclose(sw.X[row, :, 4], doy_sin)
        np.testing.assert_allclose(sw.X[row, :, 5], doy_cos)


def test_bout_net_km_present_for_moving_targets():
    frame = _fixture()
    sw = build_state_windows(frame, context_length=CL)
    net = dict(zip(sw.meta["target_idx"], sw.meta["bout_net_km"]))
    # Foraging / settle targets are outside a moving bout -> NaN; moving/launch targets carry the bout's net km.
    assert np.isnan(net[3]) and np.isnan(net[6])
    assert net[4] > 0 and net[5] > 0


def test_short_segment_skipped_and_no_cross_segment_windows():
    long_seg = _segment("segA", 7, STEPS)
    short_seg = _segment("segB", 9, [0.01, 0.01])  # 3 fixes; T == CL -> no windows
    frame = pd.concat([long_seg, short_seg], ignore_index=True)

    sw = build_state_windows(frame, context_length=CL)
    # Only the long segment contributes; the short one is dropped, and nothing spans the boundary.
    assert set(sw.meta["segment_id"]) == {"segA"}
    assert len(sw.y) == 4


def test_matches_canonical_labeller():
    # The dataset's labels must equal the canonical per-fix state at the target index (no relabelling drift).
    frame = _fixture()
    labelled = label_migratory_launches(label_regimes(frame))
    sw = build_state_windows(frame, context_length=CL)
    for t, y in zip(sw.meta["target_idx"], sw.y):
        assert STATE_TO_IDX[labelled["state"].to_numpy()[t]] == y
