# GenAI reference: Used GenAI for these tests from the behaviours I outlined (these were 3 invalid coord bounds, exact duplicates
# keep one row while conflicting pair is dropped, a missing_deploy is never flagged and more). The teleport track and timestamp
# precision came from bugs I git, and every expected speed is a north only step. All have been check. 
import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.qc import (
    CEILING_MULTIPLIERS,
    INSTANT_INTERVAL_S,
    SPEED_CEILINGS,
    SUSTAINED_INTERVAL_S,
    QCReport,
    ceiling_at_interval,
    flag_invalid_coordinates,
    flag_outside_deployment,
    flag_speed,
    geodesic_speed_kmh,
    resolve_duplicates,
    structural_qc,
)

STORK = "Ciconia ciconia"


def _ts(*values):
    return pd.to_datetime(list(values), utc=True)


def test_invalid_coordinates_catches_all_three_grounds():
    frame = pd.DataFrame(
        {
            "lat": [52.1, 0.0, 91.0, np.nan, -33.8],
            "lon": [13.4, 0.0, 10.0, 5.0, 18.4],
        }
    )
    report = QCReport()
    out = flag_invalid_coordinates(frame, report=report)

    assert list(out["qc_invalid_coord"]) == [False, True, True, True, False]
    assert (report.null_island, report.out_of_range, report.missing_coord) == (1, 1, 1)
    assert report.invalid_coord == 3


def test_zero_zero_is_flagged_but_a_near_zero_real_fix_is_not():
    """Null Island is exactly (0,0); 0.01 degrees away is a legitimate position."""
    frame = pd.DataFrame({"lat": [0.0, 0.01, 0.0], "lon": [0.0, 0.0, 0.01]})
    out = flag_invalid_coordinates(frame)
    assert list(out["qc_invalid_coord"]) == [True, False, False]


def test_exact_duplicates_keep_exactly_one():
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01", "2020-01-02"),
            "lat": [52.0, 52.0, 53.0],
            "lon": [13.0, 13.0, 13.0],
        }
    )
    report = QCReport()
    out = resolve_duplicates(frame, report=report)

    assert list(out["qc_duplicate"]) == [False, True, False]
    assert report.duplicate_rows == 1
    # Identical positions are not a conflict.
    assert report.duplicate_conflicting == 0


def test_gps_conflict_drops_the_whole_group():
    """Two different positions at one instant, no quality field: no principled winner."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01"),
            "lat": [52.0, 48.0],
            "lon": [13.0, 2.0],
        }
    )
    report = QCReport()
    out = resolve_duplicates(frame, report=report)

    assert list(out["qc_duplicate"]) == [True, True]
    assert report.duplicate_conflicting == 1
    assert any("no quality field" in n for n in report.notes)


def test_argos_conflict_keeps_the_best_location_class():
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01", "2020-01-01"),
            "lat": [52.0, 48.0, 40.0],
            "lon": [13.0, 2.0, 1.0],
            "location_quality": ["B", "3", "A"],
        }
    )
    out = resolve_duplicates(frame, quality_col="location_quality")
    # Class 3 is the best available and is the only survivor.
    survivors = out.loc[~out["qc_duplicate"], "location_quality"].tolist()
    assert survivors == ["3"]


def test_argos_ties_break_on_the_smaller_error():
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01"),
            "lat": [52.0, 48.0],
            "lon": [13.0, 2.0],
            "location_quality": ["A", "A"],
            "semi_major_m": [8000.0, 1200.0],
        }
    )
    out = resolve_duplicates(
        frame, quality_col="location_quality", error_col="semi_major_m"
    )
    survivors = out.loc[~out["qc_duplicate"], "semi_major_m"].tolist()
    assert survivors == [1200.0]


def test_unknown_quality_codes_lose_to_known_ones():
    """An unrecognised class must not silently win by sorting first."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01"),
            "lat": [52.0, 48.0],
            "lon": [13.0, 2.0],
            "location_quality": ["?", "B"],
        }
    )
    out = resolve_duplicates(frame, quality_col="location_quality")
    assert out.loc[~out["qc_duplicate"], "location_quality"].tolist() == ["B"]


def test_duplicates_are_scoped_to_one_individual():
    """Two animals at the same instant are not duplicates of each other."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 2],
            "timestamp": _ts("2020-01-01", "2020-01-01"),
            "lat": [52.0, 48.0],
            "lon": [13.0, 2.0],
        }
    )
    out = resolve_duplicates(frame)
    assert not out["qc_duplicate"].any()


def test_deployment_window_flags_both_ends():
    frame = pd.DataFrame(
        {
            "deployment_id": [7, 7, 7],
            "timestamp": _ts("2019-12-31", "2020-06-01", "2021-01-01"),
        }
    )
    deployments = pd.DataFrame(
        {
            "deployment_id": [7],
            "deploy_on_timestamp": _ts("2020-01-01"),
            "deploy_off_timestamp": _ts("2020-12-31"),
        }
    )
    report = QCReport()
    out = flag_outside_deployment(frame, deployments, report=report)

    assert list(out["qc_before_deploy_on"]) == [True, False, False]
    assert list(out["qc_after_deploy_off"]) == [False, False, True]
    assert (report.before_deploy_on, report.after_deploy_off) == (1, 1)


def test_missing_deploy_off_never_flags():
    """57% of deployments have no deploy_off. Absence of evidence must not become a flag."""
    frame = pd.DataFrame({"deployment_id": [7, 7], "timestamp": _ts("2019-01-01", "2029-01-01")})
    deployments = pd.DataFrame(
        {
            "deployment_id": [7],
            "deploy_on_timestamp": _ts("2020-01-01"),
            "deploy_off_timestamp": [pd.NaT],
        }
    )
    report = QCReport()
    out = flag_outside_deployment(frame, deployments, report=report)

    assert list(out["qc_before_deploy_on"]) == [True, False]
    assert not out["qc_after_deploy_off"].any()  # unknowable, so not flagged
    assert report.deployments_with_off == 0


def test_structural_qc_flags_without_deleting():
    """The contract: nothing is dropped, the caller decides."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1, 1],
            "timestamp": _ts("2020-01-01", "2020-01-01", "2020-01-02"),
            "lat": [0.0, 52.0, 52.0],
            "lon": [0.0, 13.0, 13.0],
        }
    )
    out, report = structural_qc(frame)

    assert len(out) == len(frame)
    assert report.rows_in == 3
    assert report.null_island == 1
    assert {"qc_invalid_coord", "qc_duplicate"} <= set(out.columns)


def test_report_is_serialisable():
    frame = pd.DataFrame(
        {"individual_id": [1], "timestamp": _ts("2020-01-01"), "lat": [1.0], "lon": [1.0]}
    )
    _, report = structural_qc(frame)
    row = report.as_dict()
    assert isinstance(row["notes"], str)
    assert row["rows_in"] == 1


def test_missing_key_columns_fail_loudly():
    """A frame without the grouping keys must raise, not silently skip deduplication."""
    with pytest.raises(KeyError):
        resolve_duplicates(pd.DataFrame({"lat": [1.0], "lon": [1.0]}))


# --- speed layer ---------------------------------------------------------------------


def test_geodesic_speed_against_a_known_distance():
    """One degree of latitude is ~111.19 km; over one hour that is the speed."""
    speed = geodesic_speed_kmh(
        np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([3600.0])
    )
    assert np.isnan(speed[0])
    assert speed[1] == pytest.approx(111.19, rel=1e-3)


def test_ceiling_decays_from_instant_to_sustained():
    stork = SPEED_CEILINGS[STORK]
    short = ceiling_at_interval(np.array([1.0, INSTANT_INTERVAL_S]), stork)
    long = ceiling_at_interval(np.array([SUSTAINED_INTERVAL_S, 10 * SUSTAINED_INTERVAL_S]), stork)
    mid = ceiling_at_interval(np.array([3600.0]), stork)[0]

    assert (short == stork.instant_kmh).all()  # flat below the instant interval
    assert (long == stork.sustained_kmh).all()  # flat above the sustained interval
    assert stork.sustained_kmh < mid < stork.instant_kmh  # monotone between


def test_ceiling_is_monotone_non_increasing_in_interval():
    """A longer averaging window can never permit a higher sustained speed."""
    grid = np.logspace(0, 6, 200)
    bound = ceiling_at_interval(grid, SPEED_CEILINGS[STORK])
    assert (np.diff(bound) <= 1e-9).all()


def test_multiplier_scales_the_whole_curve():
    stork = SPEED_CEILINGS[STORK]
    base = ceiling_at_interval(np.array([600.0]), stork)
    for m in CEILING_MULTIPLIERS:
        assert ceiling_at_interval(np.array([600.0]), stork, m) == pytest.approx(base * m)


def _teleport_track():
    """A clean 10-minute track with one null-island fix in the middle."""
    return pd.DataFrame(
        {
            "individual_id": [1] * 7,
            "timestamp": pd.date_range("2020-01-01", periods=7, freq="10min", tz="UTC"),
            "lat": [52.00, 52.01, 52.02, 0.0, 52.04, 52.05, 52.06],
            "lon": [13.00, 13.01, 13.02, 0.0, 13.04, 13.05, 13.06],
        }
    )


def test_ceiling_flags_the_teleport_and_its_innocent_successor():
    """A bad fix corrupts two steps: the one into it and the one out of it."""
    out = flag_speed(_teleport_track(), SPEED_CEILINGS[STORK])
    assert list(out["qc_over_ceiling"]) == [False, False, False, True, True, False, False]


def test_spike_identifies_which_of_the_two_is_actually_bad():
    """The point of the spike rule: the successor is innocent and must not be flagged."""
    out = flag_speed(_teleport_track(), SPEED_CEILINGS[STORK])
    assert list(out["qc_spike"]) == [False, False, False, True, False, False, False]


def test_ordinary_flight_is_not_flagged():
    """A stork covering 30 km in 30 minutes (60 km/h) is unremarkable and must survive."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30"], utc=True),
            "lat": [52.0, 52.27],
            "lon": [13.0, 13.0],
        }
    )
    out = flag_speed(frame, SPEED_CEILINGS[STORK])
    assert out["qc_speed_kmh"].iloc[1] == pytest.approx(60.0, rel=0.02)
    assert not out["qc_over_ceiling"].any()


def test_no_step_is_computed_across_two_animals():
    frame = pd.DataFrame(
        {
            "individual_id": [1, 2],
            "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:10"], utc=True),
            "lat": [52.0, -33.0],
            "lon": [13.0, 18.0],
        }
    )
    out = flag_speed(frame, SPEED_CEILINGS[STORK])
    assert not out["qc_over_ceiling"].any()
    assert out["qc_speed_kmh"].isna().all()


def test_row_order_is_preserved():
    """The function sorts internally; the caller's frame must come back unpermuted."""
    frame = _teleport_track().sample(frac=1, random_state=0)
    out = flag_speed(frame, SPEED_CEILINGS[STORK])
    assert list(out.index) == list(frame.index)
    assert bool(out.sort_index()["qc_spike"].iloc[3])


def test_marine_ceiling_is_loose_enough_for_argos_error():
    """A 5 km Argos error across an hour must not read as impossible movement."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"], utc=True),
            "lat": [-60.0, -60.045],  # ~5 km apart
            "lon": [0.0, 0.0],
        }
    )
    out = flag_speed(frame, SPEED_CEILINGS["Mirounga leonina"])
    assert out["qc_speed_kmh"].iloc[1] == pytest.approx(5.0, rel=0.05)
    assert not out["qc_over_ceiling"].any()


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_speed_is_independent_of_timestamp_precision(unit):
    """Regression: DuckDB returns datetime64[us], pd.date_range gives [ns].

    An int64 view divided by 1e9 is correct for one and wrong by a factor of 1000 for the
    other, which flagged 55% of Canis lupus as over-ceiling before it was caught. The
    fixtures were the reason it slipped through, so precision is parametrised here.
    """
    times = pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 01:00:00"], utc=True)
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": times.as_unit(unit) if hasattr(times, "as_unit") else times,
            "lat": [0.0, 1.0],  # ~111.19 km apart
            "lon": [0.0, 0.0],
        }
    )
    out = flag_speed(frame, SPEED_CEILINGS[STORK])
    assert out["qc_speed_kmh"].iloc[1] == pytest.approx(111.19, rel=1e-3)


def test_a_walking_animal_is_never_over_its_ceiling():
    """Sanity floor: 4 km in 24 h is 0.17 km/h and must not trip any flag."""
    frame = pd.DataFrame(
        {
            "individual_id": [1, 1],
            "timestamp": pd.to_datetime(["2003-05-06 08:00", "2003-05-07 08:00"], utc=True),
            "lat": [45.518521, 45.558223],
            "lon": [93.450498, 93.462025],
        }
    )
    out = flag_speed(frame, SPEED_CEILINGS["Canis lupus"])
    assert out["qc_speed_kmh"].iloc[1] < 1.0
    assert not out["qc_over_ceiling"].any()


def test_every_registered_ceiling_is_internally_consistent():
    for name, ceiling in SPEED_CEILINGS.items():
        assert ceiling.instant_kmh >= ceiling.sustained_kmh, name
        assert ceiling.sustained_kmh > 0, name
        assert ceiling.source.strip(), f"{name} has no stated provenance"
