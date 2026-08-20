# GenAI reference: I used GenAI to help with these test cases, mirroring the structure of the other model test suites.
# I asked for GenAI to confirm solar shift correctness (longitude vs clock), the circular/unit circle encoding,
# vectorised vs scalar agreement, and also an independent recomputation of the documented formula. And to double check 
# I verified them by running against known anchor points (midnight -> (0, 1), noon -> (0, -1)) computed by hand.
"""Tests for the canonical local-solar time-of-day feature (`features/time.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.features import time
from animal_movement_fm.features.time import (
    DAYS_PER_YEAR,
    DEG_LON_PER_HOUR,
    HOURS_PER_DAY,
    day_of_year_sincos,
    local_solar_sincos,
)


class TestSolarShift:
    def test_longitude_shift_equals_clock_shift(self):
        # +15 deg of longitude is exactly +1 h of solar time, so shifting either gives the same phase.
        a = local_solar_sincos(pd.Timestamp("2024-06-01 15:00"), 15.0)
        b = local_solar_sincos(pd.Timestamp("2024-06-01 16:00"), 0.0)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_midnight_maps_to_0_1(self):
        # 00:00 solar -> angle 0 -> (sin, cos) = (0, 1)
        s, c = local_solar_sincos(pd.Timestamp("2024-01-01 00:00"), 0.0)
        np.testing.assert_allclose([s, c], [0.0, 1.0], atol=1e-12)

    def test_noon_maps_to_0_neg1(self):
        # 12:00 solar -> angle pi -> (sin, cos) = (0, -1)
        s, c = local_solar_sincos(pd.Timestamp("2024-01-01 12:00"), 0.0)
        np.testing.assert_allclose([s, c], [0.0, -1.0], atol=1e-12)

    def test_full_rotation_longitude_wraps(self):
        # lon -360 deg is a whole rotation (-24 h), which mods back to the uncorrected phase.
        a = local_solar_sincos(pd.Timestamp("2024-06-01 08:00"), 0.0)
        b = local_solar_sincos(pd.Timestamp("2024-06-01 08:00"), -360.0)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_circular_continuity_across_midnight(self):
        # 23:00 and 01:00 are 2 h apart on the 24 h circle -> small chord, not opposite ends.
        s1, c1 = local_solar_sincos(pd.Timestamp("2024-01-01 23:00"), 0.0)
        s2, c2 = local_solar_sincos(pd.Timestamp("2024-01-02 01:00"), 0.0)
        assert np.hypot(s1 - s2, c1 - c2) < 0.6


class TestUnitCircle:
    def test_scalar_on_unit_circle(self):
        s, c = local_solar_sincos(pd.Timestamp("2024-03-14 07:23"), 42.0)
        assert np.isclose(s**2 + c**2, 1.0)

    def test_vector_on_unit_circle(self):
        ts = pd.to_datetime(
            ["2024-06-01 03:00", "2024-06-01 09:07", "2024-06-01 21:59"]
        ).to_numpy()
        lon = np.array([-73.0, 10.0, 151.2])
        s, c = local_solar_sincos(ts, lon)
        np.testing.assert_allclose(s**2 + c**2, np.ones(3), atol=1e-12)


class TestVectorisedMatchesScalar:
    def test_array_equals_elementwise(self):
        ts = pd.to_datetime(
            ["2024-01-02 21:30", "2024-07-15 05:45", "2024-12-31 23:59"]
        )
        lon = np.array([-73.0, 0.0, 174.7])
        sv, cv = local_solar_sincos(ts.to_numpy(), lon)
        for i in range(len(ts)):
            s, c = local_solar_sincos(ts[i], float(lon[i]))
            assert np.isclose(sv[i], s)
            assert np.isclose(cv[i], c)

    def test_array_output_shapes(self):
        ts = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 06:00"]).to_numpy()
        s, c = local_solar_sincos(ts, np.array([0.0, 12.0]))
        assert s.shape == (2,)
        assert c.shape == (2,)


class TestReferenceFormula:
    # Independent recomputation of the documented formula (not a second call to the function).
    @pytest.mark.parametrize(
        "ts,lon",
        [
            ("2024-01-02 21:30", -73.0),
            ("2024-06-21 12:00", 0.0),
            ("2024-11-05 03:15", 151.21),
        ],
    )
    def test_matches_manual_formula(self, ts, lon):
        t = pd.Timestamp(ts)
        utc_hour = t.hour + t.minute / 60.0
        solar_hour = (utc_hour + lon / DEG_LON_PER_HOUR) % HOURS_PER_DAY
        angle = 2.0 * np.pi * (solar_hour / HOURS_PER_DAY)
        s, c = local_solar_sincos(t, lon)
        np.testing.assert_allclose([s, c], [np.sin(angle), np.cos(angle)], atol=1e-12)

# GenAI generated, but again, I broke down what I wanted it to test, reformated some of the code slighlty and verified the output
class TestDayOfYearAnchors:
    def test_new_year_maps_to_0_1(self):
        # 1 Jan is day 1 -> angle 0 -> (sin, cos) = (0, 1)
        s, c = day_of_year_sincos(pd.Timestamp("2023-01-01 09:00"))
        np.testing.assert_allclose([s, c], [0.0, 1.0], atol=1e-12)

    def test_mid_year_is_roughly_opposite(self):
        # ~3 July is day 184, i.e. about half the annual cycle -> angle ~pi -> (0, -1)
        s, c = day_of_year_sincos(pd.Timestamp("2023-07-03"))
        np.testing.assert_allclose([s, c], [0.0, -1.0], atol=0.02)

    def test_circular_continuity_across_new_year(self):
        # 31 Dec and 1 Jan are one day apart on the annual circle -> tiny chord, not opposite ends.
        s1, c1 = day_of_year_sincos(pd.Timestamp("2023-12-31"))
        s2, c2 = day_of_year_sincos(pd.Timestamp("2024-01-01"))
        assert np.hypot(s1 - s2, c1 - c2) < 0.05

    def test_autumn_and_spring_migration_windows_differ(self):
        # The feature must separate the two migration seasons (Aug departure vs Mar return).
        aug = np.array(day_of_year_sincos(pd.Timestamp("2023-08-20")))
        mar = np.array(day_of_year_sincos(pd.Timestamp("2023-03-20")))
        assert np.hypot(*(aug - mar)) > 1.0


class TestDayOfYearProperties:
    def test_on_unit_circle(self):
        s, c = day_of_year_sincos(pd.Timestamp("2023-09-14"))
        assert np.isclose(s**2 + c**2, 1.0)

    def test_same_date_different_years_matches(self):
        # Cyclic only: the encoding carries no absolute year, so it cannot memorise a given year.
        a = day_of_year_sincos(pd.Timestamp("2019-08-20"))
        b = day_of_year_sincos(pd.Timestamp("2023-08-20"))
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_leap_day_is_handled(self):
        s, c = day_of_year_sincos(pd.Timestamp("2024-02-29"))
        assert np.isfinite(s) and np.isfinite(c)

    def test_no_longitude_argument_needed(self):
        # Season is one calendar cycle corpus-wide; the helper takes only a timestamp.
        with pytest.raises(TypeError):
            day_of_year_sincos(pd.Timestamp("2023-08-20"), 15.0)


class TestDayOfYearVectorised:
    def test_array_equals_elementwise(self):
        ts = pd.to_datetime(["2023-01-15", "2023-06-30", "2023-11-02"])
        sv, cv = day_of_year_sincos(ts.to_numpy())
        for i in range(len(ts)):
            s, c = day_of_year_sincos(ts[i])
            assert np.isclose(sv[i], s)
            assert np.isclose(cv[i], c)

    def test_array_output_shapes(self):
        ts = pd.to_datetime(["2023-01-01", "2023-07-01"]).to_numpy()
        s, c = day_of_year_sincos(ts)
        assert s.shape == (2,)
        assert c.shape == (2,)


class TestDayOfYearReferenceFormula:
    # Independent recomputation of the documented formula (not a second call to the function).
    @pytest.mark.parametrize("ts", ["2023-01-02", "2023-08-20", "2024-02-29", "2024-12-31"])
    def test_matches_manual_formula(self, ts):
        t = pd.Timestamp(ts)
        angle = 2.0 * np.pi * ((t.dayofyear - 1.0) / DAYS_PER_YEAR)
        s, c = day_of_year_sincos(t)
        np.testing.assert_allclose([s, c], [np.sin(angle), np.cos(angle)], atol=1e-12)


# --------------------------------------------------------------------------------------------------
# The epoch-second variants added for Tier 2 must agree with the Tier 1 definitions above, which is
# what makes them a faster path rather than a second definition. GenAI addition; see ref above file
# --------------------------------------------------------------------------------------------------


def _epoch(ts):
    """Epoch seconds, via an explicit unit conversion.

    Deliberately not `astype("int64") / 1e9`: DuckDB hands back datetime64[us] while `pd.date_range`
    hands back [ns], and that divisor is right for one and 1000x wrong for the other. Asking numpy for
    seconds takes the divisor out of the problem.
    """
    return pd.DatetimeIndex(ts).to_numpy().astype("datetime64[s]").astype("int64").astype(float)


def test_epoch_solar_matches_the_pandas_definition():
    ts = pd.date_range("2019-03-01", periods=400, freq="97min", tz=None)
    lon = np.linspace(-179.0, 179.0, len(ts))

    want = time.local_solar_sincos(ts, lon)
    got = time.local_solar_sincos_from_epoch(_epoch(ts), lon)
    assert np.allclose(got[0], want[0], atol=1e-12)
    assert np.allclose(got[1], want[1], atol=1e-12)


def test_epoch_solar_differs_from_pandas_only_by_dropped_seconds():
    """The Tier 1 function truncates to hour + minute/60 to stay identical to the XGBoost feature."""
    ts = pd.date_range("2019-03-01 00:00:37", periods=50, freq="53min")
    lon = np.zeros(len(ts))
    want = time.local_solar_sincos(ts, lon)
    got = time.local_solar_sincos_from_epoch(_epoch(ts), lon)
    # 59 s of a 24 h cycle is 0.0043 rad; the encoding is bounded by that, not equal to it.
    assert np.max(np.abs(np.array(got) - np.array(want))) < 2 * np.pi * 60 / 86400


def test_epoch_day_of_year_matches_the_pandas_definition_including_leap_years():
    ts = pd.DatetimeIndex(
        ["2019-01-01", "2019-12-31", "2020-02-29", "2020-12-31", "2021-06-15", "2024-01-01"]
    )
    want = time.day_of_year_sincos(ts)
    got = time.day_of_year_sincos_from_epoch(_epoch(ts))  # no lat: no hemisphere flip
    assert np.allclose(got[0], want[0], atol=1e-12)
    assert np.allclose(got[1], want[1], atol=1e-12)


def test_hemisphere_flip_is_half_a_year_of_phase():
    """Rotation by pi, i.e. a sign flip on both components. Applied per fix, never per species."""
    ts = pd.DatetimeIndex(["2021-01-15", "2021-07-15"])
    e = _epoch(ts)

    north = time.day_of_year_sincos_from_epoch(e, lat=np.array([45.0, 45.0]))
    south = time.day_of_year_sincos_from_epoch(e, lat=np.array([-45.0, -45.0]))
    assert np.allclose(np.array(south), -np.array(north), atol=1e-12)

    # January in the south sits where July in the north does. 15 Jan is day 15 and 15 Jul is day 196,
    # while half a year is 182.625 days, so these two dates are 1.625 days short of exactly opposite --
    # 0.028 rad, which is the tolerance below rather than a slack figure.
    jan_south = np.array([south[0][0], south[1][0]])
    jul_north = np.array([north[0][1], north[1][1]])
    assert np.allclose(jan_south, jul_north, atol=2 * np.pi * 1.625 / DAYS_PER_YEAR)

    # The equator is assigned north, and the split is strictly on the sign of the latitude.
    at_zero = time.day_of_year_sincos_from_epoch(e, lat=np.zeros(2))
    assert np.allclose(np.array(at_zero), np.array(north), atol=1e-12)
