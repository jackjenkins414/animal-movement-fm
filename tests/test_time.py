# GenAI reference: I used GenAI to help with these test cases, mirroring the structure of the other model test suites.
# I asked for GenAI to confirm solar shift correctness (longitude vs clock), the circular/unit circle encoding,
# vectorised vs scalar agreement, and also an independent recomputation of the documented formula. And to double check 
# I verified them by running against known anchor points (midnight -> (0, 1), noon -> (0, -1)) computed by hand.
"""Tests for the canonical local-solar time-of-day feature (`features/time.py`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.features.time import (
    DEG_LON_PER_HOUR,
    HOURS_PER_DAY,
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
