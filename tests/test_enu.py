# GenAI reference: used to write these tests from the behaviours I gave which were scaling round trips, scale floor not clamp, 
# antimeridian wrapping, heading threshold. Also replaced an older determinism test that wasn’t checking leakage. I checked all 
# the tests. 
from __future__ import annotations

import numpy as np
import pytest

from animal_movement_fm.data import enu


def _clock(batch, positions, start="2021-06-15T00:00:00"):
    """Epoch seconds on an exact one-bin grid. Explicit now that `timestamp_s` has no default."""
    t0 = np.datetime64(start).astype("datetime64[s]").astype("int64")
    return np.tile(t0 + np.arange(positions) * enu.BIN_SECONDS, (batch, 1)).astype(float)


def test_scale_never_below_floor_and_is_a_floor_not_a_clamp():
    """A stationary window floors at sigma; a moving one is strictly above both inputs."""
    stationary = np.zeros((1, 11, 2))
    assert enu.window_scale(stationary)[0] == pytest.approx(enu.SCALE_FLOOR_M)

    steps = np.full((1, 11, 2), 3000.0)
    median_magnitude = np.hypot(3000.0, 3000.0)  # 4242.6407 m
    scale = enu.window_scale(steps)[0]
    assert scale > median_magnitude  # Quadrature adds: a clamp would have returned the median unchanged.
    assert scale == pytest.approx(np.hypot(median_magnitude, enu.SCALE_FLOOR_M), rel=1e-12)


def test_zero_scale_window_does_not_produce_nan():
    """*C. ciconia* has a p0.1 window scale of 0.0 m. Without the floor this is a divide by zero."""
    lon = np.zeros((1, 12))
    lat = np.zeros((1, 12))
    features, scale = enu.encode_window(lon, lat, timestamp_s=_clock(1, 12))
    assert np.isfinite(features).all()
    assert scale[0] == pytest.approx(enu.SCALE_FLOOR_M)


def test_target_scaling_round_trips_exactly():
    """The inverse must be exact: reporting is in metres and goes through it."""
    rng = np.random.default_rng(42)
    target = rng.normal(0, 5000, size=(64, 2))
    scale = rng.uniform(30, 20000, size=64)
    recovered = enu.unscale_prediction(enu.scale_target(target, scale), scale)
    assert np.allclose(recovered, target, rtol=0, atol=1e-9)


def test_longitude_wrap_across_the_antimeridian():
    """179.9E to 179.9W is 0.2 degrees, not 359.8."""
    steps = enu.enu_steps(np.array([[179.9, -179.9]]), np.array([[-60.0, -60.0]]))
    east = steps[0, 0, 0]
    assert abs(east) < 12_000  # ~11.1 km at this latitude, not ~20,000 km.
    assert enu.wrap_longitude(np.array([np.pi * 1.9]))[0] == pytest.approx(-0.1 * np.pi, rel=1e-9)


def test_heading_is_gated_off_below_the_threshold():
    """Turning angle is singular at near-zero steps; the gate is what keeps direction meaningful."""
    tiny = 1e-6
    lon = np.array([[0.0, tiny, 2 * tiny, 3 * tiny, 4 * tiny, 5 * tiny,
                     6 * tiny, 7 * tiny, 8 * tiny, 9 * tiny, 10 * tiny, 11 * tiny]])
    lat = np.zeros((1, 12))
    features, _ = enu.encode_window(lon, lat, timestamp_s=_clock(1, 12))
    cos_i = enu.FEATURES.index("heading_cos")
    sin_i = enu.FEATURES.index("heading_sin")
    assert np.allclose(features[..., cos_i], 0.0)
    assert np.allclose(features[..., sin_i], 0.0)


def test_heading_is_a_unit_vector_when_moving():
    lon = np.linspace(0, 1.0, 12)[None, :]
    lat = np.zeros((1, 12))
    features, _ = enu.encode_window(lon, lat, timestamp_s=_clock(1, 12))
    cos_i, sin_i = enu.FEATURES.index("heading_cos"), enu.FEATURES.index("heading_sin")
    norms = np.hypot(features[..., cos_i], features[..., sin_i])
    assert np.allclose(norms, 1.0, atol=1e-9)


def test_feature_count_matches_the_declared_channel_list():
    lon = np.linspace(0, 0.1, 12)[None, :]
    features, _ = enu.encode_window(lon, np.zeros((1, 12)), timestamp_s=_clock(1, 12))
    assert features.shape == (1, 11, enu.N_FEATURES)
    assert len(enu.FEATURES) == enu.N_FEATURES


def test_tangent_plane_inverts():
    """enu_to_lonlat is the inverse of the step construction at the anchor."""
    lon = np.array([[10.0, 10.5]])
    lat = np.array([[-45.0, -44.8]])
    step = enu.enu_steps(lon, lat)[0, 0]
    back_lon, back_lat = enu.enu_to_lonlat(step[0], step[1], lon[0, -1], lat[0, -1])
    # The plane is anchored at the LAST position, so inverting from it recovers a point one step beyond.
    assert back_lon == pytest.approx(10.5 + 0.5, rel=1e-6)
    assert back_lat == pytest.approx(-44.8 + 0.2, rel=1e-6)


def test_the_encoder_scales_whatever_it_is_handed():
    """
    `encode_window` has no notion of a target: it normalises over every position it receives.

    So the no-leakage guarantee cannot live here. It lives in `SpeciesWindows.gather`, which hands the
    encoder the first C positions only, and it is tested there
    (`test_target_perturbation_does_not_move_the_scale`). This test exists to pin down *where* the
    guarantee is, because the previous version of it encoded the same window twice and asserted
    equality -- a determinism check wearing a leakage check's name.
    """
    # Six short steps then five long ones, so the median sits on a short step and one more long step
    # is enough to move it. A single outlier appended to eleven equal steps would not: the scale is a
    # median, and a test that cannot fail is how the old one survived.
    step = np.array([0.001] * 6 + [0.01] * 5)
    context = np.concatenate([[0.0], np.cumsum(step)])  # 12 positions
    with_target = np.append(context, context[-1] + 0.01)  # 13 positions

    features_c, scale_c = enu.encode_window(context[None, :], np.zeros((1, 12)), timestamp_s=_clock(1, 12))
    features_t, scale_t = enu.encode_window(with_target[None, :], np.zeros((1, 13)), timestamp_s=_clock(1, 13))

    # The 13th position became an input token rather than being held out as a target.
    assert features_c.shape[1] == 11 and features_t.shape[1] == 12
    assert scale_t[0] > scale_c[0]


def test_the_twelve_registered_channels_are_present_and_in_order():
    """The channel list is a registered constant in preregistration_tier2_tuning.md section 5-6."""
    assert enu.FEATURES == (
        "d_east_scaled", "d_north_scaled", "log_dt", "dt_bins", "log_scale",
        "heading_cos", "heading_sin", "log_speed",
        "solar_sin", "solar_cos", "doy_sin", "doy_cos",
    )
    assert enu.N_FEATURES == 12
    assert "gap" not in enu.FEATURES  # structurally zero under the A2 continuity rule


def test_solar_time_is_phase_aligned_across_longitudes():
    """
    Two windows at the same LOCAL solar hour must encode identically despite different longitudes.

    This is the whole point of the longitude/15 correction: 15:00 UTC is mid-afternoon for an eastern
    animal and late morning for a western one, and without the shift the diel signal smears across the
    corpus rather than lining up.
    """
    sin_i, cos_i = enu.FEATURES.index("solar_sin"), enu.FEATURES.index("solar_cos")
    lat = np.zeros((1, 12))

    east = np.full((1, 12), 60.0)                            # 60E, i.e. UTC+4 of solar time.
    west = np.full((1, 12), 0.0)
    clock_east = _clock(1, 12, start="2021-06-15T00:00:00")
    clock_west = _clock(1, 12, start="2021-06-15T04:00:00")  # same solar hour, four hours later UTC.

    f_e, _ = enu.encode_window(east, lat, timestamp_s=clock_east)
    f_w, _ = enu.encode_window(west, lat, timestamp_s=clock_west)
    assert np.allclose(f_e[..., [sin_i, cos_i]], f_w[..., [sin_i, cos_i]], atol=1e-12)


def test_day_of_year_is_hemisphere_aligned_per_fix():
    """
    Austral and boreal summer must land at the same seasonal phase.

    Flipped by the fix's own latitude, not the species': a white stork crosses the equator inside a
    single track, so this has to be per token. Half a year of phase is a rotation by pi, which is a
    sign flip on both components -- hemisphere costs one bit, where feeding latitude itself would
    reintroduce the absolute position the representation deliberately excludes.
    """
    sin_i, cos_i = enu.FEATURES.index("doy_sin"), enu.FEATURES.index("doy_cos")
    lon = np.linspace(0, 0.1, 12)[None, :]
    clock = _clock(1, 12)

    north, _ = enu.encode_window(lon, np.full((1, 12), 45.0), timestamp_s=clock)
    south, _ = enu.encode_window(lon, np.full((1, 12), -45.0), timestamp_s=clock)

    assert np.allclose(north[..., [sin_i, cos_i]], -south[..., [sin_i, cos_i]], atol=1e-12)
    # And the flip is decided per token, not per window.
    mixed, _ = enu.encode_window(
        lon, np.array([[45.0] * 6 + [-45.0] * 6]), timestamp_s=clock
    )
    assert not np.allclose(mixed[..., sin_i], mixed[0, 0, sin_i])


def test_clock_features_are_tagged_to_the_fix_the_step_lands_on():
    """The Tier 1 convention (`transformer.py` uses dest_ts / dest_lon), so both tiers encode the same thing."""
    from animal_movement_fm.features.time import local_solar_sincos_from_epoch

    sin_i = enu.FEATURES.index("solar_sin")
    lon = np.linspace(0.0, 1.1, 12)[None, :]
    clock = _clock(1, 12)
    features, _ = enu.encode_window(lon, np.zeros((1, 12)), timestamp_s=clock)

    expected, _ = local_solar_sincos_from_epoch(clock[:, 1:], lon[:, 1:])
    assert np.allclose(features[..., sin_i], expected, atol=1e-12)


def test_dt_bins_follows_the_grid_and_nothing_else_changes():
    rng = np.random.default_rng(0)
    lon = 10.0 + np.cumsum(rng.normal(0, 1e-3, (3, 12)), axis=1)
    lat = 50.0 + np.cumsum(rng.normal(0, 1e-3, (3, 12)), axis=1)
    clock = _clock(3, 12) / enu.BIN_SECONDS * 3600  # an exact hourly grid
    six_h, scale_six = enu.encode_window(lon, lat, timestamp_s=clock)
    one_h, scale_one = enu.encode_window(lon, lat, timestamp_s=clock, bin_seconds=3600)
    dt = enu.FEATURES.index("dt_bins")
    assert np.allclose(one_h[..., dt], 1.0)
    assert np.allclose(six_h[..., dt], 1 / 6)
    others = [i for i in range(enu.N_FEATURES) if i != dt]
    assert np.array_equal(one_h[..., others], six_h[..., others])
    assert np.array_equal(scale_one, scale_six)
