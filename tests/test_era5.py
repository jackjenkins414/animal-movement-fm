# GenAI reference: I specified these tests for era5.py; synthetic packed ERA5 grid, interpolation/antimeridian/time edge checks, decoding/poles/z-g 
# and token helpers and GenAI drafted this. I checked the code. 
from __future__ import annotations

import netCDF4
import numpy as np
import pytest

from animal_movement_fm.data import era5

UNITS = "hours since 1900-01-01 00:00:00.0"
EPOCH_1900 = np.datetime64("1900-01-01T00:00:00", "s").astype(np.int64)


def _epoch(stamp: str) -> int:
    return int(np.datetime64(stamp, "s").astype(np.int64))


def _write(path, name, lon, lat, raw, *, hours_s=None, scale=1.0, offset=0.0, fill=None, missing=None):
    """A packed int16 field as ERA5 stores it: (time, latitude, longitude), or (latitude, longitude) with no hours."""
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("longitude", len(lon))
        ds.createDimension("latitude", len(lat))
        ds.createVariable("longitude", "f4", ("longitude",))[:] = lon
        ds.createVariable("latitude", "f4", ("latitude",))[:] = lat
        dims = ("latitude", "longitude")
        if hours_s is not None:
            ds.createDimension("time", len(hours_s))
            time = ds.createVariable("time", "i4", ("time",))
            time.units, time.calendar = UNITS, "gregorian"
            time[:] = (np.asarray(hours_s, dtype=np.int64) - EPOCH_1900) // era5.HOUR_S
            dims = ("time",) + dims
        var = ds.createVariable(name, "i2", dims, fill_value=fill)
        var.set_auto_maskandscale(False)
        var.scale_factor, var.add_offset = scale, offset
        if missing is not None:
            var.missing_value = np.int16(missing)
        var[:] = raw
    return path


def _lonlat(step, lon0):
    lon = lon0 + step * np.arange(int(round(360 / step)))
    lat = 90.0 - step * np.arange(int(round(180 / step)) + 1)
    return lon, lat


def test_hour_floor_never_reads_a_later_hour():
    t0 = _epoch("2020-06-15T01:00:00")
    stamps = np.array([t0, t0 + 1, t0 + 3599, t0 + 3599.999, t0 + 3600, -1.0])
    assert era5.era5_hour(stamps).tolist() == [t0, t0, t0, t0, t0 + 3600, -3600]


def test_bilinear_is_exact_on_a_field_linear_in_lon_and_lat(tmp_path):
    lon, lat = _lonlat(1.0, -180.0)
    field = 10 + 2 * lat[:, None] + 0.5 * (lon[None, :] + 180)  # multiples of 0.25, so packing at 0.25 is exact
    hour = _epoch("2020-06-15T00:00:00")
    path = _write(tmp_path / "f.nc", "t2m", lon, lat, np.round(field / 0.25)[None].astype(np.int16), hours_s=[hour],
                  scale=0.25)
    rng = np.random.default_rng(0)
    plon, plat = rng.uniform(-179.5, 178.9, 500), rng.uniform(-89.9, 89.9, 500)  # off the seam
    values, outside = era5.read_field(path, "t2m", np.full(500, hour), plon, plat)
    assert outside == 0
    np.testing.assert_allclose(values, 10 + 2 * plat + 0.5 * (plon + 180), rtol=0, atol=1e-9)


@pytest.mark.parametrize("lon0", [-180.0, 0.0])
def test_antimeridian_under_both_longitude_conventions(tmp_path, lon0):
    step, scale = 0.25, 0.01
    lon, lat = _lonlat(step, lon0)

    def g(lon_deg, lat_deg):  # one physical field, continuous across the seam, quantised to the packing
        x = np.radians(lon_deg)
        return np.round((100 + 50 * np.cos(x) + 30 * np.sin(x) + 0.2 * lat_deg) / scale) * scale

    hour = _epoch("2020-06-15T00:00:00")
    raw = np.round(g(lon[None, :], lat[:, None]) / scale)[None].astype(np.int16)
    path = _write(tmp_path / "a.nc", "u10", lon, lat, raw, hours_s=[hour], scale=scale)
    # on latitude nodes, so only the longitude weight matters
    plon, plat = np.array([179.9, -179.9]), np.array([10.0, -20.25])
    values, _ = era5.read_field(path, "u10", np.full(2, hour), plon, plat)
    expected = [0.4 * g(179.75, 10.0) + 0.6 * g(180.0, 10.0), 0.6 * g(-180.0, -20.25) + 0.4 * g(-179.75, -20.25)]
    np.testing.assert_allclose(values, expected, rtol=0, atol=1e-9)


def test_a_fix_reads_its_own_hour_or_the_one_before(tmp_path):
    lon, lat = _lonlat(10.0, -180.0)
    hours = [_epoch("2020-06-15T00:00:00") + h * 3600 for h in range(3)]
    raw = np.broadcast_to(np.arange(3, dtype=np.int16)[:, None, None], (3, len(lat), len(lon)))
    path = _write(tmp_path / "h.nc", "blh", lon, lat, raw, hours_s=hours)
    stamps = np.array([hours[1], hours[1] + 3599, hours[1] - 0.1, hours[2]], dtype=float)
    values, outside = era5.read_field(path, "blh", era5.era5_hour(stamps), np.zeros(4), np.zeros(4))
    assert outside == 0 and values.tolist() == [1.0, 1.0, 0.0, 2.0]


def test_month_edges_read_the_file_that_holds_the_hour(tmp_path):
    lon, lat = _lonlat(10.0, -180.0)
    jan = [_epoch("2020-01-31T22:00:00"), _epoch("2020-01-31T23:00:00")]
    feb = [_epoch("2020-02-01T00:00:00"), _epoch("2020-02-01T01:00:00")]
    shape = (len(lat), len(lon))
    files = {
        np.datetime64("2020-01", "M"): _write(tmp_path / "jan.nc", "msl", lon, lat,
                                              np.stack([np.full(shape, 22), np.full(shape, 23)]).astype(np.int16),
                                              hours_s=jan),
        np.datetime64("2020-02", "M"): _write(tmp_path / "feb.nc", "msl", lon, lat,
                                              np.stack([np.full(shape, 100), np.full(shape, 101)]).astype(np.int16),
                                              hours_s=feb),
    }
    stamps = np.array([_epoch("2020-01-31T23:59:59"), _epoch("2020-02-01T00:00:00"), _epoch("2020-02-01T00:30:00")],
                      dtype=float)
    hours = era5.era5_hour(stamps)
    months = era5.era5_month(hours)
    assert months.tolist() == list(np.array(["2020-01", "2020-02", "2020-02"], dtype="datetime64[M]"))
    values = [era5.read_field(files[m], "msl", [h], [0.0], [0.0])[0][0] for m, h in zip(months, hours)]
    assert values == [23.0, 100.0, 100.0]
    # the February fix is outside the January file
    stray, outside = era5.read_field(files[months[0]], "msl", hours[1:2], [0.0], [0.0])
    assert np.isnan(stray[0]) and outside == 1


def test_out_of_range_hours_are_nan_and_counted(tmp_path):
    lon, lat = _lonlat(10.0, -180.0)
    hours = [_epoch("2020-06-15T00:00:00") + h * 3600 for h in range(3)]
    path = _write(tmp_path / "r.nc", "tp", lon, lat, np.full((3, len(lat), len(lon)), 7, dtype=np.int16),
                  hours_s=hours)
    asked = np.array([hours[0] - 3600, hours[1], hours[2] + 3 * 3600])
    values, outside = era5.read_field(path, "tp", asked, np.zeros(3), np.zeros(3))
    assert outside == 2 and np.isnan(values[[0, 2]]).all() and values[1] == 7.0
    values, outside = era5.read_field(path, "tp", asked[[0, 2]], np.zeros(2), np.zeros(2))
    assert outside == 2 and np.isnan(values).all()


def test_packing_and_fill_decode():
    raw = np.array([0, 100, -32767, -32768, 5], dtype=np.int16)
    values = era5.decode(raw, scale_factor=0.5, add_offset=1000.0, fill_values=(-32768, 5))
    np.testing.assert_array_equal(values[:3], [1000.0, 1050.0, 1000.0 - 16383.5])
    assert np.isnan(values[3:]).all() and values.dtype == np.float64


def test_a_fill_corner_gives_nan_and_others_decode_with_offset(tmp_path):
    lon, lat = _lonlat(10.0, -180.0)
    raw = np.full((1, len(lat), len(lon)), 200, dtype=np.int16)
    raw[0, 9, 18] = -32767  # lat 0, lon 0
    raw[0, 3, 3] = -32766  # lat 60, lon -150, flagged by missing_value
    hour = _epoch("2020-06-15T00:00:00")
    path = _write(tmp_path / "p.nc", "t2m", lon, lat, raw, hours_s=[hour], scale=0.5, offset=250.0, fill=-32767,
                  missing=-32766)
    values, outside = era5.read_field(path, "t2m", np.full(3, hour), [5.0, -145.0, 100.0], [5.0, 55.0, -40.0])
    assert outside == 0 and np.isnan(values[:2]).all() and values[2] == 350.0


def test_latitude_is_clamped_at_the_poles(tmp_path):
    lon, lat = _lonlat(10.0, -180.0)
    grid = era5.Grid.from_coords(lon, lat)
    j0, j1, wy, *_ = era5.grid_corners(grid, np.zeros(5), np.array([90.0, 90.5, -90.0, -91.0, 85.0]))
    assert j0.tolist() == [0, 0, 17, 17, 0] and j1.tolist() == [1, 1, 18, 18, 1]
    np.testing.assert_allclose(wy, [0.0, 0.0, 1.0, 1.0, 0.5])
    field = np.round(np.broadcast_to(lat[:, None] * 2 + lon[None, :] / 10, (len(lat), len(lon))))
    hour = _epoch("2020-06-15T00:00:00")
    path = _write(tmp_path / "pole.nc", "lsm", lon, lat, field[None].astype(np.int16), hours_s=[hour])
    values, _ = era5.read_field(path, "lsm", np.full(2, hour), [5.0, 5.0], [90.0, -90.0])
    np.testing.assert_allclose(values, [(field[0, 18] + field[0, 19]) / 2, (field[-1, 18] + field[-1, 19]) / 2])


def test_grid_must_span_the_globe():
    with pytest.raises(ValueError, match="span the globe"):
        era5.Grid.from_coords(np.arange(0, 180, 1.0), np.arange(90, -91, -1.0))


@pytest.mark.parametrize("with_time", [False, True])
def test_elevation_is_geopotential_over_g(tmp_path, with_time):
    assert era5.elevation_from_geopotential(era5.GRAVITY * 1234.5) == pytest.approx(1234.5, abs=1e-12)
    lon, lat = _lonlat(10.0, -180.0)
    z = np.full((len(lat), len(lon)), 1000, dtype=np.int16)
    raw = np.stack([z, z * 3, z * 3]) if with_time else z
    hours = [_epoch("2020-06-15T00:00:00") + h * 3600 for h in range(3)] if with_time else None
    path = _write(tmp_path / "z.nc", "z", lon, lat, raw, hours_s=hours, scale=9.80665)
    # a static field reads its first step whatever the fix's hour, even an hour the file does not hold
    values, outside = era5.read_field(path, "z", np.array([0, _epoch("2031-01-01T00:00:00")]), [3.0, -60.0],
                                      [12.0, 40.0], static=True)
    assert outside == 0
    np.testing.assert_allclose(era5.elevation_from_geopotential(values), [1000.0, 1000.0], rtol=1e-12)


def test_wind_components_follow_the_token_heading():
    u, v = np.array([5.0, 5.0, 5.0]), np.array([2.0, 2.0, 2.0])
    tail, cross = era5.wind_components(u, v, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert tail.tolist() == [5.0, 2.0, 0.0]  # east, north, gated
    assert cross.tolist() == [2.0, -5.0, 0.0]  # positive to the left of the heading


def test_token_channels_flag_and_zero_unobserved_tokens():
    raw = np.arange(1.0, 21.0).reshape(1, 2, len(era5.RAW))
    raw[0, 1, era5.RAW.index("tp")] = np.nan
    out = era5.token_channels(raw, np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]]))
    assert out.shape == (1, 2, era5.N_ENV_CHANNELS)
    np.testing.assert_array_equal(out[0, 0, : len(era5.RAW)], raw[0, 0])
    channel = {name: out[0, 0, i] for i, name in enumerate(era5.ENV_CHANNELS)}
    assert (channel["tail10"], channel["cross10"]) == (raw[0, 0, 0], raw[0, 0, 1])
    assert (channel["tail100"], channel["cross100"]) == (raw[0, 0, 2], raw[0, 0, 3])
    assert channel["env_observed"] == 1.0
    assert (out[0, 1] == 0).all()
    with pytest.raises(ValueError, match="channels"):
        era5.token_channels(raw[..., :5], np.ones((1, 2)), np.zeros((1, 2)))


def _channels(rng, n, tokens):
    raw = rng.normal(size=(n, tokens, len(era5.RAW))) * 10 + 100
    raw[0, 0, 0] = np.nan  # one unobserved token
    return era5.token_channels(raw, rng.uniform(-1, 1, (n, tokens)), rng.uniform(-1, 1, (n, tokens)))


def test_weighted_statistics_and_standardisation():
    rng = np.random.default_rng(1)
    channels, weights = _channels(rng, 6, 3), np.array([0.5, 0.5, 0.1, 0.1, 0.1, 0.2])
    mean, sd = era5.stats_from_sums(era5.moment_sums(channels, weights))
    w = (weights[:, None] * channels[..., -1]).reshape(-1)
    x = channels[..., :-1].reshape(-1, era5.N_STANDARDISED)
    hand_mean = (w @ x) / w.sum()
    np.testing.assert_allclose(mean, hand_mean, rtol=1e-12)
    np.testing.assert_allclose(sd, np.sqrt((w @ (x - hand_mean) ** 2) / w.sum()), rtol=1e-9)
    # chunks add
    chunked = era5.moment_sums(channels[:2], weights[:2]) + era5.moment_sums(channels[2:], weights[2:])
    np.testing.assert_allclose(chunked, era5.moment_sums(channels, weights), rtol=1e-12)

    out = era5.standardise(channels, mean, sd)
    np.testing.assert_allclose(out[1:, :, :-1], (channels[1:, :, :-1] - mean) / sd)
    assert (out[0, 0] == 0).all() and (out[..., -1] == channels[..., -1]).all()


def test_standardise_clips_beyond_10_sd_and_leaves_the_flag_and_unobserved_zeros():
    assert era5.CLIP_SD == 10.0
    rng = np.random.default_rng(3)
    channels = _channels(rng, 5, 4)
    mean, sd = channels[1:, :, :-1].reshape(-1, era5.N_STANDARDISED).mean(axis=0), np.full(era5.N_STANDARDISED, 0.5)
    unclipped = era5.standardise(channels, mean, sd, clip=None)
    np.testing.assert_allclose(unclipped[1:, :, :-1], (channels[1:, :, :-1] - mean) / sd)
    assert (np.abs(unclipped[..., :-1]) > 10).any() and (np.abs(unclipped[..., :-1]) < 10).any()

    out = era5.standardise(channels, mean, sd)
    assert np.array_equal(out[..., :-1], np.clip(unclipped[..., :-1], -10, 10))
    inside = np.abs(unclipped[..., :-1]) <= 10
    assert np.array_equal(out[..., :-1][inside], unclipped[..., :-1][inside])  # values within the clip are untouched
    assert (out[..., :-1][unclipped[..., :-1] > 10] == 10).all() and (out[..., :-1][unclipped[..., :-1] < -10] == -10).all()
    assert (out[0, 0] == 0).all() and (out[..., -1] == channels[..., -1]).all()  # unobserved token, flag
    assert np.array_equal(era5.standardise(channels, mean, sd, clip=2.0)[..., :-1], np.clip(unclipped[..., :-1], -2, 2))


def test_a_channel_without_spread_is_refused():
    rng = np.random.default_rng(2)
    channels = _channels(rng, 4, 2)
    channels[..., era5.ENV_CHANNELS.index("lsm")] = 1.0
    with pytest.raises(ValueError, match="lsm"):
        era5.stats_from_sums(era5.moment_sums(channels, np.ones(4)))
