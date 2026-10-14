"""
Sample ERA5 covariates and build environmental features for the 1 h bird study.

Each fix uses the ERA5 hour at or before its timestamp. Token features use values sampled at context fixes only; targets do not enter the covariates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

HOUR_S = 3_600

# convert ERA5 geopotential to height
GRAVITY = 9.80665  

# raw covariate order: hourly fields, then static fields
HOURLY = ("u10", "v10", "u100", "v100", "t2m", "blh", "tp", "msl")
STATIC = ("lsm", "elevation")
RAW = HOURLY + STATIC

# environmental channels appended after the 12 movement features
DERIVED = ("tail10", "cross10", "tail100", "cross100")
ENV_CHANNELS = RAW + DERIVED + ("env_observed",)
N_ENV_CHANNELS = len(ENV_CHANNELS)
N_STANDARDISED = N_ENV_CHANNELS - 1  # leave observation flag unscaled

_WIND = {
    "10": (RAW.index("u10"), RAW.index("v10")),
    "100": (RAW.index("u100"), RAW.index("v100")),
}


def era5_hour(epoch_s: np.ndarray) -> np.ndarray:
    """
    Round each timestamp down to the hour, returning int64 epoch seconds.
    """
    return (np.floor(np.asarray(epoch_s, dtype=float) / HOUR_S) * HOUR_S).astype(np.int64)


def era5_month(hour_s: np.ndarray) -> np.ndarray:
    """
    Return each hour's calendar month as datetime64[M].
    """
    return np.asarray(hour_s, dtype=np.int64).astype("datetime64[s]").astype("datetime64[M]")


@dataclass(frozen=True)
class Grid:
    """
    A regular global grid with wrapping longitude and ascending or descending latitude.
    """

    lon0: float
    dlon: float
    nlon: int
    lat0: float
    dlat: float  # signed; ERA5 latitude usually decreases from 90 to -90
    nlat: int

    @classmethod
    def from_coords(cls, lon: np.ndarray, lat: np.ndarray, rtol: float = 1e-6) -> "Grid":
        """
        Build a grid from evenly spaced longitude and latitude coordinates.
        """
        lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
        if len(lon) < 2 or len(lat) < 2:
            raise ValueError("a grid needs at least two longitudes and two latitudes")

        dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
        if not (
            np.allclose(np.diff(lon), dlon, rtol=rtol)
            and np.allclose(np.diff(lat), dlat, rtol=rtol)
        ):
            raise ValueError("long and lat must be evenly spaced")
        if dlon <= 0 or not np.isclose(len(lon) * dlon, 360.0, rtol=rtol):
            raise ValueError(f"long must increase and span the globe; {len(lon)} x {dlon} is not 360")

        return cls(float(lon[0]), float(dlon), len(lon), float(lat[0]), float(dlat), len(lat))


def grid_corners(grid: Grid, lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Return (j0, j1, wy, k0, k1, wx) for bilinear interpolation.

    Rows are j and columns are k; wy and wx weight j1 and k1. Longitude wraps across the grid seam, while latitude is clamped to the outer rows.
    """
    x = np.mod((np.asarray(lon_deg, dtype=float) - grid.lon0) / grid.dlon, grid.nlon)
    column_floor = np.floor(x)
    wx = x - column_floor
    k0 = column_floor.astype(np.int64) % grid.nlon
    k1 = (k0 + 1) % grid.nlon

    y = np.clip((np.asarray(lat_deg, dtype=float) - grid.lat0) / grid.dlat, 0.0, grid.nlat - 1)
    j0 = np.minimum(np.floor(y).astype(np.int64), grid.nlat - 2)
    wy = y - j0
    return j0, j0 + 1, wy, k0, k1, wx


def decode(
    raw: np.ndarray,
    *,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    fill_values=(),
) -> np.ndarray:
    """
    Unpack values to float64 using scale and offset, replacing fill values with NaN.
    """
    raw = np.asarray(raw)
    values = raw.astype(np.float64) * float(scale_factor) + float(add_offset)
    for fill in fill_values:
        if fill is not None:
            values[raw == np.asarray(fill, dtype=raw.dtype)] = np.nan
    return values


def bilinear(corners: np.ndarray, wy: np.ndarray, wx: np.ndarray) -> np.ndarray:
    """
    Interpolate corners ordered (j0 k0, j0 k1, j1 k0, j1 k1) in float64.
    """
    v00, v01, v10, v11 = np.asarray(corners, dtype=np.float64)
    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    return top * (1.0 - wy) + bottom * wy


def sample_field(
    raw: np.ndarray,
    time_index: np.ndarray,
    grid: Grid,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    scale_factor: float = 1.0,
    add_offset: float = 0.0,
    fill_values=(),
) -> np.ndarray:
    """
    Sample a packed (time, lat, lon) field, decoding corners before interpolation.
    """
    j0, j1, wy, k0, k1, wx = grid_corners(grid, lon_deg, lat_deg)
    time_positions = np.asarray(time_index, dtype=np.int64)
    corners = np.stack([
        raw[time_positions, j0, k0],
        raw[time_positions, j0, k1],
        raw[time_positions, j1, k0],
        raw[time_positions, j1, k1],
    ])
    return bilinear(
        decode(corners, scale_factor=scale_factor, add_offset=add_offset, fill_values=fill_values),
        wy,
        wx,
    )


def _coord(dataset, names: tuple[str, ...]) -> np.ndarray:
    """
    Read the first coordinate variable matching one of the supplied names.
    """
    for name in names:
        if name in dataset.variables:
            return np.asarray(dataset.variables[name][:], dtype=float)
    raise KeyError(f"none of {names} in the file; its variables are {sorted(dataset.variables)}")


def _epoch_hours(time_var) -> np.ndarray:
    """
    Convert a netCDF time coordinate to int64 epoch seconds.
    """
    import netCDF4

    dates = netCDF4.num2date(
        time_var[:],
        time_var.units,
        calendar=getattr(time_var, "calendar", "standard"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.array(dates, dtype="datetime64[s]").astype(np.int64)


def read_field(
    path: str | Path,
    variable: str,
    hour_s: np.ndarray,
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    static: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Sample one ERA5 variable and return (values, number of missing hours).

    Read only the time slab needed. Hours absent from the file produce NaN; static fields use the first time step, or the whole field when it is 2D.
    """
    import netCDF4

    hour_s = np.asarray(hour_s, dtype=np.int64)
    with netCDF4.Dataset(path) as dataset:
        grid = Grid.from_coords(
            _coord(dataset, ("longitude", "lon")),
            _coord(dataset, ("latitude", "lat")),
        )
        field = dataset.variables[variable]
        field.set_auto_maskandscale(False)
        dimensions = field.dimensions
        packing = dict(
            scale_factor=getattr(field, "scale_factor", 1.0),
            add_offset=getattr(field, "add_offset", 0.0),
            fill_values=(getattr(field, "_FillValue", None), getattr(field, "missing_value", None)),
        )
        if static:
            raw = np.asarray(field[:] if len(dimensions) == 2 else field[0]).reshape(1, grid.nlat, grid.nlon)
            return sample_field(
                raw, np.zeros(len(hour_s), dtype=np.int64), grid, lon_deg, lat_deg, **packing
            ), 0
        if len(dimensions) != 3:
            raise ValueError(f"{variable} has dimensions {dimensions}; expected (time, latitude, longitude)")

        times = _epoch_hours(dataset.variables[dimensions[0]])
        time_positions = np.clip(np.searchsorted(times, hour_s), 0, len(times) - 1)
        inside = times[time_positions] == hour_s
        values = np.full(len(hour_s), np.nan)
        if inside.any():
            first, last = int(time_positions[inside].min()), int(time_positions[inside].max())
            raw = np.asarray(field[first : last + 1])
            values[inside] = sample_field(
                raw,
                time_positions[inside] - first,
                grid,
                np.asarray(lon_deg, dtype=float)[inside],
                np.asarray(lat_deg, dtype=float)[inside],
                **packing,
            )
    return values, int((~inside).sum())


def elevation_from_geopotential(z: np.ndarray) -> np.ndarray:
    """
    Convert surface geopotential (m^2/s^2) to elevation in metres.
    """
    return np.asarray(z, dtype=float) / GRAVITY


def wind_components(
    u: np.ndarray,
    v: np.ndarray,
    heading_cos: np.ndarray,
    heading_sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return tailwind and crosswind relative to the token's heading.

    Crosswind is positive to the left. A zero heading gives zero components when the wind values are finite.
    """
    return u * heading_cos + v * heading_sin, v * heading_cos - u * heading_sin


def token_channels(raw: np.ndarray, heading_cos: np.ndarray, heading_sin: np.ndarray) -> np.ndarray:
    """
    Combine raw covariates, heading-relative winds and an observation flag.
    """
    raw = np.asarray(raw, dtype=float)
    if raw.shape[-1] != len(RAW):
        raise ValueError(f"raw covariates have {raw.shape[-1]} channels; expected {len(RAW)} in the order {RAW}")

    observed = np.isfinite(raw).all(axis=-1)
    raw = np.where(observed[..., None], raw, 0.0)
    derived = []
    for level in ("10", "100"):
        u, v = (raw[..., i] for i in _WIND[level])
        derived += wind_components(u, v, heading_cos, heading_sin)
    derived = np.where(observed[..., None], np.stack(derived, axis=-1), 0.0)
    return np.concatenate([raw, derived, observed[..., None].astype(float)], axis=-1)


def moment_sums(channels: np.ndarray, window_weight: np.ndarray) -> np.ndarray:
    """
    Return weighted sums (w, w*x, w*x^2) for each standardised channel; each observed token carries its window's weight. Chunk sums can be added.
    """
    channels = np.asarray(channels, dtype=float)
    weights = np.asarray(window_weight, dtype=float)[:, None] * channels[..., -1]
    values = channels[..., :N_STANDARDISED].reshape(-1, N_STANDARDISED)
    weights = weights.reshape(-1)
    return np.stack([np.full(N_STANDARDISED, weights.sum()), weights @ values, weights @ (values * values)])


def stats_from_sums(sums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return channel means and standard deviations, requiring positive weight and spread.
    """
    total, first, second = np.asarray(sums, dtype=float)
    if not (total > 0).all():
        raise ValueError("no observed token carries weight; covariate statistics are undefined")

    mean = first / total
    sd = np.sqrt(np.maximum(second / total - mean**2, 0.0))
    if not (np.isfinite(sd).all() and (sd > 0).all()):
        flat = [ENV_CHANNELS[i] for i in np.flatnonzero(~(np.isfinite(sd) & (sd > 0)))]
        raise ValueError(f"channels {flat} have no spread over the training windows")
    return mean, sd


def standardise(channels: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """
    Standardise environmental channels, preserving the flag and zeroing unobserved tokens.
    """
    channels = np.asarray(channels, dtype=float)
    mean, sd = np.asarray(mean, dtype=float), np.asarray(sd, dtype=float)
    if mean.shape != (N_STANDARDISED,) or sd.shape != (N_STANDARDISED,) or not (sd > 0).all():
        raise ValueError(f"mean and sd need {N_STANDARDISED} values each, sd positive")

    observed = channels[..., -1:] > 0
    scaled = np.where(observed, (channels[..., :N_STANDARDISED] - mean) / sd, 0.0)
    return np.concatenate([scaled, channels[..., -1:]], axis=-1)
