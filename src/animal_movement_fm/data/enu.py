"""
Encode Tier 2 movement windows as local ENU displacements with reversible per window scaling and time covariates. 
We fix the representation by preregistration; metric steps are normalised by window scale, with elapsed time, heading, 
speed, local solar time, and season added as features. More to be added later...
"""

from __future__ import annotations

import numpy as np

from animal_movement_fm.features.time import (
    day_of_year_sincos_from_epoch,
    local_solar_sincos_from_epoch,
)

# mean Earth radius (IUGG) metres
EARTH_RADIUS_M = 6371008.8

# registered floor for per window scaling
SCALE_FLOOR_M = 30.0

# gate heading near zero movement where direction is mostly noise
HEADING_GATE_M = 10.0

# modelling cadence in seconds
BIN_SECONDS = 6 * 60 * 60

# per token feature order
FEATURES = (
    "d_east_scaled",
    "d_north_scaled",
    "log_dt",
    "dt_bins",
    "log_scale",
    "heading_cos",
    "heading_sin",
    "log_speed",
    "solar_sin",
    "solar_cos",
    "doy_sin",
    "doy_cos",
)
N_FEATURES = len(FEATURES)

# trying these for single docstrings now
def wrap_longitude(delta_rad: np.ndarray) -> np.ndarray:
    """
    Wrap longitude differences into `(-pi, pi]` to handle antimeridian crossings.
    """
    return (delta_rad + np.pi) % (2 * np.pi) - np.pi


# Ref: https://github.com/geospace-code/pymap3d/blob/main/src/pymap3d/enu.py
# Note: The above is local EN tangent plane representation following the ENU convention used by PyMap3D; but here we use a 
# first order spherical approximation because the trajectory windows span small spatial extents relative to the Earth's radius, 
# so local curvature effects are somewhat nonexistent.
def enu_steps(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """
    Return consecutive (east, north) displacements in metres using one tangent plane anchored at the final context position.
    """
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    lat = np.radians(np.asarray(lat_deg, dtype=float))

    anchor_lat = lat[..., -1:]  # last observed fix of the context window
    d_lon = wrap_longitude(lon[..., 1:] - lon[..., :-1])
    d_lat = lat[..., 1:] - lat[..., :-1]

    east = EARTH_RADIUS_M * np.cos(anchor_lat) * d_lon
    north = EARTH_RADIUS_M * d_lat
    return np.stack([east, north], axis=-1)


def window_scale(steps_m: np.ndarray, sigma: float = SCALE_FLOOR_M) -> np.ndarray:
    """
    Return the per-window median step scale, floored in quadrature by sigma
    """
    magnitude = np.linalg.norm(np.asarray(steps_m, dtype=float), axis=-1)
    median = np.median(magnitude, axis=-1)
    return np.sqrt(median**2 + float(sigma) ** 2)


def encode_window(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    timestamp_s: np.ndarray,
    sigma: float = SCALE_FLOOR_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode context positions into the registered feature tensor and return the reversible window scale.

    timestamp_s must align with lon_deg/lat_deg; elapsed time and clock features are derived directly from it rather than 
    assuming fixed 6h spacing.
    """
    steps = enu_steps(lon_deg, lat_deg)  # (..., T-1, 2)
    scale = window_scale(steps, sigma=sigma) # (...,)
    magnitude = np.linalg.norm(steps, axis=-1) # (..., T-1)

    timestamp_s = np.asarray(timestamp_s, dtype=float)
    if timestamp_s.shape != np.shape(lon_deg):
        raise ValueError(
            f"timestamp_s has shape {timestamp_s.shape}, expected {np.shape(lon_deg)}: one timestamp "
            "per position, not per step"
        )
    elapsed_s = np.diff(timestamp_s, axis=-1)

    inv = 1.0 / scale[..., None]
    d_scaled = steps * inv[..., None]

    dt_bins = elapsed_s / BIN_SECONDS

    # direction is undefined near zero movement so gate heading there.
    moving = magnitude > HEADING_GATE_M
    safe = np.where(moving, magnitude, 1.0)
    heading_cos = np.where(moving, steps[..., 0] / safe, 0.0)
    heading_sin = np.where(moving, steps[..., 1] / safe, 0.0)

    log_scale = np.log(scale)[..., None] * np.ones_like(magnitude)
    log_dt = np.log(np.maximum(elapsed_s, 1.0))
    log_speed = np.log1p(magnitude / np.maximum(elapsed_s, 1.0))

    # clock features belong to the destination fix of each displacement
    dest_lon, dest_lat, dest_ts = lon_deg[..., 1:], lat_deg[..., 1:], timestamp_s[..., 1:]
    solar_sin, solar_cos = local_solar_sincos_from_epoch(dest_ts, dest_lon)
    doy_sin, doy_cos = day_of_year_sincos_from_epoch(dest_ts, lat=dest_lat)

    features = np.stack(
        [
            d_scaled[..., 0],
            d_scaled[..., 1],
            log_dt,
            dt_bins,
            log_scale,
            heading_cos,
            heading_sin,
            log_speed,
            solar_sin,
            solar_cos,
            doy_sin,
            doy_cos,
        ],
        axis=-1,
    )
    return features, scale


def scale_target(target_step_m: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Put the target displacement into the windows normalised space. Inverse of unscale_prediction.
    """
    return np.asarray(target_step_m, dtype=float) / np.asarray(scale, dtype=float)[..., None]


def unscale_prediction(pred_scaled: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Convert a normalised displacement prediction back to metres.
    """
    return np.asarray(pred_scaled, dtype=float) * np.asarray(scale, dtype=float)[..., None]


def enu_to_lonlat(
    d_east_m: np.ndarray, d_north_m: np.ndarray, anchor_lon_deg: np.ndarray, anchor_lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert the tangent plane; a metric displacement from the anchor back to (lon, lat) degrees.
    """
    anchor_lat = np.radians(np.asarray(anchor_lat_deg, dtype=float))
    d_lat = np.asarray(d_north_m, dtype=float) / EARTH_RADIUS_M
    d_lon = np.asarray(d_east_m, dtype=float) / (EARTH_RADIUS_M * np.cos(anchor_lat))
    return (
        np.asarray(anchor_lon_deg, dtype=float) + np.degrees(d_lon),
        np.asarray(anchor_lat_deg, dtype=float) + np.degrees(d_lat),
    )
