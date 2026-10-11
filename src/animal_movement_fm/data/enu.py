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
    bin_seconds: int = BIN_SECONDS,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode context positions into the registered feature tensor and return the reversible window scale.

    timestamp_s must align with lon_deg/lat_deg; elapsed time and clock features are derived directly from it rather than 
    assuming fixed 6h spacing.
    """
    steps = enu_steps(lon_deg, lat_deg)  # (..., T-1, 2)
    scale = window_scale(steps, sigma=sigma) # (...,)

    timestamp_s = np.asarray(timestamp_s, dtype=float)
    if timestamp_s.shape != np.shape(lon_deg):
        raise ValueError(
            f"timestamp_s has shape {timestamp_s.shape}, expected {np.shape(lon_deg)}: one timestamp "
            "per position, not per step"
        )
    elapsed_s = np.diff(timestamp_s, axis=-1)
    features = _token_features(
        steps, scale, elapsed_s, lon_deg[..., 1:], lat_deg[..., 1:], timestamp_s[..., 1:],
        bin_seconds=bin_seconds, clip=clip,
    )
    return features, scale


def _token_features(
    steps: np.ndarray,
    scale: np.ndarray,
    elapsed_s: np.ndarray,
    dest_lon: np.ndarray,
    dest_lat: np.ndarray,
    dest_ts: np.ndarray,
    *,
    bin_seconds: int,
    clip: float | None = None,
    span: np.ndarray | None = None,
) -> np.ndarray:
    """
    The registered per token channels for steps in metres, each landing on its destination fix.

    `span` divides the scaled step by the bins it covers (gap tokens); `clip` bounds the scaled step only so log_speed
    and log_dt keep the true magnitude
    """
    magnitude = np.linalg.norm(steps, axis=-1) 
    inv = 1.0 / scale[..., None]
    d_scaled = steps * inv[..., None]
    if span is not None:
        d_scaled = d_scaled / span[..., None]
    if clip is not None:
        d_scaled = np.clip(d_scaled, -clip, clip)

    dt_bins = elapsed_s / bin_seconds  # in bins of the grid the window was built on

    # direction is undefined near zero movement so gate heading there.
    moving = magnitude > HEADING_GATE_M
    safe = np.where(moving, magnitude, 1.0)
    heading_cos = np.where(moving, steps[..., 0] / safe, 0.0)
    heading_sin = np.where(moving, steps[..., 1] / safe, 0.0)

    log_scale = np.log(scale)[..., None] * np.ones_like(magnitude)
    log_dt = np.log(np.maximum(elapsed_s, 1.0))
    log_speed = np.log1p(magnitude / np.maximum(elapsed_s, 1.0))

    # clock features belong to the destination fix of each displacement
    solar_sin, solar_cos = local_solar_sincos_from_epoch(dest_ts, dest_lon)
    doy_sin, doy_cos = day_of_year_sincos_from_epoch(dest_ts, lat=dest_lat)

    return np.stack(
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


def previous_observed(observed: np.ndarray, n_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (source column, observed) for the tokens ending at columns 1 ... n_tokens of an (N, C) occupancy grid.

    A token's source is the last observed column before its destination; it is observed when its destination is and
    such a source exists. Unobserved tokens carry source 0.
    """
    column = np.arange(observed.shape[1])
    last = np.maximum.accumulate(np.where(observed, column, -1), axis=1)
    source = last[:, :n_tokens]  # last observed column strictly before columns 1 ... n_tokens
    valid = observed[:, 1 : n_tokens + 1] & (source >= 0)
    return np.where(valid, source, 0), valid


def encode_history(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    *,
    timestamp_s: np.ndarray,
    observed: np.ndarray,
    core: int,
    sigma: float = SCALE_FLOOR_M,
    bin_seconds: int = BIN_SECONDS,
    clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode movement history with missing observations into features, a mask, and a scale.

    Last `core` bins must be observed and use the same encoding and scale as encode_window. Earlier steps use the previous observed 
    position, with gaps accounted for in the scaled displacement and time features; missing tokens are zeroed and masked out.
    """
    lon, lat, ts = (np.asarray(a, dtype=float) for a in (lon_deg, lat_deg, timestamp_s))
    observed = np.asarray(observed, dtype=bool)
    if not observed[:, -core:].all():
        raise ValueError(f"every window's last {core} bins must be observed; they are the contiguous core")
    core_features, scale = encode_window(
        lon[:, -core:], lat[:, -core:], timestamp_s=ts[:, -core:], sigma=sigma, bin_seconds=bin_seconds, clip=clip
    )
    core_mask = np.ones(core_features.shape[:2], dtype=bool)
    n_extended = lon.shape[1] - core
    if n_extended == 0:
        return core_features, core_mask, scale

    # unobserved bins take t's values so nothing non-finite reaches the arithmetic; their tokens are zeroed below
    lon, lat, ts = (np.where(observed, a, a[:, -1:]) for a in (lon, lat, ts))
    source, valid = previous_observed(observed, n_extended)
    rows = np.arange(len(lon))[:, None]
    dest = np.arange(1, n_extended + 1)[None, :]
    lon_rad, lat_rad = np.radians(lon), np.radians(lat)
    east = EARTH_RADIUS_M * np.cos(lat_rad[:, -1:]) * wrap_longitude(lon_rad[rows, dest] - lon_rad[rows, source])
    north = EARTH_RADIUS_M * (lat_rad[rows, dest] - lat_rad[rows, source])
    span = np.where(valid, dest - source, 1)
    extended = _token_features(
        np.stack([east, north], axis=-1), scale, ts[rows, dest] - ts[rows, source],
        lon[rows, dest], lat[rows, dest], ts[rows, dest], bin_seconds=bin_seconds, clip=clip, span=span,
    )
    extended = np.where(valid[..., None], extended, 0.0)
    return np.concatenate([extended, core_features], axis=1), np.concatenate([valid, core_mask], axis=1), scale


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

# targets are encoded by enu_steps on the (anchor, target) pair, so east is measured at the target's latitude; enu_to_lonlat inverts at the anchor's
def enu_to_lonlat_exact(
    d_east_m: np.ndarray, d_north_m: np.ndarray, anchor_lon_deg: np.ndarray, anchor_lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    The exact inverse of a target step from enu_steps: latitude from north first, then east at that latitude.
    """
    lat = np.asarray(anchor_lat_deg, dtype=float) + np.degrees(np.asarray(d_north_m, dtype=float) / EARTH_RADIUS_M)
    d_lon = np.asarray(d_east_m, dtype=float) / (EARTH_RADIUS_M * np.cos(np.radians(lat)))
    return np.asarray(anchor_lon_deg, dtype=float) + np.degrees(d_lon), lat
