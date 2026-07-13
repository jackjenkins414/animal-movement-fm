"""
Local solar time of day feature (canonical definition).

This is the single source of truth for the sin/cos time of day feature used across the project. We first had it in 
`models/xgboost.py`; that definition is lifted here so the neural rungs (LSTM, Transformer) consume the identical 
feature, and XGBoost will be refactored too (remove once done....)

Why local solar, not raw UTC
----------------------------
Storks span ~3 h of longitude, so a raw UTC hour smears the diurnal (thermal soaring) rhythm across birds because 15:00 
UTC is mid-afternoon for an eastern bird but late morning for a western one. Shifting the UTC hour by `longitude / 15` 
(since Earth rotates 15 deg of longitude per hour) maps every bird's solar afternoon peak to the same corrected hour, so 
the feature is phase aligned across the corpus. Note we use mean solar time only; the equation of time wobble (<= ~15 min) 
is negligible at 6 h cadence, so it is omitted.

Encoding
--------
The corrected solar hour is mapped onto the unit circle as (sin, cos) of `2*pi * solar_hour / 24`. The circular encoding 
makes 23:00 and 01:00 sit close together (as they should) rather than at opposite ends of a linear [0, 24) scale.

Precision note: to stay identical to the original XGBoost feature, the UTC hour of day is taken as `hour + minute/60` (seconds 
dropped); sub minute precision is irrelevant on a 6 h grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0
DEG_LON_PER_HOUR = 15.0  # 360 deg / 24 h: earth's rotation rate in longitude


def local_solar_sincos(timestamp, lon):
    """
    Map (timestamp, longitude) to a local solar time-of-day (sin, cos) pair.

    Vectorised: pass a single timestamp + longitude to get two scalars, or equal length array likes (e.g. every fix in a window) 
    to get two arrays. `timestamp` may be anything `pandas.to_datetime` accepts (a `pd.Timestamp`, a numpy `datetime64`, or an 
    array/Series/Index of them); `lon` is a float or array like of degrees, aligned with `timestamp` (or broadcastable against it).

    Parameters
    ----------
    timestamp : pd.Timestamp | datetime64 | array-like of them
        The time(s) whose local solar phase we want. For the models this is the time of the step being predicted (target time) or of 
        each fix in the window.
    lon : float | array-like of float
        Longitude(s) in degrees, used for the `longitude / 15` solar correction. For a single step this is the longitude of the fix 
        where the bird sits when it takes the step.

    Returns
    -------
    (sin, cos) : tuple of float or np.ndarray
        The circular encoding of the corrected solar hour. Scalar in, scalar out; array in, array out.
    """
    ts = pd.to_datetime(timestamp)

    if isinstance(ts, pd.Timestamp):
        utc_hour = ts.hour + ts.minute / 60.0
    else:
        ts = pd.DatetimeIndex(ts)
        utc_hour = ts.hour.to_numpy(dtype=float) + ts.minute.to_numpy(dtype=float) / 60.0

    lon_arr = np.asarray(lon, dtype=float)
    solar_hour = np.mod(utc_hour + lon_arr / DEG_LON_PER_HOUR, HOURS_PER_DAY)
    angle = 2.0 * np.pi * (solar_hour / HOURS_PER_DAY)
    return np.sin(angle), np.cos(angle)
