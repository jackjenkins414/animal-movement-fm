"""
Cyclic time features (canonical definitions): local solar time of day, and day of year.

This is the single source of truth for the sin/cos time features used across the project. We first had time of day in
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

Day of year (season)
--------------------
`day_of_year_sincos` encodes where in the annual cycle a fix sits, motivated by the diagnostics' seasonal migration
component and by the state work (season is what should distinguish a migratory departure from a local trip precisely
when the recent movement history is uninformative, i.e. the cold first departure). We deliberately encode ONE seasonal
resolution: month, season and day of year are the same signal coarsened, so carrying several would only add redundant
channels. It is cyclic only, with no absolute year or monotonic time index, which would invite memorising particular
years rather than the annual cycle.

Unlike time of day, no longitude or hemisphere correction is applied. The population follows a single annual cycle
(breeding in Europe, wintering in Africa), so flipping the phase below the equator would misalign birds that are simply
at the southern end of that one cycle; the model should learn "August means autumn departure" corpus wide.

The mechanism also differs from time of day, which matters for interpreting the architectures. Solar hour varies
strongly WITHIN a 24 fix (6 day) window, which is what lets attention weight individual positions by it; day of year is
near constant across such a window, so it acts as a conditioning signal about what kind of period this is rather than
something attention can differentiate positions by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOURS_PER_DAY = 24.0
DEG_LON_PER_HOUR = 15.0  # 360 deg / 24 h: earth's rotation rate in longitude
DAYS_PER_YEAR = 365.25 # averages the leap cycle, so the phase does not drift year to year


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


def day_of_year_sincos(timestamp):
    """
    Map a timestamp to a day of year (seasonal) (sin, cos) pair.

    Vectorised in the same way as `local_solar_sincos`: pass a single timestamp to get two scalars, or an array like of
    them (e.g. every fix in a window) to get two arrays. `timestamp` may be anything `pandas.to_datetime` accepts. No
    longitude is needed, since the seasonal cycle is taken as one calendar cycle for the whole corpus (see module
    docstring).

    Day of year is 1-based, so we shift by one before scaling: 1 Jan maps to angle 0, mirroring the way hour 0 maps to
    angle 0 in the time-of-day feature. The period is `DAYS_PER_YEAR = 365.25`, which averages the leap cycle so the
    same calendar date lands at the same phase in any year. The circular encoding makes 31 Dec and 1 Jan neighbours
    rather than opposite ends of a linear [1, 365] scale.

    Parameters
    ----------
    timestamp : pd.Timestamp | datetime64 | array-like of them
        The time(s) whose seasonal phase we want. For the models this is the time of each fix in the window.

    Returns
    -------
    (sin, cos) : tuple of float or np.ndarray
        The circular encoding of the day of year. Scalar in, scalar out; array in, array out.
    """
    ts = pd.to_datetime(timestamp)

    if isinstance(ts, pd.Timestamp):
        doy = float(ts.dayofyear)
    else:
        ts = pd.DatetimeIndex(ts)
        doy = ts.dayofyear.to_numpy(dtype=float)

    angle = 2.0 * np.pi * ((doy - 1.0) / DAYS_PER_YEAR)
    return np.sin(angle), np.cos(angle)

# epoch second variants for our Tier 2 ingest; note hese avoid pandas datetime conversion and stay compatible with 
# our Tier 1 definitions

SECONDS_PER_DAY = 86400.0
SECONDS_PER_HOUR = 3600.0


def local_solar_sincos_from_epoch(epoch_s, lon):
    """
    Return local solar time of day `(sin, cos)` from epoch seconds and longitude.

    Matches our `local_solar_sincos`, but works directly from epoch seconds and keeps full sub minute precision. Used by 
    Tier 2 ingest to avoid repeated pandas datetime conversion across large batches
    """
    epoch_s = np.asarray(epoch_s, dtype=float)
    utc_hour = np.mod(epoch_s, SECONDS_PER_DAY) / SECONDS_PER_HOUR
    solar_hour = np.mod(
        utc_hour + np.asarray(lon, dtype=float) / DEG_LON_PER_HOUR,
        HOURS_PER_DAY,
    )
    angle = 2.0 * np.pi * (solar_hour / HOURS_PER_DAY)
    return np.sin(angle), np.cos(angle)


def day_of_year_sincos_from_epoch(epoch_s, lat=None):
    """
    Return seasonal `(sin, cos)` features from epoch seconds.

    When latitude is provided, southern hemisphere fixes are shifted by half a year so austral and boreal seasons align in 
    phase. The correction uses only the sign of latitude rather than absolute position, and is applied per fix so tracks 
    crossing the equator are handled correctly.
    """
    seconds = np.asarray(epoch_s, dtype="float64").astype("int64")
    day = seconds.astype("datetime64[s]").astype("datetime64[D]")
    year_start = day.astype("datetime64[Y]").astype("datetime64[D]")
    doy = (day - year_start).astype("int64") + 1.0

    angle = 2.0 * np.pi * ((doy - 1.0) / DAYS_PER_YEAR)
    sin, cos = np.sin(angle), np.cos(angle)

    if lat is not None:
        # half year phase shift south of the equator
        sign = np.where(np.asarray(lat, dtype=float) < 0.0, -1.0, 1.0)
        sin, cos = sin * sign, cos * sign

    return sin, cos