"""
Distance metrics for movement prediction. 

Lat/lon is not a Euclidean coordinate system, since one degree of 
longitude is about 111km at the equator, ~78km at 45 degrees latitude, 
and 0km at the poles. Thus, Euclidean metrics under estmate distance at 
low latitudes and produce nonsense near the poles. For tracking data 
spanning high latitudes (polar bears, emperor penguins etc) this is a 
meaningful error in the metric itself, not just the model. 

We choose to use Haversine instead, which computes the great-circle
distance over a spherical Earth. This is a standard metric in the 
movement ecology literature (ADD MY REF LATER). 

We treat Earth as a sphere of radius 6_371_000m. The error from ignoring
oblateness is well under 1%, far smaller than any GPS or model error. 
"""

from __future__ import annotations

import numpy as np 

# Readability recommendation  
EARTH_RADIUS_M = 6_371_000.0

def haversine_distance(
        lon1: np.ndarray | float, 
        lat1: np.ndarray | float,
        lon2: np.ndarray | float, 
        lat2: np.ndarray | float, 
) -> np.ndarray | float:
    """
    Great circle distance between two points (or array of points) in metres.

    All inputs are in degrees. Vectorised: arrays of any matching shape work, 
    broadcasting follows numpy rules. 
    """

    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    lon2_r = np.radians(lon2)
    lat2_r = np.radians(lat2)

    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r

    # a = sin^2(dlat/2) + cos(lat1) * cos(lat2) * sin^2(dlon/2)
    # c = 2 * arcsin(sqr(a)); distance = R * c
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2

    # Clip guards small floating point overshoot > 1.0 which would NaN arcsin 
    # Angle between the two points from the centre of the Earth
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    return EARTH_RADIUS_M * c 


def euclidean_distance_degrees(
        lon1: np.ndarray | float, 
        lat1: np.ndarray | float,
        lon2: np.ndarray | float, 
        lat2: np.ndarray | float, 
) -> np.ndarray | float:
    """
    Naive euclidean distance in degrees. NOT a real distance. 

    Included only as a debugging aid. If haversine and euclidean disagree wildly 
    for nearby points, that should be a bug. For trajectories within a small region 
    they should agree up to the latitude scaling factor (~111 km/deg at equator)
    """

    return np.sqrt((lon2 - lon1) ** 2 + (lat2 - lat1) ** 2)

def summarise_errors(errors_m: np.ndarray) -> dict[str, float]:
    """
    Compute summary stats over per-prediction errors (metres). 

    We choose to report the median in addition to mean because animal-movement 
    error distributions are (sometimes) heavily right skewed. Obviously this is 
    dependent on other factors like species, sampling resolution and more. 
    Median is robust to those and is often the more honest headline number. 
    """

    return {
        "n": int(errors_m.size), 
        "mean_m": float(np.mean(errors_m)), 
        "median_m": float(np.median(errors_m)), 
        "p90_m": float(np.percentile(errors_m, 90)), 
        "p99_m": float(np.percentile(errors_m, 99)),
        "std_m": float(np.std(errors_m)),
    }
