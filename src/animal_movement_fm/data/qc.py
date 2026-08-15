# GenAi reference: Located in 3 places in this file; see the specific lines. 
"""
Structural QC for raw movement fixes before resampling. Checks flag invalid coordinates, duplicate fixes, deployment window 
errors, implausible species ranges and speed anomalies without deleting rows; corpus derived thresholds are kept out of this 
layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FLAG_COLUMNS = (
    "qc_invalid_coord",
    "qc_invalid_time",
    "qc_duplicate",
    "qc_before_deploy_on",
    "qc_after_deploy_off",
    "qc_out_of_species_range",
    "qc_over_ceiling",
    "qc_spike",
)

EARTH_RADIUS_M = 6371008.8

# impossibility check since some masks couldnt catch all fixes 
SPECIES_LATITUDE_BOUNDS: dict[str, tuple[float, float]] = {
    "Mirounga leonina": (-90.0, 0.0),
    "Mirounga angustirostris": (0.0, 90.0),
    "Leptonychotes weddellii": (-90.0, -50.0),
    "Lobodon carcinophaga": (-90.0, -40.0),
    "Odobenus rosmarus": (45.0, 90.0),
}

CEILING_MULTIPLIERS = (0.90, 1.00, 1.08)

INSTANT_INTERVAL_S = 60.0
SUSTAINED_INTERVAL_S = 21600.0  # 6 h, the modelling cadence Tier 1 calibrated against


@dataclass(frozen=True)
class SpeedCeiling:
    """
    Ground speed ceiling in km h at instantaneous and sustained timescales.
    
    Speed celing is scale dependent; fastest speed an animal can remain at descreases as the averaging interval grows. 
    These numbers come from the literature, and they are set generously, since their job is to catch the impossible 
    steps in the data. 
    """

    instant_kmh: float
    sustained_kmh: float
    source: str


SPEED_CEILINGS: dict[str, SpeedCeiling] = {
    "Ciconia ciconia": SpeedCeiling(
        150.0,
        100.0,
        "sustained bound registered in Tier 1 (step3_error_dist.py, 90/100/108 km h); cross country migration averages" \
        " ~40 km/h, so ~2.5x headroom",
    ),
    "Larus fuscus": SpeedCeiling(
        150.0,
        100.0,
        "soaring flight models give ~20-25 m/s airspeed and normal cruise 40-48 km/h; matched to the stork bound as the " \
        "other flapping/soaring flier",
    ),
    "Cervus elaphus": SpeedCeiling(
        70.0,
        40.0,
        "inherited from the Canis lupus bound rather than separately cited; bc both are large cursorial mammals and the " \
        "bound is loose enough that a deer specific figure would not move it",
    ),
    "Canis lupus": SpeedCeiling(
        70.0,
        40.0,
        "mean travel speed 2.2 km/h and 18-27 km/day, up to ~50 km/day; instantaneous bound set above sprint speed",
    ),
    "Mirounga leonina": SpeedCeiling(
        25.0,
        15.0,
        "mean swim speed 0.91-1.66 m/s with in dive maxima to 3.0 m/s, ie, ~11 km/h. The bound is deliberately far above that;" \
        " Argos error is kilometres, so a 5 km error across an hour reads as 5 km/h, and a tighter bound could delete real " \
        "positions on measurement noise. This looseness is the missing state space filter showing through",
    ),
    "Mirounga angustirostris": SpeedCeiling(
        25.0, 15.0, "same genus as M. leonina; see that entry"
    ),
}


def haversine_m(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great circle distance between two sets of positions, metres."""

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def geodesic_speed_kmh(
    lat: np.ndarray, lon: np.ndarray, seconds: np.ndarray
) -> np.ndarray:
    """Great circle speed between consecutive positions, km/h. First element is nan."""

    metres = haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    speed = np.full(metres.shape, np.nan)
    np.divide(metres * 3.6, seconds, out=speed, where=seconds > 0)
    return np.insert(speed, 0, np.nan)

def ceiling_at_interval(
    seconds: np.ndarray, ceiling: SpeedCeiling, multiplier: float = 1.0
) -> np.ndarray:
    """Interpolate the permitted speed between instantaneous and sustained bounds."""
    seconds = np.asarray(seconds, dtype=float)
    lo, hi = np.log(INSTANT_INTERVAL_S), np.log(SUSTAINED_INTERVAL_S)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip((np.log(np.maximum(seconds, 1e-9)) - lo) / (hi - lo), 0.0, 1.0)
    bound = ceiling.instant_kmh + frac * (ceiling.sustained_kmh - ceiling.instant_kmh)
    return bound * multiplier


@dataclass
class QCReport:
    """QC counts and notes for one run; doesnt have to rerun once auditted."""

    rows_in: int = 0
    invalid_coord: int = 0
    invalid_time: int = 0
    null_island: int = 0
    out_of_range: int = 0
    missing_coord: int = 0
    duplicate_rows: int = 0
    duplicate_conflicting: int = 0
    before_deploy_on: int = 0
    after_deploy_off: int = 0
    out_of_species_range: int = 0
    species_bounds: str = ""
    over_ceiling: int = 0
    spikes: int = 0
    speed_outliers: int = 0
    ceiling_multiplier: float = 1.0
    deployments_with_on: int = 0
    deployments_with_off: int = 0
    deployments_total: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        d["notes"] = "; ".join(self.notes)
        return d


def flag_invalid_coordinates(
    frame: pd.DataFrame,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
    report: QCReport | None = None,
) -> pd.DataFrame:
    """Flag missing invalid range and Null Island (default value that a failed fix decays to) coordinates."""
    lat = pd.to_numeric(frame[lat_col], errors="coerce")
    lon = pd.to_numeric(frame[lon_col], errors="coerce")

    missing = lat.isna() | lon.isna()
    out_of_range = (lat.abs() > 90) | (lon.abs() > 180)
    null_island = (lat == 0) & (lon == 0)
    invalid = (missing | out_of_range | null_island).to_numpy()

    out = frame.copy()
    out["qc_invalid_coord"] = invalid
    if report is not None:
        report.missing_coord += int(missing.sum())
        report.out_of_range += int(out_of_range.sum())
        report.null_island += int(null_island.sum())
        report.invalid_coord += int(invalid.sum())
    return out


def flag_invalid_time(
    frame: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    report: QCReport | None = None,
) -> pd.DataFrame:
    """Flag fixes with missing or unusable timestamps. These rows cannot support ordering or speed calculations, so they are 
    explicitly flagged rather than being removed indirectly during duplicate handling."""
    out = frame.copy()
    times = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out["qc_invalid_time"] = times.isna().to_numpy()
    if report is not None:
        report.invalid_time += int(out["qc_invalid_time"].sum())
    return out


def flag_outside_species_range(
    frame: pd.DataFrame,
    species: str,
    *,
    lat_col: str = "lat",
    report: QCReport | None = None,
) -> pd.DataFrame:
    """Flag fixes outside a registered species latitude bound."""
    out = frame.copy()
    bounds = SPECIES_LATITUDE_BOUNDS.get(species)
    if bounds is None:
        out["qc_out_of_species_range"] = False
        if report is not None:
            report.notes.append(f"no latitude bound registered for {species}")
        return out

    lat = pd.to_numeric(out[lat_col], errors="coerce")
    low, high = bounds
    out["qc_out_of_species_range"] = ((lat < low) | (lat > high)).fillna(False).to_numpy()
    if report is not None:
        report.out_of_species_range += int(out["qc_out_of_species_range"].sum())
        report.species_bounds = f"{species}: {low} to {high}"
    return out


def resolve_duplicates(
    frame: pd.DataFrame,
    *,
    individual_col: str = "individual_id",
    time_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    quality_col: str | None = None,
    # best first; "G" is Fastloc; tens of metres against kilometres for every Argos class, so it heads the order
    quality_order: tuple[str, ...] = ("G", "3", "2", "1", "0", "A", "B", "Z"),
    error_col: str | None = None,
    report: QCReport | None = None,
) -> pd.DataFrame:
    """Resolve duplicate timestamps using quality/error ranking where available."""
    out = frame.copy()
    out["qc_duplicate"] = False
    keys = [individual_col, time_col]

    # GenAI reference: used to finsish the duplicate and ranking logic from her below. Initially I called duplicated(keep="first")
    # on (individual_id, timestamp), but this keeps the row that arrives first in file order rather than best fix and a repeat becomes
    # 2 disagrreing positions. GenAI suggested seperating the 2 cases, ranking by Argos quality as a tiebreak under a stable sort, and 
    # mapping flags back to original position so the returned frame keeps its input row order. 
    grp = out.groupby(keys, sort=False, dropna=False)
    sizes = grp[lat_col].transform("size").to_numpy()
    if not (sizes > 1).any():
        return out

    spread = (
        grp[lat_col].transform("max") - grp[lat_col].transform("min")
    ).abs().to_numpy() + (
        grp[lon_col].transform("max") - grp[lon_col].transform("min")
    ).abs().to_numpy()
    conflicting = (sizes > 1) & (spread > 0)

    rankable = quality_col is not None and quality_col in out.columns

    if rankable:
        order = {code: i for i, code in enumerate(quality_order)}
        rank = out[quality_col].astype(str).map(order).fillna(len(order)).to_numpy(dtype=float)
        tiebreak = (
            pd.to_numeric(out[error_col], errors="coerce").to_numpy(dtype=float)
            if error_col is not None and error_col in out.columns
            else np.zeros(len(out))
        )
        # `_lat`/`_lon` are the last resort, not evidence. Where class and error radius both
        # tie -- 72.7% of SOES conflicting groups, which carries no error column at all -- a
        # stable sort would otherwise settle the winner by file order, and file order is not a
        # property of the data: it is what cost us a split that differed between Mac and Gadi
        # under an identical seed. Ordering on the coordinates themselves makes the survivor a
        # pure function of (positions, class, error) and nothing else.
        ordering = pd.DataFrame(
            {
                "_ind": out[individual_col].to_numpy(),
                "_time": out[time_col].to_numpy(),
                "_rank": rank,
                "_tie": np.nan_to_num(tiebreak, nan=np.inf),
                "_lat": pd.to_numeric(out[lat_col], errors="coerce").to_numpy(dtype=float),
                "_lon": pd.to_numeric(out[lon_col], errors="coerce").to_numpy(dtype=float),
                "_pos": np.arange(len(out)),
            }
        ).sort_values(
            ["_ind", "_time", "_rank", "_tie", "_lat", "_lon"], kind="stable"
        )
        flags = np.zeros(len(out), dtype=bool)
        flags[ordering["_pos"].to_numpy()] = ordering.duplicated(
            subset=["_ind", "_time"], keep="first"
        ).to_numpy()
    else:
        flags = out.duplicated(subset=keys, keep="first").to_numpy() | conflicting

    out["qc_duplicate"] = flags
    if report is not None:
        report.duplicate_rows += int(flags.sum())
        report.duplicate_conflicting += int(
            out.loc[conflicting, keys].drop_duplicates().shape[0]
        )
        if not rankable and conflicting.any():
            report.notes.append(
                "conflicting duplicates dropped wholesale: no quality field to rank them"
            )
    return out
# finished here


def flag_outside_deployment(
    frame: pd.DataFrame,
    deployments: pd.DataFrame,
    *,
    deployment_col: str = "deployment_id",
    time_col: str = "timestamp",
    on_col: str = "deploy_on_timestamp",
    off_col: str = "deploy_off_timestamp",
    report: QCReport | None = None,
) -> pd.DataFrame:
    """Flag fixes outside known deployment start/end times."""
    out = frame.copy()
    out["qc_before_deploy_on"] = False
    out["qc_after_deploy_off"] = False
    if deployments.empty or deployment_col not in out.columns:
        if report is not None:
            report.notes.append("no deployment metadata; window checks not applied")
        return out

    windows = deployments.set_index(deployment_col)
    times = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    on = pd.to_datetime(
        out[deployment_col].map(windows[on_col]) if on_col in windows else pd.NaT,
        utc=True,
        errors="coerce",
    )
    off = pd.to_datetime(
        out[deployment_col].map(windows[off_col]) if off_col in windows else pd.NaT,
        utc=True,
        errors="coerce",
    )

    out["qc_before_deploy_on"] = (times < on).fillna(False).to_numpy()
    out["qc_after_deploy_off"] = (times > off).fillna(False).to_numpy()

    if report is not None:
        report.before_deploy_on += int(out["qc_before_deploy_on"].sum())
        report.after_deploy_off += int(out["qc_after_deploy_off"].sum())
        report.deployments_total += len(windows)
        if on_col in windows:
            report.deployments_with_on += int(windows[on_col].notna().sum())
        if off_col in windows:
            report.deployments_with_off += int(windows[off_col].notna().sum())
    return out


def flag_speed(
    frame: pd.DataFrame,
    ceiling: SpeedCeiling,
    *,
    multiplier: float = 1.0,
    robust_k: float | None = 6.0,
    individual_col: str = "individual_id",
    time_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
    report: QCReport | None = None, 
) -> pd.DataFrame:
    """Flag interval ceiling breaches, out and back spikes and robust speed outliers."""
    out = frame.copy()
    # GenAI reference: speed, spike and robust outlier casing here was completed from my initial attempt by GenAI. At the start I applied 
    # one fixed kmg celing to every gap; which failed bc ceiling is scale dependent, so it passed everything at 6h hour gaps and flagged
    # ordinary flight across short ones and some other issues. GenAI suggested (and them did) interpolating the bound in log interval 
    # between the instantaneous and sustained numbers, and so detecting spikes by comparing net displacement across the pair against the 
    # summed trip, and added median/MAD outlier test on log speed per individual. From below to... 
    order = out.sort_values([individual_col, time_col], kind="stable").index
    block = out.loc[order]

    lat = pd.to_numeric(block[lat_col], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(block[lon_col], errors="coerce").to_numpy(dtype=float)
    times = pd.to_datetime(block[time_col], utc=True, errors="coerce")

    seconds = times.diff().dt.total_seconds().to_numpy(dtype=float)

    same = np.concatenate([[False], block[individual_col].to_numpy()[1:] == block[individual_col].to_numpy()[:-1]])
    speed = geodesic_speed_kmh(lat, lon, seconds[1:])
    speed = np.where(same, speed, np.nan)

    bound = ceiling_at_interval(seconds, ceiling, multiplier)
    over = np.greater(speed, bound, where=np.isfinite(speed), out=np.zeros_like(bound, dtype=bool))

    over_next = np.concatenate([over[1:], [False]])
    net_km = np.full(len(block), np.nan)
    if len(block) > 2:
        phi1, phi2 = np.radians(lat[:-2]), np.radians(lat[2:])
        dlam = np.radians(lon[2:] - lon[:-2])
        h = np.sin((phi2 - phi1) / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        net_km[1:-1] = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1))) / 1000.0
    out_km = np.where(np.isfinite(speed), speed * seconds / 3600.0, np.nan)
    excursion = out_km + np.concatenate([out_km[1:], [np.nan]])
    spike = over & over_next & (net_km < 0.25 * excursion)
    spike = np.nan_to_num(spike, nan=False).astype(bool)

    outlier = np.zeros(len(block), dtype=bool)
    if robust_k is not None:
        logs = pd.Series(np.log(np.where(speed > 0, speed, np.nan)))
        grp = pd.Series(block[individual_col].to_numpy())
        med = logs.groupby(grp).transform("median")
        mad = (logs - med).abs().groupby(grp).transform("median")
        outlier = ((logs - med) > robust_k * mad.replace(0, np.nan)).fillna(False).to_numpy()

    flags = pd.DataFrame(
        {
            "qc_over_ceiling": over,
            "qc_spike": spike,
            "qc_speed_outlier": outlier,
            "qc_speed_kmh": speed,
        },
        index=block.index,
    )
    out = out.join(flags)

    if report is not None:
        report.over_ceiling += int(over.sum())
        report.spikes += int(spike.sum())
        report.speed_outliers += int(outlier.sum())
        report.ceiling_multiplier = multiplier
        report.notes.append(f"speed ceiling: {ceiling.source}")
    return out
# here

def structural_qc(
    frame: pd.DataFrame,
    *,
    species: str | None = None,
    deployments: pd.DataFrame | None = None,
    quality_col: str | None = None,
    error_col: str | None = None,
    **columns: str,
) -> tuple[pd.DataFrame, QCReport]:
    """Run structural checks and return the flagged frame plus QC report.

    Nothing is removed. `qc_invalid_coord` and `qc_invalid_time` are the two flags we recommend dropping unconditionally: neither marks a 
    low quality observation, both mark s row that is not an observation.
    """
    report = QCReport(rows_in=len(frame))
    lat_col = columns.get("lat_col", "lat")
    lon_col = columns.get("lon_col", "lon")

    out = flag_invalid_coordinates(frame, lat_col=lat_col, lon_col=lon_col, report=report)
    out = flag_invalid_time(
        out, report=report, **{k: v for k, v in columns.items() if k == "time_col"}
    )
    if species is not None:
        out = flag_outside_species_range(out, species, lat_col=lat_col, report=report)

    # GenAI reference: from here to below since its version filters kwargs to the keys each function accepts at each call site
    out = resolve_duplicates(
        out,
        lat_col=lat_col,
        lon_col=lon_col,
        quality_col=quality_col,
        error_col=error_col,
        report=report,
        **{k: v for k, v in columns.items() if k in {"individual_col", "time_col"}},
    )
    if deployments is not None:
        out = flag_outside_deployment(
            out,
            deployments,
            report=report,
            **{k: v for k, v in columns.items() if k in {"deployment_col", "time_col"}},
        )
    # to here
    return out, report
