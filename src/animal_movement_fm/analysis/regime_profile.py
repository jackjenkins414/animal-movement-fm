
# Reference: This is a very large file, and has taken me ~1 week working on; I had an initial implementation, but it was
# considerably improved upon by GenAI. I have added the full details at the bottom of this file to avoid clutter here at
# the top of the file.
"""
Movement regime descriptors, and the low displacement threshold used to split Tier 2 metrics.

We define here the shared meaning of a low displacement step. We reuse the same definition when selecting
species (`experiments/data/regime_profile.py`) and when reporting Tier 2 evaluation results, so it lives
here rather than being duplicated across experiment scripts.

Naming
------
The two strata we use are low displacement and travel, not "stationary" and "travelling". A step length
threshold cannot support a behavioural claim, and for Argos data short steps are possibly observation error
(Argos has more error just generally). See preregistration amendment A4.

Recall we previously defined "stationary" elsewhere, where there we refer to the Tier 1 movement state from
`analysis/regime.py`, which is a different quantity built from the 60 m hysteresis cut and white stork constants.
This module deals only with the step length boundary, and uses the low displacement names throughout.

The boundary
------------

The preregistered operational default is 10% of the median one-bin step length among a species' training individuals.
We do this species relative boundary to stratify Tier 2 prediction errors; it is not assumed to identify a biological
state. The 10% multiplier is an analysis choice and is not itself tested; what the registered diagnostics check is
whether the anchoring quantile is safe.

Only training individuals are used, because the boundary decides which test windows are reported as travel regime.
Deriving it from test data would place a data dependent reporting choice inside our sealed evaluation. Only steps
spanning exactly one modelling bin are counted, because resampled segments deliberately tolerate internal gaps of up
to 8 bins. A step crossing one of those covers 12 to 48 hours rather than 6, so it is systematically longer than a step
at the cadence. Including them increased the stork median from 609.5 m to 625.9 m and the resulting boundary from 60.9m
to 62.6 m. See amendment A1.

The median anchor can fail when most steps fall within the low displacement mode. Because the problem is the anchor rather
than the multiplier, `select_threshold_rule` applies our amendment A4 from p50 to p75 and then p90 when the registered
guards fire. The selected quantile is universal across species; no species receives a bespoke formula. Mixture crossings
are used only to test the anchors and are never used directly as evaluation boundaries.

Displacements use a local equirectangular approximation; see `local_en_metres`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from animal_movement_fm.data.resample import BIN_SECONDS, resample_trajectory

# metres per degree; see the geodesy note on local_en_metres
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0

# boundary = LOW_DISPLACEMENT_FRACTION x p50 (one bin steps, training individuals)
LOW_DISPLACEMENT_FRACTION = 0.10


LOW_FRACTION_GUARD = 0.45  # >= this fraction below the p50 anchor trips the fallback
MIN_MIXTURE_STEPS = 100  # fewer one bin steps than this and a mixture is not supported
MIN_MIXTURE_COMPONENT_WEIGHT = 0.05  # each component must carry at least this weight
MIN_MIXTURE_BIC_IMPROVEMENT = 10.0  # two components must beat one by at least this much BIC
MIXTURE_RANDOM_STATE = 42
LOG_STEP_FLOOR_M = 1e-3  # zero length steps are floored before taking logs
STORK_REFERENCE_SPECIES = "white_stork"
STORK_REFERENCE_BOUNDARY_M = 60.9  # the A1 figure; fallback anchors are calibrated to preserve it

MIN_TURNS_FOR_RHO = 100  # fewer qualifying turn pairs and rho is not reported
MIN_STEPS_PER_AMPLITUDE_BIN = 30  # hour/month bins below this are dropped before amplitude

DESCRIPTORS = [
    "log10_step_median_m",
    "log10_step_p90_m",
    "tail_ratio",
    "step_cv",
    "intermittency",
    "rho",
    "diel_amplitude",
    "seasonal_amplitude",
]

# transform descriptors according to their measurement scale before standardisation; positive ratio scale variables use 
# log10, proportions use logit, and bounded correlations stay on their original scale
DESCRIPTOR_TRANSFORMS: dict[str, str] = {
    "log10_step_median_m": "identity",  # already logged
    "log10_step_p90_m": "identity",
    "tail_ratio": "log10",
    "step_cv": "log10",
    "intermittency": "logit",
    "rho": "identity",
    "diel_amplitude": "log10",
    "seasonal_amplitude": "log10",
}

# avoid undefined logits at exactly 0 or 1
LOGIT_EPSILON = 1e-6


def transform_descriptors(
    values: pd.DataFrame,
    transforms: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Transform descriptor columns onto their registered measurement scales.

    The returned frame keeps the same index and column order as `values`. Columns default to `DESCRIPTOR_TRANSFORMS`, with 
    unspecified descriptors left unchanged.
    """
    transforms = dict(transforms or DESCRIPTOR_TRANSFORMS)
    transformed = {}

    for column in values.columns:
        transform = transforms.get(column, "identity")
        series = values[column].astype(float)

        if transform == "identity":
            transformed[column] = series

        elif transform == "log10":
            invalid = series <= 0
            if invalid.any():
                offenders = values.index[invalid].tolist()
                raise ValueError(
                    f"{column} is registered for log10 transformation, but rows "
                    f"{offenders} contain non-positive values"
                )
            transformed[column] = np.log10(series)

        elif transform == "logit":
            clipped = series.clip(
                LOGIT_EPSILON,
                1.0 - LOGIT_EPSILON,
            )
            transformed[column] = np.log(clipped / (1.0 - clipped))

        else:
            raise ValueError(
                f"unknown descriptor transform {transform!r} for {column}"
            )

    return pd.DataFrame(
        transformed,
        index=values.index,
    )[list(values.columns)]
REQUIRED_COLUMNS = {"individual_id", "timestamp", "lat", "lon"}


def local_en_metres(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return consecutive displacements in local east-north metres.

    Longitude is scaled by the midpoint latitude so that geographic distortion is not mistaken for differences in
    movement. The equirectangular approximation is sufficient for these descriptors.
    """

    mid_lat = np.radians((lat[1:] + lat[:-1]) / 2.0)
    delta_lon = (np.diff(lon) + 180.0) % 360.0 - 180.0
    east = delta_lon * M_PER_DEG_LON_EQ * np.cos(mid_lat)
    north = np.diff(lat) * M_PER_DEG_LAT
    return east, north

def normalise_individual_ids(values) -> np.ndarray:
    """
    Normalise individual ids to comparable strings.

    Movebank stores `individual_id` as an integer, but parquet round trips can widen it to float, and RAATD uses a
    string id such as "CAR2013_sel_a_m_03". Comparing on the raw values therefore fails in one direction or the
    other, and the previous integer only coercion silently produced an empty frame for every RAATD species, i.e.
    the entire marine leg.

    Integral floats are collapsed to their integer form so 12345, 12345.0 and "12345" all match, and everything else
    is compared as a stripped string. Non integral floats keep their decimal form rather than being truncated, so
    two distinct ids can never be merged.
    """

    series = pd.Series(np.asarray(values, dtype=object).ravel())

    numeric = pd.to_numeric(series, errors="coerce")
    integral = numeric.notna() & np.isclose(numeric % 1.0, 0.0)

    out = series.astype(str).str.strip()
    out[integral] = numeric[integral].astype("int64").astype(str)
    out[numeric.notna() & ~integral] = numeric[numeric.notna() & ~integral].astype(str)
    return out.to_numpy()


def restrict_to_individuals(frame: pd.DataFrame, individual_ids) -> pd.DataFrame:
    """
    Keep rows belonging to the specified individuals, matching ids across storage types.
    """

    wanted = set(normalise_individual_ids(individual_ids))
    if not wanted:
        raise ValueError("no individual ids supplied")

    ids = normalise_individual_ids(frame["individual_id"].to_numpy())
    kept = frame[np.isin(ids, list(wanted))]
    if kept.empty:
        available = pd.unique(ids)[:3]
        raise ValueError(
            "restricting to the supplied individuals left no rows. The manifest ids and the frame's "
            f"individual_id column do not correspond: wanted e.g. {sorted(wanted)[:3]}, "
            f"frame has e.g. {list(available)}"
        )
    return kept


def train_individuals(split) -> np.ndarray:
    """
    Training individuals of a frozen `Split`.

    The registered boundary is defined on training individuals only, so this exists to make the correct choice the
    obvious one at every call site.
    """

    return np.asarray(split.train)

@dataclass(frozen=True)
class CadenceSteps:
    """
    Steps spanning exactly one modelling interval.

    The displacement arrays are aligned, one value per retained step, and `hour`/`month` refer to the destination fix.
    `turn` is shorter: it holds one value per consecutive pair of clearly moving retained steps, so it is not aligned
    with the others by design.
    """

    east_m: np.ndarray
    north_m: np.ndarray
    mag_m: np.ndarray
    hour: np.ndarray
    month: np.ndarray
    turn: np.ndarray

    def __len__(self) -> int:
        return len(self.mag_m)


def ensure_resampled(
    frame: pd.DataFrame,
    *,
    bin_seconds: int = BIN_SECONDS,
    group_col: str = "segment_id",
) -> pd.DataFrame:
    """
    Return a frame carrying a `bin_index`.

    Frames that already have one are returned unchanged. Otherwise raw fixes are resampled per individual with the
    pipeline's own resampler, so a candidate species can be profiled before a full processed dataset has been built
    for it.
    """

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")

    if "bin_index" in frame.columns:
        if group_col not in frame.columns:
            raise ValueError(f"resampled frame must carry grouping column {group_col!r}")
        return frame

    rows = []
    ordered = frame.sort_values(["individual_id", "timestamp"])
    for individual_id, track in ordered.groupby("individual_id", sort=False):
        segments = resample_trajectory(
            pd.to_datetime(track["timestamp"]).to_numpy(),
            track["lat"].to_numpy(dtype=float),
            track["lon"].to_numpy(dtype=float),
            bin_seconds=bin_seconds,
        )
        for s, seg in enumerate(segments):
            rows.append(
                pd.DataFrame(
                    {
                        "segment_id": f"{individual_id}_{s}",
                        "individual_id": individual_id,
                        "bin_index": seg.bin_index,
                        "timestamp": seg.timestamp,
                        "lat": seg.lat,
                        "lon": seg.lon,
                        "centre_offset_s": seg.centre_offset_s,
                    }
                )
            )

    if not rows:
        raise ValueError("resampling produced no segments; check cadence and input tracks..")
    return pd.concat(rows, ignore_index=True)


def cadence_steps(
    frame: pd.DataFrame,
    *,
    group_col: str = "segment_id",
    moving_threshold_m: float | None = None,
) -> CadenceSteps:
    """
    Extract one bin steps, and turning angles between consecutive moving steps.

    Only steps within the same segment and exactly one bin apart are retained, so every species is compared at the
    same sampling interval. Turning angles are computed only when both steps exceed `moving_threshold_m` (default:
    the median retained step), because turning angle is singular at near zero steps and GPS error alone
    manufactures spurious reversals there.
    """

    if group_col not in frame.columns:
        raise ValueError(f"frame must carry the grouping column {group_col!r}")
    if "bin_index" not in frame.columns:
        raise ValueError("frame must be resampled first; call ensure_resampled()")

    frame = frame.sort_values([group_col, "bin_index"]).reset_index(drop=True)
    if len(frame) < 2:
        empty = np.empty(0, dtype=float)
        return CadenceSteps(empty, empty, empty, empty.astype(int), empty.astype(int), empty)

    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)
    bin_index = frame["bin_index"].to_numpy()
    group = frame[group_col].to_numpy()
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))

    # a step is retained only if both fixes are in the same segment AND exactly one bin apart
    same_group = group[1:] == group[:-1]
    one_bin = np.diff(bin_index) == 1
    finite_position = (
        np.isfinite(lon[:-1]) & np.isfinite(lon[1:]) & np.isfinite(lat[:-1]) & np.isfinite(lat[1:])
    )
    keep = same_group & one_bin & finite_position

    east, north = local_en_metres(lon, lat)
    mag = np.hypot(east, north)

    # turning angle needs two consecutive retained steps sharing their middle fix
    bearing = np.arctan2(east, north)
    if moving_threshold_m is None:
        moving_threshold_m = float(np.median(mag[keep])) if keep.any() else 0.0
    moving = mag >= moving_threshold_m
    pair = keep[:-1] & keep[1:] & moving[:-1] & moving[1:]
    if pair.any():
        delta = bearing[1:][pair] - bearing[:-1][pair]
        turn = (delta + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi)
    else:
        turn = np.empty(0, dtype=float)

    dest = stamps[1:][keep]
    return CadenceSteps(
        east_m=east[keep],
        north_m=north[keep],
        mag_m=mag[keep],
        hour=dest.hour.to_numpy(),
        month=dest.month.to_numpy(),
        turn=turn,
    )

def median_step_m(
    frame: pd.DataFrame,
    *,
    individual_ids=None,
    group_col: str = "segment_id",
    bin_seconds: int = BIN_SECONDS,
) -> float:
    """
    Median one bin step length in metres, optionally restricted to given individuals.
    """

    frame = ensure_resampled(frame, bin_seconds=bin_seconds, group_col=group_col)
    if individual_ids is not None:
        frame = restrict_to_individuals(frame, individual_ids)
    steps = cadence_steps(frame, group_col=group_col)
    if len(steps) == 0:
        raise ValueError(
            "no steps span exactly one bin; the tracks may be too gappy for this cadence"
        )
    return float(np.median(steps.mag_m))


def low_displacement_threshold_m(
    frame: pd.DataFrame,
    *,
    individual_ids,
    group_col: str = "segment_id",
    bin_seconds: int = BIN_SECONDS,
) -> float:
    """
    The registered default boundary for one species, in metres.

    10% of the median one bin step length over that species' training individuals. If the A4 guards fire,
    `select_threshold_rule` supplies the anchor instead and this function is not the production rule.
    """

    return LOW_DISPLACEMENT_FRACTION * median_step_m(
        frame,
        individual_ids=individual_ids,
        group_col=group_col,
        bin_seconds=bin_seconds,
    )

def is_low_displacement(step_magnitudes_m: np.ndarray, threshold_m: float) -> np.ndarray:
    """
    Low displacement mask; equality belongs to the travel regime ("below" is strict).
    """

    return np.asarray(step_magnitudes_m, dtype=float) < threshold_m

def directional_persistence_from_turns(turn_angles: np.ndarray) -> float:
    """
    Mean cosine of turning angles; matches our rho in ``analysis.randomness``.
    """

    turns = np.asarray(turn_angles, dtype=float)
    turns = turns[np.isfinite(turns)]
    return float(np.cos(turns).mean()) if turns.size else np.nan

@dataclass(frozen=True)
class MixtureDiagnostics:
    """
    Two component lognormal fit to one bin step lengths.

    This is a diagnostic. Its crossing is never used as an evaluation boundary; it only decides whether a quantile anchor
    has fallen inside a low displacement mode. `supported` encodes every adequacy rule registered in A4, so callers must
    never inspect the crossing without checking it.
    """

    supported: bool
    reason: str
    n_steps: int
    n_floor_clipped: int
    crossing_m: float = np.nan
    crossing_fraction: float = np.nan
    bic_improvement: float = np.nan
    weights: tuple[float, ...] = field(default=())
    means_log: tuple[float, ...] = field(default=())
    sds_log: tuple[float, ...] = field(default=())


def _unsupported_mixture(n_steps: int, n_floor_clipped: int, reason: str) -> MixtureDiagnostics:
    return MixtureDiagnostics(
        supported=False,
        reason=reason,
        n_steps=int(n_steps),
        n_floor_clipped=int(n_floor_clipped),
    )


def _between_means_crossing(
    weights: np.ndarray, means: np.ndarray, sds: np.ndarray
) -> float | None:
    """
    Log space point where the two weighted component densities are equal.

    Solves ``w1 N(x; m1, s1) == w2 N(x; m2, s2)`` and returns the root strictly between the two means, or None if
    there is none there. Requiring the root to lie between the means is what makes this a genuine crossing rather
    than an artefact of one component's tail overtaking the other far from both modes.
    """

    (w1, w2), (m1, m2), (s1, s2) = weights, means, sds
    lo, hi = (m1, m2) if m1 <= m2 else (m2, m1)

    a = 1.0 / (2.0 * s2**2) - 1.0 / (2.0 * s1**2)
    b = m1 / s1**2 - m2 / s2**2
    c = m2**2 / (2.0 * s2**2) - m1**2 / (2.0 * s1**2) + np.log(w1 / w2) + np.log(s2 / s1)

    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return None
        roots = np.array([-c / b])
    else:
        disc = b**2 - 4.0 * a * c
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        roots = np.array([(-b + sqrt_disc) / (2.0 * a), (-b - sqrt_disc) / (2.0 * a)])

    inside = roots[(roots > lo) & (roots < hi)]
    if inside.size == 0:
        return None
    return float(inside[0])


def fit_step_mixture(step_magnitudes_m: np.ndarray) -> MixtureDiagnostics:
    """
    Fit a two component lognormal mixture to step lengths and apply our A4 adequacy rules.

    A mixture is supported only if there are at least `MIN_MIXTURE_STEPS` steps, the fit converged, both components
    carry at least `MIN_MIXTURE_COMPONENT_WEIGHT`, a density crossing exists strictly between the component means, and
    two components beat one by at least `MIN_MIXTURE_BIC_IMPROVEMENT` BIC. Anything else returns `supported=False` with
    a reason, and an unsupported mixture is not evidence in either direction.
    """

    from sklearn.mixture import GaussianMixture

    mag = np.asarray(step_magnitudes_m, dtype=float)
    mag = mag[np.isfinite(mag)]
    n_floor_clipped = int((mag < LOG_STEP_FLOOR_M).sum())
    mag = np.maximum(mag, LOG_STEP_FLOOR_M)
    n_steps = int(mag.size)

    if n_steps < MIN_MIXTURE_STEPS:
        return _unsupported_mixture(
            n_steps, n_floor_clipped, f"fewer than {MIN_MIXTURE_STEPS} one bin steps"
        )

    x = np.log(mag).reshape(-1, 1)

    two = GaussianMixture(n_components=2, random_state=MIXTURE_RANDOM_STATE, n_init=5)
    two.fit(x)
    one = GaussianMixture(n_components=1, random_state=MIXTURE_RANDOM_STATE)
    one.fit(x)

    bic_improvement = float(one.bic(x) - two.bic(x))

    if not two.converged_:
        return _unsupported_mixture(n_steps, n_floor_clipped, "two component fit did not converge")

    weights = np.asarray(two.weights_, dtype=float).ravel()
    means = np.asarray(two.means_, dtype=float).ravel()
    sds = np.sqrt(np.asarray(two.covariances_, dtype=float).ravel())

    order = np.argsort(means)
    weights, means, sds = weights[order], means[order], sds[order]

    if weights.min() < MIN_MIXTURE_COMPONENT_WEIGHT:
        return _unsupported_mixture(
            n_steps,
            n_floor_clipped,
            f"a component carries less than {MIN_MIXTURE_COMPONENT_WEIGHT:.0%} weight",
        )
    if bic_improvement < MIN_MIXTURE_BIC_IMPROVEMENT:
        return _unsupported_mixture(
            n_steps,
            n_floor_clipped,
            f"BIC improvement {bic_improvement:.1f} below {MIN_MIXTURE_BIC_IMPROVEMENT}",
        )

    crossing_log = _between_means_crossing(weights, means, sds)
    if crossing_log is None:
        return _unsupported_mixture(
            n_steps, n_floor_clipped, "no density crossing between the component means"
        )

    crossing_m = float(np.exp(crossing_log))
    return MixtureDiagnostics(
        supported=True,
        reason="supported",
        n_steps=n_steps,
        n_floor_clipped=n_floor_clipped,
        crossing_m=crossing_m,
        crossing_fraction=float((mag < crossing_m).mean()),
        bic_improvement=bic_improvement,
        weights=tuple(float(w) for w in weights),
        means_log=tuple(float(m) for m in means),
        sds_log=tuple(float(s) for s in sds),
    )


def _anchor_inside_low_component(anchor_m: float, mixture: MixtureDiagnostics) -> bool:
    """
    True when a supported mixture places this anchor inside its low displacement component.
    """

    if not mixture.supported or not np.isfinite(mixture.crossing_m):
        return False
    return bool(anchor_m <= mixture.crossing_m)


def profile_species(
    frame: pd.DataFrame,
    *,
    individual_ids=None,
    group_col: str = "segment_id",
    bin_seconds: int = BIN_SECONDS,
    min_turns_for_rho: int = MIN_TURNS_FOR_RHO,
    min_steps_per_amplitude_bin: int = MIN_STEPS_PER_AMPLITUDE_BIN,
) -> dict:
    """
    Summarise one species with the descriptors used for selection, plus the A4 decision inputs.

    The same descriptors are computed for every species so movement patterns can be compared directly. Taxonomic domain
    enters only at final selection. Nothing here fits a forecasting model or touches a prediction target, so this remains
    dataset property exploration under our use of Hofman et al.
    """

    resampled = ensure_resampled(frame, bin_seconds=bin_seconds, group_col=group_col)
    # None means every individual in the frame, matching `median_step_m`. The CLI uses this to
    # profile a candidate species that has no frozen split yet, before committing to a full ingest;
    # such a row is marked threshold_decision_ready=False and cannot feed the A4 decision.
    scoped = resampled if individual_ids is None else restrict_to_individuals(resampled, individual_ids)

    steps = cadence_steps(scoped, group_col=group_col)
    if len(steps) == 0:
        raise ValueError("no steps span exactly one bin; nothing to profile at this cadence")

    mag = steps.mag_m
    p50, p75, p90 = (float(value) for value in np.percentile(mag, [50, 75, 90]))
    mean = float(mag.mean())

    boundary = LOW_DISPLACEMENT_FRACTION * p50
    low_fraction = float(is_low_displacement(mag, boundary).mean())

    # scale free measure of how often the species is in low displacement relative to its typical step
    intermittency = low_fraction if p50 > 0 else np.nan

    # directional persistence: 1 straight, 0 random, -1 repeated reversals. Reported only with enough qualifying pairs, because
    # a rho from a handful of turns is noise that would otherwise be imputed to the mean in descriptor space and silently move
    # a selection decision.
    n_turns = int(steps.turn.size)
    rho = directional_persistence_from_turns(steps.turn) if n_turns >= min_turns_for_rho else np.nan

    def amplitude(keys: np.ndarray) -> float:
        """
        Range of mean step length across temporal bins, relative to the overall mean.

        Bins below `min_steps_per_amplitude_bin` are dropped first: max minus min is a statistic of the extremes, so a single
        thinly populated hour or month would otherwise dominate, and candidate species differ enormously in sample size.
        """

        grouped = pd.Series(mag).groupby(keys)
        counts = grouped.size()
        means = grouped.mean()[counts >= min_steps_per_amplitude_bin]
        if mean <= 0 or len(means) < 2:
            return np.nan
        return float((means.max() - means.min()) / mean)

    mixture = fit_step_mixture(mag)

    return {
        "n_rows": int(len(scoped)),
        "n_individuals": int(pd.unique(normalise_individual_ids(scoped["individual_id"])).size),
        "n_steps_at_cadence": int(mag.size),
        "n_turns_used": n_turns,
        # selection descriptors
        "log10_step_median_m": float(np.log10(max(p50, 1e-6))),
        "log10_step_p90_m": float(np.log10(max(p90, 1e-6))),
        "tail_ratio": float(np.percentile(mag, 99) / p50) if p50 > 0 else np.nan,
        "step_cv": float(mag.std() / mean) if mean > 0 else np.nan,
        "intermittency": intermittency,
        "rho": rho,
        "diel_amplitude": amplitude(steps.hour),
        "seasonal_amplitude": amplitude(steps.month),
        # --- A4 decision inputs ---
        "step_p50_m": p50,
        "step_p75_m": p75,
        "step_p90_m": p90,
        "median_anchor_boundary_m": boundary,
        "median_anchor_low_fraction": low_fraction,
        "mixture_supported": mixture.supported,
        "mixture_failure_reason": mixture.reason,
        "mixture_crossing_m": mixture.crossing_m,
        "mixture_crossing_fraction": mixture.crossing_fraction,
        "mixture_bic_improvement": mixture.bic_improvement,
        "p50_inside_low_component": _anchor_inside_low_component(p50, mixture),
        "p75_inside_low_component": _anchor_inside_low_component(p75, mixture),
        "p90_inside_low_component": _anchor_inside_low_component(p90, mixture),
        # the multiplier is emitted so results/regime_profiles.csv records which constant produced
        # the boundary, rather than leaving it implicit in the code version that wrote the row
        "median_anchor_multiplier": LOW_DISPLACEMENT_FRACTION,
    }


@dataclass(frozen=True)
class ThresholdDecision:
    """The universal boundary formula selected by `select_threshold_rule`."""

    anchor_quantile: int
    multiplier: float
    reference_species: str
    reference_boundary_m: float
    pooled_only_species: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def rule_name(self) -> str:
        return f"{self.multiplier:.6g} x p{self.anchor_quantile}"


def _profile_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return isinstance(value, str) and value.strip().lower() == "true"


def _mixture_guard(table: pd.DataFrame, quantile: int) -> pd.Series:
    supported = table["mixture_supported"].map(_profile_bool)
    anchor = pd.to_numeric(table[f"step_p{quantile}_m"], errors="coerce")
    crossing = pd.to_numeric(table["mixture_crossing_m"], errors="coerce")
    return supported & (anchor <= crossing)


def select_threshold_rule(
    table: pd.DataFrame,
    *,
    reference_species: str = STORK_REFERENCE_SPECIES,
    reference_boundary_m: float = STORK_REFERENCE_BOUNDARY_M,
) -> ThresholdDecision:
    """
    Apply the threshold branch registered in preregistration amendment A4.

    `table` must hold one row per mechanically eligible species, profiled on training individuals only. The chosen quantile
    and multiplier are universal. The only species level outcome is the registered terminal branch: a species whose
    supported low component still contains p90 is reported pooled rather than being given a bespoke threshold.
    """

    required = {
        "species",
        "step_p50_m",
        "step_p75_m",
        "step_p90_m",
        "median_anchor_low_fraction",
        "mixture_supported",
        "mixture_crossing_m",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"profile table is missing required columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("profile table is empty")
    if table["species"].duplicated().any():
        duplicates = sorted(
            table.loc[table["species"].duplicated(), "species"].astype(str).unique()
        )
        raise ValueError(f"profile table contains duplicate species: {duplicates}")

    fractions = pd.to_numeric(table["median_anchor_low_fraction"], errors="coerce")
    if fractions.isna().any():
        raise ValueError("median_anchor_low_fraction must be finite for every species")
    if ((fractions < 0) | (fractions > 1)).any():
        raise ValueError("median_anchor_low_fraction must lie between zero and one")

    anchors = table[["step_p50_m", "step_p75_m", "step_p90_m"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(anchors.to_numpy()).all() or (anchors <= 0).any().any():
        raise ValueError("all quantile anchors must be finite and positive")
    if (
        (anchors["step_p50_m"] > anchors["step_p75_m"])
        | (anchors["step_p75_m"] > anchors["step_p90_m"])
    ).any():
        raise ValueError("quantile anchors must be ordered p50 <= p75 <= p90")

    supported = table["mixture_supported"].map(_profile_bool)
    supported_crossings = pd.to_numeric(table.loc[supported, "mixture_crossing_m"], errors="coerce")
    if not np.isfinite(supported_crossings.to_numpy()).all() or (supported_crossings <= 0).any():
        raise ValueError("a supported mixture must have a finite positive crossing")

    reasons: list[str] = []
    high_fraction = fractions >= LOW_FRACTION_GUARD
    for species in table.loc[high_fraction, "species"].astype(str):
        reasons.append(f"{species}: median-anchor low fraction >= {LOW_FRACTION_GUARD:.0%}")

    p50_inside = _mixture_guard(table, 50)
    for species in table.loc[p50_inside, "species"].astype(str):
        reasons.append(f"{species}: p50 inside supported low component")

    # branch 3: both p50 guards clear, so the registered default stands for everyone
    if not high_fraction.any() and not p50_inside.any():
        return ThresholdDecision(
            anchor_quantile=50,
            multiplier=LOW_DISPLACEMENT_FRACTION,
            reference_species=reference_species,
            reference_boundary_m=reference_boundary_m,
            pooled_only_species=(),
            reasons=("all species passed the median-anchor guards",),
        )

    reference = table.loc[table["species"].astype(str) == reference_species]
    if len(reference) != 1:
        raise ValueError(
            f"fallback calibration requires exactly one {reference_species!r} profile row"
        )

    # branch 4: move every species to a stork-calibrated p75 anchor
    p75_inside = _mixture_guard(table, 75)
    if not p75_inside.any():
        reference_anchor = float(reference["step_p75_m"].iloc[0])
        if not np.isfinite(reference_anchor) or reference_anchor <= 0:
            raise ValueError("reference p75 must be finite and positive")
        return ThresholdDecision(
            anchor_quantile=75,
            multiplier=reference_boundary_m / reference_anchor,
            reference_species=reference_species,
            reference_boundary_m=reference_boundary_m,
            pooled_only_species=(),
            reasons=tuple(reasons),
        )

    for species in table.loc[p75_inside, "species"].astype(str):
        reasons.append(f"{species}: p75 inside supported low component")

    # branches 5 and 6: p90 anchor for everyone, and anything still inside is pooled only
    reference_anchor = float(reference["step_p90_m"].iloc[0])
    if not np.isfinite(reference_anchor) or reference_anchor <= 0:
        raise ValueError("reference p90 must be finite and positive")
    p90_inside = _mixture_guard(table, 90)
    pooled_only = tuple(table.loc[p90_inside, "species"].astype(str))
    for species in pooled_only:
        reasons.append(f"{species}: p90 inside supported low component; pooled only")

    return ThresholdDecision(
        anchor_quantile=90,
        multiplier=reference_boundary_m / reference_anchor,
        reference_species=reference_species,
        reference_boundary_m=reference_boundary_m,
        pooled_only_species=pooled_only,
        reasons=tuple(reasons),
    )


def apply_threshold_decision(
    table: pd.DataFrame, decision: ThresholdDecision
) -> pd.DataFrame:
    """Annotate profile rows with the selected universal formula and per species boundary."""

    anchor_col = f"step_p{decision.anchor_quantile}_m"
    if anchor_col not in table.columns:
        raise ValueError(f"profile table is missing {anchor_col!r}")
    result = table.copy()
    result["selected_anchor_quantile"] = decision.anchor_quantile
    result["selected_multiplier"] = decision.multiplier
    result["selected_boundary_m"] = (
        pd.to_numeric(result[anchor_col], errors="coerce") * decision.multiplier
    )
    result["pooled_only"] = result["species"].astype(str).isin(decision.pooled_only_species)
    return result


def descriptor_standardisation(table: pd.DataFrame, *, descriptors=None) -> dict:
    """
    Centre and scale for descriptor space, computed once over a fixed set of species.

    Standardising against whatever rows happen to be in the CSV would make a pair's distance change every time an
    unrelated candidate is profiled. Secondary prediction 2 uses regime distance as the explanatory variable for
    uneven transfer, so the distances have to be a fixed measurement: compute this once over the final eligible set,
    save it, and pass it back in thereafter.
    """

    descriptors = list(descriptors or DESCRIPTORS)
    transforms = {d: DESCRIPTOR_TRANSFORMS.get(d, "identity") for d in descriptors}
    values = transform_descriptors(table[descriptors].astype(float), transforms)
    scale = values.std(ddof=0).replace(0, 1.0)
    return {
        "descriptors": descriptors,
        # Recorded in the frozen payload, not just applied, so a distance can never be reproduced
        # against a different transform than the one it was measured under.
        "transforms": transforms,
        "centre": values.mean().to_dict(),
        "scale": scale.to_dict(),
        "species": table["species"].astype(str).tolist(),
    }


def coverage_distances(
    table: pd.DataFrame,
    *,
    descriptors: list[str] | None = None,
    standardisation: dict | None = None,
) -> pd.DataFrame:
    """
    Pairwise distance between species in standardised descriptor space.

    Larger distance means more complementary movement regimes. This is a decision aid, not an optimiser: it prints the
    evidence for the selection, and a human makes and records the call.

    Missing descriptors are rejected rather than imputed. Filling a NaN with the column mean makes a species whose value
    could not be measured look perfectly average, which is not a neutral default when the resulting distances feed species
    selection. A species that cannot produce a descriptor fails the mechanical eligibility gate; it does not get a quietly
    invented one.
    """

    descriptors = list(descriptors or (standardisation or {}).get("descriptors") or DESCRIPTORS)
    values = table[descriptors].astype(float)

    incomplete = values.isna().any(axis=1)
    if incomplete.any():
        offenders = table.loc[incomplete, "species"].astype(str).tolist()
        cols = values.columns[values.isna().any(axis=0)].tolist()
        raise ValueError(
            f"missing descriptors for {offenders} (columns {cols}). These species do not clear the "
            "mechanical eligibility gate; exclude them explicitly rather than imputing a value."
        )

    if standardisation is None:
        standardisation = descriptor_standardisation(table, descriptors=descriptors)
    if "transforms" not in standardisation:
        raise ValueError(
            "this frozen standardisation predates the descriptor transform rule and carries no "
            "'transforms' key. Re-freeze it rather than reading it: centres and scales taken on raw "
            "descriptors do not describe transformed ones, and mixing the two produces a number that "
            "looks like a distance and is not."
        )
    centre = pd.Series(standardisation["centre"])[descriptors]
    scale = pd.Series(standardisation["scale"])[descriptors].replace(0, 1.0)

    z = (transform_descriptors(values, standardisation["transforms"]) - centre) / scale
    names = table["species"].tolist()
    return pd.DataFrame(
        [
            [float(np.linalg.norm(z.iloc[i] - z.iloc[j])) for j in range(len(z))]
            for i in range(len(z))
        ],
        index=names,
        columns=names,
    )

# GenAI reference: I implemented the original regime profiling pipeline, which included the descriptor set, exact one bin cadence rule, local
# east/north displacement calculation, training individual only profiling, low displacement boundary, species summaries and pairwise descriptor
# space distances. I then gave this file and the Tier 2 preregistration to GenAI to check my implementation was thorough and it identified a gap
# in my 10% of median rule. Specifically, for a species dominated by very short or observation error scale steps, the median itself can lie inside
# the low displacement mode. Which that case, multiplying the median by 0.10 does not meaningfully separate low displacement from travel, and
# simply varying the multiplier would not detect the problem because the anchor quantile is wrong.
#
# I asked GenAI to implement the preregistration amendment used to address this. It added (note, I do not take credit for the 'idea' of these
# additions as well, these are genuine GenAI suggestions AND implementations ontop of my original work) a two component lognormal mixture diagnostic
# with minimum sample, component weight, convergence, density crossing and BIC checks; guards for detecting when p50, p75 or p90 falls inside a supported
# low displacement component; a universal fallback decision tree that moves all species from a p50 to p75 or p90 anchor when required while preserving the
# registered stork boundary; and an explicit pooled only outcome for species whose p90 still lies inside the low component. The mixture crossing is
# diagnostic only and is never used directly as a species specific evaluation threshold.
#
# GenAI also helped add supporting safeguards such as the terminology change from behavioural "stationary" to descriptive "low displacement", with
# aliases retained for existing imports; minimum sample requirements for directional persistence and diel/seasonal amplitude; rejection rather than
# mean imputation of missing descriptors; and the ability to freeze descriptor standardisation so distances between existing species do not change when
# another candidate is later added.
#
# GenAI also wrote the accompanying tests from behaviours I specified. They are synthetic fixtures covering each branch these additons, the mixture adequacy
# rules, ID normalisation across integer, float and string forms, the descriptor sample gates, and rejection of incomplete profiles. I ran the full suite on
# and it passes. I reviewed the implemented logic line by line against amendment A4 and confirmed the originally registered rule remains the default whenever
# none of the registered guards fires.
