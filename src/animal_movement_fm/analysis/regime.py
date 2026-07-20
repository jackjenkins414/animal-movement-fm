"""
Movement state (regime) labelling; canonical definition.

Single source of truth for the four movement states used across the project JUST FOR WHITE STORKS (need to think of how 
to label this more generally speaking):

  foraging   incoming step  < ~60 m   near-stationary; persistence near optimal
  moving     incoming step >= ~60 m   directed movement; learned models earn their keep
  launch     foraging -> moving       onset of a directed bout (the hard, valuable event)
  settle     moving  -> foraging      settling back down

We used this earlier, and this version of the labelling logic was written and verified for the Step 6 regime decomposition
(`experiments/eda/step6_regime.py`) and is lifted here unchanged, so the new state work (classification, transition timing, state-conditioned
forecasting) and the existing diagnostics share one identical definition; exactly as `features/time.py` did for the time of day feature. Step 6 
and `dump_predictions.py` will be refactored to import from here...

Definition
----------
A fix's state is set by the step that ARRIVED at it: `step_km[i] = haversine(fix[i-1], fix[i])`, so the first fix of every segment has no incoming 
step and is labelled 'unknown'. This matches `true_step_km` in the prediction dumps (haversine(prev_true, target)) exactly.

For white stalks, foraging/moving is assigned with hysteresis around the 60 m trough; clearly moving above `HYST_HIGH_KM`, clearly foraging below 
`HYST_LOW_KM`, and a step in the grey band inherits the previous step's clear state. This stops threshold jitter at the cut from manufacturing spurious
switches. A state CHANGE from the previous fix is then split by direction into launch/settle.

Labels are windowing-agnostic; every fix in a segment frame is labelled, independent of any context length. Downstream windowing (e.g. for a state 
classifier) is applied on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from animal_movement_fm.eval.metrics import haversine_distance

REGIME_CUT_KM = 0.06  # foraging/moving trough 
HYST_LOW_KM = 0.04  # clearly foraging below this
HYST_HIGH_KM = 0.10  # clearly moving above this; 0.04-0.10 is the grey band

REGIMES = ("foraging", "moving", "launch", "settle")

# migratory launch refinement. We claim launch opens a moving bout; we call that launch 'migratory' when the bout's net (straight line) displacement reaches the 
# C. ciconia migratory-day threshold, which from our literature search, we can classify a white stork DAY as migratory vs stopover at 37 km moved between 
# consecutive night roosts; so we apply the same 37 km to a bout's net displacement, since a bout reaching 37 km net in <1 day is at least a migratory day's worth 
# of directed travel. So this sits above the resident foraging range (<=28 km breeding/ <=48 km to landfills 
MIGRATORY_NET_KM = 37.0

# four base states with 'launch' split by bout scale 
REFINED_STATES = ("foraging", "moving", "local_launch", "migratory_launch", "settle")


def add_step_length_km(
    df: pd.DataFrame,
    *,
    group_col: str = "segment_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    timestamp_col: str = "timestamp",
    out_col: str = "step_km",
) -> pd.DataFrame:
    """
    Add the incoming great-circle step length (km) per fix.

    `step_km[i] = haversine(fix[i-1], fix[i])` within each `group_col`; the first fix of each group has no predecessor and gets NaN. Rows 
    are sorted by (`group_col`, `timestamp_col`) first so the shift is in time order, matching our prev Step 6 dump. Returns a copy with 
    a reset index and the new column.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format fixes; must contain `group_col`, `lon_col`, `lat_col`, `timestamp_col`.
    group_col, lon_col, lat_col, timestamp_col : str
        Column names. Defaults match the segment product `stork_segments.parquet`.
    out_col : str
        Name of the step-length column to write.

    Returns
    -------
    pd.DataFrame
        Copy of `df`, sorted and reindexed, with `out_col` added (km; NaN at each segment start).
    """
    df = df.sort_values([group_col, timestamp_col]).reset_index(drop=True)
    prev_lon = df.groupby(group_col)[lon_col].shift(1)
    prev_lat = df.groupby(group_col)[lat_col].shift(1)

    has_prev = prev_lon.notna()
    step_km = np.full(len(df), np.nan)
    step_km[has_prev.to_numpy()] = haversine_distance(
        prev_lon[has_prev].to_numpy(),
        prev_lat[has_prev].to_numpy(),
        df.loc[has_prev, lon_col].to_numpy(),
        df.loc[has_prev, lat_col].to_numpy(),
    ) / 1000.0
    df[out_col] = step_km
    return df

def _hysteresis_regime(step_km: np.ndarray) -> np.ndarray:
    """
    Foraging/moving label per step with hysteresis, computed within one segment.

    A step is 'moving' above HYST_HIGH_KM, 'foraging' below HYST_LOW_KM; a step in the grey band inherits the previous steps' clear 
    regime. This stops threshold jitter around the 60 m cut from manufacturing spurious switches. NaN steps (first-in-segment) 
    stay 'unknown'.
    """
    out = np.empty(len(step_km), dtype=object)
    prev = None
    for i, s in enumerate(step_km):
        if not np.isfinite(s):
            out[i] = "unknown"
            continue
        if s >= HYST_HIGH_KM:
            cur = "moving"
        elif s < HYST_LOW_KM:
            cur = "foraging"
        else:
            cur = prev if prev is not None else ("moving" if s >= REGIME_CUT_KM else "foraging")
        out[i] = cur
        prev = cur
    return out


def tag_regime(
    df: pd.DataFrame,
    *,
    group_col: str = "segment_id",
    step_col: str = "step_km",
) -> pd.DataFrame:
    """
    Add a 4-way regime label: foraging / moving / launch / settle.

    Base foraging/moving is assigned with hysteresis. A regime CHANGE from the previous step is then split by direction, which we 
    give as:

      launch  foraging -> moving  (onset; expected hard)
      settle  moving  -> foraging (settling down; expected easy)

    `step_col` is the incoming step length in km per fix. Pass the default when the frame came through `add_step_length_km`; the 
    Step 6 dump passes `step_col="true_step_km"`.

    Returns a copy with `base_regime` and `regime` columns added.
    """
    df = df.copy()
    base_parts = []
    for _, seg in df.groupby(group_col, sort=False):
        base_parts.append(pd.Series(
            _hysteresis_regime(seg[step_col].to_numpy()), index=seg.index))
    df["base_regime"] = pd.concat(base_parts).reindex(df.index)

    prev = df.groupby(group_col)["base_regime"].shift(1)
    launch = (prev == "foraging") & (df["base_regime"] == "moving")
    settle = (prev == "moving") & (df["base_regime"] == "foraging")

    regime = df["base_regime"].to_numpy().copy()
    regime[launch.to_numpy()] = "launch"
    regime[settle.to_numpy()] = "settle"
    df["regime"] = regime
    return df


def label_regimes(
    df: pd.DataFrame,
    *,
    group_col: str = "segment_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    End to end per-fix state labelling for a raw segment frame.

    Computes the incoming step length then tags the four states, returning a copy (sorted, reset index) with `step_km`, `base_regime`, 
    and `regime` columns added. Every fix is labelled; the first fix of each segment is 'unknown'. No windowing is applied here; that 
    is the caller's job (e,g, building context windows for a state classifier).
    """
    labelled = add_step_length_km(
        df,
        group_col=group_col,
        lon_col=lon_col,
        lat_col=lat_col,
        timestamp_col=timestamp_col,
    )
    return tag_regime(labelled, group_col=group_col)


def _moving_bout_ids(lab: pd.DataFrame, *, group_col: str = "segment_id") -> np.ndarray:
    """
    Per fix moving-bout id: 0 outside a moving bout, 1..K within the k-th bout (in row order)

    A bout is a maximal run of base_regime == 'moving' within a group. `lab` must be sorted with a reset index (as label_regimes returns), 
    so positional order is time order within each group
    """
    seg = lab[group_col].to_numpy()
    is_mov = (lab["base_regime"] == "moving").to_numpy()

    prev_mov = np.zeros(len(lab), dtype=bool)
    prev_mov[1:] = is_mov[:-1] & (seg[1:] == seg[:-1])
    new_bout = is_mov & ~prev_mov
    ids = np.cumsum(new_bout)  # constant within a bout; read only on moving fixes
    return np.where(is_mov, ids, 0)


def moving_bouts(
    lab: pd.DataFrame,
    *,
    group_col: str = "segment_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    step_col: str = "step_km",
) -> pd.DataFrame:
    """
    One row per moving bout, with its displacement/straightness/length.

    A bout is a maximal run of base_regime == 'moving' within a segment (i.e,, the span a launch opens up to the next settle); `lab` must be a 
    label_regimes output (sorted, reset index) so positional index == row order; returns columns:

      segment_id, bout_id, first_idx, last_idx, n_steps,
      net_km        straight-line distance from the pre-launch fix to the bout's last moving fix
      path_km       summed step length over the bout (includes the launch step)
      straightness  net_km / path_km  (1 = straight transit, ~0 = out-and-back local trip)

    first_idx /last_idx are positional indices into `lab`, so callers can map bout properties back onto individual fixes.
    """
    bout_id = _moving_bout_ids(lab, group_col=group_col)
    seg = lab[group_col].to_numpy()
    lon = lab[lon_col].to_numpy()
    lat = lab[lat_col].to_numpy()
    step_km = lab[step_col].to_numpy()

    mov = np.where(bout_id > 0)[0]
    mdf = pd.DataFrame({"bout": bout_id[mov], "idx": mov, "seg": seg[mov], "step_km": step_km[mov]})
    agg = mdf.groupby("bout", sort=True).agg(
        first_idx=("idx", "first"), last_idx=("idx", "last"),
        n_steps=("idx", "size"), path_km=("step_km", "sum"), seg=("seg", "first"),
    )
    first_idx = agg["first_idx"].to_numpy()
    last_idx = agg["last_idx"].to_numpy()
    bout_seg = agg["seg"].to_numpy()
    path_km = agg["path_km"].to_numpy()

    # anchor = the pre-launch fix (one before the bout), when it exists in the same segment
    prev_idx = first_idx - 1
    same_seg = (first_idx > 0) & (seg[np.maximum(prev_idx, 0)] == bout_seg)
    anchor_idx = np.where(same_seg, prev_idx, first_idx)

    net_km = haversine_distance(
        lon[anchor_idx], lat[anchor_idx], lon[last_idx], lat[last_idx]) / 1000.0
    with np.errstate(divide="ignore", invalid="ignore"):
        straightness = np.where(path_km > 0, net_km / path_km, np.nan)

    return pd.DataFrame({
        "segment_id": bout_seg,
        "bout_id": agg.index.to_numpy(),
        "first_idx": first_idx, "last_idx": last_idx, "n_steps": agg["n_steps"].to_numpy(),
        "net_km": net_km, "path_km": path_km, "straightness": straightness,
    })


def label_migratory_launches(
    lab: pd.DataFrame,
    *,
    migratory_net_km: float = MIGRATORY_NET_KM,
    group_col: str = "segment_id",
) -> pd.DataFrame:
    """
    Refine 'launch' into 'migratory_launch' / 'local_launch' by the scale of the bout it opens.

    See earlier comment. 

      bout_net_km : net displacement (km) of the moving bout each moving fix belongs to; NaN elsewhere
      state       : REFINED_STATES; identical to `regime` except each launch is relabelled migratory_launch / local_launch

    Every launch fix is the first fix of a bout, so its `bout_net_km` is always defined.
    """
    lab = lab.copy()
    bout_id = _moving_bout_ids(lab, group_col=group_col)
    net_by_bout = moving_bouts(lab, group_col=group_col).set_index("bout_id")["net_km"]
    lab["bout_net_km"] = pd.Series(bout_id, index=lab.index).map(net_by_bout)

    state = lab["regime"].to_numpy().astype(object).copy()
    is_launch = state == "launch"
    migratory = is_launch & (lab["bout_net_km"].to_numpy() >= migratory_net_km)
    state[is_launch & ~migratory] = "local_launch"
    state[migratory] = "migratory_launch"
    lab["state"] = state
    return lab