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
