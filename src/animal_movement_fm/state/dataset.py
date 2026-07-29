"""
Windowed dataset for the state classifier (window -> next state), Part B.

Turns a labelled segment frame into supervised examples for forecasting the movement state one step ahead. The input
representation is IDENTICAL to our prev movement rungs (`models/lstm.py`, `models/transformer.py`), so a state classifier
trained here can later become a second head on the same neural trunk without re-windowing.

Window -> target alignment
--------------------------
For a context of `context_length` positions ending just before fix `t`, the label is the state OF fix `t`; i.e. the
same fix whose POSITION the movement head predicts. The window is fixes `[t - context_length, t)`, so fix `t` itself is
never in the window: its state depends on the (unseen) step that arrives at it, making this a genuine one step FORECAST,
not a nowcast. (A nowcast would read the state of the last inwindow fix, which is trivially given by its incoming step.

Input channels (mirror our neural rungs exactly)
------------------------------------------------
`context_length` positions are differenced into `context_length - 1` displacements (`np.diff` over the window), so each
row is one step. Channels, in order:

    dlon, dlat                       always; raw degree displacements
    tod_sin, tod_cos                 if use_time_features; local solar time of day of the fix the step LANDS on
    doy_sin, doy_cos                 if use_season_features; day of year of the fix the step lands on

Displacement `j` spans fix `j -> j+1`, so it carries fix `j+1`'s clock/date, exactly as `LSTMForecaster` tags its
covariates (`_covariate_channels(win_ts[1:], history[1:, 0])`).

Note: Displacement channels are returned RAW. Tree models (our chosen classifier) are scale-invariant so need no scaling;
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from animal_movement_fm.analysis.regime import (
    REFINED_STATES,
    label_migratory_launches,
    label_regimes,
)
from animal_movement_fm.eval.harness import contiguous_run_slices
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos

# int. coding of our five refined states; fixed by REFINED_STATES order so it is stable across runs/splits
STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(REFINED_STATES)}
IDX_TO_STATE: dict[int, str] = {i: s for s, i in STATE_TO_IDX.items()}


@dataclass(frozen=True)
class StateWindows:
    """
    A built state classification dataset.

    Attributes
    ----------
    X : np.ndarray, shape (N, context_length - 1, n_channels)
        Per-window step features; channel order given by `channels`.
    y : np.ndarray, shape (N,)
        Integer label per window (STATE_TO_IDX of the target fix's refined state).
    meta : pd.DataFrame, N rows
        Columns: segment_id, individual_id, target_timestamp, target_idx, prev_base_regime, can_launch, bout_net_km.
        `prev_base_regime` is the base foraging/moving regime of the last in-window fix; `can_launch` is
        (prev_base_regime == 'foraging'), i.e, the window sits where a launch is mechanically possible (the hard
        subset). `bout_net_km` is the target fix's moving-bout net displacement (NaN when the target is not moving).
    channels : tuple[str, ...]
        Names of the last-axis channels of X.
    context_length : int
        Number of positions per window (X has context_length - 1 steps).
    """

    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    channels: tuple[str, ...]
    context_length: int


def _channel_names(use_time_features: bool, use_season_features: bool) -> tuple[str, ...]:
    names = ["dlon", "dlat"]
    if use_time_features:
        names += ["tod_sin", "tod_cos"]
    if use_season_features:
        names += ["doy_sin", "doy_cos"]
    return tuple(names)


def _segment_windows(
    coords: np.ndarray,
    timestamps: np.ndarray,
    context_length: int,
    use_time_features: bool,
    use_season_features: bool,
) -> np.ndarray | None:
    """
    Build the (N_seg, context_length - 1, n_channels) feature block for one segment, or None if too short.

    `coords` is (T, 2) as (lon, lat); `timestamps` is (T,). Targets are t = context_length .. T-1, so N_seg = T -
    context_length. Displacement rows for target t are `diff(coords)[t - CL : t - 1]`; covariate rows are the
    per-fix features of the landed on fixes `[t - CL + 1 : t]`. Both are gathered with a sliding-window view.
    """
    # seg length 
    T = len(coords)
    cl = context_length
    if T <= cl:
        return None

    # positions to displacements 
    step = np.diff(coords, axis=0)  # (T-1, 2); step[j] = fix j -> j+1
    
    # number of steps per window; cl positions contains cl - 1 transtions 
    L = cl - 1

    # displacement blocks; for target t, rows are step[t-CL : t-1]; over all targets this is starts 0 .. T-1-CL.
    disp_sv = np.lib.stride_tricks.sliding_window_view(step, L, axis=0)  # (T-CL+1, 2, L)
    disp = np.moveaxis(disp_sv[: T - cl], 1, 2)  # (N_seg, L, 2)

    # additonal features can be appended if enabled
    chans = [disp]

    if use_time_features or use_season_features:
        lon = coords[:, 0]
        # per fix covariates for the LANDED-ON fixes; target t uses fixes [t-CL+1 : t] -> sliding starts 1 .. T-CL.
        if use_time_features:
            tod_sin, tod_cos = local_solar_sincos(timestamps, lon)
            for arr in (tod_sin, tod_cos):
                sv = np.lib.stride_tricks.sliding_window_view(arr, L) # (T-L+1, L)
                chans.append(sv[1 : T - cl + 1][..., None]) # (N_seg, L, 1)
        if use_season_features:
            doy_sin, doy_cos = day_of_year_sincos(timestamps)
            for arr in (doy_sin, doy_cos):
                sv = np.lib.stride_tricks.sliding_window_view(arr, L)
                chans.append(sv[1 : T - cl + 1][..., None])

    return np.concatenate(chans, axis=-1) # (N_seg, L, n_channels)


def build_state_windows(
    frame: pd.DataFrame,
    *,
    context_length: int = 24,
    use_time_features: bool = False,
    use_season_features: bool = False,
    group_col: str = "segment_id",
) -> StateWindows:
    """
    Build the window -> next state dataset from a raw segment frame.

    `frame` is a long segment product with columns [segment_id, individual_id,
    bin_index, timestamp, lon, lat] (as `data.benchmark.load_split_frame`
    returns). Each coarse segment is split into one-bin contiguous runs before
    labelling and windowing, matching the movement harness. Frames without
    `bin_index` retain the legacy one-run-per-segment behaviour for synthetic
    and hand-built tests.

    Parameters
    ----------
    context_length : int
        Positions per window; X has context_length - 1 steps. Default 24 (the project's standard window).
    use_time_features, use_season_features : bool
        Append the (sin, cos) time-of-day / day-of-year covariate channels, identical to the neural rungs.
    group_col : str
        Segment identifier to window within. Default "segment_id".

    Returns
    -------
    StateWindows
        X, integer labels y, a per-window meta frame, the channel names, and the context length.
    """

    channels = _channel_names(use_time_features, use_season_features)
    if "bin_index" not in frame.columns:
        raise ValueError(
            "build_state_windows requires 'bin_index' so internal cadence gaps "
            "cannot be treated as contiguous"
        )

    # temp collection lists 
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    has_individual = "individual_id" in frame.columns

    ordered = frame.sort_values([group_col, "bin_index"]).reset_index(drop=True)
    for seg_id, coarse_seg in ordered.groupby(group_col, sort=False):
        coarse_seg = coarse_seg.reset_index(drop=True)
        for run_start, run_stop in contiguous_run_slices(coarse_seg):
            if run_stop - run_start <= context_length:
                continue
            raw_run = coarse_seg.iloc[run_start:run_stop].copy()
            labelled = label_migratory_launches(
                label_regimes(raw_run, group_col=group_col), group_col=group_col
            )
            coords = labelled[["lon", "lat"]].to_numpy()
            timestamps = labelled["timestamp"].to_numpy()
            block = _segment_windows(
                coords, timestamps, context_length, use_time_features, use_season_features
            )
            # The length guard above guarantees at least one window.
            assert block is not None

            T = len(labelled)
            run_target_idx = np.arange(context_length, T)
            # Keep target_idx in its historical meaning: position inside the
            # coarse segment, not inside the derived contiguous run.
            target_idx = run_start + run_target_idx
            state = labelled["state"].to_numpy()
            base_regime = labelled["base_regime"].to_numpy()
            bout_net_km = labelled["bout_net_km"].to_numpy()

            y_seg = np.array(
                [STATE_TO_IDX[state[t]] for t in run_target_idx], dtype=np.int64
            )
            prev_base = base_regime[run_target_idx - 1]

            X_parts.append(block)
            y_parts.append(y_seg)
            meta_parts.append(
                pd.DataFrame(
                    {
                        "segment_id": seg_id,
                        "individual_id": (
                            labelled["individual_id"].to_numpy()[run_target_idx]
                            if has_individual
                            else np.nan
                        ),
                        "target_timestamp": timestamps[run_target_idx],
                        "target_idx": target_idx,
                        "prev_base_regime": prev_base,
                        "can_launch": prev_base == "foraging",
                        "bout_net_km": bout_net_km[run_target_idx],
                    }
                )
            )

    if not X_parts:
        empty = np.empty((0, context_length - 1, len(channels)), dtype=float)
        return StateWindows(
            X=empty,
            y=np.empty((0,), dtype=np.int64),
            meta=pd.DataFrame(
                columns=[
                    "segment_id",
                    "individual_id",
                    "target_timestamp",
                    "target_idx",
                    "prev_base_regime",
                    "can_launch",
                    "bout_net_km",
                ]
            ),
            channels=channels,
            context_length=context_length,
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)
    return StateWindows(X=X, y=y, meta=meta, channels=channels, context_length=context_length)
