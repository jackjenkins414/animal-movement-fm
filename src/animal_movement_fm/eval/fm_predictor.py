"""
Adapter between `MovementFM` and our evaluation harness.

The model predicts a normalised ENU displacement, while the harness evaluates `(lon, lat)` positions in metres. This module 
keeps that conversion in one place; encode the context, run the model, unscale the predicted displacement, and convert it 
back to geographic coordinates. Both single window and batched prediction use the same path.
"""

from __future__ import annotations

import numpy as np
import torch

from animal_movement_fm.data.enu import (
    SCALE_FLOOR_M,
    encode_window,
    enu_to_lonlat,
    unscale_prediction,
)
from animal_movement_fm.models.fm import MovementFM


def _epoch_seconds(timestamps) -> np.ndarray:
    """
    Convert harness timestamps to epoch seconds with an explicit datetime unit.

    Converting through `datetime64[s]` avoids assuming nanosecond precision, which we found is unsafe for DuckDB backed 
    timestamps. Object arrays are normalised through pandas first so timezone aware values are handled correctly.
    """
    values = np.asarray(timestamps)

    if values.dtype == object:
        import pandas as pd

        # `to_datetime` takes 1D only, and this function is always handed a (batch, context) array, so flatten and restore the 
        # shape. Naive timestamps skip this branch entirely, which is why it went unexercised until a species arrived with 
        # tz aware ones
        values = (
            pd.DatetimeIndex(pd.to_datetime(values.ravel(), utc=True))
            .tz_localize(None)
            .to_numpy()
            .reshape(values.shape)
        )

    return values.astype("datetime64[s]").astype("int64").astype(float)


class MovementFMPredictor:
    """
    Wrap one trained `MovementFM` for use with the evaluation harness.
    """

    def __init__(
        self,
        model: MovementFM,
        *,
        device: torch.device | str | None = None,
        name: str | None = None,
        sigma: float = SCALE_FLOOR_M,
    ) -> None:
        self.device = torch.device(device or next(model.parameters()).device)
        self.model = model.to(self.device).eval()
        self.name = name or "movement_fm"
        # must match the scale floor the model was trained under; registered arms use the registered
        # value at both ends, and a Tier E variant carries its own in its result file
        self.sigma = float(sigma)

    def predict_batch(
        self,
        history: np.ndarray,
        history_timestamps,
    ) -> np.ndarray:
        """
        Predict `(lon, lat)` for a batch of context windows.

        `history must have shape `(B, C, 2)` in `(lon, lat)` order, with `history_timestamps` aligned as `(B, C)`. 
        Predictions are produced in normalised ENU space, then converted back to metres, then anchored at the final 
        context position.
        """
        history = np.asarray(history, dtype=float)
        if history.ndim != 3 or history.shape[2] != 2:
            raise ValueError(f"history must shape (B, C, 2); got {history.shape}")

        timestamps = _epoch_seconds(history_timestamps).reshape(history.shape[:2])
        features, scale = encode_window(
            history[..., 0],
            history[..., 1],
            timestamp_s=timestamps,
            sigma=self.sigma,
        )

        with torch.no_grad():
            output = self.model(
                torch.as_tensor(
                    features,
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        # only the forecast output is scored if the reconstruction head is enabled
        prediction = output[0] if isinstance(output, tuple) else output

        step_m = unscale_prediction(
            prediction.cpu().numpy().astype(float),
            scale,
        )

        anchor = history[:, -1, :]
        lon, lat = enu_to_lonlat(
            step_m[:, 0],
            step_m[:, 1],
            anchor[:, 0],
            anchor[:, 1],
        )
        return np.stack([lon, lat], axis=-1)

    def predict(self, context) -> np.ndarray:
        """
        Predict one `(lon, lat)` position through the harness `Predictor` protocol.

        Real history timestamps are required because elapsed time and cyclic time features are part of the registered 
        input representation.
        """
        if context.history_timestamps is None:
            raise ValueError(
                "MovementFMPredictor needs history_timestamps: three of the twelve registered "
                "channels are elapsed time and the two clock pairs, and without real times they "
                "would silently fall back to a constant, which is the defect the loader was fixed "
                "for. `evaluate` populates this field."
            )

        history = np.asarray(context.history, dtype=float)
        timestamps = np.asarray(context.history_timestamps)

        return self.predict_batch(
            history[None, ...],
            timestamps[None, ...],
        )[0]


def evaluate_batched(
    predictor: MovementFMPredictor,
    df,
    *,
    context_length: int,
    individual_col: str = "individual_id",
    batch_size: int = 4096,
):
    """
    Evaluate a predictor in batches while preserving harness output format.

    Window construction and error metrics still come from the standard evaluation harness; only model inference is 
    batched. The returned DataFrame therefore matches the ordinary evaluation path row for row.
    """
    import pandas as pd

    from animal_movement_fm.eval.harness import iter_windows
    from animal_movement_fm.eval.metrics import (
        euclidean_distance_degrees,
        haversine_distance,
    )

    columns = [
        individual_col,
        "timestamp",
        "target_lon",
        "target_lat",
        "pred_lon",
        "pred_lat",
        "error_m",
        "error_deg",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.sort_values([individual_col, "timestamp"]).reset_index(drop=True)

    ids = []
    histories = []
    window_stamps = []
    targets = []
    stamps = []

    for individual_id, traj in df.groupby(individual_col, sort=False):
        traj_timestamps = traj["timestamp"].to_numpy()

        for history, target, ts, index in iter_windows(
            traj,
            context_length=context_length,
        ):
            ids.append(individual_id)
            histories.append(history)
            window_stamps.append(
                traj_timestamps[index - context_length : index]
            )
            targets.append(target)
            stamps.append(ts)

    if not histories:
        return pd.DataFrame(columns=columns)

    histories = np.stack(histories)
    window_stamps = np.stack(window_stamps)
    targets = np.stack(targets)

    predictions = np.concatenate(
        [
            predictor.predict_batch(
                histories[i : i + batch_size],
                window_stamps[i : i + batch_size],
            )
            for i in range(0, len(histories), batch_size)
        ]
    )

    return pd.DataFrame(
        {
            individual_col: ids,
            "timestamp": stamps,
            "target_lon": targets[:, 0],
            "target_lat": targets[:, 1],
            "pred_lon": predictions[:, 0],
            "pred_lat": predictions[:, 1],
            "error_m": haversine_distance(
                predictions[:, 0],
                predictions[:, 1],
                targets[:, 0],
                targets[:, 1],
            ),
            "error_deg": euclidean_distance_degrees(
                predictions[:, 0],
                predictions[:, 1],
                targets[:, 0],
                targets[:, 1],
            ),
        }
    )
