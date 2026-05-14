from __future__ import annotations

import logging
import warnings

import numpy as np

from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

logger = logging.getLogger(__name__)


class ARIMA:

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 2),
        min_history: int = 10,
        name: str | None = None,
    ) -> None:
        if not (isinstance(order, tuple) and len(order) == 3):
            raise ValueError(f"order must be a 3-tuple (p, d, q); got {order!r}")
        p, d, q = order
        for label, val in (("p", p), ("d", d), ("q", q)):
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValueError(
                    f"order component {label} must be a non-negative integer, got {val!r}"
                )
        if not isinstance(min_history, (int, np.integer)) or min_history < 2:
            raise ValueError(
                f"min_history must be an integer >= 2; got {min_history!r}"
            )
        # Need enough points to identify the model after differencing.
        required = max(p, q) + d + 2
        if min_history < required:
            raise ValueError(
                f"min_history={min_history} is too small for order={order}; "
                f"need at least {required}"
            )

        self.order = (int(p), int(d), int(q))
        self.min_history = int(min_history)
        self.name = (
            name if name is not None
            else f"arima_{self.order[0]}{self.order[1]}{self.order[2]}"
        )

    def _forecast_channel(self, series: np.ndarray) -> float | None:
        # Guard against near-constant series (MLE ill-posed).
        if np.allclose(series, series[0]):
            return None

        try:
            # statsmodels warns liberally about convergence and date frequency; we don't care 
            # here because this is a reference baseline, not a tuned model. We catch real 
            # failures via the except below.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = _SMARIMA(series, order=self.order)
                fit = model.fit(method_kwargs={"warn_convergence": False})
                forecast = fit.forecast(steps=1)
            return float(np.asarray(forecast).ravel()[0])
        except (ValueError, np.linalg.LinAlgError, Exception) as exc:
            # statsmodels raises a wide variety of exceptions for non-convergence,
            # singular Hessians, etc. We log and fall back; this is a reference 
            # baseline, not a model we'll defend the convergence of.
            logger.warning(
                "ARIMA fit failed (%s); falling back to persistence for this channel",
                exc.__class__.__name__,
            )
            return None

    def predict(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"history must have shape (T, 2); got {history.shape} instead"
            )
        T = history.shape[0]
        if T < 1:
            raise ValueError("history must contain at least one observation")

        # Short history: persistence fallback.
        if T < self.min_history:
            return history[-1].copy()

        displacements = np.diff(history, axis=0)  # (T-1, 2)

        next_pos = history[-1].copy()
        for ch in range(2):
            dx_hat = self._forecast_channel(displacements[:, ch])
            if dx_hat is None:
                # Per-channel persistence fallback: next displacement = 0 on this axis.
                continue
            next_pos[ch] = next_pos[ch] + dx_hat
        return next_pos