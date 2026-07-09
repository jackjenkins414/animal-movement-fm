"""
ARIMA baseline for animal movement prediction.

Fits a per-trajectory ARIMA model to the displacement series (delta lon, delta lat) and 
predicts the next displacement, which is added to the most recent position to produce 
the next-position forecast. Fitting on displacements gives us stationarity for free 
(raw lon/lat are non-stationary along a trajectory); fitting ARMA(p, q) on first differences
is equivalent to ARIMA(p, 1, q) on positions.

Note, this model is a reference point, not a tuned model. The (p, d, q) order is fixed at 
construction; so we don't search over it. The two coordinate channels are fit as independent 
univariate ARIMA models, so cross-coupling between lon and lat is not captured. The whole point
of this baseline is to check: 
    does a linear stochastic process with short memory beat persistence and CVM on synthetic 
    CRW/ switching CRW data? 
If not, something is wrong; if yes, by how much, and where? 

The model also fits at predict time rather than fit-then-store. This matches how the evaluation 
harness slides windows: each call to `predict` sees a fresh history of length `context_length`, 
and ARIMA fits on that window only. Wasteful (the same trajectory gets refit O(T) times) but keeps 
the model honest: it sees exactly the same data the deep learning baselines will see, no more.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

# Make private to avoid import shadowing 
from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

logger = logging.getLogger(__name__)


class ARIMA:
    """
    Per-trajectory ARIMA baseline. Fits at predict time on the displacement series.

    Fits two independent univariate ARIMA models, one per coordinate channel (Delta lon,  Delta lat), 
    and adds the one-step forecasts to `history[-1]` to produce the next-position prediction. 
    Falls back to persistence on short windows (T < min_history) or per-channel fit failures.

    Parameters
    ----------
    order : tuple[int, int, int], default (2, 0, 2)
        ARIMA(p, d, q) order applied to the displacement series. Default is ARMA(2, 2) on Delta,
        equivalent to ARIMA(2, 1, 2) on positions. Not tuned.
    min_history : int, default 10
        Window length cutoff below which `predict` returns persistence without attempting 
        a fit. Must be >= 2.
    name : str, optional
        Override for the auto-generated name. Defaults to f"arima_{p}{d}{q}".
    """
    #Note: 
    # p=2; next displacement a function of last 2 displacements 
    # q=2; the last 2 noise shocks 
    # d=0; don't difference again, input already stationary 
    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 2), # (p,d,q) tuple 
        min_history: int = 10,
        name: str | None = None,
    ) -> None:
        # We need 3 checks: 
        # 1. `order` is tupe of length 3
        if not (isinstance(order, tuple) and len(order) == 3):
            raise ValueError(f"order must be a 3-tuple (p, d, q); got {order!r}")
        p, d, q = order
        # 2. Each element is non-negative integer 
        for label, val in (("p", p), ("d", d), ("q", q)):
            if not isinstance(val, (int, np.integer)) or val < 0:
                raise ValueError(
                    # 3. !r calls repr() on bad value (string rep). 
                    f"order component {label} must be a non-negative integer, got {val!r}"
                )
            
        # Can't compute even 1 displacement with < 2 positions 
        if not isinstance(min_history, (int, np.integer)) or min_history < 2:
            raise ValueError(
                f"min_history must be an integer >= 2; got {min_history!r}"
            )

        # For order in predict 
        self.order = (int(p), int(d), int(q))
        self.min_history = int(min_history)

        # For use in Harness
        self.name = (
            name if name is not None
            else f"arima_{self.order[0]}{self.order[1]}{self.order[2]}"
        )

    def _forecast_channel(self, series: np.ndarray) -> float | None:
        """
        Fit ARIMA on a single 1D displacement channel and return the one-step forecast, or 
        None if the fit could not be produced.

        Parameters
        ----------
        series : np.ndarray
            Shape (T-1,). The displacement series for one coordinate channel.

        Returns
        -------
        float or None
            The forecasted next-step displacement on this channel, or None to signal that the 
            caller should fall back to persistence (zero displacement) for this channel. Returns 
            None for near-constant input series and for any exception raised during fit/forecast.
        """

        # Guard against near-constant series (MLE ill-posed); which means the animal isn't moving 
        #  No variance in the series, so MLE objective is flat or undefined; thus crash (I think)
        if np.allclose(series, series[0]):
            return None

        try:
            # Note, reference baseline only, not real baseline 
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Construct our statsmodels ARMINA object 
                model = _SMARIMA(series, order=self.order)

                # Run MLE estimation 
                fit = model.fit(method_kwargs={"warn_convergence": False})

                # Produce one step ahead point forecast 
                forecast = fit.forecast(steps=1)

            return float(np.asarray(forecast).ravel()[0]) # defencing unboxing; GenAI suggested this change 
        
        except Exception as exc:
            # GenAI reference: This suggestion was added since statsmodels raises a wide variety of
            # exceptions for non-convergence singular Hessians, optimiser failures, etc. Broad catch 
            # is deliberate: any fit failure falls back to persistence for this channel, and the
            # exception class is logged so genuine bugs (e.g. AttributeError) are still visible.
            logger.warning(
                "ARIMA fit failed (%s); falling back to persistence for this channel",
                exc.__class__.__name__,
            )
            return None

    def predict(self, context) -> np.ndarray:
        """
        Return predicted next (lon, lat) for the supplied history window.

        The history is first differenced to produce a displacement series, and an ARIMA model 
        with the constructor supplied order is fit independently on each coordinate channel. 
        The one-step-ahead forecasts are added to `history[-1]` to produce the predicted 
        next position.

        Parameters
        ----------
        history : np.ndarray
            Shape (T, 2), columns (lon, lat). T here equals the harness's`context_length`.

        Returns
        -------
        np.ndarray
            Shape (2,), the predicted next (lon, lat). On short windows (T < min_history) or 
            per-channel fit failures, the corresponding axis falls back to persistence 
            (zero displacement).

        Raises
        ------
        ValueError
            If `history` is not 2D with two columns, or is empty.
        """
        # Shape validation first 
        history = np.asarray(context.history)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"history must have shape (T, 2); got {history.shape} instead"
            )
        T = history.shape[0]
        if T < 1:
            raise ValueError("history must contain at least one observation")

        # If the harness provides a window with fewer than min_history, fall back to persistence
        if T < self.min_history:

            # GenAI reference: I returned history[-1] originally, but this is a bug since it returns
            # a view into the input array, and if we mutated the return value, we mutate the input
            return history[-1].copy()
        

        # I in ARMA; compute y_t - y_{t-1} along axis 0 (time)
        displacements = np.diff(history, axis=0)  # (T-1, 2)

        # ch=0 is longitude, ch=1 is latitude
        next_pos = history[-1].copy()
        for ch in range(2):
            dx_hat = self._forecast_channel(displacements[:, ch])
            if dx_hat is None:
                # Per-channel persistence fallback: next displacement = 0 on this axis.
                continue
            next_pos[ch] = next_pos[ch] + dx_hat
        return next_pos