"""
Displacement-averaging model baselines (of constant velocity). 

Predict the next position as `current + mean(last k displacement vectors)`. 
With `k=1`, this is the constant velocity model (CVM), a storng baseline
in the pedestrian trajectory literature (see below for add later). With 
`k>1`, the predictin smooths noisy single-step velocities at the cost of lag. 

These models have no learned parameters; they read the hyperparameter `k` 
from the constructor only. 

https://arxiv.org/pdf/1903.07933
"""

from __future__ import annotations
import numpy as np 

class DisplacementAveraging: 
    """
    Predict next position by extrapolating by the mean of recent displacements. 

    Let `d_t = history[t] - history[t-1]` be the displacement from step `t-1` to 
    `t`. With context legnth `T` there are `T - 1` such displacements. This model
    returns: 
                history[-1] + mean(d_{T-k+1}, ..., d_{T - 1})
    where the mean is taken over the most recent `min(k, T - 1)` displacements. 

    Parameters
    -----
    k: int, default = 1
       Number of recent displacement vectors to average. `k=1` is the constant 
       velocity model (CVM). Must be >= 1. 
    name: str, optional
       Override the default name (for results table purposes, that include multiple 
       `k` values, e.g. `averaging_k10`). If not given, defaults to `faveraging_k{k}`. 

    Notes
    -----
    We fall back to persistence (zero displacement) when `history` has fewer than 
    2 rows, since no displacement can be computed. 

    Displacements are computed in raw (lon, lat) degrees, NOT on the sphere. This is a 
    small step approximation and is fine at the per-step scale of typical movement data 
    (~km), but it will degrade as we move towards the poles. The harness still scores 
    predictions with haversine, so any approximation error shows up in the report 
    metrics; no hidden bias.

    >>>>>>>Should I add examples?<<<<<<

    >>>> Will comment more later<<<<<

    """

    def __init__(self, k: int = 1, name: str | None = None) -> None:
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r} instead")
        self.k = int(k)
        self.name= name if name is not None else f"averaging_k{self.k}"

    def predict(self, context) -> np.ndarray:
        """
        Return predicted next (lon, lat).

        Parameters
        ----------
        context: PredictionContext
          Carries `history`, shape (T, 2), columns (lon, lat), T == context_length. Averaging ignores the timestamp.

        Returns
        -----
        np.ndarray
          Shape (2,), the predicted next (lon, lat).
        """
        history = np.asarray(context.history)

        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(f"history must have shape (T, 2); got {history.shape} instead")
        T = history.shape[0]
        if T < 1:
            raise ValueError("history must contain at least one observation")

        if T < 2:
            # no displacements availible; behave like persistence
            return history[-1].copy()

        # displacements; shape (T - 1, 2)
        displacements = np.diff(history, axis=0)

        # use most recent min(k, T - 1) of them
        k_eff = min(self.k, displacements.shape[0])
        mean_disp = displacements[-k_eff:].mean(axis=0)
        return history[-1] + mean_disp


