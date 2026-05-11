"""Persistence baseline.

The simplest possible predictor, where the next position simply equals the
current position. This is the floor that every other model must beat to justify 
its existence, and it is Bayes-optimal on pure Brownian motion 
(no autocorrelation to exploit).

Note for self add reference here 
"""

from __future__ import annotations

import numpy as np

class Persistence:
    """Predict the next position as the most recent observation.

    Satisfies the ``Predictor`` protocol in
    ``animal_movement_fm.eval.harness``: provides ``predict(history) -> (2,)``.

    No ``fit`` method since this model has no parameters to learn.
    """

    # Class attribute for tables produced by eval harness 
    name = "persistence"

    def predict(self, history: np.ndarray) -> np.ndarray:
        """Return the last (lon, lat) in ``history``.

        Parameters
        ----------
        history : np.ndarray
            Shape (T, 2), columns (lon, lat). T == context_length.

        Returns
        -------
        np.ndarray
            Shape (2,), the predicted next (lon, lat).
        """
        history = np.asarray(history)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"history must have shape (T, 2); got {history.shape}"
            )
        if history.shape[0] < 1:
            raise ValueError("history must contain at least one observation")
        return history[-1].copy()
