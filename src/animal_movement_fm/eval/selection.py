"""
Tier 2 inner loop selection metric and configuration ranking.

Each pseudo target fold is scored by persistence skill on the individual weighted travel regime p90. Travel windows are 
defined from persistence error using the species' A4 boundary, fold scores are averaged, and ties are broken by minimum 
skill, parameter count, then grid index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAVEL_P90 = 0.90


def individual_weighted_quantile(
    results: pd.DataFrame,
    quantile: float = TRAVEL_P90,
    *,
    value_col: str = "error_m",
    individual_col: str = "individual_id",
) -> float:
    """
    Return a quantile where each individual contributes equal total weight.

    Within an individual, weight is divided evenly across that animal's rows. This matches the A5 definition and prevents 
    long deployments from dominating the reported statistic.
    """
    if results.empty:
        return float("nan")

    groups = [
        group[value_col].to_numpy(float)
        for _, group in results.groupby(individual_col)
    ]
    groups = [group for group in groups if group.size]

    if not groups:
        return float("nan")

    values = np.concatenate(groups)
    weights = np.concatenate(
        [
            np.full(group.size, 1.0 / (len(groups) * group.size))
            for group in groups
        ]
    )

    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]

    index = int(np.searchsorted(np.cumsum(weights), quantile))
    return float(values[min(index, values.size - 1)])


def travel_regime_mask(
    persistence_results: pd.DataFrame,
    boundary_m: float,
) -> np.ndarray:
    """
    Return the travel regime mask using persistence error as realised step length.

    Persistence predicts the last observed position, so its `error_m` is the animal's actual displacement for that 
    forecast window. A window is travel regime when this clears the species' A4 boundary.
    """
    return persistence_results["error_m"].to_numpy(float) >= float(boundary_m)


@dataclass(frozen=True)
class FoldScore:
    """
    Selection metric result for one pseudo target fold.
    """

    fold: str
    skill: float
    model_p90_m: float
    persistence_p90_m: float
    n_travel_windows: int
    n_windows: int

    @property
    def travel_fraction(self) -> float:
        return (
            self.n_travel_windows / self.n_windows
            if self.n_windows
            else float("nan")
        )


def score_fold(
    fold: str,
    model_results: pd.DataFrame,
    persistence_results: pd.DataFrame,
    *,
    boundary_m: float,
) -> FoldScore:
    """
    Score one pseudo target fold using travel regime p90 skill against persistence.

    Model and persistence results must describe the same windows in the same order, because the travel mask is 
    defined from the persistence run and then applied to both result sets.
    """
    if len(model_results) != len(persistence_results):
        raise ValueError(
            f"{len(model_results)} model rows against {len(persistence_results)} persistence rows. "
            "Both must come from one pass over same windows, or the regime labels belong to different rows than " \
            "the errors they are applied to."
        )

    mask = travel_regime_mask(persistence_results, boundary_m)
    if not mask.any():
        raise ValueError(
            f"no window on {fold} clears the A4 boundary of {boundary_m:.1f} m, so travel-regime p90 is undefined. " \
            "Likely is a data or boundary problem not model result"
        )

    model_p90 = individual_weighted_quantile(model_results.loc[mask])
    persistence_p90 = individual_weighted_quantile(
        persistence_results.loc[mask]
    )

    return FoldScore(
        fold=fold,
        skill=1.0 - model_p90 / persistence_p90,
        model_p90_m=model_p90,
        persistence_p90_m=persistence_p90,
        n_travel_windows=int(mask.sum()),
        n_windows=int(mask.size),
    )


def combine_folds(scores) -> float:
    """
    Return the mean skill across scored pseudo target folds.
    """
    values = [score.skill for score in scores]
    if not values:
        raise ValueError("no folds to combine")
    return float(np.mean(values))


def select_configuration(candidates) -> dict:
    """
    Rank candidate configurations and return the winner.

    Mean fold skill is the primary criterion. Ties are broken by highest minimum fold skill, then fewer parameters,
    then the lowest grid index.
    """
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            -combine_folds(candidate["scores"]),
            -min(score.skill for score in candidate["scores"]),
            candidate["n_parameters"],
            candidate["grid_index"],
        ),
    )

    if not ranked:
        raise ValueError("no candidate configuratiosn were scored")

    winner = dict(ranked[0])
    winner["mean_skill"] = combine_folds(winner["scores"])
    winner["min_skill"] = min(score.skill for score in winner["scores"])
    return winner
