# GenAI reference: Used GenAI for these; I said the adapter returns coordinates, zeroed heads stay on the last fix, 
# batch/single predictions and evaluation agree, missing timestamps are rejected, p90/travel labels follow A5, 
# skill/empty strata behave correctly; I checked all the tests were correct.

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
import torch

from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import PredictionContext, evaluate
from animal_movement_fm.eval.selection import (
    FoldScore,
    combine_folds,
    individual_weighted_quantile,
    score_fold,
    select_configuration,
    travel_regime_mask,
)
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.models.persistence import Persistence

C = 12


def _track(n_fixes=40, individual="a", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "individual_id": individual,
            "timestamp": pd.Timestamp("2020-03-01")
            + pd.to_timedelta(np.arange(n_fixes) * 6.0 + rng.uniform(0, 4, n_fixes), unit="h"),
            "lon": 10.0 + np.cumsum(rng.normal(0, 0.01, n_fixes)),
            "lat": -40.0 + np.cumsum(rng.normal(0, 0.01, n_fixes)),
        }
    )


def _model(seed=0):
    torch.manual_seed(seed)
    return MovementFM(d_model=16, nhead=2, num_layers=1)


# ------------------------------------------------------------------------------- the adapter


def test_the_predictor_returns_a_position_near_the_context(tmp_path):
    """An untrained model still has to produce a coordinate, not a normalised displacement."""
    track = _track()
    predictor = MovementFMPredictor(_model())
    history = track[["lon", "lat"]].to_numpy()[:C]
    context = PredictionContext(
        history=history,
        timestamp=track["timestamp"].iloc[C],
        history_timestamps=track["timestamp"].to_numpy()[:C],
    )

    prediction = predictor.predict(context)

    assert prediction.shape == (2,)
    assert abs(prediction[0] - history[-1, 0]) < 5.0  # degrees, not metres, and not a raw z-score
    assert abs(prediction[1] - history[-1, 1]) < 5.0


def test_a_zero_displacement_prediction_lands_exactly_on_the_last_fix():
    """
    The inverse must be exact, because every reported metre goes through it.

    With the output head zeroed the model predicts no displacement, so the adapter must return the
    anchor itself. Any drift here is a systematic bias added to every number in Section 3.
    """
    model = _model()
    torch.nn.init.zeros_(model.head.weight)
    torch.nn.init.zeros_(model.head.bias)

    track = _track()
    history = track[["lon", "lat"]].to_numpy()[:C]
    predictor = MovementFMPredictor(model)
    prediction = predictor.predict(
        PredictionContext(
            history=history,
            timestamp=track["timestamp"].iloc[C],
            history_timestamps=track["timestamp"].to_numpy()[:C],
        )
    )

    assert prediction == pytest.approx(history[-1], abs=1e-9)


def test_batched_and_single_prediction_agree():
    """One code path, two entry points. A batched scoring route that drifts is a second experiment."""
    track = _track(seed=3)
    predictor = MovementFMPredictor(_model())
    stamps = track["timestamp"].to_numpy()

    windows = np.stack([track[["lon", "lat"]].to_numpy()[i : i + C] for i in range(5)])
    window_stamps = np.stack([stamps[i : i + C] for i in range(5)])

    batched = predictor.predict_batch(windows, window_stamps)
    singles = np.stack(
        [
            predictor.predict(
                PredictionContext(history=windows[i], history_timestamps=window_stamps[i])
            )
            for i in range(5)
        ]
    )
    assert np.allclose(batched, singles, atol=1e-9)


def test_the_predictor_refuses_a_context_without_timestamps():
    """Three of the twelve channels are temporal; without real times they fall back to a constant."""
    predictor = MovementFMPredictor(_model())
    with pytest.raises(ValueError, match="history_timestamps"):
        predictor.predict(PredictionContext(history=np.zeros((C, 2))))


def test_the_predictor_runs_through_the_registered_harness():
    track = _track(n_fixes=30)
    results = evaluate(
        MovementFMPredictor(_model()), track, context_length=C, show_progress=False
    )

    assert len(results) == 30 - C
    assert results["error_m"].notna().all()
    assert (results["error_m"] >= 0).all()


# ------------------------------------------------------------------------ the selection metric


def test_p90_gives_every_individual_equal_weight():
    """
    Pooling lets the longest deployment dominate; A5 says it must not.

    One individual with 1,000 small errors and one with 25 large ones: pooled, the p90 sits in the
    large-sample individual's range. Individual weighted, the two count alike.
    """
    results = pd.DataFrame(
        {
            "individual_id": ["big"] * 1000 + ["small"] * 25,
            "error_m": list(np.full(1000, 100.0)) + list(np.full(25, 10_000.0)),
        }
    )

    pooled = float(np.percentile(results["error_m"], 90))
    weighted = individual_weighted_quantile(results)

    assert pooled == pytest.approx(100.0)  # the 25 animals are invisible
    assert weighted == pytest.approx(10_000.0)


def test_travel_regime_is_read_off_persistence():
    """Persistence's error IS the realised step, so the labels and the comparator share one pass."""
    track = _track(n_fixes=30)
    persistence = evaluate(Persistence(), track, context_length=C, show_progress=False)

    mask = travel_regime_mask(persistence, boundary_m=persistence["error_m"].median())

    assert mask.sum() == pytest.approx(len(persistence) / 2, abs=1)
    assert (persistence.loc[mask, "error_m"] >= persistence["error_m"].median()).all()


def test_skill_is_zero_when_the_model_is_persistence():
    """A model that reproduces the comparator scores exactly 0, by construction of the metric."""
    track = _track(n_fixes=30)
    persistence = evaluate(Persistence(), track, context_length=C, show_progress=False)

    score = score_fold("stork", persistence, persistence, boundary_m=0.0)

    assert score.skill == pytest.approx(0.0)
    assert score.n_travel_windows == score.n_windows


def test_skill_is_positive_when_the_model_is_better():
    track = _track(n_fixes=30)
    persistence = evaluate(Persistence(), track, context_length=C, show_progress=False)
    better = persistence.copy()
    better["error_m"] = better["error_m"] * 0.5

    assert score_fold("stork", better, persistence, boundary_m=0.0).skill == pytest.approx(0.5)


def test_a_mismatched_pair_of_runs_is_refused():
    track = _track(n_fixes=30)
    persistence = evaluate(Persistence(), track, context_length=C, show_progress=False)

    with pytest.raises(ValueError, match="same windows"):
        score_fold("stork", persistence.iloc[:5], persistence, boundary_m=0.0)


def test_an_empty_travel_stratum_is_an_error_not_a_score():
    track = _track(n_fixes=30)
    persistence = evaluate(Persistence(), track, context_length=C, show_progress=False)

    with pytest.raises(ValueError, match="clears the A4 boundary"):
        score_fold("stork", persistence, persistence, boundary_m=1e12)


# -------------------------------------------------------------------------------- the tie breaks


def _candidate(index, params, skills):
    return {
        "grid_index": index,
        "n_parameters": params,
        "scores": [
            FoldScore(f"fold{i}", s, 1.0, 1.0, 10, 20) for i, s in enumerate(skills)
        ],
    }


def test_the_primary_criterion_is_the_mean_across_folds():
    winner = select_configuration(
        [_candidate(1, 100, [0.1, 0.1, 0.1]), _candidate(2, 100, [0.0, 0.2, 0.25])]
    )
    assert winner["grid_index"] == 2
    assert combine_folds(winner["scores"]) == pytest.approx(0.15)


def test_ties_go_to_the_higher_minimum_then_fewer_parameters_then_grid_order():
    """§4's order, applied as §4 states it."""
    # equal means; the more uniform configuration wins
    assert select_configuration(
        [_candidate(1, 100, [0.0, 0.3]), _candidate(2, 100, [0.14, 0.16])]
    )["grid_index"] == 2

    # equal means and equal minima; the smaller model wins
    assert select_configuration(
        [_candidate(1, 900, [0.1, 0.2]), _candidate(2, 100, [0.1, 0.2])]
    )["grid_index"] == 2

    # equal on everything; the lowest grid index breaks it without a judgement call
    assert select_configuration(
        [_candidate(5, 100, [0.1, 0.2]), _candidate(3, 100, [0.1, 0.2])]
    )["grid_index"] == 3


def test_batched_evaluation_reproduces_the_harness_row_for_row():
    """
    Not a second evaluation protocol: same `iter_windows`, same `haversine_distance`.

    Only the prediction call is batched. If this ever diverges, the scored number and the fast number
    are two different experiments, so it is asserted rather than assumed.
    """
    track = pd.concat([_track(n_fixes=30, individual="a", seed=1),
                       _track(n_fixes=25, individual="b", seed=2)], ignore_index=True)
    predictor = MovementFMPredictor(_model())

    slow = evaluate(predictor, track, context_length=C, show_progress=False)
    fast = evaluate_batched(predictor, track, context_length=C)

    assert list(fast.columns) == list(slow.columns)
    assert len(fast) == len(slow)
    for column in ("individual_id", "timestamp", "target_lon", "target_lat"):
        assert (fast[column].to_numpy() == slow[column].to_numpy()).all(), column
    for column in ("pred_lon", "pred_lat", "error_m", "error_deg"):
        assert np.allclose(fast[column].to_numpy(), slow[column].to_numpy(), atol=1e-9), column
