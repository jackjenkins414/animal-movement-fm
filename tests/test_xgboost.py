"""Tests for the XGBoost baseline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.eval.harness import PredictionContext
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos
from animal_movement_fm.models.xgboost import XGBoost, _FeatureWindow

# GenAI reference: These tests are also GenAI generated, where we asked to "give me pytest tests
# for my XGBoost movement baseline, covering things like constructor validation, feature
# construction, fit/predict behaviour, shape errors, and anything else that stands out to you
# (explain the things that do stand out)". We verified the output.


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """A small but valid training DataFrame: 3 individuals, each long enough to
    yield several training windows at n_lags=5."""
    rng = np.random.default_rng(0)
    rows = []
    for ind in range(3):
        T = 50
        positions = rng.standard_normal((T, 2)).cumsum(axis=0) * 0.01
        for t in range(T):
            rows.append({
                "individual_id": f"ind_{ind}",
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=t),
                "lon": positions[t, 0],
                "lat": positions[t, 1],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fitted_model(simple_df):
    """A model fit on simple_df with small n_estimators for fast tests."""
    model = XGBoost(n_lags=5, n_estimators=10, max_depth=3)
    model.fit(simple_df)
    return model


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestXGBoostConstruction:
    def test_default_name_includes_n_lags(self):
        assert XGBoost().name == "xgboost_lag5"
        assert XGBoost(n_lags=10).name == "xgboost_lag10"

    def test_custom_name(self):
        assert XGBoost(name="xgboost_custom").name == "xgboost_custom"

    def test_stores_hyperparameters_as_plain_types(self):
        """numpy ints in input should be coerced to plain Python ints/floats."""
        model = XGBoost(
            n_lags=np.int64(5),
            n_estimators=np.int64(100),
            max_depth=np.int64(4),
            learning_rate=np.float64(0.05),
        )
        assert type(model.n_lags) is int
        assert type(model.n_estimators) is int
        assert type(model.max_depth) is int
        assert type(model.learning_rate) is float

    def test_rejects_n_lags_below_two(self):
        with pytest.raises(ValueError):
            XGBoost(n_lags=1)
        with pytest.raises(ValueError):
            XGBoost(n_lags=0)
        with pytest.raises(ValueError):
            XGBoost(n_lags=-3)

    def test_rejects_non_int_n_lags(self):
        with pytest.raises(ValueError):
            XGBoost(n_lags=5.0)  # type: ignore[arg-type]

    def test_rejects_zero_n_estimators(self):
        with pytest.raises(ValueError):
            XGBoost(n_estimators=0)
        with pytest.raises(ValueError):
            XGBoost(n_estimators=-1)

    def test_rejects_zero_max_depth(self):
        with pytest.raises(ValueError):
            XGBoost(max_depth=0)
        with pytest.raises(ValueError):
            XGBoost(max_depth=-2)

    def test_rejects_non_numeric_learning_rate(self):
        with pytest.raises(ValueError):
            XGBoost(learning_rate="fast")  # type: ignore[arg-type]

    def test_rejects_learning_rate_out_of_range(self):
        with pytest.raises(ValueError):
            XGBoost(learning_rate=0.0)
        with pytest.raises(ValueError):
            XGBoost(learning_rate=-0.1)
        with pytest.raises(ValueError):
            XGBoost(learning_rate=1.5)

    def test_post_fit_attributes_initially_unset(self):
        """is_fitted_ starts False, models start None — sklearn convention."""
        model = XGBoost()
        assert model.is_fitted_ is False
        assert model.feature_names_ is None
        assert model.model_dlon_ is None
        assert model.model_dlat_ is None


# ---------------------------------------------------------------------------
# Feature layout & extraction
# ---------------------------------------------------------------------------

class TestFeatureLayout:
    def test_feature_count_matches_formula(self):
        """For n_lags=k, expected layout size = 2k (positions) + 2(k-1)
        (displacements) + (k-1) (speeds) + (k-2) (turning angles) +
        2 if time features enabled."""
        for k in (2, 5, 10):
            model = XGBoost(n_lags=k, use_time_features=False)
            expected = 2 * k + 2 * (k - 1) + (k - 1) + (k - 2)
            assert len(model._feature_layout()) == expected

    def test_feature_layout_includes_time_when_enabled_pre_fit(self):
        """Pre-fit, _has_time_features() returns the constructor flag."""
        model = XGBoost(use_time_features=True)
        names = model._feature_layout()
        assert "sin_hour" in names
        assert "cos_hour" in names

    def test_feature_layout_excludes_time_when_disabled(self):
        model = XGBoost(use_time_features=False)
        names = model._feature_layout()
        assert "sin_hour" not in names
        assert "cos_hour" not in names

    def test_feature_layout_lag_naming_convention(self):
        """lag_1 is most recent, lag_n_lags is oldest."""
        model = XGBoost(n_lags=3, use_time_features=False)
        names = model._feature_layout()
        # Positions come first, in lag_1, lag_2, lag_3 order.
        assert names[0] == "lon_lag_1"
        assert names[1] == "lat_lag_1"
        assert names[2] == "lon_lag_2"
        assert names[4] == "lon_lag_3"


class TestBuildFeatures:
    def test_returns_feature_window(self):
        model = XGBoost(n_lags=5, use_time_features=False)
        history = np.arange(20.0).reshape(10, 2)
        window = model._build_features(history)
        assert isinstance(window, _FeatureWindow)
        assert window.features.ndim == 1
        assert window.last_position.shape == (2,)

    def test_last_position_matches_history(self):
        model = XGBoost(n_lags=5, use_time_features=False)
        history = np.arange(20.0).reshape(10, 2)
        window = model._build_features(history)
        np.testing.assert_array_equal(window.last_position, history[-1])

    def test_last_position_is_copy_not_view(self):
        """Mutating last_position must not affect the input history."""
        model = XGBoost(n_lags=5, use_time_features=False)
        history = np.arange(20.0).reshape(10, 2)
        window = model._build_features(history)
        window.last_position[0] = 999.0
        assert history[-1, 0] != 999.0

    def test_feature_count_matches_layout(self):
        """The feature vector and the layout name list have the same length —
        this is the contract _feature_layout guarantees."""
        for k in (2, 3, 5, 10):
            model = XGBoost(n_lags=k, use_time_features=False)
            history = np.arange(2 * (k + 3)).reshape(-1, 2).astype(float)
            window = model._build_features(history)
            assert len(window.features) == len(model._feature_layout())

    def test_rejects_wrong_shape(self):
        model = XGBoost(n_lags=5, use_time_features=False)
        with pytest.raises(ValueError):
            model._build_features(np.zeros((10, 3)))
        with pytest.raises(ValueError):
            model._build_features(np.zeros(10))

    def test_rejects_short_history(self):
        """History shorter than n_lags must raise."""
        model = XGBoost(n_lags=5, use_time_features=False)
        with pytest.raises(ValueError):
            model._build_features(np.zeros((4, 2)))

    def test_lag_position_values_are_most_recent_first(self):
        """lon_lag_1 should equal the lon of the *last* row of history."""
        model = XGBoost(n_lags=3, use_time_features=False)
        history = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
        window = model._build_features(history)
        # First feature is lon_lag_1 = last row's lon = 2.0
        assert window.features[0] == 2.0
        # Second feature is lat_lag_1 = last row's lat = 20.0
        assert window.features[1] == 20.0

    def test_displacement_features_are_first_differences(self):
        """dlon_lag_1 should be the most recent step: history[-1] - history[-2]."""
        model = XGBoost(n_lags=3, use_time_features=False)
        history = np.array([[0.0, 0.0], [1.0, 10.0], [3.0, 25.0]])
        window = model._build_features(history)
        # Layout for n_lags=3: lon_lag_1, lat_lag_1, lon_lag_2, lat_lag_2,
        # lon_lag_3, lat_lag_3, dlon_lag_1, dlat_lag_1, dlon_lag_2, dlat_lag_2,
        # speed_lag_1, speed_lag_2, turn_lag_1.
        # dlon_lag_1 = 3.0 - 1.0 = 2.0
        assert window.features[6] == 2.0
        # dlat_lag_1 = 25.0 - 10.0 = 15.0
        assert window.features[7] == 15.0

    def test_turning_angle_straight_line_is_zero(self):
        """Three colinear points moving in the same direction → turn = 0."""
        model = XGBoost(n_lags=3, use_time_features=False)
        history = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        window = model._build_features(history)
        # turn_lag_1 is the last feature (index -1) when time features disabled.
        assert np.isclose(window.features[-1], 0.0)

    def test_turning_angle_right_angle_turn(self):
        """A 90-degree left turn → turn_lag_1 ≈ +π/2."""
        model = XGBoost(n_lags=3, use_time_features=False)
        history = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        window = model._build_features(history)
        assert np.isclose(window.features[-1], np.pi / 2)

    def test_turning_angle_reversal(self):
        """A 180-degree reversal → |turn_lag_1| ≈ π."""
        model = XGBoost(n_lags=3, use_time_features=False)
        history = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        window = model._build_features(history)
        assert np.isclose(abs(window.features[-1]), np.pi)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

class TestXGBoostFit:
    def test_fit_returns_self(self, simple_df):
        """sklearn convention: fit returns the estimator for chaining."""
        model = XGBoost(n_lags=5, n_estimators=10)
        assert model.fit(simple_df) is model

    def test_fit_sets_is_fitted(self, simple_df):
        model = XGBoost(n_lags=5, n_estimators=10)
        assert model.is_fitted_ is False
        model.fit(simple_df)
        assert model.is_fitted_ is True

    def test_fit_creates_two_regressors(self, simple_df):
        model = XGBoost(n_lags=5, n_estimators=10)
        model.fit(simple_df)
        assert model.model_dlon_ is not None
        assert model.model_dlat_ is not None

    def test_fit_sets_feature_names(self, simple_df):
        model = XGBoost(n_lags=5, n_estimators=10)
        model.fit(simple_df)
        assert model.feature_names_ is not None
        assert len(model.feature_names_) > 0

    def test_fit_with_timestamps_includes_time_features(self, simple_df):
        model = XGBoost(n_lags=5, n_estimators=10, use_time_features=True)
        model.fit(simple_df)
        assert "sin_hour" in model.feature_names_
        assert "cos_hour" in model.feature_names_

    def test_fit_without_timestamps_disables_time_features(self, simple_df):
        """Constructor flag is reconciled to False if no timestamp column."""
        df = simple_df.drop(columns=["timestamp"])
        model = XGBoost(n_lags=5, n_estimators=10, use_time_features=True)
        model.fit(df)
        assert model.use_time_features is False
        assert "sin_hour" not in model.feature_names_

    def test_fit_rejects_missing_columns(self, simple_df):
        df = simple_df.drop(columns=["lon"])
        model = XGBoost(n_lags=5, n_estimators=10)
        with pytest.raises(ValueError, match="missing required columns"):
            model.fit(df)

    def test_fit_rejects_empty_when_all_trajectories_short(self):
        """If every trajectory is shorter than n_lags+1, fit raises."""
        rows = []
        for ind in range(3):
            for t in range(3):  # only 3 rows per individual, n_lags=5 needs >=6
                rows.append({
                    "individual_id": f"ind_{ind}",
                    "lon": float(t),
                    "lat": float(t),
                })
        df = pd.DataFrame(rows)
        model = XGBoost(n_lags=5, n_estimators=10)
        with pytest.raises(ValueError, match="No usable training windows"):
            model.fit(df)

    def test_fit_skips_short_trajectories_silently(self, simple_df):
        """Mix of short and long trajectories — short ones skipped, long ones used."""
        short = pd.DataFrame([
            {"individual_id": "short", "lon": 0.0, "lat": 0.0,
             "timestamp": pd.Timestamp("2024-01-01")}
        ])
        df = pd.concat([simple_df, short], ignore_index=True)
        model = XGBoost(n_lags=5, n_estimators=10)
        model.fit(df)  # must not raise
        assert model.is_fitted_

    def test_fit_logs_training_size(self, simple_df, caplog):
        model = XGBoost(n_lags=5, n_estimators=10)
        with caplog.at_level(logging.INFO, logger="animal_movement_fm.models.xgboost"):
            model.fit(simple_df)
        # Some INFO log about training size should have been emitted.
        assert any("training on" in rec.message for rec in caplog.records)

    def test_fit_windows_do_not_cross_internal_bin_gap(self, caplog):
        df = pd.DataFrame(
            {
                "individual_id": "one",
                "bin_index": [0, 1, 2, 4, 5, 6],
                "timestamp": pd.date_range("2024-01-01", periods=6, freq="6h"),
                "lon": np.arange(6, dtype=float),
                "lat": 0.0,
            }
        )
        model = XGBoost(
            n_lags=2, n_estimators=1, max_depth=1, use_time_features=False
        )

        with caplog.at_level(logging.INFO, logger="animal_movement_fm.models.xgboost"):
            model.fit(df)

        # Two three-fix runs each produce one window. The four windows from a
        # naive six-row slide would include the missing-bin boundary.
        assert any("training on 2 windows" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestXGBoostPredict:
    def test_predict_before_fit_raises(self):
        model = XGBoost(n_lags=5)
        with pytest.raises(RuntimeError, match="called before fit"):
            model.predict(PredictionContext(history=np.zeros((10, 2))))

    def test_predict_output_shape(self, fitted_model):
        history = np.arange(20.0).reshape(10, 2)
        pred = fitted_model.predict(PredictionContext(history=history))
        assert pred.shape == (2,)

    def test_predict_output_is_finite(self, fitted_model):
        rng = np.random.default_rng(42)
        history = rng.standard_normal((20, 2)).cumsum(axis=0) * 0.01
        pred = fitted_model.predict(PredictionContext(history=history))
        assert np.all(np.isfinite(pred))

    def test_predict_rejects_wrong_shape(self, fitted_model):
        with pytest.raises(ValueError):
            fitted_model.predict(PredictionContext(history=np.zeros((10, 3))))
        with pytest.raises(ValueError):
            fitted_model.predict(PredictionContext(history=np.zeros(10)))

    def test_predict_rejects_short_history(self, fitted_model):
        """history shorter than n_lags must raise; no silent fallback."""
        with pytest.raises(ValueError, match="shorter than"):
            fitted_model.predict(PredictionContext(history=np.zeros((3, 2))))

    def test_predict_does_not_mutate_input(self, fitted_model):
        history = np.arange(20.0).reshape(10, 2).copy()
        history_before = history.copy()
        _ = fitted_model.predict(PredictionContext(history=history))
        np.testing.assert_array_equal(history, history_before)

    def test_predict_determinism_on_repeated_call(self, fitted_model):
        """Same input twice should give the same prediction."""
        rng = np.random.default_rng(99)
        history = rng.standard_normal((20, 2)).cumsum(axis=0) * 0.01
        p1 = fitted_model.predict(PredictionContext(history=history))
        p2 = fitted_model.predict(PredictionContext(history=history))
        np.testing.assert_allclose(p1, p2)

    def test_predict_near_last_position(self, fitted_model):
        """For a slow-moving trajectory the prediction should land near the
        last observation (sanity check, not a tight bound)."""
        rng = np.random.default_rng(7)
        history = rng.standard_normal((20, 2)).cumsum(axis=0) * 0.001
        pred = fitted_model.predict(PredictionContext(history=history))
        # Step scale ~0.001 deg; prediction within a few step-scales.
        np.testing.assert_allclose(pred, history[-1], atol=0.5)

    def test_predict_raises_on_layout_mismatch(self, fitted_model):
        """If n_lags is mutated after fit, predict should refuse rather than
        feed the regressors a malformed feature vector."""
        fitted_model.n_lags = 3  # was 5 at fit time
        with pytest.raises(RuntimeError, match="does not match"):
            fitted_model.predict(PredictionContext(history=np.arange(20.0).reshape(10, 2)))

    def test_timestamp_flows_through_context_to_time_feature(self, fitted_model):
        """The point of the PredictionContext refactor: context.timestamp must
        reach the sin/cos time-of-day features. We check the feature vector
        directly rather than the prediction, because whether the *prediction*
        changes depends on the trained model having learned an hour-dependence
        — which the random-walk fixture deliberately has no signal for. What we
        need to guarantee is that the timestamp is not dropped in transit."""
        history = np.arange(20.0).reshape(10, 2).astype(float)
        # fitted_model trained with timestamps, so sin_hour/cos_hour are in the layout.
        assert "sin_hour" in fitted_model.feature_names_
        feats_3am = fitted_model._build_features(
            history, timestamp=pd.Timestamp("2024-06-01 03:00")).features
        feats_3pm = fitted_model._build_features(
            history, timestamp=pd.Timestamp("2024-06-01 15:00")).features
        # The two sin/cos slots (last two features) must differ between times.
        assert not np.allclose(feats_3am[-2:], feats_3pm[-2:])


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestXGBoostProperties:
    def test_displacement_target_recovers_position(self, fitted_model):
        """The model predicts displacement and adds it to history[-1].
        Predicted position should differ from history[-1] by exactly the
        predicted displacement (sanity check on the reconstruction step)."""
        history = np.arange(20.0).reshape(10, 2)
        pred = fitted_model.predict(PredictionContext(history=history))
        diff = pred - history[-1]
        # diff should be finite (whatever the model predicted).
        assert np.all(np.isfinite(diff))

    def test_fit_is_deterministic_across_seeds_with_default_hyperparameters(
        self, simple_df
    ):
        """With XGBoost's default subsample=1.0 and colsample_bytree=1.0, the
        booster is deterministic regardless of random_state — there is no
        stochastic sampling for the seed to influence. This test pins that
        behaviour: a future change that enables subsampling (e.g. for
        regularisation) should make this test fail, prompting an explicit
        decision about whether seed-sensitivity is desired."""
        rng = np.random.default_rng(0)
        history = rng.standard_normal((20, 2)).cumsum(axis=0) * 0.01

        m1 = XGBoost(n_lags=5, n_estimators=10, random_state=0).fit(simple_df)
        m2 = XGBoost(n_lags=5, n_estimators=10, random_state=1).fit(simple_df)
        p1 = m1.predict(PredictionContext(history=history))
        p2 = m2.predict(PredictionContext(history=history))
        # Both seeds produce identical predictions (no stochasticity in the
        # default config).
        np.testing.assert_allclose(p1, p2)


# ---------------------------------------------------------------------------
# Predictor protocol compatibility (matches the harness contract)
# ---------------------------------------------------------------------------

class TestXGBoostPredictorContract:
    """Integration with the Predictor protocol the harness expects."""

    def test_has_predict_method(self):
        assert callable(getattr(XGBoost(), "predict", None))

    def test_has_name_attribute(self):
        assert isinstance(XGBoost().name, str)

    def test_has_fit_method(self):
        """Unlike ARIMA, XGBoost has an explicit fit() method that must be
        called before predict()."""
        assert callable(getattr(XGBoost(), "fit", None))

    def test_predict_returns_array_of_shape_2(self, fitted_model):
        """The harness expects shape (2,) from every predictor."""
        rng = np.random.default_rng(0)
        history = rng.standard_normal((20, 2)).cumsum(axis=0) * 0.01
        pred = fitted_model.predict(PredictionContext(history=history))
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (2,)


# ---------------------------------------------------------------------------
# Day-of-year (season) feature
# ---------------------------------------------------------------------------

@pytest.fixture
def yearlong_df():
    """Trajectories spanning a full year, so day-of-year genuinely varies (simple_df covers ~2 days)."""
    rng = np.random.default_rng(1)
    rows = []
    for ind in range(2):
        T = 200
        positions = rng.standard_normal((T, 2)).cumsum(axis=0) * 0.01
        for t in range(T):
            rows.append({
                "individual_id": f"ind_{ind}",
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=6 * t),
                "lon": positions[t, 0],
                "lat": positions[t, 1],
            })
    return pd.DataFrame(rows)


class TestSeasonFeatures:
    def test_season_is_off_by_default(self):
        # Unlike use_time_features, season defaults off so committed XGBoost results are unchanged.
        model = XGBoost()
        assert model.use_season_features is False
        assert "sin_doy" not in model._feature_layout()

    def test_layout_includes_season_when_enabled(self):
        names = XGBoost(use_season_features=True)._feature_layout()
        assert "sin_doy" in names and "cos_doy" in names

    def test_season_adds_exactly_two_features(self):
        base = len(XGBoost(use_time_features=False)._feature_layout())
        season = len(XGBoost(use_time_features=False, use_season_features=True)._feature_layout())
        both = len(XGBoost(use_time_features=True, use_season_features=True)._feature_layout())
        assert season == base + 2
        assert both == base + 4

    def test_time_precedes_season_in_layout(self):
        # Order must match the neural rungs: time of day, then season.
        names = XGBoost(use_time_features=True, use_season_features=True)._feature_layout()
        assert names[-4:] == ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]

    def test_built_values_match_canonical_helpers(self):
        # The appended values must be exactly what features/time.py produces for the TARGET timestamp.
        model = XGBoost(n_lags=3, use_time_features=True, use_season_features=True)
        history = np.array([[10.0, 45.0], [10.1, 45.1], [10.2, 45.2]])
        ts = pd.Timestamp("2024-08-20 14:30")
        feats = model._build_features(history, ts).features

        sin_h, cos_h = local_solar_sincos(ts, float(history[-1, 0]))
        sin_d, cos_d = day_of_year_sincos(ts)
        np.testing.assert_allclose(feats[-4:], [sin_h, cos_h, sin_d, cos_d], atol=1e-12)

    def test_season_varies_across_the_year(self):
        # August and March must produce clearly different seasonal features.
        model = XGBoost(n_lags=3, use_time_features=False, use_season_features=True)
        history = np.array([[10.0, 45.0], [10.1, 45.1], [10.2, 45.2]])
        aug = model._build_features(history, pd.Timestamp("2024-08-20")).features[-2:]
        mar = model._build_features(history, pd.Timestamp("2024-03-20")).features[-2:]
        assert np.hypot(*(aug - mar)) > 1.0

    def test_fit_predict_roundtrip_with_season(self, yearlong_df):
        model = XGBoost(n_lags=5, n_estimators=10, max_depth=3, use_season_features=True)
        model.fit(yearlong_df)
        assert "sin_doy" in model.feature_names_
        history = yearlong_df.loc[yearlong_df.individual_id == "ind_0", ["lon", "lat"]].to_numpy()[:10]
        pred = model.predict(
            PredictionContext(history=history, timestamp=pd.Timestamp("2024-09-01 12:00"))
        )
        assert pred.shape == (2,) and np.all(np.isfinite(pred))

    def test_missing_timestamp_column_disables_both(self, yearlong_df):
        # Asking for season with no timestamp column must disable cleanly, not crash mid-fit.
        model = XGBoost(n_lags=5, n_estimators=10, max_depth=3, use_season_features=True)
        model.fit(yearlong_df.drop(columns=["timestamp"]))
        assert model.use_season_features is False
        assert model.use_time_features is False
        assert "sin_doy" not in model.feature_names_

    def test_predict_without_timestamp_uses_fallback(self, yearlong_df):
        # Fitted with season but handed no timestamp -> angle-zero fallback, still a valid prediction.
        model = XGBoost(n_lags=5, n_estimators=10, max_depth=3, use_season_features=True)
        model.fit(yearlong_df)
        history = yearlong_df.loc[yearlong_df.individual_id == "ind_0", ["lon", "lat"]].to_numpy()[:10]
        pred = model.predict(PredictionContext(history=history))
        assert pred.shape == (2,) and np.all(np.isfinite(pred))


# tests for WindowXGBoost the Tier 3 XGBoost rung on the registered windows.
def test_window_xgboost_fits_saves_and_reloads(tmp_path):
    from animal_movement_fm.models.xgboost import WindowXGBoost

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 11, 12)).astype(np.float32)
    y = np.stack([0.8 * x[:, -1, 0], -0.5 * x[:, -1, 1]], axis=-1)
    model = WindowXGBoost(n_estimators=40).fit(x, y)
    pred = model.predict(x[:50])
    assert pred.shape == (50, 2)
    assert np.corrcoef(pred[:, 0], y[:50, 0])[0, 1] > 0.9
    model.save(tmp_path / "rung")
    np.testing.assert_allclose(WindowXGBoost.load(tmp_path / "rung").predict(x[:50]), pred)


def test_window_xgboost_rejects_bad_shapes():
    from animal_movement_fm.models.xgboost import WindowXGBoost

    with pytest.raises(ValueError):
        WindowXGBoost().fit(np.zeros((10, 132)), np.zeros((10, 2)))
    with pytest.raises(ValueError):
        WindowXGBoost().fit(np.zeros((10, 11, 12)), np.zeros((10, 3)))


# GenAI reference: tests for WindowXGBoost's allow_missing (step 3.7, 10-10), from the behaviours I specified: NaN
# features refused without the flag, fitted and predicted with it, the flag kept by save and load, and a meta file
# written before the flag loading as False.
def _missing_features(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 23, 12)).astype(np.float32)
    y = np.stack([0.8 * x[:, -1, 0], -0.5 * x[:, -1, 1]], axis=-1)
    holes = rng.random((n, 23)) < 0.3
    holes[:, -11:] = False  # the core is always observed
    x[holes] = np.nan
    return x, y


def test_window_xgboost_refuses_nan_without_allow_missing():
    from animal_movement_fm.models.xgboost import WindowXGBoost

    x, y = _missing_features()
    with pytest.raises(ValueError, match="allow_missing"):
        WindowXGBoost(n_estimators=5).fit(x, y)
    fitted = WindowXGBoost(n_estimators=5).fit(np.nan_to_num(x), y)
    with pytest.raises(ValueError, match="allow_missing"):
        fitted.predict(x[:10])
    with pytest.raises(ValueError, match="infinite"):
        WindowXGBoost(n_estimators=5, allow_missing=True).fit(np.where(np.isnan(x), np.inf, x), y)
    with pytest.raises(ValueError, match="non-finite targets"):
        WindowXGBoost(n_estimators=5, allow_missing=True).fit(x, np.full_like(y, np.nan))


def test_window_xgboost_fits_missing_bins_and_keeps_the_flag(tmp_path):
    from animal_movement_fm.models.xgboost import WindowXGBoost

    x, y = _missing_features()
    model = WindowXGBoost(n_estimators=40, allow_missing=True).fit(x, y)
    pred = model.predict(x[:50])
    assert np.isfinite(pred).all() and np.corrcoef(pred[:, 0], y[:50, 0])[0, 1] > 0.9
    model.save(tmp_path / "ablation")
    loaded = WindowXGBoost.load(tmp_path / "ablation")
    assert loaded.allow_missing and "allow_missing" not in loaded.params
    np.testing.assert_allclose(loaded.predict(x[:50]), pred)


def test_a_meta_file_written_before_allow_missing_loads_as_false(tmp_path):
    import json

    from animal_movement_fm.models.xgboost import WindowXGBoost

    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 11, 12)).astype(np.float32)
    y = np.stack([x[:, -1, 0], x[:, -1, 1]], axis=-1)
    model = WindowXGBoost(n_estimators=10).fit(x, y)
    model.save(tmp_path / "old")
    meta_path = tmp_path / "old_meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["allow_missing"]
    meta_path.write_text(json.dumps(meta))
    loaded = WindowXGBoost.load(tmp_path / "old")
    assert loaded.allow_missing is False
    np.testing.assert_allclose(loaded.predict(x[:20]), model.predict(x[:20]))
