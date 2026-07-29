"""XGBoost baseline for animal movement prediction.

The fourth rung on the baseline ladder. and the first nonlinear model in the project, and 
the last one that uses hand-engineered features rather than learned representations.

Role
----
Previously ARIMA set the linear ceiling. XGBoost sets the nonlinear-with-hand-features ceiling: how
much extra signal is recoverable from a tabular learner given domain-informed features (lag positions,
velocity, turning angles, diurnal encodings). If the LSTM cannot beat this comfortably, learned features 
are not buying us anything over hand-engineered ones, and this would be a meaningful negative result for 
the deep learning story (which hopefully won't happen). 

Approach
-------
- Per-axis regressors: one XGBRegressor for Delta lon, one for Delta lat. Predicting displacements (not 
  raw positions) keeps the target stationary, mirroring the ARIMA decision.
- Hand-engineered features per window: flattened lag positions, recent step displacements, speeds, 
  turning angles, and sin/cos time-of-day if timestamps are available (are for actual data). 
- Trained globally across all trajectories in the training split (unlike ARIMA, which fits per-trajectory). 
  This is the first model in the ladder that benefits from cross-individual data,  a property the 
  foundation model will exploit at scale.

Note: Switching-CRW is again the diagnostic dataset; XGBoost should improve over ARIMA on the long tail 
(nonlinearity helps separate foraging from transit steps in feature space), but will likely still leave 
headroom for sequence models (we hope). 
"""

from __future__ import annotations

import logging 
from dataclasses import dataclass

import numpy as np 
import pandas as pd 

from xgboost import XGBRegressor

from animal_movement_fm.eval.harness import iter_windows
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos

logger = logging.getLogger(__name__)

@dataclass
class _FeatureWindow:
    """
    Container for features extracted from one sliding window, plus the reconstruction metadata
    needed to turn a displacement prediction back into a position.

    Attributes
    ----------
    features: np.ndarray 
        Shape (n_features,). Flat feature vector ready to be passed to an XGBRegressor. Layout 
        is fixed across windows (same feature in the same column) so the trained regressors get
        a consistent input. 
    last_position: np.ndarray
        Shape (2,). The (lon, lat) at the end of the window, i.e. history [-1]. Stored so 
        `predict` can compute `next_pos = last_pos + predicted_displacement` without the caller
        re-passing the history.
    """
    features: np.ndarray
    last_position: np.ndarray

class XGBoost: 
    """
    XGBoost baseline for animal movement prediction. 

    Predicts the next (lon, lat) position as: 

        next = history[-1] + (Delta_lon_hat + Delta_lat_hat)
    
    where Delta_lon_hat ahd Delta_lat_hat come from two independent XGBRegressors trained on 
    hand-engineered features (lag positions, lag displacements, speed, turning angles, optimial
    time of day). Trained globally across all trajectories in the training split; unlike our 
    previous ARIMA baseline. 

    Reference
    ---------
    We base our implementation largely from: Wijeyakulasuriya et al (2020), "Machine learning 
    for modeling animal movement", PLOS One 15(7): e0235750. 
    Code: https://github.com/dzw5320/animalmov.

    GenAI reference
    ---------------
    Because this is a supporting baseline rather than the main focus of the Honours project, GenAI 
    was used to assist with parts of the implementation that were more time consuming than conceptually 
    central. In particular, GenAI helped with structuring some of the XGBoost-specific feature 
    engineering, validation, and fit/predict logic, allowing more project time to remain focused on the 
    primary foundation model component.

    Parameters
    ----------
    n_lags : int, default 5
        Number of historical steps to include as lag features. The total feature vector 
        includes 2*n_lags lag positions and 2*n_lags lag displacements (plus speed, 
        turning angle, and time features).
        Must be >= 2 so that at least one turning angle can be computed.
    n_estimators : int, default 200
        Number of boosting rounds for each XGBRegressor. Reference baseline only; not tuned.
    max_depth : int, default 6
        Maximum tree depth. XGBoost default. Not tuned.
    learning_rate : float, default 0.1
        Boosting step size. Literature default; not tuned.
    use_time_features : bool, default True
        If True and `fit` is given a `timestamp` column, include sin/cos time-of-day features.
        Falls back to False silently if no timestamps are available at fit time.
    use_season_features : bool, default False
        If True and `fit` is given a `timestamp` column, include sin/cos day-of-year (season) features.
        Defaults False, unlike `use_time_features`, so the committed XGBoost results (including the
        time-of-day ablation) are reproduced unchanged unless season is asked for explicitly.
    random_state : int, default 0
        Seed passed to XGBRegressor for reproducibility.
    name : str, optional
        Override for the auto-generated name. Defaults to f"xgboost_lag{n_lags}".

    Attributes (populated by `fit`)
    -------------------------------
    is_fitted_ : bool
        True once `fit` has been called successfully.
    feature_names_ : list[str]
        Names of features in the order they appear in the feature vector. Used to assert fit-time
        /predict-time consistency.
    model_dlon_, model_dlat_ : XGBRegressor
        The two trained per-axis regressors. None until `fit` runs.
    """

    def __init__(
            self, 
            n_lags: int = 5, 
            n_estimators: int = 200,
            max_depth: int = 6,
            learning_rate: float = 0.1, 
            use_time_features: bool = True,
            use_season_features: bool = False,
            random_state: int = 0,
            name: str | None = None, 
    ) -> None: 
        # n_lags >= 2, since turning angle needs 2 consecutive displacement vectors, which needs
        # 3 consecutive positions 
        if not isinstance(n_lags, (int, np.integer)) or n_lags < 2: 
            raise ValueError(f"n_lags must be an integer >= 2; got {n_lags!r}")
        
        # n_estimators > 0 since 0 rounds gives an untrained booster 
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 1: 
            raise ValueError (f"n_estimators must be a positive interger; got {n_estimators!r}")
        
        # max_depth >= 1 since depth 0 trees are generate single leaf stumps
        if not isinstance(max_depth, (int, np.integer)) or max_depth < 1: 
            raise ValueError (f"max_depth must be a positive integer; got {max_depth!r}")
        
        # learning_rate is (0, 1], since XGBoost clamps internally but we want the validation 
        # message to fire here and not deep in xgboost 
        if not isinstance(learning_rate, (int, float, np.floating)): 
            raise ValueError(f"learning rate must be numeric; got {learning_rate!r}")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be (0, 1]; got {learning_rate!r}")
        
        # Store hyperparameters 
        self.n_lags = int(n_lags)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.use_time_features = bool(use_time_features)
        self.use_season_features = bool(use_season_features)
        self.random_state = int(random_state)

        self.name = (
            name if name is not None 
            else f"xgboost_lag{self.n_lags}"
        )

        # Post fit scale; populated by `fit`, consulted by `predict`
        self.is_fitted_: bool = False
        self.feature_names_: list[str] | None = None
        self.model_dlon_: XGBRegressor | None = None 
        self.model_dlat_: XGBRegressor | None = None 
        self._warned_missing_timestamp_: bool = False

    def _feature_layout(self) -> list[str]:
        """
        Return the canonical feature-name list, in the order features appear in the flat vector. Single 
        source of truth for feature layout, used at both fit and predict time to assert consistency.

        Features
        --------
          - lag positions: lon_lag_{1..n_lags}, lat_lag_{1..n_lags}
          - lag displacements: dlon_lag_{1..n_lags-1}, dlat_lag_{1..n_lags-1}
          - lag speeds: speed_lag_{1..n_lags-1}
          - lag turning angles: turn_lag_{1..n_lags-2}
          - time-of-day (only if use_time_features and fit saw timestamps): sin_hour, cos_hour
          - season (only if use_season_features and fit saw timestamps): sin_doy, cos_doy
        """
        names: list[str] = []

        # Lag positions; last n_lags positions (relative indexed; lag_1 is the most recent observation, 
        # lag_n_lags is the oldest in the window)
        for k in range(1, self.n_lags + 1):
            names.append(f"lon_lag_{k}")
            names.append(f"lat_lag_{k}")

        # Lag displacements; n_lags-1 first differences between consecutive lag positions. This captures 
        # velocity directly
        for k in range(1, self.n_lags):
            names.append(f"dlon_lag_{k}")
            names.append(f"dlat_lag_{k}")

        # Lag speeds; magnitude of each lag displacement. Redundant with (dlon, dlat) for a linear model
        # but useful to a tree learner which can split on it without doing arithmetic across columns
        for k in range(1, self.n_lags):
            names.append(f"speed_lag_{k}")

        # Lag turning angles; angle between consecutive displacement vectors. Needs three positions per 
        # angle, so we get n_lags-2 of them
        for k in range(1, self.n_lags - 1):
            names.append(f"turn_lag_{k}")

        # Time features appended at the end if available. The flag here is advisory; the actual decision 
        # happens in fit() based on whether timestamps were supplied.
        if self.use_time_features and self._has_time_features():
            names.append("sin_hour")
            names.append("cos_hour")

        # Season appended after time of day, matching the channel order used by the neural rungs.
        if self.use_season_features and self._has_season_features():
            names.append("sin_doy")
            names.append("cos_doy")

        return names

    def _has_time_features(self) -> bool:
        """
        Whether the trained model includes time-of-day features. At predict time, we read `feature_names_` 
        (set by fit) and check whether it ends with the time-feature columns. Before fit, returns the 
        constructor flag (best guess; reconciled at fit time).
        """
        if self.feature_names_ is None:
            return self.use_time_features
        return "sin_hour" in self.feature_names_

    def _has_season_features(self) -> bool:
        """
        Whether the trained model includes day-of-year (season) features. Mirrors `_has_time_features`: after fit we
        read the committed layout, before fit we fall back to the constructor flag.
        """
        if self.feature_names_ is None:
            return self.use_season_features
        return "sin_doy" in self.feature_names_

    def _build_features(
        self,
        history: np.ndarray,
        timestamp: pd.Timestamp | None = None,
    ) -> _FeatureWindow:
        """
        Extract the feature vector for one (T, 2) history window.

        Parameters
        ----------
        history : np.ndarray
            Shape (T, 2), columns (lon, lat). T must be >= n_lags.
        timestamp : pd.Timestamp, optional
            Timestamp of the *target* observation (the step we're predicting, not the last step in history). 
            Used only if time features are enabled. Ignored otherwise.

        Returns
        -------
        _FeatureWindow
            With `features` shape (n_features,) and `last_position` shape (2,).

        Raises
        ------
        ValueError
            If history is malformed or too short for the configured n_lags.
        """
        # GenAI was used to help work through the feature building logic, especially the lagged displacements 
        # and turning angles
        history = np.asarray(history, dtype=float)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"history must have shape (T, 2); got {history.shape}"
            )
        T = history.shape[0]
        if T < self.n_lags:
            raise ValueError(
                f"history length {T} is shorter than n_lags={self.n_lags}; "
                "the harness should not be calling this on short windows"
            )

        # Take the last n_lags rows. lag[0] is the most recent observation, lag[-1] is the oldest in 
        # the window. Using a slice (not a view of the whole history) keeps the feature shape constant regardless of
        # how long the input was
        lag = history[-self.n_lags:][::-1]  # shape (n_lags, 2), most-recent-first

        feats: list[float] = []

        # Lag positions in order lag_1, lag_2, ..., lag_n_lags.
        for k in range(self.n_lags):
            feats.append(lag[k, 0])  # lon
            feats.append(lag[k, 1])  # lat

        # Lag displacements: lag_k - lag_{k+1} (most-recent-first index space); dlon_lag_1 = lon_lag_1 - lon_lag_2 
        # = (most recent step) 
        displacements = lag[:-1] - lag[1:]  # shape (n_lags-1, 2)
        for k in range(self.n_lags - 1):
            feats.append(displacements[k, 0])  # dlon
            feats.append(displacements[k, 1])  # dlat

        # Speeds: Euclidean magnitude of each displacement
        speeds = np.linalg.norm(displacements, axis=1)  # shape (n_lags-1,)
        feats.extend(speeds.tolist())

        # Turning angles: angle between consecutive displacement vectors, in the range [-pi, pi]. Computed 
        # with atan2 of the 2D cross/dot product, which is numerically stable for small angles (unlike
        # acos of the dot product).
        for k in range(self.n_lags - 2):
            v1 = displacements[k + 1]   # earlier displacement
            v2 = displacements[k]       # later (more recent) displacement
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            feats.append(float(np.arctan2(cross, dot)))

        # Optional cyclic time covariates, both computed from the TARGET timestamp (the step being predicted).
        # These now call the canonical helpers in `features/time.py` rather than an inline copy of the maths, so
        # XGBoost and the neural rungs consume an identical definition. The formula is unchanged, so previously
        # committed XGBoost numbers are reproduced exactly.
        use_time, use_season = self._has_time_features(), self._has_season_features()
        if (use_time or use_season) and timestamp is None:
            if not self._warned_missing_timestamp_:
                logger.warning(
                    "XGBoost.predict called without a timestamp but model was fit with time/season features; "
                    "using the angle-zero fallback (midnight / 1 Jan)"
                )
                self._warned_missing_timestamp_ = True
            if use_time:
                feats.extend([0.0, 1.0])
            if use_season:
                feats.extend([0.0, 1.0])
        else:
            if use_time:
                # Longitude of the last observed position (where the bird is when it takes the step); over a 6h step its
                # longitude drift is <=~1-2 deg (<=~8 min of solar correction), so the last fix longitude is the right
                # simple choice.
                sin_h, cos_h = local_solar_sincos(timestamp, float(history[-1, 0]))
                feats.extend([float(sin_h), float(cos_h)])
            if use_season:
                # No longitude needed: season is one calendar cycle corpus-wide (see features/time.py).
                sin_d, cos_d = day_of_year_sincos(timestamp)
                feats.extend([float(sin_d), float(cos_d)])

        return _FeatureWindow(
            features=np.asarray(feats, dtype=float),
            last_position=history[-1].copy(),
        )
    # GenAI reference: We used GenAI here to help structure the training pipeline in this method such as creating one
    # supervised training row per sliding window, and fitting separate XGBoost models for dlon and dlat   
    def fit(self, df_train: pd.DataFrame, *, individual_col: str = "individual_id") -> XGBoost:
        """
        Train the two per-axis regressors on a DataFrame of trajectories.

        Iterates each individual, slides windows of length n_lags across the trajectory, builds 
        a feature vector + (Delta lon, Delta lat) target per window, stacks everything into a 
        tabular dataset, and trains one XGBRegressor for each output axis.

        Parameters
        ----------
        df_train : pd.DataFrame
            Long format trajectory data with columns [individual_col, lon, lat]. Optionally a 
            `timestamp` column; if present and `use_time_features` is True, sin/cos time of day 
            features are included.
        individual_col : str, default 'individual_id'
            Column identifying which trajectory each row belongs to. Matches the harness convention.

        Returns
        -------
        self : XGBoost
            Returns self for method chaining, sklearn convention.

        Raises
        ------
        ValueError
            If the DataFrame is missing required columns, has no usable
            windows, or is otherwise malformed.
        """
        # Required-column check up front. Fail loudly here rather than KeyError like 40 lines later 
        # inside loop
        required = {individual_col, "lon", "lat"}
        missing = required - set(df_train.columns)
        if missing:
            raise ValueError(
                f"df_train is missing required columns: {sorted(missing)}"
            )

        # Decide once whether timestamps are usable. The constructor flag says "use them if available"; 
        # this is where "available" gets resolved. Mutating the flag (rather than passing it around)
        # means _has_time_features() and _build_features() agree without extra args
        wants_timestamps = self.use_time_features or self.use_season_features
        has_timestamps = wants_timestamps and "timestamp" in df_train.columns
        # Reconcile the constructor's optimistic flags with reality. If the user asked for time or season
        # features but the data has none, log it and disable both.
        if wants_timestamps and not has_timestamps:
            logger.info(
                "XGBoost.fit: time/season features requested but no 'timestamp' "
                "column found; disabling them"
            )
            self.use_time_features = False
            self.use_season_features = False

        # Lock the feature layout *before* the loop so _build_features sees a stable answer 
        # from _has_time_features(). We assign feature_names_ provisionally here, then verify post-loop that the 
        # actual feature vectors match this layout
        provisional_layout = self._feature_layout()

        # Per-window accumulators. We collect into lists and stack at the end rather than incrementally 
        # extending a numpy array (which would reallocate on every append).
        X_rows: list[np.ndarray] = []
        y_dlon: list[float] = []
        y_dlat: list[float] = []

        # group by individual so each trajectory is processed contiguously. sort=False preserves input order; 
        # we don't care about group order here but it makes debugging easier (the first group out is the
        # first individual_id in the input)
        for ind_id, group in df_train.groupby(individual_col, sort=False):
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp").reset_index(drop=True)
                window_group = group
            else:
                group = group.reset_index(drop=True)
                # XGBoost deliberately supports timestamp free synthetic frames; t he shared iterator needs an aligned target token,
                # so use row position internally; it is never exposed as a time feature because has_timestamps is false
                window_group = group.assign(timestamp=np.arange(len(group)))
            # Trajectories shorter than n_lags + 1 yield zero windows (n_lags rows for history + 1 row for 
            # target). Skip silently; this is common with irregular real-world data.
            if len(group) < self.n_lags + 1:
                continue

            # delegate to the same contiguous run iterator as evaluation and the neural rungs. With bin_index present, no feature or target
            # can cross an internal missing bin
            for history, target, ts, i in iter_windows(window_group, self.n_lags):
                # Build features. The timestamp passed in is the target timestamp (the step being predicted), 
                # matching how _build_features documents it
                target_ts = (pd.Timestamp(ts)
                    if has_timestamps
                    else None
                )
                try:
                    window = self._build_features(history, target_ts)
                except ValueError as exc:
                    # Should not happen given the length check above, but log defensively rather than crash 
                    # on a bad row
                    logger.warning(
                        "Skipping window at individual=%s, index=%d: %s",
                        ind_id, i, exc,
                    )
                    continue

                # Sanity check on the first window only — feature vector length matches the layout. Cheap 
                # insurance against silent drift if _build_features and _feature_layout disagree
                if not X_rows and len(window.features) != len(provisional_layout):
                    raise RuntimeError(
                        f"Feature vector length ({len(window.features)}) does "
                        f"not match _feature_layout() ({len(provisional_layout)}). "
                        "This is a bug in _build_features."
                    )

                X_rows.append(window.features)
                y_dlon.append(target[0] - history[-1, 0])
                y_dlat.append(target[1] - history[-1, 1])

        # Refuse to train on nothing — better a clear ValueError now than a
        # mystery XGBoost error 
        if not X_rows:
            raise ValueError(
                f"No usable training windows produced from df_train "
                f"(n_lags={self.n_lags}, n_individuals="
                f"{df_train[individual_col].nunique()}). All trajectories "
                f"may be shorter than n_lags+1."
            )

        X = np.asarray(X_rows, dtype=float)            # (n_windows, n_features)
        y_dlon_arr = np.asarray(y_dlon, dtype=float)   # (n_windows,)
        y_dlat_arr = np.asarray(y_dlat, dtype=float)   # (n_windows,)

        logger.info(
            "XGBoost.fit: training on %d windows from %d individuals, "
            "%d features",
            X.shape[0],
            df_train[individual_col].nunique(),
            X.shape[1],
        )

        # Two regressors, identical hyperparameters, different targets. Same random_state on both so the only 
        # difference between the fitted models is the target, reproducibility is per-axis-symmetric
        common_kwargs = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            objective="reg:squarederror",
            verbosity=0,
        )
        self.model_dlon_ = XGBRegressor(**common_kwargs)
        self.model_dlat_ = XGBRegressor(**common_kwargs)

        self.model_dlon_.fit(X, y_dlon_arr)
        self.model_dlat_.fit(X, y_dlat_arr)

        # Commit the layout and mark as fitted. This ordering matters: only after both regressors 
        # successfully fit do we set is_fitted_=True, so a half-failed fit doesn't leave the model 
        # in a "looks fitted but isn't really" state.
        self.feature_names_ = provisional_layout
        self._warned_missing_timestamp_ = False
        self.is_fitted_ = True
        return self
    
    # GenAI reference:  We used GenAI here to to help check the predict time logic (which it refined from my initial implementation), but
    # especially for the reconstruction of the final next position forecast from predicted dlon/dlat
    def predict(self, context) -> np.ndarray:
        """
        Return predicted next (lon, lat) for the supplied context.

        Builds the feature vector from `context.history`, runs both per-axis
        regressors, and adds the predicted displacement to `history[-1]` to produce
        the next-position forecast. If the model was fit with time features,
        `context.timestamp` (the target step's time) drives the sin/cos time-of-day
        feature; the harness now supplies this, so the midnight fallback fires only
        when the timestamp is genuinely absent.

        Parameters
        ----------
        context : PredictionContext
            Carries `history`, shape (T, 2), columns (lon, lat), T == context_length,
            T >= n_lags; and `timestamp`, the target observation's time (used only if
            the model was fit with time features).

        Returns
        -------
        np.ndarray
            Shape (2,), the predicted next (lon, lat).

        Raises
        ------
        RuntimeError
            If `predict` is called before `fit`.
        ValueError
            If `history` is malformed or shorter than n_lags.
        """
        # Refuse to predict before fitting. The trailing-underscore convention makes this check sklearn-style; 
        # the error message tells the user exactly what is needed 
        if not self.is_fitted_:
            raise RuntimeError(
                "XGBoost.predict called before fit. Call fit(df_train) first."
            )

        history = np.asarray(context.history, dtype=float)
        timestamp = context.timestamp

        # Shape validation. _build_features will also check this, but the error message here can be more specific
        # about being called from the harness path
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(
                f"history must have shape (T, 2); got {history.shape}"
            )
        if history.shape[0] < self.n_lags:
            # Unlike ARIMA, we cannot gracefully fall back to persistence on a short window: the regressors 
            # expect a fixed-length feature vector and there is no sensible way to produce one from 
            # insufficient history. The harness's context_length should be configured to be >= n_lags; 
            # if it isn't, that's a config bug the user needs to see, not a silent fallback
            raise ValueError(
                f"history length {history.shape[0]} is shorter than "
                f"n_lags={self.n_lags}; configure the harness with "
                f"context_length >= {self.n_lags}"
            )

        # Build features. _build_features handles the time-feature branch internally based on 
        # _has_time_features()
        window = self._build_features(history, timestamp=timestamp)

        # Layout consistency check: the feature vector we just built has to match the layout the regressors
        #  were trained on. If it doesn't, something has mutated between fit and predict 
        # (e.g. n_lags changed, or use_time_features was toggled) — refuse to predict rather than silently 
        # giving the regressors malformed input
        if len(window.features) != len(self.feature_names_):
            raise RuntimeError(
                f"Feature vector length ({len(window.features)}) does not "
                f"match the layout used at fit time "
                f"({len(self.feature_names_)}). The model's hyperparameters "
                "may have been mutated after fit."
            )

        # XGBRegressor.predict expects a 2D array (n_samples, n_features)
        # We have one sample, so reshape (n_features,) -> (1, n_features)
        X = window.features.reshape(1, -1)

        # Two independent forecasts, one per axis
        dlon_hat = float(self.model_dlon_.predict(X)[0])
        dlat_hat = float(self.model_dlat_.predict(X)[0])

        # Reconstruct the predicted next position. last_position is history[-1] copied at feature-build time,
        # so we don't have to worry about whether the caller mutates `history` after the call
        next_pos = window.last_position.copy()
        next_pos[0] += dlon_hat
        next_pos[1] += dlat_hat
        return next_pos