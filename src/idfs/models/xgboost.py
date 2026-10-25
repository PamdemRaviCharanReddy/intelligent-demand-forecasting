"""Global XGBoost forecaster for IDFS.

Implements cross-series learning across all stores and product families
with early stopping and feature importance tracking.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from idfs.features.pipeline import FeaturePipeline
from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(Forecaster):
    """Global XGBoost Forecaster.

    Trains across all series simultaneously using tabular lag, rolling,
    and calendar features. Conforms strictly to the Forecaster ABC.

    Attributes:
        n_estimators: Maximum tree boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Shrinkage step size.
        subsample: Subsample ratio of training instances.
        colsample_bytree: Subsample ratio of columns.
        seed: Random seed.
    """

    def __init__(
        self,
        name: str = "XGBoost-global",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 10,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.seed = seed

        self.pipeline_: FeaturePipeline | None = None
        self.model_: xgb.XGBRegressor | None = None
        self.feature_names_: list[str] = []
        self.residual_std_: float = 1.0

    @property
    def supports_global(self) -> bool:
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        early_stopping_rounds: int = 50,
        **kwargs: Any,
    ) -> XGBoostForecaster:
        """Fit global XGBoost model on multi-series training data.

        Args:
            train_df: Training DataFrame with dates, targets, and series IDs.
            target_col: Column name of sales target.
            series_id_cols: Identifier columns (e.g. ['store_id', 'family']).
            val_df: Optional validation DataFrame for early stopping.
            early_stopping_rounds: Patience rounds for validation loss.
            **kwargs: Extra parameters passed to XGBRegressor.

        Returns:
            self (fitted instance).
        """
        series_id_cols = series_id_cols or ["store_id", "family"]
        self.pipeline_ = FeaturePipeline(
            target_col=target_col,
            series_id_cols=series_id_cols,
        )

        logger.info("Transforming training features for global XGBoost...")
        train_features = self.pipeline_.fit_transform(train_df)

        # Drop initial rows where long lags (lag 28) produce NaNs
        clean_train = train_features.dropna(subset=self.pipeline_.feature_cols_).copy()
        X_train = clean_train[self.pipeline_.feature_cols_].copy()
        y_train = clean_train[target_col].to_numpy(dtype=np.float64)

        # Convert categoricals to category dtype for XGBoost native handling
        for col in X_train.select_dtypes(include=["object"]).columns:
            X_train[col] = X_train[col].astype("category")

        eval_set = None
        if val_df is not None:
            # Transform validation set using train-fitted pipeline
            val_features = self.pipeline_.transform(val_df)
            clean_val = val_features.dropna(subset=self.pipeline_.feature_cols_).copy()
            if len(clean_val) > 0:
                X_val = clean_val[self.pipeline_.feature_cols_].copy()
                y_val = clean_val[target_col].to_numpy(dtype=np.float64)
                for col in X_val.select_dtypes(include=["object"]).columns:
                    X_val[col] = X_val[col].astype("category")
                eval_set = [(X_val, y_val)]

        self.model_ = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            random_state=self.seed,
            enable_categorical=True,
            early_stopping_rounds=early_stopping_rounds if eval_set is not None else None,
            tree_method="hist",
        )

        logger.info(
            "Fitting XGBoost regressor (rows=%d, features=%d)...",
            len(X_train),
            len(self.pipeline_.feature_cols_),
        )
        self.model_.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self.feature_names_ = list(self.pipeline_.feature_cols_)
        # In-sample residual standard deviation for prediction intervals
        preds_train = self.model_.predict(X_train)
        residuals = y_train - preds_train
        self.residual_std_ = float(np.std(residuals)) if len(residuals) > 1 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate point forecasts for future horizon.

        Args:
            horizon: Number of periods.
            features_df: DataFrame containing pre-engineered features or raw test inputs.

        Returns:
            DataFrame with [date, store_id, family, 'forecast'].
        """
        if not self._is_fitted or self.model_ is None or self.pipeline_ is None:
            raise RuntimeError("XGBoostForecaster must be fitted before predict")
        if features_df is None:
            raise ValueError("XGBoost requires features_df for multi-step inference")

        # If features_df is already transformed with feature_cols_
        if set(self.pipeline_.feature_cols_).issubset(set(features_df.columns)):
            X_test = features_df[self.pipeline_.feature_cols_].copy()
            meta_df = features_df
        else:
            # Transform with fitted pipeline
            transformed = self.pipeline_.transform(features_df)
            X_test = transformed[self.pipeline_.feature_cols_].copy()
            meta_df = transformed

        for col in X_test.select_dtypes(include=["object"]).columns:
            X_test[col] = X_test[col].astype("category")

        raw_preds = self.model_.predict(X_test)
        # Demand cannot be negative
        clamped_preds = np.maximum(0.0, raw_preds)

        out_df = pd.DataFrame()
        if "date" in meta_df.columns:
            out_df["date"] = meta_df["date"].values
        for id_col in self.pipeline_.series_id_cols:
            if id_col in meta_df.columns:
                out_df[id_col] = meta_df[id_col].values

        out_df["forecast"] = clamped_preds
        return out_df

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate point forecasts with prediction intervals."""
        point_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)
        half_width = z * self.residual_std_

        point_df["lower"] = np.maximum(0.0, point_df["forecast"] - half_width)
        point_df["upper"] = point_df["forecast"] + half_width
        return point_df

    @property
    def feature_importances_(self) -> pd.DataFrame:
        """Return table of feature importances sorted in descending order."""
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("Model not fitted yet")
        scores = self.model_.feature_importances_
        df = (
            pd.DataFrame({"feature": self.feature_names_, "importance": scores})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return df
