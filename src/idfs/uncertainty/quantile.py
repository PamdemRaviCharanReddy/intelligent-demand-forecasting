"""Quantile Gradient Boosting Forecaster for asymmetric and multi-quantile demand modeling (Novelty N2).

Trains gradient boosted decision trees optimizing the pinball loss objective:
    L_tau(y, q) = max(tau * (y - q), (1 - tau) * (q - y))
Guarantees non-crossing quantiles via monotonic sorting post-processing.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from idfs.features.pipeline import FeaturePipeline
from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class QuantileXGBoostForecaster(Forecaster):
    """Global Quantile XGBoost Forecaster.

    Fits separate gradient boosting trees for specified quantiles (e.g. 0.05, 0.50, 0.95).
    Guarantees non-crossing quantiles and non-negativity of demand.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        seed: int = 42,
        pipeline: FeaturePipeline | None = None,
        name: str = "QuantileXGBoost",
    ) -> None:
        super().__init__(name=name)
        self.quantiles = sorted(quantiles or [0.05, 0.50, 0.95])
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.seed = seed
        self.pipeline = pipeline

        self.models_: dict[float, xgb.XGBRegressor] = {}
        self.pipeline_: FeaturePipeline | None = None
        self._target_col: str = "sales"

    @property
    def supports_global(self) -> bool:
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> QuantileXGBoostForecaster:
        """Fit quantile regressors across all series."""
        self._target_col = target_col
        id_cols = series_id_cols or ["store_id", "family"]
        self.pipeline_ = self.pipeline or FeaturePipeline(series_id_cols=id_cols)

        logger.info("Transforming training features for Quantile XGBoost...")
        train_features = self.pipeline_.fit_transform(train_df)

        clean_train = train_features.dropna(subset=self.pipeline_.feature_cols_).copy()
        X_train = clean_train[self.pipeline_.feature_cols_].copy()
        y_train = clean_train[target_col].to_numpy(dtype=np.float64)

        for col in X_train.select_dtypes(include=["object"]).columns:
            X_train[col] = X_train[col].astype("category")

        eval_set = None
        if val_df is not None:
            val_features = self.pipeline_.transform(val_df)
            clean_val = val_features.dropna(subset=self.pipeline_.feature_cols_).copy()
            if len(clean_val) > 0:
                X_val = clean_val[self.pipeline_.feature_cols_].copy()
                y_val = clean_val[target_col].to_numpy(dtype=np.float64)
                for col in X_val.select_dtypes(include=["object"]).columns:
                    X_val[col] = X_val[col].astype("category")
                eval_set = [(X_val, y_val)]

        self.models_.clear()
        for tau in self.quantiles:
            logger.info("Training Quantile XGBoost for tau=%.2f...", tau)
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=tau,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                min_child_weight=self.min_child_weight,
                random_state=self.seed,
                tree_method="hist",
                enable_categorical=True,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            self.models_[tau] = model

        self._is_fitted = True
        return self

    def predict_quantiles(
        self,
        features_df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Predict all fitted quantiles with non-crossing monotonic correction."""
        if not self._is_fitted or self.pipeline_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        if set(self.pipeline_.feature_cols_).issubset(set(features_df.columns)):
            X_test = features_df[self.pipeline_.feature_cols_].copy()
            meta_df = features_df
        else:
            transformed = self.pipeline_.transform(features_df)
            X_test = transformed[self.pipeline_.feature_cols_].copy()
            meta_df = transformed

        for col in X_test.select_dtypes(include=["object"]).columns:
            X_test[col] = X_test[col].astype("category")

        # Collect raw predictions for each quantile
        raw_matrix = np.zeros((len(X_test), len(self.quantiles)), dtype=np.float64)
        for idx, tau in enumerate(self.quantiles):
            raw_matrix[:, idx] = self.models_[tau].predict(X_test)

        # Monotonic sorting post-processing: enforce q_tau1 <= q_tau2 <= q_tau3
        sorted_matrix = np.sort(raw_matrix, axis=1)
        clamped_matrix = np.maximum(0.0, sorted_matrix)

        out_df = pd.DataFrame()
        if "date" in meta_df.columns:
            out_df["date"] = meta_df["date"].values
        for col in self.pipeline_.series_id_cols:
            if col in meta_df.columns:
                out_df[col] = meta_df[col].values

        for idx, tau in enumerate(self.quantiles):
            out_df[f"q_{tau:.2f}"] = clamped_matrix[:, idx]

        return out_df

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return median (tau=0.50) point forecast."""
        if features_df is None:
            raise ValueError("QuantileXGBoost requires features_df")

        q_df = self.predict_quantiles(features_df=features_df, **kwargs)
        # Select median or closest quantile
        if 0.50 in self.quantiles:
            median_col = "q_0.50"
        else:
            # find closest to 0.5
            closest = min(self.quantiles, key=lambda q: abs(q - 0.50))
            median_col = f"q_{closest:.2f}"

        out_df = q_df.copy()
        out_df["forecast"] = q_df[median_col]
        return out_df

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return prediction interval matching requested coverage."""
        if features_df is None:
            raise ValueError("QuantileXGBoost requires features_df")

        q_df = self.predict_quantiles(features_df=features_df, **kwargs)
        target_low = round((1.0 - coverage) / 2.0, 2)
        target_high = round(1.0 - target_low, 2)

        # Closest available quantiles
        closest_low = min(self.quantiles, key=lambda q: abs(q - target_low))
        closest_high = min(self.quantiles, key=lambda q: abs(q - target_high))

        out_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        out_df["lower"] = q_df[f"q_{closest_low:.2f}"]
        out_df["upper"] = q_df[f"q_{closest_high:.2f}"]
        return out_df
