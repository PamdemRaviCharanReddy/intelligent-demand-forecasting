"""Unified feature engineering pipeline for IDFS.

Orchestrates lag generation, shifted rolling statistics, calendar encodings,
and train-only categorical transformations.
"""

from __future__ import annotations

import logging

import pandas as pd

from idfs.features.calendar import (
    build_calendar_features,
    build_holiday_proximity_features,
)
from idfs.features.encoders import FrequencyEncoder, TargetEncoder
from idfs.features.transformers import (
    build_diff_features,
    build_lag_features,
    build_rolling_features,
)

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """End-to-end feature engineering pipeline.

    Coordinates temporal transformers and categorical encodings.
    Guarantees that stateful transformers (TargetEncoder, FrequencyEncoder)
    are fitted exclusively on training folds.
    """

    def __init__(
        self,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_cols: list[str] | None = None,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        categorical_cols: list[str] | None = None,
    ) -> None:
        self.target_col = target_col
        self.date_col = date_col
        self.series_id_cols = series_id_cols or ["store_id", "family"]
        self.lags = lags or [1, 2, 3, 7, 14, 21, 28]
        self.rolling_windows = rolling_windows or [7, 14, 28]
        self.categorical_cols = categorical_cols or ["store_id", "family"]

        self.target_encoder = TargetEncoder(
            columns=self.categorical_cols, target_col=self.target_col
        )
        self.freq_encoder = FrequencyEncoder(columns=self.categorical_cols)
        self.feature_cols_: list[str] = []
        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> FeaturePipeline:
        """Fit stateful encoders on training data."""
        self.target_encoder.fit(train_df)
        self.freq_encoder.fit(train_df)
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame into complete feature matrix."""
        if not self.is_fitted_:
            raise RuntimeError("FeaturePipeline must be fitted on training fold before transform")

        # 1. Sort by series and date
        out = df.sort_values([*self.series_id_cols, self.date_col]).copy()

        # 2. Lag features (k >= 1)
        out = build_lag_features(
            out,
            lags=self.lags,
            target_col=self.target_col,
            group_cols=self.series_id_cols,
            date_col=self.date_col,
        )

        # 3. Shifted rolling statistics (applied on shift(1))
        out = build_rolling_features(
            out,
            windows=self.rolling_windows,
            target_col=self.target_col,
            group_cols=self.series_id_cols,
            date_col=self.date_col,
        )

        # 4. Differenced features
        out = build_diff_features(
            out,
            periods=[1, 7],
            target_col=self.target_col,
            group_cols=self.series_id_cols,
        )

        # 5. Calendar and cyclical features
        out = build_calendar_features(out, date_col=self.date_col)

        # 6. Holiday proximity
        out = build_holiday_proximity_features(out, date_col=self.date_col)

        # 7. Apply fitted categorical encoders
        out = self.target_encoder.transform(out)
        out = self.freq_encoder.transform(out)

        # Exclude non-feature columns
        non_feature_cols = {self.date_col, self.target_col, "name"}
        # Retain series_id_cols for identification or drop
        feature_candidates = [c for c in out.columns if c not in non_feature_cols]
        self.feature_cols_ = feature_candidates

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit on train and transform."""
        return self.fit(train_df).transform(train_df)
