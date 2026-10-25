"""Unit tests for feature transformers and pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from idfs.features.calendar import build_calendar_features, build_holiday_proximity_features
from idfs.features.encoders import FrequencyEncoder, TargetEncoder
from idfs.features.pipeline import FeaturePipeline
from idfs.features.transformers import (
    build_diff_features,
    build_lag_features,
    build_rolling_features,
)


@pytest.fixture
def sample_sales_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    records = []
    for store in ["S1", "S2"]:
        for d in dates:
            records.append(
                {
                    "date": d,
                    "store_id": store,
                    "family": "FAM1",
                    "sales": float(d.day),
                    "is_holiday": 1 if d.day in [5, 20] else 0,
                }
            )
    return pd.DataFrame(records)


class TestTransformers:
    def test_build_lag_features(self, sample_sales_df: pd.DataFrame) -> None:
        df = build_lag_features(sample_sales_df, lags=[1, 7], target_col="sales")
        assert "sales_lag_1" in df.columns
        assert "sales_lag_7" in df.columns
        # For second row of S1, sales_lag_1 should match sales of first row
        s1 = df[df["store_id"] == "S1"].reset_index(drop=True)
        assert s1.loc[1, "sales_lag_1"] == s1.loc[0, "sales"]

    def test_lag_less_than_one_raises(self, sample_sales_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            build_lag_features(sample_sales_df, lags=[0])

    def test_build_rolling_features(self, sample_sales_df: pd.DataFrame) -> None:
        df = build_rolling_features(sample_sales_df, windows=[7], target_col="sales")
        assert "sales_roll_mean_7" in df.columns
        assert "sales_roll_std_7" in df.columns

    def test_build_diff_features(self, sample_sales_df: pd.DataFrame) -> None:
        df = build_diff_features(sample_sales_df, periods=[1], target_col="sales")
        assert "sales_diff_1" in df.columns


class TestCalendarFeatures:
    def test_calendar_and_cyclical(self, sample_sales_df: pd.DataFrame) -> None:
        df = build_calendar_features(sample_sales_df)
        expected = ["day_of_week", "month", "sin_dow", "cos_dow", "sin_month", "cos_month"]
        for col in expected:
            assert col in df.columns

    def test_holiday_proximity(self, sample_sales_df: pd.DataFrame) -> None:
        df = build_holiday_proximity_features(sample_sales_df)
        assert "days_to_next_holiday" in df.columns
        assert "days_since_last_holiday" in df.columns
        # On a holiday day itself, distance is 0
        hols = df[df["is_holiday"] == 1]
        assert (hols["days_to_next_holiday"] == 0).all()
        assert (hols["days_since_last_holiday"] == 0).all()


class TestEncoders:
    def test_target_encoder(self, sample_sales_df: pd.DataFrame) -> None:
        train = sample_sales_df[sample_sales_df["date"] < "2023-01-25"]
        val = sample_sales_df[sample_sales_df["date"] >= "2023-01-25"]

        te = TargetEncoder(columns=["store_id"], target_col="sales")
        te.fit(train)
        trans_val = te.transform(val)
        assert "store_id_target_enc" in trans_val.columns
        assert not trans_val["store_id_target_enc"].isna().any()

    def test_frequency_encoder(self, sample_sales_df: pd.DataFrame) -> None:
        fe = FrequencyEncoder(columns=["store_id"])
        trans = fe.fit_transform(sample_sales_df)
        assert "store_id_freq_enc" in trans.columns
        # S1 and S2 are equal frequency
        assert abs(trans["store_id_freq_enc"].iloc[0] - 0.5) < 0.01


class TestFeaturePipeline:
    def test_pipeline_fit_transform(self, sample_sales_df: pd.DataFrame) -> None:
        pipe = FeaturePipeline()
        transformed = pipe.fit_transform(sample_sales_df)
        assert len(pipe.feature_cols_) > 10
        assert "sales_lag_1" in transformed.columns
        assert "sales_roll_mean_7" in transformed.columns
        assert "sin_dow" in transformed.columns
