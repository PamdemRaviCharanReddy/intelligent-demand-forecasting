"""Adversarial leakage tests for feature engineering.

Proves that for any target day t, future sales at t+1 or later have
strictly zero influence on feature values computed at day t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from idfs.features.pipeline import FeaturePipeline
from idfs.features.transformers import build_lag_features, build_rolling_features


def test_rolling_features_zero_leakage() -> None:
    """Modifying future actuals at day t+1 must NOT alter rolling features at day t."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df1 = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S1",
            "family": "FAM1",
            "sales": [float(i) for i in range(30)],
        }
    )
    # df2 is identical up to day 20, but day 21 has an extreme spike (1,000,000 units)
    df2 = df1.copy()
    df2.loc[20, "sales"] = 1_000_000.0

    r1 = build_rolling_features(df1, windows=[7], target_col="sales")
    r2 = build_rolling_features(df2, windows=[7], target_col="sales")

    # At row 20 (day 21), r1 and r2 rolling features must be identical
    # because rolling uses shift(1), which looks at day 20 and earlier!
    val1 = r1.loc[20, "sales_roll_mean_7"]
    val2 = r2.loc[20, "sales_roll_mean_7"]
    assert val1 == val2, f"Leakage detected! Row 20 changed: {val1} vs {val2}"


def test_lag_features_zero_leakage() -> None:
    """Modifying future actuals at day t+1 must NOT alter lag features at day t."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df1 = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S1",
            "family": "FAM1",
            "sales": [float(i) for i in range(30)],
        }
    )
    df2 = df1.copy()
    df2.loc[25, "sales"] = 999_999.0

    l1 = build_lag_features(df1, lags=[1, 7], target_col="sales")
    l2 = build_lag_features(df2, lags=[1, 7], target_col="sales")

    # Prior to row 25, lag features must be bit-for-bit identical
    for row_idx in range(25):
        assert np.nan_to_num(l1.loc[row_idx, "sales_lag_1"]) == np.nan_to_num(
            l2.loc[row_idx, "sales_lag_1"]
        )
        assert np.nan_to_num(l1.loc[row_idx, "sales_lag_7"]) == np.nan_to_num(
            l2.loc[row_idx, "sales_lag_7"]
        )


def test_pipeline_adversarial_future_perturbation() -> None:
    """Full pipeline feature matrix prior to cut date T must remain identical after future perturbation."""
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    df1 = pd.DataFrame(
        {
            "date": dates,
            "store_id": "S1",
            "family": "FAM1",
            "sales": [float(i % 10) for i in range(60)],
            "is_holiday": [0] * 60,
        }
    )
    df2 = df1.copy()
    # Shock all values from day 45 onwards
    df2.loc[45:, "sales"] = 5000.0

    pipe1 = FeaturePipeline(series_id_cols=["store_id", "family"])
    pipe2 = FeaturePipeline(series_id_cols=["store_id", "family"])

    # Fit both strictly on first 40 days
    pipe1.fit(df1.loc[:39])
    pipe2.fit(df2.loc[:39])

    f1 = pipe1.transform(df1)
    f2 = pipe2.transform(df2)

    # Compare all feature values up to index 44 (day before the future shock)
    for col in pipe1.feature_cols_:
        s1 = f1.loc[:44, col].to_numpy()
        s2 = f2.loc[:44, col].to_numpy()
        np.testing.assert_array_equal(
            np.nan_to_num(s1),
            np.nan_to_num(s2),
            err_msg=f"Leakage detected in column {col} before shock date!",
        )
