"""Unit tests for the walk-forward backtester and model evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from idfs.evaluation.backtester import Backtester
from idfs.models.base import NaiveForecaster, SeasonalNaiveForecaster
from idfs.models.xgboost import XGBoostForecaster


@pytest.fixture
def backtest_sample_df() -> pd.DataFrame:
    # 150 days of data across 2 series
    dates = pd.date_range("2023-01-01", periods=150, freq="D")
    records = []
    for store in ["S1", "S2"]:
        for d in dates:
            # Deterministic weekly pattern + trend
            sales_val = 50.0 + (d.dayofweek * 5.0) + (d.day / 10.0)
            records.append(
                {
                    "date": d,
                    "store_id": store,
                    "family": "GROCERY",
                    "sales": float(sales_val),
                    "is_holiday": 1 if d.day in [10, 25] else 0,
                    "oil_price": 75.0,
                }
            )
    return pd.DataFrame(records)


class TestBacktester:
    def test_generate_folds(self, backtest_sample_df: pd.DataFrame) -> None:
        bt = Backtester(n_folds=3, val_days=14, gap_days=0, test_days=28)
        folds = bt.generate_folds(backtest_sample_df)
        assert len(folds) == 3

        # Validate that each fold's validation window has length 14
        for _tr_s, tr_e, v_s, v_e in folds:
            assert v_s > tr_e
            val_len = (v_e - v_s).days + 1
            assert val_len == 14

    def test_run_naive_forecaster(self, backtest_sample_df: pd.DataFrame) -> None:
        bt = Backtester(n_folds=2, val_days=7, test_days=14)
        model = NaiveForecaster()
        res = bt.run(model, backtest_sample_df)

        assert res.model_name == "Naive"
        assert len(res.folds) == 2
        assert "mae" in res.mean_metrics
        assert "wape" in res.mean_metrics
        assert res.mean_metrics["mae"] >= 0.0

    def test_run_seasonal_naive_forecaster(self, backtest_sample_df: pd.DataFrame) -> None:
        bt = Backtester(n_folds=2, val_days=7, test_days=14)
        model = SeasonalNaiveForecaster(seasonality=7)
        res = bt.run(model, backtest_sample_df)

        assert "SeasonalNaive" in res.model_name
        assert res.mean_metrics["mae"] >= 0.0

    def test_run_xgboost_forecaster(self, backtest_sample_df: pd.DataFrame) -> None:
        bt = Backtester(n_folds=2, val_days=7, test_days=14)
        model = XGBoostForecaster(n_estimators=30, max_depth=3)
        res = bt.run(model, backtest_sample_df)

        assert "XGBoost" in res.model_name
        assert res.mean_metrics["mae"] >= 0.0
        assert "coverage_90" in res.mean_metrics

    def test_compare_models(self, backtest_sample_df: pd.DataFrame) -> None:
        bt = Backtester(n_folds=2, val_days=7, test_days=14)
        models = [
            NaiveForecaster(),
            SeasonalNaiveForecaster(seasonality=7),
            XGBoostForecaster(n_estimators=20, max_depth=3),
        ]
        comp_df = bt.compare_models(models, backtest_sample_df)
        assert len(comp_df) == 3
        assert "model" in comp_df.columns
        assert "wape_mean" in comp_df.columns
        assert "mae_mean" in comp_df.columns
