"""Unit tests for Phase 3 models: SARIMAX, Prophet, LSTM, Croston, TSB, Router, and Ensemble."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idfs.models.ensemble import WeightedEnsembleForecaster
from idfs.models.intermittent import CrostonForecaster, TSBForecaster
from idfs.models.lstm import LSTMForecaster
from idfs.models.prophet_model import ProphetForecaster
from idfs.models.router import DemandPatternRouterForecaster, classify_demand_pattern
from idfs.models.statistical import ARIMAForecaster, SARIMAXForecaster


@pytest.fixture
def sample_daily_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a 90-day training set and a 14-day test set for two series."""
    np.random.seed(42)
    dates_train = pd.date_range("2023-01-01", periods=90, freq="D")
    dates_test = pd.date_range("2023-04-01", periods=14, freq="D")

    train_rows = []
    for s_id in [1, 2]:
        for d in dates_train:
            base = 50.0 if s_id == 1 else 20.0
            dow_eff = (d.dayofweek % 7) * 5.0
            noise = np.random.normal(0, 3.0)
            sales = max(0.0, base + dow_eff + noise)
            train_rows.append(
                {
                    "date": d,
                    "store_id": s_id,
                    "family": "GROCERY",
                    "sales": sales,
                    "onpromotion": 1 if np.random.rand() > 0.8 else 0,
                }
            )

    test_rows = []
    for s_id in [1, 2]:
        for d in dates_test:
            test_rows.append(
                {
                    "date": d,
                    "store_id": s_id,
                    "family": "GROCERY",
                    "sales": 50.0,
                    "onpromotion": 0,
                }
            )

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


@pytest.fixture
def intermittent_data() -> pd.DataFrame:
    """Generates an intermittent zero-inflated demand series."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    rows = []
    for d in dates:
        # Only ~15% of days have positive demand
        val = np.random.choice([0.0, 10.0, 25.0], p=[0.85, 0.10, 0.05])
        rows.append(
            {
                "date": d,
                "store_id": 1,
                "family": "HARDWARE",
                "sales": float(val),
            }
        )
    return pd.DataFrame(rows)


class TestStatisticalModels:
    def test_sarimax_fit_predict(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        forecaster = SARIMAXForecaster(
            order=(1, 0, 0), seasonal_order=(0, 1, 0, 7), maxiter=20, name="FastSARIMAX"
        )
        forecaster.fit(train_df, series_id_cols=["store_id", "family"])
        assert forecaster.is_fitted

        preds = forecaster.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)
        assert "forecast" in preds.columns
        assert (preds["forecast"] >= 0.0).all()

        intervals = forecaster.predict_interval(horizon=14, coverage=0.90, features_df=test_df)
        assert (intervals["lower"] <= intervals["forecast"]).all()
        assert (intervals["upper"] >= intervals["forecast"]).all()

    def test_arima_alias(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        forecaster = ARIMAForecaster(order=(1, 1, 0), maxiter=20)
        forecaster.fit(train_df, series_id_cols=["store_id", "family"])
        preds = forecaster.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)


class TestProphetModel:
    def test_prophet_fit_predict(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        forecaster = ProphetForecaster(
            weekly_seasonality=True,
            yearly_seasonality=False,
            regressors=["onpromotion"],
        )
        forecaster.fit(train_df, series_id_cols=["store_id", "family"])
        assert forecaster.is_fitted

        preds = forecaster.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)
        assert (preds["forecast"] >= 0.0).all()

        intervals = forecaster.predict_interval(horizon=14, coverage=0.90, features_df=test_df)
        assert (intervals["lower"] <= intervals["forecast"]).all()
        assert (intervals["upper"] >= intervals["forecast"]).all()


class TestLSTMModel:
    def test_lstm_fit_predict(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        forecaster = LSTMForecaster(
            seq_len=14,
            horizon=14,
            hidden_dim=16,
            num_layers=1,
            epochs=3,
            device="cpu",
        )
        forecaster.fit(train_df, series_id_cols=["store_id", "family"])
        assert forecaster.is_fitted

        preds = forecaster.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)
        assert (preds["forecast"] >= 0.0).all()

        intervals = forecaster.predict_interval(horizon=14, coverage=0.90, features_df=test_df)
        assert (intervals["lower"] <= intervals["forecast"]).all()
        assert (intervals["upper"] >= intervals["forecast"]).all()


class TestIntermittentModels:
    def test_croston_sba(self, intermittent_data):
        forecaster = CrostonForecaster(alpha=0.1, variant="sba")
        forecaster.fit(intermittent_data, series_id_cols=["store_id", "family"])
        assert forecaster.is_fitted

        preds = forecaster.predict(horizon=14)
        assert len(preds) == 14
        assert (preds["forecast"] >= 0.0).all()
        # Croston forecast should be a constant positive rate
        assert np.isclose(preds["forecast"].iloc[0], preds["forecast"].iloc[-1])

    def test_tsb_forecaster(self, intermittent_data):
        forecaster = TSBForecaster(alpha_d=0.2, alpha_p=0.2)
        forecaster.fit(intermittent_data, series_id_cols=["store_id", "family"])
        assert forecaster.is_fitted

        preds = forecaster.predict(horizon=14)
        assert len(preds) == 14
        assert (preds["forecast"] >= 0.0).all()


class TestDemandPatternRouter:
    def test_classification_quadrants(self):
        # 1. Smooth: non-zero everyday, low variance
        smooth = np.array([10.0, 11.0, 10.5, 9.8, 10.2, 11.1, 10.0] * 10)
        adi, cv2, cat = classify_demand_pattern(smooth)
        assert adi < 1.32
        assert cv2 < 0.49
        assert cat == "smooth"

        # 2. Intermittent: demand every 4 days, low variance when it occurs
        intermittent = np.array([0.0, 0.0, 0.0, 10.0] * 20)
        adi, cv2, cat = classify_demand_pattern(intermittent)
        assert adi >= 1.32
        assert cat in ["intermittent", "lumpy"]

    def test_router_end_to_end(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        router = DemandPatternRouterForecaster(
            smooth_model=ARIMAForecaster(order=(1, 0, 0), maxiter=10),
            intermittent_model=CrostonForecaster(alpha=0.1),
        )
        router.fit(train_df, series_id_cols=["store_id", "family"])
        assert router.is_fitted
        assert len(router.series_patterns_) == 2

        preds = router.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)
        assert (preds["forecast"] >= 0.0).all()

        intervals = router.predict_interval(horizon=14, coverage=0.90, features_df=test_df)
        assert len(intervals) == len(test_df)


class TestWeightedEnsemble:
    def test_ensemble_blending(self, sample_daily_data):
        train_df, test_df = sample_daily_data
        m1 = ARIMAForecaster(order=(1, 0, 0), maxiter=10, name="M1-ARIMA")
        m2 = CrostonForecaster(alpha=0.1, name="M2-Croston")

        ensemble = WeightedEnsembleForecaster(
            models=[m1, m2],
            weighting_strategy="inverse_mae",
        )
        ensemble.fit(
            train_df,
            series_id_cols=["store_id", "family"],
            val_df=train_df.iloc[-28:],
        )
        assert ensemble.is_fitted
        assert len(ensemble.weights_) == 2
        assert np.isclose(sum(ensemble.weights_), 1.0)

        preds = ensemble.predict(horizon=14, features_df=test_df)
        assert len(preds) == len(test_df)
        assert (preds["forecast"] >= 0.0).all()
