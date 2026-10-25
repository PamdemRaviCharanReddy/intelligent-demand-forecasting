"""Unit tests for Phase 4: Quantile XGBoost, Split Conformal Calibration, and Inventory Simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idfs.inventory.simulation import (
    ConformalQuantilePolicy,
    ForecastDrivenPolicy,
    NaivePolicy,
    Simulator,
    simulate_portfolio,
)
from idfs.uncertainty.conformal import ConformalCalibrator
from idfs.uncertainty.quantile import QuantileXGBoostForecaster


@pytest.fixture
def synthetic_train_cal_test() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates train (120 days), cal (28 days), and test (14 days) splits for 2 series."""
    np.random.seed(42)
    dates_train = pd.date_range("2023-01-01", periods=120, freq="D")
    dates_cal = pd.date_range("2023-05-01", periods=28, freq="D")
    dates_test = pd.date_range("2023-05-29", periods=14, freq="D")

    def make_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for s_id in [1, 2]:
            for d in dates:
                base = 40.0 if s_id == 1 else 15.0
                dow = (d.dayofweek % 7) * 4.0
                noise = np.random.normal(0, 3.0)
                sales = max(0.0, base + dow + noise)
                rows.append(
                    {
                        "date": d,
                        "store_id": s_id,
                        "family": "BEVERAGES",
                        "sales": sales,
                        "onpromotion": 1 if np.random.rand() > 0.8 else 0,
                    }
                )
        return pd.DataFrame(rows)

    return make_df(dates_train), make_df(dates_cal), make_df(dates_test)


class TestQuantileXGBoost:
    def test_fit_and_monotonicity(self, synthetic_train_cal_test):
        train_df, _, test_df = synthetic_train_cal_test
        model = QuantileXGBoostForecaster(
            quantiles=[0.05, 0.50, 0.95],
            n_estimators=30,
            max_depth=3,
        )
        model.fit(train_df, series_id_cols=["store_id", "family"])
        assert model.is_fitted

        q_df = model.predict_quantiles(features_df=test_df)
        assert "q_0.05" in q_df.columns
        assert "q_0.50" in q_df.columns
        assert "q_0.95" in q_df.columns

        # Verify strict non-crossing monotonicity
        assert (q_df["q_0.05"] <= q_df["q_0.50"] + 1e-5).all()
        assert (q_df["q_0.50"] <= q_df["q_0.95"] + 1e-5).all()
        # Verify non-negativity
        assert (q_df["q_0.05"] >= 0.0).all()

    def test_predict_intervals(self, synthetic_train_cal_test):
        train_df, _, test_df = synthetic_train_cal_test
        model = QuantileXGBoostForecaster(n_estimators=30, max_depth=3)
        model.fit(train_df, series_id_cols=["store_id", "family"])

        intervals = model.predict_interval(horizon=14, coverage=0.90, features_df=test_df)
        assert len(intervals) == len(test_df)
        assert (intervals["lower"] <= intervals["forecast"] + 1e-5).all()
        assert (intervals["upper"] >= intervals["forecast"] - 1e-5).all()


class TestSplitConformalPrediction:
    def test_conformal_calibration(self, synthetic_train_cal_test):
        train_df, cal_df, test_df = synthetic_train_cal_test
        base_model = QuantileXGBoostForecaster(n_estimators=30, max_depth=3)
        base_model.fit(train_df, series_id_cols=["store_id", "family"])

        calibrator = ConformalCalibrator(base_forecaster=base_model, alpha=0.10)
        calibrator.calibrate(cal_df=cal_df)
        assert calibrator.is_calibrated_
        assert calibrator.q_hat_ is not None

        cal_intervals = calibrator.predict_interval(horizon=14, features_df=test_df)
        assert len(cal_intervals) == len(test_df)
        assert (cal_intervals["lower"] <= cal_intervals["upper"] + 1e-5).all()

        metrics = calibrator.evaluate_calibration(test_df=test_df)
        assert "empirical_coverage" in metrics
        assert "mean_interval_width" in metrics
        assert 0.0 <= metrics["empirical_coverage"] <= 1.0


class TestInventorySimulation:
    def test_pipeline_order_arrival_delay(self):
        """Verify an order placed at t=0 with lead_time=3 arrives at t=3."""
        policy = NaivePolicy(avg_daily_demand=20.0, lead_time=3, review_period=7)
        sim = Simulator(
            policy=policy,
            lead_time=3,
            holding_cost=0.10,
            stockout_cost=5.00,
            ordering_cost=50.00,
            initial_inventory=0.0,
        )

        demand = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        forecasts = np.array([10.0] * 7)

        res = sim.run(demand=demand, forecasts=forecasts)
        log = res.daily_log

        # Day 0: order placed (inventory was 0)
        assert log.loc[0, "order_qty"] > 0
        assert log.loc[0, "receipts"] == 0.0

        # Day 1 and 2: in transit
        assert log.loc[1, "receipts"] == 0.0
        assert log.loc[2, "receipts"] == 0.0

        # Day 3: received!
        assert log.loc[3, "receipts"] == log.loc[0, "order_qty"]

    def test_stockout_accounting_and_costs(self):
        """Verify stockout is penalized and costs sum up accurately."""
        policy = NaivePolicy(avg_daily_demand=10.0)
        sim = Simulator(
            policy=policy,
            holding_cost=0.50,
            stockout_cost=10.00,
            ordering_cost=25.00,
            initial_inventory=5.0,
            lead_time=2,
            review_period=5,
        )

        demand = np.array([15.0, 15.0, 15.0])
        forecasts = np.array([15.0, 15.0, 15.0])

        res = sim.run(demand=demand, forecasts=forecasts)
        costs = res.total_cost

        assert costs["holding"] >= 0.0
        assert costs["stockout"] > 0.0  # definitely stocked out on day 0 (demand 15 vs on-hand 5)
        assert costs["ordering"] >= 0.0
        assert np.isclose(costs["total"], costs["holding"] + costs["stockout"] + costs["ordering"])
        assert 0.0 <= res.fill_rate <= 1.0

    def test_conformal_quantile_policy(self):
        policy = ConformalQuantilePolicy(review_period=7)
        sim = Simulator(policy=policy, lead_time=3, initial_inventory=50.0)

        demand = np.array([20.0] * 14)
        forecasts = np.array([20.0] * 14)
        upper_q = np.array([25.0] * 14)

        res = sim.run(demand=demand, forecasts=forecasts, upper_quantiles=upper_q)
        assert res.fill_rate >= 0.80
        assert res.n_orders > 0

    def test_simulate_portfolio(self, synthetic_train_cal_test):
        _, _, test_df = synthetic_train_cal_test
        policy = ForecastDrivenPolicy(service_level=0.95, review_period=7)
        sim = Simulator(policy=policy, lead_time=3)

        forecast_df = test_df.copy()
        forecast_df["forecast"] = forecast_df["sales"] * 1.05
        forecast_df["upper"] = forecast_df["sales"] * 1.25

        portfolio_res = simulate_portfolio(
            simulator=sim,
            actual_df=test_df,
            forecast_df=forecast_df,
            series_id_cols=["store_id", "family"],
            target_col="sales",
            upper_col="upper",
        )

        assert "total_cost" in portfolio_res
        assert "fill_rate" in portfolio_res
        assert 0.0 <= portfolio_res["fill_rate"] <= 1.0
        assert len(portfolio_res["series_details"]) == 2
