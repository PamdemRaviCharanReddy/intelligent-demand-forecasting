"""Tests for evaluation metrics.

Covers all point, interval, and inventory metrics with edge cases:
- Zero actuals (MAPE, WAPE)
- Constant series (MASE)
- Empty arrays
- Perfect forecasts
- Single-element arrays
"""

from __future__ import annotations

import math

import pytest

from idfs.evaluation.metrics import (
    compute_all_point_metrics,
    empirical_coverage,
    fill_rate,
    interval_width,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    stockout_periods,
    total_inventory_cost,
    wape,
)

# ── MAE ──────────────────────────────────────────────────────────────────────


class TestMAE:
    def test_perfect_forecast(self) -> None:
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self) -> None:
        assert mae([1, 2, 3], [2, 3, 4]) == 1.0

    def test_symmetric(self) -> None:
        assert mae([10, 20], [12, 18]) == mae([12, 18], [10, 20])

    def test_single_element(self) -> None:
        assert mae([5], [3]) == 2.0

    def test_negative_values(self) -> None:
        assert mae([-1, -2], [-1, -2]) == 0.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mae([], [])

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            mae([1, 2], [1])


# ── RMSE ─────────────────────────────────────────────────────────────────────


class TestRMSE:
    def test_perfect_forecast(self) -> None:
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self) -> None:
        # errors: 1, 1, 1 → MSE = 1 → RMSE = 1
        assert rmse([1, 2, 3], [2, 3, 4]) == 1.0

    def test_penalises_large_errors(self) -> None:
        # MAE is the same, but RMSE differs
        a = [0, 0]
        # Spread errors: MAE = 2, RMSE = 2
        p1 = [2, 2]
        # Concentrated error: MAE = 2, RMSE = sqrt(8) ≈ 2.83
        p2 = [4, 0]
        assert rmse(a, p1) < rmse(a, p2)

    def test_rmse_ge_mae(self) -> None:
        a = [1, 5, 10, 20]
        p = [2, 4, 12, 15]
        assert rmse(a, p) >= mae(a, p)


# ── MAPE ─────────────────────────────────────────────────────────────────────


class TestMAPE:
    def test_perfect_forecast(self) -> None:
        assert mape([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self) -> None:
        # errors/actuals: 1/1, 1/2, 1/3 → mean ≈ 61.11%
        result = mape([1, 2, 3], [2, 3, 4])
        assert abs(result - 61.11) < 0.1

    def test_all_zeros_returns_nan(self) -> None:
        with pytest.warns(match="zero"):
            result = mape([0, 0, 0], [1, 2, 3])
        assert math.isnan(result)

    def test_some_zeros_masked(self) -> None:
        # Only non-zero actuals used: error = |10-12|/10 = 20%
        result = mape([0, 10], [5, 12])
        assert abs(result - 20.0) < 0.01


# ── WAPE ─────────────────────────────────────────────────────────────────────


class TestWAPE:
    def test_perfect_forecast(self) -> None:
        assert wape([10, 20, 30], [10, 20, 30]) == 0.0

    def test_known_value(self) -> None:
        # sum|errors| = 3, sum|actuals| = 6
        result = wape([1, 2, 3], [2, 3, 4])
        assert abs(result - 0.5) < 0.01

    def test_all_zeros_returns_nan(self) -> None:
        with pytest.warns(match="zero"):
            result = wape([0, 0], [1, 2])
        assert math.isnan(result)


# ── MASE ─────────────────────────────────────────────────────────────────────


class TestMASE:
    def test_perfect_forecast(self) -> None:
        train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert mase([11, 12], [11, 12], train) == 0.0

    def test_naive_equivalent_is_one(self) -> None:
        # If test errors equal the naive training errors, MASE = 1
        train = [1, 2, 3, 4, 5]
        # Naive errors: |2-1|=1, |3-2|=1, |4-3|=1, |5-4|=1 → mean=1
        # Test errors: [1, 1] → mean=1 → MASE = 1
        result = mase([6, 7], [5, 6], train, seasonality=1)
        assert abs(result - 1.0) < 0.01

    def test_constant_training_returns_nan(self) -> None:
        with pytest.warns(match="constant"):
            result = mase([5, 6], [5, 5], [3, 3, 3, 3, 3])
        assert math.isnan(result)

    def test_short_training_raises(self) -> None:
        with pytest.raises(ValueError, match="must exceed"):
            mase([1], [1], [1], seasonality=7)

    def test_seasonal_mase(self) -> None:
        # Weekly seasonality
        train = list(range(1, 22))  # 21 values, seasonality 7
        actual = [22, 23]
        predicted = [22, 23]
        result = mase(actual, predicted, train, seasonality=7)
        assert result == 0.0


# ── Pinball Loss ─────────────────────────────────────────────────────────────


class TestPinballLoss:
    def test_perfect_at_median(self) -> None:
        assert pinball_loss([5, 10], [5, 10], tau=0.5) == 0.0

    def test_over_prediction_at_high_quantile(self) -> None:
        # tau=0.9: over-predicting is penalised less
        loss_under = pinball_loss([10], [5], tau=0.9)  # under-predict
        loss_over = pinball_loss([10], [15], tau=0.9)  # over-predict
        assert loss_under > loss_over

    def test_invalid_tau_raises(self) -> None:
        with pytest.raises(ValueError, match="tau"):
            pinball_loss([1], [1], tau=0.0)
        with pytest.raises(ValueError, match="tau"):
            pinball_loss([1], [1], tau=1.0)


# ── Coverage & Width ─────────────────────────────────────────────────────────


class TestCoverageAndWidth:
    def test_perfect_coverage(self) -> None:
        assert empirical_coverage([5, 10], [0, 5], [10, 15]) == 1.0

    def test_zero_coverage(self) -> None:
        assert empirical_coverage([5, 10], [6, 11], [7, 12]) == 0.0

    def test_partial_coverage(self) -> None:
        assert empirical_coverage([5, 10], [0, 11], [6, 12]) == 0.5

    def test_interval_width(self) -> None:
        assert interval_width([0, 5], [10, 15]) == 10.0

    def test_width_shape_mismatch(self) -> None:
        with pytest.raises(ValueError):
            interval_width([0], [10, 15])


# ── Inventory Metrics ────────────────────────────────────────────────────────


class TestInventoryMetrics:
    def test_fill_rate_full(self) -> None:
        assert fill_rate([10, 20], [10, 20]) == 1.0

    def test_fill_rate_partial(self) -> None:
        assert fill_rate([10, 20], [5, 10]) == 0.5

    def test_fill_rate_zero_demand(self) -> None:
        assert fill_rate([0, 0], [0, 0]) == 1.0

    def test_stockout_periods_count(self) -> None:
        assert stockout_periods([10, 0, 5, -2, 0]) == 3

    def test_total_cost_structure(self) -> None:
        result = total_inventory_cost(
            inventory_levels=[100, 50, 0],
            stockout_units=[0, 0, 20],
            n_orders=2,
            holding_cost_per_unit_day=0.10,
            stockout_cost_per_unit=5.0,
            ordering_cost_per_order=50.0,
        )
        assert set(result.keys()) == {"holding", "stockout", "ordering", "total"}
        assert result["holding"] == 15.0  # (100+50+0) * 0.10
        assert result["stockout"] == 100.0  # 20 * 5.0
        assert result["ordering"] == 100.0  # 2 * 50
        assert result["total"] == 215.0


# ── Convenience Function ─────────────────────────────────────────────────────


class TestComputeAllPointMetrics:
    def test_returns_all_keys(self) -> None:
        result = compute_all_point_metrics([1, 2, 3], [2, 3, 4])
        assert {"mae", "rmse", "wape", "mape", "mase"} == set(result.keys())

    def test_mase_nan_without_training(self) -> None:
        result = compute_all_point_metrics([1, 2, 3], [2, 3, 4])
        assert math.isnan(result["mase"])

    def test_with_training_data(self) -> None:
        train = list(range(1, 20))
        result = compute_all_point_metrics([20, 21, 22], [20, 21, 22], training_actual=train)
        assert result["mase"] == 0.0
