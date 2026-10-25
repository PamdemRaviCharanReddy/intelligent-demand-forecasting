"""Evaluation metrics for demand forecasting.

All metrics are implemented as pure functions with consistent signatures.
Edge cases (zero actuals, empty arrays, short series) are handled explicitly.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike


def _to_arrays(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float64 numpy arrays and validate."""
    a = np.asarray(actual, dtype=np.float64).ravel()
    p = np.asarray(predicted, dtype=np.float64).ravel()
    if a.shape != p.shape:
        raise ValueError(f"Shape mismatch: actual {a.shape} vs predicted {p.shape}")
    if len(a) == 0:
        raise ValueError("Cannot compute metrics on empty arrays")
    return a, p


# ── Point Forecast Metrics ───────────────────────────────────────────────────


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error.

    Args:
        actual: True values.
        predicted: Forecast values.

    Returns:
        MAE (non-negative float). Lower is better.
    """
    a, p = _to_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Squared Error.

    Penalises large errors more than MAE.

    Args:
        actual: True values.
        predicted: Forecast values.

    Returns:
        RMSE (non-negative float). Lower is better.
    """
    a, p = _to_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Percentage Error (zero-masked).

    Observations where actual == 0 are excluded to avoid division by zero.
    Returns NaN if all actuals are zero.

    Args:
        actual: True values.
        predicted: Forecast values.

    Returns:
        MAPE as a percentage (e.g. 15.2 means 15.2%). Lower is better.
    """
    a, p = _to_arrays(actual, predicted)
    mask = a != 0
    if not mask.any():
        warnings.warn("All actual values are zero; MAPE is undefined.", stacklevel=2)
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def wape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Weighted Absolute Percentage Error.

    WAPE = sum(|actual - predicted|) / sum(|actual|).
    More robust than MAPE for series with zeros.
    Returns NaN if sum of actuals is zero.

    Args:
        actual: True values.
        predicted: Forecast values.

    Returns:
        WAPE as a ratio (e.g. 0.15 means 15%). Lower is better.
    """
    a, p = _to_arrays(actual, predicted)
    total = np.sum(np.abs(a))
    if total == 0:
        warnings.warn("Sum of actual values is zero; WAPE is undefined.", stacklevel=2)
        return float("nan")
    return float(np.sum(np.abs(a - p)) / total)


def mase(
    actual: ArrayLike,
    predicted: ArrayLike,
    training_actual: ArrayLike,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scaled by the in-sample naive (seasonal) forecast error.
    MASE < 1 means the model beats the naive baseline.

    Args:
        actual: True test values.
        predicted: Forecast values.
        training_actual: In-sample (training) actual values for scaling.
        seasonality: Seasonal period (1 for naive, 7 for weekly, etc.).

    Returns:
        MASE (non-negative float). Lower is better. < 1 beats naive.

    Raises:
        ValueError: If training series is too short for the given seasonality.
    """
    a, p = _to_arrays(actual, predicted)
    train = np.asarray(training_actual, dtype=np.float64).ravel()

    if len(train) <= seasonality:
        raise ValueError(
            f"Training series length ({len(train)}) must exceed seasonality ({seasonality})"
        )

    # In-sample naive seasonal error
    naive_errors = np.abs(train[seasonality:] - train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0:
        warnings.warn(
            "Training series has zero naive error (constant series); MASE is undefined.",
            stacklevel=2,
        )
        return float("nan")

    return float(np.mean(np.abs(a - p)) / scale)


# ── Interval / Uncertainty Metrics ───────────────────────────────────────────


def pinball_loss(actual: ArrayLike, predicted_quantile: ArrayLike, tau: float) -> float:
    """Pinball (quantile) loss.

    Args:
        actual: True values.
        predicted_quantile: Predicted quantile values.
        tau: Quantile level (e.g. 0.1, 0.5, 0.9).

    Returns:
        Average pinball loss (non-negative float). Lower is better.

    Raises:
        ValueError: If tau not in (0, 1).
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")

    a, q = _to_arrays(actual, predicted_quantile)
    diff = a - q
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def empirical_coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Empirical coverage of prediction intervals.

    Args:
        actual: True values.
        lower: Lower bound of prediction intervals.
        upper: Upper bound of prediction intervals.

    Returns:
        Coverage as a ratio in [0, 1]. E.g. 0.90 means 90% of actuals
        fall within the intervals.
    """
    a = np.asarray(actual, dtype=np.float64).ravel()
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if not (a.shape == lo.shape == hi.shape):
        raise ValueError("All arrays must have the same shape")
    if len(a) == 0:
        raise ValueError("Cannot compute coverage on empty arrays")
    covered = (a >= lo) & (a <= hi)
    return float(np.mean(covered))


def interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Mean width of prediction intervals.

    Narrower intervals (at the same coverage) indicate sharper forecasts.

    Args:
        lower: Lower bounds.
        upper: Upper bounds.

    Returns:
        Mean interval width (non-negative float). Lower is better (at same coverage).
    """
    lo = np.asarray(lower, dtype=np.float64).ravel()
    hi = np.asarray(upper, dtype=np.float64).ravel()
    if lo.shape != hi.shape:
        raise ValueError("Lower and upper must have the same shape")
    widths = hi - lo
    if np.any(widths < 0):
        warnings.warn("Some intervals have upper < lower.", stacklevel=2)
    return float(np.mean(np.abs(widths)))


# ── Inventory Metrics ────────────────────────────────────────────────────────


def fill_rate(demand: ArrayLike, fulfilled: ArrayLike) -> float:
    """Fill rate: fraction of total demand that was fulfilled.

    Args:
        demand: True demand quantities per period.
        fulfilled: Quantities actually fulfilled per period.

    Returns:
        Fill rate in [0, 1]. Higher is better.
    """
    d = np.asarray(demand, dtype=np.float64).ravel()
    f = np.asarray(fulfilled, dtype=np.float64).ravel()
    total_demand = np.sum(d)
    if total_demand == 0:
        return 1.0  # No demand → trivially fulfilled
    return float(np.clip(np.sum(f) / total_demand, 0.0, 1.0))


def stockout_periods(inventory_levels: ArrayLike) -> int:
    """Count the number of periods where inventory was zero or negative.

    Args:
        inventory_levels: End-of-period inventory levels.

    Returns:
        Number of stockout periods (non-negative int).
    """
    inv = np.asarray(inventory_levels, dtype=np.float64).ravel()
    return int(np.sum(inv <= 0))


def total_inventory_cost(
    inventory_levels: ArrayLike,
    stockout_units: ArrayLike,
    n_orders: int,
    holding_cost_per_unit_day: float,
    stockout_cost_per_unit: float,
    ordering_cost_per_order: float,
) -> dict[str, float]:
    """Compute total inventory cost broken down by component.

    Args:
        inventory_levels: End-of-period inventory levels (positive = stock on hand).
        stockout_units: Unmet demand per period (non-negative).
        n_orders: Total number of replenishment orders placed.
        holding_cost_per_unit_day: Cost per unit per day of holding inventory.
        stockout_cost_per_unit: Cost per unit of unmet demand.
        ordering_cost_per_order: Fixed cost per order placed.

    Returns:
        Dict with keys 'holding', 'stockout', 'ordering', 'total'.
    """
    inv = np.asarray(inventory_levels, dtype=np.float64).ravel()
    so = np.asarray(stockout_units, dtype=np.float64).ravel()

    holding = float(np.sum(np.maximum(inv, 0)) * holding_cost_per_unit_day)
    stockout = float(np.sum(so) * stockout_cost_per_unit)
    ordering = float(n_orders * ordering_cost_per_order)
    total = holding + stockout + ordering

    return {
        "holding": round(holding, 2),
        "stockout": round(stockout, 2),
        "ordering": round(ordering, 2),
        "total": round(total, 2),
    }


# ── Convenience ──────────────────────────────────────────────────────────────


def compute_all_point_metrics(
    actual: ArrayLike,
    predicted: ArrayLike,
    training_actual: ArrayLike | None = None,
    seasonality: int = 1,
) -> dict[str, float]:
    """Compute all point forecast metrics at once.

    Args:
        actual: True values.
        predicted: Forecast values.
        training_actual: Training series (needed for MASE). If None, MASE is NaN.
        seasonality: Seasonal period for MASE.

    Returns:
        Dict of metric_name → value.
    """
    results = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "wape": wape(actual, predicted),
        "mape": mape(actual, predicted),
    }

    if training_actual is not None:
        try:
            results["mase"] = mase(actual, predicted, training_actual, seasonality)
        except ValueError:
            results["mase"] = float("nan")
    else:
        results["mase"] = float("nan")

    return results
