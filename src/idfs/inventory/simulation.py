"""Inventory policies and simulation.

Implements periodic and continuous review policies with configurable
assumptions. Simulates daily inventory dynamics driven by forecasts
and compares against a naive baseline policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Results from an inventory simulation run."""

    daily_log: (
        pd.DataFrame
    )  # Columns: date, demand, forecast, order_qty, inventory, stockout_qty, fulfilled
    total_cost: dict[str, float]  # holding, stockout, ordering, total
    fill_rate: float
    stockout_periods: int
    avg_inventory: float
    n_orders: int
    service_level: float


class InventoryPolicy(ABC):
    """Abstract inventory policy.

    Subclasses implement the ordering decision rule.
    All policies use the same Simulator, ensuring fair comparison.
    """

    @abstractmethod
    def compute_order(
        self,
        current_inventory: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        **kwargs: Any,
    ) -> float:
        """Decide how much to order.

        Args:
            current_inventory: Stock on hand.
            forecast: Point forecasts for the next `lead_time + review_period` days.
            forecast_std: Forecast error standard deviation (for safety stock).
            lead_time: Supplier lead time in days.
            **kwargs: Policy-specific parameters.

        Returns:
            Order quantity (non-negative). 0 means no order.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for reporting."""


class NaivePolicy(InventoryPolicy):
    """Naive inventory policy: order a fixed quantity when inventory < reorder point.

    Uses historical average demand (no forecasting model).
    This is the baseline that forecast-driven policies must beat.
    """

    def __init__(
        self,
        avg_daily_demand: float,
        lead_time: int = 5,
        safety_days: int = 3,
        review_period: int = 7,
    ) -> None:
        self._avg = avg_daily_demand
        self._lead_time = lead_time
        self._safety_days = safety_days
        self._review_period = review_period

    @property
    def name(self) -> str:
        return "NaivePolicy"

    def compute_order(
        self,
        current_inventory: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        **kwargs: Any,
    ) -> float:
        reorder_point = self._avg * (lead_time + self._safety_days)
        if current_inventory <= reorder_point:
            order_up_to = self._avg * (lead_time + self._review_period + self._safety_days)
            return max(0, order_up_to - current_inventory)
        return 0.0


class ForecastDrivenPolicy(InventoryPolicy):
    """Forecast-driven periodic review policy.

    Uses model forecasts and prediction intervals to set safety stock
    and order-up-to levels. This is the policy that should beat NaivePolicy
    by leveraging better demand estimates.
    """

    def __init__(
        self,
        service_level: float = 0.95,
        review_period: int = 7,
    ) -> None:
        from scipy import stats

        self._service_level = service_level
        self._review_period = review_period
        self._z = stats.norm.ppf(service_level)

    @property
    def name(self) -> str:
        return f"ForecastDriven-SL{self._service_level:.0%}"

    def compute_order(
        self,
        current_inventory: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        **kwargs: Any,
    ) -> float:
        protection_period = lead_time + self._review_period
        # Demand over protection period
        demand_forecast = float(np.sum(forecast[:protection_period]))
        # Safety stock
        safety_stock = self._z * forecast_std * np.sqrt(protection_period)
        # Order-up-to level
        order_up_to = demand_forecast + safety_stock
        order_qty = max(0.0, float(order_up_to - current_inventory))
        return float(order_qty)


class Simulator:
    """Daily inventory simulation engine.

    Steps through time day by day, applying an inventory policy,
    receiving orders after lead time, and recording all events.

    Usage:
        simulator = Simulator(policy, lead_time=5, ...)
        result = simulator.run(demand_series, forecast_series, forecast_std)
    """

    def __init__(
        self,
        policy: InventoryPolicy,
        lead_time: int = 5,
        holding_cost: float = 0.10,
        stockout_cost: float = 5.00,
        ordering_cost: float = 50.00,
        review_period: int = 7,
        initial_inventory: float = 100.0,
    ) -> None:
        self.policy = policy
        self.lead_time = lead_time
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.review_period = review_period
        self.initial_inventory = initial_inventory

    def run(
        self,
        demand: np.ndarray,
        forecasts: np.ndarray,
        forecast_std: float,
        dates: pd.DatetimeIndex | None = None,
    ) -> SimulationResult:
        """Run the inventory simulation.

        Args:
            demand: Daily true demand.
            forecasts: Daily point forecasts (aligned with demand).
            forecast_std: Forecast error std (scalar for simplicity).
            dates: Optional date index for the daily log.

        Returns:
            SimulationResult with detailed daily log and summary metrics.
        """
        # Stub — full implementation in Phase 4
        raise NotImplementedError("Full simulation implementation in Phase 4")


class DriftMonitor:
    """Rolling-error drift monitor.

    Tracks forecast error over a rolling window and triggers a retrain
    signal when error exceeds a threshold relative to the baseline.

    Attributes:
        window_size: Number of recent observations to monitor.
        threshold_multiplier: Trigger retrain if rolling error > baseline * this.
    """

    def __init__(
        self,
        window_size: int = 28,
        threshold_multiplier: float = 1.5,
        metric: str = "mae",
    ) -> None:
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.metric = metric
        self._baseline_error: float | None = None
        self._history: list[dict[str, Any]] = []

    def set_baseline(self, baseline_error: float) -> None:
        """Set the baseline error from initial training.

        Args:
            baseline_error: Error metric value from initial backtest.
        """
        self._baseline_error = baseline_error

    def update(
        self,
        actual: float,
        predicted: float,
        timestamp: pd.Timestamp | None = None,
    ) -> bool:
        """Add a new observation and check for drift.

        Args:
            actual: True demand value.
            predicted: Forecasted value.
            timestamp: Optional timestamp.

        Returns:
            True if drift is detected (retrain recommended).
        """
        # Stub — full implementation in Phase 5
        raise NotImplementedError("Full drift monitor in Phase 5")

    @property
    def should_retrain(self) -> bool:
        """Whether the most recent check detected drift."""
        raise NotImplementedError("Full drift monitor in Phase 5")
