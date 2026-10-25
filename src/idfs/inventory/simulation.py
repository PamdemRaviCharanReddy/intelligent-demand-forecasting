"""Inventory policies and discrete-event simulation engine (Novelty N1).

Simulates daily multi-echelon or single-echelon inventory dynamics driven by
forecasts, lead times, safety stocks, and cost parameters:
    - Holding cost ($c_h$)
    - Stockout penalty cost ($c_s$)
    - Fixed ordering cost ($c_o$)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from idfs.evaluation.metrics import total_inventory_cost

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from an inventory simulation run."""

    daily_log: pd.DataFrame
    total_cost: dict[str, float]  # holding, stockout, ordering, total
    fill_rate: float
    stockout_periods: int
    avg_inventory: float
    n_orders: int
    service_level: float


class InventoryPolicy(ABC):
    """Abstract inventory replenishment policy."""

    @abstractmethod
    def compute_order(
        self,
        current_inventory_position: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        upper_quantile: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        """Decide how much to order.

        Args:
            current_inventory_position: On-hand + In-transit - Backorders.
            forecast: Point forecasts for protection period (lead_time + review_period).
            forecast_std: Estimated forecast error std.
            lead_time: Lead time in days.
            upper_quantile: Optional upper conformal/quantile forecast (for CQR policy).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for reporting."""


class NaivePolicy(InventoryPolicy):
    """Naive inventory policy: orders up to historical average demand level.

    Does not use forecasting models or dynamic lead-time uncertainty adjustments.
    """

    def __init__(
        self,
        avg_daily_demand: float,
        lead_time: int = 5,
        safety_days: int = 3,
        review_period: int = 7,
        name: str = "NaivePolicy",
    ) -> None:
        self._avg = max(0.0, avg_daily_demand)
        self._lead_time = lead_time
        self._safety_days = safety_days
        self._review_period = review_period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute_order(
        self,
        current_inventory_position: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        upper_quantile: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        protection = lead_time + self._review_period
        order_up_to = self._avg * (protection + self._safety_days)
        return max(0.0, float(order_up_to - current_inventory_position))


class ForecastDrivenPolicy(InventoryPolicy):
    """Standard Parametric (R, S) policy with Gaussian safety stock.

    Order-up-to level:
        S = sum(forecast) + z_alpha * forecast_std * sqrt(L + R)
    """

    def __init__(
        self,
        service_level: float = 0.95,
        review_period: int = 7,
        name: str | None = None,
    ) -> None:
        self._service_level = service_level
        self._review_period = review_period
        self._z = float(stats.norm.ppf(service_level))
        self._name = name or f"Gaussian-(R,S)-SL{service_level:.0%}"

    @property
    def name(self) -> str:
        return self._name

    def compute_order(
        self,
        current_inventory_position: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        upper_quantile: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        protection = lead_time + self._review_period
        window = min(len(forecast), protection)
        mean_demand = float(np.sum(forecast[:window])) if window > 0 else 0.0
        safety_stock = self._z * max(forecast_std, 0.1) * np.sqrt(protection)
        order_up_to = mean_demand + safety_stock
        return max(0.0, float(order_up_to - current_inventory_position))


class ConformalQuantilePolicy(InventoryPolicy):
    """Uncertainty-Aware Conformal Quantile Policy (Novelty N1 + N2).

    Directly leverages calibrated upper prediction intervals (e.g. q_0.95 or conformal bound)
    over the protection window (L + R), guaranteeing robust protection against stockouts
    under non-Gaussian, skewed, or intermittent demand without parametric assumptions.
    """

    def __init__(
        self,
        review_period: int = 7,
        name: str = "ConformalQuantile-(R,S)",
    ) -> None:
        self._review_period = review_period
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute_order(
        self,
        current_inventory_position: float,
        forecast: np.ndarray,
        forecast_std: float,
        lead_time: int,
        upper_quantile: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        protection = lead_time + self._review_period
        if upper_quantile is not None and len(upper_quantile) > 0:
            window = min(len(upper_quantile), protection)
            # The sum of upper bounds over the lead time + review period forms order-up-to
            order_up_to = float(np.sum(upper_quantile[:window]))
        else:
            # Fallback to mean + 2 std
            window = min(len(forecast), protection)
            mean_demand = float(np.sum(forecast[:window])) if window > 0 else 0.0
            order_up_to = mean_demand + 1.96 * max(forecast_std, 0.1) * np.sqrt(protection)

        return max(0.0, float(order_up_to - current_inventory_position))


class Simulator:
    """Daily discrete-event inventory simulation engine.

    Simulates day-by-day inventory dynamics:
    1. Order receipts from pipeline
    2. Customer demand fulfillment (lost sales assumption)
    3. Holding and stockout cost accumulation
    4. Periodic replenishment review and order placement
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
        self.lead_time = max(1, lead_time)
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.ordering_cost = ordering_cost
        self.review_period = max(1, review_period)
        self.initial_inventory = max(0.0, initial_inventory)

    def run(
        self,
        demand: np.ndarray,
        forecasts: np.ndarray,
        forecast_std: float = 1.0,
        upper_quantiles: np.ndarray | None = None,
        dates: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    ) -> SimulationResult:
        """Execute daily simulation across the evaluation horizon.

        Args:
            demand: True demand array.
            forecasts: Point forecasts aligned with demand.
            forecast_std: Standard deviation of forecast error.
            upper_quantiles: Upper quantile / conformal bound array.
            dates: Optional date timestamps.
        """
        n_days = len(demand)
        if n_days == 0:
            raise ValueError("Cannot simulate on empty demand array")

        # In-transit order pipeline: arrival_day -> order_qty
        pipeline: dict[int, float] = {}

        on_hand = self.initial_inventory
        daily_records: list[dict[str, Any]] = []

        holding_costs: list[float] = []
        stockout_costs: list[float] = []
        ordering_costs: list[float] = []

        total_demand = 0.0
        total_fulfilled = 0.0
        stockout_days = 0
        orders_placed = 0

        dates_arr = np.asarray(dates) if dates is not None else None

        for t in range(n_days):
            date_val = dates_arr[t] if dates_arr is not None else t

            # Step 1: Receive arriving orders
            receipts = pipeline.pop(t, 0.0)
            on_hand += receipts

            # Step 2: Customer demand fulfillment (lost sales)
            d_t = float(demand[t])
            total_demand += d_t
            fulfilled = min(on_hand, d_t)
            total_fulfilled += fulfilled
            unmet = max(0.0, d_t - fulfilled)
            if unmet > 0:
                stockout_days += 1

            on_hand -= fulfilled

            # Step 3: Cost calculation
            c_hold = on_hand * self.holding_cost
            c_stockout = unmet * self.stockout_cost
            c_order = 0.0

            holding_costs.append(c_hold)
            stockout_costs.append(c_stockout)

            # Step 4: Periodic replenishment review
            order_qty = 0.0
            if t % self.review_period == 0:
                # Calculate Inventory Position: On-hand + In-transit
                in_transit = sum(pipeline.values())
                inv_position = on_hand + in_transit

                f_window = forecasts[t : t + self.lead_time + self.review_period]
                u_window = (
                    upper_quantiles[t : t + self.lead_time + self.review_period]
                    if upper_quantiles is not None
                    else None
                )

                order_qty = self.policy.compute_order(
                    current_inventory_position=inv_position,
                    forecast=f_window,
                    forecast_std=forecast_std,
                    lead_time=self.lead_time,
                    upper_quantile=u_window,
                )

                if order_qty > 0.0:
                    arrival_day = t + self.lead_time
                    pipeline[arrival_day] = pipeline.get(arrival_day, 0.0) + order_qty
                    c_order = self.ordering_cost
                    orders_placed += 1

            ordering_costs.append(c_order)

            daily_records.append(
                {
                    "date": date_val,
                    "demand": d_t,
                    "forecast": float(forecasts[t]) if t < len(forecasts) else 0.0,
                    "receipts": receipts,
                    "inventory": on_hand,
                    "fulfilled": fulfilled,
                    "stockout_qty": unmet,
                    "order_qty": order_qty,
                    "holding_cost": c_hold,
                    "stockout_cost": c_stockout,
                    "ordering_cost": c_order,
                    "total_cost": c_hold + c_stockout + c_order,
                }
            )

        log_df = pd.DataFrame(daily_records)
        costs = total_inventory_cost(
            inventory_levels=log_df["inventory"],
            stockout_units=log_df["stockout_qty"],
            n_orders=orders_placed,
            holding_cost_per_unit_day=self.holding_cost,
            stockout_cost_per_unit=self.stockout_cost,
            ordering_cost_per_order=self.ordering_cost,
        )

        fill_rate = float(total_fulfilled / total_demand) if total_demand > 0 else 1.0
        service_level = float((n_days - stockout_days) / n_days) if n_days > 0 else 1.0
        avg_inv = float(log_df["inventory"].mean())

        return SimulationResult(
            daily_log=log_df,
            total_cost=costs,
            fill_rate=fill_rate,
            stockout_periods=stockout_days,
            avg_inventory=avg_inv,
            n_orders=orders_placed,
            service_level=service_level,
        )


def simulate_portfolio(
    simulator: Simulator,
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    series_id_cols: list[str],
    date_col: str = "date",
    target_col: str = "sales",
    forecast_col: str = "forecast",
    upper_col: str | None = None,
) -> dict[str, Any]:
    """Run simulation across multi-series portfolio and compute aggregate metrics."""
    total_holding = 0.0
    total_stockout = 0.0
    total_ordering = 0.0
    total_overall = 0.0
    all_demands = 0.0
    all_fulfilled = 0.0
    total_orders = 0
    total_stockout_periods = 0
    series_results: list[dict[str, Any]] = []

    f_cols_to_drop = [c for c in [target_col] if c in forecast_df.columns]
    f_df_clean = forecast_df.drop(columns=f_cols_to_drop)
    merged = pd.merge(
        actual_df,
        f_df_clean,
        on=[date_col, *series_id_cols],
        how="inner",
    ).sort_values(date_col)

    for key, group in merged.groupby(series_id_cols, observed=True):
        k = key if isinstance(key, tuple) else (key,)
        demands = group[target_col].to_numpy(dtype=np.float64)
        f_vals = group[forecast_col].to_numpy(dtype=np.float64)
        u_vals = group[upper_col].to_numpy(dtype=np.float64) if upper_col in group else None
        f_std = float(np.std(demands - f_vals)) if len(demands) > 1 else 1.0
        dates = pd.to_datetime(group[date_col]).dt.tz_localize(None)

        res = simulator.run(
            demand=demands,
            forecasts=f_vals,
            forecast_std=f_std,
            upper_quantiles=u_vals,
            dates=dates,
        )

        total_holding += res.total_cost["holding"]
        total_stockout += res.total_cost["stockout"]
        total_ordering += res.total_cost["ordering"]
        total_overall += res.total_cost["total"]
        all_demands += float(np.sum(demands))
        all_fulfilled += float(res.daily_log["fulfilled"].sum())
        total_orders += res.n_orders
        total_stockout_periods += res.stockout_periods

        series_dict: dict[str, Any] = {
            "series": str(k),
            "fill_rate": res.fill_rate,
            "total_cost": res.total_cost["total"],
            "holding_cost": res.total_cost["holding"],
            "stockout_cost": res.total_cost["stockout"],
            "ordering_cost": res.total_cost["ordering"],
        }
        for i, col in enumerate(series_id_cols):
            series_dict[col] = k[i]
        series_results.append(series_dict)

    portfolio_fill_rate = all_fulfilled / all_demands if all_demands > 0 else 1.0

    return {
        "policy": simulator.policy.name,
        "lead_time": simulator.lead_time,
        "holding_cost": round(total_holding, 2),
        "stockout_cost": round(total_stockout, 2),
        "ordering_cost": round(total_ordering, 2),
        "total_cost": round(total_overall, 2),
        "fill_rate": round(portfolio_fill_rate, 4),
        "stockout_periods": total_stockout_periods,
        "n_orders": total_orders,
        "series_details": pd.DataFrame(series_results),
    }
