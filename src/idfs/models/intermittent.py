"""Intermittent demand forecasting models: Croston and Teunter-Syntetos-Babai (TSB).

Specialised methods for zero-inflated time series with intermittent or lumpy demand patterns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class CrostonForecaster(Forecaster):
    """Croston's method for intermittent demand forecasting.

    Separately forecasts the non-zero demand size and the inter-arrival time
    between demand occurrences using exponential smoothing.

    Includes the Syntetos-Boylan Approximation (SBA) correction factor
    (1 - alpha/2) to eliminate positive bias.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        variant: str = "sba",
        name: str = "Croston",
    ) -> None:
        super().__init__(name=name)
        self.alpha = alpha
        self.variant = variant.lower()  # 'classic' or 'sba'
        self._forecasts: dict[tuple, float] = {}
        self._residual_stds: dict[tuple, float] = {}
        self._last_dates: dict[tuple, pd.Timestamp] = {}
        self._series_id_cols: list[str] = []

    @property
    def supports_global(self) -> bool:
        return False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> CrostonForecaster:
        self._series_id_cols = series_id_cols or []
        self._forecasts.clear()
        self._residual_stds.clear()
        self._last_dates.clear()

        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        groups: list[tuple[Any, pd.DataFrame]] = (
            list(df.groupby(self._series_id_cols, observed=True))
            if self._series_id_cols
            else [(None, df)]
        )

        for key, group in groups:
            k = key if isinstance(key, tuple) else (key,)
            sorted_group = group.sort_values("date")
            y = sorted_group[target_col].to_numpy(dtype=np.float64)
            self._last_dates[k] = sorted_group["date"].iloc[-1]

            non_zero_idx = np.where(y > 0)[0]
            if len(non_zero_idx) < 2:
                # Flat or almost all zero
                mean_val = float(np.mean(y)) if len(y) > 0 else 0.0
                self._forecasts[k] = mean_val
                self._residual_stds[k] = float(np.std(y)) if len(y) > 1 else 1.0
                continue

            # Initialise with first non-zero demand
            first_idx = non_zero_idx[0]
            z = float(y[first_idx])  # Demand size
            p = float(first_idx + 1)  # Demand interval
            q = 1  # Periods since last non-zero demand

            for t in range(first_idx + 1, len(y)):
                if y[t] > 0:
                    z = self.alpha * y[t] + (1 - self.alpha) * z
                    p = self.alpha * q + (1 - self.alpha) * p
                    q = 1
                else:
                    q += 1

            if p <= 0:
                p = 1.0

            forecast_rate = z / p
            if self.variant == "sba":
                # Syntetos-Boylan correction
                forecast_rate *= 1.0 - (self.alpha / 2.0)

            self._forecasts[k] = max(0.0, float(forecast_rate))
            # Empirical residual standard deviation
            resids = y - forecast_rate
            self._residual_stds[k] = float(np.std(resids)) if len(resids) > 1 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict")

        rows = []
        if features_df is not None and "date" in features_df.columns and self._series_id_cols:
            feat_df = features_df.copy()
            feat_df["date"] = pd.to_datetime(feat_df["date"])
            for key, group in feat_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                rate = self._forecasts.get(k, 0.0)
                sorted_group = group.sort_values("date")
                for d in sorted_group["date"]:
                    row: dict[str, Any] = {"date": d, "forecast": rate}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)
        else:
            for k, last_date in self._last_dates.items():
                start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                dates = pd.date_range(start=start_dt, periods=horizon, freq="D")
                rate = self._forecasts.get(k, 0.0)
                for d in dates:
                    row = {"date": d, "forecast": rate}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)

        return pd.DataFrame(rows)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pred_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)

        half_widths = []
        if self._series_id_cols and set(self._series_id_cols).issubset(set(pred_df.columns)):
            for tup in pred_df[self._series_id_cols].itertuples(index=False):
                k = tuple(tup)
                std = self._residual_stds.get(k, 1.0)
                half_widths.append(z * max(std, 0.1))
        else:
            default_std = (
                float(np.mean(list(self._residual_stds.values()))) if self._residual_stds else 1.0
            )
            half_widths = [z * max(default_std, 0.1)] * len(pred_df)

        arr_half = np.array(half_widths)
        pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - arr_half)
        pred_df["upper"] = pred_df["forecast"] + arr_half
        return pred_df


class TSBForecaster(Forecaster):
    """Teunter-Syntetos-Babai (TSB) forecaster.

    Updates demand probability at every period and demand size only when
    demand occurs, solving the obsolescence problem in Croston's method.
    """

    def __init__(
        self,
        alpha_d: float = 0.2,
        alpha_p: float = 0.2,
        name: str = "TSB",
    ) -> None:
        super().__init__(name=name)
        self.alpha_d = alpha_d  # Smoothing for demand size
        self.alpha_p = alpha_p  # Smoothing for demand probability
        self._forecasts: dict[tuple, float] = {}
        self._residual_stds: dict[tuple, float] = {}
        self._last_dates: dict[tuple, pd.Timestamp] = {}
        self._series_id_cols: list[str] = []

    @property
    def supports_global(self) -> bool:
        return False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> TSBForecaster:
        self._series_id_cols = series_id_cols or []
        self._forecasts.clear()
        self._residual_stds.clear()
        self._last_dates.clear()

        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        groups: list[tuple[Any, pd.DataFrame]] = (
            list(df.groupby(self._series_id_cols, observed=True))
            if self._series_id_cols
            else [(None, df)]
        )

        for key, group in groups:
            k = key if isinstance(key, tuple) else (key,)
            sorted_group = group.sort_values("date")
            y = sorted_group[target_col].to_numpy(dtype=np.float64)
            self._last_dates[k] = sorted_group["date"].iloc[-1]

            non_zero_idx = np.where(y > 0)[0]
            if len(non_zero_idx) == 0:
                self._forecasts[k] = 0.0
                self._residual_stds[k] = 1.0
                continue

            first_idx = non_zero_idx[0]
            z = float(y[first_idx])  # Demand size
            p = 1.0 / (first_idx + 1)  # Demand probability

            for t in range(first_idx + 1, len(y)):
                is_demand = 1.0 if y[t] > 0 else 0.0
                p = self.alpha_p * is_demand + (1.0 - self.alpha_p) * p
                if is_demand > 0:
                    z = self.alpha_d * y[t] + (1.0 - self.alpha_d) * z

            forecast_rate = max(0.0, p * z)
            self._forecasts[k] = float(forecast_rate)
            resids = y - forecast_rate
            self._residual_stds[k] = float(np.std(resids)) if len(resids) > 1 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict")

        rows = []
        if features_df is not None and "date" in features_df.columns and self._series_id_cols:
            feat_df = features_df.copy()
            feat_df["date"] = pd.to_datetime(feat_df["date"])
            for key, group in feat_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                rate = self._forecasts.get(k, 0.0)
                sorted_group = group.sort_values("date")
                for d in sorted_group["date"]:
                    row: dict[str, Any] = {"date": d, "forecast": rate}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)
        else:
            for k, last_date in self._last_dates.items():
                start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                dates = pd.date_range(start=start_dt, periods=horizon, freq="D")
                rate = self._forecasts.get(k, 0.0)
                for d in dates:
                    row = {"date": d, "forecast": rate}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)

        return pd.DataFrame(rows)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pred_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)

        half_widths = []
        if self._series_id_cols and set(self._series_id_cols).issubset(set(pred_df.columns)):
            for tup in pred_df[self._series_id_cols].itertuples(index=False):
                k = tuple(tup)
                std = self._residual_stds.get(k, 1.0)
                half_widths.append(z * max(std, 0.1))
        else:
            default_std = (
                float(np.mean(list(self._residual_stds.values()))) if self._residual_stds else 1.0
            )
            half_widths = [z * max(default_std, 0.1)] * len(pred_df)

        arr_half = np.array(half_widths)
        pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - arr_half)
        pred_df["upper"] = pred_df["forecast"] + arr_half
        return pred_df
