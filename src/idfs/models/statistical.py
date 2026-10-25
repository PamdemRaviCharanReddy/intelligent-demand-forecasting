"""Statistical forecasting models: ARIMA, SARIMAX, and Exponential Smoothing.

Fits classical time-series models per series using statsmodels.
Implements robust convergence safeguards and automatic fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class SARIMAXForecaster(Forecaster):
    """Seasonal Autoregressive Integrated Moving Average (SARIMAX) forecaster.

    Fits independent SARIMAX models for each time series in the dataset.
    Default parameters are tailored for daily retail data with weekly seasonality:
    order=(1, 1, 1), seasonal_order=(0, 1, 1, 7).
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7),
        maxiter: int = 50,
        name: str = "SARIMAX",
    ) -> None:
        super().__init__(name=name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self._fitted_models: dict[tuple, Any] = {}
        self._residual_stds: dict[tuple, float] = {}
        self._last_dates: dict[tuple, pd.Timestamp] = {}
        self._series_id_cols: list[str] = []
        self._last_values: dict[tuple, float] = {}

    @property
    def supports_global(self) -> bool:
        return False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> SARIMAXForecaster:
        """Fit SARIMAX model per series.

        Args:
            train_df: Training data with date and target columns.
            target_col: Target column name.
            series_id_cols: Series identifier columns.
            **kwargs: Extra parameters.
        """
        self._series_id_cols = series_id_cols or []
        self._fitted_models.clear()
        self._residual_stds.clear()
        self._last_dates.clear()
        self._last_values.clear()

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
            series_data = sorted_group[target_col].to_numpy(dtype=np.float64)
            last_date = sorted_group["date"].iloc[-1]
            self._last_dates[k] = last_date
            self._last_values[k] = float(series_data[-1]) if len(series_data) > 0 else 0.0

            if len(series_data) < (self.seasonal_order[3] * 2):
                # Series too short for seasonal estimation; record fallback std
                self._fitted_models[k] = None
                diff = np.diff(series_data)
                self._residual_stds[k] = float(np.std(diff)) if len(diff) > 0 else 1.0
                continue

            try:
                model = SARIMAX(
                    series_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=self.maxiter)
                self._fitted_models[k] = res
                resids = res.resid
                self._residual_stds[k] = float(np.std(resids)) if len(resids) > 0 else 1.0
            except Exception as e:
                logger.debug("SARIMAX fit failed for series %s: %s. Using naive fallback.", k, e)
                # Try simple AR(1) fallback
                try:
                    fallback_model = SARIMAX(
                        series_data,
                        order=(1, 0, 0),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = fallback_model.fit(disp=False, maxiter=20)
                    self._fitted_models[k] = res
                    self._residual_stds[k] = float(np.std(res.resid))
                except Exception:
                    self._fitted_models[k] = None
                    self._residual_stds[k] = (
                        float(np.std(series_data)) if len(series_data) > 1 else 1.0
                    )

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Forecast horizon steps ahead for each series."""
        if not self._is_fitted:
            raise RuntimeError("Forecaster must be fitted before predict")

        rows = []
        if features_df is not None and "date" in features_df.columns and self._series_id_cols:
            feat_df = features_df.copy()
            feat_df["date"] = pd.to_datetime(feat_df["date"])
            for key, group in feat_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                steps = len(group)
                fitted_res = self._fitted_models.get(k)
                if fitted_res is not None:
                    try:
                        preds = fitted_res.forecast(steps=steps)
                    except Exception:
                        preds = np.full(steps, self._last_values.get(k, 0.0))
                else:
                    preds = np.full(steps, self._last_values.get(k, 0.0))

                preds = np.maximum(0.0, np.nan_to_num(preds, nan=self._last_values.get(k, 0.0)))
                sorted_group = group.sort_values("date")
                for d, p in zip(sorted_group["date"], preds, strict=False):
                    row: dict[str, Any] = {"date": d, "forecast": float(p)}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)
        else:
            # Generate synthetic date range
            for k, last_date in self._last_dates.items():
                start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                dates = pd.date_range(start=start_dt, periods=horizon, freq="D")
                fitted_res = self._fitted_models.get(k)
                if fitted_res is not None:
                    try:
                        preds = fitted_res.forecast(steps=horizon)
                    except Exception:
                        preds = np.full(horizon, self._last_values.get(k, 0.0))
                else:
                    preds = np.full(horizon, self._last_values.get(k, 0.0))

                preds = np.maximum(0.0, np.nan_to_num(preds, nan=self._last_values.get(k, 0.0)))
                for d, p in zip(dates, preds, strict=False):
                    row = {"date": d, "forecast": float(p)}
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
        """Compute forecast prediction intervals."""
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


class ARIMAForecaster(SARIMAXForecaster):
    """Non-seasonal ARIMA forecaster (alias with default seasonal_order=(0, 0, 0, 0))."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        maxiter: int = 50,
        name: str = "ARIMA",
    ) -> None:
        super().__init__(
            order=order,
            seasonal_order=(0, 0, 0, 0),
            maxiter=maxiter,
            name=name,
        )
