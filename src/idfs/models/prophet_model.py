"""Additive forecasting models: Facebook Prophet and Fourier Harmonic Regression.

Decomposes time series into trend, weekly seasonality, yearly seasonality,
and exogenous effects (promotions, holidays).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)

# Check prophet availability
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetForecaster(Forecaster):
    """Facebook Prophet additive forecaster.

    Decomposes demand into trend and seasonality with optional holiday and
    promotional regressors. Fits per series independently.
    """

    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False,
        interval_width: float = 0.90,
        regressors: list[str] | None = None,
        name: str = "Prophet",
    ) -> None:
        super().__init__(name=name)
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.interval_width = interval_width
        self.regressors = regressors or []
        self._fitted_models: dict[tuple, Any] = {}
        self._last_dates: dict[tuple, pd.Timestamp] = {}
        self._series_id_cols: list[str] = []
        self._last_values: dict[tuple, float] = {}
        self._residual_stds: dict[tuple, float] = {}

    @property
    def supports_global(self) -> bool:
        return False

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> ProphetForecaster:
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is not installed. Install with `uv pip install prophet` or use FourierHarmonicForecaster."
            )

        self._series_id_cols = series_id_cols or []
        self._fitted_models.clear()
        self._last_dates.clear()
        self._last_values.clear()
        self._residual_stds.clear()

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
            last_date = sorted_group["date"].iloc[-1]
            self._last_dates[k] = last_date
            last_val = float(sorted_group[target_col].iloc[-1]) if len(sorted_group) > 0 else 0.0
            self._last_values[k] = last_val

            # Format dataframe for prophet (needs ds and y)
            prophet_df = pd.DataFrame(
                {
                    "ds": sorted_group["date"],
                    "y": sorted_group[target_col].astype(float),
                }
            )

            # Add available regressors
            active_regressors = []
            for reg in self.regressors:
                if reg in sorted_group.columns:
                    prophet_df[reg] = sorted_group[reg].astype(float).fillna(0.0)
                    active_regressors.append(reg)

            if len(prophet_df) < 5:
                self._fitted_models[k] = None
                self._residual_stds[k] = 1.0
                continue

            try:
                # Suppress prophet's cmdstan verbose logging
                m = Prophet(
                    weekly_seasonality=self.weekly_seasonality,
                    yearly_seasonality=self.yearly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    interval_width=self.interval_width,
                )
                for reg in active_regressors:
                    m.add_regressor(reg)

                # Fit silently
                import io
                import sys

                saved_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    m.fit(prophet_df)
                finally:
                    sys.stdout = saved_stdout

                self._fitted_models[k] = (m, active_regressors)
                in_sample = m.predict(prophet_df[["ds", *active_regressors]])
                resids = prophet_df["y"].to_numpy(dtype=float) - in_sample["yhat"].to_numpy(
                    dtype=float
                )
                self._residual_stds[k] = float(np.std(resids))
            except Exception as e:
                logger.debug("Prophet fit failed for %s: %s", k, e)
                self._fitted_models[k] = None
                self._residual_stds[k] = (
                    float(np.std(sorted_group[target_col])) if len(sorted_group) > 1 else 1.0
                )

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
                model_tuple = self._fitted_models.get(k)
                sorted_group = group.sort_values("date")

                if model_tuple is not None:
                    m, active_regs = model_tuple
                    future_df = pd.DataFrame({"ds": sorted_group["date"]})
                    for reg in active_regs:
                        if reg in sorted_group.columns:
                            future_df[reg] = sorted_group[reg].astype(float).fillna(0.0)
                        else:
                            future_df[reg] = 0.0

                    try:
                        preds = m.predict(future_df)["yhat"].to_numpy()
                    except Exception:
                        preds = np.full(len(sorted_group), self._last_values.get(k, 0.0))
                else:
                    preds = np.full(len(sorted_group), self._last_values.get(k, 0.0))

                preds = np.maximum(0.0, np.nan_to_num(preds, nan=self._last_values.get(k, 0.0)))
                for d, p in zip(sorted_group["date"], preds, strict=False):
                    row: dict[str, Any] = {"date": d, "forecast": float(p)}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)
        else:
            for k, last_date in self._last_dates.items():
                start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                dates = pd.date_range(start=start_dt, periods=horizon, freq="D")
                model_tuple = self._fitted_models.get(k)
                if model_tuple is not None:
                    m, active_regs = model_tuple
                    future_df = pd.DataFrame({"ds": dates})
                    for reg in active_regs:
                        future_df[reg] = 0.0
                    try:
                        preds = m.predict(future_df)["yhat"].to_numpy()
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
