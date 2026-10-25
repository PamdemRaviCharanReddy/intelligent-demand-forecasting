"""Abstract base class for all forecasting models.

Every model (Naive, ARIMA, XGBoost, LSTM, etc.) implements this interface,
which guarantees it plugs into the same backtest, simulation, and evaluation
pipeline without special-casing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """Abstract forecaster interface.

    Subclasses must implement fit(), predict(), and predict_interval().
    The backtest harness calls these methods with consistent signatures.

    Attributes:
        _name: Human-readable model name.
        _is_fitted: Whether fit() has been called.
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self._name = name
        self._is_fitted = False
        self._params = kwargs

    @property
    def name(self) -> str:
        """Human-readable model name (e.g. 'XGBoost-global')."""
        return self._name

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    @abstractmethod
    def supports_global(self) -> bool:
        """Whether this model can train on multiple series at once.

        Global models (XGBoost, LSTM) train on all series jointly.
        Local models (ARIMA, Prophet) train on one series at a time.
        """

    @abstractmethod
    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> Forecaster:
        """Fit the model on training data.

        Args:
            train_df: Training DataFrame with at least [date, target_col] columns.
                For global models, also contains series_id_cols and features.
            target_col: Name of the target column.
            series_id_cols: Columns identifying individual series.
            **kwargs: Model-specific extra arguments.

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate point forecasts.

        Args:
            horizon: Number of periods to forecast.
            features_df: Future feature values (required for ML models).
            **kwargs: Model-specific extra arguments.

        Returns:
            DataFrame with columns [date, series_id_cols..., 'forecast'].
        """

    @abstractmethod
    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate prediction intervals.

        Args:
            horizon: Number of periods to forecast.
            coverage: Nominal coverage level (e.g. 0.90 for 90% intervals).
            features_df: Future feature values (required for ML models).
            **kwargs: Model-specific extra arguments.

        Returns:
            DataFrame with columns [date, series_id_cols..., 'forecast',
            'lower', 'upper'].
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}', fitted={self._is_fitted})"


class NaiveForecaster(Forecaster):
    """Last-value naive forecast.

    Predicts the last observed value for all future periods.
    The simplest possible baseline — any useful model must beat this.
    """

    def __init__(self, name: str = "Naive") -> None:
        super().__init__(name="Naive")
        self._last_values: dict[tuple, float] = {}
        self._residual_stds: dict[tuple, float] = {}
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
    ) -> NaiveForecaster:
        self._series_id_cols = series_id_cols or []
        if self._series_id_cols:
            for key, group in train_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                sorted_group = group.sort_values("date")
                self._last_values[k] = float(sorted_group[target_col].iloc[-1])
                diffs = sorted_group[target_col].diff().dropna()
                self._residual_stds[k] = float(diffs.std()) if len(diffs) > 1 else 1.0
        else:
            sorted_df = train_df.sort_values("date")
            self._last_values[()] = float(sorted_df[target_col].iloc[-1])
            diffs = sorted_df[target_col].diff().dropna()
            self._residual_stds[()] = float(diffs.std()) if len(diffs) > 1 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if features_df is not None and len(features_df) > 0:
            out_df = features_df.copy()
            forecasts = []
            if self._series_id_cols:
                for tup in features_df[self._series_id_cols].itertuples(index=False):
                    k = tuple(tup)
                    forecasts.append(self._last_values.get(k, 0.0))
            else:
                forecasts = [self._last_values.get((), 0.0)] * len(features_df)

            res_cols = ["date"] if "date" in out_df.columns else []
            for col in self._series_id_cols:
                if col in out_df.columns and col not in res_cols:
                    res_cols.append(col)
            result = out_df[res_cols].copy() if res_cols else pd.DataFrame(index=out_df.index)
            result["forecast"] = forecasts
            return result

        rows = []
        for key, value in self._last_values.items():
            for h in range(horizon):
                r_dict: dict[str, Any] = {"horizon_step": h + 1, "forecast": value}
                if key and self._series_id_cols:
                    for i, col_val in enumerate(key):
                        r_dict[self._series_id_cols[i]] = col_val
                rows.append(r_dict)
        return pd.DataFrame(rows)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from scipy import stats

        pred_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)

        if self._series_id_cols and set(self._series_id_cols).issubset(set(pred_df.columns)):
            half_widths = []
            for tup in pred_df[self._series_id_cols].itertuples(index=False):
                k = tuple(tup)
                std = self._residual_stds.get(k, 1.0)
                half_widths.append(z * std)
            pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - np.array(half_widths))
            pred_df["upper"] = pred_df["forecast"] + np.array(half_widths)
        else:
            default_std = (
                float(np.mean(list(self._residual_stds.values()))) if self._residual_stds else 1.0
            )
            pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - z * default_std)
            pred_df["upper"] = pred_df["forecast"] + z * default_std

        return pred_df


class SeasonalNaiveForecaster(Forecaster):
    """Seasonal naive forecast.

    Repeats the values from the last observed seasonal cycle.
    For daily data with weekly seasonality, predicts using values from 7 days ago.
    """

    def __init__(self, seasonality: int = 7, name: str | None = None) -> None:
        super().__init__(name=name or f"SeasonalNaive-{seasonality}")
        self._seasonality = seasonality
        self._last_season: dict[tuple, np.ndarray] = {}
        self._residual_stds: dict[tuple, float] = {}
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
    ) -> SeasonalNaiveForecaster:
        self._series_id_cols = series_id_cols or []
        if self._series_id_cols:
            for key, group in train_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                sorted_group = group.sort_values("date")
                values = np.asarray(sorted_group[target_col], dtype=np.float64)
                self._last_season[k] = values[-self._seasonality :]
                # Residual std: std of seasonal differences
                if len(values) > self._seasonality:
                    diffs = values[self._seasonality :] - values[: -self._seasonality]
                    self._residual_stds[k] = float(np.std(diffs))
                else:
                    self._residual_stds[k] = float(np.std(values)) if len(values) > 1 else 1.0
        else:
            sorted_df = train_df.sort_values("date")
            values = np.asarray(sorted_df[target_col], dtype=np.float64)
            self._last_season[()] = values[-self._seasonality :]
            if len(values) > self._seasonality:
                diffs = values[self._seasonality :] - values[: -self._seasonality]
                self._residual_stds[()] = float(np.std(diffs))
            else:
                self._residual_stds[()] = float(np.std(values)) if len(values) > 1 else 1.0

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if features_df is not None and len(features_df) > 0:
            out_df = features_df.copy()
            forecasts = []
            if self._series_id_cols:
                series_counters: dict[tuple, int] = {}
                for tup in features_df[self._series_id_cols].itertuples(index=False):
                    k = tuple(tup)
                    step = series_counters.get(k, 0)
                    season_vals = self._last_season.get(k, np.array([0.0]))
                    idx = step % self._seasonality
                    forecasts.append(float(season_vals[idx]) if idx < len(season_vals) else 0.0)
                    series_counters[k] = step + 1
            else:
                for step in range(len(features_df)):
                    season_vals = self._last_season.get((), np.array([0.0]))
                    idx = step % self._seasonality
                    forecasts.append(float(season_vals[idx]) if idx < len(season_vals) else 0.0)

            res_cols = ["date"] if "date" in out_df.columns else []
            for col in self._series_id_cols:
                if col in out_df.columns and col not in res_cols:
                    res_cols.append(col)
            result = out_df[res_cols].copy() if res_cols else pd.DataFrame(index=out_df.index)
            result["forecast"] = forecasts
            return result

        rows = []
        for key, season_vals in self._last_season.items():
            for h in range(horizon):
                idx = h % self._seasonality
                forecast = float(season_vals[idx]) if idx < len(season_vals) else 0.0
                r_dict: dict[str, Any] = {"horizon_step": h + 1, "forecast": forecast}
                if key and self._series_id_cols:
                    for i, col_val in enumerate(key):
                        r_dict[self._series_id_cols[i]] = col_val
                rows.append(r_dict)
        return pd.DataFrame(rows)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        from scipy import stats

        pred_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)

        if self._series_id_cols and set(self._series_id_cols).issubset(set(pred_df.columns)):
            half_widths = []
            for tup in pred_df[self._series_id_cols].itertuples(index=False):
                k = tuple(tup)
                std = self._residual_stds.get(k, 1.0)
                half_widths.append(z * std)
            pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - np.array(half_widths))
            pred_df["upper"] = pred_df["forecast"] + np.array(half_widths)
        else:
            default_std = (
                float(np.mean(list(self._residual_stds.values()))) if self._residual_stds else 1.0
            )
            pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - z * default_std)
            pred_df["upper"] = pred_df["forecast"] + z * default_std

        return pred_df
