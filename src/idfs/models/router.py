"""Demand-pattern router based on Syntetos-Boylan ADI vs CV^2 classification (Novelty N3).

Categorises demand series into 4 distinct quadrants:
1. Smooth: ADI < 1.32, CV^2 < 0.49
2. Erratic: ADI < 1.32, CV^2 >= 0.49
3. Intermittent: ADI >= 1.32, CV^2 < 0.49
4. Lumpy: ADI >= 1.32, CV^2 >= 0.49

Routes each time series to the optimal forecasting model family.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from idfs.models.base import Forecaster, SeasonalNaiveForecaster
from idfs.models.intermittent import CrostonForecaster, TSBForecaster
from idfs.models.xgboost import XGBoostForecaster

logger = logging.getLogger(__name__)


def classify_demand_pattern(
    series: np.ndarray,
    adi_cutoff: float = 1.32,
    cv2_cutoff: float = 0.49,
) -> tuple[float, float, str]:
    """Compute ADI and CV^2 and determine Syntetos-Boylan demand quadrant.

    Args:
        series: Array of historical demand observations.
        adi_cutoff: Threshold for Average Demand Interval (default: 1.32).
        cv2_cutoff: Threshold for squared coefficient of variation (default: 0.49).

    Returns:
        tuple of (adi, cv2, category_name)
    """
    n = len(series)
    if n == 0:
        return 1.0, 0.0, "smooth"

    non_zero = series[series > 0]
    n_non_zero = len(non_zero)

    if n_non_zero == 0:
        # All zero demand
        return float(n), 0.0, "intermittent"

    adi = float(n) / float(n_non_zero)

    mean_z = float(np.mean(non_zero))
    if mean_z <= 1e-6:
        cv2 = 0.0
    else:
        std_z = float(np.std(non_zero, ddof=1)) if n_non_zero > 1 else 0.0
        cv2 = (std_z / mean_z) ** 2

    # Determine quadrant
    if adi < adi_cutoff:
        category = "smooth" if cv2 < cv2_cutoff else "erratic"
    else:
        category = "intermittent" if cv2 < cv2_cutoff else "lumpy"

    return adi, cv2, category


class DemandPatternRouterForecaster(Forecaster):
    """Dynamic demand-pattern routing forecaster (Novelty N3).

    Inspects each series during train_df.fit() and delegates to:
    - Smooth series -> Global XGBoost (or SARIMAX)
    - Erratic series -> Seasonal Naive (or tuned XGBoost)
    - Intermittent series -> Croston (SBA)
    - Lumpy series -> TSB
    """

    def __init__(
        self,
        adi_cutoff: float = 1.32,
        cv2_cutoff: float = 0.49,
        smooth_model: Forecaster | None = None,
        erratic_model: Forecaster | None = None,
        intermittent_model: Forecaster | None = None,
        lumpy_model: Forecaster | None = None,
        name: str = "DemandPatternRouter",
    ) -> None:
        super().__init__(name=name)
        self.adi_cutoff = adi_cutoff
        self.cv2_cutoff = cv2_cutoff

        # Default expert assignments
        self.smooth_model = smooth_model or XGBoostForecaster(name="Router-Smooth-XGB")
        self.erratic_model = erratic_model or SeasonalNaiveForecaster(name="Router-Erratic-SNaive")
        self.intermittent_model = intermittent_model or CrostonForecaster(
            name="Router-Intermittent-Croston"
        )
        self.lumpy_model = lumpy_model or TSBForecaster(name="Router-Lumpy-TSB")

        self.series_patterns_: dict[tuple, dict[str, Any]] = {}
        self._series_id_cols: list[str] = []

    @property
    def supports_global(self) -> bool:
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> DemandPatternRouterForecaster:
        """Classify each series and train designated expert models."""
        self._series_id_cols = series_id_cols or []
        self.series_patterns_.clear()

        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        groups: list[tuple[Any, pd.DataFrame]] = (
            list(df.groupby(self._series_id_cols, observed=True))
            if self._series_id_cols
            else [(None, df)]
        )

        quadrant_series: dict[str, list[tuple]] = {
            "smooth": [],
            "erratic": [],
            "intermittent": [],
            "lumpy": [],
        }

        # Step 1: Compute metrics on historical train split only (zero leakage)
        for key, group in groups:
            k = key if isinstance(key, tuple) else (key,)
            sorted_group = group.sort_values("date")
            y = sorted_group[target_col].to_numpy(dtype=np.float64)

            adi, cv2, cat = classify_demand_pattern(
                y, adi_cutoff=self.adi_cutoff, cv2_cutoff=self.cv2_cutoff
            )
            self.series_patterns_[k] = {
                "adi": adi,
                "cv2": cv2,
                "category": cat,
            }
            quadrant_series[cat].append(k)

        logger.info(
            "Syntetos-Boylan Demand Routing: Smooth=%d, Erratic=%d, Intermittent=%d, Lumpy=%d",
            len(quadrant_series["smooth"]),
            len(quadrant_series["erratic"]),
            len(quadrant_series["intermittent"]),
            len(quadrant_series["lumpy"]),
        )

        # Helper to filter DataFrame by series keys
        def filter_df_by_keys(target_df: pd.DataFrame, keys: list[tuple]) -> pd.DataFrame:
            if not keys or not self._series_id_cols:
                return pd.DataFrame(columns=target_df.columns)
            mask = np.zeros(len(target_df), dtype=bool)
            for k in keys:
                k_mask = np.ones(len(target_df), dtype=bool)
                for i, col in enumerate(self._series_id_cols):
                    k_mask &= target_df[col] == k[i]
                mask |= k_mask
            return target_df[mask].copy()

        # Step 2: Fit designated models
        # Smooth model
        smooth_train = filter_df_by_keys(train_df, quadrant_series["smooth"])
        if len(smooth_train) > 0:
            smooth_val = (
                filter_df_by_keys(val_df, quadrant_series["smooth"]) if val_df is not None else None
            )
            self.smooth_model.fit(
                smooth_train,
                target_col=target_col,
                series_id_cols=self._series_id_cols,
                val_df=smooth_val,
                **kwargs,
            )

        # Erratic model
        erratic_train = filter_df_by_keys(train_df, quadrant_series["erratic"])
        if len(erratic_train) > 0:
            erratic_val = (
                filter_df_by_keys(val_df, quadrant_series["erratic"])
                if val_df is not None
                else None
            )
            self.erratic_model.fit(
                erratic_train,
                target_col=target_col,
                series_id_cols=self._series_id_cols,
                val_df=erratic_val,
                **kwargs,
            )

        # Intermittent model
        intermittent_train = filter_df_by_keys(train_df, quadrant_series["intermittent"])
        if len(intermittent_train) > 0:
            self.intermittent_model.fit(
                intermittent_train,
                target_col=target_col,
                series_id_cols=self._series_id_cols,
                **kwargs,
            )

        # Lumpy model
        lumpy_train = filter_df_by_keys(train_df, quadrant_series["lumpy"])
        if len(lumpy_train) > 0:
            self.lumpy_model.fit(
                lumpy_train,
                target_col=target_col,
                series_id_cols=self._series_id_cols,
                **kwargs,
            )

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Route predictions to their designated model and assemble result."""
        if not self._is_fitted:
            raise RuntimeError("DemandPatternRouterForecaster must be fitted before predict")

        if features_df is None:
            raise ValueError("Router requires features_df for multi-series routing")

        # Partition features_df by category
        quadrant_dfs: dict[str, list[pd.DataFrame]] = {
            "smooth": [],
            "erratic": [],
            "intermittent": [],
            "lumpy": [],
        }

        if self._series_id_cols:
            for key, group in features_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                cat = self.series_patterns_.get(k, {}).get("category", "smooth")
                quadrant_dfs[cat].append(group)
        else:
            quadrant_dfs["smooth"].append(features_df)

        pred_parts: list[pd.DataFrame] = []

        # Smooth
        if quadrant_dfs["smooth"] and self.smooth_model.is_fitted:
            smooth_feats = pd.concat(quadrant_dfs["smooth"], ignore_index=True)
            p = self.smooth_model.predict(horizon=horizon, features_df=smooth_feats, **kwargs)
            pred_parts.append(p)

        # Erratic
        if quadrant_dfs["erratic"] and self.erratic_model.is_fitted:
            erratic_feats = pd.concat(quadrant_dfs["erratic"], ignore_index=True)
            p = self.erratic_model.predict(horizon=horizon, features_df=erratic_feats, **kwargs)
            pred_parts.append(p)

        # Intermittent
        if quadrant_dfs["intermittent"] and self.intermittent_model.is_fitted:
            inter_feats = pd.concat(quadrant_dfs["intermittent"], ignore_index=True)
            p = self.intermittent_model.predict(horizon=horizon, features_df=inter_feats, **kwargs)
            pred_parts.append(p)

        # Lumpy
        if quadrant_dfs["lumpy"] and self.lumpy_model.is_fitted:
            lumpy_feats = pd.concat(quadrant_dfs["lumpy"], ignore_index=True)
            p = self.lumpy_model.predict(horizon=horizon, features_df=lumpy_feats, **kwargs)
            pred_parts.append(p)

        if not pred_parts:
            return pd.DataFrame(columns=["date", "forecast", *self._series_id_cols])

        out_df = pd.concat(pred_parts, ignore_index=True)
        return out_df

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Route interval predictions to each expert model."""
        if not self._is_fitted:
            raise RuntimeError("DemandPatternRouterForecaster must be fitted before predict")

        if features_df is None:
            raise ValueError("Router requires features_df")

        quadrant_dfs: dict[str, list[pd.DataFrame]] = {
            "smooth": [],
            "erratic": [],
            "intermittent": [],
            "lumpy": [],
        }

        if self._series_id_cols:
            for key, group in features_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                cat = self.series_patterns_.get(k, {}).get("category", "smooth")
                quadrant_dfs[cat].append(group)
        else:
            quadrant_dfs["smooth"].append(features_df)

        interval_parts: list[pd.DataFrame] = []

        if quadrant_dfs["smooth"] and self.smooth_model.is_fitted:
            s_df = pd.concat(quadrant_dfs["smooth"], ignore_index=True)
            interval_parts.append(
                self.smooth_model.predict_interval(
                    horizon=horizon, coverage=coverage, features_df=s_df, **kwargs
                )
            )

        if quadrant_dfs["erratic"] and self.erratic_model.is_fitted:
            e_df = pd.concat(quadrant_dfs["erratic"], ignore_index=True)
            interval_parts.append(
                self.erratic_model.predict_interval(
                    horizon=horizon, coverage=coverage, features_df=e_df, **kwargs
                )
            )

        if quadrant_dfs["intermittent"] and self.intermittent_model.is_fitted:
            i_df = pd.concat(quadrant_dfs["intermittent"], ignore_index=True)
            interval_parts.append(
                self.intermittent_model.predict_interval(
                    horizon=horizon, coverage=coverage, features_df=i_df, **kwargs
                )
            )

        if quadrant_dfs["lumpy"] and self.lumpy_model.is_fitted:
            l_df = pd.concat(quadrant_dfs["lumpy"], ignore_index=True)
            interval_parts.append(
                self.lumpy_model.predict_interval(
                    horizon=horizon, coverage=coverage, features_df=l_df, **kwargs
                )
            )

        if not interval_parts:
            return pd.DataFrame(
                columns=["date", "forecast", "lower", "upper", *self._series_id_cols]
            )

        return pd.concat(interval_parts, ignore_index=True)
