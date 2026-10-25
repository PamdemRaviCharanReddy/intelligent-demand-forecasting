"""Model Stacking and Weighted Ensembling.

Combines predictions from multiple distinct forecaster families (Statistical,
Tree-based, Deep Learning, Baselines) using inverse-validation-error or equal weighting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from idfs.evaluation.metrics import mae
from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class WeightedEnsembleForecaster(Forecaster):
    """Blends multiple candidate forecasters via convex combination of predictions.

    Weights can be set manually, computed uniformly, or fitted dynamically
    based on inverse validation MAE:
        w_k proportional to (1 / (MAE_k + eps)^gamma)
    """

    def __init__(
        self,
        models: list[Forecaster],
        weights: list[float] | None = None,
        weighting_strategy: str = "inverse_mae",  # 'uniform', 'inverse_mae', 'manual'
        gamma: float = 1.0,
        name: str = "WeightedEnsemble",
    ) -> None:
        super().__init__(name=name)
        if not models:
            raise ValueError("WeightedEnsembleForecaster requires at least one model")
        self.models = models
        self.weighting_strategy = weighting_strategy
        self.gamma = gamma
        self.weights_: list[float] = weights or [1.0 / len(models)] * len(models)
        self._series_id_cols: list[str] = []

    @property
    def supports_global(self) -> bool:
        return any(m.supports_global for m in self.models)

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> WeightedEnsembleForecaster:
        """Fit all constituent models and compute validation weights."""
        self._series_id_cols = series_id_cols or []

        # Fit each constituent model
        for m in self.models:
            logger.info("Fitting ensemble constituent model: %s", m.name)
            m.fit(
                train_df,
                target_col=target_col,
                series_id_cols=series_id_cols,
                val_df=val_df,
                **kwargs,
            )

        # Dynamic validation weighting
        if self.weighting_strategy == "inverse_mae" and val_df is not None and len(val_df) > 0:
            maes = []
            horizon = val_df["date"].nunique()
            y_true = val_df[target_col].to_numpy(dtype=np.float64)

            for m in self.models:
                try:
                    preds_df = m.predict(horizon=horizon, features_df=val_df)
                    # Align prediction with val_df order
                    if self._series_id_cols:
                        merged = pd.merge(
                            val_df[["date", *self._series_id_cols]],
                            preds_df,
                            on=["date", *self._series_id_cols],
                            how="left",
                        )
                        y_pred = merged["forecast"].fillna(0.0).to_numpy(dtype=np.float64)
                    else:
                        y_pred = preds_df["forecast"].to_numpy(dtype=np.float64)[: len(y_true)]

                    m_err = mae(y_true, y_pred)
                    maes.append(m_err)
                except Exception as e:
                    logger.warning("Failed validation evaluation for model %s: %s", m.name, e)
                    maes.append(1e6)

            # Invert MAEs: w_i = (1 / (mae_i + 1e-4))^gamma
            inv_maes = np.array([1.0 / ((m + 1e-4) ** self.gamma) for m in maes])
            norm_weights = inv_maes / np.sum(inv_maes)
            self.weights_ = norm_weights.tolist()
            logger.info(
                "Ensemble weights computed: %s",
                dict(
                    zip(
                        [m.name for m in self.models],
                        [round(w, 4) for w in self.weights_],
                        strict=False,
                    )
                ),
            )
        elif self.weighting_strategy == "uniform":
            self.weights_ = [1.0 / len(self.models)] * len(self.models)

        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate ensemble prediction via weighted linear combination."""
        if not self._is_fitted:
            raise RuntimeError("WeightedEnsembleForecaster must be fitted before predict")

        all_preds: list[pd.DataFrame] = []
        for m in self.models:
            p = m.predict(horizon=horizon, features_df=features_df, **kwargs)
            all_preds.append(p)

        # Merge base predictions
        base_df = all_preds[0].copy()
        join_cols = ["date"] + [c for c in self._series_id_cols if c in base_df.columns]

        combined_forecast = np.zeros(len(base_df), dtype=np.float64)
        for w, p in zip(self.weights_, all_preds, strict=False):
            if join_cols:
                aligned = pd.merge(base_df[join_cols], p, on=join_cols, how="left")
                f_vals = aligned["forecast"].fillna(0.0).to_numpy()
            else:
                f_vals = p["forecast"].to_numpy()
            combined_forecast += w * f_vals

        base_df["forecast"] = np.maximum(0.0, combined_forecast)
        return base_df

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Compute ensemble prediction intervals via convex combination of bounds."""
        if not self._is_fitted:
            raise RuntimeError("WeightedEnsembleForecaster must be fitted before predict")

        all_intervals: list[pd.DataFrame] = []
        for m in self.models:
            p = m.predict_interval(
                horizon=horizon, coverage=coverage, features_df=features_df, **kwargs
            )
            all_intervals.append(p)

        base_df = all_intervals[0].copy()
        join_cols = ["date"] + [c for c in self._series_id_cols if c in base_df.columns]

        combined_forecast = np.zeros(len(base_df), dtype=np.float64)
        combined_lower = np.zeros(len(base_df), dtype=np.float64)
        combined_upper = np.zeros(len(base_df), dtype=np.float64)

        for w, p in zip(self.weights_, all_intervals, strict=False):
            if join_cols:
                aligned = pd.merge(base_df[join_cols], p, on=join_cols, how="left")
                f_vals = aligned["forecast"].fillna(0.0).to_numpy()
                l_vals = aligned["lower"].fillna(0.0).to_numpy()
                u_vals = aligned["upper"].fillna(0.0).to_numpy()
            else:
                f_vals = p["forecast"].to_numpy()
                l_vals = p["lower"].to_numpy()
                u_vals = p["upper"].to_numpy()

            combined_forecast += w * f_vals
            combined_lower += w * l_vals
            combined_upper += w * u_vals

        base_df["forecast"] = np.maximum(0.0, combined_forecast)
        base_df["lower"] = np.maximum(0.0, combined_lower)
        base_df["upper"] = np.maximum(base_df["lower"], combined_upper)
        return base_df
