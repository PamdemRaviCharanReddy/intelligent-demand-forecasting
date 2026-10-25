"""Split Conformal Calibration and Conformalized Quantile Regression (CQR) (Novelty N2).

Provides finite-sample statistical coverage guarantees:
    P(y_{t+h} in C(x_{t+h})) >= 1 - alpha
without assuming Gaussianity, symmetry, or constant variance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from idfs.evaluation.metrics import empirical_coverage, interval_width, pinball_loss
from idfs.models.base import Forecaster
from idfs.uncertainty.quantile import QuantileXGBoostForecaster

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """Inductive Split Conformal Prediction and Conformalized Quantile Regression (CQR).

    Calibrates prediction intervals using a held-out calibration partition,
    guaranteeing valid empirical coverage >= 1 - alpha.
    """

    def __init__(
        self,
        base_forecaster: Forecaster | QuantileXGBoostForecaster,
        alpha: float = 0.10,  # 1 - alpha is target nominal coverage (e.g. 0.90)
    ) -> None:
        self.base_forecaster = base_forecaster
        self.alpha = alpha
        self.coverage_target = 1.0 - alpha
        self.q_hat_: float = 0.0
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        cal_df: pd.DataFrame,
        target_col: str = "sales",
    ) -> ConformalCalibrator:
        """Compute non-conformity scores and empirical adjustment quantile on calibration split.

        Args:
            cal_df: Held-out calibration DataFrame with features and true target values.
            target_col: Name of actual sales column.
        """
        if not self.base_forecaster.is_fitted:
            raise RuntimeError("Base forecaster must be fitted before conformal calibration")

        horizon = cal_df["date"].nunique()
        y_cal = cal_df[target_col].to_numpy(dtype=np.float64)
        n = len(y_cal)
        if n < 5:
            raise ValueError(f"Calibration set too small: {n} samples")

        # Conformalized Quantile Regression (CQR) if base model supports quantiles
        if isinstance(self.base_forecaster, QuantileXGBoostForecaster):
            pred_quantiles = self.base_forecaster.predict_quantiles(features_df=cal_df)
            low_col = f"q_{self.alpha / 2:.2f}"
            high_col = f"q_{1.0 - (self.alpha / 2):.2f}"

            if low_col in pred_quantiles.columns and high_col in pred_quantiles.columns:
                q_low = pred_quantiles[low_col].to_numpy()
                q_high = pred_quantiles[high_col].to_numpy()
            else:
                # Fallback to closest
                intervals = self.base_forecaster.predict_interval(
                    horizon=horizon, coverage=self.coverage_target, features_df=cal_df
                )
                q_low = intervals["lower"].to_numpy()
                q_high = intervals["upper"].to_numpy()

            # CQR non-conformity score: distance to boundary (negative if inside interval)
            scores = np.maximum(q_low - y_cal, y_cal - q_high)
        else:
            # Standard residual conformal prediction
            preds = self.base_forecaster.predict(horizon=horizon, features_df=cal_df)
            y_hat = preds["forecast"].to_numpy(dtype=np.float64)
            scores = np.abs(y_cal - y_hat)

        # Finite-sample conformal correction index
        # k = ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * self.coverage_target) / n)
        self.q_hat_ = float(np.quantile(scores, level, method="higher"))

        logger.info(
            "Conformal calibration completed: N=%d, alpha=%.2f, q_hat=%.4f",
            n,
            self.alpha,
            self.q_hat_,
        )
        self.is_calibrated_ = True
        return self

    def predict_interval(
        self,
        horizon: int,
        features_df: pd.DataFrame,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate conformalized prediction intervals on test set."""
        if not self.is_calibrated_:
            raise RuntimeError("ConformalCalibrator must be calibrated before predict_interval")

        # Get base bounds
        base_intervals = self.base_forecaster.predict_interval(
            horizon=horizon, coverage=self.coverage_target, features_df=features_df, **kwargs
        )

        out_df = base_intervals.copy()
        if isinstance(self.base_forecaster, QuantileXGBoostForecaster):
            # CQR adjustment: expand/contract bounds by q_hat
            out_df["lower"] = np.maximum(0.0, base_intervals["lower"] - self.q_hat_)
            out_df["upper"] = np.maximum(out_df["lower"], base_intervals["upper"] + self.q_hat_)
        else:
            # Additive residual adjustment
            out_df["lower"] = np.maximum(0.0, base_intervals["forecast"] - self.q_hat_)
            out_df["upper"] = base_intervals["forecast"] + self.q_hat_

        return out_df

    def evaluate_calibration(
        self,
        test_df: pd.DataFrame,
        target_col: str = "sales",
    ) -> dict[str, float]:
        """Compute coverage, sharpness (width), and pinball loss on test set."""
        horizon = test_df["date"].nunique()
        intervals = self.predict_interval(horizon=horizon, features_df=test_df)
        y_true = test_df[target_col].to_numpy(dtype=np.float64)

        lower = intervals["lower"].to_numpy(dtype=np.float64)
        upper = intervals["upper"].to_numpy(dtype=np.float64)
        forecast = intervals["forecast"].to_numpy(dtype=np.float64)

        cov = empirical_coverage(y_true, lower, upper)
        width = interval_width(lower, upper)
        pb_lower = pinball_loss(y_true, lower, tau=self.alpha / 2)
        pb_upper = pinball_loss(y_true, upper, tau=1.0 - (self.alpha / 2))
        pb_med = pinball_loss(y_true, forecast, tau=0.50)

        return {
            "target_coverage": self.coverage_target,
            "empirical_coverage": cov,
            "mean_interval_width": width,
            "pinball_loss_lower": pb_lower,
            "pinball_loss_upper": pb_upper,
            "pinball_loss_mean": float(np.mean([pb_lower, pb_upper, pb_med])),
        }
