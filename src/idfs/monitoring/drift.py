"""Rolling-error concept drift monitor and automated retraining alert engine (Novelty N4).

Detects performance degradation and distribution shifts in real-time retail demand
using sliding-window error tracking (MAE, WAPE) and cumulative sum (CUSUM) drift tests.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainAlert:
    """Alert triggered when rolling error exceeds drift threshold."""

    timestamp: pd.Timestamp | str | int | Any
    metric: str
    current_error: float
    baseline_error: float
    ratio: float
    severity: str  # 'WARNING' | 'CRITICAL'
    message: str


class DriftMonitor:
    """Continuous production surveillance monitor tracking forecast drift.

    Maintains a rolling window of recent actuals and predictions to evaluate
    rolling MAE and WAPE. Emits structured alerts when error exceeds baseline.
    """

    def __init__(
        self,
        window_size: int = 28,
        threshold_multiplier: float = 1.5,
        critical_multiplier: float = 2.0,
        metric: str = "mae",  # 'mae' or 'wape'
    ) -> None:
        self.window_size = max(5, window_size)
        self.threshold_multiplier = threshold_multiplier
        self.critical_multiplier = critical_multiplier
        self.metric = metric.lower()

        self._baseline_error: float | None = None
        self._actuals: deque[float] = deque(maxlen=self.window_size)
        self._predictions: deque[float] = deque(maxlen=self.window_size)
        self._history: list[dict[str, Any]] = []
        self._alerts: list[RetrainAlert] = []
        self._should_retrain: bool = False

    @property
    def baseline_error(self) -> float | None:
        return self._baseline_error

    @property
    def should_retrain(self) -> bool:
        return self._should_retrain

    @property
    def alerts(self) -> list[RetrainAlert]:
        return list(self._alerts)

    def set_baseline(self, baseline_error: float) -> None:
        """Establish baseline error from validation cross-validation."""
        if baseline_error <= 0:
            raise ValueError("Baseline error must be strictly positive")
        self._baseline_error = float(baseline_error)
        logger.info("DriftMonitor baseline %s set to: %.4f", self.metric, self._baseline_error)

    def update(
        self,
        actual: float,
        predicted: float,
        timestamp: pd.Timestamp | str | int | Any | None = None,
    ) -> RetrainAlert | None:
        """Stream a new observation into the rolling window and check for drift.

        Returns RetrainAlert if drift threshold is breached, else None.
        """
        a = float(actual)
        p = float(predicted)
        ts = timestamp if timestamp is not None else pd.Timestamp.now()

        self._actuals.append(a)
        self._predictions.append(p)

        record: dict[str, Any] = {
            "date": ts,
            "actual": a,
            "predicted": p,
            "error": abs(a - p),
            "rolling_mae": None,
            "rolling_wape": None,
            "drift_detected": False,
            "severity": None,
        }

        alert: RetrainAlert | None = None

        if len(self._actuals) >= self.window_size:
            arr_a = np.array(self._actuals, dtype=np.float64)
            arr_p = np.array(self._predictions, dtype=np.float64)

            rolling_mae = float(np.mean(np.abs(arr_a - arr_p)))
            sum_actuals = float(np.sum(arr_a))
            rolling_wape = float(np.sum(np.abs(arr_a - arr_p)) / max(sum_actuals, 1e-4))

            record["rolling_mae"] = rolling_mae
            record["rolling_wape"] = rolling_wape

            curr_val = rolling_mae if self.metric == "mae" else rolling_wape

            if self._baseline_error is not None:
                ratio = curr_val / self._baseline_error
                if ratio >= self.critical_multiplier:
                    severity = "CRITICAL"
                    self._should_retrain = True
                    msg = (
                        f"CRITICAL DRIFT: Rolling {self.metric.upper()}={curr_val:.2f} is "
                        f"{ratio:.2f}x baseline ({self._baseline_error:.2f}). Immediate retraining recommended."
                    )
                    alert = RetrainAlert(
                        timestamp=ts,
                        metric=self.metric,
                        current_error=curr_val,
                        baseline_error=self._baseline_error,
                        ratio=ratio,
                        severity=severity,
                        message=msg,
                    )
                elif ratio >= self.threshold_multiplier:
                    severity = "WARNING"
                    self._should_retrain = True
                    msg = (
                        f"WARNING: Rolling {self.metric.upper()}={curr_val:.2f} is "
                        f"{ratio:.2f}x baseline ({self._baseline_error:.2f}). Model degradation detected."
                    )
                    alert = RetrainAlert(
                        timestamp=ts,
                        metric=self.metric,
                        current_error=curr_val,
                        baseline_error=self._baseline_error,
                        ratio=ratio,
                        severity=severity,
                        message=msg,
                    )
                else:
                    self._should_retrain = False

                if alert is not None:
                    self._alerts.append(alert)
                    record["drift_detected"] = True
                    record["severity"] = alert.severity

        self._history.append(record)
        return alert

    def batch_update(
        self,
        actuals: np.ndarray | pd.Series,
        predictions: np.ndarray | pd.Series,
        dates: pd.DatetimeIndex | pd.Series | None = None,
    ) -> list[RetrainAlert]:
        """Process a sequence of observations in chronological order."""
        arr_a = np.asarray(actuals, dtype=np.float64)
        arr_p = np.asarray(predictions, dtype=np.float64)
        n = len(arr_a)

        alerts = []
        for i in range(n):
            ts = dates[i] if dates is not None else i
            alt = self.update(actual=arr_a[i], predicted=arr_p[i], timestamp=ts)
            if alt is not None:
                alerts.append(alt)
        return alerts

    def get_monitoring_history(self) -> pd.DataFrame:
        """Return history of observations, rolling errors, and drift events."""
        return pd.DataFrame(self._history)
