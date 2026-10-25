"""Walk-forward backtester.

Implements rolling-origin (expanding window) cross-validation for time series.
Ensures identical folds for every model — no leakage by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from idfs.models.base import Forecaster


@dataclass
class FoldResult:
    """Results from a single backtest fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    actuals: pd.DataFrame
    forecasts: pd.DataFrame
    metrics: dict[str, float]


@dataclass
class BacktestResult:
    """Aggregated results from all backtest folds."""

    model_name: str
    folds: list[FoldResult]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]


class Backtester:
    """Walk-forward (rolling-origin) backtester.

    Creates time-based splits with expanding training windows and
    fixed-size validation windows. The test set is never touched.

    The same fold boundaries are used for every model, ensuring
    fair comparison.

    Attributes:
        n_folds: Number of validation folds.
        val_days: Size of each validation window in days.
        gap_days: Gap between training end and validation start.
        test_days: Size of the held-out test set (excluded from folds).
    """

    def __init__(
        self,
        n_folds: int = 3,
        val_days: int = 14,
        gap_days: int = 0,
        test_days: int = 28,
    ) -> None:
        self.n_folds = n_folds
        self.val_days = val_days
        self.gap_days = gap_days
        self.test_days = test_days

    def generate_folds(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate fold boundaries (train_start, train_end, val_start, val_end).

        Args:
            df: Full dataset with a date column.
            date_col: Name of the date column.

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples.

        Raises:
            ValueError: If the dataset is too short for the requested folds.
        """
        dates = sorted(df[date_col].unique())
        total = len(dates)

        # Reserve test set at the end
        available = total - self.test_days
        needed_per_fold = self.val_days + self.gap_days
        total_val_space = needed_per_fold * self.n_folds

        if available < total_val_space + 30:  # Need at least 30 days training
            raise ValueError(
                f"Not enough data for {self.n_folds} folds. "
                f"Have {available} days available, need {total_val_space + 30}"
            )

        folds = []
        for i in range(self.n_folds):
            # Walk forward: each fold extends the training window
            val_end_idx = available - (self.n_folds - 1 - i) * needed_per_fold
            val_start_idx = val_end_idx - self.val_days
            train_end_idx = val_start_idx - self.gap_days

            folds.append(
                (
                    pd.Timestamp(dates[0]),
                    pd.Timestamp(dates[train_end_idx - 1]),
                    pd.Timestamp(dates[val_start_idx]),
                    pd.Timestamp(dates[val_end_idx - 1]),
                )
            )

        return folds

    def run(
        self,
        model: Forecaster,
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> BacktestResult:
        """Run walk-forward backtest for a single model.

        Args:
            model: Forecaster instance to evaluate.
            df: Full dataset.
            target_col: Target column name.
            date_col: Date column name.
            series_id_cols: Series identifier columns.
            **kwargs: Passed to model.fit() and model.predict().

        Returns:
            BacktestResult with per-fold and aggregated metrics.
        """
        # Stub — full implementation in Phase 2
        raise NotImplementedError("Full backtester implementation in Phase 2")
