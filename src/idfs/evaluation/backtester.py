"""Walk-forward backtester for IDFS.

Implements rolling-origin (expanding window) cross-validation for time series.
Guarantees identical folds for all candidate models — no leakage by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from idfs.evaluation.metrics import (
    compute_all_point_metrics,
    empirical_coverage,
    interval_width,
)
from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single backtest fold."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    metrics: dict[str, float]
    actuals: pd.DataFrame
    forecasts: pd.DataFrame


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
        dates = sorted(pd.to_datetime(df[date_col].unique()))
        total = len(dates)

        # Reserve test set at the end
        available = total - self.test_days
        needed_per_fold = self.val_days + self.gap_days
        total_val_space = needed_per_fold * self.n_folds

        if available < total_val_space + 30:
            raise ValueError(
                f"Not enough data for {self.n_folds} folds. "
                f"Have {available} days available, need at least {total_val_space + 30}"
            )

        folds = []
        for i in range(self.n_folds):
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
            **kwargs: Extra arguments passed to model.fit().

        Returns:
            BacktestResult with per-fold and aggregated metrics.
        """
        series_id_cols = series_id_cols or ["store_id", "family"]
        folds = self.generate_folds(df, date_col=date_col)
        fold_results: list[FoldResult] = []

        logger.info(
            "Starting walk-forward backtest for model '%s' across %d folds...",
            model.name,
            len(folds),
        )

        for fold_idx, (tr_start, tr_end, v_start, v_end) in enumerate(folds):
            logger.info(
                "Fold %d: Train [%s to %s] -> Val [%s to %s]",
                fold_idx + 1,
                tr_start.strftime("%Y-%m-%d"),
                tr_end.strftime("%Y-%m-%d"),
                v_start.strftime("%Y-%m-%d"),
                v_end.strftime("%Y-%m-%d"),
            )

            # Slice training and validation subsets
            train_fold = df[(df[date_col] >= tr_start) & (df[date_col] <= tr_end)].copy()
            val_fold = df[(df[date_col] >= v_start) & (df[date_col] <= v_end)].copy()

            # Fit model
            model.fit(
                train_df=train_fold,
                target_col=target_col,
                series_id_cols=series_id_cols,
                val_df=val_fold,
                **kwargs,
            )

            # Point forecast & interval forecast
            horizon = self.val_days
            pred_df = model.predict(horizon=horizon, features_df=val_fold)

            # Merge predictions with validation actuals
            merge_cols = [
                date_col,
                *[c for c in series_id_cols if c in val_fold.columns and c in pred_df.columns],
            ]
            if len(merge_cols) > 0:
                merged = pd.merge(
                    val_fold,
                    pred_df,
                    on=merge_cols,
                    how="inner",
                    suffixes=("_actual", "_pred"),
                )
            else:
                merged = val_fold.copy()
                merged["forecast"] = pred_df["forecast"].values

            actuals = merged[target_col].to_numpy(dtype=np.float64)
            preds = merged["forecast"].to_numpy(dtype=np.float64)
            train_actuals = train_fold[target_col].to_numpy(dtype=np.float64)

            # Compute standard metrics
            metrics = compute_all_point_metrics(
                actual=actuals,
                predicted=preds,
                training_actual=train_actuals,
                seasonality=7,
            )

            # If model supports intervals, evaluate coverage and width
            try:
                interval_df = model.predict_interval(
                    horizon=horizon, coverage=0.90, features_df=val_fold
                )
                if "lower" in interval_df.columns and "upper" in interval_df.columns:
                    merged_intervals = pd.merge(merged, interval_df, on=merge_cols, how="inner")
                    if len(merged_intervals) > 0:
                        cov = empirical_coverage(
                            merged_intervals[target_col],
                            merged_intervals["lower"],
                            merged_intervals["upper"],
                        )
                        w = interval_width(merged_intervals["lower"], merged_intervals["upper"])
                        metrics["coverage_90"] = round(cov, 4)
                        metrics["width_90"] = round(w, 2)
            except Exception:
                pass

            fold_results.append(
                FoldResult(
                    fold_id=fold_idx + 1,
                    train_start=tr_start,
                    train_end=tr_end,
                    val_start=v_start,
                    val_end=v_end,
                    metrics=metrics,
                    actuals=val_fold,
                    forecasts=pred_df,
                )
            )

        # Compute mean and standard deviation across folds
        metric_keys = fold_results[0].metrics.keys()
        mean_metrics = {}
        std_metrics = {}
        for k in metric_keys:
            vals = [f.metrics[k] for f in fold_results if not np.isnan(f.metrics[k])]
            mean_metrics[k] = float(np.mean(vals)) if vals else float("nan")
            std_metrics[k] = float(np.std(vals)) if vals else float("nan")

        return BacktestResult(
            model_name=model.name,
            folds=fold_results,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
        )

    def compare_models(
        self,
        models: list[Forecaster],
        df: pd.DataFrame,
        target_col: str = "sales",
        date_col: str = "date",
        series_id_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run all candidate models across identical folds and tabulate comparisons.

        Args:
            models: List of Forecaster instances.
            df: Full dataset.
            target_col: Sales column name.
            date_col: Date column name.
            series_id_cols: Series identifier columns.

        Returns:
            Comparison summary DataFrame sorted by WAPE.
        """
        rows = []
        for model in models:
            res = self.run(
                model=model,
                df=df,
                target_col=target_col,
                date_col=date_col,
                series_id_cols=series_id_cols,
                **kwargs,
            )
            row: dict[str, Any] = {"model": res.model_name}
            for k, val in res.mean_metrics.items():
                row[f"{k}_mean"] = round(val, 3)
                row[f"{k}_std"] = round(res.std_metrics.get(k, 0.0), 3)
            rows.append(row)

        comp_df = pd.DataFrame(rows)
        if "wape_mean" in comp_df.columns:
            comp_df = comp_df.sort_values("wape_mean").reset_index(drop=True)
        return comp_df
