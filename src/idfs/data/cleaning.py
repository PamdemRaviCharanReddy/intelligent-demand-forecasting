"""Data cleaning, outlier mitigation, and missing date imputation.

Ensures time-series continuity, non-negativity, and consistency before feature engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CleaningReport:
    """Audit log of cleaning operations performed on raw data."""

    initial_rows: int
    final_rows: int
    duplicates_removed: int
    negative_values_clamped: int
    missing_dates_imputed: int
    outliers_capped: int


def clean_sales(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
    series_id_cols: list[str] | None = None,
    impute_missing_dates: bool = True,
    winsorize_quantile: float | None = 0.999,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean and standardize sales transactions.

    Performs:
    1. Datetime parsing and validation.
    2. Deduplication of identical (date, series_id) records.
    3. Non-negativity enforcement (clamping negative sales to 0.0).
    4. Continuous date grid imputation per active series.
    5. Upper-tail outlier winsorization.

    Args:
        df: Raw sales DataFrame.
        date_col: Date column name.
        target_col: Target demand column name.
        series_id_cols: Identifier columns for individual series.
        impute_missing_dates: Whether to fill unobserved calendar days with 0.
        winsorize_quantile: Upper quantile threshold to cap extreme spikes.

    Returns:
        Tuple of (cleaned_df, cleaning_report).
    """
    initial_rows = len(df)
    clean_df = df.copy()
    series_id_cols = series_id_cols or ["store_id", "family"]
    # Filter to existing columns in df
    active_id_cols = [c for c in series_id_cols if c in clean_df.columns]

    # 1. Parse date
    clean_df[date_col] = pd.to_datetime(clean_df[date_col])

    # 2. Deduplicate
    group_cols = [date_col, *active_id_cols]
    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=group_cols, keep="first")
    duplicates_removed = before_dedup - len(clean_df)

    # 3. Non-negativity
    negative_values_clamped = int((clean_df[target_col] < 0).sum())
    clean_df[target_col] = np.maximum(0.0, clean_df[target_col].astype(np.float64))

    # 4. Impute missing dates per series
    missing_dates_imputed = 0
    if impute_missing_dates and active_id_cols:
        min_date = clean_df[date_col].min()
        max_date = clean_df[date_col].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq="D")

        # Create multi-index of all dates x unique series combinations
        unique_series = clean_df[active_id_cols].drop_duplicates()
        idx_tuples = []
        for d in all_dates:
            for row in unique_series.itertuples(index=False):
                idx_tuples.append((d, *row))

        full_index_df = pd.DataFrame(idx_tuples, columns=[date_col, *active_id_cols])
        merged = pd.merge(
            full_index_df,
            clean_df,
            on=[date_col, *active_id_cols],
            how="left",
        )
        missing_dates_imputed = int(merged[target_col].isna().sum())
        merged[target_col] = merged[target_col].fillna(0.0)

        # Fill promotion if present
        if "onpromotion" in merged.columns:
            merged["onpromotion"] = merged["onpromotion"].fillna(0).astype(int)

        clean_df = merged

    # 5. Outlier Winsorization
    outliers_capped = 0
    if winsorize_quantile is not None and len(clean_df) > 0:
        threshold = float(clean_df[target_col].quantile(winsorize_quantile))
        is_outlier = clean_df[target_col] > threshold
        outliers_capped = int(is_outlier.sum())
        clean_df.loc[is_outlier, target_col] = threshold

    clean_df = clean_df.sort_values([*active_id_cols, date_col]).reset_index(drop=True)

    report = CleaningReport(
        initial_rows=initial_rows,
        final_rows=len(clean_df),
        duplicates_removed=duplicates_removed,
        negative_values_clamped=negative_values_clamped,
        missing_dates_imputed=missing_dates_imputed,
        outliers_capped=outliers_capped,
    )

    return clean_df, report


def clean_oil(
    oil_df: pd.DataFrame,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Clean commodity oil prices by making dates continuous and forward/back-filling.

    Args:
        oil_df: Raw oil price DataFrame.
        start_date: Desired starting date for continuous grid.
        end_date: Desired ending date for continuous grid.

    Returns:
        Cleaned oil DataFrame with zero gaps.
    """
    df = oil_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"], keep="first")

    s_date = start_date or df["date"].min()
    e_date = end_date or df["date"].max()
    full_dates = pd.date_range(start=s_date, end=e_date, freq="D")

    grid = pd.DataFrame({"date": full_dates})
    merged = pd.merge(grid, df, on="date", how="left")
    # Forward fill weekend gaps, then backfill initial gap
    merged["oil_price"] = merged["oil_price"].ffill().bfill()
    # If still NaN (e.g. empty table), fill with 0.0
    merged["oil_price"] = merged["oil_price"].fillna(0.0)

    return merged.sort_values("date").reset_index(drop=True)


def clean_holidays(holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Clean holiday events by deduplicating and consolidating multi-event days.

    Args:
        holidays_df: Raw holidays DataFrame.

    Returns:
        Deduplicated holiday DataFrame with binary is_holiday indicator.
    """
    df = holidays_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # If 'transferred' is present, ignore transferred holidays
    if "transferred" in df.columns:
        df = df[~df["transferred"]].copy()

    # Consolidate duplicate dates: keep first event or national over regional
    if "type" in df.columns:
        # Sort so National comes before Regional and Local
        type_order = {"National": 0, "Regional": 1, "Local": 2, "Event": 3}
        df["type_rank"] = df["type"].map(lambda t: type_order.get(str(t), 4))
        df = df.sort_values(["date", "type_rank"]).drop(columns=["type_rank"])

    dedup = df.drop_duplicates(subset=["date"], keep="first").copy()
    dedup["is_holiday"] = 1
    return dedup.sort_values("date").reset_index(drop=True)
