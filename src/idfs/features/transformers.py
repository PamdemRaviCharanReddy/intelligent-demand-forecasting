"""Lag and rolling statistical feature transformers for IDFS.

Enforces zero-leakage by construction: all rolling windows strictly invoke
.shift(1) before calculating rolling statistics.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def build_lag_features(
    df: pd.DataFrame,
    lags: list[int] | None = None,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate lag features for time series grouped by series identifiers.

    For any lag k >= 1, the feature value at row t equals y_{t-k}.

    Args:
        df: Input DataFrame sorted by group_cols and date_col.
        lags: List of integer lags (e.g. [1, 2, 7, 14, 28]). Must all be >= 1.
        target_col: Column to lag.
        group_cols: Series identifier columns (e.g. ['store_id', 'family']).
        date_col: Timestamp column.

    Returns:
        DataFrame with added lag feature columns.

    Raises:
        ValueError: If any lag is < 1 (leakage risk).
    """
    lags = lags or [1, 2, 3, 7, 14, 21, 28]
    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent data leakage, got {lag}")

    out_df = df.copy()
    group_cols = group_cols or ["store_id", "family"]
    active_groups = [c for c in group_cols if c in out_df.columns]

    if active_groups:
        grouped = out_df.groupby(active_groups, observed=True)[target_col]
        for lag in lags:
            out_df[f"{target_col}_lag_{lag}"] = grouped.shift(lag)
    else:
        for lag in lags:
            out_df[f"{target_col}_lag_{lag}"] = out_df[target_col].shift(lag)

    return out_df


def build_rolling_features(
    df: pd.DataFrame,
    windows: list[int] | None = None,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate shifted rolling window statistical aggregates.

    LEAKAGE PROTECTION INVARIANT:
    Computes rolling mean and standard deviation on df[target_col].shift(1).
    This strictly ensures observation at date t incorporates observations up to t-1 only.

    Args:
        df: Input DataFrame sorted by group_cols and date_col.
        windows: List of window sizes in days (e.g. [7, 14, 28]).
        target_col: Column to compute rolling statistics over.
        group_cols: Series identifier columns.
        date_col: Timestamp column.

    Returns:
        DataFrame with added rolling mean and std columns.
    """
    windows = windows or [7, 14, 28]
    out_df = df.copy()
    group_cols = group_cols or ["store_id", "family"]
    active_groups = [c for c in group_cols if c in out_df.columns]

    if active_groups:
        # Shift 1 first to guarantee no future row enters the rolling window
        grouped_shifted = out_df.groupby(active_groups, observed=True)[target_col].shift(1)
        # Group again for rolling calculation per series
        grouped = grouped_shifted.groupby([out_df[c] for c in active_groups], observed=True)
        for w in windows:
            out_df[f"{target_col}_roll_mean_{w}"] = grouped.transform(
                lambda s, win=w: s.rolling(win, min_periods=max(1, win // 2)).mean()
            )
            out_df[f"{target_col}_roll_std_{w}"] = grouped.transform(
                lambda s, win=w: s.rolling(win, min_periods=max(1, win // 2)).std()
            ).fillna(0.0)
    else:
        shifted = out_df[target_col].shift(1)
        for w in windows:
            out_df[f"{target_col}_roll_mean_{w}"] = shifted.rolling(
                w, min_periods=max(1, w // 2)
            ).mean()
            out_df[f"{target_col}_roll_std_{w}"] = (
                shifted.rolling(w, min_periods=max(1, w // 2)).std().fillna(0.0)
            )

    return out_df


def build_diff_features(
    df: pd.DataFrame,
    periods: list[int] | None = None,
    target_col: str = "sales",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute seasonal differencing features based on lag differences.

    e.g. diff_7 = sales_lag_1 - sales_lag_8 (change in demand week-over-week).

    Args:
        df: Input DataFrame.
        periods: Differencing steps (e.g. [1, 7]).
        target_col: Column name.
        group_cols: Series identifier columns.

    Returns:
        DataFrame with differenced signals.
    """
    periods = periods or [1, 7]
    out_df = df.copy()
    group_cols = group_cols or ["store_id", "family"]
    active_groups = [c for c in group_cols if c in out_df.columns]

    if active_groups:
        # Shift 1 first, then diff
        shifted = out_df.groupby(active_groups, observed=True)[target_col].shift(1)
        grouped = shifted.groupby([out_df[c] for c in active_groups], observed=True)
        for p in periods:
            out_df[f"{target_col}_diff_{p}"] = grouped.diff(p).fillna(0.0)
    else:
        shifted = out_df[target_col].shift(1)
        for p in periods:
            out_df[f"{target_col}_diff_{p}"] = shifted.diff(p).fillna(0.0)

    return out_df
