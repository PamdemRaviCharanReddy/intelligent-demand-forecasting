"""Calendar, cyclical, and holiday proximity feature engineering for IDFS."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_calendar_features(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Generate calendar-derived features and cyclical sine/cosine encodings.

    Features generated:
    - day_of_week (0-6)
    - day_of_month (1-31)
    - month (1-12)
    - day_of_year (1-366)
    - is_weekend (0 or 1)
    - sin_dow, cos_dow
    - sin_month, cos_month
    - sin_doy, cos_doy

    Args:
        df: Input DataFrame with date column.
        date_col: Name of the timestamp column.

    Returns:
        DataFrame enriched with calendar features.
    """
    out_df = df.copy()
    dates = pd.to_datetime(out_df[date_col])

    # Basic discrete calendar attributes
    dow = dates.dt.dayofweek
    dom = dates.dt.day
    month = dates.dt.month
    doy = dates.dt.dayofyear

    out_df["day_of_week"] = dow.astype(int)
    out_df["day_of_month"] = dom.astype(int)
    out_df["month"] = month.astype(int)
    out_df["day_of_year"] = doy.astype(int)
    out_df["is_weekend"] = (dow >= 5).astype(int)

    # Cyclical sine/cosine transformations
    out_df["sin_dow"] = np.sin(2 * np.pi * dow / 7.0)
    out_df["cos_dow"] = np.cos(2 * np.pi * dow / 7.0)

    out_df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out_df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    out_df["sin_doy"] = np.sin(2 * np.pi * (doy - 1) / 365.25)
    out_df["cos_doy"] = np.cos(2 * np.pi * (doy - 1) / 365.25)

    return out_df


def build_holiday_proximity_features(
    df: pd.DataFrame,
    date_col: str = "date",
    is_holiday_col: str = "is_holiday",
    max_distance: int = 14,
) -> pd.DataFrame:
    """Compute days to next holiday and days since last holiday.

    Clipped to max_distance to prevent uninformative distant lookups.

    Args:
        df: Input DataFrame.
        date_col: Date column name.
        is_holiday_col: Binary holiday indicator column.
        max_distance: Maximum lookahead/lookback cap in days.

    Returns:
        DataFrame with days_to_next_holiday and days_since_last_holiday.
    """
    out_df = df.copy()
    dates = pd.to_datetime(out_df[date_col])

    if is_holiday_col not in out_df.columns:
        out_df["days_to_next_holiday"] = max_distance
        out_df["days_since_last_holiday"] = max_distance
        return out_df

    # Extract all unique holiday dates
    holiday_dates = pd.to_datetime(out_df.loc[out_df[is_holiday_col] == 1, date_col].unique())

    if len(holiday_dates) == 0:
        out_df["days_to_next_holiday"] = max_distance
        out_df["days_since_last_holiday"] = max_distance
        return out_df

    holiday_dates = pd.DatetimeIndex(holiday_dates).sort_values()
    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()

    # Map each unique date to next and previous holiday delta
    days_to_next_map: dict[pd.Timestamp, int] = {}
    days_since_last_map: dict[pd.Timestamp, int] = {}

    for d in unique_dates:
        # Next holiday (future or today)
        future_hols = holiday_dates[holiday_dates >= d]
        if len(future_hols) > 0:
            delta_next = int((future_hols[0] - d) / np.timedelta64(1, "D"))
            days_to_next_map[d] = min(delta_next, max_distance)
        else:
            days_to_next_map[d] = max_distance

        # Past holiday (past or today)
        past_hols = holiday_dates[holiday_dates <= d]
        if len(past_hols) > 0:
            delta_past = int((d - past_hols[-1]) / np.timedelta64(1, "D"))
            days_since_last_map[d] = min(delta_past, max_distance)
        else:
            days_since_last_map[d] = max_distance

    out_df["days_to_next_holiday"] = dates.map(days_to_next_map).fillna(max_distance).astype(int)
    out_df["days_since_last_holiday"] = (
        dates.map(days_since_last_map).fillna(max_distance).astype(int)
    )

    return out_df
