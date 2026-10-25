"""Synthetic data generator for IDFS.

Generates realistic retail sales data with:
- Multiple stores and product families
- Trend, weekly seasonality, yearly seasonality
- Promotional uplift
- Holiday effects
- Intermittent demand items
- Injected concept drift
- Stock-out censoring simulation

Output conforms to the project's data schema (see docs/DATA_CONTRACT.md).
All randomness is seeded for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    n_stores: int = 5
    n_families: int = 10
    n_days: int = 730  # 2 years
    start_date: str = "2021-01-01"
    seed: int = 42
    intermittent_frac: float = 0.15  # Fraction of series that are intermittent
    drift_start_day: int = 600  # Day when concept drift begins
    drift_magnitude: float = 0.3  # Drift as fraction of base demand
    stockout_frac: float = 0.05  # Fraction of observations censored by stock-outs
    promo_freq: float = 0.10  # Fraction of days with promotions


def generate_synthetic_data(config: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate a complete synthetic dataset.

    Produces data that mimics the Favorita competition structure:
    - sales: daily sales by store × family
    - stores: store metadata
    - holidays: holiday calendar
    - oil: external regressor (oil price proxy)
    - promotions: promotion events

    Args:
        config: Generation configuration. Uses defaults if None.

    Returns:
        Dict with keys 'sales', 'stores', 'holidays', 'oil', 'promotions'.
        All DataFrames have the correct schema for downstream pipeline use.
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    dates = pd.date_range(start=cfg.start_date, periods=cfg.n_days, freq="D")
    store_ids = [f"S{i:03d}" for i in range(cfg.n_stores)]
    families = [f"FAM_{i:02d}" for i in range(cfg.n_families)]

    # ── Store metadata ──
    store_types = rng.choice(["A", "B", "C", "D"], size=cfg.n_stores)
    store_clusters = rng.integers(1, 18, size=cfg.n_stores)
    stores_df = pd.DataFrame(
        {
            "store_id": store_ids,
            "city": [f"City_{i}" for i in range(cfg.n_stores)],
            "state": [f"State_{i % 3}" for i in range(cfg.n_stores)],
            "type": store_types,
            "cluster": store_clusters,
        }
    )

    # ── Holiday calendar ──
    holidays_df = _generate_holidays(dates, rng)

    # ── Oil price (external regressor) ──
    oil_df = _generate_oil(dates, rng)

    # ── Sales data ──
    n_series = cfg.n_stores * cfg.n_families
    intermittent_count = int(n_series * cfg.intermittent_frac)

    # Determine which series are intermittent
    series_indices = list(range(n_series))
    rng.shuffle(series_indices)
    intermittent_set = set(series_indices[:intermittent_count])

    all_records: list[dict[str, Any]] = []
    promo_records: list[dict[str, Any]] = []

    series_idx = 0
    for store_id in store_ids:
        for family in families:
            is_intermittent = series_idx in intermittent_set

            sales, promos = _generate_series(
                dates=dates,
                store_id=store_id,
                family=family,
                is_intermittent=is_intermittent,
                drift_start_day=cfg.drift_start_day,
                drift_magnitude=cfg.drift_magnitude,
                stockout_frac=cfg.stockout_frac,
                promo_freq=cfg.promo_freq,
                rng=rng,
            )

            for i, date in enumerate(dates):
                all_records.append(
                    {
                        "date": date,
                        "store_id": store_id,
                        "family": family,
                        "sales": max(0, sales[i]),
                        "onpromotion": promos[i],
                    }
                )
                if promos[i]:
                    promo_records.append(
                        {
                            "date": date,
                            "store_id": store_id,
                            "family": family,
                            "promo_type": rng.choice(["discount", "bogo", "bundle", "seasonal"]),
                            "promo_strength": round(float(rng.uniform(0.1, 0.5)), 2),
                        }
                    )

            series_idx += 1

    sales_df = pd.DataFrame(all_records)
    sales_df["sales"] = sales_df["sales"].round().astype(int)

    promos_df = (
        pd.DataFrame(promo_records)
        if promo_records
        else pd.DataFrame(columns=["date", "store_id", "family", "promo_type", "promo_strength"])
    )

    return {
        "sales": sales_df,
        "stores": stores_df,
        "holidays": holidays_df,
        "oil": oil_df,
        "promotions": promos_df,
    }


def _generate_series(
    dates: pd.DatetimeIndex,
    store_id: str,
    family: str,
    is_intermittent: bool,
    drift_start_day: int,
    drift_magnitude: float,
    stockout_frac: float,
    promo_freq: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a single sales series with all effects.

    Returns:
        Tuple of (sales_array, promotion_flag_array).
    """
    n = len(dates)
    day_indices = np.arange(n)

    # Base demand level varies by series
    base = rng.uniform(0.5, 3.0) if is_intermittent else rng.uniform(20, 200)

    # ── Trend ──
    trend_slope = rng.uniform(-0.01, 0.05) * base / 365
    trend = trend_slope * day_indices

    # ── Weekly seasonality ──
    dow = np.array([d.weekday() for d in dates])
    weekly_pattern = np.zeros(7)
    weekly_pattern[5] = rng.uniform(0.1, 0.4)  # Saturday boost
    weekly_pattern[6] = rng.uniform(-0.1, 0.3)  # Sunday varies
    weekly_pattern[0] = rng.uniform(-0.2, 0.0)  # Monday dip
    weekly = base * weekly_pattern[dow]

    # ── Yearly seasonality ──
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    yearly_amp = rng.uniform(0.05, 0.3) * base
    yearly_phase = rng.uniform(0, 2 * np.pi)
    yearly = yearly_amp * np.sin(2 * np.pi * day_of_year / 365.25 + yearly_phase)

    # ── Promotions ──
    promos = rng.random(n) < promo_freq
    promo_uplift = np.where(promos, base * rng.uniform(0.2, 0.6), 0.0)

    # ── Concept drift (gradual) ──
    drift = np.zeros(n)
    if drift_start_day < n:
        drift_days = day_indices[drift_start_day:]
        drift_progress = (drift_days - drift_start_day) / max(n - drift_start_day, 1)
        drift[drift_start_day:] = drift_magnitude * base * drift_progress

    # ── Noise ──
    noise_std = base * rng.uniform(0.05, 0.2)
    noise = rng.normal(0, noise_std, n)

    # ── Combine ──
    sales = base + trend + weekly + yearly + promo_uplift + drift + noise

    # ── Intermittent demand ──
    if is_intermittent:
        # ADI > 1.32: many zero periods
        zero_mask = rng.random(n) > rng.uniform(0.15, 0.40)
        sales = np.where(zero_mask, 0.0, np.abs(sales))

    # ── Stock-out censoring ──
    stockout_mask = rng.random(n) < stockout_frac
    sales = np.where(stockout_mask, sales * rng.uniform(0.0, 0.3, n), sales)

    # Floor at 0, round
    sales = np.maximum(sales, 0.0)

    return sales, promos.astype(int)


def _generate_holidays(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic holiday calendar."""
    records = []

    # Fixed holidays (approximate)
    fixed_holidays = [
        (1, 1, "New Year", "National"),
        (5, 1, "Labour Day", "National"),
        (8, 10, "Independence Day", "National"),
        (12, 25, "Christmas", "National"),
        (11, 2, "Day of the Dead", "National"),
        (2, 14, "Valentines Day", "Local"),
        (10, 31, "Halloween", "Local"),
    ]

    for date in dates:
        for month, day, name, htype in fixed_holidays:
            if date.month == month and date.day == day:
                records.append(
                    {
                        "date": date,
                        "type": htype,
                        "name": name,
                        "transferred": False,
                    }
                )

        # Random regional events (~2% of days)
        if rng.random() < 0.02:
            records.append(
                {
                    "date": date,
                    "type": "Regional",
                    "name": f"Regional_Event_{date.strftime('%Y%m%d')}",
                    "transferred": rng.random() < 0.1,
                }
            )

    return (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["date", "type", "name", "transferred"])
    )


def _generate_oil(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic oil price series (random walk with drift)."""
    n = len(dates)
    price = np.zeros(n)
    price[0] = rng.uniform(40, 80)

    for i in range(1, n):
        shock = rng.normal(0, 1.0)
        drift = 0.01  # Slight upward drift
        price[i] = price[i - 1] + drift + shock
        price[i] = max(price[i], 10)  # Floor

    return pd.DataFrame({"date": dates, "oil_price": np.round(price, 2)})


def validate_synthetic_schema(data: dict[str, pd.DataFrame]) -> list[str]:
    """Validate that synthetic data conforms to the expected schema.

    Args:
        data: Dict from generate_synthetic_data().

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    # Sales schema
    if "sales" not in data:
        errors.append("Missing 'sales' table")
    else:
        sales = data["sales"]
        required_cols = {"date", "store_id", "family", "sales", "onpromotion"}
        missing = required_cols - set(sales.columns)
        if missing:
            errors.append(f"Sales missing columns: {missing}")
        if len(sales) == 0:
            errors.append("Sales table is empty")
        if "sales" in sales.columns and (sales["sales"] < 0).any():
            errors.append("Sales has negative values")
        if "date" in sales.columns and sales["date"].isna().any():
            errors.append("Sales has null dates")

    # Stores schema
    if "stores" not in data:
        errors.append("Missing 'stores' table")
    else:
        stores = data["stores"]
        required_cols = {"store_id", "city", "state", "type", "cluster"}
        missing = required_cols - set(stores.columns)
        if missing:
            errors.append(f"Stores missing columns: {missing}")

    # Holidays schema
    if "holidays" not in data:
        errors.append("Missing 'holidays' table")
    else:
        holidays = data["holidays"]
        required_cols = {"date", "type", "name"}
        missing = required_cols - set(holidays.columns)
        if missing:
            errors.append(f"Holidays missing columns: {missing}")

    # Oil schema
    if "oil" not in data:
        errors.append("Missing 'oil' table")
    else:
        oil = data["oil"]
        if "date" not in oil.columns or "oil_price" not in oil.columns:
            errors.append("Oil missing 'date' or 'oil_price' column")

    return errors
