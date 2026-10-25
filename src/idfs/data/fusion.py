"""Multi-source dataset fusion for IDFS.

Merges cleaned sales transactions with store metadata, holiday events,
and commodity oil prices into a unified analysis-ready feature table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from idfs.data.cleaning import clean_holidays, clean_oil, clean_sales
from idfs.data.ingestion import RawDataset, load_raw_dataset
from idfs.data.validation import validate_raw_dataset

logger = logging.getLogger(__name__)


def fuse_dataset(raw: RawDataset) -> pd.DataFrame:
    """Clean and fuse all available raw tables into a single analytical table.

    Performs:
    1. Sales cleaning & missing date grid completion.
    2. Store metadata enrichment.
    3. Holiday calendar merging with binary flags.
    4. Oil price continuous regressor merging.
    5. Promotion details joining.
    6. Categorical memory optimization.

    Args:
        raw: Ingested RawDataset.

    Returns:
        Unified processed DataFrame ready for feature engineering.
    """
    logger.info("Cleaning sales transactions...")
    sales, cleaning_report = clean_sales(raw.sales)
    logger.info(
        "Sales cleaned: %d rows -> %d rows (%d missing dates imputed)",
        cleaning_report.initial_rows,
        cleaning_report.final_rows,
        cleaning_report.missing_dates_imputed,
    )

    fused = sales.copy()

    # ── 1. Enrich with Stores Metadata ──
    if raw.stores is not None:
        stores = raw.stores.copy()
        if "type" in stores.columns:
            stores = stores.rename(columns={"type": "store_type"})
        if "cluster" in stores.columns:
            stores = stores.rename(columns={"cluster": "store_cluster"})

        fused = pd.merge(fused, stores, on="store_id", how="left")
        logger.info("Enriched with store metadata: %s", list(stores.columns))

    # ── 2. Enrich with Holiday Indicators ──
    if raw.holidays is not None:
        holidays = clean_holidays(raw.holidays)
        holiday_cols = ["date", "is_holiday"]
        if "type" in holidays.columns:
            holidays = holidays.rename(columns={"type": "holiday_type"})
            holiday_cols.append("holiday_type")
        if "name" in holidays.columns:
            holiday_cols.append("name")

        fused = pd.merge(fused, holidays[holiday_cols], on="date", how="left")
        fused["is_holiday"] = fused["is_holiday"].fillna(0).astype(int)
        if "holiday_type" in fused.columns:
            fused["holiday_type"] = fused["holiday_type"].fillna("None")
        logger.info("Enriched with holiday indicators.")
    else:
        fused["is_holiday"] = 0
        fused["holiday_type"] = "None"

    # ── 3. Enrich with Oil Prices ──
    if raw.oil is not None:
        min_date = fused["date"].min()
        max_date = fused["date"].max()
        oil = clean_oil(raw.oil, start_date=min_date, end_date=max_date)
        fused = pd.merge(fused, oil[["date", "oil_price"]], on="date", how="left")
        fused["oil_price"] = fused["oil_price"].ffill().bfill().fillna(0.0)
        logger.info("Enriched with continuous oil prices.")
    else:
        fused["oil_price"] = 0.0

    # ── 4. Enrich with Promotions if separate ──
    if raw.promotions is not None and "promo_type" in raw.promotions.columns:
        promo_sub = raw.promotions[
            ["date", "store_id", "family", "promo_type", "promo_strength"]
        ].drop_duplicates(subset=["date", "store_id", "family"])
        fused = pd.merge(fused, promo_sub, on=["date", "store_id", "family"], how="left")
        fused["promo_type"] = fused["promo_type"].fillna("None")
        fused["promo_strength"] = fused["promo_strength"].fillna(0.0)

    # ── 5. Standardize Categorical Columns ──
    cat_candidates = [
        "store_id",
        "family",
        "city",
        "state",
        "store_type",
        "holiday_type",
        "promo_type",
    ]
    for col in cat_candidates:
        if col in fused.columns:
            fused[col] = fused[col].astype("category")

    fused = fused.sort_values(["store_id", "family", "date"]).reset_index(drop=True)
    return fused


def run_data_pipeline(
    source_dir: str | Path,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Execute end-to-end data ingestion, validation, cleaning, and fusion.

    Args:
        source_dir: Directory containing raw files (CSV or Parquet).
        output_path: Optional path to save the fused Parquet table.

    Returns:
        Cleaned, fused DataFrame.

    Raises:
        ValueError: If raw data fails contract validation.
    """
    logger.info("Starting data pipeline for source: %s", source_dir)
    raw = load_raw_dataset(source_dir)

    validation = validate_raw_dataset(raw)
    if not validation.is_valid:
        raise ValueError(f"Dataset validation failed:\n{validation.summary()}")

    logger.info("Validation successful: %s", validation.summary())

    fused = fuse_dataset(raw)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fused.to_parquet(out_p, index=False)
        logger.info("Saved fused dataset to %s (%d rows)", out_p, len(fused))

    return fused
