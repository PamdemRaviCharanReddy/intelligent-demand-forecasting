"""Data ingestion module for IDFS.

Loads retail sales datasets and exogenous sources from CSV or Parquet files.
Detects available tables and degrades gracefully if optional sources are missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RawDataset:
    """Container for ingested raw tables."""

    sales: pd.DataFrame
    stores: pd.DataFrame | None = None
    holidays: pd.DataFrame | None = None
    oil: pd.DataFrame | None = None
    promotions: pd.DataFrame | None = None
    source_format: str = "unknown"
    source_dir: str = ""


def _find_file(directory: Path, base_names: list[str]) -> Path | None:
    """Find a file matching any of the base names with .csv or .parquet extension."""
    for base in base_names:
        for ext in [".parquet", ".csv"]:
            candidate = directory / f"{base}{ext}"
            if candidate.exists():
                return candidate
    return None


def _read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame with standardized types."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def load_raw_dataset(source_dir: str | Path) -> RawDataset:
    """Ingest raw dataset from a directory containing CSV or Parquet files.

    Searches for:
    - sales / train / transactions
    - stores
    - holidays / holidays_events
    - oil
    - promotions

    Args:
        source_dir: Path to directory containing raw files.

    Returns:
        RawDataset with loaded DataFrames.

    Raises:
        FileNotFoundError: If the mandatory sales table cannot be found.
    """
    p = Path(source_dir)
    if not p.exists():
        raise FileNotFoundError(f"Source directory not found: {p}")

    # 1. Sales (mandatory)
    sales_path = _find_file(p, ["sales", "train"])
    if sales_path is None:
        raise FileNotFoundError(f"Mandatory sales table ('sales' or 'train') not found in {p}")

    logger.info("Loading sales data from %s", sales_path)
    sales = _read_table(sales_path)

    # Standardize column naming if from Kaggle competition
    rename_map = {
        "store_nbr": "store_id",
        "unit_sales": "sales",
    }
    sales = sales.rename(columns={k: v for k, v in rename_map.items() if k in sales.columns})

    # Ensure store_id is string
    if "store_id" in sales.columns:
        sales["store_id"] = sales["store_id"].astype(str)

    # 2. Stores (optional)
    stores_path = _find_file(p, ["stores"])
    stores = None
    if stores_path:
        logger.info("Loading stores metadata from %s", stores_path)
        stores = _read_table(stores_path)
        if "store_nbr" in stores.columns:
            stores = stores.rename(columns={"store_nbr": "store_id"})
        stores["store_id"] = stores["store_id"].astype(str)
    else:
        logger.warning("Optional 'stores' table not found; store metadata disabled.")

    # 3. Holidays (optional)
    holidays_path = _find_file(p, ["holidays", "holidays_events"])
    holidays = None
    if holidays_path:
        logger.info("Loading holidays from %s", holidays_path)
        holidays = _read_table(holidays_path)
    else:
        logger.warning("Optional 'holidays' table not found; holiday regressors disabled.")

    # 4. Oil (optional)
    oil_path = _find_file(p, ["oil"])
    oil = None
    if oil_path:
        logger.info("Loading oil commodity prices from %s", oil_path)
        oil = _read_table(oil_path)
        if "dcoilwtico" in oil.columns:
            oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    else:
        logger.warning("Optional 'oil' table not found; oil price regressor disabled.")

    # 5. Promotions (optional)
    promos_path = _find_file(p, ["promotions"])
    promotions = None
    if promos_path:
        logger.info("Loading promotions from %s", promos_path)
        promotions = _read_table(promos_path)
        if "store_nbr" in promotions.columns:
            promotions = promotions.rename(columns={"store_nbr": "store_id"})
        if "store_id" in promotions.columns:
            promotions["store_id"] = promotions["store_id"].astype(str)

    fmt = sales_path.suffix.lstrip(".")
    return RawDataset(
        sales=sales,
        stores=stores,
        holidays=holidays,
        oil=oil,
        promotions=promotions,
        source_format=fmt,
        source_dir=str(p),
    )
