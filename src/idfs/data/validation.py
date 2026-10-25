"""Data validation and schema integrity checks for IDFS.

Validates ingested tables against the formal data contract (docs/DATA_CONTRACT.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from idfs.data.ingestion import RawDataset


@dataclass
class ValidationResult:
    """Outcome of dataset validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Produce human-readable validation summary."""
        status = "PASSED" if self.is_valid else "FAILED"
        msg = [f"Validation {status} (Errors: {len(self.errors)}, Warnings: {len(self.warnings)})"]
        for err in self.errors:
            msg.append(f"  [ERROR] {err}")
        for warn in self.warnings:
            msg.append(f"  [WARN]  {warn}")
        return "\n".join(msg)


def validate_raw_dataset(raw: RawDataset) -> ValidationResult:
    """Validate raw dataset against data contract specifications.

    Args:
        raw: RawDataset instance.

    Returns:
        ValidationResult indicating validity, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Sales Validation ──
    sales = raw.sales
    if sales is None or len(sales) == 0:
        errors.append("Sales table is missing or empty.")
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    required_sales_cols = {"date", "sales"}
    missing_cols = required_sales_cols - set(sales.columns)
    if missing_cols:
        errors.append(f"Sales table missing mandatory columns: {missing_cols}")

    if "date" in sales.columns:
        if sales["date"].isna().any():
            errors.append("Sales table contains null dates.")
        try:
            pd.to_datetime(sales["date"])
        except Exception as e:
            errors.append(f"Sales 'date' column cannot be parsed as datetime: {e}")

    if "sales" in sales.columns:
        if sales["sales"].isna().any():
            n_nulls = int(sales["sales"].isna().sum())
            errors.append(f"Sales column contains {n_nulls} null values.")
        if (sales["sales"] < 0).any():
            n_neg = int((sales["sales"] < 0).sum())
            warnings.append(
                f"Sales contains {n_neg} negative values (will be clamped to zero during cleaning)."
            )

    if "store_id" not in sales.columns:
        warnings.append("Sales table lacks 'store_id'; treated as single-store dataset.")

    if "family" not in sales.columns:
        warnings.append("Sales table lacks 'family'; treated as single-item dataset.")

    # ── 2. Stores Validation ──
    if raw.stores is not None:
        stores = raw.stores
        if "store_id" not in stores.columns:
            errors.append("Stores table missing mandatory 'store_id' primary key.")
        else:
            if stores["store_id"].duplicated().any():
                errors.append("Stores table contains duplicate 'store_id' records.")
            # Cross-table referential integrity
            if "store_id" in sales.columns:
                sales_stores = set(sales["store_id"].unique())
                known_stores = set(stores["store_id"].unique())
                unknown = sales_stores - known_stores
                if unknown:
                    warnings.append(
                        f"Sales contains store IDs not listed in stores metadata: {list(unknown)[:5]}..."
                    )

    # ── 3. Holidays Validation ──
    if raw.holidays is not None:
        holidays = raw.holidays
        if "date" not in holidays.columns:
            errors.append("Holidays table missing mandatory 'date' column.")
        else:
            try:
                pd.to_datetime(holidays["date"])
            except Exception as e:
                errors.append(f"Holidays 'date' column cannot be parsed as datetime: {e}")

    # ── 4. Oil Validation ──
    if raw.oil is not None:
        oil = raw.oil
        if "date" not in oil.columns or "oil_price" not in oil.columns:
            errors.append("Oil table missing 'date' or 'oil_price' column.")
        else:
            if (oil["oil_price"] < 0).any():
                warnings.append("Oil price contains negative values.")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
