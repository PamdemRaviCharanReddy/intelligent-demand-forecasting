"""Unit tests for Phase 1 data ingestion, validation, cleaning, and fusion."""

from __future__ import annotations

import pandas as pd

from idfs.data.cleaning import clean_holidays, clean_oil, clean_sales
from idfs.data.fusion import fuse_dataset
from idfs.data.ingestion import RawDataset
from idfs.data.validation import validate_raw_dataset


class TestDataCleaning:
    def test_clean_sales_clamps_negative_and_deduplicates(self) -> None:
        raw_df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-03"],
                "store_id": ["S1", "S1", "S1", "S1"],
                "family": ["F1", "F1", "F1", "F1"],
                "sales": [10.0, 10.0, -5.0, 20.0],
            }
        )
        cleaned, report = clean_sales(raw_df, impute_missing_dates=False)
        assert len(cleaned) == 3
        assert report.duplicates_removed == 1
        assert report.negative_values_clamped == 1
        assert (cleaned["sales"] >= 0).all()

    def test_clean_sales_imputes_missing_dates(self) -> None:
        # Date gap between 2023-01-01 and 2023-01-04 (missing 01-02 and 01-03)
        raw_df = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-04"],
                "store_id": ["S1", "S1"],
                "family": ["F1", "F1"],
                "sales": [10.0, 25.0],
            }
        )
        cleaned, report = clean_sales(raw_df, impute_missing_dates=True)
        assert len(cleaned) == 4
        assert report.missing_dates_imputed == 2
        # Imputed values should be 0.0
        imputed_sales = cleaned.loc[cleaned["date"].isin(["2023-01-02", "2023-01-03"]), "sales"]
        assert (imputed_sales == 0.0).all()

    def test_clean_oil_continuous_and_filled(self) -> None:
        raw_oil = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-04"],
                "oil_price": [75.0, 80.0],
            }
        )
        cleaned_oil = clean_oil(raw_oil)
        assert len(cleaned_oil) == 4
        assert not cleaned_oil["oil_price"].isna().any()
        # Days 2 and 3 should forward-fill 75.0
        assert cleaned_oil.loc[cleaned_oil["date"] == "2023-01-02", "oil_price"].iloc[0] == 75.0

    def test_clean_holidays_flags_binary(self) -> None:
        raw_hols = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-05-01"],
                "type": ["National", "National"],
                "name": ["New Year", "Labour Day"],
            }
        )
        clean_h = clean_holidays(raw_hols)
        assert len(clean_h) == 2
        assert (clean_h["is_holiday"] == 1).all()


class TestDataValidation:
    def test_validates_healthy_dataset(self) -> None:
        raw = RawDataset(
            sales=pd.DataFrame(
                {
                    "date": ["2023-01-01", "2023-01-02"],
                    "store_id": ["S1", "S1"],
                    "family": ["F1", "F1"],
                    "sales": [10.0, 15.0],
                }
            )
        )
        res = validate_raw_dataset(raw)
        assert res.is_valid
        assert len(res.errors) == 0

    def test_rejects_empty_sales(self) -> None:
        raw = RawDataset(sales=pd.DataFrame())
        res = validate_raw_dataset(raw)
        assert not res.is_valid
        assert any("missing or empty" in e for e in res.errors)

    def test_rejects_missing_date_column(self) -> None:
        raw = RawDataset(sales=pd.DataFrame({"sales": [10.0]}))
        res = validate_raw_dataset(raw)
        assert not res.is_valid
        assert any("mandatory" in e for e in res.errors)


class TestDataFusion:
    def test_fuses_sales_and_exogenous(self) -> None:
        sales = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02"],
                "store_id": ["S1", "S1"],
                "family": ["F1", "F1"],
                "sales": [10.0, 15.0],
            }
        )
        stores = pd.DataFrame(
            {
                "store_id": ["S1"],
                "city": ["Quito"],
                "type": ["A"],
                "cluster": [1],
            }
        )
        holidays = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "type": ["National"],
                "name": ["New Year"],
            }
        )
        raw = RawDataset(sales=sales, stores=stores, holidays=holidays)
        fused = fuse_dataset(raw)

        assert len(fused) == 2
        assert "city" in fused.columns
        assert "is_holiday" in fused.columns
        # Day 1 is a holiday, Day 2 is not
        assert fused.loc[fused["date"] == "2023-01-01", "is_holiday"].iloc[0] == 1
        assert fused.loc[fused["date"] == "2023-01-02", "is_holiday"].iloc[0] == 0
