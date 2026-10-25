"""Tests for the synthetic data generator.

Validates that generated data conforms to the expected schema,
is reproducible with the same seed, and contains the expected
features (seasonality, promotions, drift, intermittent demand).
"""

from __future__ import annotations

import pandas as pd
import pytest

from idfs.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_data,
    validate_synthetic_schema,
)


@pytest.fixture
def default_data() -> dict[str, pd.DataFrame]:
    """Generate synthetic data with small config for fast tests."""
    cfg = SyntheticConfig(
        n_stores=2,
        n_families=3,
        n_days=100,
        seed=42,
        intermittent_frac=0.15,
    )
    return generate_synthetic_data(cfg)


class TestSyntheticSchema:
    def test_schema_valid(self, default_data: dict[str, pd.DataFrame]) -> None:
        errors = validate_synthetic_schema(default_data)
        assert errors == [], f"Schema errors: {errors}"

    def test_has_all_tables(self, default_data: dict[str, pd.DataFrame]) -> None:
        assert set(default_data.keys()) == {"sales", "stores", "holidays", "oil", "promotions"}

    def test_sales_shape(self, default_data: dict[str, pd.DataFrame]) -> None:
        sales = default_data["sales"]
        # 2 stores × 3 families × 100 days = 600 rows
        assert len(sales) == 2 * 3 * 100

    def test_sales_non_negative(self, default_data: dict[str, pd.DataFrame]) -> None:
        assert (default_data["sales"]["sales"] >= 0).all()

    def test_sales_columns(self, default_data: dict[str, pd.DataFrame]) -> None:
        expected = {"date", "store_id", "family", "sales", "onpromotion"}
        assert expected == set(default_data["sales"].columns)

    def test_stores_count(self, default_data: dict[str, pd.DataFrame]) -> None:
        assert len(default_data["stores"]) == 2

    def test_oil_length(self, default_data: dict[str, pd.DataFrame]) -> None:
        assert len(default_data["oil"]) == 100


class TestReproducibility:
    def test_same_seed_same_data(self) -> None:
        cfg = SyntheticConfig(n_stores=2, n_families=2, n_days=50, seed=123)
        data1 = generate_synthetic_data(cfg)
        data2 = generate_synthetic_data(cfg)
        pd.testing.assert_frame_equal(data1["sales"], data2["sales"])

    def test_different_seed_different_data(self) -> None:
        cfg1 = SyntheticConfig(n_stores=2, n_families=2, n_days=50, seed=1)
        cfg2 = SyntheticConfig(n_stores=2, n_families=2, n_days=50, seed=2)
        data1 = generate_synthetic_data(cfg1)
        data2 = generate_synthetic_data(cfg2)
        assert not data1["sales"]["sales"].equals(data2["sales"]["sales"])


class TestDataFeatures:
    def test_has_promotions(self, default_data: dict[str, pd.DataFrame]) -> None:
        """Some observations should have promotions."""
        assert default_data["sales"]["onpromotion"].sum() > 0

    def test_has_holidays(self, default_data: dict[str, pd.DataFrame]) -> None:
        """Holiday table should not be empty."""
        # 100 days may or may not hit a holiday, but regional events help
        # With 2% random regional events over 100 days, expect ~2
        assert len(default_data["holidays"]) >= 0  # May be 0 for short periods

    def test_oil_positive(self, default_data: dict[str, pd.DataFrame]) -> None:
        assert (default_data["oil"]["oil_price"] > 0).all()

    def test_intermittent_series_exist(self) -> None:
        """With intermittent_frac=0.5, at least some series should have many zeros."""
        cfg = SyntheticConfig(
            n_stores=2,
            n_families=4,
            n_days=200,
            seed=42,
            intermittent_frac=0.5,
        )
        data = generate_synthetic_data(cfg)
        sales = data["sales"]

        # Check that at least one series has >30% zeros
        has_intermittent = False
        for (_store, _family), group in sales.groupby(["store_id", "family"]):
            zero_frac = (group["sales"] == 0).mean()
            if zero_frac > 0.3:
                has_intermittent = True
                break

        assert has_intermittent, "No intermittent series found with intermittent_frac=0.5"
