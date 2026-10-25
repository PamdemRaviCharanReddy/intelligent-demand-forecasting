"""End-to-end integration test for Phase 1 data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from idfs.data.eda import generate_eda_figures
from idfs.data.fusion import run_data_pipeline


@pytest.mark.integration
class TestDataPipelineE2E:
    def test_run_pipeline_on_synthetic_data(self, tmp_path: Path) -> None:
        source_dir = Path("data/synthetic")
        output_file = tmp_path / "processed_fused.parquet"

        # 1. Execute end-to-end data pipeline
        fused_df = run_data_pipeline(source_dir=source_dir, output_path=output_file)

        # 2. Assert output file exists and has rows
        assert output_file.exists()
        loaded = pd.read_parquet(output_file)
        assert len(loaded) > 0
        assert len(loaded) == len(fused_df)

        # 3. Assert core schemas
        expected_cols = {
            "date",
            "store_id",
            "family",
            "sales",
            "onpromotion",
            "is_holiday",
            "oil_price",
        }
        assert expected_cols.issubset(set(fused_df.columns))

        # 4. Assert non-negativity and no null dates
        assert (fused_df["sales"] >= 0).all()
        assert not fused_df["date"].isna().any()

        # 5. Generate EDA figures
        fig_dir = tmp_path / "figures"
        tbl_dir = tmp_path / "tables"
        figs = generate_eda_figures(fused_df, output_dir=fig_dir, tables_dir=tbl_dir)

        assert "sales_trend" in figs
        assert Path(figs["sales_trend"]).exists()
        assert "day_of_week_seasonality" in figs
        assert Path(figs["day_of_week_seasonality"]).exists()
        assert "demand_pattern_scatter" in figs
        assert Path(figs["demand_pattern_scatter"]).exists()
        assert "eda_summary_table" in figs
        assert Path(figs["eda_summary_table"]).exists()
