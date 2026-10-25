"""Smoke and integration tests for Streamlit application, RunStorage, and Report Builder (Phase 6)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from idfs.app.storage import RunStorage
from idfs.reporting.report_builder import export_metrics_csv, generate_executive_html_report


def test_run_storage_crud(tmp_path: Path) -> None:
    """Verify SQLite run storage inserts, reads, and updates records cleanly."""
    db_file = tmp_path / "test_runs.db"
    storage = RunStorage(db_path=db_file)

    # 1. Insert a simulation run
    run_id = storage.save_run(
        model_name="QuantileXGBoost",
        policy_name="ConformalQuantilePolicy",
        lead_time=5,
        holding_cost=120.50,
        stockout_cost=450.00,
        ordering_cost=200.00,
        total_cost=770.50,
        fill_rate=0.965,
        stockout_periods=2,
        n_orders=4,
        planner_decision="PENDING",
        planner_notes="Awaiting warehouse capacity confirmation",
    )
    assert run_id is not None
    assert len(run_id) > 0

    # 2. Retrieve history
    history = storage.get_all_runs()
    assert len(history) == 1
    assert history.iloc[0]["run_id"] == run_id
    assert history.iloc[0]["model_name"] == "QuantileXGBoost"
    assert history.iloc[0]["policy_name"] == "ConformalQuantilePolicy"
    assert history.iloc[0]["planner_decision"] == "PENDING"

    # 3. Fetch single run by ID
    record = storage.get_run_by_id(run_id)
    assert record is not None
    assert record["total_cost"] == pytest.approx(770.50)

    # 4. Update planner decision and notes
    updated = storage.update_planner_review(
        run_id=run_id,
        decision="APPROVED",
        notes="Approved after confirming with supplier lead time",
    )
    assert updated is True

    updated_record = storage.get_run_by_id(run_id)
    assert updated_record is not None
    assert updated_record["planner_decision"] == "APPROVED"
    assert "lead time" in updated_record["planner_notes"]


def test_executive_html_report_builder(tmp_path: Path) -> None:
    """Verify HTML executive report renders valid markup with tables and notes."""
    comp_df = pd.DataFrame(
        {
            "model": ["XGBoost", "LSTM"],
            "mae": [4.2, 3.8],
            "wape": [0.18, 0.16],
        }
    )
    inv_df = pd.DataFrame(
        {
            "policy": ["ConformalQuantile", "Naive"],
            "total_cost": [12500.0, 24000.0],
            "fill_rate": [0.97, 0.88],
        }
    )
    alerts_df = pd.DataFrame(
        {
            "timestamp": ["2026-10-25"],
            "alert": ["No drift detected"],
        }
    )
    save_file = tmp_path / "executive_report.html"

    html = generate_executive_html_report(
        model_comparison_df=comp_df,
        inventory_results_df=inv_df,
        alerts_df=alerts_df,
        planner_notes="System ready for academic viva demonstration.",
        save_path=save_file,
    )

    assert "<!DOCTYPE html>" in html
    assert "Intelligent Demand Forecasting System" in html
    assert "XGBoost" in html
    assert "ConformalQuantile" in html
    assert "System ready for academic viva demonstration." in html
    assert save_file.exists()
    assert len(save_file.read_text(encoding="utf-8")) > 500


def test_export_metrics_csv(tmp_path: Path) -> None:
    """Verify CSV export utility writes valid CSV file."""
    df = pd.DataFrame({"metric": ["MAE", "RMSE"], "value": [1.25, 2.10]})
    csv_file = tmp_path / "metrics_export.csv"

    exported = export_metrics_csv(df, csv_file)
    assert exported.exists()
    read_back = pd.read_csv(exported)
    assert len(read_back) == 2
    assert list(read_back["metric"]) == ["MAE", "RMSE"]


def test_app_main_initial_render() -> None:
    """Verify Streamlit app renders initial page cleanly without exceptions."""
    app_path = str(
        (
            Path(__file__).resolve().parent.parent.parent / "src" / "idfs" / "app" / "main.py"
        ).resolve()
    )
    at = AppTest.from_file(app_path, default_timeout=30).run(timeout=30)
    assert not at.exception, f"AppTest raised exception on initial load: {at.exception}"
    radio = at.sidebar.radio(key="main_nav_radio")
    assert radio.value == "1. Data Explorer & EDA"


@pytest.mark.parametrize(
    "page_name",
    [
        "2. Forecast Studio",
        "3. Inventory & Alerts (N1+N2)",
        "4. Drift & Explainability (N4+N5)",
        "5. Executive Report Generator",
    ],
)
def test_app_pages_render_headless(page_name: str) -> None:
    """Verify Streamlit app renders each secondary page without exceptions."""
    app_path = str(
        (
            Path(__file__).resolve().parent.parent.parent / "src" / "idfs" / "app" / "main.py"
        ).resolve()
    )
    at = AppTest.from_file(app_path, default_timeout=30).run(timeout=30)
    radio = at.sidebar.radio(key="main_nav_radio")
    radio.set_value(page_name).run(timeout=30)
    assert not at.exception, f"Exception on {page_name}: {at.exception}"
