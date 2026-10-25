"""Main Streamlit Application Entrypoint (P6-T01)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from idfs.app.storage import RunStorage
from idfs.app.views.data_explorer import render_data_explorer
from idfs.app.views.drift_explain_view import render_drift_explain_view
from idfs.app.views.forecast_studio import render_forecast_studio
from idfs.app.views.inventory_view import render_inventory_view
from idfs.reporting.report_builder import generate_executive_html_report

# Ensure src is on sys.path for direct `streamlit run` invocation
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def main() -> None:
    st.set_page_config(
        page_title="IDFS - Intelligent Demand Forecasting System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar Navigation
    st.sidebar.title("📈 IDFS")
    st.sidebar.markdown(
        "**Intelligent Demand Forecasting System**\n\n*ICFAI / IcfaiTech Special Project*"
    )
    st.sidebar.divider()

    page = st.sidebar.radio(
        "Navigation Menu:",
        [
            "1. Data Explorer & EDA",
            "2. Forecast Studio",
            "3. Inventory & Alerts (N1+N2)",
            "4. Drift & Explainability (N4+N5)",
            "5. Executive Report Generator",
        ],
        key="main_nav_radio",
    )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Team:**
        - Pamdem Ravi Charan Reddy
        - Jaiti Vijaya Shravani
        - Vankayalapati Srujana
        - Pottipothu V M Krishna Swamy

        **Mentors:**
        - Dr. Sukanta Das
        - Dr. Sandeep Kumar Panda
        """
    )

    if page == "1. Data Explorer & EDA":
        render_data_explorer()
    elif page == "2. Forecast Studio":
        render_forecast_studio()
    elif page == "3. Inventory & Alerts (N1+N2)":
        render_inventory_view()
    elif page == "4. Drift & Explainability (N4+N5)":
        render_drift_explain_view()
    elif page == "5. Executive Report Generator":
        render_report_generator()


def render_report_generator() -> None:
    st.header("Executive Report Generator & Planner Decision Review (P6-T07)")
    st.markdown(
        "Review recent system performance, record human planner decisions, and export executive reports."
    )

    storage = RunStorage()
    runs_df = storage.get_recent_runs(limit=15)

    st.subheader("Historical Simulation Runs in Database")
    if not runs_df.empty:
        st.dataframe(runs_df, use_container_width=True)
    else:
        st.info("No saved runs recorded yet. Save a simulation run from the Inventory page.")

    st.subheader("Planner Decision & Rationales")
    c1, c2 = st.columns([1, 2])
    with c1:
        decision_val = st.selectbox(
            "Overall Replenishment Plan Decision:",
            [
                "APPROVED (Production Release)",
                "APPROVED WITH MODIFICATIONS",
                "REJECTED (Needs Review)",
            ],
        )
    with c2:
        planner_notes = st.text_area(
            "Executive Comments & Viva Notes:",
            "All forecasts reviewed and validated against historical 3-fold cross-validation. "
            "Conformal Quantile Policy adopted for high-margin category replenishment to minimize stockouts.",
        )

    # Load summary tables for HTML report compilation
    models_df = (
        pd.read_csv("reports/tables/phase3_model_comparison.csv")
        if Path("reports/tables/phase3_model_comparison.csv").exists()
        else None
    )
    inv_df = (
        pd.read_csv("reports/tables/inventory_simulation_results.csv")
        if Path("reports/tables/inventory_simulation_results.csv").exists()
        else None
    )
    alerts_df = (
        pd.read_csv("reports/tables/drift_monitoring_log.csv")
        if Path("reports/tables/drift_monitoring_log.csv").exists()
        else None
    )
    shap_df = (
        pd.read_csv("reports/tables/shap_feature_importance.csv")
        if Path("reports/tables/shap_feature_importance.csv").exists()
        else None
    )

    html_report = generate_executive_html_report(
        model_comparison_df=models_df,
        inventory_results_df=inv_df,
        alerts_df=alerts_df[alerts_df["drift_detected"]].tail(5) if alerts_df is not None else None,
        shap_importance_df=shap_df,
        planner_notes=f"Decision: {decision_val}\nNotes: {planner_notes}",
    )

    st.subheader("Download Executive Artifacts")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button(
            label="Download Styled Executive Report (HTML)",
            data=html_report,
            file_name="IDFS_Executive_Report.html",
            mime="text/html",
        )
    with btn_col2:
        if inv_df is not None:
            st.download_button(
                label="Download Inventory Policy Summary (CSV)",
                data=inv_df.to_csv(index=False),
                file_name="inventory_policy_summary.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
