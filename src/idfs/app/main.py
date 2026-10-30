"""Main Streamlit Application Entrypoint (P6-T01) with professional UI styling and tooltips."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from idfs.app.storage import RunStorage
from idfs.app.views.backend_view import render_backend_view
from idfs.app.views.data_explorer import render_data_explorer
from idfs.app.views.drift_explain_view import render_drift_explain_view
from idfs.app.views.forecast_studio import render_forecast_studio
from idfs.app.views.inventory_view import render_inventory_view
from idfs.reporting.report_builder import generate_executive_html_report

# Ensure src is on sys.path for direct `streamlit run` invocation
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def apply_custom_styles() -> None:
    """Inject clean, modern, professional CSS styles."""
    st.markdown(
        """
        <style>
        /* Sleek card container styles */
        .stMetric {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            padding: 14px 18px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .stMetric:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(37, 99, 235, 0.15);
            border-color: rgba(37, 99, 235, 0.4);
        }
        /* Top executive banner */
        .top-banner {
            background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
            border-left: 4px solid #3b82f6;
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .banner-pill {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-left: 6px;
        }
        /* Expander headers */
        .streamlit-expanderHeader {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            border-radius: 6px;
        }
        /* Tab formatting */
        button[data-baseweb="tab"] {
            font-weight: 600;
            font-size: 0.92rem;
        }
        /* Sidebar badge */
        .sidebar-badge {
            background: linear-gradient(135deg, #2563eb, #7c3aed);
            color: white;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 14px;
            display: inline-block;
            margin-bottom: 8px;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header() -> None:
    """Render top executive status banner across all dashboard views."""
    st.markdown(
        """
        <div class="top-banner">
            <div>
                <span style="font-weight:700; color:#f8fafc; font-size:1.05rem;">IDFS Enterprise Platform</span>
                <span class="banner-pill">🟢 15 Stores Active</span>
                <span class="banner-pill">📦 20 Retail Products</span>
                <span class="banner-pill">📊 219K Sales Records</span>
                <span class="banner-pill">⚡ FastAPI REST (Port 8000)</span>
            </div>
            <div style="font-size:0.8rem; color:#94a3b8;">
                ICFAI Engineering Special Project
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="IDFS - Intelligent Demand Forecasting System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_styles()

    # Sidebar Header
    st.sidebar.markdown(
        '<span class="sidebar-badge">v1.0 • Academic Production</span>', unsafe_allow_html=True
    )
    st.sidebar.title("📈 IDFS")
    st.sidebar.markdown(
        "**Intelligent Demand Forecasting System**\n\n*ICFAI / IcfaiTech Special Project*",
        help="ℹ️ Autonomous, multi-paradigm AI demand forecasting and cost-optimal inventory replenishment system.",
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
            "6. 🔌 Backend API & Architecture",
        ],
        key="main_nav_radio",
        help="ℹ️ Navigate across data profiling, forecasting, inventory optimization, production surveillance, reports, and backend API.",
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

    render_top_header()

    if page == "1. Data Explorer & EDA":
        render_data_explorer()
    elif page == "2. Forecast Studio":
        render_forecast_studio()
    elif page == "3. Inventory & Alerts (N1+N2)":
        render_inventory_view()
    elif page == "4. Drift & Explainability (N4+N5)":
        render_drift_explain_view()
    elif page == "5. Executive Report Generator":
        render_reporting_view()
    elif page == "6. 🔌 Backend API & Architecture":
        render_backend_view()


def render_reporting_view() -> None:
    """Render executive summary and report export page."""
    st.header("Executive Reporting & Audit Trail (P6-T07)")
    st.caption(
        "Review recent system performance, record human planner decisions, and export executive reports."
    )

    storage = RunStorage()
    runs_df = storage.get_recent_runs(limit=15)

    with st.expander("🗄️ Historical Database Runs (SQLite Audit Trail)", expanded=True):
        if not runs_df.empty:
            st.dataframe(runs_df, use_container_width=True)
        else:
            st.info("No saved runs recorded yet. Save a simulation run from the Inventory page.")

    with st.expander("✍️ Human Review & Executive Rationales", expanded=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            decision_val = st.selectbox(
                "Overall Replenishment Plan Decision:",
                [
                    "APPROVED (Production Release)",
                    "APPROVED WITH MODIFICATIONS",
                    "REJECTED (Needs Review)",
                ],
                help="ℹ️ Official sign-off status recorded in the executive report.",
            )
        with c2:
            planner_notes = st.text_area(
                "Executive Comments & Viva Notes:",
                "All forecasts reviewed and validated against historical 3-fold cross-validation. "
                "Conformal Quantile Policy adopted for high-margin category replenishment to minimize stockouts.",
                help="ℹ️ Qualitative notes and justification included in the downloaded report.",
            )

    # Load summary tables for HTML report compilation
    models_df = (
        pd.read_csv("reports/tables/final_model_evaluation.csv")
        if Path("reports/tables/final_model_evaluation.csv").exists()
        else (
            pd.read_csv("reports/tables/phase3_model_comparison.csv")
            if Path("reports/tables/phase3_model_comparison.csv").exists()
            else None
        )
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

    with st.expander("📥 Download & Export Executive Deliverables", expanded=True):
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            st.download_button(
                label="📄 Download Styled Executive Report (HTML)",
                data=html_report,
                file_name="IDFS_Executive_Report.html",
                mime="text/html",
                help="ℹ️ Generates a self-contained, publication-ready executive report with embedded tables and SHAP drivers.",
            )
        with btn_col2:
            if inv_df is not None:
                st.download_button(
                    label="📊 Download Inventory Policy Summary (CSV)",
                    data=inv_df.to_csv(index=False),
                    file_name="inventory_policy_summary.csv",
                    mime="text/csv",
                    help="ℹ️ Tabular CSV export of inventory policy benchmark costs and fill rates.",
                )

    with st.expander("👁️ Live Preview of Executive HTML Report", expanded=False):
        st.components.v1.html(html_report, height=600, scrolling=True)


if __name__ == "__main__":
    main()
