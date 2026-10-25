"""Surveillance Drift Monitoring & Explainability View (P6-T05) with enhanced visual analytics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_drift_explain_view() -> None:
    """Render the Drift Monitor and Model Explainability page with interactive visualizations."""
    st.header("Model Surveillance & Explainability (Novelties N4 & N5)")
    st.caption(
        "Monitor concept drift in production, examine Tree-SHAP demand drivers, and formulate counterfactual What-If scenarios."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📡 Production Drift Monitor (N4)",
            "🌳 Tree-SHAP Explainability (N5)",
            "⚡ What-If Scenario Simulator (N5)",
        ]
    )

    # --- TAB 1: DRIFT MONITOR ---
    with tab1:
        st.subheader("Concept Drift Surveillance & Retrain Alerts")
        st.caption(
            "Active monitoring tracks rolling error metrics against historical baseline thresholds."
        )

        log_p = Path("reports/tables/drift_monitoring_log.csv")
        fig_p = Path("reports/figures/drift_detection_demo.png")

        if fig_p.exists():
            st.image(str(fig_p), use_container_width=True)

        if log_p.exists():
            drift_df = pd.read_csv(log_p)
            drift_events = drift_df[drift_df["drift_detected"] == True]  # noqa: E712

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric(
                    "Surveillance Days Tracked",
                    f"{len(drift_df):,}",
                    help="ℹ️ Total continuous production days under automated error surveillance.",
                )
            with c2:
                st.metric(
                    "Drift Trigger Events",
                    f"{len(drift_events)}",
                    delta="Threshold Breached" if len(drift_events) > 0 else "0 Breaches",
                    delta_color="inverse",
                    help="ℹ️ Occasions where rolling WAPE exceeded 1.5x baseline error, indicating model decay.",
                )
            with c3:
                status = "CRITICAL ALERT" if len(drift_events) > 0 else "HEALTHY"
                st.metric(
                    "Model Health Status",
                    status,
                    delta="Retrain Recommended" if len(drift_events) > 0 else "Normal Operation",
                    delta_color="inverse",
                    help="ℹ️ Recommended operational action for human planners and ML engineering.",
                )

            with st.expander(
                "📋 View Surveillance Telemetry Log (Last 15 Observations)", expanded=False
            ):
                st.dataframe(drift_df.tail(15), use_container_width=True)
        else:
            st.info(
                "Run `python scripts/run_drift_and_explainability.py` to populate drift telemetry."
            )

    # --- TAB 2: SHAP EXPLAINABILITY ---
    with tab2:
        st.subheader("Tree-SHAP Feature Importance & Driver Attribution")
        st.caption(
            "Mathematically exact Shapley value decomposition of the global gradient-boosted forecaster."
        )

        shap_table_p = Path("reports/tables/shap_feature_importance.csv")
        summary_fig_p = Path("reports/figures/shap_summary.png")
        waterfall_fig_p = Path("reports/figures/shap_waterfall.png")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Global Demand Attribution")
            if summary_fig_p.exists():
                st.image(str(summary_fig_p), use_container_width=True)
            if shap_table_p.exists():
                shap_df = pd.read_csv(shap_table_p)
                st.dataframe(
                    shap_df.head(10).style.format(
                        {"mean_abs_shap": "{:.2f}", "relative_importance": "{:.2%}"}
                    ),
                    use_container_width=True,
                )

        with col_right:
            st.markdown("#### Local Prediction Waterfall")
            if waterfall_fig_p.exists():
                st.image(str(waterfall_fig_p), use_container_width=True)
                st.caption(
                    "Deconstructs the additive contributions pushing individual daily demand above or below base value."
                )

    # --- TAB 3: WHAT-IF SIMULATOR ---
    with tab3:
        st.subheader("Counterfactual What-If Scenario Stress Testing")
        st.caption(
            "Simulate demand responsiveness to promotional interventions and macroeconomic commodity shifts."
        )

        with st.expander("⚙️ Configure Scenario Parameters (Click to open/close)", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                promo_toggle = st.toggle(
                    "Activate Full-Chain Promotional Campaign",
                    value=True,
                    help="ℹ️ Simulates the effect of actively promoting the category across all store circulars and digital ads.",
                )
            with c2:
                oil_shock = st.slider(
                    "Macro Oil Price Shock (%):",
                    min_value=-50,
                    max_value=50,
                    value=20,
                    step=5,
                    help="ℹ️ Simulates commodity fuel/transportation inflation shocks that impact retail purchasing power.",
                )

        # Baseline calculation
        base_volume = 120158.0
        promo_impact = 0.1837 if promo_toggle else 0.0
        oil_impact = (oil_shock / 100.0) * -0.05  # Negative price elasticity
        total_lift_pct = promo_impact + oil_impact
        scenario_volume = base_volume * (1.0 + total_lift_pct)
        delta = scenario_volume - base_volume

        promo_units = base_volume * promo_impact
        oil_units = base_volume * oil_impact

        # Metric Badges
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric(
            "Baseline Demand Volume",
            f"{base_volume:,.0f} units",
            help="ℹ️ Expected baseline demand under normal operating conditions without interventions.",
        )
        res_c2.metric(
            "Scenario Projected Volume",
            f"{scenario_volume:,.0f} units",
            delta=f"{delta:+,.0f} units",
            help="ℹ️ Projected volume incorporating promotional lift and macroeconomic elasticity.",
        )
        res_c3.metric(
            "Net Projected Demand Lift",
            f"{total_lift_pct:+.2%}",
            delta=f"{total_lift_pct:+.2%}",
            help="ℹ️ Percentage change in sales volume compared to baseline.",
        )

        st.divider()

        # Side-by-Side Comparison & Waterfall Analytics
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Scenario Volume Comparison")
            comp_fig = go.Figure()
            comp_fig.add_trace(
                go.Bar(
                    x=["Baseline", "What-If Scenario"],
                    y=[base_volume, scenario_volume],
                    marker_color=["#64748b", "#2563eb"],
                    text=[f"{base_volume:,.0f}", f"{scenario_volume:,.0f} ({total_lift_pct:+.1%})"],
                    textposition="auto",
                    width=0.45,
                )
            )
            comp_fig.update_layout(
                yaxis_title="Demand Units",
                template="plotly_white",
                yaxis=dict(range=[0, max(base_volume, scenario_volume) * 1.25]),
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(comp_fig, use_container_width=True)

        with chart_col2:
            st.markdown("#### Demand Lift Waterfall Decomposition")
            wf_fig = go.Figure(
                go.Waterfall(
                    name="Impact Decomposition",
                    orientation="v",
                    measure=["absolute", "relative", "relative", "total"],
                    x=["Base Volume", "Promotions Lift", "Oil Shock Impact", "Projected Total"],
                    textposition="outside",
                    text=[
                        f"{base_volume:,.0f}",
                        f"{promo_units:+,.0f}",
                        f"{oil_units:+,.0f}",
                        f"{scenario_volume:,.0f}",
                    ],
                    y=[base_volume, promo_units, oil_units, scenario_volume],
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    increasing={"marker": {"color": "#10b981"}},
                    totals={"marker": {"color": "#2563eb"}},
                )
            )
            wf_fig.update_layout(
                yaxis_title="Units",
                template="plotly_white",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(wf_fig, use_container_width=True)
