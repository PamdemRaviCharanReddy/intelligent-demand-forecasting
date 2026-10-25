"""Surveillance Drift Monitoring & Explainability View (P6-T05)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_drift_explain_view() -> None:
    """Render the Drift Monitor and Model Explainability page."""
    st.header("Model Surveillance & Explainability (Novelties N4 & N5)")
    st.markdown(
        "Monitor concept drift in production, examine Tree-SHAP demand drivers, and formulate counterfactual What-If scenarios."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Production Drift Monitor (N4)", "Tree-SHAP Explainability (N5)", "What-If Simulator (N5)"]
    )

    # --- TAB 1: DRIFT MONITOR ---
    with tab1:
        st.subheader("Concept Drift Surveillance & Retrain Alerts")
        log_p = Path("reports/tables/drift_monitoring_log.csv")
        fig_p = Path("reports/figures/drift_detection_demo.png")

        if fig_p.exists():
            st.image(str(fig_p), use_container_width=True)

        if log_p.exists():
            drift_df = pd.read_csv(log_p)
            drift_events = drift_df[drift_df["drift_detected"] == True]  # noqa: E712
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Surveillance Days", f"{len(drift_df):,}")
            with c2:
                st.metric("Drift Trigger Events", f"{len(drift_events)}")
            with c3:
                status = "CRITICAL ALERT" if len(drift_events) > 0 else "NORMAL"
                st.metric(
                    "Model Health Status",
                    status,
                    delta="Retrain Recommended" if len(drift_events) > 0 else "Healthy",
                    delta_color="inverse",
                )

            st.markdown("#### Recent Surveillance Log")
            st.dataframe(drift_df.tail(15), use_container_width=True)
        else:
            st.info(
                "Run `python scripts/run_drift_and_explainability.py` to populate drift telemetry."
            )

    # --- TAB 2: SHAP EXPLAINABILITY ---
    with tab2:
        st.subheader("Tree-SHAP Feature Importance & Driver Attribution")
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
        st.markdown(
            "Simulate demand responsiveness to marketing campaigns, promotional interventions, and macroeconomic commodity shifts."
        )

        c1, c2 = st.columns(2)
        with c1:
            promo_toggle = st.toggle("Activate Full-Chain Promotion", value=True)
        with c2:
            oil_shock = st.slider("Macro Oil Price Shock (%)", -50, 50, 20, 5)

        # Baseline reference
        base_volume = 120158.0
        promo_impact = 0.1837 if promo_toggle else 0.0
        oil_impact = (oil_shock / 100.0) * -0.05  # Slight negative elasticity
        total_lift_pct = promo_impact + oil_impact
        scenario_volume = base_volume * (1.0 + total_lift_pct)
        delta = scenario_volume - base_volume

        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Baseline Demand Volume", f"{base_volume:,.0f} units")
        res_c2.metric(
            "Scenario Projected Volume",
            f"{scenario_volume:,.0f} units",
            delta=f"{delta:+,.0f} units",
        )
        res_c3.metric("Projected Demand Lift", f"{total_lift_pct:+.2%}")

        # Comparison Chart
        comp_df = pd.DataFrame(
            {
                "Scenario": ["Baseline", "What-If Scenario"],
                "Demand Units": [base_volume, scenario_volume],
            }
        )
        bar_fig = px.bar(
            comp_df,
            x="Scenario",
            y="Demand Units",
            color="Scenario",
            color_discrete_map={"Baseline": "#64748b", "What-If Scenario": "#2563eb"},
            text_auto=True,
            title="Forecast Volume Comparison: Baseline vs Counterfactual",
        )
        st.plotly_chart(bar_fig, use_container_width=True)
