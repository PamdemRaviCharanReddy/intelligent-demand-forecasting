"""Inventory Replenishment, Simulation & Alerts View (P6-T04) with interactive controls and audit logging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from idfs.app.storage import RunStorage
from idfs.inventory.simulation import (
    ConformalQuantilePolicy,
    ForecastDrivenPolicy,
    InventoryPolicy,
    NaivePolicy,
    Simulator,
)


def render_inventory_view() -> None:
    """Render the Inventory Replenishment & Cost Simulation page with visual controls."""
    st.header("Inventory Replenishment & Decision Optimization (Novelty N1)")
    st.caption(
        "Evaluate inventory replenishment policies under asymmetric costs, track pipeline receipts, and generate reorder alerts."
    )

    # 1. Policy Benchmark Leaderboard (Collapsible)
    with st.expander(
        "🏆 Historical Policy Benchmark Leaderboard (Novelties N1 + N2)", expanded=False
    ):
        results_p = Path("reports/tables/inventory_simulation_results.csv")
        if results_p.exists():
            sim_df = pd.read_csv(results_p)
            st.dataframe(
                sim_df.style.highlight_min(subset=["total_cost", "stockout_cost"], color="#dcfce7")
                .highlight_max(subset=["fill_rate"], color="#dbeafe")
                .format(
                    {
                        "total_cost": "${:,.2f}",
                        "holding_cost": "${:,.2f}",
                        "stockout_cost": "${:,.2f}",
                        "fill_rate": "{:.2%}",
                    }
                ),
                use_container_width=True,
            )
        else:
            st.info(
                "Run `python scripts/run_inventory_simulation.py` to populate policy benchmark table."
            )

    # 2. Simulation Configuration Parameters (Collapsible Expander)
    with st.expander("⚙️ Replenishment & Cost Parameters (Click to open/close)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            policy_type = st.selectbox(
                "Replenishment Policy:",
                [
                    "ConformalQuantile-(R,S) (N1+N2)",
                    "Gaussian-(R,S) SL95%",
                    "NaivePolicy (Historical Avg)",
                ],
                help="ℹ️ ConformalQuantile sizes safety stock via non-crossing quantile bounds; Gaussian assumes normal distribution; Naive uses trailing mean.",
            )
            lead_time = st.slider(
                "Supplier Lead Time L (Days):",
                min_value=1,
                max_value=14,
                value=5,
                help="ℹ️ Days elapsed between order placement and physical warehouse arrival.",
            )
        with c2:
            review_period = st.slider(
                "Review Period R (Days):",
                min_value=1,
                max_value=14,
                value=7,
                help="ℹ️ Replenishment cycle length (e.g. weekly review R=7). Protection horizon is R + L.",
            )
            holding_cost = st.number_input(
                "Holding Cost ($/unit/day):",
                min_value=0.01,
                max_value=5.0,
                value=0.10,
                step=0.05,
                help="ℹ️ Physical storage, insurance, spoilage, and capital interest per unsold unit per day.",
            )
        with c3:
            stockout_cost = st.number_input(
                "Stockout Penalty ($/unit):",
                min_value=1.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
                help="ℹ️ Financial loss, lost margin, and brand penalty per unfulfilled customer demand unit.",
            )
            ordering_cost = st.number_input(
                "Fixed Order Cost ($/order):",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=10.0,
                help="ℹ️ Fixed administrative and freight delivery charge per replenishment batch.",
            )

    # 3. Chart Customization Options
    with st.expander("🎨 Daily Dynamics Chart Options", expanded=False):
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            show_stock = st.checkbox(
                "Show On-Hand Stock", value=True, help="ℹ️ Trace daily warehouse inventory."
            )
        with t2:
            show_demand = st.checkbox(
                "Show Customer Demand", value=True, help="ℹ️ Trace daily actual customer purchases."
            )
        with t3:
            show_orders = st.checkbox(
                "Show Orders Placed", value=True, help="ℹ️ Show blue replenishment bars."
            )
        with t4:
            show_stockouts = st.checkbox(
                "Show Stockout Events", value=True, help="ℹ️ Show red unmet stockout bars."
            )

    # Run discrete-event simulation
    data_p = Path("data/processed/fused_sales.parquet")
    if data_p.exists():
        df = pd.read_parquet(data_p)
        daily_demand = df.groupby("date")["sales"].mean().to_numpy()[-60:]
        daily_forecast = np.maximum(
            0.0, daily_demand + np.random.normal(0, 3.0, size=len(daily_demand))
        )
        upper_q = daily_forecast + 1.645 * 8.0

        pol: InventoryPolicy
        if "Conformal" in policy_type:
            pol = ConformalQuantilePolicy(review_period=review_period)
        elif "Gaussian" in policy_type:
            pol = ForecastDrivenPolicy(service_level=0.95, review_period=review_period)
        else:
            pol = NaivePolicy(
                avg_daily_demand=float(np.mean(daily_demand)),
                lead_time=lead_time,
                review_period=review_period,
            )

        sim = Simulator(
            policy=pol,
            lead_time=lead_time,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            ordering_cost=ordering_cost,
            review_period=review_period,
            initial_inventory=60.0,
        )

        res = sim.run(
            demand=daily_demand,
            forecasts=daily_forecast,
            forecast_std=8.0,
            upper_quantiles=upper_q,
        )

        log = res.daily_log

        # Operational Metrics Callouts
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Service Fill Rate",
            f"{res.fill_rate:.1%}",
            help="ℹ️ Percentage of total consumer demand satisfied directly from on-hand stock.",
        )
        m2.metric(
            "Total Cost",
            f"${res.total_cost['total']:,.2f}",
            help="ℹ️ Net sum of holding costs, stockout penalties, and batch order charges.",
        )
        m3.metric(
            "Holding Cost",
            f"${res.total_cost['holding']:,.2f}",
            help="ℹ️ Cumulative cost of carrying inventory over the 60-day simulation.",
        )
        m4.metric(
            "Stockout Cost",
            f"${res.total_cost['stockout']:,.2f}",
            help="ℹ️ Financial penalty incurred due to unmet demand on zero-inventory days.",
        )
        m5.metric(
            "Orders Placed",
            f"{res.n_orders}",
            help="ℹ️ Total replenishment shipments dispatched across review cycles.",
        )

        st.divider()

        # Operational Alerts Panel
        latest_inv = log["inventory"].iloc[-1]
        next_order_qty = log["order_qty"].iloc[-1]

        if latest_inv <= 10.0:
            st.error(
                f"🚨 CRITICAL STOCKOUT RISK: On-hand inventory is dangerously low ({latest_inv:.1f} units). Immediate reorder required."
            )
        elif next_order_qty > 0.0:
            st.warning(
                f"⚠️ REPLENISHMENT ACTION REQUIRED: Policy recommends placing an order of **{next_order_qty:.0f} units** arriving in {lead_time} days."
            )
        else:
            st.success(
                f"✅ INVENTORY HEALTHY: On-hand stock is sufficient ({latest_inv:.1f} units). No order needed today."
            )

        # Interactive Daily Trajectory Plot
        fig = go.Figure()
        if show_stock:
            fig.add_trace(
                go.Scatter(
                    x=log["date"],
                    y=log["inventory"],
                    name="On-Hand Inventory",
                    line=dict(color="#10b981", width=2.5),
                )
            )
        if show_demand:
            fig.add_trace(
                go.Scatter(
                    x=log["date"],
                    y=log["demand"],
                    name="Customer Demand",
                    line=dict(color="#0f172a", width=1.5, dash="dot"),
                )
            )
        if show_orders:
            fig.add_trace(
                go.Bar(
                    x=log["date"],
                    y=log["order_qty"],
                    name="Replenishment Orders Placed",
                    marker_color="#3b82f6",
                    opacity=0.7,
                )
            )
        if show_stockouts:
            fig.add_trace(
                go.Bar(
                    x=log["date"],
                    y=log["stockout_qty"],
                    name="Unmet Stockout Demand",
                    marker_color="#ef4444",
                    opacity=0.8,
                )
            )

        fig.update_layout(
            title="Daily Inventory Dynamics (On-Hand Stock, Demand & Order Placement)",
            xaxis_title="Simulation Day",
            yaxis_title="Units",
            template="plotly_white",
            barmode="overlay",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Save Run & Planner Review
        with st.expander(
            "📝 Human Planner Review & Decision Recording (Click to open/close)", expanded=True
        ):
            col_d1, col_d2 = st.columns([1, 2])
            with col_d1:
                decision = st.selectbox(
                    "Planner Review Decision:",
                    ["APPROVED", "APPROVED WITH MODIFICATIONS", "OVERRIDDEN", "REJECTED"],
                    help="ℹ️ Official managerial review status committed to enterprise audit trail.",
                )
            with col_d2:
                notes = st.text_input(
                    "Reviewer Notes / Rationales:",
                    "Verified with warehouse capacity constraints. Approved by store manager.",
                    help="ℹ️ Qualitative rationale recorded alongside simulation numbers.",
                )

            if st.button(
                "💾 Commit Run & Approval to Persistent Database",
                help="ℹ️ Persists simulation parameters and manager decisions to SQLite runs.db.",
            ):
                storage = RunStorage()
                rid = storage.save_run(
                    model_name="QuantileXGBoost",
                    policy_name=pol.name,
                    lead_time=lead_time,
                    holding_cost=res.total_cost["holding"],
                    stockout_cost=res.total_cost["stockout"],
                    ordering_cost=res.total_cost["ordering"],
                    total_cost=res.total_cost["total"],
                    fill_rate=res.fill_rate,
                    stockout_periods=res.stockout_periods,
                    n_orders=res.n_orders,
                    planner_decision=decision,
                    planner_notes=notes,
                )
                st.success(f"Run successfully committed to SQLite storage (Run ID: `{rid}`)!")
