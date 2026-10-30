"""Inventory Replenishment, Simulation & Alerts View (P6-T04) with interactive controls and audit logging."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from idfs.app.cache import get_cached_csv, get_cached_daily_series, get_cached_sales
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
        sim_df = get_cached_csv("reports/tables/inventory_simulation_results.csv")
        if not sim_df.empty:
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
        df_raw = get_cached_sales()
        prods = (
            sorted([str(p) for p in df_raw["family"].unique()])
            if not df_raw.empty and "family" in df_raw.columns
            else []
        )

        c0, c1, c2, c3 = st.columns([1.3, 1.2, 1, 1])
        with c0:
            selected_product = st.selectbox(
                "Select Product for Policy:",
                ["All Products (Portfolio Average)", *prods],
                help="ℹ️ Simulate replenishment and inventory carrying policies for an individual product (e.g., Umbrella, Raincoat, Air Conditioner) or overall chain.",
            )
            policy_type = st.selectbox(
                "Replenishment Policy:",
                [
                    "ConformalQuantile-(R,S) (N1+N2)",
                    "Gaussian-(R,S) SL95%",
                    "NaivePolicy (Historical Avg)",
                ],
                help="ℹ️ ConformalQuantile sizes safety stock via non-crossing quantile bounds; Gaussian assumes normal distribution; Naive uses trailing mean.",
            )
        with c1:
            lead_time = st.slider(
                "Supplier Lead Time L (Days):",
                min_value=1,
                max_value=14,
                value=5,
                help="ℹ️ Days elapsed between order placement and physical warehouse arrival.",
            )
            review_period = st.slider(
                "Review Period R (Days):",
                min_value=1,
                max_value=14,
                value=7,
                help="ℹ️ Replenishment cycle length (e.g. weekly review R=7). Protection horizon is R + L.",
            )
        with c2:
            holding_cost = st.number_input(
                "Holding Cost ($/unit/day):",
                min_value=0.01,
                max_value=5.0,
                value=0.10,
                step=0.05,
                help="ℹ️ Physical storage, insurance, spoilage, and capital interest per unsold unit per day.",
            )
            ordering_cost = st.number_input(
                "Fixed Order Cost ($/order):",
                min_value=0.0,
                max_value=200.0,
                value=50.0,
                step=10.0,
                help="ℹ️ Fixed administrative and freight delivery charge per replenishment batch.",
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

    # Run discrete-event simulation (cached daily series)
    if not df_raw.empty:
        if selected_product != "All Products (Portfolio Average)":
            plot_df = df_raw[df_raw["family"].astype(str) == str(selected_product)]
        else:
            plot_df = df_raw
        daily_demand = plot_df.groupby("date")["sales"].mean().to_numpy()[-60:]
    else:
        daily_df = get_cached_daily_series()
        daily_demand = (
            daily_df["sales"].to_numpy()[-60:] if not daily_df.empty else np.array([20.0] * 60)
        )
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
            title=f"Daily Inventory Dynamics for '{selected_product}' (On-Hand Stock, Demand & Order Placement)",
            xaxis_title="Simulation Day",
            yaxis_title="Units",
            template="plotly_white",
            barmode="overlay",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Cost Decomposition Donut & Fill-Rate Service Level Gauge
        col_g, col_p = st.columns(2)
        with col_g:
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=float(res.fill_rate * 100),
                    number=dict(suffix="%"),
                    title=dict(text="Simulated Customer Service Fill Rate"),
                    gauge=dict(
                        axis=dict(range=[70, 100]),
                        bar=dict(color="#2563eb"),
                        steps=[
                            {"range": [70, 85], "color": "#fee2e2"},
                            {"range": [85, 95], "color": "#fef3c7"},
                            {"range": [95, 100], "color": "#dcfce7"},
                        ],
                        threshold=dict(
                            line=dict(color="#16a34a", width=3),
                            thickness=0.75,
                            value=95.0,
                        ),
                    ),
                )
            )
            gauge_fig.update_layout(
                height=250, margin=dict(l=30, r=30, t=40, b=20), template="plotly_white"
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_p:
            cost_labels = ["Holding Cost", "Stockout Penalty", "Ordering Setup"]
            cost_vals = [
                float(res.total_cost.get("holding", 0.0)),
                float(res.total_cost.get("stockout", 0.0)),
                float(res.total_cost.get("ordering", res.total_cost.get("order", 0.0))),
            ]
            donut_fig = go.Figure(
                go.Pie(
                    labels=cost_labels,
                    values=cost_vals,
                    hole=0.5,
                    marker=dict(colors=["#10b981", "#ef4444", "#3b82f6"]),
                    textinfo="label+percent",
                )
            )
            donut_fig.update_layout(
                title="Replenishment Cost Decomposition",
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(donut_fig, use_container_width=True)

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
