"""Data Explorer & Automated EDA View (P6-T02) with interactive chart customization and data editing."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from idfs.app.cache import get_cached_sales


def render_data_explorer() -> None:
    """Render the Data Explorer and EDA page with interactive chart controls and data editor."""
    st.header("Data Explorer & Exploratory Analysis")
    st.caption(
        "Explore historical demand trends, inspect dataset distributions, and customize visual analytics."
    )

    # 1. Data Source Selection
    with st.expander("📁 Data Source Selection & Custom Upload", expanded=False):
        data_source = st.radio(
            "Select Data Source:",
            ["Processed Retail Dataset (Default)", "Upload Custom CSV/Parquet"],
            horizontal=True,
            help="ℹ️ Choose between the validated pre-processed multi-echelon retail dataset or upload your own CSV/Parquet sales file.",
        )

        df: pd.DataFrame | None = None
        if data_source == "Processed Retail Dataset (Default)":
            df = get_cached_sales()
            if df.empty:
                st.warning("Default dataset not found. Please run pipeline first.")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV or Parquet file",
                type=["csv", "parquet"],
                help="ℹ️ File should contain at least: 'date', 'store_id', 'family', and 'sales' columns.",
            )
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".parquet"):
                    df = pd.read_parquet(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)

    if df is None or df.empty:
        df = get_cached_sales()

    if df is not None:
        df["date"] = pd.to_datetime(df["date"])

        # High-level KPIs with Info Tooltips
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Records",
                f"{len(df):,}",
                help="ℹ️ Total individual sales observations in the selected dataset across all stores and product families.",
            )
        with col2:
            st.metric(
                "Unique Stores",
                f"{df['store_id'].nunique() if 'store_id' in df else 'N/A'}",
                help="ℹ️ Count of distinct retail store locations tracked in this dataset.",
            )
        with col3:
            st.metric(
                "Product Families",
                f"{df['family'].nunique() if 'family' in df else 'N/A'}",
                help="ℹ️ Number of product categories (SKU families) modeled.",
            )
        with col4:
            st.metric(
                "Date Span",
                f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                help="ℹ️ Inclusive temporal range covered by the dataset.",
            )

        st.divider()

        # 2. Filter & Slicing Controls (Collapsible Expander)
        with st.expander("🔍 Filter & Slicing Controls (Click to open/close)", expanded=True):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                stores = (
                    sorted([str(s) for s in df["store_id"].unique()]) if "store_id" in df else []
                )
                selected_store = st.selectbox(
                    "Filter Store ID:",
                    ["All", *stores],
                    help="ℹ️ Isolate demand for an individual store branch or select 'All' for chain-wide aggregate volume.",
                )
            with fc2:
                families = sorted([str(f) for f in df["family"].unique()]) if "family" in df else []
                selected_family = st.selectbox(
                    "Filter Product Family:",
                    ["All", *families],
                    help="ℹ️ Filter by a specific SKU category (e.g. GROCERY, BEVERAGES) or view total retail portfolio.",
                )
            with fc3:
                min_d = df["date"].min().date()
                max_d = df["date"].max().date()
                date_range = st.date_input(
                    "Temporal Date Range:",
                    value=(min_d, max_d),
                    min_value=min_d,
                    max_value=max_d,
                    help="ℹ️ Zoom into a specific season, promotional campaign, or calendar quarter.",
                )

        # Apply Filters Safely (String matching prevents int parsing errors)
        filtered_df = df.copy()
        if selected_store != "All":
            filtered_df = filtered_df[filtered_df["store_id"].astype(str) == str(selected_store)]
        if selected_family != "All":
            filtered_df = filtered_df[filtered_df["family"].astype(str) == str(selected_family)]
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            filtered_df = filtered_df[
                (filtered_df["date"] >= start_date) & (filtered_df["date"] <= end_date)
            ]

        # 3. Chart Customization Toolbar (Collapsible Expander)
        with st.expander("🎨 Chart Customization & Analytics Options", expanded=False):
            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                chart_type = st.selectbox(
                    "Visualization Type:",
                    ["Line Chart", "Area Chart", "Bar Chart"],
                    help="ℹ️ Choose between continuous line, cumulative area ribbon, or discrete volume bars.",
                )
            with cc2:
                freq_agg = st.selectbox(
                    "Time Resampling:",
                    ["Daily", "Weekly Total", "Monthly Total"],
                    help="ℹ️ Resample daily demand into weekly (W-MON) or monthly (M) aggregates to smooth day-to-day noise.",
                )
            with cc3:
                overlay_sma = st.selectbox(
                    "Trendline Overlay:",
                    [
                        "None",
                        "7-Day Moving Average",
                        "14-Day Moving Average",
                        "28-Day Moving Average",
                    ],
                    help="ℹ️ Superimpose a Simple Moving Average (SMA) to uncover underlying macro trends.",
                )
            with cc4:
                color_theme = st.selectbox(
                    "Color Palette:",
                    [
                        "Royal Blue (#2563eb)",
                        "Emerald Green (#10b981)",
                        "Crimson Red (#e11d48)",
                        "Purple (#7c3aed)",
                    ],
                    help="ℹ️ Select the primary accent color for the chart.",
                )

        # Resample / Aggregate
        resample_rule = "1D" if freq_agg == "Daily" else ("1W" if "Weekly" in freq_agg else "1ME")
        daily_agg = (
            filtered_df.set_index("date")
            .resample(resample_rule)["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        hex_color = color_theme.split("(")[-1].replace(")", "").strip()

        # Build Interactive Plotly Figure
        fig = go.Figure()

        if chart_type == "Area Chart":
            fig.add_trace(
                go.Scatter(
                    x=daily_agg["date"],
                    y=daily_agg["sales"],
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=hex_color, width=2),
                    fillcolor=f"rgba({int(hex_color[1:3], 16)}, {int(hex_color[3:5], 16)}, {int(hex_color[5:7], 16)}, 0.2)",
                    name="Demand Volume",
                )
            )
        elif chart_type == "Bar Chart":
            fig.add_trace(
                go.Bar(
                    x=daily_agg["date"],
                    y=daily_agg["sales"],
                    marker_color=hex_color,
                    opacity=0.85,
                    name="Demand Volume",
                )
            )
        else:  # Line Chart
            fig.add_trace(
                go.Scatter(
                    x=daily_agg["date"],
                    y=daily_agg["sales"],
                    mode="lines+markers" if len(daily_agg) < 40 else "lines",
                    line=dict(color=hex_color, width=2),
                    name="Demand Volume",
                )
            )

        # Optional SMA Overlay
        if overlay_sma != "None":
            window = 7 if "7" in overlay_sma else (14 if "14" in overlay_sma else 28)
            daily_agg["sma"] = daily_agg["sales"].rolling(window=window, min_periods=1).mean()
            fig.add_trace(
                go.Scatter(
                    x=daily_agg["date"],
                    y=daily_agg["sma"],
                    mode="lines",
                    line=dict(color="#f59e0b", width=2.5, dash="dash"),
                    name=overlay_sma,
                )
            )

        fig.update_layout(
            title=f"Demand Volume Over Time ({freq_agg} Resampling - Store: {selected_store}, Family: {selected_family})",
            xaxis_title="Date",
            yaxis_title="Units Sold",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Tabbed Data Profiling, Heatmap & Interactive Data Editor
        tab_editor, tab_heatmap, tab_movers, tab_stats = st.tabs(
            [
                "✏️ Interactive Data Table & Editor",
                "🔥 Store × Family Portfolio Heatmap",
                "🏆 Top & Bottom Demand Movers",
                "📊 Statistical Distribution Summary",
            ]
        )

        with tab_editor:
            st.caption(
                "You can search, sort, and edit cells directly in the table below. Modified data can be exported."
            )
            edited_df = st.data_editor(
                filtered_df.head(200),
                use_container_width=True,
                num_rows="dynamic",
                key="interactive_data_editor",
            )
            csv_data = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Current Sliced / Edited Data to CSV",
                data=csv_data,
                file_name=f"idfs_filtered_sales_{selected_store}_{selected_family}.csv",
                mime="text/csv",
                help="ℹ️ Download the currently filtered and edited dataset to your local machine.",
            )

        with tab_heatmap:
            st.caption(
                "Cross-echelon mean daily demand matrix across all 15 Stores and 20 Product Families."
            )
            if "store_id" in df.columns and "family" in df.columns:
                pivot = (
                    df.pivot_table(
                        index="store_id",
                        columns="family",
                        values="sales",
                        aggfunc="mean",
                        observed=False,
                    )
                    .round(1)
                    .fillna(0)
                )
                heat_fig = go.Figure(
                    data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns.tolist(),
                        y=pivot.index.tolist(),
                        colorscale="Blues",
                        colorbar=dict(title="Daily Units"),
                        hoverongaps=False,
                    )
                )
                heat_fig.update_layout(
                    title="Portfolio Demand Heatmap (15 Stores × 20 Families)",
                    xaxis_title="Product Family",
                    yaxis_title="Store ID",
                    template="plotly_white",
                    height=450,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(heat_fig, use_container_width=True)

        with tab_movers:
            st.caption(
                "Portfolio ranking identifying high-velocity demand drivers and intermittent slow movers."
            )
            if "family" in df.columns and "sales" in df.columns:
                fam_sales = (
                    df.groupby("family", observed=False)["sales"]
                    .sum()
                    .reset_index()
                    .sort_values("sales", ascending=False)
                )
                col_top, col_bot = st.columns(2)
                with col_top:
                    st.markdown("##### 🚀 Top 5 Highest-Volume Families")
                    top_fig = go.Figure(
                        go.Bar(
                            x=fam_sales.head(5)["sales"],
                            y=fam_sales.head(5)["family"],
                            orientation="h",
                            marker=dict(color="#10b981"),
                        )
                    )
                    top_fig.update_layout(
                        template="plotly_white",
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(top_fig, use_container_width=True)
                with col_bot:
                    st.markdown("##### 🐢 Bottom 5 Intermittent / Niche Families")
                    bot_fig = go.Figure(
                        go.Bar(
                            x=fam_sales.tail(5)["sales"],
                            y=fam_sales.tail(5)["family"],
                            orientation="h",
                            marker=dict(color="#f59e0b"),
                        )
                    )
                    bot_fig.update_layout(
                        template="plotly_white",
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(bot_fig, use_container_width=True)

        with tab_stats:
            st.caption(
                "Parametric descriptive statistics for all numerical features in the filtered selection."
            )
            numeric_desc = filtered_df.select_dtypes(include="number").describe().round(2)
            st.dataframe(numeric_desc, use_container_width=True)
