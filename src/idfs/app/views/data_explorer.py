"""Data Explorer & Automated EDA View (P6-T02)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def render_data_explorer() -> None:
    """Render the Data Explorer and EDA page."""
    st.header("Data Explorer & Exploratory Analysis")
    st.markdown(
        "Inspect, validate, and explore historical sales data, exogenous variables (oil, holidays), and category distributions."
    )

    data_source = st.radio(
        "Select Data Source:",
        ["Processed Retail Dataset (Default)", "Upload Custom CSV/Parquet"],
        horizontal=True,
    )

    df: pd.DataFrame | None = None

    if data_source == "Processed Retail Dataset (Default)":
        data_p = Path("data/processed/fused_sales.parquet")
        if data_p.exists():
            df = pd.read_parquet(data_p)
        else:
            st.warning(f"Default dataset not found at `{data_p}`. Please run pipeline first.")
    else:
        uploaded_file = st.file_uploader("Upload CSV or Parquet file", type=["csv", "parquet"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

    if df is not None:
        df["date"] = pd.to_datetime(df["date"])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Unique Stores", f"{df['store_id'].nunique() if 'store_id' in df else 'N/A'}")
        with col3:
            st.metric("Product Families", f"{df['family'].nunique() if 'family' in df else 'N/A'}")
        with col4:
            st.metric(
                "Date Span",
                f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
            )

        st.subheader("Interactive Demand Trends")
        c1, c2 = st.columns(2)
        with c1:
            stores = sorted(df["store_id"].unique()) if "store_id" in df else []
            selected_store = st.selectbox("Filter Store ID", ["All", *[str(s) for s in stores]])
        with c2:
            families = sorted(df["family"].unique()) if "family" in df else []
            selected_family = st.selectbox("Filter Product Family", ["All", *list(families)])

        filtered_df = df.copy()
        if selected_store != "All":
            filtered_df = filtered_df[filtered_df["store_id"] == int(selected_store)]
        if selected_family != "All":
            filtered_df = filtered_df[filtered_df["family"] == selected_family]

        daily_agg = filtered_df.groupby("date")["sales"].sum().reset_index()

        fig = px.line(
            daily_agg,
            x="date",
            y="sales",
            title="Daily Sales Volume Over Time",
            template="plotly_white",
            labels={"sales": "Sales Units", "date": "Date"},
        )
        fig.update_traces(line_color="#2563eb", line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Data Summary & Distribution")
        numeric_desc = filtered_df.select_dtypes(include="number").describe().round(2)
        st.dataframe(numeric_desc, use_container_width=True)
