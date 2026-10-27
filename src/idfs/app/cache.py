"""High-speed in-memory caching layer for the IDFS Streamlit portal and API.

Provides sub-millisecond data retrieval via Streamlit's @st.cache_data decorators,
preventing repetitive disk reads and expensive Pandas aggregations.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_sales(file_path: str = "data/processed/fused_sales.parquet") -> pd.DataFrame:
    """Load and cache the fused sales dataset in RAM."""
    p = Path(file_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_csv(file_path: str) -> pd.DataFrame:
    """Load and cache report tables in RAM."""
    p = Path(file_path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_daily_series(file_path: str = "data/processed/fused_sales.parquet") -> pd.DataFrame:
    """Pre-compute and cache daily aggregate demand for instantaneous plotting."""
    df = get_cached_sales(file_path)
    if df.empty or "date" not in df.columns or "sales" not in df.columns:
        return pd.DataFrame()
    return df.groupby("date")["sales"].mean().reset_index().sort_values("date")
