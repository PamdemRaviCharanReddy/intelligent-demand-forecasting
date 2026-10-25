"""Exploratory Data Analysis (EDA) engine for IDFS.

Generates descriptive statistics, demand pattern characterizations,
and publication-ready visual diagnostic figures into reports/figures/eda/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def compute_demand_pattern_stats(
    df: pd.DataFrame,
    series_id_cols: list[str] | None = None,
    target_col: str = "sales",
) -> pd.DataFrame:
    """Compute Average Demand Interval (ADI) and squared Coefficient of Variation (CV^2).

    Implements the Syntetos-Boylan (2005) demand classification:
    - Smooth: ADI < 1.32, CV^2 < 0.49
    - Erratic: ADI < 1.32, CV^2 >= 0.49
    - Intermittent: ADI >= 1.32, CV^2 < 0.49
    - Lumpy: ADI >= 1.32, CV^2 >= 0.49

    Args:
        df: Processed sales DataFrame.
        series_id_cols: Identifier columns for series.
        target_col: Sales volume column.

    Returns:
        DataFrame with ADI, CV2, and demand_category per series.
    """
    id_cols = series_id_cols or ["store_id", "family"]
    active_cols = [c for c in id_cols if c in df.columns]

    records = []
    for key, group in df.groupby(active_cols):
        k = key if isinstance(key, tuple) else (key,)
        series_sales = group[target_col].to_numpy(dtype=np.float64)
        n = len(series_sales)

        non_zero_idx = np.where(series_sales > 0)[0]
        n_non_zero = len(non_zero_idx)

        if n_non_zero < 2:
            adi = float(n) if n_non_zero == 1 else 999.0
            cv2 = 999.0
            category = "Lumpy"
        else:
            # Average Demand Interval
            intervals = np.diff(non_zero_idx)
            adi = float(np.mean(intervals))

            # Non-zero demand CV^2
            non_zero_demand = series_sales[non_zero_idx]
            mean_demand = np.mean(non_zero_demand)
            std_demand = np.std(non_zero_demand)
            cv2 = float((std_demand / mean_demand) ** 2) if mean_demand > 0 else 999.0

            # Classification
            if adi < 1.32 and cv2 < 0.49:
                category = "Smooth"
            elif adi < 1.32 and cv2 >= 0.49:
                category = "Erratic"
            elif adi >= 1.32 and cv2 < 0.49:
                category = "Intermittent"
            else:
                category = "Lumpy"

        rec = {
            "series_id": "_".join(str(x) for x in k),
            "adi": round(adi, 3),
            "cv2": round(cv2, 3),
            "category": category,
            "mean_sales": round(float(np.mean(series_sales)), 2),
            "zero_ratio": round(float(np.mean(series_sales == 0)), 3),
        }
        for i, col in enumerate(active_cols):
            rec[col] = k[i]
        records.append(rec)

    return pd.DataFrame(records)


def generate_eda_figures(
    df: pd.DataFrame,
    output_dir: str | Path = "reports/figures/eda",
    tables_dir: str | Path = "reports/tables",
) -> dict[str, str]:
    """Generate all EDA figures and summary tables.

    Produces:
    - sales_trend.png: Total chain daily demand trend
    - day_of_week_seasonality.png: Average sales by weekday
    - demand_pattern_scatter.png: Syntetos-Boylan ADI vs CV^2 quadrants
    - promotional_uplift.png: Distribution comparison with/without promo
    - eda_summary.csv: Summary statistics per store/family

    Args:
        df: Fused processed dataset.
        output_dir: Destination folder for figures.
        tables_dir: Destination folder for tables.

    Returns:
        Dict mapping figure name to output file path.
    """
    fig_dir = Path(output_dir)
    tbl_dir = Path(tables_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, str] = {}
    sns.set_theme(style="whitegrid", palette="muted")

    # ── 1. Total Chain Daily Demand Trend ──
    plt.figure(figsize=(12, 4))
    daily_agg = df.groupby("date")["sales"].sum().reset_index()
    plt.plot(daily_agg["date"], daily_agg["sales"], color="#2b5c8f", lw=1.5)
    plt.title("Total Aggregate Daily Demand Over Time", fontsize=13, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Total Units Sold", fontsize=11)
    plt.tight_layout()
    p1 = fig_dir / "sales_trend.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    generated["sales_trend"] = str(p1)

    # ── 2. Day-of-Week Seasonality ──
    plt.figure(figsize=(8, 4))
    df_copy = df.copy()
    df_copy["dow"] = df_copy["date"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_agg = df_copy.groupby("dow")["sales"].mean().reindex(dow_order).reset_index()
    sns.barplot(data=dow_agg, x="dow", y="sales", color="#3b82f6")
    plt.title("Average Sales by Day of Week (Weekly Seasonality)", fontsize=13, fontweight="bold")
    plt.xlabel("Day of Week", fontsize=11)
    plt.ylabel("Mean Daily Sales per Series", fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()
    p2 = fig_dir / "day_of_week_seasonality.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    generated["day_of_week_seasonality"] = str(p2)

    # ── 3. Syntetos-Boylan ADI vs CV^2 Scatter Plot ──
    pattern_df = compute_demand_pattern_stats(df)
    plt.figure(figsize=(9, 6))
    category_colors = {
        "Smooth": "#10b981",
        "Erratic": "#f59e0b",
        "Intermittent": "#3b82f6",
        "Lumpy": "#ef4444",
    }
    for cat, color in category_colors.items():
        sub = pattern_df[pattern_df["category"] == cat]
        plt.scatter(
            sub["adi"],
            sub["cv2"],
            label=f"{cat} ({len(sub)})",
            color=color,
            alpha=0.75,
            edgecolors="none",
            s=40,
        )

    plt.axvline(1.32, color="gray", linestyle="--", lw=1.2, label="ADI Cutoff (1.32)")
    plt.axhline(0.49, color="gray", linestyle=":", lw=1.2, label="CV² Cutoff (0.49)")
    plt.xlim(0.8, min(float(pattern_df["adi"].max() + 0.5), 10.0))
    plt.ylim(0, min(float(pattern_df["cv2"].max() + 0.5), 10.0))
    plt.title(
        "Syntetos-Boylan Demand Pattern Categorization (N3 Router)", fontsize=13, fontweight="bold"
    )
    plt.xlabel("Average Demand Interval (ADI)", fontsize=11)
    plt.ylabel("Demand Squared Coefficient of Variation (CV²)", fontsize=11)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    p3 = fig_dir / "demand_pattern_scatter.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    generated["demand_pattern_scatter"] = str(p3)

    # ── 4. Promotional Uplift Comparison ──
    if "onpromotion" in df.columns and df["onpromotion"].nunique() > 1:
        plt.figure(figsize=(7, 4))
        promo_summary = df.groupby("onpromotion")["sales"].mean().reset_index()
        promo_summary["status"] = promo_summary["onpromotion"].map(
            {0: "Regular Day", 1: "Promotional Day"}
        )
        sns.barplot(data=promo_summary, x="status", y="sales", palette=["#94a3b8", "#f43f5e"])
        plt.title(
            "Mean Daily Demand: Regular vs. Promotional Uplift", fontsize=13, fontweight="bold"
        )
        plt.xlabel("Promotion Status", fontsize=11)
        plt.ylabel("Average Units Sold", fontsize=11)
        plt.tight_layout()
        p4 = fig_dir / "promotional_uplift.png"
        plt.savefig(p4, dpi=200)
        plt.close()
        generated["promotional_uplift"] = str(p4)

    # ── 5. Export Summary Statistics Table ──
    summary_path = tbl_dir / "eda_summary.csv"
    pattern_df.to_csv(summary_path, index=False)
    generated["eda_summary_table"] = str(summary_path)

    logger.info("Generated %d EDA figures and summary table in %s", len(generated), fig_dir)
    return generated
