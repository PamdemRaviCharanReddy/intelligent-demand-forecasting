"""Execute Phase 5: Concept Drift Monitoring & Tree-SHAP Explainability (Novelties N4 & N5).

Outputs:
- reports/tables/drift_monitoring_log.csv
- reports/tables/shap_feature_importance.csv
- reports/figures/drift_detection_demo.png
- reports/figures/shap_summary.png
- reports/figures/shap_waterfall.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from idfs.config import load_config
from idfs.explain.shap_explainer import TreeSHAPExplainer
from idfs.explain.what_if import WhatIfSimulator
from idfs.models.xgboost import XGBoostForecaster
from idfs.monitoring.drift import DriftMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDFS Phase 5 Monitoring & Explainability")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/fused_sales.parquet",
        help="Path to fused processed dataset",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data_path)
    if not data_path.exists():
        from idfs.data.fusion import run_data_pipeline

        df = run_data_pipeline("data/synthetic", data_path)
    else:
        df = pd.read_parquet(data_path)

    df["date"] = pd.to_datetime(df["date"])
    unique_dates = np.sort(df["date"].unique())

    # Split into historical train and streaming test
    split_date = unique_dates[-90]  # Last 90 days for streaming surveillance
    train_df = df[df["date"] <= split_date].copy()
    stream_df = df[df["date"] > split_date].copy()

    print(f"Training XGBoost on {len(train_df)} historical records...")
    model = XGBoostForecaster(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        seed=42,
    )
    model.fit(train_df, series_id_cols=list(cfg.data.series_id_cols))

    # Evaluate baseline validation error on last 28 days of train
    val_sub = train_df[train_df["date"] > unique_dates[-118]].copy()
    val_preds = model.predict(horizon=28, features_df=val_sub)
    merged_val = pd.merge(val_sub, val_preds, on=["date", *cfg.data.series_id_cols])
    baseline_mae = float(np.mean(np.abs(merged_val["sales"] - merged_val["forecast"])))
    print(f"Calculated Baseline Validation MAE: {baseline_mae:.2f}")

    # =========================================================================
    # PART 1: Concept Drift Monitoring Simulation (Novelty N4)
    # =========================================================================
    print("\n--- Part 1: Rolling-Error Concept Drift Monitoring (Novelty N4) ---")
    # Stream test data day-by-day. In the last 30 days, inject a +50% demand shock
    shock_date = unique_dates[-30]
    stream_copy = stream_df.copy()
    shock_mask = stream_copy["date"] >= shock_date
    stream_copy.loc[shock_mask, "sales"] *= 1.50  # 50% structural shock

    # Generate model predictions for stream
    preds = model.predict(horizon=len(unique_dates) - len(train_df["date"].unique()), features_df=stream_copy)
    merged_stream = pd.merge(stream_copy, preds, on=["date", *cfg.data.series_id_cols]).sort_values("date")

    # Group daily aggregates across stores for clean streaming surveillance
    daily_stream = (
        merged_stream.groupby("date")
        .agg({"sales": "mean", "forecast": "mean"})
        .reset_index()
    )

    monitor = DriftMonitor(
        window_size=14,
        threshold_multiplier=1.40,
        critical_multiplier=1.80,
        metric="mae",
    )
    monitor.set_baseline(baseline_mae)

    print("Streaming daily observations into DriftMonitor...")
    alerts = monitor.batch_update(
        actuals=daily_stream["sales"].values,
        predictions=daily_stream["forecast"].values,
        dates=daily_stream["date"].values,
    )

    history_df = monitor.get_monitoring_history()
    log_path = Path("reports/tables/drift_monitoring_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(log_path, index=False)
    print(f"Drift monitoring log saved ({len(alerts)} alerts triggered): {log_path}")

    # Plot Concept Drift Detection Demo Figure
    fig_path = Path("reports/figures/drift_detection_demo.png")
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(daily_stream["date"], daily_stream["sales"], label="Actual Demand (Shock Injected)", color="#0f172a", lw=2)
    plt.plot(daily_stream["date"], daily_stream["forecast"], label="Static Model Forecast", color="#2563eb", lw=1.5, ls="--")
    plt.axvline(shock_date, color="#ef4444", lw=2, ls=":", label="Injected Demand Shock (+50%)")

    # Mark alert trigger points
    alert_dates = [a.timestamp for a in alerts]
    alert_y = [daily_stream.loc[daily_stream["date"] == d, "sales"].values[0] for d in alert_dates if d in daily_stream["date"].values]
    if alert_dates:
        plt.scatter(alert_dates[:len(alert_y)], alert_y, color="#dc2626", s=80, zorder=5, label="Drift Retrain Alert")

    plt.title("Production Surveillance: Automated Concept Drift Detection (Novelty N4)", fontsize=13, pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Average Sales Volume", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved drift detection diagnostic plot: {fig_path}")

    # =========================================================================
    # PART 2: Tree-SHAP Explainability (Novelty N5)
    # =========================================================================
    print("\n--- Part 2: Tree-SHAP Explainability (Novelty N5) ---")
    explainer = TreeSHAPExplainer(model)
    explanation, sample_X = explainer.explain(stream_df, max_samples=400)

    # 1. Feature Importance Table
    shap_table = explainer.get_feature_importance_table(explanation)
    shap_table_path = Path("reports/tables/shap_feature_importance.csv")
    shap_table.to_csv(shap_table_path, index=False)
    print(f"Saved SHAP feature importance table to: {shap_table_path}")
    print(shap_table.head(8).to_string(index=False))

    # 2. Global Summary Beeswarm Plot
    shap_fig_path = Path("reports/figures/shap_summary.png")
    explainer.plot_summary(explanation, save_path=shap_fig_path, max_display=12)
    print(f"Saved SHAP summary beeswarm plot: {shap_fig_path}")

    # 3. Local Waterfall Plot on High Demand Row
    waterfall_path = Path("reports/figures/shap_waterfall.png")
    # Find row with highest predicted sales in sample
    top_row_idx = int(np.argmax(explanation.values.sum(axis=1)))
    explainer.plot_waterfall(explanation, row_idx=top_row_idx, save_path=waterfall_path, max_display=10)
    print(f"Saved local SHAP waterfall plot: {waterfall_path}")

    # =========================================================================
    # PART 3: What-If Scenario Simulator (Novelty N5)
    # =========================================================================
    print("\n--- Part 3: What-If Scenario Simulation (Novelty N5) ---")
    what_if = WhatIfSimulator(model)
    promo_result = what_if.simulate_promotion_scenario(
        features_df=stream_df.head(700),  # 14-day window across 50 series
        horizon=14,
        target_promo_status=1,
        scenario_name="Full Promotion Rollout",
    )
    print(f"Scenario: {promo_result.scenario_name}")
    print(f"  Baseline Total Demand: {promo_result.baseline_total_demand:,.0f} units")
    print(f"  Scenario Total Demand: {promo_result.scenario_total_demand:,.0f} units")
    print(f"  Projected Lift: {promo_result.absolute_delta:+,.0f} units ({promo_result.percentage_change:+.2f}%)")


if __name__ == "__main__":
    main()
