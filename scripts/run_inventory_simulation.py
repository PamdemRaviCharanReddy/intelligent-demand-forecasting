"""Execute Phase 4 Inventory Simulation and Policy Benchmark (Novelties N1 & N2).

Compares:
1. Naive Policy (Historical average)
2. Point Forecast Policy (Deterministic order-up-to, zero safety stock)
3. Gaussian (R, S) Policy (Parametric Gaussian safety stock)
4. Conformal Quantile Policy (Novelty N1 + N2: CQR calibrated upper bound)

Saves results to:
- reports/tables/inventory_simulation_results.csv
- reports/tables/inventory_sensitivity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from idfs.config import load_config
from idfs.inventory.simulation import (
    ConformalQuantilePolicy,
    ForecastDrivenPolicy,
    NaivePolicy,
    Simulator,
    simulate_portfolio,
)
from idfs.models.base import NaiveForecaster
from idfs.uncertainty.conformal import ConformalCalibrator
from idfs.uncertainty.quantile import QuantileXGBoostForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDFS Phase 4 Inventory Simulation")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/fused_sales.parquet",
        help="Path to fused processed dataset",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default="reports/tables/inventory_simulation_results.csv",
        help="Path to save simulation comparison table",
    )
    parser.add_argument(
        "--sensitivity-table",
        type=str,
        default="reports/tables/inventory_sensitivity.csv",
        help="Path to save sensitivity grid",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Data not found at {data_path}. Running fusion pipeline...")
        from idfs.data.fusion import run_data_pipeline

        df = run_data_pipeline("data/synthetic", data_path)
    else:
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)

    df["date"] = pd.to_datetime(df["date"])
    unique_dates = np.sort(df["date"].unique())
    n_days = len(unique_dates)

    # 3-way temporal split: Train (up to T-42), Calibration (T-42 to T-14), Eval (T-14 to T)
    test_days = 14
    cal_days = 28
    train_end = unique_dates[-(test_days + cal_days)]
    cal_end = unique_dates[-test_days]

    train_df = df[df["date"] <= train_end].copy()
    cal_df = df[(df["date"] > train_end) & (df["date"] <= cal_end)].copy()
    eval_df = df[df["date"] > cal_end].copy()

    print(
        f"Split sizes: Train={len(train_df)} rows, Cal={len(cal_df)} rows, Eval={len(eval_df)} rows."
    )

    # 1. Fit Quantile XGBoost on Train
    print("Fitting Quantile XGBoost forecaster on training split...")
    q_xgb = QuantileXGBoostForecaster(
        quantiles=[0.05, 0.50, 0.95],
        n_estimators=cfg.xgboost.n_estimators,
        max_depth=cfg.xgboost.max_depth,
        learning_rate=cfg.xgboost.learning_rate,
        seed=cfg.xgboost.seed,
    )
    q_xgb.fit(train_df, series_id_cols=list(cfg.data.series_id_cols))

    # 2. Calibrate Conformal Calibrator on Cal split
    print("Calibrating Conformal Prediction engine on calibration split...")
    conformal = ConformalCalibrator(base_forecaster=q_xgb, alpha=0.10)
    conformal.calibrate(cal_df=cal_df)

    # 3. Predict on Eval split
    print("Generating calibrated intervals on evaluation window...")
    conformal_intervals = conformal.predict_interval(horizon=test_days, features_df=eval_df)

    # Also compute Naive baseline
    naive_forecaster = NaiveForecaster()
    naive_forecaster.fit(train_df, series_id_cols=list(cfg.data.series_id_cols))
    naive_preds = naive_forecaster.predict(horizon=test_days, features_df=eval_df)

    avg_demand_per_series = (
        train_df.groupby(list(cfg.data.series_id_cols), observed=True)["sales"].mean().mean()
    )

    # 4. Define Competing Policies
    lead_time = cfg.inventory.lead_time_days
    review_period = cfg.inventory.review_period_days
    c_h = cfg.inventory.holding_cost_per_unit_day
    c_s = cfg.inventory.stockout_cost_per_unit
    c_o = cfg.inventory.ordering_cost_per_order

    policies = [
        (
            NaivePolicy(
                avg_daily_demand=float(avg_demand_per_series),
                lead_time=lead_time,
                review_period=review_period,
            ),
            naive_preds,
            "forecast",
        ),
        (
            ForecastDrivenPolicy(
                service_level=0.50,
                review_period=review_period,
                name="PointForecast-ZeroSafetyStock",
            ),
            conformal_intervals,
            "forecast",
        ),
        (
            ForecastDrivenPolicy(
                service_level=0.95, review_period=review_period, name="Gaussian-(R,S)-SL95%"
            ),
            conformal_intervals,
            "forecast",
        ),
        (
            ConformalQuantilePolicy(
                review_period=review_period, name="ConformalQuantile-(R,S)-N1+N2"
            ),
            conformal_intervals,
            "upper",
        ),
    ]

    print("\nRunning inventory simulations across all series...")
    sim_rows = []
    for pol, p_df, u_col in policies:
        sim = Simulator(
            policy=pol,
            lead_time=lead_time,
            holding_cost=c_h,
            stockout_cost=c_s,
            ordering_cost=c_o,
            review_period=review_period,
            initial_inventory=50.0,
        )
        res = simulate_portfolio(
            simulator=sim,
            actual_df=eval_df,
            forecast_df=p_df,
            series_id_cols=list(cfg.data.series_id_cols),
            forecast_col="forecast",
            upper_col=u_col,
        )
        sim_rows.append(
            {
                "policy": res["policy"],
                "fill_rate": res["fill_rate"],
                "total_cost": res["total_cost"],
                "holding_cost": res["holding_cost"],
                "stockout_cost": res["stockout_cost"],
                "ordering_cost": res["ordering_cost"],
                "stockout_periods": res["stockout_periods"],
                "n_orders": res["n_orders"],
            }
        )

    results_df = pd.DataFrame(sim_rows)
    out_table = Path(args.output_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_table, index=False)

    print("\n" + "=" * 85)
    print("PHASE 4: INVENTORY REPLENISHMENT SIMULATION RESULTS (Novelty N1 & N2)")
    print("=" * 85)
    print(results_df.to_string(index=False))
    print("=" * 85)
    print(f"Results saved to: {out_table}\n")

    # 5. Sensitivity Sweep Grid (Task P4-T07)
    print("Running lead time and cost ratio sensitivity sweeps...")
    sensitivity_rows = []
    lead_times = [2, 4, 7]
    cost_ratios = [10.0, 50.0, 100.0]  # c_s / c_h

    for lt in lead_times:
        for cr in cost_ratios:
            curr_cs = c_h * cr
            # Compare Conformal Quantile vs Gaussian Policy
            for pol, p_df, u_col in [
                (
                    ForecastDrivenPolicy(
                        service_level=0.95, review_period=review_period, name="Gaussian-SL95%"
                    ),
                    conformal_intervals,
                    "forecast",
                ),
                (
                    ConformalQuantilePolicy(
                        review_period=review_period, name="ConformalQuantile-CQR"
                    ),
                    conformal_intervals,
                    "upper",
                ),
            ]:
                sim = Simulator(
                    policy=pol,
                    lead_time=lt,
                    holding_cost=c_h,
                    stockout_cost=curr_cs,
                    ordering_cost=c_o,
                    review_period=review_period,
                    initial_inventory=50.0,
                )
                res = simulate_portfolio(
                    simulator=sim,
                    actual_df=eval_df,
                    forecast_df=p_df,
                    series_id_cols=list(cfg.data.series_id_cols),
                    upper_col=u_col,
                )
                sensitivity_rows.append(
                    {
                        "lead_time": lt,
                        "cost_ratio_cs_ch": cr,
                        "stockout_penalty": curr_cs,
                        "policy": pol.name,
                        "fill_rate": res["fill_rate"],
                        "total_cost": res["total_cost"],
                        "holding_cost": res["holding_cost"],
                        "stockout_cost": res["stockout_cost"],
                    }
                )

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_path = Path(args.sensitivity_table)
    sens_df.to_csv(sens_path, index=False)
    print(f"Sensitivity grid saved to: {sens_path}")


if __name__ == "__main__":
    main()
