"""Execute Phase 2 model benchmark: Naive vs Seasonal Naive vs Global XGBoost.

Evaluates all models across identical expanding walk-forward folds and saves
the comparison table to reports/tables/phase2_model_comparison.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idfs.config import load_config
from idfs.evaluation.backtester import Backtester
from idfs.models.base import NaiveForecaster, SeasonalNaiveForecaster
from idfs.models.xgboost import XGBoostForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDFS Phase 2 Model Benchmark")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/fused_sales.parquet",
        help="Path to fused processed dataset",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default="reports/tables/phase2_model_comparison.csv",
        help="Path to save comparative metrics table",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Running data pipeline first...")
        from idfs.data.fusion import run_data_pipeline

        fused_df = run_data_pipeline("data/synthetic", data_path)
    else:
        print(f"Loading processed dataset from {data_path}...")
        fused_df = pd.read_parquet(data_path)

    print(f"Loaded {len(fused_df)} rows across {fused_df['store_id'].nunique()} stores and {fused_df['family'].nunique()} families.")

    # Initialize Walk-Forward Backtester
    backtester = Backtester(
        n_folds=cfg.split.n_folds,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        test_days=cfg.split.test_days,
    )

    models = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(seasonality=7),
        XGBoostForecaster(
            n_estimators=cfg.xgboost.n_estimators,
            max_depth=cfg.xgboost.max_depth,
            learning_rate=cfg.xgboost.learning_rate,
            subsample=cfg.xgboost.subsample,
            colsample_bytree=cfg.xgboost.colsample_bytree,
            min_child_weight=cfg.xgboost.min_child_weight,
            seed=cfg.xgboost.seed,
        ),
    ]

    print("\nRunning rolling-origin walk-forward evaluation across identical folds...")
    comp_df = backtester.compare_models(
        models=models,
        df=fused_df,
        target_col=cfg.data.target_col,
        date_col=cfg.data.date_col,
        series_id_cols=list(cfg.data.series_id_cols),
    )

    out_p = Path(args.output_table)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(out_p, index=False)

    print("\n" + "=" * 80)
    print("PHASE 2 MODEL COMPARISON BENCHMARK (3 Identical Walk-Forward Folds)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)
    print(f"Results saved to: {out_p}\n")


if __name__ == "__main__":
    main()
