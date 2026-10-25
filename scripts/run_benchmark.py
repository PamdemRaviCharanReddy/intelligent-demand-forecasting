"""Execute Model Benchmark (Phase 2 & Phase 3).

Evaluates all models across identical expanding walk-forward folds and saves
the comparison table to reports/tables/phase3_model_comparison.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idfs.config import load_config
from idfs.evaluation.backtester import Backtester
from idfs.models.base import NaiveForecaster, SeasonalNaiveForecaster
from idfs.models.ensemble import WeightedEnsembleForecaster
from idfs.models.intermittent import CrostonForecaster, TSBForecaster
from idfs.models.lstm import LSTMForecaster
from idfs.models.router import DemandPatternRouterForecaster
from idfs.models.statistical import SARIMAXForecaster
from idfs.models.xgboost import XGBoostForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IDFS Model Benchmark")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/fused_sales.parquet",
        help="Path to fused processed dataset",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default="reports/tables/phase3_model_comparison.csv",
        help="Path to save comparative metrics table",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=3,
        choices=[2, 3],
        help="Benchmark phase (2 for baselines+XGBoost, 3 for full model suite)",
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

    print(
        f"Loaded {len(fused_df)} rows across {fused_df['store_id'].nunique()} stores and {fused_df['family'].nunique()} families."
    )

    # Initialize Walk-Forward Backtester
    backtester = Backtester(
        n_folds=cfg.split.n_folds,
        val_days=cfg.split.val_days,
        gap_days=cfg.split.gap_days,
        test_days=cfg.split.test_days,
    )

    xgb_model = XGBoostForecaster(
        n_estimators=cfg.xgboost.n_estimators,
        max_depth=cfg.xgboost.max_depth,
        learning_rate=cfg.xgboost.learning_rate,
        subsample=cfg.xgboost.subsample,
        colsample_bytree=cfg.xgboost.colsample_bytree,
        min_child_weight=cfg.xgboost.min_child_weight,
        seed=cfg.xgboost.seed,
    )

    if args.phase == 2:
        models = [
            NaiveForecaster(),
            SeasonalNaiveForecaster(seasonality=7),
            xgb_model,
        ]
    else:
        # Full Phase 3 multi-paradigm suite
        snaive_model = SeasonalNaiveForecaster(seasonality=7)
        croston_model = CrostonForecaster(alpha=0.1, variant="sba")
        tsb_model = TSBForecaster(alpha_d=0.2, alpha_p=0.2)
        sarimax_model = SARIMAXForecaster(
            order=(1, 0, 0), seasonal_order=(0, 1, 0, 7), maxiter=25, name="SARIMAX(1,0,0)x(0,1,0)7"
        )
        router_model = DemandPatternRouterForecaster(
            smooth_model=XGBoostForecaster(n_estimators=60, max_depth=5, name="Router-Smooth-XGB"),
            erratic_model=SeasonalNaiveForecaster(seasonality=7, name="Router-Erratic-SNaive"),
            intermittent_model=CrostonForecaster(alpha=0.1, name="Router-Intermittent-Croston"),
            lumpy_model=TSBForecaster(alpha_d=0.2, alpha_p=0.2, name="Router-Lumpy-TSB"),
            name="DemandPatternRouter(N3)",
        )
        ensemble_model = WeightedEnsembleForecaster(
            models=[
                SeasonalNaiveForecaster(seasonality=7, name="Ens-SNaive"),
                CrostonForecaster(alpha=0.1, name="Ens-Croston"),
                SARIMAXForecaster(
                    order=(1, 0, 0), seasonal_order=(0, 1, 0, 7), maxiter=20, name="Ens-SARIMAX"
                ),
            ],
            weighting_strategy="uniform",
            name="WeightedEnsemble",
        )
        lstm_model = LSTMForecaster(
            seq_len=28,
            horizon=14,
            hidden_dim=32,
            num_layers=1,
            epochs=6,
            batch_size=64,
            device="cpu",
            name="PyTorch-LSTM",
        )

        models = [
            SeasonalNaiveForecaster(seasonality=7),
            NaiveForecaster(),
            xgb_model,
            croston_model,
            tsb_model,
            sarimax_model,
            router_model,
            ensemble_model,
            lstm_model,
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
    print(f"PHASE {args.phase} MODEL COMPARISON BENCHMARK (3 Identical Walk-Forward Folds)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
    print("=" * 80)
    print(f"Results saved to: {out_p}\n")


if __name__ == "__main__":
    main()
