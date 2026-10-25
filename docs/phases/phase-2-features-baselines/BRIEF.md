# Phase 2 Brief: Features, Baselines, Backtest & XGBoost (P0 Core)

## 1. Goal
Implement the core predictive backbone of IDFS:
1. Leakage-free feature engineering pipeline: lag features ($k \ge 1$), shifted rolling statistics ($w \in \{7, 14, 28\}$ with mandatory `.shift(1)`), cyclical sine/cosine calendar encodings, holiday distance features, and train-fitted categorical target encoders.
2. Verified baseline forecasters: Naive and Seasonal Naive ($m=7$).
3. Expanding-window rolling-origin walk-forward backtest harness producing identical fold splits across all models.
4. Global XGBoost forecaster training across all series concurrently with early stopping on validation folds.
5. First reproducible benchmark comparison table comparing Naive, Seasonal Naive, and XGBoost across identical folds with MAE, RMSE, WAPE, and MASE.

## 2. Inputs Available
- Processed fused sales dataset at `data/processed/fused_sales.parquet`.
- Evaluation metrics library at `src/idfs/evaluation/metrics.py`.
- Base forecaster interface at `src/idfs/models/base.py`.
- Default hyperparameters in `configs/default.yaml`.

## 3. Constraints & Leakage Invariants
- **Non-Negotiable Leakage Rule**: For forecast origin $T_{\text{origin}}$, features for horizon step $t > T_{\text{origin}}$ cannot access target values $y_\tau$ where $\tau > T_{\text{origin}}$.
- **Pristine Test Horizon**: Final 28 days remain completely locked and excluded from backtesting folds.
- **Identical Folds**: XGBoost and baselines must be evaluated on the exact same fold start and end dates.
