# Project Scope & Boundary Definition

## 1. System Vision & Contribution Statement
IDFS is an uncertainty-aware, cost-optimised forecasting and inventory decision system that bridges the gap between predictive modeling and supply chain replenishment. It addresses the fundamental retail pain point: traditional statistical point forecasts fail to account for asymmetric cost structures, lead-time uncertainty, and concept drift. By directly driving periodic inventory policies and quantifying confidence with calibrated prediction intervals, IDFS translates demand intelligence into measurable operational savings.

---

## 2. Tiered Requirements Mapping

The project scope is organized into three tiers to guarantee delivery of all core commitments under strict time and resource bounds.

### Mapping to Report Objectives (1–9) and Novelties (N1–N6)

| ID | Capability / Feature | Report Obj | Novelty | Tier | Definition of Done |
|:---|:---|:---:|:---:|:---:|:---|
| **REQ-01** | Data Ingestion & Validation (CSV/Parquet) | (2), (3) | - | **P0** | Validates schema, handles nulls, dedupes, flags outliers |
| **REQ-02** | Synthetic Data Generator | (1) | - | **P0** | Produces multi-source retail data (trend, seasonality, promo, drift) |
| **REQ-03** | Feature Pipeline (Lags, Rolling, Calendar) | (4) | - | **P0** | Zero-leakage features with `.shift(1)` enforcement |
| **REQ-04** | Baseline Models (Naive, Seasonal Naive) | (5) | - | **P0** | Standard benchmark models for scale-free comparisons |
| **REQ-05** | Global XGBoost Forecaster | (5) | - | **P0** | Multi-series cross-learning gradient boosting model |
| **REQ-06** | Walk-Forward Backtester | (6) | - | **P0** | Rolling-origin CV over 3 identical folds; pristine test set |
| **REQ-07** | Core Evaluation Metrics (MAE, RMSE, WAPE, MASE) | (6) | - | **P0** | Comprehensive metric suite with zero-division handling |
| **REQ-08** | Prediction Intervals & Calibration | (7) | **N2** | **P0** | Quantile XGBoost + split conformal coverage verification |
| **REQ-09** | Cost-Based Inventory Simulation Engine | (8) | **N1** | **P0** | Simulates $(R, S)$ policy; compares forecast vs naive cost |
| **REQ-10** | Streamlit Multi-Page Dashboard | (9) | - | **P0** | Interactive upload, model comparison, forecast & inventory views |
| **REQ-11** | Automated Report Generation | (9) | - | **P0** | Exportable summary reports in Markdown/HTML/CSV |
| **REQ-12** | Statistical Forecasters (ARIMA / SARIMA) | (5) | - | **P1** | Per-series classical models on stratified 20–50 series subset |
| **REQ-13** | Additive Decomposition (Prophet) | (5) | - | **P1** | Generalized additive model with holiday/trend changepoints |
| **REQ-14** | Deep Learning Forecaster (PyTorch LSTM) | (5) | - | **P1** | 12-step LSTM with sequential windowing and early stopping |
| **REQ-15** | Intermittent Demand Specialist (Croston / TSB) | (5) | - | **P1** | Separate non-zero size and inter-arrival demand modeling |
| **REQ-16** | Demand-Pattern Router | (6) | **N3** | **P1** | Routes series via ADI and $CV^2$ to specialized forecasters |
| **REQ-17** | Model Stacking / Weighted Ensemble | (5) | - | **P1** | Combines top model forecasts using validation performance |
| **REQ-18** | Rolling-Error Drift Monitor & Retrain Alert | - | **N4** | **P1** | Detects statistical degradation; fires retrain warning |
| **REQ-19** | Model Explainability (SHAP Global & Local) | - | **N5** | **P1** | Attribution plots of top drivers for planner transparency |
| **REQ-20** | Interactive What-If Scenario Simulator | - | **N5** | **P2** | Dynamic slider for price/promo shock testing on demand & stock |
| **REQ-21** | Secondary Dataset Integration (M5 Walmart) | - | - | **P2** | Generalizability check across a second retail domain |
| **REQ-22** | Foundation Model Zero-Shot Baseline | - | **N6** | **P2** | Pretrained transformer (e.g. Chronos/TimesFM) zero-shot evaluation |

---

## 3. Strict Project Boundaries

### In Scope
- Daily sales demand forecasting at Store $\times$ Product Family granularity.
- Multi-source fusion: historical sales, calendar holidays, promotional flags, store metadata, and oil price regressors.
- Time-series cross-validation strictly adhering to temporal precedence.
- Inventory policy simulation evaluating holding, stockout, and ordering cost trade-offs under estimated lead-time uncertainty.
- Local Streamlit application with headless test capability.

### Out of Scope
- Hardware-level IoT devices (smart shopping carts, RFID scanners, or physical warehouse robotics mentioned in exploratory student drafts).
- Store-floor layout navigation, cashier billing algorithms, and reinforcement learning robot pathfinding.
- Intraday / hourly real-time transaction streaming.
- Multi-echelon global supply chain network optimization (factory $\to$ regional hub $\to$ store routing). Focus is localized store replenishment.

---

## 4. Result-Agnostic Success Criteria

To ensure scientific honesty and prevent publication bias, all project acceptance criteria are defined by structural and experimental integrity rather than arbitrary performance thresholds:

1. **Evaluation Consistency**: Every candidate model is evaluated on identical rolling-origin walk-forward folds without data leakage.
2. **Scale-Free Comparative Benchmarks**: All candidate models are compared against both Naive and Seasonal Naive baselines; MASE values are reported for all series.
3. **Uncertainty Calibration**: Prediction interval coverage (e.g., nominal 90%) is empirically validated against ground truth actuals and reported alongside average interval width.
4. **Economic Quantification**: Inventory policy simulation produces complete cost breakdowns (holding cost, stockout penalty, ordering cost) and achieved fill-rate percentages comparing forecast-driven policies against naive historical averages.
5. **Ablation & Sensitivity Analysis**: Impact of feature blocks (lags, calendar, promotions) and lead-time variations ($L=2$ to $L=7$ days) are systematically measured and tabulated.
6. **Negative Result Integrity**: In the event that complex models (e.g., LSTM or Prophet) are outperformed by simpler models (e.g., Seasonal Naive or XGBoost), the finding is preserved, documented, and analyzed in the final report.
