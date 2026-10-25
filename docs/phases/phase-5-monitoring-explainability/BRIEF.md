# Phase 5 Brief: Monitoring & Explainability (Tier P1 / N4 / N5)

**Owner:** Lead ML Engineer & Research Architect  
**Branch:** `phase-5-monitoring-explainability`  
**Objectives:**  
1. Implement Rolling-Error Concept Drift Monitor (Novelty N4) (`src/idfs/monitoring/drift.py`) tracking MAE/WAPE over sliding windows with CUSUM and threshold multipliers.
2. Execute simulated demand shock experiment demonstrating real-time drift detection and alert triggering (`scripts/run_drift_demo.py` $\to$ `reports/figures/drift_detection_demo.png` and `reports/tables/drift_monitoring_log.csv`).
3. Implement Tree-SHAP Global Feature Importance (Novelty N5) (`src/idfs/explain/shap_explainer.py` $\to$ `reports/figures/shap_summary.png` and `reports/tables/shap_feature_importance.csv`).
4. Implement Local SHAP Single-Prediction Waterfall (Novelty N5) (`reports/figures/shap_waterfall.png`).
5. Implement What-If Interactive Scenario Engine (Novelty N5) (`src/idfs/explain/what_if.py`) simulating promotion shocks, price shifts, and macroeconomic volatility.

---

## Novelty Stack Grounding
- **Novelty N4 (Production Drift Monitoring):** Time-series ML models in production inevitably degrade due to seasonality shifts, macro shifts (e.g. oil price shocks), or consumer behavioral drift. Our rolling-error drift monitor provides continuous surveillance and retrain alerts.
- **Novelty N5 (Explainability & What-If Simulation):** Planners in real retail settings will not adopt "black box" machine learning models without understanding *why* demand was forecast to spike or plunge. Tree-SHAP attribution plus interactive scenario analysis empowers planners to review and stress-test assumptions.
