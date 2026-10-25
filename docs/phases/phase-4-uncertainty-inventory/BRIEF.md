# Phase 4 Brief: Uncertainty Quantification & Cost-Based Inventory Simulation (P0 / N1 / N2)

**Owner:** Lead Systems Architect & Research Engineer  
**Branch:** `phase-4-uncertainty-inventory`  
**Objectives:**  
1. Implement Quantile Regression with XGBoost (`src/idfs/uncertainty/quantile.py`) predicting quantiles $\tau \in \{0.05, 0.50, 0.95\}$.
2. Implement Inductive Split Conformal Prediction & Conformalized Quantile Regression (CQR) (`src/idfs/uncertainty/conformal.py`) guaranteeing empirical coverage $\ge 1 - \alpha$.
3. Implement realistic Daily Inventory Simulation Engine (`src/idfs/inventory/simulation.py`) with lead times, pipeline orders, holding, stockout, and ordering costs.
4. Implement Replenishment Policies (`src/idfs/inventory/policies.py`):
   - `NaivePolicy` (Moving Average Reorder Point)
   - `PointForecastPolicy` (Deterministic Order-Up-To)
   - `UncertaintyAwarePolicy` (Calibrated Conformal / Quantile $(R, S)$ Policy)
5. Execute end-to-end simulation across walk-forward test periods; generate headline cost comparison table (`reports/tables/inventory_simulation_results.csv`).
6. Perform lead time $L \in \{2, 4, 7\}$ and cost ratio $c_s / c_h \in \{10, 50, 100\}$ sensitivity sweeps (`reports/tables/inventory_sensitivity.csv`).

---

## Novelty Stack Grounding
- **Novelty N1 (Cost-Based Inventory Simulation):** Directly addresses the primary problem highlighted in the student's PPT: *"a forecast alone does not say what to order."* Evaluates models by business impact (\$ costs and fill rates), not just statistical loss.
- **Novelty N2 (Calibrated Prediction Intervals):** Combines XGBoost quantile loss with finite-sample split conformal prediction to solve interval under-coverage on skewed demand.
