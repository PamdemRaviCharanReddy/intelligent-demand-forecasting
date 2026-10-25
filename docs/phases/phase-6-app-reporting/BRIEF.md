# Phase 6 Brief: Streamlit Application & Executive Reporting (Tier P0)

**Owner:** Lead Full-Stack ML Engineer  
**Branch:** `phase-6-app-reporting`  
**Objectives:**  
1. Build multi-page Streamlit web application (`src/idfs/app/main.py` and modular subpages):
   - Page 1: Data Explorer & Automated EDA
   - Page 2: Multi-Model Forecast Studio with prediction intervals
   - Page 3: Inventory Replenishment & Stockout/Overstock Alerts (Novelties N1 + N2)
   - Page 4: Surveillance Drift Monitor & Tree-SHAP Explainability (Novelties N4 + N5)
2. Implement local SQLite persistent run storage (`src/idfs/app/storage.py`) tracking user scenarios, decisions, and planner notes.
3. Build executive HTML/CSV downloadable report generator (`src/idfs/reporting/report_builder.py`).
4. Add automated headless smoke tests ensuring zero runtime errors on all app pages (`tests/smoke/test_app_smoke.py`).

---

## Alignment with Student Requirements
Directly fulfills Report Objectives (2), (8), (9) and Presentation Claims:
- Upload CSV/Excel and validate.
- Generate forecasts with calibrated uncertainty bands.
- Produce actionable inventory replenishment (order quantities, safety stocks, reorder points, stockout risk).
- Executive downloadable report generator and human planner decision review panel.
