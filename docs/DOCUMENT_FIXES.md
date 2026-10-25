# Document Fixes Log

> Lists every known inconsistency across the students' report, PPT, INTRODUCTION, and simple flow documents. The **unified** version below is what this project implements.

## 1. Model List Inconsistencies

| Source | Models Listed |
|--------|-------------|
| Report | ARIMA, Prophet, XGBoost, LSTM |
| PPT Methodology | ARIMA, Random Forest, LSTM |
| PPT Next Steps | ARIMA, XGBoost, LSTM + hybrid |
| INTRODUCTION doc | LSTM only |
| Simple Flow | XGBoost only |

### ✅ Unified Model List (this project)

| Model | Role | Tier |
|-------|------|------|
| Naive (last value) | Baseline | P0 |
| Seasonal Naive | Baseline | P0 |
| ARIMA/SARIMA | Classical statistical (per-series, subset) | P1 |
| Prophet | Additive decomposition (per-series, subset) | P1 |
| XGBoost | Global ML model (core) | P0 |
| LSTM | Global deep learning model | P1 |
| Croston/TSB | Intermittent demand specialist | P1 |
| Stacking/Weighted Ensemble | Model combination | P1 |

> Random Forest (from PPT) is not included as a separate model. XGBoost covers the tree-based approach with superior performance. If needed as an ablation, it can be added trivially.

## 2. Metric List Inconsistencies

| Source | Metrics Listed |
|--------|---------------|
| Report | MAE, RMSE, MAPE |
| PPT (some slides) | MAE, RMSE, MAPE, R² |
| Simple Flow | MAE, RMSE |
| LSTM notes | MAE, RMSE, MAPE |
| PPT (other slides) | MAE, MSE, RMSE |

### ✅ Unified Metric List (this project)

| Metric | Role | Notes |
|--------|------|-------|
| MAE | Primary point accuracy | Always reported |
| RMSE | Penalises large errors | Always reported |
| WAPE | Weighted aggregate accuracy | Replaces MAPE as headline for aggregated results |
| MASE | Scale-free comparison | Relative to naive baseline |
| MAPE | Secondary/legacy | Reported but zero-masked; not used as headline due to zero-division issues |
| Pinball Loss | Interval/quantile quality | For uncertainty evaluation |
| Empirical Coverage | Interval calibration | % of actuals within predicted intervals |
| Interval Width | Interval sharpness | Narrower is better at same coverage |

> R² is **not** used as a headline metric for time-series forecasting (it can be misleading). MSE is redundant given RMSE.

## 3. Tools/Technology Inconsistencies

| Source | Tools Listed |
|--------|-------------|
| Report | Python, Pandas, NumPy, Scikit-learn, Statsmodels, Prophet, XGBoost, TensorFlow/Keras, Streamlit, Plotly/Matplotlib, CSV/SQLite |
| PPT | Python, Pandas, NumPy, Scikit-learn, TensorFlow (missing: Streamlit, XGBoost, Prophet, Statsmodels) |

### ✅ Unified Tools List (this project)

- **Language:** Python 3.12
- **Data:** Pandas, NumPy, Polars (optional acceleration)
- **ML/Stats:** Scikit-learn, Statsmodels (ARIMA/SARIMA), XGBoost, PyTorch (LSTM — chosen over TensorFlow/Keras for 3.12 compatibility; see ADR)
- **Forecasting:** Prophet (if installable; statsforecast as fallback)
- **Visualisation:** Plotly, Matplotlib, Seaborn
- **UI:** Streamlit
- **Storage:** CSV/Parquet for data and results, SQLite for app run storage
- **Explainability:** SHAP
- **Dev tools:** ruff (lint/format), mypy (type checking), pytest (testing), uv (package management)

## 4. Scope/Content Gaps in PPT

- PPT scope slide has blanks for: number of products, number of months, number of weeks
- Reference [1] has placeholder authors (no names given)
- Lit Review IV sources (McKinsey 2011, Orłowski 2014, Elragal 2014, Disney & Towill 2003) are absent from the main reference list and are primarily about big data/ERP, not demand forecasting
- Zulfiqar et al. (2025) from the INTRODUCTION is missing from both report and PPT reference lists

## 5. Mentor List Discrepancy

- **Report:** Dr. Sukanta Das
- **PPT:** Dr. Sukanta Das and Dr. Sandeep Kumar Panda

`[VERIFY]` Which is correct — one mentor or two? The project proceeds listing both with `[VERIFY]` tags.

## 6. Student ID Discrepancy

- Jaiti Vijaya Shravani listed as 24STUCHH01010756 (8 digits after "24STUCHH0") vs others with 6 digits
- Likely correct: **24STUCHH010756** `[VERIFY]`

## 7. Reference List Issues

- No single merged reference list exists across report and PPT
- PPT references are incomplete (missing authors, years, venues)
- Report references and PPT references do not overlap
- The unified reference list is maintained in `docs/RELATED_WORK.md`

## 8. Slide Numbering

- PPT has repeated slide numbers in the provided summary

## 9. LSTM Terminology

- INTRODUCTION uses "last short-time memory" — corrected to **Long Short-Term Memory (LSTM)**
