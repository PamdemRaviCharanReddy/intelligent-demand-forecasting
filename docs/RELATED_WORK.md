# Related Work & Research Gap Analysis

This document provides a systematic review of time-series demand forecasting and supply chain inventory literature, analyzes citations referenced in the students' drafts, and establishes the formal Research Gap Table for IDFS.

---

## 1. Research Gap Analysis Table

The student presentation identifies five specific research gaps: Scalability, Model Drift, Data Sparsity, Explainability, and the Forecast-to-Decision Gap. The table below benchmarks existing literature against IDFS:

| Dimension / Paper | Scalability Across Multi-Series | Adaptive Model Drift Handling | Intermittent / Sparse Demand | Explainability (XAI) | Uncertainty-to-Inventory Optimization |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Classical ARIMA / Box & Jenkins (1970)** | ❌ Isolated per-series; O(N) compute | ❌ Static parameters; no drift alerts | ❌ Assumes Gaussian continuity | ⚠️ Linear coefficients only | ❌ Theoretical standard deviation only |
| **Prophet / Taylor & Letham (2018)** | ❌ Local additive model; slow on large collections | ⚠️ Changepoint detection; no auto-retrain trigger | ❌ Struggles with zero-inflated data | ⚠️ Additive components (trend/seasonality) | ❌ No inventory cost modeling |
| **Standard Retail XGBoost / Chen & Guestrin (2016)** | ✅ Global cross-series learning | ❌ Static tree structures | ⚠️ Needs heavy zero-feature engineering | ✅ Tree SHAP feature attribution | ❌ Standard RMSE/MAE point loss only |
| **Radhakrishnan et al. (2025)** `[VERIFY]` | ⚠️ Hybrid LSTM; limited series count | ❌ Not explicitly addressed | ❌ Evaluated on smooth aggregated sales | ❌ Black-box deep network | ⚠️ High-level reorder point discussion |
| **MCDFN (2025)** `[VERIFY]` | ⚠️ Complex multi-channel fusion | ❌ Retraining cost high | ⚠️ Multi-source assists sparsity | ✅ Attention weights visualization | ❌ Lacks asymmetric inventory simulation |
| **IDFS (This Project)** | ✅ **Global XGBoost + PyTorch LSTM cross-learning** | ✅ **Rolling error drift monitor (N4)** | ✅ **Demand router (N3) + Croston / TSB** | ✅ **SHAP global & local attribution (N5)** | ✅ **Periodic review cost simulation (N1, N2)** |

---

## 2. Analysis of Primary & Baseline Literature

### A. Classical Statistical & Additive Foundations
1. **Box, G. E. P., & Jenkins, G. M. (1970).** *Time Series Analysis: Forecasting and Control.* Holden-Day.
   - Introduced the Autoregressive Integrated Moving Average (ARIMA) methodology. Effective for stationary univariate series, but computationally intractable for multi-thousand series retail matrices and unable to ingest cross-series exogenous covariates.
2. **Hyndman, R. J., & Athanasopoulos, G. (2021).** *Forecasting: Principles and Practice* (3rd ed.). OTexts.
   - Core reference for scale-free forecasting evaluation (MASE), seasonal decompositions, and intermittent demand methods (Croston, TSB).
3. **Taylor, S. J., & Letham, B. (2018).** *Forecasting at scale.* The American Statistician, 72(1), 37-45.
   - Generalized additive model decomposing time series into piecewise linear/logistic trend, Fourier series seasonality, and holiday binary indicators. Serves as a vital benchmark for interpretable additive forecasting.

### B. Machine Learning & Deep Learning Benchmarks
4. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A scalable tree boosting system.* KDD '16, 785-794.
   - The primary workhorse of tabular and competitive retail forecasting (M5, Favorita). Efficiently captures cross-series interactions and non-linear holiday/promotional responses.
5. **Hochreiter, S., & Schmidhuber, J. (1997).** *Long short-term memory.* Neural Computation, 9(8), 1735-1780.
   - Recurrent architecture with gating mechanisms (input, forget, output gates) overcoming vanishing gradients. Applied to sequential multi-step retail demand forecasting.
6. **Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).** *The M4 Competition: 100,000 time series and 61 forecasting methods.* International Journal of Forecasting, 36(1), 54-74.
   - Conclusively demonstrated that pure ML and hybrid combinations consistently outperform pure statistical methods on large retail collections.

---

## 3. Review of Recent Literature Cited in Student Slides

*Note: Citations extracted from preliminary student drafts with incomplete publication metadata are flagged with `[VERIFY]` and must be confirmed prior to final thesis submission.*

7. **Radhakrishnan et al. (2025).** *"AI-Based Demand Forecasting System for Supply Chain Optimization."* `[VERIFY venue and author list]`
   - The students' PPT notes reported MAPE scores (Hybrid 6.8%, LSTM 8.5%, Random Forest 9.2%). *Crucial Defense Note: These numbers were obtained on a distinct, proprietary dataset; IDFS results cannot be directly compared against them as equivalent.*
8. **MCDFN (2025).** *"Supply Chain Demand Forecasting via an Explainable Multi-Channel Data Fusion Network."* Evolutionary Intelligence. `[VERIFY volume and pages]`
   - Emphasizes multi-channel fusion (pricing, promotions, holidays) and attention-based explainability. Directly informs IDFS's multi-source feature architecture.
9. **Zulfiqar et al. (2025).** *"AI-Driven Smart Shopping Carts With Real-Time Tracking and Inventory Forecasting for Enhanced Retail Efficiency."* `[VERIFY full citation]`
   - Cited in introductory drafts; IDFS strictly narrows the scope to demand forecasting and inventory replenishment, eliminating hardware cart tracking and billing mechanisms.
10. **Syntetos, A. A., & Boylan, J. E. (2005).** *The accuracy of intermittent demand estimates.* International Journal of Forecasting, 21(2), 303-314.
    - Foundational formulation of the Average Demand Interval (ADI) and squared coefficient of variation ($CV^2$) categorization plane used in the IDFS Demand Pattern Router (Novelty N3).
