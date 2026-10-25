# Phase 3 Brief: Additional Models, Intermittent Forecasters & Demand Routing (Tier P1)

**Owner:** Lead Engineer & ML Architect  
**Branch:** `phase-3-additional-models`  
**Objectives:**  
1. Implement Classical Statistical Models: Auto-ARIMA / SARIMAX (`src/idfs/models/statistical.py`).
2. Implement Additive Decomposition Forecaster: Prophet (`src/idfs/models/prophet_model.py`) with fallback to Fourier harmonic regression if Stan compiler binary fails.
3. Implement 12-Step PyTorch Deep Learning Forecaster: Long Short-Term Memory (`src/idfs/models/lstm.py`) matching the students' documented 12-step sequence architecture.
4. Implement Intermittent Demand Specialists: Croston's Method and Teunter-Syntetos-Babai (TSB) Method (`src/idfs/models/intermittent.py`).
5. Implement Demand-Pattern Router (Novelty N3) (`src/idfs/models/router.py`) dynamically classifying series via Syntetos-Boylan ADI vs. $CV^2$ grid into Smooth, Erratic, Intermittent, and Lumpy categories.
6. Implement Model Stacking / Weighted Ensemble Forecaster (`src/idfs/models/ensemble.py`).
7. Update walk-forward benchmark and generate comprehensive comparison table across all model paradigms (`reports/tables/phase3_model_comparison.csv`).

---

## Technical Specifications

### 1. Classical Statistical Models (ARIMA / SARIMAX)
- Uses `statsmodels.tsa.statespace.sarimax.SARIMAX`.
- Supports automated order selection $(p, d, q) \times (P, D, Q)_s$ with $s=7$ for seasonal retail series.
- Per-series fitting with robust convergence safeguards and fallback to simpler orders $(1, 1, 1)$ or seasonal naive upon failure.

### 2. Prophet / Additive Forecaster
- Captures trend changepoints, weekly seasonality, yearly seasonality, and holiday binary regressors.
- Graceful degradation: If `prophet` C++ Stan binaries encounter toolchain issues on Windows, falls back cleanly to a Fourier Harmonic Seasonal Regressor (`HarmonicRegressionForecaster`) without breaking the pipeline.

### 3. 12-Step PyTorch LSTM
- Implements the exact 12 steps from the student documentation:
  1. Data collection & ingestion.
  2. Data availability check.
  3. Integration & cleaning.
  4. Sequence creation: sliding window length $L \in [14, 28]$ predicting step $t+1 \dots t+H$.
  5. Train-only MinMax normalisation ($[0, 1]$ scale).
  6. Neural network architecture: Input layer $\to$ LSTM cell with forget, input, output gates and cell state $\to$ Linear dense layer.
  7. Loss function: MSE / Smooth L1 loss with Adam optimizer.
  8. Early stopping with patience on validation split to prevent overfitting.
  9. CPU/GPU automated device selection.
  10. Inference with inverse scale transformation and non-negativity clamping.
  11. Metric evaluation (MAE, RMSE, WAPE).
  12. Actionable replenishment integration.

### 4. Intermittent Forecasters (Croston & TSB)
- **Croston's Method (1972):** Separates demand intervals $p_t$ from non-zero demand sizes $z_t$. Updates both via exponential smoothing with parameter $\alpha \in (0, 1]$. Forecast $= z_t / p_t$.
- **Teunter-Syntetos-Babai (TSB, 2011):** Updates the probability of demand occurrence rather than inter-arrival time, resolving Croston's obsolescence limitation when demand ceases.

### 5. Demand-Pattern Router (Novelty N3)
- Evaluates each time series over historical training window:
  - $\text{ADI} = \frac{N}{\sum_{t=1}^N \mathbf{1}(y_t > 0)}$ (Average Demand Interval).
  - $CV^2 = \left(\frac{\sigma_z}{\mu_z}\right)^2$ (Squared Coefficient of Variation of non-zero demands).
- Routing matrix:
  - **Smooth** ($\text{ADI} < 1.32, CV^2 < 0.49$) $\to$ `XGBoostForecaster` / `SARIMAXForecaster`
  - **Erratic** ($\text{ADI} < 1.32, CV^2 \ge 0.49$) $\to$ `XGBoostForecaster` / `SeasonalNaiveForecaster`
  - **Intermittent** ($\text{ADI} \ge 1.32, CV^2 < 0.49$) $\to$ `CrostonForecaster`
  - **Lumpy** ($\text{ADI} \ge 1.32, CV^2 \ge 0.49$) $\to$ `TSBForecaster`

### 6. Weighted Stacking Ensemble
- Computes validation performance weights per model:
  $$w_m = \frac{1 / \text{MAE}_m^\gamma}{\sum_{k} 1 / \text{MAE}_k^\gamma}$$
- Blends predictions dynamically with convex combination $\sum w_m = 1$.
