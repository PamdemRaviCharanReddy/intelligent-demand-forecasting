# Phase 3 Pre-Mortem: What Could Go Wrong & Mitigations

**Phase:** Phase 3 (Additional Models & Demand Routing - Tier P1)  
**Date:** 2026-10-25  

---

## 1. Potential Failure Modes

### Risk 1: Fitting 50 individual ARIMA/SARIMAX models in cross-validation is slow or crashes on non-stationary series
- **Probability:** High  
- **Impact:** Medium (slow test runs or non-convergence errors)  
- **Mitigation:**
  - Enforce max iterations (`maxiter=50`) and catch convergence warnings/exceptions cleanly.
  - Fall back to seasonal naive or simple exponential smoothing if SARIMAX fails to invert Hessian.
  - For comprehensive backtests across 50 series, run ARIMA on a stratified subset (e.g. top series per family) or configure fast orders $(1, 0, 0) \times (0, 1, 0)_7$.

### Risk 2: Prophet installation fails due to C++ compiler / cmdstanpy requirements on Windows
- **Probability:** High  
- **Impact:** Low  
- **Mitigation:**
  - Attempt installing `prophet` via uv.
  - Implement a dual architecture: if `prophet` is imported successfully, use `ProphetForecaster`; otherwise, gracefully fall back to `FourierHarmonicForecaster` which provides exact deterministic additive decomposition (trend + Fourier harmonics for 7-day and 365-day cycles + regressors) using pure scikit-learn Ridge regression without C++ compilation requirements.

### Risk 3: PyTorch LSTM training instability, slow CPU training, or shape mismatches
- **Probability:** Medium  
- **Impact:** High  
- **Mitigation:**
  - Design a vectorized sequence dataset (`TimeSeriesDataset`) sliding 28-day windows.
  - Use lightweight 1-2 layer LSTM with hidden dimension 32 or 64, batch size 64, Adam optimizer (lr 1e-3), and early stopping with patience 5.
  - On CPU, keep epochs $\le 15$.
  - Strictly fit the MinMax scaler on training window only; apply `transform` to test and `inverse_transform` to predictions.

### Risk 4: Zero-division or degeneracies in Croston / TSB on completely flat or zero series
- **Probability:** Medium  
- **Impact:** Medium  
- **Mitigation:**
  - Check if series has fewer than 2 non-zero points; if all-zero, forecast 0.0 with epsilon safeguard.
  - In router, handle edge cases where variance is 0 ($CV^2 = 0$).

### Risk 5: Leakage in Router Classification
- **Probability:** Medium  
- **Impact:** Critical (violates Operating Principle 5)  
- **Mitigation:**
  - The router MUST compute ADI and $CV^2$ strictly on the training partition of each walk-forward fold.
  - Never classify series using future test observations.
