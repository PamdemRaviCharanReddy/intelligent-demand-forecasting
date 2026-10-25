# Project Pre-Mortem: Failure Modes & Mitigations

This pre-mortem analyzes 10 specific ways this engineering project could fail, along with automated architectural mitigations built into IDFS.

---

## 1. Risk: Future Information Leakage in Rolling Feature Calculations
- **Failure Mode**: A rolling average computed at date $t$ inadvertently incorporates sales from day $t$, yielding artificially low backtest errors that collapse during inference.
- **Severity**: Critical.
- **Architectural Defense**:
  1. Every rolling window operation strictly calls `df['sales'].shift(1).rolling(w)`.
  2. The test suite includes explicit adversarial tests (`tests/unit/test_features_leakage.py`) where future sales are spiked; if feature values change at day $t$, the build fails.

---

## 2. Risk: Prediction Interval Miscalibration (Undercoverage)
- **Failure Mode**: Quantile XGBoost generates nominal 90% prediction intervals, but empirical backtest coverage achieves only 70%, leading to unexpected stockouts.
- **Severity**: High.
- **Architectural Defense**:
  1. Conformal prediction layer computes non-conformity scores on validation residuals to calibrate interval widths.
  2. If coverage falls below nominal threshold by $> 3\%$, the calibrator widens bounds via empirical conformal adjustments.

---

## 3. Risk: Memory Exhaustion During Deep Learning (LSTM) Training
- **Failure Mode**: Creating 3D sliding sequence arrays $(N \times L \times D)$ across millions of retail rows exhausts 12GB host RAM.
- **Severity**: High.
- **Architectural Defense**:
  1. PyTorch `Dataset` uses lazy on-the-fly window slicing rather than pre-materializing all sequence matrices in RAM.
  2. Sequence length defaults to a conservative $L=28$; batch size defaults to 64.

---

## 4. Risk: CPU Bottleneck on Classical Statistical Models (ARIMA / SARIMA)
- **Failure Mode**: Running `auto_arima` or SARIMAX on all 1,782 series takes 15+ hours, blocking continuous development.
- **Severity**: Medium.
- **Architectural Defense**:
  1. Local statistical models are budgeted to a representative 20–50 series benchmark subset.
  2. Global models (XGBoost) train across all series concurrently in minutes.

---

## 5. Risk: Division by Zero in Intermittent Demand Metrics (MAPE)
- **Failure Mode**: Slow-moving or intermittent products have true actuals $y_t = 0$, producing infinite or undefined MAPE values that crash evaluation scripts.
- **Severity**: Medium.
- **Architectural Defense**:
  1. Pure metrics library implements zero-masked MAPE with explicit warnings.
  2. Headline reporting prioritizes WAPE and MASE, which are mathematically stable on intermittent series.

---

## 6. Risk: Incomplete or Missing Exogenous Data Streams
- **Failure Mode**: Real-world uploaded CSVs lack oil price indices or holiday calendars, causing `KeyError` during fusion.
- **Severity**: Medium.
- **Architectural Defense**:
  1. Multi-source fusion engine implements graceful degradation: missing regressors are zero-imputed with logged warnings.
  2. The pipeline continues functioning end-to-end on sales-only core features.

---

## 7. Risk: Unrealistic Inventory Simulation Dynamics
- **Failure Mode**: Simulation assumes instantaneous replenishment ($L=0$) or unlimited capacity, rendering inventory savings claims unconvincing in academic defense.
- **Severity**: High.
- **Architectural Defense**:
  1. Orders in transit are explicitly tracked across a non-zero lead time ($L \in [2, 7]$).
  2. Complete sensitivity sweeps evaluate multiple cost ratios ($c_s / c_h$) to demonstrate robustness under varying operational scenarios.

---

## 8. Risk: Over-Fitting on Validation Folds
- **Failure Mode**: Hyperparameters tuned excessively on validation folds fail to generalize to the final evaluation period.
- **Severity**: High.
- **Architectural Defense**:
  1. A 28-day final test period is sealed and remains untouched until Phase 7.
  2. Multi-fold rolling-origin cross-validation (3 folds) guards against overfitting to a single temporal window.

---

## 9. Risk: Concept Drift Following Economic or Seasonal Shocks
- **Failure Mode**: Model trained on pre-pandemic or seasonal data degrades severely after structural market changes.
- **Severity**: Medium.
- **Architectural Defense**:
  1. Rolling error drift monitor (N4) computes tracking signals over a 28-day sliding window.
  2. Automatically triggers visual alerts and retrain suggestions when error exceeds 1.5× baseline.

---

## 10. Risk: Streamlit Application Coupling with Computational Core
- **Failure Mode**: UI crashes render underlying forecasting algorithms untestable or un-evaluable.
- **Severity**: Medium.
- **Architectural Defense**:
  1. Strict architectural separation: all forecasting, inventory, and metric logic lives in `src/idfs/`.
  2. Streamlit pages only call imported library functions; CLI and automated scripts reproduce all results independently.
