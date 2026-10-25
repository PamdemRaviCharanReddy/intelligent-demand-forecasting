# Phase 4 Pre-Mortem: What Could Go Wrong & Mitigations

**Phase:** Phase 4 (Uncertainty Quantification & Cost Simulation - P0 / N1 / N2)  
**Date:** 2026-10-25  

---

## 1. Potential Failure Modes

### Risk 1: Crossing Quantiles in Multi-Quantile Regression
- **Failure:** An independent quantile model for $\tau=0.05$ could output a value higher than $\tau=0.95$ for outliers.
- **Mitigation:**
  - Enforce monotonic post-processing: $\hat{q}_{0.05} \le \hat{q}_{0.50} \le \hat{q}_{0.95}$.
  - Non-negativity constraint: $\hat{q}_\tau = \max(0, \hat{q}_\tau)$.

### Risk 2: Conformal Coverage Gap under Distributional Shift
- **Failure:** Calibration data from past months may not match future test distribution if seasonality shifts.
- **Mitigation:**
  - Use recent calibration split immediately preceding the evaluation window.
  - Apply the standard finite-sample adjustment quantile: $\lceil (N_{\text{cal}} + 1)(1 - \alpha) \rceil / N_{\text{cal}}$.

### Risk 3: Unbounded Inventory Accumulation or Double-Counting Pipeline Orders
- **Failure:** Ordering on day $t$ without tracking orders placed on day $t-1$ that arrive on day $t+L$ causes massive over-ordering (the classic Bullwhip Effect).
- **Mitigation:**
  - Strict inventory position accounting:
    $$\text{IP}_t = I_t + \sum_{\tau=t}^{t+L-1} \text{InTransit}_\tau - \text{Backorders}_t$$
  - Order quantity $Q_t = \max(0, S - \text{IP}_t)$ evaluated strictly once per review cycle $R$.

### Risk 4: Leakage between Calibration Split and Backtest Evaluation
- **Failure:** Conformal calibration using future data or data from the test period.
- **Mitigation:**
  - Calibration split must be strictly partitioned from the training fold (e.g. last 28 days of the training fold), never touching the out-of-fold test horizon.
