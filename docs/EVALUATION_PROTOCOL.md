# Evaluation Protocol & Validation Methodology

This document specifies the experimental standards, temporal split methodology, mathematical definitions of all metrics, and statistical comparison tests used throughout IDFS.

---

## 1. Walk-Forward (Rolling-Origin) Backtest Protocol

Standard random $k$-fold cross-validation destroys the temporal ordering of time-series observations and induces catastrophic forward-looking data leakage. IDFS strictly uses an **expanding-window rolling-origin walk-forward protocol**.

### A. Fold Structure & Split Boundaries
- **Total Historical Span**: $T$ days.
- **Pristine Test Horizon**: The final $T_{\text{test}} = 28$ days are sealed and remain completely untouched until Phase 7.
- **Validation Folds**: $K = 3$ walk-forward folds evaluated over the pre-test history.
- **Validation Window**: $H = 14$ days per fold.
- **Origin Cadence**: For fold $k \in \{0, 1, 2\}$, the forecast origin advances by $H$ days:
  $$\text{Train}_k = [0, \; T_{\text{origin}, k}], \quad \text{Val}_k = (T_{\text{origin}, k}, \; T_{\text{origin}, k} + H]$$
- **Guarantee**: Every candidate model (Naive, Seasonal Naive, XGBoost, LSTM, ARIMA, Prophet) is evaluated on the exact same fold slices.

---

## 2. Mathematical Metric Definitions

### A. Point Accuracy Metrics

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Primary point metric. Linear error penalty.*

2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Penalizes large outlier forecast errors.*

3. **Weighted Absolute Percentage Error (WAPE)**:
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}$$
   *Headline percentage metric across series. Avoids the zero-division pathology of MAPE.*

4. **Mean Absolute Scaled Error (MASE)**:
   $$\text{MASE} = \frac{\frac{1}{H} \sum_{t=1}^H |y_t - \hat{y}_t|}{\frac{1}{T-m} \sum_{t=m+1}^T |y_t - y_{t-m}|}$$
   *Scale-free comparison against an in-sample seasonal naive benchmark (seasonality $m=7$). $\text{MASE} < 1.0$ confirms the model outperforms a simple naive repeat.*

5. **Mean Absolute Percentage Error (MAPE, Zero-Masked)**:
   $$\text{MAPE} = \frac{1}{|M|} \sum_{i \in M} \frac{|y_i - \hat{y}_i|}{y_i} \times 100\%, \quad M = \{i \mid y_i \ne 0\}$$
   *Secondary metric retained for alignment with student report; zero values are masked to prevent $\infty$. Never used as sole headline.*

---

### B. Uncertainty & Interval Metrics

1. **Pinball Loss (Quantile Loss at $\tau$)**:
   $$\mathcal{L}_\tau(y, \hat{q}_\tau) = \max \Big( \tau(y - \hat{q}_\tau), \; (\tau - 1)(y - \hat{q}_\tau) \Big)$$
   *Evaluated at $\tau \in \{0.05, 0.50, 0.95\}$.*

2. **Empirical Coverage (Nominal $1-\alpha$)**:
   $$\text{Coverage} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\Big( \hat{L}_i \le y_i \le \hat{U}_i \Big)$$
   *For a nominal $90\%$ interval, empirical coverage should satisfy $\text{Coverage} \approx 0.90$.*

3. **Mean Interval Width (Sharpness)**:
   $$\text{Width} = \frac{1}{N} \sum_{i=1}^N (\hat{U}_i - \hat{L}_i)$$
   *Among models achieving valid empirical coverage, narrower width indicates superior predictive sharpness.*

---

### C. Inventory Performance Metrics

1. **Item Fill Rate**:
   $$\text{Fill Rate} = \frac{\sum_{t=1}^T \text{Units Fulfilled}_t}{\sum_{t=1}^T \text{Demand}_t}$$
   *Proportion of total customer demand satisfied immediately from stock.*

2. **Stockout Periods**:
   $$\text{Stockouts} = \sum_{t=1}^T \mathbb{I}(I_t \le 0)$$
   *Count of days where stock on hand was completely exhausted.*

3. **Total Inventory Operating Cost**:
   $$\text{Cost}_{\text{total}} = c_h \sum_{t=1}^T \max(0, I_t) + c_s \sum_{t=1}^T \max(0, D_t - I_t) + c_o \cdot N_{\text{orders}}$$
   *Combines daily holding cost ($c_h = \$0.10/\text{unit-day}$), stockout penalty ($c_s = \$5.00/\text{unit}$), and order processing cost ($c_o = \$50.00/\text{order}$).*

---

## 3. Statistical Significance Testing

To verify whether performance differences between candidate models and baselines are statistically genuine or mere random fluctuations:
- **Wilcoxon Signed-Rank Test**: Applied to paired series-level MAE and MASE scores across the evaluation folds.
- **Diebold-Mariano Test**: Applied to multi-step forecast error trajectories for individual critical series.
- Differences are reported with explicit $p$-values, adhering to scientific rigor.
