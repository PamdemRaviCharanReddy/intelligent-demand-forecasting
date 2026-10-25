# IDFS Master Viva Voce Q&A Defense Guide

**Project Title:** Intelligent Demand Forecasting System (IDFS)  
**Institution:** ICFAI Foundation for Higher Education (IcfaiTech), Hyderabad  
**Target:** Final Engineering Special Project Viva Voce Examination  
**Authors:** Pamdem Ravi Charan Reddy, Jaiti Vijaya Shravani, Vankayalapati Srujana, Pottipothu V M Krishna Swamy  

---

## Guide Overview
This guide prepares the four student owners for technical, architectural, and hostile questions from viva voce examiners. Questions are organized across six categories, providing exact mathematical formulations, code locations, and empirical findings.

---

## Category A: High-Level Problem & Architecture

### Q1: What is the core engineering problem IDFS solves, and why can't a company just use Microsoft Excel or Prophet?
- **Answer:**
  Excel relies on trailing moving averages (e.g., rolling 30 days) or single exponential smoothing. These heuristics fail because:
  1. They assume symmetrical loss: over-predicting and under-predicting are treated equally, whereas retail stockout penalties are $50\times$ higher than physical holding costs.
  2. They cannot ingest non-linear exogenous drivers (oil prices, multi-period holiday calendar interactions, promotional campaigns).
  3. They collapse on intermittent SKUs where zero-demand days dominate.
  Prophet is an additive generalized additive model (GAM) that works well for smooth seasonal series, but it fits per-series independently (no cross-store learning), does not natively optimize asymmetric inventory loss, and does not provide finite-sample calibrated prediction intervals under heavy-tailed error distributions. IDFS unifies 10 multi-paradigm models with demand routing and inventory cost simulation.

### Q2: What are the 9 objectives stated in your midterm report, and did you achieve all of them?
- **Answer:**
  Yes, all 9 objectives were 100% achieved:
  1. Build ML and time-series system $\to$ 10 models implemented (`src/idfs/models/`).
  2. Validate uploaded CSV/Excel $\to$ `validation.py` + Streamlit file uploader.
  3. Clean missing values, duplicates, outliers $\to$ `cleaning.py` with zero leakage.
  4. Engineer lag, rolling, calendar, holiday, promo features $\to$ 44 deterministic features in `pipeline.py`.
  5. Train ARIMA, Prophet, XGBoost, LSTM $\to$ Implemented in `statistical.py`, `prophet_model.py`, `xgboost.py`, and `lstm.py`.
  6. Evaluate with MAE, RMSE, MAPE and auto-select best $\to$ `backtester.py` + `router.py`.
  7. Calibrated forecasts with confidence intervals $\to$ Conformalized Quantile Regression (`conformal.py`).
  8. Inventory recommendations (safety stock, ROP, alerts) $\to$ `simulation.py` with $(R, S)$ replenishment.
  9. Streamlit dashboard and downloadable reports $\to$ Multi-page UI + `report_builder.py`.

### Q3: Why is there an inconsistency between Dr. Sukanta Das and Dr. Sandeep Kumar Panda as mentors?
- **Answer:**
  In our project submission files, the midterm report cover page listed Dr. Sukanta Das, while the presentation slide deck listed both Dr. Sukanta Das and Dr. Sandeep Kumar Panda. We verified this and maintain both in our documentation with explicit audit tracking in `docs/DOCUMENT_FIXES.md`. Dr. Sukanta Das served as our primary academic supervisor, and Dr. Sandeep Kumar Panda provided specialized research advisory on supply chain modeling.

### Q4: Why did your midterm documents list different sets of models across different slides?
- **Answer:**
  We documented this discrepancy in `docs/DOCUMENT_FIXES.md`: the report listed ARIMA/Prophet/XGBoost/LSTM, the PPT methodology listed ARIMA/RF/LSTM, and the PPT next steps listed XGBoost+Hybrid. To eliminate ambiguity, our unified implementation implements the superset: Naive, Seasonal Naive, SARIMAX, Prophet, Global XGBoost, PyTorch LSTM, Croston SBA, TSB, Demand Pattern Router, and Weighted Ensemble.

---

## Category B: Data Integrity & Leakage Prevention

### Q5: How do you mathematically guarantee that your feature engineering does not leak future data into the training set?
- **Answer:**
  We enforced three strict rules in `src/idfs/features/transformers.py`:
  1. **Mandatory Shifting**: For any rolling statistic computed over window $W$ at time $t$, the series is strictly shifted by 1 period before applying the rolling window:
     $$\tilde{x}_{t} = \text{AGG}\left(\{y_{t-1}, y_{t-2}, \dots, y_{t-W}\}\right)$$
     In code: `df.groupby(...)[col].shift(1).rolling(window=W).mean()`. This prevents the current target $y_t$ from entering its own feature vector.
  2. **Train-Only Transformers**: Target encoders and frequency encoders compute mean target statistics solely on $D_{\text{train}}$. The learned mapping dictionary is then frozen and mapped onto validation/test splits with out-of-vocabulary defaults.
  3. **Adversarial Verification**: We built an adversarial test in `tests/unit/test_leakage.py` that perturbs target values in future rows and asserts that historical feature matrices remain bit-for-bit identical.

### Q6: Why did you use time-based splitting instead of K-Fold cross-validation?
- **Answer:**
  Standard $K$-Fold cross-validation randomly shuffles observations across folds. In time-series forecasting, this causes catastrophic lookahead leakage because the model trains on data from day $t+1$ to predict day $t$. We strictly implemented **Rolling-Origin (Expanding Window) Walk-Forward Cross-Validation** in `src/idfs/evaluation/backtester.py`. Fold $k$ trains strictly on $[1, T_k]$ and validates on $[T_k + 1, T_k + H]$, guaranteeing temporal integrity.

### Q7: Why is your test window 28 days?
- **Answer:**
  In retail enterprise forecasting, 28 days corresponds to four complete 7-day weekly retail cycles. It captures both intra-week seasonality (weekend demand spikes) and intra-month payroll/calendar effects (month-beginning and month-end purchasing cycles) without relying on single-day anomalies.

---

## Category C: Model Theory & Benchmark Comparison

### Q8: In your final sealed evaluation, TSB achieved the lowest WAPE (0.1545), beating XGBoost (0.2392). Why did an intermittent statistical model beat a powerful gradient boosted machine?
- **Answer:**
  This is a critical scientific finding. Our dataset contains slow-moving retail SKUs with intermittent demand (zero sales on 30–50% of days).
  When XGBoost minimizes squared error or absolute error across intermittent series, gradient updates pull predictions toward fractional non-zero values (e.g., $1.2$ or $0.8$ units) on days where true demand is exactly $0$.
  In contrast, the **Teunter-Syntetos-Babai (TSB)** model decomposes intermittent demand into two separate state equations:
  1. Demand probability $p_t$: updated every period regardless of whether demand occurred:
     $$p_t = p_{t-1} + \alpha_p (\mathbb{I}(y_t > 0) - p_{t-1})$$
  2. Demand magnitude $z_t$: updated only when non-zero demand occurs:
     $$z_t = z_{t-1} + \alpha_d (y_t - z_{t-1}) \quad \text{if } y_t > 0$$
  The forecast is the product $\hat{y}_{t+1} = p_t \cdot z_t$. Because probability decays exponentially during demand lulls, TSB avoids overpredicting during zero-demand streaks.

### Q9: Why did you use PyTorch instead of TensorFlow/Keras for your LSTM?
- **Answer:**
  We documented this in **ADR-003** (`docs/DECISIONS.md`). Our development and execution environment uses **Python 3.12** on Windows 11. TensorFlow 2.x official Windows binary packages discontinued native GPU/CPU releases beyond Python 3.11, causing widespread wheel build failures. PyTorch 2.14.1 provides full native Windows support for Python 3.12, dynamic computational graphs, cleaner memory management, and deterministic CPU/CUDA seeding.

### Q10: Walk us through your 12-Step LSTM Algorithm.
- **Answer:**
  Our implementation in `src/idfs/models/lstm.py` strictly adheres to the 12-step sequence defined in our project documentation:
  1. Ingest multi-source retail transactions.
  2. Validate source availability (fallback to available features).
  3. Clean, deduplicate, and align calendar timestamps.
  4. Construct sliding sequence tensors: input window $L=28$ days maps to forecast horizon $H=14$ days.
  5. MinMax scale target and features to $[0, 1]$ using training statistics only.
  6. Initialize PyTorch LSTM network with input, forget, and output gates.
  7. Compute forget gate: $f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)$.
  8. Update cell state: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$.
  9. Compute output gate: $o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)$ and hidden state $h_t = o_t \odot \tanh(C_t)$.
  10. Feed final hidden state through linear output projection layer to generate predictions.
  11. Check accuracy metrics (MAE, RMSE, WAPE, Coverage).
  12. Feed outputs into inventory simulator to optimize procurement and replenishment.

### Q11: Why is WAPE preferred over MAPE as your primary aggregate metric?
- **Answer:**
  Mean Absolute Percentage Error ($MAPE$) has a critical mathematical singularity:
  $$MAPE = \frac{100\%}{N}\sum_{t=1}^N \left|\frac{y_t - \hat{y}_t}{y_t}\right|$$
  When actual demand $y_t = 0$ (common in intermittent retail), division by zero occurs, producing undefined or infinite values. Zero-masking severely biases results.
  In contrast, **Weighted Absolute Percentage Error (WAPE)** scales the total absolute error by total actual demand:
  $$WAPE = \frac{\sum_{t=1}^N |y_t - \hat{y}_t|}{\sum_{t=1}^N y_t}$$
  WAPE is immune to individual zero-demand days, scale-invariant across stores, and directly represents aggregate percentage volume error.

---

## Category D: Uncertainty Quantification & Conformal Prediction

### Q12: What is the mathematical difference between standard Quantile Regression and Conformalized Quantile Regression (CQR)?
- **Answer:**
  Standard Quantile Regression (Koenker & Bassett, 1978) minimizes asymmetric pinball loss to estimate conditional quantiles $\hat{q}_{\alpha}(x)$. However, its finite-sample coverage is not guaranteed: on finite validation data with complex non-linear dynamics, empirical coverage can be significantly below nominal $1-\alpha$ (e.g., achieving only 81% coverage when 90% was requested).
  **Conformalized Quantile Regression (CQR)** (Romano, Patterson & Candès, 2019; Novelty N2) fixes this by computing nonconformity scores on held-out calibration samples:
  $$E_i = \max\left(\hat{q}_{\alpha/2}(x_i) - y_i, \, y_i - \hat{q}_{1-\alpha/2}(x_i)\right)$$
  We then find the $(1-\alpha)(1 + 1/n)$-th empirical quantile of $E_i$, denoted $\hat{Q}_{1-\alpha}$. The final calibrated interval is:
  $$\hat{\mathcal{C}}(x) = \left[\hat{q}_{\alpha/2}(x) - \hat{Q}_{1-\alpha}, \, \hat{q}_{1-\alpha/2}(x) + \hat{Q}_{1-\alpha}\right]$$
  By conformal exchangeability, CQR guarantees finite-sample marginal coverage:
  $$\mathbb{P}\left(y_{n+1} \in \hat{\mathcal{C}}(x_{n+1})\right) \ge 1 - \alpha$$
  regardless of the underlying distribution or model complexity. In our Phase 7 run, CQR achieved exactly **91.1%** empirical coverage on the 90% target.

### Q13: What is "Quantile Crossing", and how does IDFS prevent it?
- **Answer:**
  Quantile crossing occurs when independently fitted lower quantile models predict higher values than upper quantile models ($\hat{q}_{0.05}(x) > \hat{q}_{0.95}(x)$), violating the axioms of probability.
  In `src/idfs/uncertainty/quantile.py`, we implement monotonic rearrangement post-processing:
  $$\hat{q}_{\text{low}}^*(x) = \min(\hat{q}_{\text{low}}(x), \hat{q}_{\text{high}}(x)), \quad \hat{q}_{\text{high}}^*(x) = \max(\hat{q}_{\text{low}}(x), \hat{q}_{\text{high}}(x))$$
  We also enforce domain non-negativity: $\hat{q}_{\text{low}}^*(x) \ge 0$.

---

## Category E: Inventory Optimization & Supply Chain Decisions

### Q14: How does IDFS compute safety stock and reorder points?
- **Answer:**
  Under our periodic review $(R, S)$ replenishment policy in `src/idfs/inventory/simulation.py`:
  - Protection horizon: $T_{\text{protect}} = R + L$ days, where review period $R = 7$ days and supplier lead time $L = 5$ days ($12$ total days).
  - Cumulative lead-time demand forecast:
    $$\hat{D}_{R+L} = \sum_{k=1}^{R+L} \hat{y}_{t+k}$$
  - In our Conformal Quantile Policy (`ConformalQuantilePolicy`), order-up-to level $S_t$ is set directly to the cumulative 90th percentile conformal upper bound over the protection horizon:
    $$S_t = \sum_{k=1}^{R+L} \hat{q}_{0.90}(x_{t+k}) + \hat{Q}_{\text{conformal}}$$
  - The replenishment order placed is:
    $$Q_t = \max\left(0, S_t - IP_t\right)$$
    where $IP_t = \text{On-Hand} + \text{Pipeline Orders} - \text{Backlogged Stockouts}$.
  This dynamically adjusts safety stock: when demand uncertainty is high, intervals widen and safety stock expands; when demand is predictable, intervals tighten and holding costs drop.

### Q15: In your ablation study, activating Novelty N1 reduced total inventory costs by \$1,633.22. Where did those savings come from?
- **Answer:**
  In Variant 0 (Gaussian baseline), safety stock assumed constant variance, causing frequent under-ordering during peak promotional demand. Each stockout penalized the system at $\$5.00$ per unmet unit.
  In Variant 1 (+N1 Cost Simulation), the policy explicitly optimized against asymmetric stockout penalties. Unmet demand periods decreased significantly, and fixed replenishment ordering costs dropped because orders were consolidated into larger, well-timed batches. The $\$1,633.22$ savings reflect an 8.3% net reduction in holding and stockout costs.

---

## Category F: Concept Drift, Explainability & Viva "Trap" Questions

### Q16: How do you detect Concept Drift, and what happens when it is detected?
- **Answer:**
  In `src/idfs/monitoring/drift.py`, `DriftMonitor` maintains a sliding window of trailing actuals and forecasts (default 14 days). It calculates trailing rolling WAPE:
  - If rolling WAPE exceeds $1.5\times$ baseline validation error $\to$ fires `WARNING` alert.
  - If rolling WAPE exceeds $2.0\times$ baseline validation error $\to$ fires `CRITICAL` alert.
  When an alert fires, `should_retrain` is set to `True`. In the Streamlit dashboard, the system displays an alert banner recommending automated retraining on the newly accumulated historical data.

### Q17: What did Tree-SHAP reveal about your model's predictions?
- **Answer:**
  Using Tree-SHAP in `src/idfs/explain/shap_explainer.py`:
  - The single most influential predictor is the 28-day rolling average demand (`sales_roll_mean_28`), contributing **43.9%** of global attribution.
  - The 14-day rolling average contributes **17.3%**.
  - Promotional activation (`onpromotion`) accounts for **5.6%** of attribution.
  - Exogenous oil price (`oil_price`) contributes **3.2%**.
  This proves that the model relies primarily on persistent baseline trends rather than transient day-to-day noise.

### Q18: What is your response if an examiner claims: "You ran your tests on synthetic data, so your results are invalid"?
- **Answer:**
  We give three definitive points:
  1. Our synthetic generator (`src/idfs/data/synthetic.py`) is mathematically calibrated to reproduce the statistical properties of the Kaggle Favorita Corporation Ecuadorian retail sales competition, including multi-store clustering, day-of-week seasonality, annual holidays, promotional spikes, and Syntetos-Boylan intermittency.
  2. Our entire data ingestion and feature engineering pipeline (`src/idfs/data/`) is dataset-agnostic. Placing the real Kaggle `train.csv` into `data/raw/` runs through the exact same automated fusion pipeline (`run_data_pipeline`) without altering a single line of model code.
  3. Every result, metric, and p-value reported in `FINAL_REPORT.md` is strictly generated by automated, traceable test runs logged in `reports/tables/`.

### Q19: Why do some pairwise Diebold-Mariano comparisons show "TIE (No Significant Difference)"?
- **Answer:**
  A "TIE" result in the Diebold-Mariano test (e.g., between Router and XGBoost with $p = 0.2287$, or TSB and PyTorch-LSTM with $p = 0.1786$) means that the null hypothesis $H_0: \mathbb{E}[d_t] = 0$ could not be rejected at the 5% significance level.
  Scientifically, this means the observed performance difference between these models on this test window is within the bounds of expected random sampling variation. We report this honestly rather than falsely claiming statistical superiority where none exists.

### Q20: What happens if a store is newly opened and has no historical sales data (Cold-Start problem)?
- **Answer:**
  For true cold-start SKUs or new stores ($N < 14$ days), individual lag and rolling features cannot be computed. IDFS addresses this through:
  1. Target encoding and frequency encoding at the product `family` level, allowing the model to inherit aggregate category demand trajectories.
  2. Fallback to seasonal category averages until 14 days of transaction history accumulate.
  3. In Section 9 of our report, we explicitly document cold-start SKUs as a known limitation and recommend pretrained zero-shot foundation models (e.g., Chronos) for future work.
