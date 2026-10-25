# IDFS Viva Voce Presentation Deck: Slide-by-Slide Outline & Speaker Notes

**Project Title:** Intelligent Demand Forecasting System (IDFS)  
**Institution:** ICFAI Foundation for Higher Education (IcfaiTech), Hyderabad  
**Team Members:** Pamdem Ravi Charan Reddy, Jaiti Vijaya Shravani, Vankayalapati Srujana, Pottipothu V M Krishna Swamy  
**Faculty Mentors:** Dr. Sukanta Das, Dr. Sandeep Kumar Panda  

---

## Slide 1: Title & Team Introduction
- **Visuals:** Project Title, University Logo, Team Names & Enrolment IDs, Mentors.
- **Key Points:**
  - Project Title: Intelligent Demand Forecasting System (IDFS)
  - Team: Pamdem Ravi Charan Reddy (Lead Systems Engineer), Jaiti Vijaya Shravani (Uncertainty & Statistical Modeling), Vankayalapati Srujana (Deep Learning & Monitoring), Pottipothu V M Krishna Swamy (Inventory Simulation)
  - Date: October 2026
- **Speaker Notes (Ravi Charan):**
  > "Good morning respected examiners and mentors. Today our team presents the Intelligent Demand Forecasting System, or IDFS. In this project, we built a production-grade, multi-paradigm machine learning and inventory optimization engine that transforms raw retail sales into calibrated, cost-optimal supply chain procurement decisions."

---

## Slide 2: The Core Problem: Why Modern Retail Forecasting Fails
- **Visuals:** Split comparison diagram (Spreadsheets/Moving Averages vs. Real-world retail shocks).
- **Key Points:**
  - Modern retail planners rely heavily on spreadsheets and 30-day moving averages.
  - Misses non-linear calendar seasonality, promotions, and macroeconomic shocks (e.g., fuel/oil prices).
  - Intermittent demand: 40%+ of retail SKUs have zero sales on most days; classical regression overpredicts.
  - The Decoupling Gap: A point forecast does not provide safety stock, reorder quantities, or risk bounds.
- **Speaker Notes (Ravi Charan):**
  > "The root cause of supply chain failure in retail is not just inaccurate forecasts—it is the decoupling between forecasting and inventory costs. When a store over-orders, holding costs accumulate; when it under-orders, customer stockout penalties are severe. Spreadsheets cannot model this asymmetry."

---

## Slide 3: Research Novelty Stack (N1 to N5)
- **Visuals:** Layered architectural stack diagram showing N1 to N5.
- **Key Points:**
  - **[N1] Cost-Based Inventory Simulation**: Evaluates replenishment orders under asymmetric holding vs. stockout penalties.
  - **[N2] Calibrated Prediction Intervals**: Conformalized Quantile Regression (CQR) delivering finite-sample $\ge 90\%$ coverage.
  - **[N3] Demand-Pattern Router**: Syntetos-Boylan ADI vs. $CV^2$ categorizing intermittent vs. smooth series.
  - **[N4] Surveillance Drift Monitoring**: Rolling error monitoring with automated retrain warnings.
  - **[N5] Explainability & What-If**: Tree-SHAP driver ranking and counterfactual price/promotional shock testing.
- **Speaker Notes (Jaiti Shravani):**
  > "To address these gaps, we formulated a 5-tier novelty stack. Rather than just training an isolated model, our platform integrates cost-optimal inventory policies, distribution-free conformal calibration, pattern-based routing, concept drift surveillance, and counterfactual explainability."

---

## Slide 4: Data Pipeline & Zero-Leakage Protocol
- **Visuals:** Data flow diagram: Ingestion $\to$ Cleaning $\to$ Feature Transformer $\to$ Strict Temporal Split.
- **Key Points:**
  - 36,500 fused transactions across 5 stores and product categories.
  - 44 engineered features: Lags ($t-1, \dots, t-28$), rolling windows ($7, 14, 28$), calendar cyclicals, holidays, promotions.
  - Strict No-Leakage Rules:
    1. Rolling features apply `.shift(1)` before window computation.
    2. Encoders/normalizers fit on training set only.
    3. Sealed 28-day final test window untouched until final benchmark.
- **Speaker Notes (Ravi Charan):**
  > "A primary pitfall in time-series engineering is future data leakage. We enforced rigorous programmatic guarantees: every rolling statistic shifts by at least one time step before aggregation, and all normalizers fit strictly on the training partition. Our test window remained sealed until Phase 7."

---

## Slide 5: Multi-Paradigm Forecaster Suite
- **Visuals:** Table of 10 forecasters grouped by paradigm (Baselines, Statistical, ML, Deep Learning, Intermittent).
- **Key Points:**
  - Baselines: Naive, Seasonal Naive (7-day lag).
  - Classical Statistical: SARIMAX(1,0,0)x(0,1,0)7, Facebook Prophet.
  - Tree-Based ML: Global XGBoost Regressor (44 features, tree depth 5).
  - Deep Learning: PyTorch 12-Step LSTM with memory cell states and dropout.
  - Intermittent Specialists: Croston (SBA) and Teunter-Syntetos-Babai (TSB).
  - Ensembles: Inverse-MAE weighted ensemble and Demand-Pattern Router.
- **Speaker Notes (Vankayalapati Srujana):**
  > "Retail portfolios contain diverse time-series behaviors. No single algorithm dominates every SKU. We implemented ten distinct models across statistical, machine learning, deep learning, and state-space intermittent paradigms."

---

## Slide 6: Demand Pattern Routing (Novelty N3)
- **Visuals:** Syntetos-Boylan 2D quadrant scatter plot (ADI on x-axis, $CV^2$ on y-axis).
- **Key Points:**
  - Average Demand Interval ($\text{ADI} = 1.32$) & Squared Coefficient of Variation ($CV^2 = 0.49$).
  - Smooth $\to$ Global XGBoost / SARIMAX.
  - Erratic $\to$ Seasonal Naive / Ensemble.
  - Intermittent $\to$ Croston SBA.
  - Lumpy $\to$ TSB (separating demand size from demand arrival probability).
- **Speaker Notes (Vankayalapati Srujana):**
  > "Novelty N3 dynamically categorizes series based on arrival intervals and transaction variance. For slow-moving items with intermittent zeros, standard neural networks overpredict; the router directs them to Croston or TSB, reducing WAPE substantially."

---

## Slide 7: Calibrated Uncertainty Quantification (Novelty N2)
- **Visuals:** Graph comparing uncalibrated Gaussian intervals (undercovering) vs. Conformal CQR intervals.
- **Key Points:**
  - Standard Gaussian intervals assume normality; real retail errors have heavy tails and skewness.
  - Quantile XGBoost fits 5th and 95th percentiles via pinball loss.
  - Conformalized Quantile Regression (CQR) applies finite-sample nonconformity score adjustment $\hat{Q}_{1-\alpha}$.
  - Result: Achieved exact empirical coverage of **91.1%** against 90% nominal target.
- **Speaker Notes (Jaiti Shravani):**
  > "Point forecasts are dangerous when placing replenishment orders. Using Conformalized Quantile Regression, we adjust quantile boundaries using nonconformity scores on held-out validation data. This provides mathematical finite-sample coverage guarantees without assuming Gaussian errors."

---

## Slide 8: Inventory Cost Simulation & Order Decisions (Novelty N1)
- **Visuals:** Discrete-event simulation timeline showing On-Hand Stock, Pipeline Orders, and Unmet Demand.
- **Key Points:**
  - Periodic review $(R, S)$ policy with supplier lead time $L = 5$ days.
  - Cost structure: Holding $c_h = \$0.10/\text{unit/day}$, Stockout $c_s = \$5.00/\text{unit}$, Ordering $c_o = \$50/\text{order}$.
  - Dynamic order-up-to level: $S_t = \sum_{k=1}^{R+L} \hat{y}_{t+k} + k_\alpha \sigma$.
  - Conformal-driven policy cuts orders in half while boosting fill rate to 90.6%.
- **Speaker Notes (Krishna Swamy):**
  > "Novelty N1 bridges the gap between forecast and procurement. Our discrete-event simulator tracks on-hand inventory and lead-time pipelines every single day. Using calibrated quantile bounds, the policy orders exactly what is needed, cutting stockout penalties dramatically."

---

## Slide 9: Production Drift Surveillance (Novelty N4)
- **Visuals:** Rolling error chart showing normal operation, error spike, and automated Retrain Alert trigger.
- **Key Points:**
  - Tracks trailing rolling WAPE and MAE across a 14-day sliding window.
  - Triggers `WARNING` at $1.5\times$ baseline error, `CRITICAL` at $2.0\times$.
  - Generates automated retraining recommendations before supply chain stockouts occur.
- **Speaker Notes (Vankayalapati Srujana):**
  > "Models decay in production due to macroeconomic shifts and consumer behavior changes. Novelty N4 provides real-time surveillance. The system flags concept drift as soon as rolling WAPE breaches the threshold, prompting the human planner to retrain."

---

## Slide 10: Model Explainability & What-If Simulation (Novelty N5)
- **Visuals:** Tree-SHAP global importance bar chart + counterfactual What-If promotion slider.
- **Key Points:**
  - Tree-SHAP decomposes predictions into exact additive feature attributions.
  - Key global drivers: 28-day rolling average (43.9%), 14-day rolling average (17.3%), promotional presence (5.6%).
  - What-If Simulator: Counterfactual testing shows $+18.37\%$ volume lift from chain-wide promotion rollout.
- **Speaker Notes (Jaiti Shravani):**
  > "To ensure planners trust the system, we integrated Tree-SHAP. The store manager can see exactly why a forecast is high—for instance, driven by a 28-day baseline trend or promotional uplift—and simulate hypothetical price or promotion changes in real time."

---

## Slide 11: Sealed Test Evaluation Benchmark Results
- **Visuals:** Final benchmark leaderboard table on 28-day held-out test data.
- **Key Points:**
  - **TSB**: WAPE 0.1545, MASE 0.777 (Rank 1).
  - **Prophet**: WAPE 0.1607, MASE 0.808 (Rank 2).
  - **PyTorch-LSTM**: WAPE 0.1797, MASE 0.903 (Rank 4).
  - **XGBoost-Global**: WAPE 0.2392, MASE 1.202 (Rank 9).
  - **Naive Baseline**: WAPE 0.3179, MASE 1.598 (Rank 10).
- **Speaker Notes (Ravi Charan):**
  > "Here are the unsealed 28-day test results. Notice that TSB and Prophet outperformed complex global tree models because our retail SKU mix contains significant intermittency. PyTorch-LSTM demonstrated strong sequence learning, achieving 0.1797 WAPE."

---

## Slide 12: Novelty Ablation Study
- **Visuals:** Ablation matrix table (Variants 0 through 5).
- **Key Points:**
  - Variant 0 (Vanilla Baseline): Cost = \$19,687.87, Fill Rate = 87.7%.
  - Variant 1 (+N1 Cost Sim): Cost = \$18,054.65 (**-\$1,633.22 cost reduction**), Fill Rate = 90.1%.
  - Variant 2 (+N2 Calibrated CQR): Coverage reaches **91.1%** (exact 90% target).
  - Variant 3 (+N3 Demand Router): Forecast WAPE improves from 0.2392 to **0.2121**.
  - Variants 4 & 5: Adds active drift surveillance and full Tree-SHAP explainability.
- **Speaker Notes (Krishna Swamy):**
  > "Our ablation study validates each novelty independently. Simply activating Novelty N1 saves \$1,633.22 in inventory costs per cycle. Adding Conformal CQR fixes the undercoverage issue, and Novelty N3 reduces forecast error by 2.7 percentage points."

---

## Slide 13: Statistical Significance Verification
- **Visuals:** Pairwise Diebold-Mariano test results table with $p$-values.
- **Key Points:**
  - TSB significantly beats Seasonal Naive ($DM = -5.7946, p < 0.0001$).
  - TSB significantly beats XGBoost ($DM = 2.6098, p = 0.0103$).
  - PyTorch-LSTM significantly beats Seasonal Naive ($DM = -2.0646, p = 0.0413$).
  - PyTorch-LSTM significantly beats XGBoost ($DM = 2.4124, p = 0.0175$).
  - Confirms improvements are statistically significant, not random noise.
- **Speaker Notes (Jaiti Shravani):**
  > "To ensure scientific rigor, we conducted Diebold-Mariano and Wilcoxon signed-rank hypothesis tests. Both TSB and PyTorch-LSTM achieved statistically significant improvements over standard baselines and XGBoost with p-values well below 0.05."

---

## Slide 14: System Implementation & Software Architecture
- **Visuals:** Streamlit UI screenshots + SQLite database schema (`runs.db`).
- **Key Points:**
  - 5-page interactive Streamlit portal (`main.py`, `data_explorer.py`, `forecast_studio.py`, `inventory_view.py`, `drift_explain_view.py`).
  - Persistent SQLite storage logging simulations and planner decisions (`APPROVED`, `OVERRIDDEN`).
  - One-click executive HTML and CSV report generator.
  - 117 automated unit and headless smoke tests passing in CI.
- **Speaker Notes (Ravi Charan):**
  > "The entire system is deployed as a reactive multi-page Streamlit portal. Store planners can explore sales trends, adjust forecast horizons, test inventory replenishment orders, and commit review decisions directly to an embedded SQLite database."

---

## Slide 15: Conclusion & Viva Voce Summary
- **Visuals:** Checklist of 9 original project objectives marked 100% achieved.
- **Key Points:**
  - All 9 midterm objectives successfully delivered.
  - Realized \$1,633.22 supply chain savings per review cycle.
  - Calibrated 91.1% interval coverage via Conformal Prediction.
  - Reproducible codebase: fixed seeds, zero leakage, 117 passing automated tests.
  - Ready for live system demonstration.
- **Speaker Notes (Ravi Charan):**
  > "In conclusion, IDFS bridges the divide between machine learning forecasts and commercial supply chain decisions. We now invite the examiners to view our live 5-minute interactive demonstration."
