# Team Viva Guide: Plain-Language Explanations & Examiner Q&A

**Project Title**: Intelligent Demand Forecasting System (IDFS)  
**Authors**: Pamdem Ravi Charan Reddy, Jaiti Vijaya Shravani, Vankayalapati Srujana, Pottipothu V M Krishna Swamy  
**Academic Institution**: ICFAI Foundation for Higher Education (IcfaiTech), Hyderabad  
**Mentors**: Dr. Sukanta Das, Dr. Sandeep Kumar Panda `[VERIFY]`

---

## 1. Executive Summary in Plain English

Imagine running a major supermarket chain. Every evening, you have to decide how many cartons of milk, bags of rice, or bottles of cooking oil to order from suppliers.
- **If you order too few**: Your shelves are empty when customers arrive. You lose revenue and customers leave frustrated (**Stockout Cost**).
- **If you order too many**: Your warehouse fills up, items expire or get damaged, and working capital is locked up (**Holding Cost**).

Most store managers currently use basic moving averages or Excel spreadsheets. These fail because demand changes when promotions run, during holidays, or with paydays. Furthermore, standard machine learning projects stop after producing a forecast number (e.g. "Tomorrow's demand is 120 units"), without telling the manager **how many units to actually purchase** considering supplier delivery delays and demand uncertainty.

**What IDFS does**:
1. Ingests sales, store characteristics, promotions, holidays, and economic factors.
2. Trains a range of models (from baselines to XGBoost, LSTM, and ARIMA) on identical historical walk-forward folds.
3. Automatically categorizes demand patterns (smooth vs. intermittent) and produces **prediction intervals** (e.g., "Demand will be between 105 and 138 units with 95% confidence").
4. Feeds that uncertainty directly into an **Inventory Simulation** that computes safety stock and recommends order quantities to minimize total inventory costs.
5. Monitors future demand in real-time to alert the manager if consumer behavior shifts (**Concept Drift**).

---

## 2. Walkthrough of Every Code Module

### A. Data Layer (`src/idfs/data/`)
- **What it is**: Ingestion, validation, and multi-source fusion.
- **Why it exists**: Raw sales CSVs contain formatting errors, missing calendar days, negative sales (returns), or disconnected holiday lists. This module cleans and merges them into a single coherent table.
- **Key Functionality**: Graceful fallback. If oil prices or holiday lists are missing, it does not crash; it logs a warning and proceeds with the available sales data.

### B. Feature Engineering (`src/idfs/features/`)
- **What it is**: Calculates historical lags ($y_{t-1}, y_{t-7}$), rolling averages, and calendar flags (day-of-week, distance to holiday).
- **Why it exists**: Machine learning models like XGBoost cannot understand dates directly; they need numerical signals.
- **The Golden Rule (Zero Leakage)**: A rolling average for day $t$ can **never** see day $t$'s sales. We always call `.shift(1)` before computing rolling statistics!

### C. Models Layer (`src/idfs/models/`)
- **What it is**: Contains all forecasters:
  - `NaiveForecaster`: Repeats yesterday's value (sanity check).
  - `SeasonalNaiveForecaster`: Repeats last week's value (e.g. last Monday's sales predict next Monday).
  - `XGBoost`: Global gradient-boosted decision trees learning patterns across all stores and items simultaneously.
  - `LSTM`: Long Short-Term Memory recurrent neural network capturing sequential patterns.
  - `ARIMA / SARIMA & Prophet`: Classical statistical time-series models.
  - `Croston / TSB`: Specialized models for intermittent items that have mostly zeros with occasional purchases.
- **Why it exists**: No single model wins on every product. Smooth fast-moving items behave differently from slow-moving spare parts.

### D. Evaluation Layer (`src/idfs/evaluation/`)
- **What it is**: Computes MAE, RMSE, WAPE, and MASE, and runs the walk-forward backtester.
- **Why it exists**: Random train/test splits cheat by predicting past sales using future data. Our walk-forward backtester slides an expanding window through time, exactly mimicking real daily operations.

### E. Uncertainty & Prediction Intervals (`src/idfs/uncertainty/`)
- **What it is**: Produces upper and lower bounds using Quantile Regression and Conformal Calibration.
- **Why it exists**: A single number (point forecast) is not enough to calculate safety stock. We need to know the range of likely outcomes.

### F. Inventory Engine (`src/idfs/inventory/`)
- **What it is**: Periodic Review $(R, S)$ policy and daily simulation engine.
- **Why it exists**: Translates the forecast into financial numbers: how much holding cost and stockout penalty did our forecast save compared to a basic moving average?

### G. Monitoring Layer (`src/idfs/monitoring/`)
- **What it is**: Drift detector tracking rolling forecast errors.
- **Why it exists**: When inflation strikes or a new competitor opens nearby, historical models degrade. This module flags when it is time to retrain.

### H. Application & Reporting (`src/idfs/app/`, `src/idfs/reporting/`)
- **What it is**: Streamlit interactive web application and downloadable executive reports.
- **Why it exists**: Non-technical store planners need clean interactive charts, alert cards, and one-click order recommendations.

---

## 3. Likely Viva Questions & Winning Defense Answers

### Q1: "How did you prevent data leakage in your feature engineering and cross-validation?"
> **Examiner's trap**: They suspect your rolling averages or scaling leaked future test data into training folds.  
> **Your Answer**:  
> "We implemented temporal precedence by construction in two ways. First, all rolling statistical features apply a strict `shift(1)` before calculating window means and standard deviations, ensuring day $t$ features only contain information from $t-1$ and earlier. We even built an automated leakage test in `tests/unit/` that spikes future actuals to prove past features do not shift. Second, we strictly avoided standard $k$-fold cross-validation. Instead, we used a rolling-origin walk-forward backtest across 3 expanding temporal folds, keeping a final 28-day test period locked until the final evaluation."

### Q2: "Why didn't you use MAPE as your headline evaluation metric?"
> **Examiner's trap**: Checking if you understand mathematical edge cases in retail data.  
> **Your Answer**:  
> "Retail sales data frequently contains zero-sales days, especially for intermittent product categories. Calculating MAPE involves dividing by the actual value $y_t$. When $y_t = 0$, MAPE divides by zero and explodes to infinity. While we report a zero-masked MAPE for historical comparison with the mid-term report, our headline metrics are **WAPE** (Weighted Absolute Percentage Error) and **MASE** (Mean Absolute Scaled Error), which remain mathematically well-behaved and scale-free across all product types."

### Q3: "Why did you use PyTorch instead of TensorFlow for the LSTM when your early report mentioned TensorFlow?"
> **Examiner's trap**: Testing if you understand platform constraints and software architecture.  
> **Your Answer**:  
> "We formalized this in Architecture Decision Record ADR-003. TensorFlow has deprecated native Windows GPU support in recent versions and exhibits severe binary incompatibilities with modern Python 3.12 runtimes on Windows. PyTorch 2.x provides stable native Windows execution, detects our host NVIDIA RTX 3050 Laptop GPU out of the box, and allows clean object-oriented sequence modeling. Furthermore, our abstract `Forecaster` base class completely encapsulates the model, ensuring the external interface, loss tracking, and gate logic (input, forget, output gates) remain identical regardless of the underlying tensor library."

### Q4: "Public datasets like Kaggle Favorita don't have inventory or lead-time data. How can you claim inventory cost savings?"
> **Examiner's trap**: Accusing you of fabricating warehouse numbers.  
> **Your Answer**:  
> "We explicitly disclose in our assumptions that public datasets record point-of-sale transactions rather than warehouse telemetry. To evaluate operational value, we built a parameterized discrete-event simulation engine with realistic, documented assumptions: lead times of 2 to 7 days, holding cost of \$0.10 per unit per day, and a stockout penalty of \$5.00 per unit. We ran both our forecast-driven periodic review policy and a benchmark moving-average policy through the exact same demand realization and lead-time delays. This proves the relative cost savings of forecast accuracy under controlled, transparent conditions."

### Q5: "What is the primary research novelty of your project beyond standard Kaggle competitions?"
> **Examiner's trap**: Suggesting your work is just a standard XGBoost script.  
> **Your Answer**:  
> "Kaggle competitions optimize exclusively for isolated point metrics like RMSLE. Our project makes five specific contributions addressing the research gaps in our literature review:
> 1. **N1 Cost-Based Inventory Simulation**: Directly connecting predictions to safety stock, order-up-to levels, and financial cost comparisons against empirical baselines.
> 2. **N2 Calibrated Prediction Intervals**: Combining quantile regression with split conformal calibration to guarantee empirical coverage.
> 3. **N3 Demand-Pattern Router**: Using Syntetos-Boylan ADI/$CV^2$ classification to route intermittent series to Croston/TSB and smooth series to XGBoost.
> 4. **N4 Drift Monitoring**: Real-time rolling error tracking that signals when structural consumer shifts necessitate retraining.
> 5. **N5 Explainability & What-If Simulation**: Tree-SHAP attribution plots and interactive pricing/promotional scenario testing for supply chain planners."
