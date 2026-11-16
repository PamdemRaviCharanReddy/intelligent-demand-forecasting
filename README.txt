================================================================================
INTELLIGENT DEMAND FORECASTING SYSTEM (IDFS)
A Multi-Model Approach to Retail Demand Prediction, Conformal Uncertainty 
Quantification, Tree-SHAP Explainability, and Autonomous (R, S) Inventory Optimization
================================================================================

1. ACADEMIC & TEAM METADATA
--------------------------------------------------------------------------------
Institution       : ICFAI Foundation for Higher Education (IFHE), Hyderabad
Department        : Department of Computer Science & Engineering, IcfaiTech
Academic Year     : 2026
GitHub Repository : https://github.com/PamdemRaviCharanReddy/intelligent-demand-forecasting

PROJECT TEAM:
1. Pamdem Ravi Charan Reddy (Lead)  |  Enrollment No: 24STUCHH010156
2. Jaiti Vijaya Shravani            |  Enrollment No: 24STUCHH010756
3. Vankayalapati Srujana            |  Enrollment No: 24STUCHH010749
4. Pottipothu V M Krishna Swamy     |  Enrollment No: 24STUCHH010207

PROJECT GUIDES:
- Dr. Sukanta Das         : Associate Professor, Dept. of CSE, IcfaiTech
- Dr. Sandeep Kumar Panda : Professor & Research Advisor, Dept. of CSE, IcfaiTech


2. EXECUTIVE SUMMARY & PROBLEM STATEMENT
--------------------------------------------------------------------------------
Retailers face an asymmetric operational dilemma: overstocking causes heavy capital
lockup and inventory holding costs, while understocking leads to stockouts, lost 
revenue, and damaged customer retention. Traditional forecasting tools (e.g., 
moving averages or single point estimates) fail to manage intermittent demand, 
promotional spikes, and forecast uncertainty.

The Intelligent Demand Forecasting System (IDFS) bridges machine learning with 
supply chain operations research. Built on 219,000+ daily transactions across 15 
stores and 20 seasonal products spanning 4 years (2013-2017), the system:
- Evaluates 10 diverse models across statistical, ML, and deep learning paradigms.
- Employs Teunter-Syntetos-Babai (TSB) as the champion model (15.45% WAPE).
- Calibrates distribution-free 90% prediction intervals via Conformal Prediction.
- Simulates periodic review (R, S) inventory replenishment to minimize costs.
- Provides Tree-SHAP interpretability and real-time MLOps drift monitoring.


3. 5 CORE RESEARCH NOVELTIES
--------------------------------------------------------------------------------
[N1] Cost-Optimized (R, S) Inventory Simulation:
     A discrete-event simulation tracking holding costs ($1.50/unit), stockout 
     penalties ($15.00/unit), and ordering overhead ($50.00/order) under lead-time 
     uncertainty, cutting order fragmentation by 49%.

[N2] Conformalized Quantile Regression (CQR):
     Distribution-free uncertainty intervals calibrated on held-out data to 
     guarantee finite-sample 90% empirical coverage without assuming Gaussian errors.

[N3] Syntetos-Boylan Demand-Pattern Router:
     Meta-classification using Average Inter-demand Interval (ADI) and squared 
     Coefficient of Variation (CV^2) to categorize SKUs into Smooth, Intermittent, 
     Erratic, or Lumpy demand, routing each to its optimal model family.

[N4] Automated Drift Monitoring & MLOps Guardrails:
     Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests 
     to detect covariate and concept drift with automated retraining alerts.

[N5] Tree-SHAP Model Explainability:
     Game-theoretic feature attribution decomposing predictions into exact marginal 
     contributions, showing 28-day rolling average (43.9%) and promotions (5.6%) 
     drive over 60% of forecast decisions.


4. 10-MODEL BENCHMARK TOURNAMENT LEADERBOARD
--------------------------------------------------------------------------------
Rank | Model Name              | Paradigm       | MAE   | RMSE  | WAPE   | MASE | Cov90%
-----+-------------------------+----------------+-------+-------+--------+------+-------
 1   | TSB (Champion)          | Statistical    | 27.77 | 36.50 | 15.45% | 0.78 | 82.1%
 2   | Prophet (Meta)          | Bayesian       | 28.90 | 40.17 | 16.07% | 0.81 | 84.8%
 3   | Croston-SBA             | Statistical    | 32.13 | 42.23 | 17.87% | 0.90 | 80.4%
 4   | PyTorch-LSTM            | Deep Learning  | 32.30 | 43.34 | 17.97% | 0.90 | 80.4%
 5   | Weighted Ensemble       | Hybrid         | 32.81 | 46.96 | 18.25% | 0.92 | 83.0%
 6   | Seasonal Naive (7-Day)  | Baseline       | 37.55 | 58.13 | 20.89% | 1.05 | 85.7%
 7   | SARIMAX(1,0,0)x(0,1,0)7 | Time Series    | 37.61 | 58.15 | 20.92% | 1.05 | 84.8%
 8   | Demand Pattern Router   | Meta-Learning  | 38.14 | 49.48 | 21.21% | 1.07 | 84.8%
 9   | XGBoost-Global          | Gradient Boost | 43.00 | 51.76 | 23.92% | 1.20 | 81.2%
 10  | Naive Baseline          | Heuristic      | 57.16 | 65.44 | 31.79% | 1.60 | 92.9%


5. DATASET & 20 SEASONAL PRODUCT CATALOG
--------------------------------------------------------------------------------
Total Transactions : 219,000+ records
History            : 4 years (daily granularity)
Store Footprint    : 15 retail stores across multiple geographic clusters

Product Catalog by Seasonality:
- Monsoon Demand : Umbrella, Raincoat, Waterproof Boots
- Summer Demand  : Air Conditioner, Ceiling Fan, Cold Drinks, Ice Cream,
                   Sunscreen Lotion, Reusable Water Bottle
- Winter Demand  : Sweater, Winter Jacket, Thermal Wear, Room Heater,
                   Winter Blankets, Coffee & Hot Cocoa, Electric Kettle
- Festive/Yearly : Sweets Gift Box, Festive Decorative Lights, Sunglasses,
                   Mosquito Repellent


6. SYSTEM ARCHITECTURE & PIPELINE
--------------------------------------------------------------------------------
1. Data Sources         : 15 Stores x 20 Products historical transactions (SQLite/Parquet)
2. Feature Pipeline     : 44 engineered features (lags, rolling stats, cyclical calendar)
3. Model Tournament     : 10 parallel statistical, ML, and deep learning models
4. Conformal Quantile   : 90% prediction intervals calibrated via CQR
5. Inventory Engine     : Discrete-event periodic review (R, S) policy simulator
6. Decoupled Delivery   : FastAPI REST backend (Port 8000) + Streamlit UI (Port 8501)


7. TECHNOLOGY STACK
--------------------------------------------------------------------------------
- Core Language       : Python 3.12 (managed via uv)
- Backend API         : FastAPI, Uvicorn, Pydantic
- Frontend UI         : Streamlit, Plotly, Matplotlib, Seaborn
- Machine Learning    : Scikit-learn, XGBoost, PyTorch (LSTM), Prophet
- Statistical Models  : Statsmodels, custom TSB & Croston implementations
- Explainability      : SHAP (Tree-SHAP)
- Storage & Data      : SQLite, Apache Parquet, Pandas, NumPy
- Code Quality        : Pytest, Ruff


8. HOW TO RUN THE SYSTEM
--------------------------------------------------------------------------------
Step 1: Setup Environment
    uv venv --python 3.12 .venv
    .venv\Scripts\Activate.ps1
    uv pip install -e ".[dev]"

Step 2: Start Backend REST API (Port 8000)
    uvicorn idfs.api.main:app --host 127.0.0.1 --port 8000 --reload
    * Interactive API Documentation: http://127.0.0.1:8000/docs

Step 3: Start Frontend Streamlit Dashboard (Port 8501)
    streamlit run streamlit_app.py --server.port 8501
    * Interactive Dashboard: http://localhost:8501


9. API ENDPOINTS SUMMARY
--------------------------------------------------------------------------------
GET   /health              : System status and package version
POST  /forecast/predict    : Generate demand forecast with 90% CQR intervals
GET   /models/benchmark    : Retrieve 10-model tournament leaderboard
POST  /explainability/shap : Return Tree-SHAP feature importance attributions
POST  /inventory/simulate  : Run (R, S) periodic review inventory cost simulation
GET   /drift/status        : Return Population Stability Index (PSI) drift status

================================================================================
END OF README
================================================================================
