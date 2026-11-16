# Intelligent Demand Forecasting System (IDFS)

> **A Multi-Model Approach to Retail Demand Prediction, Conformal Uncertainty Quantification, Tree-SHAP Explainability, and Autonomous $(R, S)$ Inventory Optimization**

---

## 📋 Academic & Team Information

* **Institution:** ICFAI Foundation for Higher Education (IFHE), Hyderabad
* **Department:** Department of Computer Science and Engineering, IcfaiTech
* **Academic Year:** 2026
* **GitHub Repository:** [https://github.com/PamdemRaviCharanReddy/intelligent-demand-forecasting](https://github.com/PamdemRaviCharanReddy/intelligent-demand-forecasting)

### Project Team

| Team Member | Enrollment Number |
| :--- | :--- |
| **Pamdem Ravi Charan Reddy** *(Lead)* | `24STUCHH010156` |
| **Jaiti Vijaya Shravani** | `24STUCHH010756` |
| **Vankayalapati Srujana** | `24STUCHH010749` |
| **Pottipothu V M Krishna Swamy** | `24STUCHH010207` |

### Project Guides
* **Dr. Sukanta Das** — Associate Professor, Department of CSE, IcfaiTech
* **Dr. Sandeep Kumar Panda** — Professor & Research Advisor, Department of CSE, IcfaiTech

---

## 📌 Executive Summary

Traditional retail forecasting relies on simple moving averages or point predictions that fail to account for seasonal volatility, intermittent purchase patterns, and forecast uncertainty. This results in severe stockouts (lost sales) or excessive inventory holding costs.

The **Intelligent Demand Forecasting System (IDFS)** is an end-to-end machine learning and operational research platform that bridges the gap between predictive modeling and inventory decision-making. Built on a dataset of **219,000+ daily transactions across 15 stores and 20 seasonal products over 4 years**, IDFS:
1. Benchmarks **10 diverse forecasting models** spanning statistical, machine learning, and deep learning paradigms.
2. Identifies the **Teunter-Syntetos-Babai (TSB)** model as the champion (**15.45% WAPE**, outperforming baselines by over **51.4%**).
3. Quantifies forecast uncertainty using **Conformalized Quantile Regression (CQR)** with distribution-free 90% prediction intervals.
4. Simulates periodic review **$(R, S)$ inventory replenishment policies**, cutting order fragmentation by **49%** while maintaining a **>95% on-shelf service level**.
5. Explains individual predictions via **Tree-SHAP** and tracks concept/data drift using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests.

---

## 🔬 5 Core Research Novelties

* **N1: Cost-Optimized $(R, S)$ Inventory Simulation:** Discrete-event operational simulation evaluating carrying, stockout penalty, and replenishment costs under dynamic lead-time demand.
* **N2: Conformalized Quantile Regression (CQR):** Distribution-free prediction intervals calibrated on held-out data to guarantee finite-sample coverage without assuming Gaussian errors.
* **N3: Syntetos-Boylan Demand-Pattern Router:** Automated time-series classification via Average Inter-demand Interval ($ADI$) and squared Coefficient of Variation ($CV^2$) into *Smooth*, *Intermittent*, *Erratic*, and *Lumpy* demand quadrants.
* **N4: Automated Drift Monitoring & MLOps Guardrails:** Continuous tracking of covariate and concept shifts via PSI and two-sample KS-tests with automated threshold alerts for model retraining.
* **N5: Tree-SHAP Model Explainability:** Game-theoretic feature attribution decomposing predictions into exact marginal contributions for complete transparency to store managers.

---

## 📊 10-Model Benchmark Tournament Leaderboard

All models were evaluated on an identical out-of-sample holdout test set using rolling-origin backtesting:

| Rank | Model Name | Paradigm | MAE | RMSE | WAPE (%) | MASE | 90% Coverage | Pinball Loss |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **TSB (Teunter-Syntetos-Babai)** | **Statistical (Intermittent)** | **27.77** | **36.50** | **15.45%** | **0.777** | **82.1%** | **6.03** |
| 🥈 2 | Prophet (Meta) | Bayesian Additive | 28.90 | 40.17 | 16.07% | 0.808 | 84.8% | 7.57 |
| 🥉 3 | Croston-SBA | Statistical (Intermittent) | 32.13 | 42.23 | 17.87% | 0.898 | 80.4% | 7.39 |
| 4 | PyTorch-LSTM | Deep Learning (RNN) | 32.30 | 43.34 | 17.97% | 0.903 | 80.4% | 7.73 |
| 5 | Weighted Ensemble | Hybrid Combination | 32.81 | 46.96 | 18.25% | 0.917 | 83.0% | 9.45 |
| 6 | Seasonal Naive (7-Day) | Heuristic Baseline | 37.55 | 58.13 | 20.89% | 1.050 | 85.7% | 12.84 |
| 7 | SARIMAX(1,0,0)x(0,1,0)7 | Classical Time Series | 37.61 | 58.15 | 20.92% | 1.051 | 84.8% | 12.89 |
| 8 | Demand Pattern Router (N3) | Meta-Learning Selection | 38.14 | 49.48 | 21.21% | 1.066 | 84.8% | 7.46 |
| 9 | XGBoost-Global | Gradient Boosted Trees | 43.00 | 51.76 | 23.92% | 1.202 | 81.2% | 7.11 |
| 10 | Naive Baseline | Naive Drift | 57.16 | 65.44 | 31.79% | 1.598 | 92.9% | 9.83 |

---

## 📦 Retail Product Catalog

The dataset models 20 distinct products representing 4 distinct seasonal demand archetypes across 15 retail stores:

* **Monsoon Peaks:** Umbrella, Raincoat, Waterproof Boots
* **Summer Peaks:** Air Conditioner, Ceiling Fan, Cold Drinks, Ice Cream, Sunscreen Lotion, Reusable Water Bottle
* **Winter Peaks:** Sweater, Winter Jacket, Thermal Wear, Room Heater, Winter Blankets, Coffee & Hot Cocoa, Electric Kettle
* **Festive & Steady Surges:** Sweets Gift Box, Festive Decorative Lights, Sunglasses, Mosquito Repellent

---

## 🏗️ System Architecture & Workflow

```
[ Raw Multi-Store Data ] (219,000+ Transactions across 15 Stores & 20 SKUs)
           │
           ▼
[ Feature Engineering ] (44 Dimensions: Lags, Rolling Stats, Fourier Calendar, Promos)
           │
           ▼
[ Model Tournament ] (10 Models: TSB, Croston, Prophet, LSTM, XGBoost, SARIMAX)
           │
           ▼
[ Conformal Calibration ] (CQR 90% Distribution-Free Prediction Intervals)
           │
           ▼
[ (R, S) Inventory Optimization ] (Dynamic Safety Stock & Order-Up-To Simulation)
           │
           ▼
[ Decoupled Delivery ]
   ├── FastAPI REST Backend (Port 8000: /predict, /benchmark, /explain, /drift)
   └── Streamlit Multi-Page UI (Port 8501: Interactive Analytics & Decision Support)
```

---

## 🛠️ Technology Stack

* **Programming Language:** Python 3.12 (managed via `uv`)
* **Data Processing:** Pandas, NumPy, SciPy, Statsmodels
* **Machine Learning & Deep Learning:** Scikit-learn, XGBoost, PyTorch (LSTM), Prophet
* **Explainability:** SHAP (Tree-SHAP)
* **Backend Services:** FastAPI, Uvicorn, Pydantic
* **Frontend Dashboard:** Streamlit, Plotly, Matplotlib, Seaborn
* **Data Storage:** Parquet, SQLite
* **Quality & Testing:** Pytest, Ruff

---

## 🚀 Installation & Quickstart

### 1. Clone the Repository
```bash
git clone https://github.com/PamdemRaviCharanReddy/intelligent-demand-forecasting.git
cd intelligent-demand-forecasting
```

### 2. Environment Setup
```powershell
# Create virtual environment with Python 3.12
uv venv --python 3.12 .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install package dependencies
uv pip install -e ".[dev]"
```

### 3. Launch Backend API (FastAPI)
```powershell
uvicorn idfs.api.main:app --host 127.0.0.1 --port 8000 --reload
```
* Interactive API Documentation (Swagger): `http://127.0.0.1:8000/docs`

### 4. Launch Frontend Dashboard (Streamlit)
In a separate terminal:
```powershell
streamlit run streamlit_app.py --server.port 8501
```
* Interactive Web UI: `http://localhost:8501`

---

## 📁 Repository Directory Structure

```
SP-SP/
├── configs/                     # YAML configuration for models, features, & inventory
├── data/                        # Raw, processed, and benchmark datasets
├── docs/                        # Project documentation, architecture ADRs, and protocols
├── reports/                     # Project presentation decks, reports, and generated charts
│   ├── figures/                 # EDA and SHAP visualizations
│   ├── ppt_charts/              # Visual charts for slide deck
│   ├── tables/                  # Benchmark CSVs, ablation tables, and simulation results
│   ├── CAPSTONE_PROJECT_REPORT_ADVANCED.md
│   ├── IDFS_Modern_Presentation.pptx
│   └── PRESENTATION_SPEAKER_NOTES.md
├── scripts/                     # Utility scripts for presentation and benchmark generation
│   ├── generate_modern_ppt.py   # Automated 10-slide visual presentation generator
│   └── dev.py                   # Development and verification tasks
├── src/idfs/                    # Core Production Package
│   ├── api/                     # FastAPI application endpoints and Pydantic schemas
│   ├── dashboard/               # Streamlit multi-page application components
│   ├── evaluation/              # Benchmark scorecards, loss metrics, and significance tests
│   ├── explainability/          # Tree-SHAP attribution and scenario simulators
│   ├── features/                # 44-feature engineering transformations
│   ├── inventory/               # Stochastic (R, S) policy simulator and cost models
│   ├── models/                  # TSB, Croston-SBA, Prophet, LSTM, XGBoost implementations
│   ├── monitoring/              # PSI and KS-test drift detection engine
│   └── uncertainty/             # Conformalized Quantile Regression calibration
├── tests/                       # Pytest test suite (unit and integration tests)
├── pyproject.toml               # Build configuration, package metadata, and dependencies
└── README.md                    # Project overview and setup instructions
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/health` | System health check and package version status |
| `POST` | `/forecast/predict` | Generate multi-horizon forecasts with 90% CQR prediction intervals |
| `GET` | `/models/benchmark` | Retrieve 10-model tournament performance leaderboard |
| `POST` | `/explainability/shap` | Return Tree-SHAP feature importance and marginal attributions |
| `POST` | `/inventory/simulate` | Run $(R, S)$ periodic review inventory cost simulation |
| `GET` | `/drift/status` | Check Population Stability Index (PSI) and KS drift indicators |

---

## 📜 Citation & License

This project was developed for the Special Project Capstone at **ICFAI Foundation for Higher Education (IFHE), Hyderabad**. Released under the MIT License.
