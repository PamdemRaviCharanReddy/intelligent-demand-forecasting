# Intelligent Demand Forecasting System (IDFS)

An uncertainty-aware, cost-optimised forecasting and inventory decision system designed for retail supply chains.

## Overview

IDFS bridges the critical gap between demand forecasting and operational inventory replenishment. Standard retail forecasting models optimize statistical error (e.g., RMSE, MAE) in isolation, yet inventory decisions require calibrated uncertainty quantification and asymmetric cost management (holding vs. stockout vs. ordering costs).

### Core Capabilities & Research Novelties
- **N1 Cost-Based Inventory Simulation**: Translates demand forecasts and prediction intervals directly into inventory policies (safety stock, reorder point, periodic review), comparing forecast-driven policies against standard empirical baselines.
- **N2 Calibrated Prediction Intervals**: Leverages quantile modeling and conformal calibration with empirical coverage tracking.
- **N3 Demand-Pattern Routing**: Automatically categorizes series via Syntetos-Boylan / Kostenko-Hyndman classification (ADI vs. $CV^2$) into smooth, erratic, intermittent, or lumpy patterns to select optimal model families (Croston/TSB vs. ARIMA/XGBoost).
- **N4 Drift Monitoring & Retrain Triggers**: Continuous rolling error evaluation tracking concept drift with automated alerting.
- **N5 Explainability & What-If Simulation**: SHAP-driven feature importance coupled with pricing/promotion what-if scenario testing.

---

## Architecture & Project Structure

```
SP-SP/
├── configs/               # Declarative experiment and runtime configurations
├── data/                  # Raw, processed, and synthetic datasets (git-ignored)
├── docs/                  # Architectural records, contracts, protocols, and team guide
├── reports/               # Results tables, experiment registries, and figures
├── src/idfs/              # Core production package
│   ├── app/               # Streamlit multi-page UI application
│   ├── data/              # Ingestion, validation, and multi-source fusion
│   ├── evaluation/        # Point, interval, and inventory metrics; backtesting
│   ├── explain/           # Model explainability and SHAP attribution
│   ├── features/          # Leakage-free lag, rolling, and calendar transformers
│   ├── inventory/         # Policies, safety stock engine, simulation harness
│   ├── models/            # Baselines, statistical, tree-based (XGBoost), deep learning
│   ├── monitoring/        # Drift detection and alert mechanisms
│   ├── reporting/         # Automated report generation
│   └── uncertainty/       # Quantile loss and conformal prediction calibration
├── tests/                 # Unit, integration, and smoke test suites
└── scripts/               # Developer tools and task runners
```

---

## Installation & Quickstart

### Prerequisites
- Python 3.11 or 3.12 (Python 3.12.15 managed via `uv` recommended)
- Git

### 1. Environment Setup
```powershell
# Create virtual environment
uv venv --python 3.12 .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### 2. Run Verification Suite
```powershell
# Run linting, type-checking, and test suite via the cross-platform dev runner
python scripts/dev.py check
```

### 3. Generate Synthetic Benchmark Dataset
```powershell
python scripts/dev.py generate
```

### 4. Launch the Dashboard
```powershell
streamlit run src/idfs/app/main.py
```

---

## Team & Academic Metadata
- **Institution**: ICFAI Foundation for Higher Education, Hyderabad (IcfaiTech)
- **Project**: Special Project (Demand Forecasting & Supply Chain Optimization)
- **Students**: Pamdem Ravi Charan Reddy, Jaiti Vijaya Shravani, Vankayalapati Srujana, Pottipothu V M Krishna Swamy
- **Mentors**: Dr. Sukanta Das, Dr. Sandeep Kumar Panda
