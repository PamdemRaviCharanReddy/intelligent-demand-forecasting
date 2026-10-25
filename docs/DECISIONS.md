# Architecture Decision Records (ADRs)

This document formalizes every major technical and methodological design decision in IDFS. Each decision compares 2–3 viable options against explicit criteria: scientific correctness, CPU feasibility, explainability to viva examiners, and implementation risk.

---

## [ADR-001] Python Runtime & Virtual Environment Management
- **Context**: Host system defaults to Python 3.14.5 and 3.14.7. Major machine learning packages (e.g., PyTorch, XGBoost, Statsmodels) lack stable pre-compiled wheels for Python 3.14 on Windows.
- **Options Considered**:
  1. *Python 3.14 with source compilation*: High failure risk on C extensions and Cuda libraries.
  2. *Python 3.12 via `uv` (Selected)*: Mature binary wheels across PyPI, extremely fast isolated environments, guaranteed compatibility with data science stack.
  3. *Python 3.9 via Conda*: Matches the minimum in student report, but slower and larger disk footprint.
- **Decision**: Use Python 3.12.15 provisioned and managed via `uv` in `.venv`.
- **Consequences**: Fast setup, standard typing features (PEP 695/PEP 585), stable builds.
- **Revisit-If**: Official universal wheels for 3.14 become available across all project dependencies.

---

## [ADR-002] Task Runner & Developer Tooling
- **Context**: The student report lists Windows/Linux environments. Traditional `Makefile` fails natively on Windows PowerShell due to lack of `make`.
- **Options Considered**:
  1. *GNU Make via Chocolatey/MSYS2*: Requires extra external dependencies and admin rights.
  2. *Python CLI runner (`scripts/dev.py`) (Selected)*: Zero external dependencies beyond standard library, cross-platform compatibility across Windows/macOS/Linux.
  3. *Taskfile (Go) / Nox*: Requires additional package installations and unfamiliar syntax for students.
- **Decision**: Implement `scripts/dev.py` supporting `check`, `lint`, `format`, `test`, `typecheck`, and `generate`.
- **Consequences**: Single command execution (`python scripts/dev.py check`) regardless of student operating system.
- **Revisit-If**: Team explicitly requests CI integration via standard GNU make.

---

## [ADR-003] Deep Learning Framework for LSTM: PyTorch vs. TensorFlow/Keras
- **Context**: The mid-term report mentions TensorFlow/Keras for the 12-step LSTM algorithm. However, TensorFlow on Windows with modern Python versions has deprecated native GPU support (WSL2 required) and has complex wheel installation issues.
- **Options Considered**:
  1. *TensorFlow/Keras*: Matches verbatim report text, but exhibits severe installation hurdles on Windows with Python 3.12, and CPU-only builds are cumbersome.
  2. *PyTorch 2.x (Selected)*: Seamless Windows support, native CUDA detection on the RTX 3050 Laptop GPU, cleaner object-oriented sequence modeling, highly respected in academic viva.
  3. *Keras 3 with PyTorch backend*: Extra abstraction layer with potential version mismatches.
- **Decision**: Implement the 12-step LSTM using PyTorch (`torch.nn.LSTM`). The `Forecaster` ABC completely isolates the internal framework, so examiners see the exact same sequential interface and mathematical formulation (gates, hidden states, cell states).
- **Consequences**: Documented in `docs/DOCUMENT_FIXES.md` and explained in viva guide.
- **Revisit-If**: Examiner insists strictly on Keras syntax, in which case Keras 3 with PyTorch backend can be slotted into `src/idfs/models/lstm.py`.

---

## [ADR-004] Prediction Interval Estimation: Quantile Regression vs. Conformal Prediction
- **Context**: Point forecasts alone fail to support risk-averse inventory replenishment. We require valid prediction intervals.
- **Options Considered**:
  1. *Quantile Regression Only*: Fast via XGBoost quantile objective (`reg:quantileerror`), but quantiles may suffer from crossing or poor empirical coverage under heteroscedastic noise.
  2. *Residual Bootstrapping / Gaussian Assumption*: Strong parametric assumption ($e \sim \mathcal{N}(0, \sigma^2)$) frequently violated by intermittent and skewed retail demand.
  3. *Dual Hybrid: Quantile XGBoost + Split Conformal Calibration (Selected)*: Quantile regression generates candidate upper/lower quantiles, and conformal prediction on validation residuals calibrates nominal $(1-\alpha)$ coverage with finite-sample statistical guarantees.
- **Decision**: Adopt the Dual Hybrid approach (Novelty N2).
- **Consequences**: Provides mathematically guaranteed empirical coverage tables for viva defense.
- **Revisit-If**: Computational overhead during backtesting exceeds time budget (can fall back to conformalized residuals around point forecast).

---

## [ADR-005] Inventory Policy Formulation & Review Cadence
- **Context**: Retail demand models must drive tangible inventory savings (Novelty N1). Public datasets lack warehouse stock levels, supplier lead times, and unit purchase costs.
- **Options Considered**:
  1. *Continuous Review $(s, S)$*: Complex event-driven state tracking; requires real-time transaction streams often absent in batch retail planning.
  2. *Periodic Review $(R, S)$ Order-Up-To Policy (Selected)*: Review stock every $R$ days (e.g. weekly $R=7$), placing an order to bring inventory position up to level $S = \hat{D}_{R+L} + z \cdot \hat{\sigma}_{R+L}$. Matches real retail purchasing rhythms.
  3. *Simple Constant Reorder Point*: Fixed threshold ignoring demand trends and promotional spikes.
- **Decision**: Implement a Periodic Review $(R, S)$ policy with configurable lead times ($L \in [2, 7]$ days), holding costs, stockout penalties, and target service levels ($95\%$).
- **Consequences**: Evaluated alongside a baseline `NaivePolicy` that relies on static moving averages.
- **Revisit-If**: Real warehouse continuous telemetry data is supplied by industry partner.

---

## [ADR-006] Dataset Granularity & Hierarchical Aggregation
- **Context**: Kaggle Favorita contains 54 stores and 33 product families ($54 \times 33 = 1,782$ series over 1,684 days $\approx 3$ million rows). Training per-series models (ARIMA, Prophet) on 1,782 series is computationally prohibitive on a laptop.
- **Options Considered**:
  1. *Single Aggregate Series (Entire Chain)*: Simplistic, hides intermittent demand patterns and cross-store variability.
  2. *Full Store $\times$ Family Matrix (1,782 series)*: Feasible for global XGBoost/LSTM, but completely infeasible for statistical models (ARIMA takes ~5-10 seconds per series).
  3. *Tiered Granularity / Representative Stratified Subset (Selected)*: Global models (XGBoost, LSTM) train across all series (or a 50-series stratified benchmark); local models (ARIMA, Prophet) train on a 20–50 series subset representing diverse demand patterns (smooth, erratic, intermittent, lumpy).
- **Decision**: Standardize on Store $\times$ Family level, using a 50-series representative benchmark (5 stores $\times$ 10 families) for model comparison and walk-forward CV.
- **Consequences**: Balances CPU feasibility (~minutes instead of hours) with rich statistical variety.
- **Revisit-If**: Dedicated cloud compute cluster with parallel multiprocessing becomes available.

---

## [ADR-007] Forecast Horizon & Origin Cadence
- **Context**: Daily operational planning requires a realistic forecasting window.
- **Options Considered**:
  1. *1-Day Ahead*: Too short for supply chain ordering with supplier lead times of 2–7 days.
  2. *14-Day Ahead Daily (Selected)*: Matches standard two-week retail replenishment horizon and Favorita competition evaluation period (16 days).
  3. *90-Day Ahead Daily*: High forecast uncertainty accumulation, unrealistic without macro-level variables.
- **Decision**: Default to a 14-day daily forecast horizon ($H=14$), with weekly and monthly aggregation views in the UI.
- **Consequences**: Protection period for inventory equals $R + L = 7 + 5 = 12$ days, well within the 14-day forecast window.
- **Revisit-If**: User selects custom horizon in the Streamlit interface.

---

## [ADR-008] Walk-Forward Cross-Validation Design
- **Context**: Time-series cross-validation must strictly avoid data leakage. Standard $k$-fold shuffle is invalid.
- **Options Considered**:
  1. *Single Train/Test Split (e.g. 80/20)*: Sensitive to temporary shocks or seasonal anomalies in the single test slice.
  2. *Expanding Window (Rolling-Origin) Walk-Forward (Selected)*: 3 folds of 14 days each, preceded by an expanding historical training window. A pristine 28-day final test period is held out untouched until Phase 7.
  3. *Sliding Window*: Drops historical data; unnecessary when global models benefit from deep history.
- **Decision**: 3-fold rolling-origin walk-forward CV with 14-day validation windows and a locked 28-day final test set.
- **Consequences**: Guarantees identical folds for all evaluated models, preventing split bias.
- **Revisit-If**: Historical data length is under 90 days.

---

## [ADR-009] Model Architecture: Global vs. Local Model Partitioning
- **Context**: Time-series forecasting includes both local methods (fit on one series independently) and global methods (fit on all series simultaneously with cross-learning).
- **Options Considered**:
  1. *All Local Models*: High training overhead; cannot share parameters or learn categorical store/family embeddings.
  2. *All Global Models*: Prevents running traditional statistical baselines (ARIMA/SARIMA) specified in the student report.
  3. *Hybrid Architecture via Common `Forecaster` ABC (Selected)*: Global models (XGBoost, LSTM) leverage all series with categorical indicators and shared lag dynamics. Local models (ARIMA, Prophet) are wrapped in a vectorizer or per-series manager that conforms to the exact same `Forecaster` API.
- **Decision**: Enforce common `Forecaster` ABC where `supports_global` indicates data partitioning strategy.
- **Consequences**: Evaluation harness treats every model identically.
- **Revisit-If**: None; modular design accommodates any future model type.

---

## [ADR-010] Experiment Tracking & Results Registry
- **Context**: Need reproducible record of runs, hyperparameters, commits, and metrics.
- **Options Considered**:
  1. *Heavyweight Platform (MLflow / Weights & Biases)*: Requires running local daemon/server or online accounts; prone to port conflicts on student laptops.
  2. *Lightweight File-Based CSV/Parquet Registry (Selected)*: Every experiment run writes an immutable record to `reports/tables/results_registry.csv` keyed by 8-char config hash, git commit hash, timestamp, and seed.
  3. *Git Notes*: Inconvenient to parse and display within Streamlit dashboard.
- **Decision**: Build an explicit file-based experiment registry in `reports/tables/`.
- **Consequences**: Simple, human-inspectable, version-controllable, zero daemon overhead.
- **Revisit-If**: Distributed team experiments require remote centralized synchronization.

---

## [ADR-011] Application Architecture & State Management
- **Context**: The student report calls for a Streamlit web application.
- **Options Considered**:
  1. *Monolithic `app.py`*: Becomes unmaintainable as pages for EDA, forecasting, inventory, drift, and reports are added.
  2. *Multi-Page Streamlit App with Clean Separation (Selected)*: `src/idfs/app/main.py` serves as the entrypoint, with pages under `src/idfs/app/pages/` (Data Overview, Forecast Studio, Inventory Planner, Model Comparison, Drift Monitor). All business logic resides in `src/idfs/` core modules; UI files only render components.
  3. *FastAPI Backend + React Frontend*: Disproportionate engineering complexity; student report specifically prescribes Streamlit.
- **Decision**: Multi-page Streamlit application strictly decoupling UI rendering from computational core.
- **Consequences**: Headless automated smoke testing of application logic without running browser drivers.
- **Revisit-If**: Multi-tenant concurrent web access requirements emerge.

---

## [ADR-012] Persistent Data Storage
- **Context**: Need local storage for uploaded datasets, generated features, and saved run scenarios.
- **Options Considered**:
  1. *PostgreSQL / MySQL*: Excessive installation overhead for student project grading viva.
  2. *Parquet + SQLite (Selected)*: Parquet for columnar numerical time-series feature caching (fast I/O, compression); SQLite for application metadata, user profiles, and saved run records.
  3. *Plain CSV*: Inefficient for large feature matrices with 50+ lag/rolling columns.
- **Decision**: Dual storage strategy: Parquet for analytics data, SQLite for app session persistence.
- **Consequences**: Exactly satisfies report specification ("CSV/SQLite") while providing production-grade Parquet performance.
- **Revisit-If**: Data size exceeds single-node disk capacity.

---

## [ADR-013] Package Layout & Dependency Inversion
- **Context**: Ensuring clean architecture with strictly unidirectional dependencies.
- **Options Considered**:
  1. *Flat script folder*: Spaghetti dependencies; logic duplication across CLI and UI.
  2. *Layered `src/idfs` Architecture (Selected)*:
     `data` → `features` → `models` → `evaluation` / `uncertainty` → `inventory` → `reporting` / `app`.
     Dependencies flow strictly downwards; upstream layers never import downstream layers.
- **Decision**: Standard `src/idfs` layout managed with Hatchling build backend.
- **Consequences**: Guaranteed modularity, zero circular dependencies, high testability.
- **Revisit-If**: Monorepo split into separate distributed microservices.
