# Master Project Plan (Phases 0–7)

`docs/PLAN.md` is the single source of truth for all project commitments, task breakdowns, dependencies, and definitions of done. Any modification to scope or task priority requires a formal Plan Change Record in `docs/PLAN_CHANGES.md`.

- **Project Timeline**: Relative 8-week schedule (W1 to W8); `deadline: null` in config.
- **Critical Path**: P0 Data Pipeline $\to$ P0 Features & Backtester $\to$ P0 XGBoost $\to$ P0 Interval Calibration (N2) $\to$ P0 Inventory Cost Simulation (N1) $\to$ P0 Streamlit App $\to$ P0 Final Held-Out Evaluation.

---

## Task Breakdown Across All Phases

### Phase 0: Discovery, Scope, Architecture & Skeleton (Week 1) — Tier: P0
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P0-T01** | Environment & tooling discovery | Lead Architect | S | None | P0 | ✅ DONE | `docs/ENVIRONMENT.md` and `CLAUDE.md` created |
| **P0-T02** | Source digest & inconsistencies audit | Researcher | S | None | P0 | ✅ DONE | `PROJECT_SOURCE_DIGEST.md` & `DOCUMENT_FIXES.md` written |
| **P0-T03** | Architectural brief & decision records | Lead Architect | M | T01 | P0 | ✅ DONE | `BRIEF.md` and 13 ADRs recorded in `docs/DECISIONS.md` |
| **P0-T04** | Formal contracts & protocols | Lead Architect | M | T03 | P0 | ✅ DONE | `SCOPE.md`, `ARCHITECTURE.md`, `DATA_CONTRACT.md`, `EVALUATION_PROTOCOL.md`, `INVENTORY_MODEL.md` |
| **P0-T05** | Master plan & risk pre-mortem | Lead Architect | M | T04 | P0 | ✅ DONE | `PLAN.md`, `PLAN_CHANGES.md`, `RISKS.md`, `ASSUMPTIONS.md` |
| **P0-T06** | Package structure & config loader | Software Eng | S | T01 | P0 | ✅ DONE | `pyproject.toml`, `src/idfs/config.py`, default YAML |
| **P0-T07** | Pure metric engine & test suite | ML Engineer | M | T06 | P0 | ✅ DONE | `metrics.py` fully tested across all edge cases |
| **P0-T08** | Synthetic multi-source generator | ML Engineer | M | T06 | P0 | ✅ DONE | Realistic sales, stores, holidays, oil data generator |
| **P0-T09** | Typed stubs for Forecaster & Policies | Software Eng | S | T06 | P0 | ✅ DONE | Base interfaces with docstrings & typing |
| **P0-T10** | Cross-platform developer runner | Software Eng | S | T06 | P0 | ✅ DONE | `scripts/dev.py` supporting check/lint/test/format |
| **P0-T11** | Red-team architectural critique | Lead Researcher | M | T04 | P0 | 🔄 ACTIVE | 10 examiner critiques with documented defenses |
| **P0-T12** | Human gate checkpoint review | Lead Architect | S | All | P0 | 🔄 ACTIVE | Gate message presented; awaiting human approval |

---

### Phase 1: Foundation & Data Engineering (Week 2) — Tier: P0
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P1-T01** | Data ingestion module (CSV / Parquet) | Data Eng | S | P0-T09 | P0 | ✅ DONE | Reads sales, metadata, external regressors |
| **P1-T02** | Schema validation & quality asserts | Data Eng | S | P1-T01 | P0 | ✅ DONE | Rejects negative sales, flags nulls & gaps |
| **P1-T03** | Data cleaning & outlier mitigation | Data Eng | M | P1-T02 | P0 | ✅ DONE | Imputes missing dates; winsorizes anomalies |
| **P1-T04** | Multi-source fusion engine | Data Eng | M | P1-T03 | P0 | ✅ DONE | Joins sales, stores, holidays, oil; graceful fallback |
| **P1-T05** | Exploratory Data Analysis (EDA) script | Data Eng | M | P1-T04 | P0 | ✅ DONE | Auto-generates figures into `reports/figures/eda/` |
| **P1-T06** | End-to-end data pipeline integration test| QA Engineer | S | P1-T04 | P0 | ✅ DONE | Test validates ingestion $\to$ fusion on synthetic data |

---

### Phase 2: Feature Engineering, Baselines & XGBoost (Week 3) — Tier: P0
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P2-T01** | Leakage-free lag & rolling transformers | ML Engineer | M | P1-T04 | P0 | ✅ DONE | Implements `.shift(1)` rolling stats; passes leakage tests |
| **P2-T02** | Calendar, cyclical & holiday distance | ML Engineer | S | P2-T01 | P0 | ✅ DONE | Encodes cyclical DOW/DOY and holiday deltas |
| **P2-T03** | Categorical encoding (Target/Entity) | ML Engineer | S | P2-T01 | P0 | ✅ DONE | Fits encodings strictly on train folds |
| **P2-T04** | Naive & Seasonal Naive forecasters | ML Engineer | S | P0-T09 | P0 | ✅ DONE | Implements baseline predictions and scale references |
| **P2-T05** | Walk-forward cross-validation harness | ML Engineer | M | P2-T01 | P0 | ✅ DONE | Evaluates models across 3 rolling-origin folds |
| **P2-T06** | Global XGBoost forecaster | ML Engineer | M | P2-T05 | P0 | ✅ DONE | Cross-series gradient booster with early stopping |
| **P2-T07** | Baseline vs XGBoost comparison table | Researcher | S | P2-T06 | P0 | ✅ DONE | Generates first comparative metrics CSV in `reports/tables/` |

---

### Phase 3: Additional Models & Demand Routing (Week 4) — Tier: P1
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P3-T01** | Classical ARIMA / SARIMA wrapper | ML Engineer | M | P2-T05 | P1 | ✅ DONE | Fits per-series ARIMA on 20–50 series subset |
| **P3-T02** | Additive Prophet forecaster | ML Engineer | M | P2-T05 | P1 | ✅ DONE | Decomposes trend, holidays, weekly seasonality |
| **P3-T03** | 12-Step PyTorch LSTM model | Deep Learning | L | P2-T05 | P1 | ✅ DONE | Windowed sequence modeling; train-only MinMax scaling |
| **P3-T04** | Intermittent Croston & TSB forecaster | ML Engineer | M | P2-T05 | P1 | ✅ DONE | Models inter-arrival times and demand sizes |
| **P3-T05** | Demand-pattern router (N3) | Researcher | M | P3-T04 | P1 | ✅ DONE | Classifies ADI vs $CV^2$; routes to best model family |
| **P3-T06** | Stacking / Weighted Ensemble | ML Engineer | S | P3-T05 | P1 | ✅ DONE | Combines candidate forecasts via validation weights |
| **P3-T07** | Comprehensive model comparison report | Researcher | S | P3-T06 | P1 | ✅ DONE | Multi-model evaluation table across all metrics |

---

### Phase 4: Uncertainty Quantification & Inventory Simulation (Week 5) — Tier: P0 / N1 / N2
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P4-T01** | Quantile XGBoost implementation | ML Engineer | M | P2-T06 | P0/N2 | ✅ DONE | Predicts $\tau \in \{0.05, 0.50, 0.95\}$ quantiles |
| **P4-T02** | Split conformal calibration engine | Researcher | M | P4-T01 | P0/N2 | ✅ DONE | Adjusts bounds to satisfy empirical nominal coverage |
| **P4-T03** | Daily inventory simulation engine | Software Eng | M | P0-T09 | P0/N1 | ✅ DONE | Steps through daily receipts, sales, stockouts |
| **P4-T04** | Periodic Review $(R, S)$ policy | Software Eng | M | P4-T03 | P0/N1 | ✅ DONE | Computes safety stock from forecast uncertainty |
| **P4-T05** | Naive moving average policy benchmark | Software Eng | S | P4-T03 | P0/N1 | ✅ DONE | Implements historical moving average reorder rule |
| **P4-T06** | Cost simulation & fill-rate evaluation | Researcher | M | P4-T04 | P0/N1 | ✅ DONE | Generates headline holding vs stockout cost table |
| **P4-T07** | Lead time & cost ratio sensitivity grid | Researcher | M | P4-T06 | P0/N1 | ✅ DONE | Sensitivity sweeps across $L \in [2, 7]$ and $c_s/c_h$ |

---

### Phase 5: Monitoring & Explainability (Week 6) — Tier: P1 / N4 / N5
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P5-T01** | Rolling-error drift monitor (N4) | ML Engineer | M | P2-T06 | P1/N4 | ✅ DONE | Computes rolling MAE/WAPE; triggers alert on threshold |
| **P5-T02** | Simulated & real drift demonstration | Researcher | M | P5-T01 | P1/N4 | ✅ DONE | Verifies detection on injected concept drift dataset |
| **P5-T03** | Global SHAP feature importance (N5) | ML Engineer | M | P2-T06 | P1/N5 | ✅ DONE | Generates summary bee-swarm and bar attribution plots |
| **P5-T04** | Local SHAP single-prediction waterfall | ML Engineer | S | P5-T03 | P1/N5 | ✅ DONE | Visualizes drivers of individual peak sales days |
| **P5-T05** | Interactive What-If simulator (N5) | Software Eng | M | P5-T03 | P2/N5 | ✅ DONE | Dynamic price/promotion slider testing demand shock |

---

### Phase 6: Streamlit Application & Reporting (Week 7) — Tier: P0
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P6-T01** | Multi-page Streamlit application core | Frontend Eng | M | P0-T06 | P0 | ✅ DONE | Modular page navigation and state caching |
| **P6-T02** | Data Explorer & EDA page | Frontend Eng | S | P1-T05 | P0 | ✅ DONE | Upload CSV/Excel, summary statistics, missing flags |
| **P6-T03** | Forecast Studio page | Frontend Eng | M | P2-T06 | P0 | ✅ DONE | Interactive forecast plotting with interval bands |
| **P6-T04** | Inventory Replenishment & Alerts page | Frontend Eng | M | P4-T06 | P0 | ✅ DONE | Displays order quantities, reorder points, stockout risk |
| **P6-T05** | Drift Monitor & Explainability page | Frontend Eng | M | P5-T03 | P1 | ✅ DONE | Renders rolling error charts and SHAP plots |
| **P6-T06** | SQLite persistent run storage | Software Eng | S | P6-T01 | P0 | ✅ DONE | Stores user scenarios and historical runs |
| **P6-T07** | Automated downloadable report builder | Software Eng | M | P6-T01 | P0 | ✅ DONE | Exports executive HTML/CSV summary reports |
| **P6-T08** | Headless smoke tests for app | QA Engineer | S | P6-T01 | P0 | ✅ DONE | Automated test ensuring pages load without exception |

---

### Phase 7: Final Evaluation, Ablations & Academic Defense (Week 8) — Tier: P0 / P2
| Task ID | Task Description | Owner Role | Size | Dep | Tier | Status | Definition of Done |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| **P7-T01** | Sealed test window evaluation | Lead Researcher | M | All Models| P0 | ✅ DONE | One-time unsealing of 28-day final test set |
| **P7-T02** | Comprehensive feature & novelty ablations| Lead Researcher | M | P7-T01 | P0 | ✅ DONE | Measures performance with each novelty on/off |
| **P7-T03** | Statistical significance hypothesis tests | Lead Researcher | S | P7-T01 | P0 | ✅ DONE | Wilcoxon and Diebold-Mariano tests with $p$-values |
| **P7-T04** | Academic report manuscript (Markdown) | Lead Author | L | P7-T02 | P0 | ✅ DONE | IEEE/ACM format paper ready for submission |
| **P7-T05** | Final slide deck & presentation outline | Lead Author | M | P7-T04 | P0 | ✅ DONE | Complete presentation structure with speaker notes |
| **P7-T06** | 5-Minute viva demonstration script | Lead Author | S | P6-T01 | P0 | ✅ DONE | Step-by-step interactive demo script for examiners |
| **P7-T07** | Comprehensive Viva Q&A defense guide | Lead Author | M | P7-T04 | P0 | ✅ DONE | Detailed examiner questions and rigorous answers |
| **P7-T08** | Release tagging & final audit (`v1.0`) | Lead Architect | S | All | P0 | ✅ DONE | Green build, clean git log, tagged `v1.0` |
