# Plan Change Records (PCR)

Any change to scope, delivery tiers, model selections, datasets, metric definitions, or phase ordering must be logged in this document. P0 items cannot be deferred or dropped unless a documented blocker makes them technically impossible.

---

## PCR Index

| PCR ID | Date | Trigger | Scope / Change Summary | Impact on P0 | Status |
|:---:|:---:|:---|:---|:---:|:---:|
| **PCR-001** | 2026-10-25 | Environment Discovery: Python 3.14 system / no prebuilt wheels | Standardized on Python 3.12 managed via `uv` | None (P0 preserved) | Approved |
| **PCR-002** | 2026-10-25 | Environment Discovery: Windows OS lacks native `make` | Implemented `scripts/dev.py` cross-platform runner | None (P0 preserved) | Approved |
| **PCR-003** | 2026-10-25 | Technical Discovery: TensorFlow Windows support constraints | Swapped LSTM backend to PyTorch 2.x via `Forecaster` ABC | None (P0 preserved) | Approved |
| **PCR-004** | 2026-10-25 | Missing Kaggle credentials on host | Defaulted pipeline to synthetic multi-source generator | None (P0 preserved) | Approved |

---

## PCR Details

### PCR-001: Target Python 3.12 Runtime
- **Trigger**: System has Python 3.14.5/3.14.7 installed, but scientific libraries (`numpy`, `pandas`, `xgboost`, `torch`) lack production wheels for Windows on 3.14.
- **Change**: Installed isolated Python 3.12.15 via `uv` in `.venv`.
- **Impact on P0**: Zero negative impact; guarantees stability and reproducibility across all phases.

### PCR-002: Cross-Platform Python Task Runner
- **Trigger**: Windows PowerShell does not provide GNU Make.
- **Change**: Replaced `Makefile` with `scripts/dev.py` supporting `check`, `lint`, `format`, `test`, `typecheck`, `generate`.
- **Impact on P0**: Zero negative impact; eliminates third-party tool installation requirements on student laptops.

### PCR-003: PyTorch Implementation for 12-Step LSTM
- **Trigger**: TensorFlow on modern Windows lacks direct native GPU acceleration and has installation issues with Python 3.12.
- **Change**: Implement the 12-step sequence LSTM using PyTorch (`torch.nn.LSTM`). The unified `Forecaster` ABC hides internal framework differences.
- **Impact on P0**: Zero negative impact; provides cleaner code and RTX 3050 CUDA acceleration.

### PCR-004: Synthetic Benchmark Data Pipeline
- **Trigger**: Kaggle API key not configured on host system.
- **Change**: Implemented realistic retail synthetic data generator (`src/idfs/data/synthetic.py`) with trend, seasonality, promotions, holidays, oil, drift, and censoring. Once students supply `kaggle.json`, the real Favorita dataset is automatically ingested.
- **Impact on P0**: Zero negative impact; permits 100% autonomous development and testing.
