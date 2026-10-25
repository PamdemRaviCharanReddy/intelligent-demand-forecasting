# Phase 0 Retrospective: Discovery, Scope & Architecture

**Date**: 2026-10-25  
**Phase**: Phase 0 (Discovery, Architecture, Skeleton & Setup)  
**Status**: Completed successfully — Human Gate Prepared

---

## 1. What Worked
- **Rapid Environment Adaptation**: Discovery revealed Windows 11 with system Python 3.14 (lacking ML pre-compiled wheels) and no installed Git or GitHub CLI. We successfully deployed PortableGit and used `uv` to provision an isolated Python 3.12 environment in `.venv`.
- **Decoupled Architecture**: Strict enforcement of the unidirectional layered architecture (`data` $\to$ `features` $\to$ `models` $\to$ `evaluation` $\to$ `inventory` $\to$ `app`) ensured that typed stubs and pure metrics can be verified in isolation.
- **Pure Metric Engine**: Built a comprehensive evaluation metrics suite with explicit edge-case handling (zero actuals, short series, constant baselines) backed by 58 unit tests passing at 100%.
- **Realistic Synthetic Benchmark**: Built `generate_synthetic_data` simulating multi-store retail dynamics (trend, weekly/yearly seasonality, promo uplifts, holidays, oil prices, concept drift, and intermittent zero-inflation) conforming to the formal data contract.
- **Cross-Platform Tooling**: Designed `scripts/dev.py` to provide consistent developer commands across Windows and Linux without requiring `make`.

---

## 2. What Surprised Us
- **Dual Mentor Discrepancy**: The student mid-term report listed Dr. Sukanta Das, while the presentation slides listed both Dr. Sukanta Das and Dr. Sandeep Kumar Panda. Both have been preserved and flagged with `[VERIFY]`.
- **Inconsistencies in Student Source Drafts**: The models listed varied from "LSTM only" to "XGBoost only" to "ARIMA, RF, LSTM" to "ARIMA, Prophet, XGBoost, LSTM". We resolved this in `docs/DOCUMENT_FIXES.md` with an all-inclusive unified model zoo and tiered priority.
- **MAPE Zero Pathology in Retail**: The student report emphasized MAPE as a headline metric; however, in retail data with intermittent zeros, standard MAPE explodes. We elevated WAPE and MASE to headline metrics while preserving zero-masked MAPE for historical comparison.

---

## 3. Plan Changes Logged
- **PCR-001**: Python 3.12 selected over 3.14 for wheel compatibility (`docs/PLAN_CHANGES.md`).
- **PCR-002**: Cross-platform Python runner `scripts/dev.py` implemented (`docs/PLAN_CHANGES.md`).
- **PCR-003**: PyTorch chosen for 12-step sequence LSTM over TensorFlow on Windows (`docs/PLAN_CHANGES.md`).
- **PCR-004**: Synthetic benchmark generator implemented as default pipeline driver pending student Kaggle token placement (`docs/PLAN_CHANGES.md`).

---

## 4. Readiness for Phase 1
All foundational contracts, directory trees, configuration structures, and baseline unit tests are in place. The project is ready to proceed to Phase 1 (Foundation: data ingestion, cleaning, validation, and multi-source fusion) immediately upon human gate approval.
