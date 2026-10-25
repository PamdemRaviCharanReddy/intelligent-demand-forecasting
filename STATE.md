# Current Project State

- **Current Phase**: Phase 0 (Discovery, Scope, Architecture & Skeleton)
- **Current Stage**: VERIFY / GATE (Preparing Phase 0 Documentation and Testing Skeleton)
- **Last Completed Task**: Created baseline code stubs, config system, evaluation metrics, synthetic generator, tests, and core project setup.
- **Next Action**: Complete all Phase 0 architectural documents, run test suite, push/commit to git, and present Phase 0 Human Gate.
- **Open Blockers**: None blocking Phase 0 completion. Kaggle API credentials and GitHub CLI absent (handled with documented fallbacks).

## Environment Summary
- **OS**: Windows 11 (NT 10.0.26300.0), PowerShell
- **Python**: Python 3.12.15 via `uv` in `.venv`
- **Git**: PortableGit 2.56.0.3 initialized locally on branch `main`
- **Hardware**: AMD Ryzen (8 logical cores), 12GB RAM, NVIDIA RTX 3050 Laptop GPU (4GB VRAM)

## Key Decisions Index
- [ADR-001] Python 3.12 selected over 3.14 for PyTorch, XGBoost, and statsmodels compatibility (`docs/DECISIONS.md`).
- [ADR-002] Cross-platform task runner via `python scripts/dev.py` instead of GNU Make (`docs/DECISIONS.md`).
- [ADR-003] PyTorch selected for LSTM implementation over TensorFlow due to Windows/Python 3.12 stability (`docs/DECISIONS.md`).
- [ADR-004] Dual interval estimation: Quantile XGBoost + Conformal split calibration (`docs/DECISIONS.md`).
- [ADR-005] Periodic review $(R, S)$ order-up-to inventory policy with empirical fill-rate and cost modeling (`docs/DECISIONS.md`).
