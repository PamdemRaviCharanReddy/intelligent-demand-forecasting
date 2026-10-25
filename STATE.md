# Current Project State

- **Current Phase**: Phase 3 (Additional Models & Demand Routing - Tier P1)
- **Current Stage**: THINK / PLAN
- **Last Completed Task**: Phase 2 Features, Baselines, Backtest & XGBoost (84 tests passing, leakage verified, benchmark logged, tagged `phase-2-complete`).
- **Next Action**: Execute Phase 3: ARIMA/SARIMAX, Prophet, 12-step PyTorch LSTM, Croston/TSB intermittent models, Syntetos-Boylan demand router (N3), and model stacking.
- **Open Blockers**: None. Data pipeline fully validated on synthetic dataset (and ready for real Kaggle Favorita dataset when placed).

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
