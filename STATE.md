# Current Project State

- **Current Phase**: Phase 4 (Uncertainty Quantification & Inventory Simulation - Tier P0 / N1 / N2)
- **Current Stage**: THINK / PLAN
- **Last Completed Task**: Phase 3 Additional Models & Ensembles (93 tests passing, SARIMAX, Prophet, PyTorch LSTM, Croston, TSB, Router N3, Stacking, benchmark logged, tagged `phase-3-complete`).
- **Next Action**: Execute Phase 4: Quantile XGBoost, split conformal calibration engine, daily inventory simulator, (R, S) periodic policy, and headline cost evaluation.
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
