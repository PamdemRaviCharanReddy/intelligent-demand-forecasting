# Current Project State

- **Current Phase**: Phase 7 (Final Evaluation, Ablations & Academic Defense — Tier P0 / P2)
- **Current Stage**: THINK / PLAN
- **Last Completed Task**: Phase 6 Streamlit Dashboard & Executive Reporting (113 tests passing, SQLite RunStorage, HTML report generator, headless AppTest suite, tagged `phase-6-complete`).
- **Next Action**: Execute Phase 7: Unseal held-out test window, run novelty ablations (N1-N5), hypothesis tests (Wilcoxon, Diebold-Mariano), draft IEEE final report, slide outline, and viva defense guide.
- **Open Blockers**: None.

## Environment Summary
- **OS**: Windows 11 (NT 10.0.26300.0), PowerShell
- **Python**: Python 3.12.15 via `uv` in `.venv`
- **Git**: PortableGit initialized locally on branch `main`
- **Hardware**: AMD Ryzen (8 logical cores), 12GB RAM, NVIDIA RTX 3050 Laptop GPU (4GB VRAM)

## Key Decisions Index
- [ADR-001] Python 3.12 selected over 3.14 for PyTorch, XGBoost, and statsmodels compatibility (`docs/DECISIONS.md`).
- [ADR-002] Cross-platform task runner via `python scripts/dev.py` instead of GNU Make (`docs/DECISIONS.md`).
- [ADR-003] PyTorch selected for LSTM implementation over TensorFlow due to Windows/Python 3.12 stability (`docs/DECISIONS.md`).
- [ADR-004] Dual interval estimation: Quantile XGBoost + Conformal split calibration (`docs/DECISIONS.md`).
- [ADR-005] Periodic review $(R, S)$ order-up-to inventory policy with empirical fill-rate and cost modeling (`docs/DECISIONS.md`).
- [ADR-006] SQLite embedded storage for historical simulations and human planner decisions (`docs/DECISIONS.md`).
