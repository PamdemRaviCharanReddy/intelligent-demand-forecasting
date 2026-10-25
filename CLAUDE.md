# CLAUDE.md — Operating Guidelines for IDFS

## 1. Operating Principles
- **Think slowly, act deliberately**: Do not start a phase before drafting `BRIEF.md`, recording trade-offs in `DECISIONS.md`, and detailing tasks in `PLAN.md`.
- **Strict plan, dynamic execution**: `docs/PLAN.md` is the single source of truth. Scope or timeline shifts require a logged Plan Change Record (PCR) in `docs/PLAN_CHANGES.md`.
- **Scientific honesty**: Zero fabrication. Every figure, table, and metric must trace back to artifacts in `reports/`. Report negative results objectively (e.g. if XGBoost beats LSTM).
- **Strictly no leakage**: Time-based splits only. Rolling features must apply `.shift(1)` before window operations. Scalers/encoders fit strictly on train folds. Held-out test set remains pristine until Phase 7.
- **Reproducibility**: Config-driven execution (`configs/`), fixed pseudo-random seeds (`seed: 42`), results written to structured CSV registries.
- **Safety**: Work only inside the workspace root. Never force push to `main`. Never commit secrets or raw API tokens. Use `.venv`.
- **Minimal human dependence**: Document assumptions in `docs/ASSUMPTIONS.md`. If genuinely blocked, log in `NEEDS_HUMAN.md` and continue on unblocked tasks.
- **Teach as you build**: Maintain `docs/GUIDE_FOR_TEAM.md` with layperson explanations and defense viva Q&A for student stakeholders.

---

## 2. Locked Scope & Architecture
- **Language & Runtime**: Python 3.12 (via `uv` venv).
- **Primary Data**: Kaggle "Store Sales – Time Series Forecasting" (Favorita multi-source: sales, stores, holidays, oil, transactions) + Synthetic Generator.
- **Models**:
  - P0: Naive, Seasonal Naive, XGBoost (global model).
  - P1: ARIMA/SARIMA (subset), Prophet (subset), PyTorch LSTM (global), Croston/TSB (intermittent), Stacking/Weighted Ensemble.
- **Validation**: Walk-forward (rolling-origin) cross-validation across identical folds.
- **Novelty Stack**:
  - N1: Cost-based inventory simulation (holding, stockout, ordering costs).
  - N2: Calibrated prediction intervals (quantile regression + conformal calibration).
  - N3: Demand-pattern router (ADI vs. $CV^2$).
  - N4: Drift monitoring (rolling error & retrain alerts).
  - N5: Explainability (SHAP & what-if scenarios).
- **Tiers**:
  - P0: Data pipeline, baseline models, XGBoost, walk-forward CV, intervals, inventory simulation, basic Streamlit app, downloadable report.
  - P1: ARIMA, Prophet, LSTM, ensemble, router, drift monitoring, SHAP.
  - P2: What-if panel, M5 stretch dataset, foundation time-series zero-shot ablation.

---

## 3. Resume Protocol & State Pointers
When resuming execution or after a context refresh:
1. Read `CLAUDE.md`, `STATE.md`, and `PROGRESS.md`.
2. Check `NEEDS_HUMAN.md` for blocking queries.
3. Review `docs/PLAN.md` to identify the current phase, critical path, and next task definition of done.
4. Verify workspace status using `git status` and recent commit logs.
5. Continue from "Next Action" in `STATE.md`.
