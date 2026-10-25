# Phase 7 Pre-Mortem: Anticipating Failure Modes

**Date:** 2026-10-25  
**Phase:** Phase 7 (Final Evaluation, Ablations & Academic Defense)  
**Lead Engineer / Researcher:** Lead Architect & University Special Project Researcher  

---

## 1. Prospective Failure Scenarios & Mitigation Strategies

| Failure Mode | Root Cause | Impact | Mitigation Strategy |
|:---|:---|:---|:---|
| **FM1: Test Data Leakage during Unsealing** | Accidentally fitting transformers or scalers on the full dataset including the 28-day test period. | Fatal scientific error invalidating results. | Ensure `pipeline.fit_transform()` is called *only* on `train_df`. The test fold is strictly passed through `pipeline.transform()`. |
| **FM2: Zero-Variance Diebold-Mariano Variance** | When comparing two models with identical forecasts or on very short horizons, auto-covariance estimate may be non-positive. | Numerical crash or `NaN` test statistic. | Use Harvey, Leybourne, and Newbold (1997) small-sample modification with variance regularizer `max(var, 1e-8)`. |
| **FM3: Fabricated or Inconsistent Viva Figures** | Report claims numbers that do not match the CSV files in `reports/tables/`. | Immediate vivavoce failure if examiner checks. | Every figure and table in `docs/FINAL_REPORT.md` must be directly parsed or referenced from code executions logged in `reports/tables/`. |
| **FM4: Negative Academic Results Concealed** | Deep learning (LSTM) being beaten by simpler models or baselines on intermittent series. | Loss of scientific credibility. | Scientific honesty principle: report all findings faithfully. If XGBoost or TSB outperforms LSTM on intermittent items, provide rigorous theoretical explanations (e.g., sample efficiency, inductive bias). |
| **FM5: CI / Type-Checking Regressions** | Adding new scripts and evaluation modules breaks `mypy` or `ruff`. | Merge blocked, build fails. | Run `python scripts/dev.py check` continuously throughout Phase 7 development. |

---

## 2. Success Criteria for Phase 7
1. Pristine 28-day test window unsealed with zero future leakage.
2. Complete comparison table generated for all 8 forecasters on held-out test data.
3. Novelty ablation matrix ($N1$ to $N5$) computed and logged to CSV.
4. Statistical significance tests (Diebold-Mariano and Wilcoxon) executed with exact $p$-values.
5. All academic defense documents (`FINAL_REPORT.md`, `SLIDES_OUTLINE.md`, `DEMO_SCRIPT.md`, `VIVA_QA_GUIDE.md`) drafted to institutional standard.
6. Clean exit gate: all CI checks passing (`ruff`, `mypy`, `pytest` 115+ tests), merge to `main`, and tag `v1.0`.
