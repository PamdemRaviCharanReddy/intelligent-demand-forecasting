# Phase 7 Brief: Final Held-Out Evaluation, Ablations & Academic Defense

**Phase:** Phase 7 (Final Evaluation, Ablations & Academic Viva Defense)  
**Tiers Covered:** P0 (Core Requirements) & P2 (Stretch Research Goals)  
**Lead Engineer / Researcher:** Lead Architect & University Special Project Researcher  
**Target Completion:** End of Week 8  

---

## 1. Goal Statement
Execute the rigorous scientific conclusion of the Intelligent Demand Forecasting System (IDFS). This entails:
1. Unsealing and evaluating all 8 models on the pristine 28-day held-out test window that has remained strictly untouched throughout Phase 0 to Phase 6.
2. Conducting formal novelty ablations ($N1$ to $N5$ on/off) to quantify the standalone contribution of each research novelty.
3. Conducting rigorous statistical hypothesis tests (Wilcoxon signed-rank and Diebold-Mariano tests) with exact $p$-values to verify that accuracy gains are statistically significant and not noise.
4. Generating comprehensive, publication-grade academic deliverables:
   - `docs/FINAL_REPORT.md`: Full IEEE/ACM format paper ready for submission.
   - `docs/SLIDES_OUTLINE.md`: Presentation structure with slide-by-slide speaker notes.
   - `docs/DEMO_SCRIPT.md`: Step-by-step 5-minute interactive viva demonstration script for examiners.
   - `docs/VIVA_QA_GUIDE.md`: Deep viva defense preparation covering 25+ anticipated hostile/technical questions with mathematical and empirical justifications.
5. Performing the final code quality audit and release tagging (`v1.0`).

---

## 2. Methodology & Mathematical Protocols

### 2.1 Final Sealed Test Evaluation
- **Split Protocol**: The final 28 days of the fused dataset (`data/processed/fused_sales.parquet`) are used strictly as test actuals. Models are trained on all data prior to this window.
- **Leakage Prevention**: Strictly zero feature computation uses target values inside the test window; rolling features use `.shift(1)` before window aggregation.
- **Models Compared**:
  - Naive & Seasonal Naive (7-day baseline)
  - SARIMAX(1,0,0)x(0,1,0)7
  - Prophet
  - Global XGBoost
  - PyTorch 12-Step LSTM
  - Croston (SBA) & TSB (Intermittent)
  - DemandPatternRouter (N3)
  - WeightedEnsemble (Inverse-variance weights)
- **Metrics Computed**:
  - MAE, RMSE, WAPE, MASE, Pinball Loss (0.10, 0.50, 0.90), Empirical Coverage (90%), Mean Interval Width.

### 2.2 Novelty Ablations Matrix
We quantify the incremental value added by the 5 novelty layers against a vanilla XGBoost baseline:
- **Variant 0 (Vanilla Baseline)**: Point-forecast XGBoost only, standard $(s, S)$ inventory using Gaussian assumptions, no drift monitor, no explainability.
- **Variant 1 (+N1 Cost Simulation)**: Discrete-event inventory simulation under asymmetric holding vs. stockout costs.
- **Variant 2 (+N2 Calibrated Intervals)**: Conformalized Quantile Regression (CQR) delivering guaranteed 90% finite-sample coverage.
- **Variant 3 (+N3 Demand Router)**: Syntetos-Boylan ADI vs. $CV^2$ categorization routing intermittent series to Croston/TSB.
- **Variant 4 (+N4 Surveillance Drift)**: Rolling error surveillance triggering automated retrain recommendations.
- **Variant 5 (+N5 Explainability & What-If)**: Tree-SHAP driver decomposition and counterfactual price/promotional shock testing.

### 2.3 Statistical Hypothesis Testing
- **Diebold-Mariano Test (1995)**: Tests null hypothesis $H_0: \mathbb{E}[d_t] = 0$ where $d_t = L(e_{1,t}) - L(e_{2,t})$ under squared error and absolute error loss.
- **Wilcoxon Signed-Rank Test**: Non-parametric paired test across series evaluation metrics to evaluate whether model differences are significant without normality assumptions.

---

## 3. Deliverables Checklist
- [ ] `src/idfs/evaluation/significance.py`: Diebold-Mariano and Wilcoxon hypothesis tests.
- [ ] `scripts/run_final_evaluation.py`: Unsealed test evaluation and ablation script generating `reports/tables/final_model_evaluation.csv`, `reports/tables/novelty_ablation_results.csv`, and `reports/tables/statistical_significance_tests.csv`.
- [ ] `docs/FINAL_REPORT.md`: Comprehensive IEEE/ACM style manuscript.
- [ ] `docs/SLIDES_OUTLINE.md`: Presentation structure and speaker notes.
- [ ] `docs/DEMO_SCRIPT.md`: 5-minute interactive viva demonstration walkthrough.
- [ ] `docs/VIVA_QA_GUIDE.md`: Comprehensive defense Q&A guide.
- [ ] `docs/phases/phase-7-final-evaluation/RETRO.md`: Phase 7 retrospective.
- [ ] Tag `v1.0` upon passing the full test suite (`python scripts/dev.py check`).
