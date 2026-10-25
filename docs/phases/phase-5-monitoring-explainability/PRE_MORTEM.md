# Phase 5 Pre-Mortem: What Could Go Wrong & Mitigations

**Phase:** Phase 5 (Monitoring & Explainability - Tier P1 / N4 / N5)  
**Date:** 2026-10-25  

---

## 1. Potential Failure Modes

### Risk 1: SHAP Calculation on Large Tabular Datasets is Slow or Memory-Intensive
- **Failure:** Running `shap.TreeExplainer` on tens of thousands of rows causes CPU slowdowns or memory exhaustion.
- **Mitigation:**
  - Subsample a representative evaluation sample (e.g. 500-1,000 observations stratified across stores and families) for calculating Shapley values.
  - Use `approximate=False` or `check_additivity=False` if floating-point discrepancies occur.

### Risk 2: Drift Detector False Positives during Known Seasonal Peaks (e.g. Holidays)
- **Failure:** Regular holiday demand spikes could be misidentified as permanent concept drift.
- **Mitigation:**
  - Rolling window $W \ge 28$ days smooths over isolated weekly or holiday spikes.
  - Configurable threshold multiplier (e.g. $1.50 \times \text{baseline}$).
  - CUSUM with drift tolerance parameter $\delta$.

### Risk 3: What-If Simulator Produces Inconsistent Counterfactual Lag Features
- **Failure:** If a user modifies "promotion=1" on day $t$, simply updating the promotion column without regenerating lag features might ignore temporal dependencies.
- **Mitigation:**
  - The What-If engine runs through the full `FeaturePipeline.transform()` logic or isolates the direct effect of the perturbed exogenous features.
