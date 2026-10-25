# Phase 2 Pre-Mortem: Failure Modes & Mitigations

1. **Autoregressive Feature Leakage During Multi-Step Forecasting**: When forecasting 14 days ahead, using actual sales from day $t+1$ to predict day $t+2$ causes extreme leakage.  
   *Mitigation*: Direct forecasting or recursive lag updating. In direct/lag-14 forecasting, all features use lags $k \ge 14$, OR recursive autoregression updates lag buffers with predicted values step-by-step.
2. **Target Encoder Overfitting**: Computing average store sales using the entire dataset leaks target information from validation folds into training features.  
   *Mitigation*: Target encoders fit strictly on the training fold slice, applying out-of-fold smoothing or additive prior regularization, and transform validation slices without refitting.
3. **Mismatched Series Indexing in Global XGBoost**: Reconstructed predictions scrambled across stores and product families.  
   *Mitigation*: Enforce deterministic row sorting on `(store_id, family, date)` before matrix conversion and carry series metadata through prediction outputs.
4. **XGBoost Memory Spikes on High-Cardinality One-Hot Encoding**: One-hot encoding 54 stores and 33 families inflates matrix dimensionality to hundreds of sparse columns.  
   *Mitigation*: Leverage XGBoost's native categorical support (`enable_categorical=True`) and integer/frequency encoding instead of dense one-hot matrices.
5. **Cold-Start / Unseen Categories in Validation Folds**: New store or family identifiers absent during training fold.  
   *Mitigation*: Target and categorical encoders handle unseen values via global prior fallback.
