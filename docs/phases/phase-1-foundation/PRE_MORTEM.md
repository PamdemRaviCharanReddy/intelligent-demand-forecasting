# Phase 1 Pre-Mortem: Failure Modes & Mitigations

1. **Discontinuous Date Index**: If a store closed for renovation, missing rows cause lag calculation failures downstream.  
   *Mitigation*: Generate a complete Cartesian product of `(date_range x store_id x family)` and left-join, imputing sales=0 and onpromotion=0 for active lifespan.
2. **Missing Exogenous Regressors**: Absence of `oil.csv` or `holidays.csv` crashes ingestion.  
   *Mitigation*: Ingestion loader detects missing optional tables, logs a warning, and creates synthetic stub columns filled with neutral defaults.
3. **Outlier Corruption**: Spurious data-entry errors (e.g. 100,000 units sold on a single day for a corner store) distort variance.  
   *Mitigation*: Implement IQR/Winsorization cleaning threshold ($99.9\text{th}$ percentile clamp) with an audit log of modified records.
4. **Data Type Mismatches**: Date columns parsed as strings or object types instead of `datetime64[ns]`.  
   *Mitigation*: Strict schema parsing with `pd.to_datetime(..., errors='coerce')` and validation assertions.
5. **Memory Bloat on Merges**: Naive SQL-style joins on millions of rows expanding memory.  
   *Mitigation*: Convert store and family identifiers to categorical types before merging.
