# Phase 1 Brief: Foundation & Data Engineering

## 1. Goal
Construct a robust, schema-enforced, leakage-free data engineering pipeline that accepts raw retail time series (CSV or Parquet, synthetic or real Kaggle Favorita files), validates schemas, handles missing values/outliers/negative sales, imputes missing dates per series, fuses multi-source exogenous tables (stores, holidays, oil, promotions) with graceful degradation, and produces a processed dataset ready for feature engineering alongside an automated EDA report with visual figures.

## 2. Inputs Available
- Synthetic data generated in `data/synthetic/*.parquet` (or real Favorita files if placed in `data/raw/`).
- Schemas locked in `docs/DATA_CONTRACT.md`.
- Default configuration in `configs/default.yaml`.

## 3. Constraints & Operating Environment
- Memory safety: processing must execute comfortably within 12 GB RAM.
- Graceful degradation: if `oil` or `holidays` or `promotions` are absent, pipeline logs a warning and proceeds without crashing.
- Data integrity: never alter raw files; write clean outputs exclusively to `data/processed/`.

## 4. Unknowns & Risks Addressed
- *Missing Dates*: Retail store series may omit dates when stores close or on holidays; we must create a continuous complete date grid per active series with zero sales.
- *Negative Sales*: Raw retail returns can produce negative values; must clamp to zero with logging.
- *Oil Price Gaps*: Weekend or holiday commodity price missing values require forward-fill and initial backfill.
