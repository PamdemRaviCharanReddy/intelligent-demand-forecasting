# Data Contract & Schema Specifications

This document defines the formal data contracts across raw ingestion, cleaned multi-source fusion, engineered feature tables, and synthetic datasets. Every module in the pipeline asserts conformance to these specifications.

---

## 1. Primary Raw Data Schemas (Favorita / Synthetic)

### A. Sales Transactions Table (`sales`)
- **Primary Key**: `(date, store_id, family)`
- **Frequency**: Daily (`freq='D'`)
- **Columns**:
  | Column Name | Type | Nullable | Validation Constraints / Domain | Description |
  |:---|:---:|:---:|:---|:---|
  | `date` | `datetime64[ns]` | No | Must be valid calendar date; monotonic per series | Transaction observation date |
  | `store_id` | `string` | No | Known store identifier (e.g. `S001` or `1`) | Unique retail store code |
  | `family` | `string` | No | Categorical product family (e.g. `GROCERY_I`) | Merchandise department/category |
  | `sales` | `float64` | No | $\ge 0.0$; non-negative float or integer | Units or currency volume sold |
  | `onpromotion` | `int64` / `bool` | No | $\in \{0, 1\}$ or count of items on promo | Active promotion indicator |

### B. Store Metadata (`stores`)
- **Primary Key**: `store_id`
- **Columns**:
  | Column Name | Type | Nullable | Domain / Constraints | Description |
  |:---|:---:|:---:|:---|:---|
  | `store_id` | `string` | No | Matches foreign key in `sales` | Unique store identifier |
  | `city` | `string` | No | Non-empty text | City location |
  | `state` | `string` | No | Non-empty text | State / province |
  | `type` | `string` | No | Categorical (e.g. `A`, `B`, `C`, `D`, `E`) | Store format / tier |
  | `cluster` | `int64` | No | $1 \le \text{cluster} \le 17$ | Demographic grouping cluster |

### C. Holiday & Calendar Events (`holidays`)
- **Primary Key**: `(date, name)`
- **Columns**:
  | Column Name | Type | Nullable | Domain / Constraints | Description |
  |:---|:---:|:---:|:---|:---|
  | `date` | `datetime64[ns]` | No | Valid calendar date | Date of celebration / holiday |
  | `type` | `string` | No | `Holiday`, `Event`, `Additional`, `Transfer` | Holiday classification |
  | `name` | `string` | No | Non-empty string | Official name of event |
  | `transferred`| `bool` | No | `True` / `False` | Whether official day off was moved |

### D. Exogenous Commodity Regressor (`oil`)
- **Primary Key**: `date`
- **Columns**:
  | Column Name | Type | Nullable | Constraints | Description |
  |:---|:---:|:---:|:---|:---|
  | `date` | `datetime64[ns]` | No | Daily calendar date | Observation day |
  | `oil_price` | `float64` | Yes | $> 0.0$ when present | Daily crude oil price (WTI index) |

---

## 2. Processed & Fused Representation (`data/processed/`)

The fused dataset merges historical sales with store metadata, holiday indicators, and oil prices on `(date, store_id)`:
```
Columns:
- date: datetime64[ns]
- store_id: category
- family: category
- sales: float64
- onpromotion: int64
- city: category
- state: category
- store_type: category
- store_cluster: int64
- is_holiday: int64 (0 or 1)
- holiday_type: category (or 'None')
- oil_price: float64 (forward-filled, then backward-filled for initial gap)
```

---

## 3. Engineered Feature Matrix (`idfs.features`)

For any target day $t$, all features are derived from information strictly available at $\tau < t$:

| Feature Group | Column Naming Pattern | Definition / Formula | Leakage Protection |
|:---|:---|:---|:---|
| **Lags** | `sales_lag_1`, `sales_lag_2`, ..., `sales_lag_14` | $y_{t-k}$ | $k \ge 1$ guaranteed |
| **Rolling Means** | `sales_roll_mean_7`, `sales_roll_mean_28` | $\frac{1}{w}\sum_{i=1}^w y_{t-i}$ | Applied on `.shift(1)` |
| **Rolling Stds** | `sales_roll_std_7`, `sales_roll_std_28` | $\sqrt{\frac{1}{w-1}\sum_{i=1}^w (y_{t-i} - \bar{y})^2}$ | Applied on `.shift(1)` |
| **Calendar Cyclical** | `sin_dow`, `cos_dow`, `sin_doy`, `cos_doy` | $\sin(2\pi d / D)$, $\cos(2\pi d / D)$ | Deterministic from date $t$ |
| **Holiday Distance**| `days_to_next_holiday`, `days_since_last_holiday` | Signed day delta to calendar events | Pre-scheduled calendar info |
| **Promotion Lag/Lead**| `promo_lag_1`, `promo_lead_0` | Current known promo status | Future promotions are known retail plans |

---

## 4. Boundary Assertions & Data Quality Rules

1. **Strict Non-Negativity**: Demand cannot be negative. If uploaded raw data contains negative entries (e.g. product returns), the ingestion layer logs a warning and clamps values to zero: $y_t = \max(0.0, y_t)$.
2. **Missing Date Imputation**: Missing dates within an active store-family series are imputed with $y=0$ (representing days with no recorded transactions or closed stores) rather than left as discontinuous index gaps.
3. **Censored Demand Caveat**: In retail point-of-sale data, recorded sales $S_t$ equal true demand $D_t$ only when inventory $I_t > 0$. When stock is exhausted ($I_t = 0$), sales drop to 0 even if latent customer demand is high. IDFS explicitly flags this unobserved demand censoring in `docs/ASSUMPTIONS.md`.
4. **Missing Source Graceful Fallback**:
   - If `oil.csv` is omitted: oil price column is filled with zeros and flagged as disabled.
   - If `holidays.csv` is omitted: default country calendar is queried via Python `holidays` package; if unavailable, holiday indicators default to 0.
