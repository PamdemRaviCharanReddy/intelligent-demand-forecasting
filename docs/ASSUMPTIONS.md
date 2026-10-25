# Project Assumptions & Constraints Register

This register documents all technical, economic, domain, and environmental assumptions underlying the design of IDFS.

---

## 1. Domain & Data Assumptions

1. **Sales as a Proxy for Latent Demand (Censored Demand Caveat)**:
   - *Assumption*: Point-of-sale recorded transactions represent true customer demand $D_t \approx S_t$ for periods where stock is available ($I_t > 0$).
   - *Limitation*: When a stockout occurs ($I_t = 0$), actual customer demand is unobservable and sales drop to zero. Unless unconstrained demand estimation is applied, historical sales under-represent true demand during stockout periods.
   - *Mitigation*: Our inventory simulation records simulated stockouts explicitly and tracks unmet demand.

2. **Temporal Stationarity & Cyclicality**:
   - *Assumption*: Consumer shopping patterns exhibit strong 7-day weekly seasonality (weekend peaks) and annual calendar seasonality, with predictable promotional uplifts.
   - *Mitigation*: The feature pipeline builds cyclical sine/cosine encoders and shifted rolling averages; the drift monitor (N4) flags structural regime shifts.

3. **Exogenous Information Precedence**:
   - *Assumption*: Holiday schedules and promotional campaigns are planned in advance and known at forecast time $t$ for future horizon steps $t+1, \dots, t+H$.
   - *Limitation*: Commodity prices (oil) are not known in advance for future horizons. In inference, oil prices are assumed to persist via random walk or forward-fill.

---

## 2. Supply Chain & Inventory Assumptions

1. **Replenishment Lead Time ($L$)**:
   - *Assumption*: Deterministic or bounded stochastic supplier lead time between $2$ and $7$ days (default $L = 5$ days).
   - *Mitigation*: Sensitivity analyses evaluate model cost savings across $L \in \{2, 3, 5, 7\}$ days.

2. **Cost Parameters**:
   - **Unit Daily Holding Cost ($c_h$)**: Default $\$0.10$ per unit per day. Reflects warehousing, insurance, cost of capital, and shrinkage.
   - **Unit Stockout Penalty ($c_s$)**: Default $\$5.00$ per unit of unmet demand. Reflects lost sales margin and customer goodwill loss ($c_s / c_h = 50$, representing typical retail asymmetry).
   - **Fixed Order Placement Cost ($c_o$)**: Default $\$50.00$ per replenishment PO. Reflects procurement handling, shipping base charge, and receiving labor.

3. **Fulfillment Policy**:
   - Backorders are not accumulated across long horizons in retail groceries (unsatisfied grocery demand is lost rather than backordered). The simulation treats unmet demand as lost sales during the period.

---

## 3. Computational & Platform Assumptions

1. **Local Single-Node Execution**:
   - *Assumption*: All code must run smoothly on a standard student laptop (AMD 8-core CPU, 12GB RAM, NVIDIA RTX 3050 Laptop GPU) without requiring cloud clusters.
   - *Mitigation*: Global models are optimized for vectorization; local models (ARIMA) are restricted to a representative 20–50 series benchmark subset.

2. **Virtual Environment Isolation**:
   - System Python (3.14) is avoided; isolated Python 3.12 managed via `uv` in `.venv` ensures reproducible dependencies.

---

## 4. Academic & Project Metadata Assumptions

1. **Mentors**: Listed as Dr. Sukanta Das and Dr. Sandeep Kumar Panda `[VERIFY]`, acknowledging both faculty members from report and presentation drafts.
2. **Student Roll Number**: Jaiti Vijaya Shravani is treated as `24STUCHH010756` (reconciling the extra digit `01010756` found in drafts) `[VERIFY]`.
3. **Project Deadline**: Set to `deadline: null` in configurations; executed against an 8-week structured roadmap.
