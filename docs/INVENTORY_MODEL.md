# Inventory Model & Cost Simulation Formulation

This document details the mathematical formulation, policy rules, and cost simulation mechanics of the Inventory Decision Engine (Novelty N1).

---

## 1. Motivation: The Forecast-to-Decision Gap

Standard supply chain forecasting projects stop at error metrics like RMSE or MAE. However, in retail inventory management, an error that causes an overstock carries a fundamentally different financial penalty (marginal holding cost $c_h$) than an error that causes an understock (lost margin, stockout penalty $c_s$). Furthermore, replenishment orders are subject to supplier lead-time delays ($L$).

The primary novelty of IDFS is demonstrating that a calibrated, uncertainty-aware forecast translates directly into lower inventory holding and stockout costs compared to standard baseline policies.

---

## 2. Inventory Policy: Periodic Review $(R, S)$

IDFS implements a standard retail **Periodic Review Order-Up-To Policy**:
- **Review Period ($R$)**: Every $R$ days (default $R = 7$ days, representing weekly supplier review), the planner reviews current stock.
- **Supplier Lead Time ($L$)**: Orders placed at time $t$ arrive at time $t + L$ (default $L = 5$ days; evaluated across $L \in [2, 7]$).
- **Protection Horizon**: The inventory ordered at review time $t$ must protect against stockouts across the entire combined period $P = R + L$ days.

### A. Order-Up-To Level Calculation ($S$)
$$S_t = \sum_{k=1}^{R+L} \hat{D}_{t+k} + \text{SS}_t$$
where:
- $\sum_{k=1}^{R+L} \hat{D}_{t+k}$ is the cumulative point demand forecast over the protection horizon.
- $\text{SS}_t$ is the dynamic safety stock buffer.

### B. Dynamic Safety Stock ($\text{SS}$)
Using the calibrated forecast standard error $\hat{\sigma}_f$ over the lead-time window:
$$\text{SS}_t = z_{\alpha} \cdot \hat{\sigma}_f \cdot \sqrt{R + L}$$
where $z_{\alpha}$ is the normal critical value corresponding to target cycle service level $\alpha$ (e.g. for $\alpha = 0.95$, $z_{0.95} \approx 1.645$).

### C. Order Quantity Decision ($Q_t$)
At review day $t$:
$$\text{Inventory Position}_t = I_t + \text{Orders In Transit}_t - \text{Backorders}_t$$
$$Q_t = \max\Big(0, \; S_t - \text{Inventory Position}_t\Big)$$
On non-review days, $Q_t = 0$.

---

## 3. Daily Simulation State Engine

The simulation steps through each day $t = 1, 2, \dots, T$:
1. **Order Delivery**: Outstanding orders with delivery date $t$ are added to on-hand inventory:
   $$I_t^{\text{start}} = I_{t-1} + \text{Arriving Orders}_t$$
2. **Demand Realization & Fulfillment**:
   Customer demand $D_t$ arrives:
   $$\text{Fulfilled}_t = \min(I_t^{\text{start}}, D_t)$$
   $$\text{Stockout Units}_t = \max(0, D_t - I_t^{\text{start}})$$
   $$I_t = \max(0, I_t^{\text{start}} - D_t)$$
3. **Review & Ordering**: If $t \pmod R == 0$, calculate target $S_t$ using the model's future $R+L$ forecast, and schedule order $Q_t$ to arrive at $t + L$.
4. **Cost Accrual**:
   $$\text{Cost}_t = c_h \cdot I_t + c_s \cdot \text{Stockout Units}_t + c_o \cdot \mathbb{I}(Q_t > 0)$$

---

## 4. Benchmark: Naive Historical Moving Average Policy

To prove the commercial value of ML forecasting, IDFS compares against `NaivePolicy`:
- Instead of using machine learning forecasts, `NaivePolicy` projects future demand as a rolling historical 28-day moving average: $\bar{D} = \frac{1}{28}\sum_{i=1}^{28} y_{t-i}$.
- Order-up-to level:
  $$S_t^{\text{naive}} = \bar{D} \cdot (R + L) + z \cdot s_D \cdot \sqrt{R + L}$$
- Evaluated on the exact same simulated demand trajectory and cost parameters.

---

## 5. Economic Parameter Defaults & Sensitivity Grids

| Parameter | Symbol | Default Value | Sensitivity Grid | Unit |
|:---|:---:|:---:|:---:|:---:|
| Holding Cost | $c_h$ | $\$0.10$ | $[0.05, 0.20]$ | \$ / unit / day |
| Stockout Penalty | $c_s$ | $\$5.00$ | $[2.00, 15.00]$ | \$ / unit unmet |
| Order Placement Cost | $c_o$ | $\$50.00$ | $[10.00, 100.00]$ | \$ / replenishment |
| Supplier Lead Time | $L$ | $5$ days | $[2, 3, 5, 7]$ | days |
| Review Period | $R$ | $7$ days | $[1, 7, 14]$ | days |
| Target Service Level | $\alpha$ | $95\%$ | $[90\%, 95\%, 99\%]$ | percentage |
