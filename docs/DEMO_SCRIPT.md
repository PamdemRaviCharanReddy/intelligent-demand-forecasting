# IDFS 5-Minute Interactive Viva Voce Demonstration Script

**Project Title:** Intelligent Demand Forecasting System (IDFS)  
**Target Audience:** University Viva Voce Examination Panel & Faculty Mentors  
**Duration:** Exactly 5 Minutes (Timed Walkthrough)  
**Presenter Role Allocations:**
- **Minute 1 (Architecture & Data Explorer):** Pamdem Ravi Charan Reddy
- **Minute 2 (Forecast Studio & Uncertainty):** Jaiti Vijaya Shravani
- **Minute 3 (Inventory Simulation & Order Decision):** Pottipothu V M Krishna Swamy
- **Minute 4 (Drift Monitoring & Tree-SHAP Explainability):** Vankayalapati Srujana
- **Minute 5 (Executive Reporting & Conclusion):** Team Open for Examiner Q&A

---

## 0. Pre-Flight Demonstration Setup (Terminal Execution)

Run these two commands in the repository root prior to the examiners entering the room:

```powershell
# 1. Activate reproducible virtual environment
.venv\Scripts\activate

# 2. Launch Streamlit portal in browser
streamlit run src/idfs/app/main.py
```

The app will launch at `http://localhost:8501`. Ensure the browser is maximized in Full-Screen mode (press `F11`).

---

## Minute 1: System Overview & Data Explorer (Ravi Charan)

### Actions in UI:
1. Direct the examiners' attention to the **Sidebar**:
   - Point out the unified university title, team enrolment IDs, and mentor names.
2. Ensure **"1. Data Explorer & EDA"** is selected.
3. In the main view:
   - Point out the metric cards: **Total Records (36,500)**, **Unique Stores (5)**, **Product Families (10)**, and the date span.
4. Interact with the dropdowns:
   - Select **Filter Store ID $\to$ S000**.
   - Select **Filter Product Family $\to$ FAM_00**.
   - Hover over the reactive Plotly line chart to demonstrate interactive hover tooltips showing daily sales volume.

### Spoken Script (Ravi Charan):
> *"Respected examiners, welcome to the live demonstration of IDFS. On Page 1, our Data Explorer handles ingestion and automated exploratory profiling. The system continuously validates schemas and missing data. As I filter by Store S000 and Family FAM_00, you can observe historical demand seasonality and promotional spikes rendered dynamically with Plotly. Crucially, all historical rolling features are built with strict zero-leakage shifting, ensuring no future information bleeds into training."*

---

## Minute 2: Forecast Studio & Calibrated Intervals (Jaiti Shravani)

### Actions in UI:
1. In the sidebar, select **"2. Forecast Studio"**.
2. Scroll to the **Benchmark Leaderboard**:
   - Highlight the walk-forward CV comparative table showing WAPE and MASE across all 10 models.
3. In the **Interactive Forecast Inspection** panel:
   - Set **Selected Forecaster** $\to$ `PyTorch-LSTM`.
   - Adjust **Forecast Horizon** slider $\to$ `14 Days`.
   - Toggle **Interval Confidence** $\to$ `90%`.
4. Point to the interactive chart:
   - Trace 1 (Grey): Historical Actuals.
   - Trace 2 (Black): Ground Truth Evaluation Demand.
   - Trace 3 (Blue Shaded Ribbon): Calibrated Conformal Prediction Interval.
   - Trace 4 (Blue Dashed): Model Point Forecast.
5. Highlight the metric callout: **Empirical Coverage = 91.1%** and **Evaluation WAPE = 17.9%**.

### Spoken Script (Jaiti Shravani):
> *"Moving to Page 2, the Forecast Studio displays our multi-paradigm leaderboard. Notice that our PyTorch LSTM and TSB forecasters strongly outperform baselines. To protect against retail uncertainty, we deploy Conformalized Quantile Regression (Novelty N2). As shown by this blue confidence band, IDFS guarantees a 90% finite-sample coverage without assuming a Gaussian error distribution. On our held-out test data, empirical coverage reached 91.1% with tight interval sharpness."*

---

## Minute 3: Cost-Based Inventory Simulation & Reorder Alerts (Krishna Swamy)

### Actions in UI:
1. In the sidebar, select **"3. Inventory & Alerts (N1+N2)"**.
2. Scroll to the **Interactive Replenishment Simulator**:
   - Point out the cost parameter inputs: **Holding Cost (\$0.10)**, **Stockout Cost (\$5.00)**, **Order Cost (\$50.00)**.
   - Select **Replenishment Policy** $\to$ `ConformalQuantile-(R,S) (N1+N2)`.
   - Set **Supplier Lead Time** $\to$ `5 Days` and **Review Period** $\to$ `7 Days`.
3. Highlight the headline metrics:
   - **Service Fill Rate:** `90.6%`
   - **Total Operational Cost:** `~$18,440` (saving over \$1,600 vs. uncalibrated baseline)
4. Show the **Operational Replenishment Alerts** card:
   - Yellow banner: *"REPLENISHMENT ACTION REQUIRED: Policy recommends placing an order of XX units arriving in 5 days."*
5. In the **Planner Decision & Run Recording** card:
   - Select **Planner Decision** $\to$ `APPROVED`.
   - Type note: *"Verified with warehouse space. Approved by Lead Planner."*
   - Click the button: **"Save Run & Approval to Persistent Database"**.
   - Show the green success confirmation: *"Run successfully committed to SQLite storage (Run ID: xxxxxxxx)!"*

### Spoken Script (Krishna Swamy):
> *"On Page 3, Novelty N1 connects the forecast to commercial reality. Rather than relying on simple averages, our discrete-event simulator tracks on-hand inventory, pipeline deliveries, and backlogged demand under asymmetric costs. By setting safety stock according to the conformal upper quantile, the policy maintains a 90.6% fill rate while reducing total holding and stockout penalties by \$1,633 per cycle. Notice our Planner-in-the-Loop workflow: I can record my managerial approval and rationale directly into our embedded SQLite database for enterprise auditability."*

---

## Minute 4: Production Surveillance Drift & Tree-SHAP Explainability (Vankayalapati Srujana)

### Actions in UI:
1. In the sidebar, select **"4. Drift & Explainability (N4+N5)"**.
2. Click **Tab 1: Production Drift Monitor (N4)**:
   - Show the surveillance timeline chart where rolling error is tracked against the adaptive baseline.
   - Point out the Status Metric: **Model Health Status = Healthy / Retrain Triggered**.
3. Click **Tab 2: Tree-SHAP Explainability (N5)**:
   - Point out the **Global Demand Attribution** chart:
     - Explain that the 28-day rolling average accounts for **43.9%** of model decisions, followed by the 14-day average (**17.3%**) and promotional flag (**5.6%**).
   - Point out the **Local Prediction Waterfall** chart showing how promotions push specific days higher.
4. Click **Tab 3: What-If Simulator (N5)**:
   - Toggle **Activate Full-Chain Promotion** $\to$ `ON`.
   - Move **Macro Oil Price Shock** slider $\to$ `+20%`.
   - Show reactive metrics: **Baseline Demand (120,158 units)** $\to$ **Scenario Projected Volume (140,993 units)**, demonstrating a **+17.34% Net Demand Lift**.

### Spoken Script (Vankayalapati Srujana):
> *"Page 4 addresses production operations and explainability. Under Tab 1, Novelty N4 provides automated surveillance: when market shocks cause rolling WAPE to breach 1.5× baseline, the system automatically triggers a retraining alert. Under Tab 2, Novelty N5 uses Tree-SHAP to deconstruct black-box predictions; store planners can see that rolling demand trends and promotions drive 66% of volume. Finally, in Tab 3, our counterfactual What-If simulator allows planners to formulate marketing campaigns and observe elasticity in real time."*

---

## Minute 5: Executive Report Generator & Examiner Q&A (Ravi Charan & Team)

### Actions in UI:
1. In the sidebar, select **"5. Executive Report Generator"**.
2. In the main view:
   - Click **"Generate Comprehensive Executive HTML Report"**.
   - Show the preview of the styled executive report with embedded tables, SHAP attributions, and planner audit trail.
   - Click **"Download Executive HTML Report"** to show instant browser export.
3. Conclude and open the floor for examination.

### Spoken Script (Ravi Charan):
> *"Finally, on Page 5, our executive reporting engine compiles all cv metrics, inventory simulation costs, SHAP driver rankings, and planner approval notes into a standalone, styled HTML document ready for corporate executive review. In conclusion, IDFS satisfies all nine project objectives with full reproducibility, strict leakage prevention, and statistically verified improvements. We now invite your questions."*

---

## Quick Reference for Live Demo Hiccups
- If the browser zoom is awkward: Press `Ctrl + 0` to reset zoom to 100%.
- If Streamlit reloads unexpectedly: The app uses caching (`st.cache_data`); pages reload in under 0.5 seconds.
- If asked about unit test coverage: Open a side terminal and run `python scripts/dev.py check` to display **117 passing automated tests** in 30 seconds.
