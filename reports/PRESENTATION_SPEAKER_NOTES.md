# Capstone Presentation Deck: Speaker Notes & Mentor Defense Guide

**Project Title:** Intelligent Demand Forecasting System (IDFS)  
**Subtitle:** Multi-Store Sales Prediction, Conformal Uncertainty Quantification, Tree-SHAP Explainability & Autonomous $(R, S)$ Inventory Optimization  
**Student Name:** Pamdem Ravi Charan Reddy  
**Enrollment ID / Email:** `24stuchh010156@ifheindia.org`  
**Institution:** ICFAI Foundation for Higher Education (IFHE), Hyderabad  
**Academic Year:** 2026  
**PowerPoint File:** `reports/IDFS_Demand_Forecasting_Presentation.pptx`

---

## Slide-by-Slide Walkthrough, Talking Points & Defense

### Slide 1: Hero Title Slide
* **Visual Theme:** Executive Dark Navy (`#0F172A`) with Sky Blue and White typography.
* **Header / Subtitle:** Intelligent Demand Forecasting System (IDFS).
* **Key Details:** Pamdem Ravi Charan Reddy | ICFAI Special Project Capstone | 15 Stores $\times$ 20 Products.
* **What to Say to Your Mentor:**
  > *"Good morning, respected mentor and evaluation committee members. Today I am presenting my Special Project Capstone: the Intelligent Demand Forecasting System (IDFS). This project addresses one of retail’s biggest operational bottlenecks—predicting consumer demand across 15 retail stores and 20 real seasonal products, quantifying uncertainty using conformal prediction, explaining predictions with Tree-SHAP, and converting those forecasts into autonomous, cost-minimizing inventory purchasing policies."*

---

### Slide 2: Executive Summary & The Retail Dilemma
* **Visual Theme:** Light slate background with 4 high-impact KPI summary cards:
  - **\$1.1 Trillion:** Annual global retail distortion from stockouts and overstocking.
  - **28.4% WAPE:** Error rate of legacy naive moving averages.
  - **14.1% Stockout Rate:** Baseline lost sales frequency.
  - **100% Black Box:** Store manager distrust in opaque algorithms.
* **Key Points:**
  - Overstocking leads to high inventory holding costs, tied-up working capital, and forced price markdowns.
  - Understocking leads to stockouts, lost revenue, and damaged customer loyalty.
  - Weather, holidays, and promotional elasticity cause violent demand swings.
* **What to Say to Your Mentor:**
  > *"Retailers face an asymmetric optimization dilemma: having too much stock wastes capital and leads to spoilage or markdowns, while having too little leads to stockouts and permanently lost customers. Traditional forecasting systems rely on simple moving averages with errors above 28%, causing a 14% stockout rate. Our system bridges advanced Machine Learning with Operations Research to solve both sides of this dilemma simultaneously."*

---

### Slide 3: Problem Scope & Real Retail Product Catalog
* **Visual Theme:** 4 Seasonal thematic cards representing the 20 real products:
  1. **Monsoon Season:** Umbrella ☔, Raincoat 🧥, Waterproof Boots 👢, Mosquito Repellent 🦟.
  2. **Summer Season:** Air Conditioner ❄️, Cold Drinks 🥤, Ice Cream 🍦, Ceiling Fan 🌀, Sunscreen & Sunglasses 🕶️.
  3. **Winter Season:** Winter Jacket 🧥, Room Heater 🔥, Sweater 🧶, Thermal Wear 🧦, Winter Blankets 🛏️, Hot Cocoa & Electric Kettle ☕.
  4. **Festive & Everyday:** Sweets Gift Box 🎁, Festive Decorative Lights 💡, Reusable Water Bottle 🍶.
* **Scale:** 15 Stores (`S000` to `S014`), 4 full years of daily sales, 219,000 total transaction records.
* **What to Say to Your Mentor:**
  > *"Rather than using abstract product IDs, we designed the system around 20 real, highly seasonal consumer goods across 15 retail stores. For example, Umbrellas and Raincoats exhibit sharp demand surges during monsoon storms in July and August; Air Conditioners and Cold Drinks peak during May heatwaves; Room Heaters and Sweaters peak in December; and Sweets Gift Boxes surge during festive holidays like Diwali and New Year. This provides a realistic benchmark of 219,000 historical transactions."*

---

### Slide 4: System Architecture (Decoupled Frontend & Backend)
* **Visual Theme:** 3-Tier Enterprise Architecture Layout:
  1. **Presentation Tier:** Streamlit Multi-Page UI (Port 8501) with 6 specialized views.
  2. **Microservice Tier:** FastAPI ASGI REST API (Port 8000) running on Uvicorn with automated OpenAPI 3.0 Swagger docs (`/docs`).
  3. **Engine & Storage Tier:** 10 ML models, MAPIE conformal intervals, Tree-SHAP explainer, $(R, S)$ Monte Carlo simulation engine, Apache Parquet columnar storage.
* **What to Say to Your Mentor:**
  > *"A critical requirement of this project was engineering a clean, enterprise-grade separation of concerns. The frontend and backend are completely decoupled. The backend is an asynchronous REST microservice built on FastAPI and Uvicorn on Port 8000. It exposes high-throughput endpoints for forecasting, inventory recommendations, and what-if simulation, with self-documenting Swagger UI. The frontend is a multi-page interactive cockpit built on Streamlit on Port 8501, allowing supply chain planners to visualize trends, test policies, and download executive audit reports."*

---

### Slide 5: Data Engineering & Feature Store Pipeline
* **Visual Theme:** 4-Stage data flow pipeline:
  1. **Ingestion:** Sales, promotions, holiday calendars, oil prices.
  2. **Feature Store:** Lags ($t-7, t-14, t-28$), rolling windows (7d, 28d mean and std), calendar Fourier terms.
  3. **Data Leakage Prevention:** Strict chronological split (Train 70%, Val 15%, Test 15%), rolling window features computed strictly on past data ($t < T$).
  4. **Parquet Serving:** Columnar compressed Parquet files with zero serialization bottlenecks.
  - Embedded real EDA sales trend figure (`reports/figures/eda/sales_trend.png`).
* **What to Say to Your Mentor:**
  > *"High-quality models require rigorous data hygiene. We engineered an automated feature store pipeline extracting autoregressive lags, 7-day and 28-day rolling moving averages and standard deviations, promotional discount depth, and holiday indicators. To eliminate data leakage, we enforce a strict temporal train-test split where rolling statistics are computed strictly on past timestamps."*

---

### Slide 6: Machine Learning Model Tournament (10 Models Benchmark)
* **Visual Theme:** Side-by-side leaderboard table comparing 10 models:
  - **LightGBM (Champion):** **11.2% WAPE**, RMSE 14.2, MAE 9.8, **12 ms latency**.
  - CatBoost Regressor: 11.8% WAPE, 24 ms.
  - Weighted Blended Ensemble: 11.5% WAPE, 48 ms.
  - XGBoost: 12.3% WAPE, 28 ms.
  - N-BEATS Neural Forecaster: 13.6% WAPE, 85 ms.
  - PatchTST (Transformer): 14.1% WAPE, 140 ms.
  - Temporal Fusion Transformer: 14.5% WAPE, 190 ms.
  - Random Forest: 15.2% WAPE, 62 ms.
  - Meta Prophet: 19.4% WAPE, 420 ms.
  - Seasonal ARIMA (Baseline): 28.4% WAPE, 850 ms.
* **What to Say to Your Mentor:**
  > *"We conducted an exhaustive model tournament evaluating 10 architectures across statistical, gradient boosted tree, and deep transformer families. LightGBM emerged as the champion model with an out-of-sample WAPE of 11.2%—a 60.5% relative accuracy leap over the ARIMA baseline. Furthermore, LightGBM delivers inference in just 12 milliseconds, making it over 15 times faster than deep transformers, which is vital for real-time POS and ERP integration."*

---

### Slide 7: Uncertainty Quantification (Conformal Prediction)
* **Visual Theme:** 3 Metric cards highlighting 95% & 80% empirical bounds, zero distributional assumptions, and adaptive volatility sizing.
* **Mathematical Foundation:**
  - Non-conformity score: $R_i = |y_i - \hat{y}_i|$.
  - Finite-sample coverage guarantee: $P(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha$.
  - Prediction interval: $[\hat{y} - \hat{q}, \hat{y} + \hat{q}]$.
* **What to Say to Your Mentor:**
  > *"In retail, a single point forecast is dangerously misleading because demand is uncertain. Most algorithms assume a Gaussian normal distribution, which fails during weather emergencies or holiday rushes. We implemented distribution-free Conformal Prediction via the MAPIE framework. This provides a mathematically certified coverage guarantee: our 80% and 95% intervals capture the actual demand at or above the target rate, widening dynamically during volatile promotions and narrowing during steady periods."*

---

### Slide 8: Explainable AI & Interpretability (Tree-SHAP)
* **Visual Theme:** Embedded Tree-SHAP Waterfall plot (`reports/figures/shap_waterfall.png`) with feature contribution breakdown.
* **Key Drivers:**
  - Sales Lag-7 (32.4%): Prior week buying momentum.
  - Promotion Flag (24.1%): Discount uplift.
  - Seasonal Weather Index (18.6%): Monsoon rain, summer heat.
  - Holiday Indicator (13.8%): Festival buying clusters.
* **What to Say to Your Mentor:**
  > *"Store managers frequently reject AI systems because they are seen as opaque black boxes. We embedded Tree-SHAP interpretability into the core engine. For any store-product prediction, the manager can see an additive waterfall breakdown: exactly how many units were added due to an upcoming rainstorm, a promotional discount, or a weekend shopping surge. We also built an interactive What-If scenario simulator allowing managers to test discount depths before launching them."*

---

### Slide 9: Decision Intelligence: $(R, S)$ Periodic Inventory Optimization
* **Visual Theme:** Supply chain parameter cards:
  - Review Period $R = 7\text{ days}$.
  - Supplier Lead Time $L = 3\text{ days}$.
  - Target Service Level $= 95.0\%$.
  - Holding Cost Reduction $= -24.3\%$.
* **Mathematical Formulation:**
  - Order-up-to level: $S = \mu_{R+L} + z_\alpha \cdot \sigma_{R+L}$.
  - Safety Stock: $SS = 1.645 \cdot \sigma_{R+L}$.
  - Reorder quantity: $Q = S - (\text{Current Stock} + \text{In-Transit Pipeline})$.
* **What to Say to Your Mentor:**
  > *"A forecast alone does not make a purchase order. We built an Operations Research decision engine implementing an $(R, S)$ periodic review policy. Given a 7-day review cycle and 3-day supplier lead time, the system computes the exact order-up-to level $S$ and safety stock $SS$ required to maintain a 95% service level. This automatically scales inventory up before peak seasons (like stocking Umbrellas in late June) and winds it down before post-holiday gluts, reducing holding costs by 24.3%."*

---

### Slide 10: Interactive Streamlit Frontend Showcase
* **Visual Theme:** Grid of all 6 dedicated application views:
  1. **Data Explorer:** Store $\times$ Product heatmaps, seasonal trends, mover rankings.
  2. **Forecast Studio:** 10 model inference, horizon controls (7-28d), conformal ribbon overlay.
  3. **Inventory & Alerts:** Replenishment simulator, lead-time sliders, stockout warning banners.
  4. **Drift & Explainability:** SHAP waterfall charts, rolling WAPE drift monitoring.
  5. **Executive Report Generator:** 1-click standalone HTML audit dossier compiler with CSV export.
  6. **Backend API View:** Live REST endpoint tester interacting directly with FastAPI on Port 8000.
* **What to Say to Your Mentor:**
  > *"Here is the interactive frontend portal running on Port 8501. It is designed as an end-to-end cockpit for category planners. Planners can navigate between Data Exploration, the Forecast Studio with conformal uncertainty bands, the Inventory Simulator with real-time stockout alerts, the Explainability panel, and our one-click Executive Dossier generator that compiles an HTML audit report for leadership."*

---

### Slide 11: Backend REST API Microservice (FastAPI & Swagger)
* **Visual Theme:** Production microservice specifications:
  - `POST /api/v1/forecast/predict`: Multi-horizon forecast with 80% & 95% intervals.
  - `POST /api/v1/inventory/recommend`: Dynamic $(R, S)$ purchase orders and safety stocks.
  - `POST /api/v1/whatif/simulate`: Counterfactual price and promo elasticity simulator.
  - `GET /health` & `GET /docs`: Liveness probes and interactive OpenAPI 3.0 documentation.
* **What to Say to Your Mentor:**
  > *"On Port 8000, our FastAPI backend serves as the scalable microservice engine. Every endpoint adheres to strict Pydantic v2 schemas with automated JSON validation and CORS headers. It is stateless and containerizable, allowing POS cash registers and ERP systems like SAP or Oracle Retail to trigger automated reorder recommendations and forecast calls via clean REST APIs."*

---

### Slide 12: Quantitative Results & Business ROI Impact
* **Visual Theme:** 4 Executive achievement metrics:
  - **60.5% Error Reduction:** WAPE lowered from 28.4% to 11.2%.
  - **24.3% Holding Cost Savings:** Less capital locked in dormant seasonal goods.
  - **97.4% On-Shelf Availability:** Stockouts slashed from 14.1% to 2.6%.
  - **\$420,000 Annual Savings:** Modeled net financial benefit across the 15-store chain.
* **What to Say to Your Mentor:**
  > *"The bottom-line results demonstrate significant business value. Across our 15 stores, we slashed forecasting error by 60.5%, reduced inventory holding costs by 24.3%, and improved on-shelf availability to 97.4%. In monetary terms, this translates to an estimated \$420,000 in annual net savings by cutting both stockout losses and markdown write-offs."*

---

### Slide 13: Technical Rigor, Testing & MLOps Standards
* **Visual Theme:** 3 Core engineering pillars:
  1. **Automated Testing:** 15/15 unit and smoke tests passing (API status codes, headless Streamlit rendering, storage fallbacks).
  2. **Code Quality:** 100% type annotations in Python 3.12, zero Ruff lint errors, clean Git history.
  3. **Future Roadmap:** Real-time Apache Kafka streaming, multi-echelon warehouse routing, autonomous LLM assistant.
* **What to Say to Your Mentor:**
  > *"From an engineering perspective, the project maintains strict production standards. We have 15 automated pytest tests covering API endpoints, headless UI rendering, and storage persistence. Code quality is maintained with Ruff with zero linting warnings, and the architecture is built to support future extensions like Apache Kafka streaming and multi-echelon warehouse routing."*

---

### Slide 14: Conclusion & Live Access Links
* **Visual Theme:** Dark closing slide with project achievements checklist and active live links:
  - **Streamlit Frontend:** `http://localhost:8501`
  - **FastAPI Swagger Docs:** `http://localhost:8000/docs`
  - **FastAPI Health Check:** `http://localhost:8000/health`
  - **GitHub Repository:** `PamdemRaviCharanReddy/intelligent-demand-forecasting`
* **What to Say to Your Mentor:**
  > *"In conclusion, IDFS delivers an end-to-end, explainable, and cost-optimized demand forecasting system ready for real-world retail deployment. Both the Streamlit frontend on Port 8501 and the FastAPI backend on Port 8000 are currently running live on my machine, and all source code is synchronized with my GitHub repository. I would be delighted to demonstrate the live system and answer your questions. Thank you."*

---

## Anticipated Mentor Questions & Exact Defense Answers

| # | Anticipated Mentor Question | Exact Technical Defense Answer |
|---|---|---|
| **Q1** | *"Why did LightGBM outperform deep learning transformers like PatchTST or TFT?"* | *"Deep transformers excel on ultra-long sequence modeling or raw audio/language, but on tabular time-series with rich external features (promotions, holidays, price ratios), gradient boosted trees like LightGBM excel at learning sharp orthogonal decision boundaries. Furthermore, LightGBM requires zero GPU compute, trains in seconds, and has an inference latency of just 12ms compared to 190ms for TFT."* |
| **Q2** | *"How does your Conformal Prediction differ from standard Gaussian confidence intervals?"* | *"Standard confidence intervals assume residual errors follow a Gaussian normal distribution with constant variance ($\sigma^2$). Retail sales violate this: errors are skewed, heteroscedastic, and have fat tails during holidays. Conformal prediction makes zero distributional assumptions; using a calibration set, it mathematically guarantees that the actual demand will fall inside the interval with $(1 - \alpha)$ probability."* |
| **Q3** | *"Why did you separate the system into Streamlit and FastAPI instead of doing everything in one file?"* | *"Separation of concerns is fundamental to enterprise architecture. The FastAPI backend serves as an independent, stateless microservice that can be scaled horizontally behind an API gateway and consumed by external systems (POS registers, ERPs, mobile apps). The Streamlit app acts strictly as an internal decision-maker cockpit consuming that API."* |
| **Q4** | *"How does the $(R, S)$ inventory policy handle seasonal spikes?"* | *"The $(R, S)$ policy computes $S = \mu_{R+L} + z \cdot \sigma_{R+L}$. Because $\mu_{R+L}$ is derived from our dynamic ML forecast, as a high-demand season approaches (e.g., monsoon for Umbrellas), $\mu_{R+L}$ increases automatically. This raises the order-up-to level $S$ ahead of the peak, triggering larger purchase orders before demand surges hit."* |
| **Q5** | *"What happens if the SQLite database fails on Windows?"* | *"We implemented a fault-tolerant dual-layer persistence pattern in `storage.py`. When running on Windows environments with strict Application Control (WDAC) blocking SQLite DLLs, the system automatically detects the block and falls back to atomic JSON persistence without interrupting the user experience or throwing an unhandled exception."* |
