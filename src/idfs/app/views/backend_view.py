"""Interactive Backend API & System Architecture View (Frontend-Backend integration)."""

from __future__ import annotations

import json
import time
from typing import Any

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"


def check_backend_health() -> tuple[bool, dict[str, Any], float]:
    """Check connectivity and latency to FastAPI backend."""
    t0 = time.time()
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3.0)
        elapsed_ms = (time.time() - t0) * 1000.0
        if resp.status_code == 200:
            return True, resp.json(), elapsed_ms
        return False, {"error": f"HTTP {resp.status_code}"}, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.time() - t0) * 1000.0
        return False, {"error": str(e)}, elapsed_ms


def render_backend_view() -> None:
    """Render the Frontend & Backend Architecture and Live API Explorer."""
    st.title("🔌 Frontend & Backend System Architecture")
    st.markdown(
        "Complete structural separation between the **Interactive Presentation Layer (Frontend)** "
        "and the **FastAPI RESTful Microservice & ML Inference Engine (Backend)**."
    )

    # 1. Live Connection Status Banner
    is_live, _health_data, latency = check_backend_health()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if is_live:
            st.metric(
                "Backend Status", "🟢 ONLINE", help="ℹ️ FastAPI ASGI server running on port 8000"
            )
        else:
            st.metric(
                "Backend Status", "🔴 OFFLINE", help="ℹ️ Backend server not reachable on port 8000"
            )
    with c2:
        st.metric(
            "API Latency", f"{latency:.1f} ms", help="ℹ️ Round-trip HTTP request-response time"
        )
    with c3:
        st.metric("Frontend Port", "8501 (Streamlit)", help="ℹ️ Presentation dashboard port")
    with c4:
        st.metric("Backend Port", "8000 (FastAPI)", help="ℹ️ REST microservice port")

    st.markdown("---")

    # 2. System Architecture Visual Mapping
    st.subheader("🏛️ End-to-End Architectural Decomposition")

    st.markdown(
        """
| Layer | Technology | Primary Responsibilities | Network Address |
| :--- | :--- | :--- | :--- |
| **Frontend** | **Streamlit (v1.50)** | Multi-page dashboard, Plotly interactive visualizations, user filters, audit review | `http://localhost:8501` |
| **API Gateway** | **FastAPI + Uvicorn (ASGI)** | High-throughput REST API, Pydantic input/output validation, OpenAPI/Swagger | `http://localhost:8000` |
| **ML Engine** | **PyTorch & XGBoost** | 12-Step LSTM neural networks, Gradient Boosted Trees, CQR intervals, TSB/Croston | In-Process Engine |
| **Data Lake & DB**| **SQLite & Parquet** | Run audit logging (`runs.db`), Snappy-compressed columnar sales (`fused_sales.parquet`) | File & SQLite Storage |
        """
    )

    st.info(
        "💡 **For viva demonstration:** Examiners can inspect the **Frontend UI** right here on port **8501**, "
        "and open the **Backend REST API Documentation** at [http://localhost:8000/docs](http://localhost:8000/docs) "
        "to test endpoints directly in Swagger UI."
    )

    st.markdown("---")

    # 3. Interactive REST API Console / Live Tester
    st.subheader("🚀 Live Backend API Console")
    st.markdown(
        "Execute live HTTP requests against the FastAPI backend and view raw JSON payloads:"
    )

    endpoint_choice = st.selectbox(
        "Select API Endpoint to Query:",
        [
            "GET  /health (Service Liveness Probe)",
            "GET  /api/v1/system/architecture (System Topology)",
            "GET  /api/v1/data/summary (Dataset & SKU Counts)",
            "GET  /api/v1/forecast/models (Model Accuracy Benchmarks)",
            "POST /api/v1/forecast/predict (Generate Forecast & Intervals)",
            "POST /api/v1/inventory/recommend ((R, S) Policy Simulation)",
            "POST /api/v1/whatif/simulate (Counterfactual Sensitivity)",
            "GET  /api/v1/system/runs (SQLite Audit Logs)",
        ],
        help="ℹ️ Choose an endpoint to test live communication between Streamlit and FastAPI",
    )

    req_payload: dict[str, Any] = {}
    method = "GET"
    path = "/health"

    if "GET  /health" in endpoint_choice:
        method = "GET"
        path = "/health"
    elif "GET  /api/v1/system/architecture" in endpoint_choice:
        method = "GET"
        path = "/api/v1/system/architecture"
    elif "GET  /api/v1/data/summary" in endpoint_choice:
        method = "GET"
        path = "/api/v1/data/summary"
    elif "GET  /api/v1/forecast/models" in endpoint_choice:
        method = "GET"
        path = "/api/v1/forecast/models"
    elif "POST /api/v1/forecast/predict" in endpoint_choice:
        method = "POST"
        path = "/api/v1/forecast/predict"
        c_sku, c_store, c_m = st.columns(3)
        with c_sku:
            item_id = st.text_input("Item ID:", value="I001")
        with c_store:
            store_id = st.text_input("Store ID:", value="S001")
        with c_m:
            model_name = st.selectbox(
                "Model:", ["TSB", "Prophet", "PyTorch-LSTM", "Global XGBoost"]
            )
        req_payload = {
            "item_id": item_id,
            "store_id": store_id,
            "model_name": model_name,
            "horizon_days": 28,
        }
    elif "POST /api/v1/inventory/recommend" in endpoint_choice:
        method = "POST"
        path = "/api/v1/inventory/recommend"
        c_sl, c_lt, c_hc = st.columns(3)
        with c_sl:
            service_level = st.slider("Target Service Level:", 0.80, 0.99, 0.95, 0.01)
        with c_lt:
            lead_time = st.number_input("Lead Time (Days):", min_value=1, max_value=30, value=7)
        with c_hc:
            holding_cost = st.number_input("Holding Cost ($/unit):", min_value=0.1, value=0.5)
        req_payload = {
            "service_level": service_level,
            "lead_time_days": int(lead_time),
            "holding_cost_per_unit": holding_cost,
            "stockout_penalty_per_unit": 10.0,
            "ordering_cost_per_order": 50.0,
        }
    elif "POST /api/v1/whatif/simulate" in endpoint_choice:
        method = "POST"
        path = "/api/v1/whatif/simulate"
        c_p, c_pr, c_l = st.columns(3)
        with c_p:
            p_pct = st.slider("Price Delta (%):", -30.0, 30.0, 0.0, 5.0)
        with c_pr:
            is_promo = st.checkbox("Promotion Active")
        with c_l:
            lt_d = st.slider("Lead Time Shift (Days):", -3, 7, 0)
        req_payload = {
            "price_change_pct": p_pct,
            "promo_active": is_promo,
            "lead_time_delta_days": lt_d,
        }
    elif "GET  /api/v1/system/runs" in endpoint_choice:
        method = "GET"
        path = "/api/v1/system/runs?limit=10"

    col_btn, col_doc1, col_doc2 = st.columns([2, 2, 2])
    with col_btn:
        trigger = st.button("🚀 Send Live HTTP Request", type="primary", use_container_width=True)
    with col_doc1:
        st.link_button(
            "📖 Open Swagger UI (/docs)", f"{BACKEND_URL}/docs", use_container_width=True
        )
    with col_doc2:
        st.link_button("📑 Open ReDoc (/redoc)", f"{BACKEND_URL}/redoc", use_container_width=True)

    if trigger:
        t_start = time.time()
        try:
            target_url = f"{BACKEND_URL}{path}"
            if method == "GET":
                resp = requests.get(target_url, timeout=5.0)
            else:
                resp = requests.post(target_url, json=req_payload, timeout=5.0)
            t_elapsed = (time.time() - t_start) * 1000.0

            st.success(f"**HTTP {resp.status_code} OK** — Completed in **{t_elapsed:.2f} ms**")

            with st.expander("📄 View Formatted Response JSON", expanded=True):
                st.json(resp.json())

            with st.expander("🔍 View Request Headers & Target URL"):
                st.code(
                    f"URL: {target_url}\nMethod: {method}\nPayload: {json.dumps(req_payload, indent=2)}"
                )
        except Exception as err:
            st.error(f"Failed to communicate with backend: {err}")
