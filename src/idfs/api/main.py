"""IDFS Backend REST API service powered by FastAPI and Uvicorn.

Provides high-performance RESTful endpoints for demand forecasting models,
conformal prediction intervals, inventory optimization simulations, and explainability.
"""

from __future__ import annotations

import datetime
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from idfs.api.schemas import (
    DataSummaryResponse,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    InventorySimRequest,
    InventorySimResponse,
    SystemArchitectureResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from idfs.api.services import BackendService

app = FastAPI(
    title="IDFS REST API - Intelligent Demand Forecasting System",
    description=(
        "Backend RESTful microservice providing machine learning forecasting inference, "
        "probabilistic prediction intervals, (R, S) inventory optimization, and Tree-SHAP explainability."
    ),
    version="1.0.0",
    contact={
        "name": "IDFS Engineering Team (ICFAI Special Project)",
        "email": "24stuchh010156@ifheindia.org",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend UI interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = BackendService()


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Root entrypoint linking to interactive Swagger documentation."""
    return {
        "message": "Welcome to IDFS Backend REST API",
        "documentation": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Service liveness probe and health check."""
    return HealthResponse(
        status="healthy",
        service="IDFS Backend REST API",
        version="1.0.0",
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )


@app.get("/api/v1/system/architecture", response_model=SystemArchitectureResponse, tags=["System"])
def system_architecture() -> SystemArchitectureResponse:
    """System component architecture mapping frontend, backend, ML engines, and data layer."""
    return SystemArchitectureResponse(
        frontend={
            "framework": "Streamlit (Multi-Page)",
            "port": 8501,
            "url": "http://localhost:8501",
            "features": [
                "Interactive Data Explorer",
                "Forecast Studio (10 Models)",
                "Inventory Replenishment & Alerts",
                "Tree-SHAP & Counterfactual Simulator",
                "Executive HTML Audit Builder",
            ],
        },
        backend={
            "framework": "FastAPI + Uvicorn (ASGI)",
            "port": 8000,
            "url": "http://localhost:8000",
            "documentation": "http://localhost:8000/docs",
            "protocol": "RESTful JSON / OpenAPI 3.0",
        },
        ml_engines={
            "deep_learning": "PyTorch LSTM (12-Step Lookback)",
            "gradient_boosted": "Global XGBoost & Quantile Trees",
            "statistical": "SARIMAX & Facebook Prophet",
            "intermittent": "Croston SBA & Teunter-Syntetos-Babai (TSB)",
            "ensembles": "Weighted Stacking & Syntetos-Boylan Router",
            "uncertainty": "Conformalized Quantile Regression (CQR)",
        },
        database={
            "storage_engine": "SQLite 3",
            "db_path": "data/runs.db",
            "data_lake": "Apache Parquet (Columnar Compressed)",
            "tables": ["simulation_runs", "fused_sales"],
        },
    )


@app.get("/api/v1/data/summary", response_model=DataSummaryResponse, tags=["Data"])
def get_data_summary() -> DataSummaryResponse:
    """Returns dataset summary metrics (SKU count, store count, date boundaries)."""
    summary = service.get_data_summary()
    return DataSummaryResponse(**summary)


@app.get("/api/v1/forecast/models", tags=["Forecasting"])
def get_supported_models() -> list[dict[str, Any]]:
    """Returns list of all 10 supported forecasting models with their official test benchmarks."""
    return service.list_models()


@app.post("/api/v1/forecast/predict", response_model=ForecastResponse, tags=["Forecasting"])
def predict_forecast(req: ForecastRequest) -> ForecastResponse:
    """Generate multi-step horizon demand forecast with P10-P90 prediction intervals."""
    try:
        res = service.predict_forecast(
            item_id=req.item_id,
            store_id=req.store_id,
            model_name=req.model_name,
            horizon_days=req.horizon_days,
        )
        return ForecastResponse(**res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/inventory/recommend", response_model=InventorySimResponse, tags=["Inventory"])
def simulate_inventory(req: InventorySimRequest) -> InventorySimResponse:
    """Simulate cost-optimal (R, S) replenishment policy under service-level targets."""
    res = service.simulate_inventory(
        service_level=req.service_level,
        lead_time_days=req.lead_time_days,
        holding_cost=req.holding_cost_per_unit,
        stockout_penalty=req.stockout_penalty_per_unit,
        ordering_cost=req.ordering_cost_per_order,
    )
    return InventorySimResponse(**res)


@app.post("/api/v1/whatif/simulate", response_model=WhatIfResponse, tags=["Explainability"])
def simulate_whatif(req: WhatIfRequest) -> WhatIfResponse:
    """Perform counterfactual stress testing on pricing, promotions, and supplier lead times."""
    res = service.simulate_whatif(
        price_change_pct=req.price_change_pct,
        promo_active=req.promo_active,
        lead_time_delta_days=req.lead_time_delta_days,
    )
    return WhatIfResponse(**res)


@app.get("/api/v1/system/runs", tags=["Audit & Persistence"])
def get_audit_runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
    """Retrieve historical execution audit runs stored in SQLite."""
    runs = service.get_runs()
    return runs[:limit]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("idfs.api.main:app", host="127.0.0.1", port=8000, reload=False)
