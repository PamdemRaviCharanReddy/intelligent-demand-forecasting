"""Unit tests for FastAPI REST API endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from idfs.api.main import app

client = TestClient(app)


def test_api_health_check() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "IDFS Backend REST API"
    assert data["version"] == "1.0.0"


def test_api_system_architecture() -> None:
    res = client.get("/api/v1/system/architecture")
    assert res.status_code == 200
    data = res.json()
    assert data["frontend"]["port"] == 8501
    assert data["backend"]["port"] == 8000
    assert "PyTorch LSTM" in data["ml_engines"]["deep_learning"]


def test_api_data_summary() -> None:
    res = client.get("/api/v1/data/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_records"] > 0
    assert data["num_skus"] > 0
    assert data["num_stores"] > 0


def test_api_forecast_models() -> None:
    res = client.get("/api/v1/forecast/models")
    assert res.status_code == 200
    models = res.json()
    assert isinstance(models, list)
    assert len(models) >= 4
    model_names = [m["model_name"] for m in models]
    assert any("TSB" in name for name in model_names)


def test_api_forecast_predict() -> None:
    payload = {
        "item_id": "I001",
        "store_id": "S001",
        "model_name": "TSB",
        "horizon_days": 14,
    }
    res = client.post("/api/v1/forecast/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["item_id"] == "I001"
    assert len(data["forecast_values"]) == 14
    assert len(data["lower_bound"]) == 14
    assert len(data["upper_bound"]) == 14
    assert data["historical_wape"] > 0


def test_api_inventory_recommend() -> None:
    payload = {
        "service_level": 0.95,
        "lead_time_days": 7,
        "holding_cost_per_unit": 0.50,
        "stockout_penalty_per_unit": 10.0,
        "ordering_cost_per_order": 50.0,
    }
    res = client.post("/api/v1/inventory/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["safety_stock"] > 0
    assert data["reorder_point"] > data["safety_stock"]
    assert data["total_cost"] > 0


def test_api_whatif_simulate() -> None:
    payload = {
        "price_change_pct": 10.0,
        "promo_active": False,
        "lead_time_delta_days": 2,
    }
    res = client.post("/api/v1/whatif/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["demand_delta_pct"] < 0  # Price increase reduces demand
    assert data["adjusted_safety_stock"] > 0
