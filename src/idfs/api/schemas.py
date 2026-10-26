"""Pydantic schemas for IDFS REST API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Health status", examples=["healthy"])
    service: str = Field(description="Service name", examples=["IDFS Backend REST API"])
    version: str = Field(description="Semantic version", examples=["1.0.0"])
    timestamp: str


class SystemArchitectureResponse(BaseModel):
    frontend: dict[str, Any]
    backend: dict[str, Any]
    ml_engines: dict[str, Any]
    database: dict[str, Any]


class DataSummaryResponse(BaseModel):
    total_records: int
    num_skus: int
    num_stores: int
    date_start: str
    date_end: str
    total_sales_volume: float
    categories: list[str]


class ForecastRequest(BaseModel):
    item_id: str = Field(default="I001", description="Item SKU ID")
    store_id: str = Field(default="S001", description="Store ID")
    model_name: str = Field(default="TSB", description="Forecasting model name")
    horizon_days: int = Field(default=28, ge=7, le=90)


class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    model_name: str
    horizon_days: int
    forecast_dates: list[str]
    forecast_values: list[float]
    lower_bound: list[float]
    upper_bound: list[float]
    historical_wape: float
    historical_mase: float


class InventorySimRequest(BaseModel):
    service_level: float = Field(default=0.95, ge=0.80, le=0.999)
    lead_time_days: int = Field(default=7, ge=1, le=30)
    holding_cost_per_unit: float = Field(default=0.50, ge=0.01)
    stockout_penalty_per_unit: float = Field(default=10.0, ge=0.1)
    ordering_cost_per_order: float = Field(default=50.0, ge=0.0)


class InventorySimResponse(BaseModel):
    policy: str
    service_level: float
    lead_time_days: int
    safety_stock: float
    reorder_point: float
    simulated_fill_rate: float
    total_cost: float
    holding_cost: float
    stockout_cost: float
    ordering_cost: float
    stockout_events: int
    orders_placed: int


class WhatIfRequest(BaseModel):
    price_change_pct: float = Field(default=0.0, ge=-50.0, le=50.0)
    promo_active: bool = Field(default=False)
    lead_time_delta_days: int = Field(default=0, ge=-5, le=14)


class WhatIfResponse(BaseModel):
    baseline_demand: float
    counterfactual_demand: float
    demand_delta_pct: float
    baseline_safety_stock: float
    adjusted_safety_stock: float
    primary_driver: str
    interpretation: str
