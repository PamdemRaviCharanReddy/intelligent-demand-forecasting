"""Configuration loading and validation for IDFS.

All experiment configuration flows through this module. No magic numbers in code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data sources and processing."""

    source: str = "synthetic"  # "synthetic" | "favorita" | "m5"
    synthetic_n_stores: int = 15
    synthetic_n_families: int = 20
    synthetic_n_series: int = 300
    synthetic_n_days: int = 730  # 2 years
    synthetic_seed: int = 42
    synthetic_intermittent_frac: float = 0.15
    synthetic_drift_start_day: int = 600
    date_col: str = "date"
    target_col: str = "sales"
    series_id_cols: list[str] = field(default_factory=lambda: ["store_id", "family"])


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/validation/test splits."""

    test_days: int = 28  # Final held-out test window (untouched until Phase 7)
    val_days: int = 14  # Each validation fold
    n_folds: int = 3  # Number of walk-forward folds
    gap_days: int = 0  # Gap between train end and val start (prevents leakage)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for forecasting."""

    horizon: int = 14  # Default 14-day daily forecast
    frequency: str = "D"  # "D" | "W" | "M"


@dataclass(frozen=True)
class XGBoostConfig:
    """XGBoost hyperparameters."""

    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10
    early_stopping_rounds: int = 50
    seed: int = 42


@dataclass(frozen=True)
class LSTMConfig:
    """LSTM hyperparameters — CPU-feasible defaults."""

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 28
    batch_size: int = 64
    max_epochs: int = 50
    learning_rate: float = 0.001
    patience: int = 10  # Early stopping patience
    seed: int = 42


@dataclass(frozen=True)
class InventoryConfig:
    """Inventory simulation parameters — configurable assumptions."""

    lead_time_days: int = 5
    review_period_days: int = 7
    service_level: float = 0.95  # Target service level (z ≈ 1.645)
    holding_cost_per_unit_day: float = 0.10
    stockout_cost_per_unit: float = 5.00
    ordering_cost_per_order: float = 50.00
    initial_stock_days: int = 30  # Start with 30 days of avg demand
    policy: str = "periodic"  # "periodic" | "continuous"


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    project_name: str = "IDFS"
    seed: int = 42
    deadline: str | None = None  # null = unknown
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a dict to a dataclass, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    filtered = {}
    for key, value in data.items():
        if key in field_types:
            ft = field_types[key]
            # Resolve string annotations for nested dataclasses
            if isinstance(ft, str):
                ft = eval(ft)
            if dataclasses.is_dataclass(ft) and isinstance(value, dict):
                filtered[key] = _dict_to_dataclass(ft, value)
            else:
                filtered[key] = value
    return cls(**filtered)


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load project configuration from a YAML file.

    Falls back to defaults if no path given or file not found.

    Args:
        path: Path to YAML config file. If None, uses defaults.

    Returns:
        Validated ProjectConfig dataclass.
    """
    if path is None:
        return ProjectConfig()

    p = Path(path)
    if not p.exists():
        return ProjectConfig()

    with open(p) as f:
        raw = yaml.safe_load(f) or {}

    return cast(ProjectConfig, _dict_to_dataclass(ProjectConfig, raw))


def config_hash(cfg: ProjectConfig) -> str:
    """Generate a short deterministic hash of a config for run identification.

    Args:
        cfg: The project configuration.

    Returns:
        8-character hex hash string.
    """
    import dataclasses

    d = dataclasses.asdict(cfg)
    serialized = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]
