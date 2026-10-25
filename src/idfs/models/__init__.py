"""Forecasting models: baselines, statistical, ML, deep learning, and routing."""

from idfs.models.base import Forecaster, NaiveForecaster, SeasonalNaiveForecaster
from idfs.models.ensemble import WeightedEnsembleForecaster
from idfs.models.intermittent import CrostonForecaster, TSBForecaster
from idfs.models.lstm import LSTMForecaster
from idfs.models.prophet_model import ProphetForecaster
from idfs.models.router import DemandPatternRouterForecaster, classify_demand_pattern
from idfs.models.statistical import ARIMAForecaster, SARIMAXForecaster
from idfs.models.xgboost import XGBoostForecaster

__all__ = [
    "ARIMAForecaster",
    "CrostonForecaster",
    "DemandPatternRouterForecaster",
    "Forecaster",
    "LSTMForecaster",
    "NaiveForecaster",
    "ProphetForecaster",
    "SARIMAXForecaster",
    "SeasonalNaiveForecaster",
    "TSBForecaster",
    "WeightedEnsembleForecaster",
    "XGBoostForecaster",
    "classify_demand_pattern",
]
