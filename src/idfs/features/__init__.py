"""Feature engineering pipeline for IDFS."""

from idfs.features.calendar import (
    build_calendar_features,
    build_holiday_proximity_features,
)
from idfs.features.encoders import FrequencyEncoder, TargetEncoder
from idfs.features.pipeline import FeaturePipeline
from idfs.features.transformers import (
    build_diff_features,
    build_lag_features,
    build_rolling_features,
)

__all__ = [
    "FeaturePipeline",
    "FrequencyEncoder",
    "TargetEncoder",
    "build_calendar_features",
    "build_diff_features",
    "build_holiday_proximity_features",
    "build_lag_features",
    "build_rolling_features",
]
