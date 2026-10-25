"""Leakage-free categorical encoders for IDFS.

Enforces strict train-only fitting: encoders fit solely on training folds
and apply learned parameters to validation/test sets with fallback priors.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class TargetEncoder:
    """Smoothed Empirical Bayes target encoder.

    Formula for category c:
        mu_c = (n_c * mean_c + smoothing * global_mean) / (n_c + smoothing)

    Attributes:
        columns: List of categorical columns to encode.
        target_col: Target continuous column.
        smoothing: Weight assigned to global prior.
    """

    def __init__(
        self,
        columns: list[str],
        target_col: str = "sales",
        smoothing: float = 10.0,
    ) -> None:
        self.columns = columns
        self.target_col = target_col
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.encodings_: dict[str, dict[Any, float]] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> TargetEncoder:
        """Fit target encodings strictly on training fold."""
        self.global_mean_ = float(df[self.target_col].mean())
        self.encodings_ = {}

        for col in self.columns:
            if col not in df.columns:
                continue
            stats = df.groupby(col, observed=True)[self.target_col].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            smoothed = (counts * means + self.smoothing * self.global_mean_) / (
                counts + self.smoothing
            )
            self.encodings_[col] = smoothed.to_dict()

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encodings to DataFrame without refitting."""
        if not self.is_fitted_:
            raise RuntimeError("TargetEncoder must be fitted before transform")

        out_df = df.copy()
        for col in self.columns:
            if col in self.encodings_:
                mapping = self.encodings_[col]
                encoded_col = f"{col}_target_enc"
                out_df[encoded_col] = (
                    out_df[col].astype(object).map(mapping).fillna(self.global_mean_).astype(float)
                )

        return out_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform on the same training slice."""
        return self.fit(df).transform(df)


class FrequencyEncoder:
    """Frequency / count encoder for categorical attributes.

    Maps each category to its relative frequency in the training data.
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.frequencies_: dict[str, dict[Any, float]] = {}
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> FrequencyEncoder:
        """Fit frequency distributions strictly on training fold."""
        self.frequencies_ = {}
        n = len(df)
        for col in self.columns:
            if col in df.columns:
                counts = df[col].value_counts()
                freqs = (counts / n).to_dict()
                self.frequencies_[col] = freqs

        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply frequency mapping to DataFrame without refitting."""
        if not self.is_fitted_:
            raise RuntimeError("FrequencyEncoder must be fitted before transform")

        out_df = df.copy()
        for col in self.columns:
            if col in self.frequencies_:
                mapping = self.frequencies_[col]
                encoded_col = f"{col}_freq_enc"
                out_df[encoded_col] = (
                    out_df[col].astype(object).map(mapping).fillna(0.0).astype(float)
                )

        return out_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform on the same training slice."""
        return self.fit(df).transform(df)
