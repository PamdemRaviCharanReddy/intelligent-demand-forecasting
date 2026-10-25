"""Tree-SHAP explainability for gradient boosted demand models (Novelty N5).

Computes exact Shapley feature attributions, decomposing individual predictions
and portfolio-wide demand drivers into interpretable additive contributions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from idfs.models.xgboost import XGBoostForecaster

logger = logging.getLogger(__name__)


class TreeSHAPExplainer:
    """Tree-SHAP attribution engine for gradient boosted decision trees."""

    def __init__(self, forecaster: XGBoostForecaster) -> None:
        if not forecaster.is_fitted or forecaster.model_ is None:
            raise RuntimeError("XGBoost forecaster must be fitted before creating SHAP explainer")
        self.forecaster = forecaster
        self.pipeline = forecaster.pipeline_
        self.explainer = shap.TreeExplainer(forecaster.model_)

    def explain(
        self,
        features_df: pd.DataFrame,
        max_samples: int = 500,
        random_seed: int = 42,
    ) -> tuple[shap.Explanation, pd.DataFrame]:
        """Compute Shapley values on a representative subset of features.

        Args:
            features_df: Preprocessed or raw DataFrame containing feature columns.
            max_samples: Maximum number of rows to evaluate for efficiency.
            random_seed: Random state for deterministic subsampling.

        Returns:
            tuple of (shap.Explanation object, sample feature DataFrame)
        """
        if self.pipeline is not None and not set(self.pipeline.feature_cols_).issubset(
            set(features_df.columns)
        ):
            transformed = self.pipeline.transform(features_df)
            X = transformed[self.pipeline.feature_cols_].copy()
        else:
            feat_cols = (
                self.pipeline.feature_cols_
                if self.pipeline is not None
                else [c for c in features_df.columns if c not in ["date", "sales"]]
            )
            X = features_df[feat_cols].copy()

        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")

        # Subsample if necessary
        if len(X) > max_samples:
            sample_X = X.sample(n=max_samples, random_state=random_seed).copy()
        else:
            sample_X = X.copy()

        # Compute SHAP explanation
        explanation = self.explainer(sample_X)
        return explanation, sample_X

    def get_feature_importance_table(self, explanation: shap.Explanation) -> pd.DataFrame:
        """Calculate mean absolute SHAP value per feature."""
        # explanation.values shape: (N, num_features)
        mean_abs = np.mean(np.abs(explanation.values), axis=0)
        feature_names = explanation.feature_names or [f"f_{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        total = df["mean_abs_shap"].sum()
        df["relative_importance"] = df["mean_abs_shap"] / max(total, 1e-6)
        return df.reset_index(drop=True)

    def plot_summary(
        self,
        explanation: shap.Explanation,
        save_path: Path | str,
        max_display: int = 15,
    ) -> None:
        """Render and save global SHAP summary beeswarm plot."""
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            explanation,
            max_display=max_display,
            show=False,
        )
        plt.title("Tree-SHAP Global Feature Importance (Demand Drivers)", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(save_p, dpi=150)
        plt.close()
        logger.info("Saved SHAP summary plot to: %s", save_p)

    def plot_waterfall(
        self,
        explanation: shap.Explanation,
        row_idx: int,
        save_path: Path | str,
        max_display: int = 10,
    ) -> None:
        """Render and save local single-prediction waterfall plot."""
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(
            explanation[row_idx],
            max_display=max_display,
            show=False,
        )
        plt.title(f"Local SHAP Prediction Waterfall (Observation #{row_idx})", fontsize=12, pad=12)
        plt.tight_layout()
        plt.savefig(save_p, dpi=150)
        plt.close()
        logger.info("Saved SHAP waterfall plot to: %s", save_p)
