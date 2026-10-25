"""Explainability and counterfactual What-If simulation (Novelty N5)."""

from idfs.explain.shap_explainer import TreeSHAPExplainer
from idfs.explain.what_if import WhatIfSimulator

__all__ = [
    "TreeSHAPExplainer",
    "WhatIfSimulator",
]
