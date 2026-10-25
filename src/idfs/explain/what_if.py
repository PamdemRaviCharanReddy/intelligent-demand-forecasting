"""Counterfactual What-If scenario simulation engine (Novelty N5).

Enables inventory and supply chain planners to stress-test future demand forecasts
against hypothetical promotional events, economic price shocks, and supply disruptions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScenarioResult:
    """Summary of a counterfactual What-If simulation."""

    scenario_name: str
    baseline_total_demand: float
    scenario_total_demand: float
    absolute_delta: float
    percentage_change: float
    comparison_df: pd.DataFrame


class WhatIfSimulator:
    """Counterfactual scenario testing engine for demand models."""

    def __init__(self, forecaster: Forecaster) -> None:
        if not forecaster.is_fitted:
            raise RuntimeError("Forecaster must be fitted before running What-If simulations")
        self.forecaster = forecaster

    def simulate_promotion_scenario(
        self,
        features_df: pd.DataFrame,
        horizon: int,
        target_promo_status: int = 1,
        store_id: int | None = None,
        family: str | None = None,
        scenario_name: str = "Promotional Surge",
    ) -> ScenarioResult:
        """Simulate demand response if promotions are activated or deactivated."""
        # 1. Baseline prediction
        base_preds = self.forecaster.predict(horizon=horizon, features_df=features_df)

        # 2. Perturb features
        counterfactual_df = features_df.copy()
        mask = np.ones(len(counterfactual_df), dtype=bool)
        if store_id is not None and "store_id" in counterfactual_df.columns:
            mask &= counterfactual_df["store_id"] == store_id
        if family is not None and "family" in counterfactual_df.columns:
            mask &= counterfactual_df["family"] == family

        if "onpromotion" in counterfactual_df.columns:
            counterfactual_df.loc[mask, "onpromotion"] = target_promo_status

        # 3. Counterfactual prediction
        scenario_preds = self.forecaster.predict(horizon=horizon, features_df=counterfactual_df)

        # 4. Compare results
        comp_df = base_preds.copy()
        comp_df["baseline_forecast"] = base_preds["forecast"]
        comp_df["scenario_forecast"] = scenario_preds["forecast"]
        comp_df["demand_delta"] = comp_df["scenario_forecast"] - comp_df["baseline_forecast"]
        comp_df["pct_change"] = np.where(
            comp_df["baseline_forecast"] > 0,
            (comp_df["demand_delta"] / comp_df["baseline_forecast"]) * 100.0,
            0.0,
        )

        base_total = float(comp_df["baseline_forecast"].sum())
        scen_total = float(comp_df["scenario_forecast"].sum())
        abs_delta = scen_total - base_total
        pct_change = (abs_delta / max(base_total, 1e-4)) * 100.0

        return ScenarioResult(
            scenario_name=scenario_name,
            baseline_total_demand=base_total,
            scenario_total_demand=scen_total,
            absolute_delta=abs_delta,
            percentage_change=pct_change,
            comparison_df=comp_df,
        )

    def simulate_macro_shock(
        self,
        features_df: pd.DataFrame,
        horizon: int,
        oil_multiplier: float = 1.20,  # e.g. +20% oil price shock
        scenario_name: str = "Macroeconomic Oil Shock (+20%)",
    ) -> ScenarioResult:
        """Simulate demand response under macroeconomic exogenous price shifts."""
        base_preds = self.forecaster.predict(horizon=horizon, features_df=features_df)

        counterfactual_df = features_df.copy()
        if "dcoilwtico" in counterfactual_df.columns:
            counterfactual_df["dcoilwtico"] *= oil_multiplier

        scenario_preds = self.forecaster.predict(horizon=horizon, features_df=counterfactual_df)

        comp_df = base_preds.copy()
        comp_df["baseline_forecast"] = base_preds["forecast"]
        comp_df["scenario_forecast"] = scenario_preds["forecast"]
        comp_df["demand_delta"] = comp_df["scenario_forecast"] - comp_df["baseline_forecast"]
        comp_df["pct_change"] = np.where(
            comp_df["baseline_forecast"] > 0,
            (comp_df["demand_delta"] / comp_df["baseline_forecast"]) * 100.0,
            0.0,
        )

        base_total = float(comp_df["baseline_forecast"].sum())
        scen_total = float(comp_df["scenario_forecast"].sum())
        abs_delta = scen_total - base_total
        pct_change = (abs_delta / max(base_total, 1e-4)) * 100.0

        return ScenarioResult(
            scenario_name=scenario_name,
            baseline_total_demand=base_total,
            scenario_total_demand=scen_total,
            absolute_delta=abs_delta,
            percentage_change=pct_change,
            comparison_df=comp_df,
        )
