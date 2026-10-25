"""Unit tests for Phase 5: Concept Drift Monitoring and SHAP Explainability (Novelties N4 & N5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from idfs.explain.shap_explainer import TreeSHAPExplainer
from idfs.explain.what_if import WhatIfSimulator
from idfs.models.xgboost import XGBoostForecaster
from idfs.monitoring.drift import DriftMonitor, RetrainAlert


@pytest.fixture
def sample_monitoring_stream() -> tuple[np.ndarray, np.ndarray]:
    """Generates 50 normal observations followed by 30 drifted observations."""
    np.random.seed(42)
    normal_actuals = np.random.normal(50.0, 5.0, size=50)
    normal_preds = np.random.normal(50.0, 2.0, size=50)

    # Injected drift: actuals jump to 80 while model keeps predicting 50
    drift_actuals = np.random.normal(85.0, 5.0, size=30)
    drift_preds = np.random.normal(50.0, 2.0, size=30)

    actuals = np.concatenate([normal_actuals, drift_actuals])
    preds = np.concatenate([normal_preds, drift_preds])
    return actuals, preds


@pytest.fixture
def fitted_xgb_and_data() -> tuple[XGBoostForecaster, pd.DataFrame]:
    """Generates a fitted XGBoostForecaster on synthetic sample."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=90, freq="D")
    rows = []
    for s_id in [1, 2]:
        for d in dates:
            rows.append(
                {
                    "date": d,
                    "store_id": s_id,
                    "family": "DAIRY",
                    "sales": float(np.random.poisson(30) + (d.dayofweek % 7) * 4),
                    "onpromotion": int(np.random.rand() > 0.8),
                    "dcoilwtico": 75.0 + np.random.normal(0, 1.0),
                }
            )
    df = pd.DataFrame(rows)
    model = XGBoostForecaster(n_estimators=25, max_depth=3)
    model.fit(df, series_id_cols=["store_id", "family"])
    return model, df


class TestDriftMonitor:
    def test_baseline_and_normal_operation(self):
        monitor = DriftMonitor(window_size=10, threshold_multiplier=1.5, metric="mae")
        monitor.set_baseline(5.0)
        assert monitor.baseline_error == 5.0
        assert not monitor.should_retrain

        # Feed normal errors (around 4.0)
        for i in range(15):
            alert = monitor.update(actual=50.0, predicted=54.0, timestamp=f"2023-01-{i + 1:02d}")
            assert alert is None

        assert not monitor.should_retrain
        hist = monitor.get_monitoring_history()
        assert len(hist) == 15
        assert hist["rolling_mae"].iloc[-1] == pytest.approx(4.0)

    def test_drift_detection_and_alerts(self, sample_monitoring_stream):
        actuals, preds = sample_monitoring_stream
        monitor = DriftMonitor(window_size=14, threshold_multiplier=1.4, critical_multiplier=1.8)
        monitor.set_baseline(5.0)

        alerts = monitor.batch_update(actuals=actuals, predictions=preds)
        assert len(alerts) > 0
        assert monitor.should_retrain

        last_alert = alerts[-1]
        assert isinstance(last_alert, RetrainAlert)
        assert last_alert.severity in ["WARNING", "CRITICAL"]
        assert last_alert.current_error > 5.0 * 1.4


class TestTreeSHAPExplainer:
    def test_shap_explanation_and_importance_table(self, fitted_xgb_and_data, tmp_path):
        model, df = fitted_xgb_and_data
        explainer = TreeSHAPExplainer(model)

        explanation, sample_X = explainer.explain(df, max_samples=50)
        assert explanation is not None
        assert explanation.values.shape[0] == len(sample_X)

        importance_df = explainer.get_feature_importance_table(explanation)
        assert "feature" in importance_df.columns
        assert "mean_abs_shap" in importance_df.columns
        assert "relative_importance" in importance_df.columns
        assert np.isclose(importance_df["relative_importance"].sum(), 1.0)

        # Plot summary
        summary_path = tmp_path / "shap_summary.png"
        explainer.plot_summary(explanation, save_path=summary_path, max_display=5)
        assert summary_path.exists()

        # Plot waterfall
        waterfall_path = tmp_path / "shap_waterfall.png"
        explainer.plot_waterfall(explanation, row_idx=0, save_path=waterfall_path, max_display=5)
        assert waterfall_path.exists()


class TestWhatIfSimulator:
    def test_promotion_scenario(self, fitted_xgb_and_data):
        model, df = fitted_xgb_and_data
        what_if = WhatIfSimulator(model)

        eval_df = df.tail(28).copy()
        res = what_if.simulate_promotion_scenario(
            features_df=eval_df,
            horizon=14,
            target_promo_status=1,
            scenario_name="Test Promo",
        )

        assert res.scenario_name == "Test Promo"
        assert res.baseline_total_demand >= 0.0
        assert res.scenario_total_demand >= 0.0
        assert "demand_delta" in res.comparison_df.columns
        assert "pct_change" in res.comparison_df.columns

    def test_macro_shock(self, fitted_xgb_and_data):
        model, df = fitted_xgb_and_data
        what_if = WhatIfSimulator(model)

        eval_df = df.tail(28).copy()
        res = what_if.simulate_macro_shock(
            features_df=eval_df,
            horizon=14,
            oil_multiplier=1.25,
        )
        assert res.scenario_name is not None
        assert len(res.comparison_df) == len(eval_df)
