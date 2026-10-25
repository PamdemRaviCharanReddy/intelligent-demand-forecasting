"""Unit tests for statistical significance testing (Phase 7 - P7-T03)."""

from __future__ import annotations

import numpy as np
import pytest

from idfs.evaluation.significance import (
    compute_pairwise_significance_matrix,
    diebold_mariano_test,
    paired_wilcoxon_test,
)


def test_diebold_mariano_identical_forecasts() -> None:
    """Two identical forecasts must produce DM statistic = 0 and p-value = 1.0."""
    actual = np.array([10.0, 12.0, 14.0, 11.0, 15.0, 13.0, 16.0, 14.0])
    forecast = np.array([10.5, 11.5, 14.2, 11.0, 14.8, 13.2, 15.7, 14.1])

    res = diebold_mariano_test(actual, forecast, forecast, horizon=1, loss_type="squared")
    assert res["dm_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert res["p_value"] == pytest.approx(1.0, abs=1e-4)


def test_diebold_mariano_superior_model() -> None:
    """A clearly superior model must yield a statistically significant negative DM statistic."""
    np.random.seed(42)
    actual = np.linspace(10, 50, 40) + np.random.normal(0, 1, 40)
    # Model 1 is near-perfect
    f1 = actual + np.random.normal(0, 0.5, 40)
    # Model 2 has large biased errors
    f2 = actual + np.random.normal(5, 4.0, 40)

    res = diebold_mariano_test(actual, f1, f2, horizon=1, loss_type="absolute")
    assert res["dm_statistic"] < 0
    assert res["p_value"] < 0.001
    assert res["mean_differential"] < 0


def test_paired_wilcoxon_test() -> None:
    """Wilcoxon test accurately identifies significant differences in error distributions."""
    np.random.seed(42)
    errors_good = np.abs(np.random.normal(0, 1.0, 30))
    errors_bad = np.abs(np.random.normal(5.0, 2.0, 30))

    res = paired_wilcoxon_test(errors_good, errors_bad)
    assert res["p_value"] < 0.001


def test_pairwise_significance_matrix() -> None:
    """Pairwise significance matrix constructs properly across multiple models."""
    actual = np.array([10.0, 20.0, 15.0, 25.0, 18.0, 22.0, 30.0, 28.0] * 3)
    preds = {
        "ModelA": actual + 1.0,
        "ModelB": actual + 5.0,
        "ModelC": actual + 10.0,
    }

    df = compute_pairwise_significance_matrix(actual, preds, horizon=1, loss_type="absolute")
    assert len(df) == 3  # (3 choose 2 = 3 pairs)
    assert "dm_stat" in df.columns
    assert "wilcoxon_p_value" in df.columns
    assert "superior_model" in df.columns
    # ModelA should beat ModelC significantly
    ac_row = df[(df["model_1"] == "ModelA") & (df["model_2"] == "ModelC")]
    assert len(ac_row) == 1
    assert ac_row.iloc[0]["dm_stat"] < 0
