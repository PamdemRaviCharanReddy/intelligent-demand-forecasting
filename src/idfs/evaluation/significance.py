"""Statistical hypothesis testing for comparative time-series forecasting (P7-T03).

Implements the Diebold-Mariano test (1995) with the Harvey, Leybourne, and Newbold (1997)
small-sample correction, and the paired Wilcoxon signed-rank test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    actual: np.ndarray | pd.Series,
    forecast_1: np.ndarray | pd.Series,
    forecast_2: np.ndarray | pd.Series,
    horizon: int = 1,
    loss_type: str = "squared",  # "squared" (MSE) or "absolute" (MAE)
) -> dict[str, float]:
    """Perform the Diebold-Mariano test of equal predictive accuracy.

    Tests the null hypothesis H0: E[d_t] = 0, where d_t = Loss(e_{1,t}) - Loss(e_{2,t}).
    A negative statistic indicates forecast_1 has smaller loss than forecast_2.

    Args:
        actual: Ground truth demand vector of length T.
        forecast_1: Predictions from model 1.
        forecast_2: Predictions from model 2.
        horizon: Multi-step forecast horizon h (controls autocovariance lag truncation).
        loss_type: 'squared' for quadratic loss, 'absolute' for linear loss.

    Returns:
        Dictionary with 'dm_statistic', 'p_value', 'mean_differential', and 'df'.
    """
    y = np.asarray(actual, dtype=float)
    f1 = np.asarray(forecast_1, dtype=float)
    f2 = np.asarray(forecast_2, dtype=float)

    if len(y) != len(f1) or len(y) != len(f2):
        raise ValueError(f"Length mismatch: actual={len(y)}, f1={len(f1)}, f2={len(f2)}")

    t_obs = len(y)
    if t_obs < 3:
        raise ValueError(f"Sample size T={t_obs} too small for Diebold-Mariano test.")

    e1 = y - f1
    e2 = y - f2

    if loss_type == "squared":
        d = e1**2 - e2**2
    elif loss_type == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Must be 'squared' or 'absolute'.")

    d_mean = float(np.mean(d))
    d_centered = d - d_mean

    # Sample autocovariances up to lag h - 1
    gamma0 = float(np.mean(d_centered**2))
    gamma_sum = 0.0

    max_lag = max(1, min(horizon - 1, t_obs - 2))
    for k in range(1, max_lag + 1):
        gamma_k = float(np.mean(d_centered[k:] * d_centered[:-k]))
        # Bartlett kernel weighting to ensure positive semi-definiteness
        weight = 1.0 - (k / (max_lag + 1))
        gamma_sum += 2.0 * weight * gamma_k

    var_d = max(gamma0 + gamma_sum, 1e-12)
    se_d = np.sqrt(var_d / t_obs)

    dm_stat = float(d_mean / se_d)

    # Harvey, Leybourne, Newbold (1997) small-sample adjustment factor
    h = max(1, horizon)
    hln_factor = np.sqrt(max((t_obs + 1 - 2 * h + (h * (h - 1)) / t_obs) / t_obs, 1e-6))
    adjusted_dm_stat = float(dm_stat * hln_factor)

    deg_f = t_obs - 1
    # Two-sided Student's t distribution p-value
    p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(adjusted_dm_stat), df=deg_f)))

    return {
        "dm_statistic": adjusted_dm_stat,
        "p_value": p_val,
        "mean_differential": d_mean,
        "df": deg_f,
    }


def paired_wilcoxon_test(
    errors_1: np.ndarray | pd.Series,
    errors_2: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Perform the Wilcoxon signed-rank test comparing two error distributions.

    Args:
        errors_1: Absolute or squared error array from model 1.
        errors_2: Absolute or squared error array from model 2.

    Returns:
        Dictionary with 'statistic' and 'p_value'.
    """
    e1 = np.asarray(errors_1, dtype=float)
    e2 = np.asarray(errors_2, dtype=float)

    diff = e1 - e2
    # Filter out exact ties
    diff = diff[diff != 0.0]
    if len(diff) < 5:
        # Fall back if insufficient non-zero differences
        return {"statistic": 0.0, "p_value": 1.0}

    res = stats.wilcoxon(e1, e2, alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def compute_pairwise_significance_matrix(
    actual: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    horizon: int = 14,
    loss_type: str = "absolute",
) -> pd.DataFrame:
    """Generate a pairwise matrix of Diebold-Mariano and Wilcoxon p-values.

    Args:
        actual: Ground truth demand array.
        model_predictions: Mapping of model name to predicted values array.
        horizon: Forecast horizon.
        loss_type: Loss type for DM test ('absolute' or 'squared').

    Returns:
        DataFrame summarizing pairwise model comparisons.
    """
    records: list[dict[str, Any]] = []
    models = list(model_predictions.keys())

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1 = models[i]
            m2 = models[j]

            pred1 = model_predictions[m1]
            pred2 = model_predictions[m2]

            dm_res = diebold_mariano_test(
                actual=actual,
                forecast_1=pred1,
                forecast_2=pred2,
                horizon=horizon,
                loss_type=loss_type,
            )

            e1 = np.abs(actual - pred1) if loss_type == "absolute" else (actual - pred1) ** 2
            e2 = np.abs(actual - pred2) if loss_type == "absolute" else (actual - pred2) ** 2

            wilc_res = paired_wilcoxon_test(e1, e2)

            better_model = m1 if dm_res["mean_differential"] < 0 else m2
            is_sig = dm_res["p_value"] < 0.05

            records.append(
                {
                    "model_1": m1,
                    "model_2": m2,
                    "dm_stat": round(dm_res["dm_statistic"], 4),
                    "dm_p_value": round(dm_res["p_value"], 4),
                    "wilcoxon_stat": round(wilc_res["statistic"], 2),
                    "wilcoxon_p_value": round(wilc_res["p_value"], 4),
                    "statistically_significant": is_sig,
                    "superior_model": better_model if is_sig else "TIE (No Significant Difference)",
                }
            )

    return pd.DataFrame(records)
