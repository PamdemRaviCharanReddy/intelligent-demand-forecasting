"""Model evaluation: metrics, backtesting, comparison."""

from idfs.evaluation.metrics import (
    compute_all_point_metrics,
    empirical_coverage,
    fill_rate,
    interval_width,
    mae,
    mape,
    mase,
    pinball_loss,
    rmse,
    stockout_periods,
    total_inventory_cost,
    wape,
)
from idfs.evaluation.significance import (
    compute_pairwise_significance_matrix,
    diebold_mariano_test,
    paired_wilcoxon_test,
)

__all__ = [
    "compute_all_point_metrics",
    "compute_pairwise_significance_matrix",
    "diebold_mariano_test",
    "empirical_coverage",
    "fill_rate",
    "interval_width",
    "mae",
    "mape",
    "mase",
    "paired_wilcoxon_test",
    "pinball_loss",
    "rmse",
    "stockout_periods",
    "total_inventory_cost",
    "wape",
]
