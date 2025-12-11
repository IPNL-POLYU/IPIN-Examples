"""
Evaluation and Visualization Module.

This module provides evaluation metrics and visualization utilities
for indoor positioning algorithms.

Modules:
    metrics: Error metrics (RMSE, CDF, NEES, NIS)
    plots: Visualization functions for trajectories, errors, and DOP
"""

from .metrics import (
    compute_error_stats,
    compute_nees,
    compute_nis,
    compute_position_errors,
    compute_rmse,
)
from .plots import (
    plot_dop_map,
    plot_error_cdf,
    plot_error_hist,
    plot_position_error_time,
    plot_rf_geometry,
    plot_trajectory_2d,
    save_figure,
)

__all__ = [
    # Metrics
    "compute_position_errors",
    "compute_rmse",
    "compute_error_stats",
    "compute_nees",
    "compute_nis",
    # Plots
    "plot_trajectory_2d",
    "plot_position_error_time",
    "plot_error_hist",
    "plot_error_cdf",
    "plot_rf_geometry",
    "plot_dop_map",
    "save_figure",
]

