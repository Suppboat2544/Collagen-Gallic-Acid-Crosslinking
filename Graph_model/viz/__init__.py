"""
Graph_model.viz
================
Visualization package for training results.

All plots are individual figures (NOT multi-panel / subplot), as requested.
"""

from .plot_results import (
    plot_training_loss,
    plot_validation_loss,
    plot_rmse_per_epoch,
    plot_mae_per_epoch,
    plot_pearson_per_epoch,
    plot_spearman_per_epoch,
    plot_learning_rate,
    plot_pred_vs_true,
    plot_residuals,
    plot_all_individual,
)

from .compare_models import (
    plot_rmse_comparison_bar,
    plot_mae_comparison_bar,
    plot_pearson_comparison_bar,
    plot_loss_overlay,
    plot_rmse_overlay,
    plot_wall_time_comparison,
    plot_all_comparisons,
)

__all__ = [
    # Individual model plots
    "plot_training_loss",
    "plot_validation_loss",
    "plot_rmse_per_epoch",
    "plot_mae_per_epoch",
    "plot_pearson_per_epoch",
    "plot_spearman_per_epoch",
    "plot_learning_rate",
    "plot_pred_vs_true",
    "plot_residuals",
    "plot_all_individual",
    # Comparison plots
    "plot_rmse_comparison_bar",
    "plot_mae_comparison_bar",
    "plot_pearson_comparison_bar",
    "plot_loss_overlay",
    "plot_rmse_overlay",
    "plot_wall_time_comparison",
    "plot_all_comparisons",
]
