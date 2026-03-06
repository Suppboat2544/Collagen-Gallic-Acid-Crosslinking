"""
Graph_model.viz.plot_results
================================
Individual plots for each model's training results.

Every function produces **one** matplotlib figure (no subplots / multi-panel)
and saves it to disk as a PNG file. Returns the Path of the saved figure.

Usage
-----
    >>> from Graph_model.viz.plot_results import plot_all_individual
    >>> paths = plot_all_individual("results/option_e_training.json", out_dir="figs/")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

# ── Shared style ──────────────────────────────────────────────────────────────

_COLORS = {
    'train': '#2196F3',    # blue
    'val':   '#F44336',    # red
    'best':  '#4CAF50',    # green
    'pred':  '#9C27B0',    # purple
    'bar':   '#FF9800',    # orange
}

_DPI = 150
_FIGSIZE = (8, 5)


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# ── Load epoch data from JSON ─────────────────────────────────────────────────

def _load_epochs(json_path: str | Path) -> tuple[str, list[dict], dict]:
    """Load training JSON and return (model_name, epoch_list, summary)."""
    with open(json_path) as f:
        data = json.load(f)
    return data.get('model_name', 'Model'), data.get('epochs', []), data.get('summary', {})


def _safe_float(val) -> float:
    """Convert value to float, NaN if impossible."""
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_training_loss(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Training loss curve (MSE) per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    ys = [_safe_float(e.get('train_loss', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(xs, ys, color=_COLORS['train'], linewidth=2, label='Train Loss (MSE)')

    best_ep = summary.get('best_epoch', None)
    if best_ep is not None and best_ep < len(ys):
        ax.axvline(best_ep, color=_COLORS['best'], linestyle='--', alpha=0.7, label=f'Best epoch ({best_ep})')

    _style_ax(ax, f"{name}\nTraining Loss", "Epoch", "MSE Loss")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_train_loss.png")


def plot_validation_loss(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Validation loss curve (MSE) per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    ys = [_safe_float(e.get('val_loss', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(xs, ys, color=_COLORS['val'], linewidth=2, label='Val Loss (MSE)')

    best_ep = summary.get('best_epoch', None)
    if best_ep is not None and best_ep < len(ys):
        ax.axvline(best_ep, color=_COLORS['best'], linestyle='--', alpha=0.7, label=f'Best epoch ({best_ep})')

    _style_ax(ax, f"{name}\nValidation Loss", "Epoch", "MSE Loss")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_val_loss.png")


def plot_rmse_per_epoch(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Validation RMSE per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    train_ys = [_safe_float(e.get('train_rmse', float('nan'))) for e in epochs]
    val_ys = [_safe_float(e.get('val_rmse', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if any(not math.isnan(y) for y in train_ys):
        ax.plot(xs, train_ys, color=_COLORS['train'], linewidth=1.5, alpha=0.7, label='Train RMSE')
    ax.plot(xs, val_ys, color=_COLORS['val'], linewidth=2, label='Val RMSE')

    _style_ax(ax, f"{name}\nRMSE per Epoch", "Epoch", "RMSE (kcal/mol)")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_rmse_epoch.png")


def plot_mae_per_epoch(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Validation MAE per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    train_ys = [_safe_float(e.get('train_mae', float('nan'))) for e in epochs]
    val_ys = [_safe_float(e.get('val_mae', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if any(not math.isnan(y) for y in train_ys):
        ax.plot(xs, train_ys, color=_COLORS['train'], linewidth=1.5, alpha=0.7, label='Train MAE')
    ax.plot(xs, val_ys, color=_COLORS['val'], linewidth=2, label='Val MAE')

    _style_ax(ax, f"{name}\nMAE per Epoch", "Epoch", "MAE (kcal/mol)")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_mae_epoch.png")


def plot_pearson_per_epoch(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Pearson correlation per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    train_ys = [_safe_float(e.get('train_pearson_r', float('nan'))) for e in epochs]
    val_ys = [_safe_float(e.get('val_pearson_r', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if any(not math.isnan(y) for y in train_ys):
        ax.plot(xs, train_ys, color=_COLORS['train'], linewidth=1.5, alpha=0.7, label='Train Pearson r')
    ax.plot(xs, val_ys, color=_COLORS['val'], linewidth=2, label='Val Pearson r')

    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    _style_ax(ax, f"{name}\nPearson Correlation per Epoch", "Epoch", "Pearson r")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_pearson_epoch.png")


def plot_spearman_per_epoch(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Spearman correlation per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    train_ys = [_safe_float(e.get('train_spearman_r', float('nan'))) for e in epochs]
    val_ys = [_safe_float(e.get('val_spearman_r', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if any(not math.isnan(y) for y in train_ys):
        ax.plot(xs, train_ys, color=_COLORS['train'], linewidth=1.5, alpha=0.7, label='Train Spearman ρ')
    ax.plot(xs, val_ys, color=_COLORS['val'], linewidth=2, label='Val Spearman ρ')

    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
    _style_ax(ax, f"{name}\nSpearman Correlation per Epoch", "Epoch", "Spearman ρ")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_spearman_epoch.png")


def plot_learning_rate(
    json_path: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Learning rate schedule per epoch."""
    assert plt is not None, "matplotlib required"
    name, epochs, summary = _load_epochs(json_path)
    xs = [e['epoch'] for e in epochs]
    ys = [_safe_float(e.get('learning_rate', float('nan'))) for e in epochs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(xs, ys, color='#607D8B', linewidth=2, label='Learning Rate')
    ax.set_yscale('log')
    _style_ax(ax, f"{name}\nLearning Rate Schedule", "Epoch", "Learning Rate")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(name)}_lr_schedule.png")


def plot_pred_vs_true(
    preds: list | np.ndarray,
    targets: list | np.ndarray,
    model_name: str = "Model",
    out_dir: str | Path = "figs",
) -> Path:
    """Scatter plot: predicted vs true ΔG."""
    assert plt is not None, "matplotlib required"
    preds = np.asarray(preds, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, preds, alpha=0.5, s=30, color=_COLORS['pred'], edgecolors='none')

    # Perfect prediction line
    lo = min(targets.min(), preds.min()) - 0.5
    hi = max(targets.max(), preds.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1.5, alpha=0.5, label='y = x')

    from ..train.metrics import regression_metrics
    m = regression_metrics(preds, targets)
    text = f"RMSE = {m['rmse']:.3f}\nMAE = {m['mae']:.3f}\nr = {m['pearson_r']:.3f}\nρ = {m['spearman_r']:.3f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    _style_ax(ax, f"{model_name}\nPredicted vs True ΔG", "True ΔG (kcal/mol)", "Predicted ΔG (kcal/mol)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(model_name)}_pred_vs_true.png")


def plot_residuals(
    preds: list | np.ndarray,
    targets: list | np.ndarray,
    model_name: str = "Model",
    out_dir: str | Path = "figs",
) -> Path:
    """Residual plot (error vs true ΔG)."""
    assert plt is not None, "matplotlib required"
    preds = np.asarray(preds, dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()
    residuals = preds - targets

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.scatter(targets, residuals, alpha=0.5, s=30, color=_COLORS['train'], edgecolors='none')
    ax.axhline(0, color='k', linewidth=1, alpha=0.5)

    # ±1 kcal/mol error bands
    ax.axhline(1, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axhline(-1, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.fill_between([targets.min()-0.5, targets.max()+0.5], -1, 1,
                    alpha=0.1, color='green', label='±1 kcal/mol')

    _style_ax(ax, f"{model_name}\nResidual Plot", "True ΔG (kcal/mol)", "Residual (Pred − True)")
    ax.legend(fontsize=10)
    return _save_fig(fig, Path(out_dir) / f"{_slug(model_name)}_residuals.png")


# ── Generate all individual plots for one model ──────────────────────────────

def plot_all_individual(
    json_path: str | Path,
    out_dir: str | Path = "figs",
    preds: list | np.ndarray | None = None,
    targets: list | np.ndarray | None = None,
) -> list[Path]:
    """
    Generate all individual plots for a model from its JSON epoch log.

    Parameters
    ----------
    json_path : path to training JSON log
    out_dir   : output directory for PNG files
    preds     : optional predictions for scatter/residual plots
    targets   : optional targets for scatter/residual plots

    Returns
    -------
    list of created file paths
    """
    paths = []
    paths.append(plot_training_loss(json_path, out_dir))
    paths.append(plot_validation_loss(json_path, out_dir))
    paths.append(plot_rmse_per_epoch(json_path, out_dir))
    paths.append(plot_mae_per_epoch(json_path, out_dir))
    paths.append(plot_pearson_per_epoch(json_path, out_dir))
    paths.append(plot_spearman_per_epoch(json_path, out_dir))
    paths.append(plot_learning_rate(json_path, out_dir))

    if preds is not None and targets is not None:
        name, _, _ = _load_epochs(json_path)
        paths.append(plot_pred_vs_true(preds, targets, name, out_dir))
        paths.append(plot_residuals(preds, targets, name, out_dir))

    return paths


# ── Slug helper ───────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Convert model name to filesystem-safe slug."""
    return name.lower().replace(' ', '_').replace('—', '_').replace('(', '').replace(')', '').replace('/', '_')
