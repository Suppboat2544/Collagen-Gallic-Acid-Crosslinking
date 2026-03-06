"""
Graph_model.viz.compare_models
==================================
Cross-model comparison plots — each as an individual figure (no subplots).

Reads comparison_summary.json and/or per-model JSON epoch logs to produce
bar charts, overlay curves, and ranking tables.

Usage
-----
    >>> from Graph_model.viz.compare_models import plot_all_comparisons
    >>> paths = plot_all_comparisons("results/", out_dir="figs/")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore

# ── Style constants ───────────────────────────────────────────────────────────

_MODEL_COLORS = {
    'A': '#2196F3',   # blue
    'B': '#4CAF50',   # green
    'C': '#FF9800',   # orange
    'D': '#9C27B0',   # purple
    'E': '#F44336',   # red (novel model highlighted)
}

_DPI = 150
_FIGSIZE = (8, 5)
_BAR_FIGSIZE = (9, 5)


def _style_ax(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _safe_float(val) -> float:
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


# ── Load helpers ──────────────────────────────────────────────────────────────

def _load_comparison(results_dir: str | Path) -> dict:
    """Load comparison_summary.json."""
    path = Path(results_dir) / "comparison_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _load_all_epoch_logs(results_dir: str | Path) -> dict[str, tuple[str, list[dict]]]:
    """Load all per-model epoch logs. Returns {key: (model_name, epochs)}."""
    results_dir = Path(results_dir)
    out = {}
    for key in ['a', 'b', 'c', 'd', 'e']:
        path = results_dir / f"option_{key}_training.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            out[key.upper()] = (data.get('model_name', f'Option {key.upper()}'), data.get('epochs', []))
    return out


# ── Bar chart: Best RMSE comparison ──────────────────────────────────────────

def plot_rmse_comparison_bar(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Bar chart comparing best validation RMSE across models."""
    assert plt is not None, "matplotlib required"
    comp = _load_comparison(results_dir)
    if not comp.get('models'):
        # Fallback: load from individual logs
        comp = _build_comparison_from_logs(results_dir)

    models = comp.get('models', {})
    keys = sorted(models.keys())
    names = [models[k].get('name', f'Option {k}') for k in keys]
    rmses = [_safe_float(models[k].get('best_val_rmse', float('nan'))) for k in keys]
    colors = [_MODEL_COLORS.get(k, '#888888') for k in keys]

    fig, ax = plt.subplots(figsize=_BAR_FIGSIZE)
    bars = ax.bar(range(len(keys)), rmses, color=colors, edgecolor='white', linewidth=1.5)

    # Value labels on bars
    for bar, val in zip(bars, rmses):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_short_name(n) for n in names], rotation=15, ha='right')
    _style_ax(ax, "Model Comparison — Best Validation RMSE", "Model", "RMSE (kcal/mol)")

    # Highlight best
    best_idx = np.nanargmin(rmses) if rmses else None
    if best_idx is not None:
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(3)

    return _save_fig(fig, Path(out_dir) / "comparison_rmse_bar.png")


def plot_mae_comparison_bar(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Bar chart comparing best validation MAE across models."""
    assert plt is not None, "matplotlib required"
    logs = _load_all_epoch_logs(results_dir)
    keys = sorted(logs.keys())

    maes = []
    names = []
    for k in keys:
        name, epochs = logs[k]
        names.append(name)
        val_maes = [_safe_float(e.get('val_mae', float('nan'))) for e in epochs]
        valid = [v for v in val_maes if not math.isnan(v)]
        maes.append(min(valid) if valid else float('nan'))

    colors = [_MODEL_COLORS.get(k, '#888888') for k in keys]

    fig, ax = plt.subplots(figsize=_BAR_FIGSIZE)
    bars = ax.bar(range(len(keys)), maes, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, maes):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_short_name(n) for n in names], rotation=15, ha='right')
    _style_ax(ax, "Model Comparison — Best Validation MAE", "Model", "MAE (kcal/mol)")

    best_idx = np.nanargmin(maes) if maes else None
    if best_idx is not None:
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(3)

    return _save_fig(fig, Path(out_dir) / "comparison_mae_bar.png")


def plot_pearson_comparison_bar(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Bar chart comparing best validation Pearson r across models."""
    assert plt is not None, "matplotlib required"
    logs = _load_all_epoch_logs(results_dir)
    keys = sorted(logs.keys())

    pearsons = []
    names = []
    for k in keys:
        name, epochs = logs[k]
        names.append(name)
        val_rs = [_safe_float(e.get('val_pearson_r', float('nan'))) for e in epochs]
        valid = [v for v in val_rs if not math.isnan(v)]
        pearsons.append(max(valid) if valid else float('nan'))

    colors = [_MODEL_COLORS.get(k, '#888888') for k in keys]

    fig, ax = plt.subplots(figsize=_BAR_FIGSIZE)
    bars = ax.bar(range(len(keys)), pearsons, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, pearsons):
        if not math.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_short_name(n) for n in names], rotation=15, ha='right')
    ax.set_ylim(-0.1, 1.1)
    _style_ax(ax, "Model Comparison — Best Validation Pearson r", "Model", "Pearson r")

    best_idx = np.nanargmax(pearsons) if pearsons else None
    if best_idx is not None:
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(3)

    return _save_fig(fig, Path(out_dir) / "comparison_pearson_bar.png")


# ── Overlay: Loss curves across models ────────────────────────────────────────

def plot_loss_overlay(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
    loss_key: str = "val_loss",
) -> Path:
    """Overlay validation loss curves from all models on one plot."""
    assert plt is not None, "matplotlib required"
    logs = _load_all_epoch_logs(results_dir)
    keys = sorted(logs.keys())

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for k in keys:
        name, epochs = logs[k]
        xs = [e['epoch'] for e in epochs]
        ys = [_safe_float(e.get(loss_key, float('nan'))) for e in epochs]
        color = _MODEL_COLORS.get(k, '#888888')
        ax.plot(xs, ys, color=color, linewidth=2, label=_short_name(name), alpha=0.85)

    title_type = "Validation" if "val" in loss_key else "Training"
    _style_ax(ax, f"All Models — {title_type} Loss Overlay", "Epoch", "MSE Loss")
    ax.legend(fontsize=9, loc='upper right')
    return _save_fig(fig, Path(out_dir) / f"comparison_{loss_key}_overlay.png")


def plot_rmse_overlay(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Overlay validation RMSE curves from all models."""
    assert plt is not None, "matplotlib required"
    logs = _load_all_epoch_logs(results_dir)
    keys = sorted(logs.keys())

    fig, ax = plt.subplots(figsize=_FIGSIZE)

    for k in keys:
        name, epochs = logs[k]
        xs = [e['epoch'] for e in epochs]
        ys = [_safe_float(e.get('val_rmse', float('nan'))) for e in epochs]
        color = _MODEL_COLORS.get(k, '#888888')
        ax.plot(xs, ys, color=color, linewidth=2, label=_short_name(name), alpha=0.85)

    _style_ax(ax, "All Models — Validation RMSE Overlay", "Epoch", "RMSE (kcal/mol)")
    ax.legend(fontsize=9, loc='upper right')
    return _save_fig(fig, Path(out_dir) / "comparison_rmse_overlay.png")


def plot_wall_time_comparison(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> Path:
    """Bar chart comparing training wall time across models."""
    assert plt is not None, "matplotlib required"
    comp = _load_comparison(results_dir)
    if not comp.get('models'):
        comp = _build_comparison_from_logs(results_dir)

    models = comp.get('models', {})
    keys = sorted(models.keys())
    names = [models[k].get('name', f'Option {k}') for k in keys]
    times = [_safe_float(models[k].get('wall_time_s', float('nan'))) for k in keys]
    colors = [_MODEL_COLORS.get(k, '#888888') for k in keys]

    fig, ax = plt.subplots(figsize=_BAR_FIGSIZE)
    bars = ax.bar(range(len(keys)), times, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, times):
        if not math.isnan(val):
            if val > 60:
                label = f'{val/60:.1f} min'
            else:
                label = f'{val:.1f} s'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([_short_name(n) for n in names], rotation=15, ha='right')
    _style_ax(ax, "Model Comparison — Training Wall Time", "Model", "Time (seconds)")
    return _save_fig(fig, Path(out_dir) / "comparison_wall_time.png")


# ── Generate all comparison plots ─────────────────────────────────────────────

def plot_all_comparisons(
    results_dir: str | Path,
    out_dir: str | Path = "figs",
) -> list[Path]:
    """Generate all cross-model comparison plots."""
    paths = []
    paths.append(plot_rmse_comparison_bar(results_dir, out_dir))
    paths.append(plot_mae_comparison_bar(results_dir, out_dir))
    paths.append(plot_pearson_comparison_bar(results_dir, out_dir))
    paths.append(plot_loss_overlay(results_dir, out_dir))
    paths.append(plot_loss_overlay(results_dir, out_dir, loss_key="train_loss"))
    paths.append(plot_rmse_overlay(results_dir, out_dir))
    paths.append(plot_wall_time_comparison(results_dir, out_dir))
    return paths


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short_name(full_name: str) -> str:
    """Shorten model name for axis labels."""
    # "Option A — Baseline GATv2" → "A: GATv2"
    import re
    m = re.match(r'Option\s+(\w+)\s*[—\-]\s*(.*)', full_name)
    if m:
        return f"{m.group(1)}: {m.group(2)[:20]}"
    return full_name[:25]


def _build_comparison_from_logs(results_dir: str | Path) -> dict:
    """Build a comparison dict from individual log files."""
    logs = _load_all_epoch_logs(results_dir)
    models = {}
    for k in sorted(logs.keys()):
        name, epochs = logs[k]
        val_losses = [_safe_float(e.get('val_loss', float('nan'))) for e in epochs]
        valid_losses = [v for v in val_losses if not math.isnan(v)]
        best_val_mse = min(valid_losses) if valid_losses else float('nan')
        best_epoch = val_losses.index(best_val_mse) if valid_losses else 0

        models[k] = {
            "name": name,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_mse ** 0.5 if not math.isnan(best_val_mse) else float('nan'),
            "total_epochs": len(epochs),
            "wall_time_s": _safe_float(epochs[-1].get('wall_time_s', 0)) if epochs else 0,
        }
    return {"models": models}
