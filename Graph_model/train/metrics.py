"""
Graph_model.train.metrics
==========================
Regression metrics and per-fold summary for LOLO-CV reporting.

Usage
-----
>>> from Graph_model.train.metrics import regression_metrics, aggregate_folds, FoldMetrics
>>> m = regression_metrics(preds, targets)
>>> print(m)                # {'rmse': ..., 'mae': ..., 'pearson_r': ..., 'spearman_r': ...}

>>> # After LOLO-CV
>>> fold_list = [FoldMetrics(fold=i, held_out=lig, n_test=n, **m) for ...]
>>> report = aggregate_folds(fold_list)
>>> print(report['rmse_mean'], report['rmse_std'])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np


# ── Scalar metrics ─────────────────────────────────────────────────────────────

def regression_metrics(
    preds:   np.ndarray | list,
    targets: np.ndarray | list,
) -> dict[str, float]:
    """
    Compute RMSE, MAE, Pearson r, Spearman ρ for a prediction vector.

    Parameters
    ----------
    preds   : array-like [N]   predicted ΔG (kcal/mol)
    targets : array-like [N]   experimental / Vinardo ΔG (kcal/mol)

    Returns
    -------
    dict with keys: 'rmse', 'mae', 'pearson_r', 'spearman_r', 'n'
    """
    preds   = np.asarray(preds,   dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()

    # Drop any NaN pairs
    valid = ~(np.isnan(preds) | np.isnan(targets))
    preds, targets = preds[valid], targets[valid]
    n = len(preds)

    if n == 0:
        return {'rmse': float('nan'), 'mae': float('nan'),
                'pearson_r': float('nan'), 'spearman_r': float('nan'), 'n': 0}

    err    = preds - targets
    rmse   = float(np.sqrt((err ** 2).mean()))
    mae    = float(np.abs(err).mean())

    # Pearson r
    if n >= 2 and preds.std() > 0 and targets.std() > 0:
        pearson_r = float(np.corrcoef(preds, targets)[0, 1])
    else:
        pearson_r = float('nan')

    # Spearman ρ (rank correlation)
    spearman_r = float(_spearman(preds, targets))

    return {
        'rmse':       rmse,
        'mae':        mae,
        'pearson_r':  pearson_r,
        'spearman_r': spearman_r,
        'n':          n,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's ρ without scipy dependency."""
    n = len(x)
    if n < 2:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d  = rx - ry
    rho = 1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1))
    return float(rho)


# ── Per-fold container ─────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """
    Metrics for one LOLO-CV fold.

    Attributes
    ----------
    fold        : int   fold index (0-based)
    held_out    : str   ligand name withheld in this fold
    n_test      : int   number of held-out test records
    rmse        : float kcal/mol
    mae         : float kcal/mol
    pearson_r   : float
    spearman_r  : float
    n_train     : int   optional — number of training records
    n_val       : int   optional — number of validation records
    best_epoch  : int   optional — epoch at which best val loss was achieved
    val_rmse    : float optional — validation RMSE at best epoch
    """
    fold:       int
    held_out:   str
    n_test:     int
    rmse:       float
    mae:        float
    pearson_r:  float
    spearman_r: float
    n_train:    int   = 0
    n_val:      int   = 0
    best_epoch: int   = 0
    val_rmse:   float = float('nan')

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"Fold {self.fold:2d} | held-out: {self.held_out:25s} | "
            f"n={self.n_test:4d} | "
            f"RMSE={self.rmse:.4f}  MAE={self.mae:.4f}  "
            f"r={self.pearson_r:+.4f}  ρ={self.spearman_r:+.4f}"
        )


# ── Cross-fold aggregation ─────────────────────────────────────────────────────

def aggregate_folds(folds: List[FoldMetrics]) -> dict[str, float]:
    """
    Compute mean and std across LOLO-CV folds.

    Returns
    -------
    dict with keys:
      rmse_mean, rmse_std,
      mae_mean,  mae_std,
      pearson_r_mean, pearson_r_std,
      spearman_r_mean, spearman_r_std,
      n_folds, n_total_test
    """
    if not folds:
        return {}

    def _stats(key: str) -> tuple[float, float]:
        vals = [getattr(f, key) for f in folds if not math.isnan(getattr(f, key))]
        if not vals:
            return float('nan'), float('nan')
        arr = np.array(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    rm, rs  = _stats('rmse')
    mm, ms  = _stats('mae')
    pm, ps  = _stats('pearson_r')
    sm, ss  = _stats('spearman_r')

    return {
        'rmse_mean':       rm, 'rmse_std':       rs,
        'mae_mean':        mm, 'mae_std':         ms,
        'pearson_r_mean':  pm, 'pearson_r_std':   ps,
        'spearman_r_mean': sm, 'spearman_r_std':  ss,
        'n_folds':         len(folds),
        'n_total_test':    sum(f.n_test for f in folds),
    }


def print_lolo_report(
    folds:   List[FoldMetrics],
    model_name: str = "Model",
) -> None:
    """Print a formatted LOLO-CV report to stdout."""
    print(f"\n{'='*72}")
    print(f" LOLO-CV Report — {model_name}  ({len(folds)} folds)")
    print(f"{'='*72}")
    for f in sorted(folds, key=lambda x: x.fold):
        print(f" {f}")
    agg = aggregate_folds(folds)
    print(f"{'-'*72}")
    print(
        f" MEAN ± STD  | "
        f"RMSE = {agg['rmse_mean']:.4f} ± {agg['rmse_std']:.4f}  "
        f"MAE = {agg['mae_mean']:.4f} ± {agg['mae_std']:.4f}  "
        f"r = {agg['pearson_r_mean']:+.4f} ± {agg['pearson_r_std']:.4f}"
    )
    print(f"{'='*72}\n")
