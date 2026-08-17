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
    dict with keys: 'rmse', 'mae', 'r2', 'pearson_r', 'spearman_r', 'n'
    """
    preds   = np.asarray(preds,   dtype=float).ravel()
    targets = np.asarray(targets, dtype=float).ravel()

    # Drop any NaN pairs
    valid = ~(np.isnan(preds) | np.isnan(targets))
    preds, targets = preds[valid], targets[valid]
    n = len(preds)

    if n == 0:
        return {'rmse': float('nan'), 'mae': float('nan'), 'r2': float('nan'),
                'pearson_r': float('nan'), 'spearman_r': float('nan'), 'n': 0}

    err    = preds - targets
    rmse   = float(np.sqrt((err ** 2).mean()))
    mae    = float(np.abs(err).mean())

    # Coefficient of determination, 1 - SS_res/SS_tot.
    #
    # Reported because correlation alone cannot distinguish a useful model from
    # a useless one: a model that predicts the training mean for every input
    # scores r ~ 0 but R2 ~ 0 too, while a model WORSE than that constant
    # predictor scores R2 < 0. Only R2 states whether the model beats "always
    # guess the mean", which is the question a reader actually has. This is the
    # regression-against-the-mean form, NOT the square of Pearson r; the two
    # differ whenever predictions are biased or mis-scaled, and reporting r^2 in
    # place of R2 hides exactly that failure.
    ss_res = float((err ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

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
        'r2':         r2,
        'pearson_r':  pearson_r,
        'spearman_r': spearman_r,
        'n':          n,
    }


def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Ranks with ties averaged ('average' method, as in scipy.stats.rankdata).

    Ties are not a corner case here: Vinardo writes affinities to two decimal
    places, so thousands of rows share values and a held-out ligand routinely
    has repeated scores.
    """
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    # Average the ranks within each run of equal values.
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman's rho, tie-corrected, without a scipy dependency.

    Computed as Pearson's r on average ranks. The previous implementation used
    the shortcut formula

        rho = 1 - 6*sum(d^2) / (n*(n^2 - 1))

    which is only valid when there are NO tied values. It also ranked with
    argsort(argsort(.)), which breaks ties arbitrarily by position, so the
    reported rho depended on row order in the CSV. With Vinardo scores rounded
    to two decimals, ties are guaranteed and the shortcut is biased.
    """
    n = len(x)
    if n < 2:
        return float('nan')
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


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
    r2          : float optional — coefficient of determination vs the mean
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
    r2:         float = float('nan')
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
            f"r={self.pearson_r:+.4f}  ρ={self.spearman_r:+.4f}  "
            f"R²={self.r2:+.4f}"
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
      r2_mean, r2_std,
      n_folds, n_total_test

    R² is aggregated alongside the correlations because it is the only one of
    them that answers "does this beat predicting the mean?". A negative
    r2_mean means it does not.
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
    r2m, r2s = _stats('r2')

    return {
        'rmse_mean':       rm, 'rmse_std':       rs,
        'mae_mean':        mm, 'mae_std':         ms,
        'pearson_r_mean':  pm, 'pearson_r_std':   ps,
        'spearman_r_mean': sm, 'spearman_r_std':  ss,
        'r2_mean':         r2m, 'r2_std':        r2s,
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
