"""
Graph_model.train.stratified_eval
====================================
Stratified LOLO-CV evaluation — Problem 6a.

Extends LOLO-CV with fine-grained breakdowns:
  • Per-box-type metrics   (GLU_cluster, LYS_cluster, …)
  • Per-receptor metrics   (collagen vs MMP-1)
  • Per-ligand-group metrics (primary, intermediate, GA_analogue)
  • Galloyl-unit strata     (0, 1, 2, 5)

Usage
-----
    >>> from Graph_model.train.stratified_eval import StratifiedEvaluator
    >>> evaluator = StratifiedEvaluator(dataset)
    >>> report = evaluator.evaluate_fold(fold, predictions, targets)
    >>> evaluator.print_stratified_report(report)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrataMetrics:
    """Metrics for a single stratum (subset of predictions)."""
    stratum_key:   str
    stratum_value: str
    n_samples:     int
    rmse:          float
    mae:           float
    pearson_r:     float
    spearman_r:    float
    median_ae:     float

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.stratum_key,
            "value": self.stratum_value,
            "n": self.n_samples,
            "rmse": self.rmse,
            "mae": self.mae,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "median_ae": self.median_ae,
        }


@dataclass
class StratifiedReport:
    """Full stratified evaluation report for one fold."""
    fold_idx:         int
    held_out_ligand:  str
    overall_rmse:     float
    overall_mae:      float
    overall_pearson:  float
    overall_spearman: float
    by_box_type:      List[StrataMetrics] = field(default_factory=list)
    by_receptor:      List[StrataMetrics] = field(default_factory=list)
    by_ligand_group:  List[StrataMetrics] = field(default_factory=list)
    by_galloyl_units: List[StrataMetrics] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold_idx,
            "held_out": self.held_out_ligand,
            "overall": {
                "rmse": self.overall_rmse,
                "mae": self.overall_mae,
                "pearson_r": self.overall_pearson,
                "spearman_r": self.overall_spearman,
            },
            "by_box_type":      [m.as_dict() for m in self.by_box_type],
            "by_receptor":      [m.as_dict() for m in self.by_receptor],
            "by_ligand_group":  [m.as_dict() for m in self.by_ligand_group],
            "by_galloyl_units": [m.as_dict() for m in self.by_galloyl_units],
        }


def _compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute regression metrics on numpy arrays."""
    from scipy.stats import pearsonr, spearmanr

    n = len(preds)
    if n < 2:
        return {"rmse": float("nan"), "mae": float("nan"),
                "pearson_r": float("nan"), "spearman_r": float("nan"),
                "median_ae": float("nan")}

    residuals = preds - targets
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae  = float(np.mean(np.abs(residuals)))
    median_ae = float(np.median(np.abs(residuals)))

    # Guard against constant arrays
    if np.std(targets) < 1e-10 or np.std(preds) < 1e-10:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson  = float(pearsonr(preds, targets)[0])
        spearman = float(spearmanr(preds, targets)[0])

    return {"rmse": rmse, "mae": mae, "pearson_r": pearson,
            "spearman_r": spearman, "median_ae": median_ae}


class StratifiedEvaluator:
    """
    Evaluates model predictions with stratification over metadata.

    Parameters
    ----------
    dataset : indexable PyG dataset
        Each data point should have:
          .ligand_name, .box_type / .box_idx, .receptor_flag,
          .galloyl_units (or derivable from ligand_name)
    """

    def __init__(self, dataset=None):
        self.dataset = dataset
        # Lazy import — only needed for galloyl lookup
        try:
            from Graph_model.data.config import (
                LIGAND_CATALOGUE, BOX_TYPE_VOCAB, LIGAND_GROUPS
            )
            self._catalogue = LIGAND_CATALOGUE
            self._box_vocab = {v: k for k, v in BOX_TYPE_VOCAB.items()}
            self._ligand_groups = LIGAND_GROUPS
        except ImportError:
            self._catalogue = {}
            self._box_vocab = {}
            self._ligand_groups = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate_fold(
        self,
        test_indices:  Sequence[int],
        predictions:   np.ndarray,
        targets:       np.ndarray,
        fold_idx:      int = 0,
        held_out:      str = "unknown",
        metadata:      Optional[List[dict]] = None,
    ) -> StratifiedReport:
        """
        Compute stratified metrics for one LOLO-CV fold.

        Parameters
        ----------
        test_indices : indices into self.dataset (used to extract metadata)
        predictions  : [N] float array of predicted ΔG
        targets      : [N] float array of true ΔG
        fold_idx     : fold number
        held_out     : held-out ligand name
        metadata     : optional list of dicts with keys:
                       box_type, receptor, ligand_group, galloyl_units
                       If None, extracted from self.dataset[test_indices]
        """
        preds = np.asarray(predictions).ravel()
        targs = np.asarray(targets).ravel()
        assert len(preds) == len(targs), "predictions and targets must have same length"

        # Overall
        overall = _compute_metrics(preds, targs)

        # Extract metadata
        if metadata is None:
            metadata = self._extract_metadata(test_indices)

        # Group indices by strata
        box_groups     = defaultdict(list)
        receptor_groups = defaultdict(list)
        ligand_groups  = defaultdict(list)
        galloyl_groups = defaultdict(list)

        for i, meta in enumerate(metadata):
            box_groups[meta.get("box_type", "unknown")].append(i)
            receptor_groups[meta.get("receptor", "unknown")].append(i)
            ligand_groups[meta.get("ligand_group", "unknown")].append(i)
            galloyl_groups[str(meta.get("galloyl_units", "?"))].append(i)

        # Compute per-stratum metrics
        by_box   = self._metrics_for_groups("box_type",      box_groups,     preds, targs)
        by_rec   = self._metrics_for_groups("receptor",      receptor_groups, preds, targs)
        by_lig   = self._metrics_for_groups("ligand_group",  ligand_groups,  preds, targs)
        by_gal   = self._metrics_for_groups("galloyl_units", galloyl_groups, preds, targs)

        return StratifiedReport(
            fold_idx=fold_idx,
            held_out_ligand=held_out,
            overall_rmse=overall["rmse"],
            overall_mae=overall["mae"],
            overall_pearson=overall["pearson_r"],
            overall_spearman=overall["spearman_r"],
            by_box_type=by_box,
            by_receptor=by_rec,
            by_ligand_group=by_lig,
            by_galloyl_units=by_gal,
        )

    def aggregate_reports(
        self,
        reports: List[StratifiedReport],
    ) -> dict[str, Any]:
        """
        Aggregate stratified reports across all LOLO-CV folds.
        Returns mean ± std for each stratum.
        """
        overall_rmses = [r.overall_rmse for r in reports]
        overall_pearsons = [r.overall_pearson for r in reports]

        # Collect per-stratum metrics across folds
        strata_collections: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for report in reports:
            for stratum_list, key_name in [
                (report.by_box_type,      "box_type"),
                (report.by_receptor,      "receptor"),
                (report.by_ligand_group,  "ligand_group"),
                (report.by_galloyl_units, "galloyl_units"),
            ]:
                for m in stratum_list:
                    label = f"{key_name}/{m.stratum_value}"
                    strata_collections[label]["rmse"].append(m.rmse)
                    strata_collections[label]["pearson_r"].append(m.pearson_r)
                    strata_collections[label]["n"].append(m.n_samples)

        # Summarize
        summary: dict[str, Any] = {
            "overall": {
                "rmse_mean": float(np.nanmean(overall_rmses)),
                "rmse_std":  float(np.nanstd(overall_rmses)),
                "pearson_mean": float(np.nanmean(overall_pearsons)),
                "pearson_std":  float(np.nanstd(overall_pearsons)),
            },
            "strata": {},
        }

        for label, metrics in sorted(strata_collections.items()):
            summary["strata"][label] = {
                "rmse_mean":    float(np.nanmean(metrics["rmse"])),
                "rmse_std":     float(np.nanstd(metrics["rmse"])),
                "pearson_mean": float(np.nanmean(metrics["pearson_r"])),
                "pearson_std":  float(np.nanstd(metrics["pearson_r"])),
                "avg_n":        float(np.mean(metrics["n"])),
            }

        return summary

    def print_stratified_report(self, report: StratifiedReport) -> None:
        """Pretty-print a single fold's stratified report."""
        print(f"\n{'=' * 70}")
        print(f"  Fold {report.fold_idx} — held out: {report.held_out_ligand}")
        print(f"  Overall RMSE={report.overall_rmse:.4f}  "
              f"Pearson={report.overall_pearson:.4f}  "
              f"Spearman={report.overall_spearman:.4f}")
        print(f"{'=' * 70}")

        for name, strata_list in [
            ("Box Type",      report.by_box_type),
            ("Receptor",      report.by_receptor),
            ("Ligand Group",  report.by_ligand_group),
            ("Galloyl Units", report.by_galloyl_units),
        ]:
            if not strata_list:
                continue
            print(f"\n  {name}:")
            for m in strata_list:
                print(f"    {m.stratum_value:24s}  n={m.n_samples:4d}  "
                      f"RMSE={m.rmse:.4f}  r={m.pearson_r:+.3f}")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extract_metadata(self, indices: Sequence[int]) -> List[dict]:
        """Pull metadata from dataset items."""
        metadata = []
        for idx in indices:
            d = self.dataset[idx]
            lig_name = getattr(d, "ligand_name", "unknown")
            if not isinstance(lig_name, str):
                lig_name = str(lig_name)

            # Box type
            box_idx = getattr(d, "box_idx", None)
            if box_idx is not None:
                box_idx = int(box_idx.item()) if hasattr(box_idx, "item") else int(box_idx)
                box_type = self._box_vocab.get(box_idx, f"box_{box_idx}")
            else:
                box_type = str(getattr(d, "box_type", "unknown"))

            # Receptor
            rec_flag = getattr(d, "receptor_flag", None)
            if rec_flag is not None:
                rec_flag = float(rec_flag.item()) if hasattr(rec_flag, "item") else float(rec_flag)
                receptor = "mmp1" if rec_flag > 0.5 else "collagen"
            else:
                receptor = str(getattr(d, "receptor", "unknown"))

            # Ligand group
            lig_group = self._ligand_groups.get(lig_name, "unknown")

            # Galloyl units
            if lig_name in self._catalogue:
                gal_units = self._catalogue[lig_name].get("galloyl_units", 0)
            else:
                gal_units = int(getattr(d, "galloyl_units", 0))

            metadata.append({
                "box_type": box_type,
                "receptor": receptor,
                "ligand_group": lig_group,
                "galloyl_units": gal_units,
            })
        return metadata

    def _metrics_for_groups(
        self,
        key_name:  str,
        groups:    dict[str, list[int]],
        preds:     np.ndarray,
        targs:     np.ndarray,
    ) -> List[StrataMetrics]:
        """Compute metrics for each group in the stratum."""
        results = []
        for value, idxs in sorted(groups.items()):
            if len(idxs) < 1:
                continue
            p = preds[idxs]
            t = targs[idxs]
            m = _compute_metrics(p, t)
            results.append(StrataMetrics(
                stratum_key=key_name,
                stratum_value=str(value),
                n_samples=len(idxs),
                rmse=m["rmse"],
                mae=m["mae"],
                pearson_r=m["pearson_r"],
                spearman_r=m["spearman_r"],
                median_ae=m["median_ae"],
            ))
        return results
