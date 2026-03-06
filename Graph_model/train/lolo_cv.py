"""
Graph_model.train.lolo_cv
===========================
Strict Leave-One-Ligand-Out Cross-Validation.

Design
------
9-fold LOLO-CV is the only honest generalization test for this dataset.
With only 9 unique molecules, any row-level split leaks molecular identity:
a model trained on gallic acid at pH 5 and tested on gallic acid at pH 7
performs interpolation, not extrapolation.

Each fold:
  Test  = ALL records for one ligand (all boxes, all pH, all temperatures)
  Train = remaining 8 ligands, 85 % of their records  (random within-ligand)
  Val   = remaining 8 ligands, 15 % of their records

Fold indices are guaranteed disjoint:  train ∩ val ∩ test = ∅

Usage
-----
By index list (when dataset is pre-loaded):
    >>> splitter = LOLOCVSplitter(val_ratio=0.15, seed=42)
    >>> for fold in splitter.split(dataset):
    ...     fold.fold, fold.held_out_ligand
    ...     train_ds = Subset(dataset, fold.train_idx)
    ...     val_ds   = Subset(dataset, fold.val_idx)
    ...     test_ds  = Subset(dataset, fold.test_idx)

By ligand → row-index dict (faster if dataset is not loaded):
    >>> groups = splitter.group_by_ligand(dataset)
    >>> for fold in splitter.split_from_groups(groups):
    ...     ...

Non-anchor records (tier ≠ 0) are assigned to the training set of every fold
and are never placed in val or test.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# Tier codes must match data/dataset.py
_TIER_ANCHOR = 0


@dataclass
class LOLOFold:
    """Single LOLO-CV fold descriptor."""
    fold:              int
    held_out_ligand:   str
    train_idx:         List[int]
    val_idx:           List[int]
    test_idx:          List[int]

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"LOLOFold(fold={self.fold}, held_out='{self.held_out_ligand}', "
            f"train={self.n_train}, val={self.n_val}, test={self.n_test})"
        )


class LOLOCVSplitter:
    """
    Strict Leave-One-Ligand-Out Cross-Validation.

    Generates exactly ``len(unique_ligands)`` folds (typically 9).
    Non-anchor records are always placed in the train split.

    Parameters
    ----------
    val_ratio  : float
        Fraction of non-held-out anchor records to use for validation.
        Stratified over box type within each ligand.  Default 0.15.
    seed       : int
        Random seed for train/val partition of non-held-out records.
    anchor_only_eval : bool
        If True (default), val/test sets contain only anchor (tier=0) records.
    """

    def __init__(
        self,
        val_ratio:        float = 0.15,
        seed:             int   = 42,
        anchor_only_eval: bool  = True,
    ) -> None:
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1); got {val_ratio}")
        self.val_ratio        = val_ratio
        self.seed             = seed
        self.anchor_only_eval = anchor_only_eval

    # ── Public API ─────────────────────────────────────────────────────────────

    def split(self, dataset) -> Iterator[LOLOFold]:
        """
        Yield LOLOFold objects for each ligand.

        Parameters
        ----------
        dataset : any indexable with attributes
            ``dataset[i].ligand_name`` (str)
            ``dataset[i].tier``        (int — 0/1/2)
        """
        groups, non_anchor = self._group_dataset(dataset)
        yield from self._folds_from_groups(groups, non_anchor)

    def group_by_ligand(
        self,
        dataset,
    ) -> tuple[dict[str, list[int]], list[int]]:
        """
        Return (anchor_groups, non_anchor_indices).

        anchor_groups : dict[ligand_name → List[int]]
        non_anchor    : List[int]   (tier ≠ 0, PDBbind / augment)
        """
        return self._group_dataset(dataset)

    def split_from_groups(
        self,
        anchor_groups:  dict[str, list[int]],
        non_anchor_idx: list[int],
    ) -> Iterator[LOLOFold]:
        """
        Yield folds from pre-computed group dictionaries.
        Useful when dataset is too large to index repeatedly.
        """
        yield from self._folds_from_groups(anchor_groups, non_anchor_idx)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _group_dataset(
        self,
        dataset,
    ) -> tuple[dict[str, list[int]], list[int]]:
        anchor_groups: dict[str, list[int]] = {}
        non_anchor:    list[int]            = []

        for i in range(len(dataset)):
            d    = dataset[i]
            tier = int(d.tier.item()) if hasattr(d.tier, 'item') else int(d.tier)
            if tier != _TIER_ANCHOR:
                non_anchor.append(i)
                continue
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            anchor_groups.setdefault(name, []).append(i)

        n_lig = len(anchor_groups)
        n_anc = sum(len(v) for v in anchor_groups.values())
        logger.info(
            "LOLOCVSplitter: %d unique anchor ligands, %d anchor records, "
            "%d non-anchor records.",
            n_lig, n_anc, len(non_anchor)
        )
        return anchor_groups, non_anchor

    def _folds_from_groups(
        self,
        anchor_groups:  dict[str, list[int]],
        non_anchor_idx: list[int],
    ) -> Iterator[LOLOFold]:
        ligands = sorted(anchor_groups.keys())   # deterministic fold order

        for fold_idx, held_out in enumerate(ligands):
            rng = random.Random(self.seed + fold_idx)

            # ── Test set: ALL records for held-out ligand ─────────────────────
            test_idx = list(anchor_groups[held_out])
            rng.shuffle(test_idx)   # shuffle for stochastic minibatch order later

            # ── Train/Val: remaining anchor records ───────────────────────────
            train_idx: list[int] = []
            val_idx:   list[int] = []

            for lig, indices in anchor_groups.items():
                if lig == held_out:
                    continue
                idxs = list(indices)
                rng.shuffle(idxs)
                n     = len(idxs)
                n_val = max(1, round(n * self.val_ratio))
                # Guard: at least 1 in train
                if n <= 1:
                    train_idx.extend(idxs)
                else:
                    val_idx.extend(idxs[:n_val])
                    train_idx.extend(idxs[n_val:])

            # Non-anchor records → training only (never in val/test)
            train_idx.extend(non_anchor_idx)

            # Final shuffle
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)

            fold = LOLOFold(
                fold            = fold_idx,
                held_out_ligand = held_out,
                train_idx       = train_idx,
                val_idx         = val_idx,
                test_idx        = test_idx,
            )
            logger.debug("  %r", fold)
            yield fold

    # ── Verification ──────────────────────────────────────────────────────────

    @staticmethod
    def verify_no_leakage(fold: LOLOFold) -> bool:
        """
        Assert that test molecules are absent from train/val.
        Returns True if clean, raises AssertionError otherwise.
        """
        train_set = set(fold.train_idx)
        val_set   = set(fold.val_idx)
        test_set  = set(fold.test_idx)

        assert len(train_set & test_set) == 0, \
            f"DATA LEAKAGE: {len(train_set & test_set)} indices shared between train and test"
        assert len(val_set & test_set) == 0, \
            f"DATA LEAKAGE: {len(val_set & test_set)} indices shared between val and test"
        assert len(train_set & val_set) == 0, \
            f"OVERLAP: {len(train_set & val_set)} indices shared between train and val"
        return True
