"""
Graph_model.data.splitter
==========================
Stratified train / validation / test split for the three-tier dataset.

Split strategy
--------------
Tier assignments
  - Transfer (PDBbind) and Augment records → train set ONLY
    (they provide pre-training signal; we never evaluate on external data)
  - Anchor records → stratified split

Stratification for anchor records
  Key: (ligand_name, box_type_canonical)
  Ensures every split contains all 9 ligands across all 8 box types.
  pH and temperature are secondarily balanced by the shuffle seed.

Default ratios  →  70 % train  /  15 % val  /  15 % test

Usage
-----
>>> from Graph_model.data.splitter import StratifiedSplitter
>>> splitter = StratifiedSplitter(val_ratio=0.15, test_ratio=0.15, seed=42)
>>> train_idx, val_idx, test_idx = splitter.split(dataset)
>>> print(len(train_idx), len(val_idx), len(test_idx))

Anchor-only split (for cross-validation)
-----------------------------------------
>>> splitter = StratifiedSplitter()
>>> train_idx, val_idx, test_idx = splitter.split_anchor_only(dataset)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Tier codes (must match dataset.py)
_TIER_ANCHOR   = 0
_TIER_TRANSFER = 1
_TIER_AUGMENT  = 2


class StratifiedSplitter:
    """
    Produce stratified train / validation / test index splits.

    Parameters
    ----------
    val_ratio   : float, default 0.15
    test_ratio  : float, default 0.15
    seed        : int, default 42
    anchor_only_val_test : bool, default True
        If True, validation and test sets contain ONLY anchor records.
        Non-anchor records go entirely to train.
    """

    def __init__(
        self,
        val_ratio:   float = 0.15,
        test_ratio:  float = 0.15,
        seed:        int   = 42,
        anchor_only_val_test: bool = True,
    ) -> None:
        if val_ratio + test_ratio >= 1.0:
            raise ValueError("val_ratio + test_ratio must be < 1.0")
        self.val_ratio   = val_ratio
        self.test_ratio  = test_ratio
        self.seed        = seed
        self.anchor_only_val_test = anchor_only_val_test

    # ── Main split ────────────────────────────────────────────────────────────

    def split(self, dataset) -> Tuple[List[int], List[int], List[int]]:
        """
        Split *dataset* into (train, val, test) index lists.

        dataset must support:
          len(dataset)
          dataset[i].tier          (int, 0/1/2)
          dataset[i].ligand_name   (str)
          dataset[i].cond          (Tensor [4], cond[2] = box_idx)
          dataset[i].sample_id     (str)

        Returns
        -------
        train_idx : List[int]
        val_idx   : List[int]
        test_idx  : List[int]
        """
        n = len(dataset)
        if n == 0:
            return [], [], []

        # Separate anchor indices from non-anchor
        anchor_idx   = []
        non_anchor_idx = []
        for i in range(n):
            d = dataset[i]
            tier = int(d.tier) if hasattr(d.tier, 'item') else d.tier
            if tier == _TIER_ANCHOR:
                anchor_idx.append(i)
            else:
                non_anchor_idx.append(i)

        # Stratified split on anchor records
        a_train, a_val, a_test = self._stratified_split(dataset, anchor_idx)

        if self.anchor_only_val_test:
            # All non-anchor → train
            train_idx = a_train + non_anchor_idx
        else:
            train_idx = a_train
            # Randomly assign non-anchor to train (90%) / val (5%) / test (5%)
            rng = np.random.default_rng(self.seed + 1)
            perm = rng.permutation(len(non_anchor_idx)).tolist()
            cut1 = int(0.9 * len(perm))
            cut2 = int(0.95 * len(perm))
            train_idx = a_train + [non_anchor_idx[i] for i in perm[:cut1]]
            a_val   = a_val   + [non_anchor_idx[i] for i in perm[cut1:cut2]]
            a_test  = a_test  + [non_anchor_idx[i] for i in perm[cut2:]]

        logger.info(
            "Split: %d train / %d val / %d test  "
            "(anchor: %d / %d / %d, non-anchor in train: %d)",
            len(train_idx), len(a_val), len(a_test),
            len(a_train), len(a_val), len(a_test),
            len(non_anchor_idx) if self.anchor_only_val_test else 0,
        )
        return sorted(train_idx), sorted(a_val), sorted(a_test)

    def split_anchor_only(self, dataset) -> Tuple[List[int], List[int], List[int]]:
        """
        Split using ONLY anchor records (ignores transfer / augment).
        Useful for cross-validation on the core 6,196-record set.
        """
        n = len(dataset)
        anchor_idx = [
            i for i in range(n)
            if (int(dataset[i].tier) if hasattr(dataset[i].tier, 'item')
                else dataset[i].tier) == _TIER_ANCHOR
        ]
        return self._stratified_split(dataset, anchor_idx)

    # ── Stratified splitter ───────────────────────────────────────────────────

    def _stratified_split(
        self,
        dataset,
        indices: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Block-stratified split by (ligand_name, box_type_idx).

        For each stratum (ligand, box_type), records are shuffled and then
        allocated to train/val/test proportionally.
        """
        rng = random.Random(self.seed)

        # Group indices by stratum key
        strata: dict[tuple, List[int]] = defaultdict(list)
        for i in indices:
            d = dataset[i]
            ligand   = d.ligand_name if d.ligand_name else "unknown"
            # box_idx is stored as cond[2] (float → int)
            box_type = int(d.cond[2].item()) if hasattr(d.cond, 'item') else int(d.cond[2])
            strata[(ligand, box_type)].append(i)

        logger.debug("StratifiedSplitter: %d strata detected.", len(strata))

        train_idx, val_idx, test_idx = [], [], []

        for key, group in strata.items():
            rng.shuffle(group)
            n = len(group)
            n_test = max(1, round(n * self.test_ratio))
            n_val  = max(1, round(n * self.val_ratio))
            # Guard: need at least 1 record in train
            if n <= 2:
                # Too few — all to train
                train_idx += group
                continue
            if n <= 3:
                test_idx.append(group[0])
                train_idx += group[1:]
                continue

            test_idx  += group[:n_test]
            val_idx   += group[n_test: n_test + n_val]
            train_idx += group[n_test + n_val:]

        # Shuffle within each split for downstream DataLoader
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

        return train_idx, val_idx, test_idx

    # ── Verification ──────────────────────────────────────────────────────────

    def verify_coverage(
        self,
        dataset,
        train_idx: List[int],
        val_idx: List[int],
        test_idx: List[int],
    ) -> dict:
        """
        Check that all ligands and box types are represented in EACH split.

        Returns a dict with 'missing_ligands' and 'missing_box_types' per split.
        """
        def _collect(idxs):
            ligands, boxes = set(), set()
            for i in idxs:
                d = dataset[i]
                if (int(d.tier) if hasattr(d.tier, 'item') else d.tier) != _TIER_ANCHOR:
                    continue
                ligands.add(d.ligand_name)
                boxes.add(int(d.cond[2].item()) if hasattr(d.cond, 'item') else int(d.cond[2]))
            return ligands, boxes

        all_ligands = set()
        all_boxes   = set()
        for i in range(len(dataset)):
            d = dataset[i]
            if (int(d.tier) if hasattr(d.tier, 'item') else d.tier) == _TIER_ANCHOR:
                all_ligands.add(d.ligand_name)
                all_boxes.add(int(d.cond[2].item()) if hasattr(d.cond, 'item')
                               else int(d.cond[2]))

        report = {}
        for name, idxs in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            lg, bx = _collect(idxs)
            report[name] = {
                "missing_ligands":   sorted(all_ligands - lg),
                "missing_box_types": sorted(all_boxes - bx),
                "ligand_count":      len(lg),
                "box_type_count":    len(bx),
            }
        return report
