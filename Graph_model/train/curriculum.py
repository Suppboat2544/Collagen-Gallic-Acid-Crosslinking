"""
Graph_model.train.curriculum
==============================
Complexity-ordered curriculum training sampler.

Scientific rationale
--------------------
Curriculum learning (Bengio et al. 2009) orders training examples from easy
(low variance, low structural complexity) to hard (high variance, high MW).

This dataset has a natural complexity gradient that matches the ligand
family hierarchy described in the paper:

  Stage 1 — Simple molecules (SD ≤ 0.36 kcal/mol across boxes)
    NHS, EDC, pyrogallol
    MW range:  115–191 Da    galloyl units: 0–1

  Stage 2 — Medium molecules
    gallic_acid, protocatechuic_acid, EDC_Oacylisourea, EDC_NHS
    MW range:  154–457 Da    galloyl units: 0–1

  Stage 3 — Complex molecules (SD up to 1.12 kcal/mol across boxes)
    ellagic_acid, PGG
    MW range:  302–940 Da    galloyl units: 2–5

Training protocol
-----------------
  Week 1 (epochs  1–N1):   sample only from Stage-1 ligands
  Week 2 (epochs N1–N2):   add Stage-2 ligands (weight 1:1 with Stage 1)
  Week 3 (epochs N2–∞):    add Stage-3 ligands (weight 1:1:1 all stages)

The sampler returns a torch.utils.data.WeightedRandomSampler whose per-sample
weights transition automatically as the active stage advances.

Usage
-----
>>> sampler = CurriculumSampler(dataset, stage_schedule=[10, 20])
>>> loader  = DataLoader(dataset, batch_sampler=None, sampler=sampler, batch_size=64)
>>> for epoch in range(30):
...     sampler.set_epoch(epoch)
...     for batch in loader:
...         ...

Design note: stage_schedule=[10, 20] means
  epochs  0–9  → Stage 1 only
  epochs 10–19 → Stages 1+2
  epochs 20+   → Stages 1+2+3

Reference
---------
Bengio Y. et al. "Curriculum Learning." ICML 2009.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

logger = logging.getLogger(__name__)


# ── Curriculum specification ──────────────────────────────────────────────────
# Three stages ordered by molecular complexity / binding-energy variance.
# These are LIGAND_CATALOGUE keys from data/config.py.

STAGE_LIGANDS: dict[int, list[str]] = {
    1: [
        "NHS",
        "EDC",
        "pyrogallol",
    ],
    2: [
        "gallic_acid",
        "protocatechuic_acid",
        "EDC_Oacylisourea",
        "EDC_NHS",
    ],
    3: [
        "ellagic_acid",
        "PGG",
    ],
}

# Relative weight multiplier per stage when active.
# Higher weight in later stages helps the model not forget simple molecules.
STAGE_WEIGHTS: dict[int, float] = {
    1: 1.0,
    2: 1.0,
    3: 1.0,
}

# All 9 ligands in order of introduction
_ALL_LIGAND_ORDER: list[str] = (
    STAGE_LIGANDS[1] + STAGE_LIGANDS[2] + STAGE_LIGANDS[3]
)


def _ligand_to_stage() -> dict[str, int]:
    """Return {ligand_name: stage_number} mapping."""
    m: dict[str, int] = {}
    for stage, names in STAGE_LIGANDS.items():
        for n in names:
            m[n] = stage
    return m


_LIGAND_STAGE_MAP = _ligand_to_stage()


# ── Main class ────────────────────────────────────────────────────────────────

class CurriculumSampler:
    """
    Epoch-aware curriculum sampler.

    Parameters
    ----------
    dataset        : indexable with ``dataset[i].ligand_name`` attribute
    stage_schedule : list[int] of length 2
        [s1_end_epoch, s2_end_epoch].
        Before s1_end_epoch  → Stage 1 only.
        Before s2_end_epoch  → Stages 1 + 2.
        After  s2_end_epoch  → Stages 1 + 2 + 3  (full dataset).
        Default: [10, 20]
    non_anchor_weight : float
        Relative weight for non-anchor (PDBbind / augment) records.
        Default 0.3 — we want them in training but anchor records dominate.
    seed           : int
    """

    def __init__(
        self,
        dataset,
        stage_schedule:      list[int]  = None,
        non_anchor_weight:   float      = 0.3,
        seed:                int        = 42,
    ) -> None:
        self.dataset           = dataset
        self.stage_schedule    = stage_schedule or [10, 20]
        self.non_anchor_weight = non_anchor_weight
        self.seed              = seed
        self._current_epoch    = 0
        self._weights: Optional[torch.Tensor] = None

        # Pre-compute ligand stage for each sample
        self._sample_stages: list[int] = self._compute_sample_stages()
        logger.info(
            "CurriculumSampler: %d samples  "
            "(stage1=%d, stage2=%d, stage3=%d, non-anchor=%d)",
            len(dataset),
            sum(s == 1 for s in self._sample_stages),
            sum(s == 2 for s in self._sample_stages),
            sum(s == 3 for s in self._sample_stages),
            sum(s == 0 for s in self._sample_stages),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch and recompute sampling weights."""
        self._current_epoch = epoch
        self._weights = self._build_weights(epoch)

    @property
    def current_stage(self) -> int:
        """Active curriculum stage (1, 2, or 3) for current epoch."""
        schedule = self.stage_schedule
        if self._current_epoch < schedule[0]:
            return 1
        elif self._current_epoch < schedule[1]:
            return 2
        return 3

    def get_sampler(
        self,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ) -> WeightedRandomSampler:
        """
        Return a WeightedRandomSampler for the current epoch.
        Call ``set_epoch(epoch)`` before each epoch.
        """
        if self._weights is None:
            self._weights = self._build_weights(self._current_epoch)
        n = num_samples or len(self.dataset)
        return WeightedRandomSampler(
            weights     = self._weights,
            num_samples = n,
            replacement = replacement,
            generator   = torch.Generator().manual_seed(self.seed + self._current_epoch),
        )

    def active_ligands(self) -> list[str]:
        """Return names of ligands active in the current epoch."""
        stage = self.current_stage
        active: list[str] = []
        for s in range(1, stage + 1):
            active.extend(STAGE_LIGANDS[s])
        return active

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_sample_stages(self) -> list[int]:
        """
        Assign each sample a stage integer (0 = non-anchor, 1/2/3 = anchor stage).
        """
        stages: list[int] = []
        for i in range(len(self.dataset)):
            d    = self.dataset[i]
            tier = int(d.tier.item()) if hasattr(d.tier, 'item') else int(d.tier)
            if tier != 0:
                stages.append(0)   # non-anchor
                continue
            name  = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            stage = _LIGAND_STAGE_MAP.get(name, 2)   # unknown → stage 2
            stages.append(stage)
        return stages

    def _build_weights(self, epoch: int) -> torch.Tensor:
        """
        Build per-sample float weights for WeightedRandomSampler.

        Samples from inactive stages have weight 0 → never sampled.
        Active stage weights are scaled by STAGE_WEIGHTS.
        Non-anchor records always have weight ``non_anchor_weight``.
        """
        schedule = self.stage_schedule
        if epoch < schedule[0]:
            active_stages = {1}
        elif epoch < schedule[1]:
            active_stages = {1, 2}
        else:
            active_stages = {1, 2, 3}

        weights = torch.zeros(len(self._sample_stages), dtype=torch.float32)
        for i, stage in enumerate(self._sample_stages):
            if stage == 0:
                weights[i] = self.non_anchor_weight
            elif stage in active_stages:
                weights[i] = STAGE_WEIGHTS[stage]
            # else: weight stays 0 (inactive stage)

        if weights.sum() == 0:
            # Fallback: uniform (should not happen)
            weights.fill_(1.0)
            logger.warning("CurriculumSampler: all weights are zero at epoch %d; using uniform.", epoch)

        # Normalise to [0, 1] for numerical stability
        weights = weights / weights.sum()
        return weights

    def stage_summary(self, epoch: int) -> str:
        """Human-readable summary of active stages at a given epoch."""
        schedule = self.stage_schedule
        if epoch < schedule[0]:
            stage, ligands = 1, STAGE_LIGANDS[1]
        elif epoch < schedule[1]:
            stage, ligands = 2, STAGE_LIGANDS[1] + STAGE_LIGANDS[2]
        else:
            stage, ligands = 3, _ALL_LIGAND_ORDER
        return (
            f"Epoch {epoch:4d} | curriculum stage {stage}/3 | "
            f"active ligands: {', '.join(ligands)}"
        )
