"""
Graph_model.model.losses
===========================
Advanced loss functions for binding energy prediction.

Problem 4a — ListMLE Rank-Aware Loss
-------------------------------------
Standard MSE treats all errors equally. ListMLE encourages the model to
rank ligands correctly within each condition (pH, box), which is the
actual scientific objective — we care about *relative* binding affinities.

ListMLE: Given a ranked list of ΔG values (ground truth order), compute
the negative log-likelihood of the predicted scores generating that
permutation via Plackett-Luce sampling.

Problem 4c — Physics-Constrained Monotonicity Loss
---------------------------------------------------
A galloyl-unit monotonicity prior: within the same box/pH/temp, molecules
with more galloyl units should (generally) bind more strongly.

  ΔG(PGG, 5-gal) ≤ ΔG(ellagic, 2-gal) ≤ ΔG(gallic, 1-gal)

This is a soft constraint: violated pairs incur a hinge loss.

Combined Loss
-------------
    L_total = α·L_mse + β·L_rank + γ·L_mono + δ·L_nll

where L_nll is heteroscedastic NLL (from uncertainty.py) if applicable.

Public API
----------
  ListMLELoss(reduction='mean')
  MonotonicityLoss(margin=0.1)
  CombinedDockingLoss(alpha, beta, gamma, delta)

References
----------
• Xia F. et al., "Listwise Approach to Learning to Rank," ICML 2008.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ── ListMLE Rank Loss ────────────────────────────────────────────────────────

class ListMLELoss(nn.Module):
    """
    ListMLE ranking loss (Xia et al. 2008).

    Given ground truth ranking (by target ΔG) and model predictions,
    computes the negative log-likelihood of the observed permutation
    under a Plackett-Luce probability model.

    Lower ΔG = stronger binding = should be ranked first.

    Parameters
    ----------
    reduction : 'mean' | 'sum' | 'none'
    eps       : numerical stability epsilon
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-10) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps       = eps

    def forward(
        self,
        pred: Tensor,      # [B, 1] or [B] predicted ΔG
        target: Tensor,    # [B, 1] or [B] true ΔG
        group: Tensor = None,  # [B] group IDs (e.g., box_idx) — rank within groups
    ) -> Tensor:
        """
        Compute ListMLE loss.

        If group is provided, ranking is computed within each group separately.
        If group is None, all predictions form a single list.
        """
        pred   = pred.view(-1)
        target = target.view(-1)
        B = pred.size(0)

        if B <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if group is None:
            return self._list_mle_single(pred, target)

        # Compute per-group ListMLE
        unique_groups = group.unique()
        losses = []
        for g in unique_groups:
            mask = group == g
            if mask.sum() <= 1:
                continue
            losses.append(self._list_mle_single(pred[mask], target[mask]))

        if not losses:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = torch.stack(losses)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    def _list_mle_single(self, pred: Tensor, target: Tensor) -> Tensor:
        """ListMLE for a single ranked list."""
        n = pred.size(0)

        # Sort by ground truth (stronger binding = lower ΔG = first)
        sorted_indices = target.argsort()  # ascending ΔG
        pred_sorted = pred[sorted_indices]

        # Plackett-Luce log-likelihood
        # log P(π) = Σ_i [s_π(i) - log(Σ_{j≥i} exp(s_π(j)))]
        # Use log-sum-exp trick for stability
        max_val = pred_sorted.max()
        shifted = pred_sorted - max_val

        # Cumulative log-sum-exp from the end
        cumsums = torch.logcumsumexp(shifted.flip(0), dim=0).flip(0)

        # Log-likelihood
        log_lik = shifted - cumsums
        return -log_lik.sum() / n


# ── Monotonicity Loss ────────────────────────────────────────────────────────

class MonotonicityLoss(nn.Module):
    """
    Physics-constrained monotonicity regularisation.

    For molecule pairs within the same condition group (pH, box, temp):
    if galloyl_units(A) > galloyl_units(B), then ΔG(A) should be ≤ ΔG(B).

    Violation penalty: max(0, ΔG(A) - ΔG(B) + margin)²

    Parameters
    ----------
    margin    : hinge margin in kcal/mol
    reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(self, margin: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(
        self,
        pred: Tensor,           # [B, 1] or [B]
        galloyl_units: Tensor,  # [B] integer galloyl unit counts
        group: Tensor = None,   # [B] condition group IDs
    ) -> Tensor:
        """
        Compute monotonicity violation loss.

        Within each group, pairs with different galloyl counts are compared.
        """
        pred = pred.view(-1)
        galloyl_units = galloyl_units.view(-1)
        B = pred.size(0)

        if B <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        violations = []

        if group is None:
            v = self._compute_violations(pred, galloyl_units)
            if v is not None:
                violations.append(v)
        else:
            unique_groups = group.unique()
            for g in unique_groups:
                mask = group == g
                if mask.sum() <= 1:
                    continue
                v = self._compute_violations(pred[mask], galloyl_units[mask])
                if v is not None:
                    violations.append(v)

        if len(violations) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        all_violations = torch.cat(violations)

        if all_violations.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.reduction == "mean":
            return all_violations.mean()
        elif self.reduction == "sum":
            return all_violations.sum()
        return all_violations

    def _compute_violations(
        self,
        pred: Tensor,
        galloyl: Tensor,
    ) -> Optional[Tensor]:
        """Compute pairwise monotonicity violations."""
        n = pred.size(0)
        violations = []

        for i in range(n):
            for j in range(i + 1, n):
                if galloyl[i] > galloyl[j]:
                    # A has more galloyl → should have lower ΔG
                    violation = F.relu(pred[i] - pred[j] + self.margin)
                    violations.append(violation ** 2)
                elif galloyl[j] > galloyl[i]:
                    violation = F.relu(pred[j] - pred[i] + self.margin)
                    violations.append(violation ** 2)

        if not violations:
            return None
        return torch.stack(violations)


# ── Pairwise Ranking Loss ───────────────────────────────────────────────────

class PairwiseRankLoss(nn.Module):
    """
    Simpler pairwise ranking loss (BPR-style) as an alternative to ListMLE.

    For each pair (i, j) where target[i] < target[j] (i binds more strongly):
      loss = log(1 + exp(pred[i] - pred[j]))
    """

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        self.margin    = margin
        self.reduction = reduction

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
        n = pred.size(0)

        if n <= 1:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Find all pairs where target[i] < target[j]
        diff_pred   = pred.unsqueeze(0) - pred.unsqueeze(1)      # [n, n]
        diff_target = target.unsqueeze(0) - target.unsqueeze(1)   # [n, n]
        mask = diff_target < 0  # i binds stronger than j

        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Loss: want pred[i] < pred[j], so penalise pred[i] - pred[j] + margin > 0
        losses = F.softplus(diff_pred[mask] + self.margin)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        return losses


# ── Combined Docking Loss ────────────────────────────────────────────────────

class CombinedDockingLoss(nn.Module):
    """
    Combined loss for docking energy prediction:

        L = α·MSE + β·ListMLE + γ·Monotonicity + δ·NLL

    Parameters
    ----------
    alpha : weight for MSE loss
    beta  : weight for rank loss (ListMLE or pairwise)
    gamma : weight for monotonicity regularisation
    delta : weight for heteroscedastic NLL (0 = not used)
    rank_type : 'listmle' or 'pairwise'
    mono_margin : margin for monotonicity hinge loss
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta:  float = 0.1,
        gamma: float = 0.05,
        delta: float = 0.0,
        rank_type: str = "listmle",
        mono_margin: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta

        self.mse_loss  = nn.MSELoss()
        self.rank_loss = ListMLELoss() if rank_type == "listmle" else PairwiseRankLoss()
        self.mono_loss = MonotonicityLoss(margin=mono_margin)

    def forward(
        self,
        pred: Tensor,                        # [B, 1]
        target: Tensor,                      # [B, 1]
        galloyl_units: Tensor = None,        # [B] (optional, for monotonicity)
        group: Tensor = None,                # [B] (optional, for within-group ranking)
        log_var: Tensor = None,              # [B, 1] (optional, for heteroscedastic NLL)
    ) -> dict[str, Tensor]:
        """
        Returns dict with 'total', 'mse', 'rank', 'mono', 'nll'.
        """
        losses = {}

        # MSE
        losses['mse'] = self.mse_loss(pred, target)

        # Rank loss
        if self.beta > 0 and pred.size(0) > 1:
            losses['rank'] = self.rank_loss(pred, target, group if hasattr(self.rank_loss, 'forward') and group is not None else None) if isinstance(self.rank_loss, ListMLELoss) else self.rank_loss(pred, target)
        else:
            losses['rank'] = torch.tensor(0.0, device=pred.device)

        # Monotonicity loss
        if self.gamma > 0 and galloyl_units is not None:
            losses['mono'] = self.mono_loss(pred, galloyl_units, group)
        else:
            losses['mono'] = torch.tensor(0.0, device=pred.device)

        # Heteroscedastic NLL
        if self.delta > 0 and log_var is not None:
            precision = torch.exp(-log_var)
            nll = 0.5 * precision * (target - pred) ** 2 + 0.5 * log_var
            losses['nll'] = nll.mean()
        else:
            losses['nll'] = torch.tensor(0.0, device=pred.device)

        # Total
        losses['total'] = (
            self.alpha * losses['mse'] +
            self.beta  * losses['rank'] +
            self.gamma * losses['mono'] +
            self.delta * losses['nll']
        )

        return losses
