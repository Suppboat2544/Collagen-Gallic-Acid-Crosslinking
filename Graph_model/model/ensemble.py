"""
Graph_model.model.ensemble
============================
Deep ensemble of M independently-trained models.

Epistemic uncertainty = std across ensemble member predictions.
Aleatoric uncertainty = mean of per-model σ̂ (from heteroscedastic heads).

Workflow
--------
1.  Train M models with different random seeds (via train.finetune).
2.  Wrap them:
        ensemble = DeepEnsemble(model_list)
3.  Predict:
        out = ensemble(data)
        # out['mu_mean']      [B, 1]  — ensemble mean prediction
        # out['mu_std']       [B, 1]  — epistemic uncertainty
        # out['sigma_mean']   [B, 1]  — mean aleatoric uncertainty (if heteroscedastic)
        # out['total_var']    [B, 1]  — σ²_epistemic + σ²_aleatoric

Reference
---------
Lakshminarayanan B., Pritzel A., Blundell C.
  "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles."
  NeurIPS 2017.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class DeepEnsemble(nn.Module):
    """
    Ensemble of M models for combined prediction + uncertainty.

    Parameters
    ----------
    models : list[nn.Module]
        Independently trained models.  Each should accept the same forward()
        signature (HeteroData) and return:
          - Tensor [B, 1]  (point-estimate model), OR
          - dict with 'mu', 'sigma' (HeteroscedasticWrapper)
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        assert len(models) >= 2, "Need ≥2 models for an ensemble"
        self.members = nn.ModuleList(models)
        self.M       = len(models)

    # ── Forward ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, data, **kwargs) -> dict[str, Tensor]:
        """
        Run all ensemble members and aggregate predictions.

        Returns
        -------
        dict with keys:
            mu_mean     : [B, 1]  ensemble mean  (best point estimate)
            mu_std      : [B, 1]  epistemic std (std across members)
            sigma_mean  : [B, 1]  mean aleatoric σ (only if heteroscedastic)
            total_var   : [B, 1]  σ²_epistemic + σ²_aleatoric
            all_mu      : [M, B, 1]  per-member predictions
        """
        all_mu:    list[Tensor] = []
        all_sigma: list[Tensor] = []
        heteroscedastic = False

        for model in self.members:
            model.eval()
            raw = model(data, **kwargs)

            if isinstance(raw, dict) and 'mu' in raw:
                # Heteroscedastic output
                heteroscedastic = True
                all_mu.append(raw['mu'])
                all_sigma.append(raw['sigma'])
            elif isinstance(raw, tuple):
                # OptionB/C return (tensor, extra)
                all_mu.append(raw[0])
            elif isinstance(raw, Tensor):
                all_mu.append(raw)
            else:
                # OptionD dict — use collagen head
                all_mu.append(raw.get('collagen', raw.get('mu')))

        stack_mu = torch.stack(all_mu, dim=0)       # [M, B, 1]
        mu_mean  = stack_mu.mean(dim=0)              # [B, 1]
        mu_std   = stack_mu.std(dim=0)               # [B, 1]  epistemic

        result = {
            'mu_mean':  mu_mean,
            'mu_std':   mu_std,
            'all_mu':   stack_mu,
        }

        if heteroscedastic and all_sigma:
            stack_sigma  = torch.stack(all_sigma, dim=0)      # [M, B, 1]
            sigma_mean   = stack_sigma.mean(dim=0)            # [B, 1]
            aleatoric_var = (stack_sigma ** 2).mean(dim=0)    # mean σ²
            epistemic_var = mu_std ** 2
            result['sigma_mean'] = sigma_mean
            result['total_var']  = aleatoric_var + epistemic_var
        else:
            result['sigma_mean'] = torch.zeros_like(mu_mean)
            result['total_var']  = mu_std ** 2

        return result

    # ── Convenience ──────────────────────────────────────────────────────────

    def predict(self, data, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns (mu_mean, epistemic_std, total_std).
        """
        out = self.forward(data, **kwargs)
        total_std = out['total_var'].sqrt()
        return out['mu_mean'], out['mu_std'], total_std

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save(self, save_dir: str | Path) -> None:
        """Save each member as member_0.pt … member_{M-1}.pt."""
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.members):
            torch.save(m.state_dict(), d / f"member_{i}.pt")

    @classmethod
    def load(
        cls,
        model_factory,
        save_dir: str | Path,
        n_members: int = 5,
        device: Optional[torch.device] = None,
    ) -> 'DeepEnsemble':
        """
        Reconstruct ensemble from saved checkpoints.

        Parameters
        ----------
        model_factory : callable() → nn.Module  (fresh model instance)
        save_dir      : directory containing member_0.pt … member_{n-1}.pt
        n_members     : expected number of members
        device        : device to load onto
        """
        d = Path(save_dir)
        models = []
        for i in range(n_members):
            m = model_factory()
            m.load_state_dict(torch.load(d / f"member_{i}.pt", map_location=device or 'cpu'))
            if device is not None:
                m = m.to(device)
            m.eval()
            models.append(m)
        return cls(models)


# ── Training helper ──────────────────────────────────────────────────────────

def train_ensemble(
    model_factory,
    train_fn,
    n_members:   int   = 5,
    base_seed:   int   = 42,
    **train_kwargs,
) -> DeepEnsemble:
    """
    Train M models with different random seeds.

    Parameters
    ----------
    model_factory : callable() → nn.Module
    train_fn      : callable(model, seed=int, **kwargs) → trained model
                    Must accept 'seed' keyword for reproducibility.
    n_members     : number of ensemble members
    base_seed     : starting seed (member i uses base_seed + i)
    **train_kwargs: passed through to train_fn

    Returns
    -------
    DeepEnsemble
    """
    models = []
    for i in range(n_members):
        seed = base_seed + i
        model = model_factory()
        trained = train_fn(model, seed=seed, **train_kwargs)
        trained.eval()
        models.append(trained)
    return DeepEnsemble(models)
