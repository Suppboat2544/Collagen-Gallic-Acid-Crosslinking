"""
Graph_model.model.uncertainty
================================
Heteroscedastic prediction head + Gaussian NLL loss.

Wraps any Option{A,B,C,D} model so that the final layer predicts *both* the
mean μ̂ and log-variance log σ̂² of binding energy.

Output
------
    (μ̂, log_σ²)     where  σ² = exp(log_σ²)

Loss   (Gaussian negative log-likelihood)
----
    𝓛 = (ΔG − μ̂)² / (2σ̂²) + ½ log σ̂²

    Numerically:
    𝓛 = 0.5 * exp(−log_σ²) * (ΔG − μ̂)²  +  0.5 * log_σ²

Interpretation: large σ̂ → molecule has box-dependent binding variation (like PGG).

Reference
---------
Nix D., Weigend A. "Estimating the Mean and Variance of the Target
Probability Distribution." ICNN 1994.
Kendall A., Gal Y. "What Uncertainties Do We Need in Bayesian Deep
Learning for Computer Vision?"  NeurIPS 2017.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ── Heteroscedastic wrapper ──────────────────────────────────────────────────

class HeteroscedasticWrapper(nn.Module):
    """
    Wrap any base model to produce (μ, log_σ²) instead of a single scalar.

    The base model's final linear layer (outputting [B, 1]) is replaced by a
    layer outputting [B, 2]:  column 0 = μ̂,  column 1 = log σ̂².

    Parameters
    ----------
    base_model   : nn.Module — any Option A/B/C/D instance
    min_log_var  : float — clamp lower bound on log σ² for numerical stability
    max_log_var  : float — clamp upper bound (avoids σ → ∞ collapse)
    """

    def __init__(
        self,
        base_model:  nn.Module,
        min_log_var: float = -10.0,
        max_log_var: float = 10.0,
    ) -> None:
        super().__init__()
        self.base    = base_model
        self.min_lv  = min_log_var
        self.max_lv  = max_log_var

        # Identify and replace the final Linear layer in the base model's MLP
        self._replace_final_layer()

    # ── Layer surgery ────────────────────────────────────────────────────────

    def _replace_final_layer(self) -> None:
        """
        Walk the base model's MLP to find the last nn.Linear(…, 1) and
        replace it with nn.Linear(…, 2).
        """
        # All four options have an `mlp` sequential (A, B, D) or `glob_mlp` (C)
        mlp_attr = None
        for name in ('mlp', 'glob_mlp'):
            if hasattr(self.base, name):
                mlp_attr = name
                break

        if mlp_attr is None:
            raise ValueError(
                "Cannot find 'mlp' or 'glob_mlp' attribute on base model. "
                "Ensure the base model is an OptionA/B/C/D instance."
            )

        mlp: nn.Sequential = getattr(self.base, mlp_attr)

        # Find last Linear layer inside the sequential
        last_idx = None
        for i, mod in enumerate(mlp):
            if isinstance(mod, nn.Linear) and mod.out_features == 1:
                last_idx = i

        if last_idx is None:
            raise ValueError("No Linear(…, 1) found inside base model MLP.")

        old = mlp[last_idx]
        new = nn.Linear(old.in_features, 2, bias=True)

        # Copy existing weight into mean channel; init variance channel near 0
        with torch.no_grad():
            new.weight[0] = old.weight[0]
            new.bias[0]   = old.bias[0]
            nn.init.zeros_(new.weight[1])
            # Start with log σ² ≈ −2 ⇒ σ ≈ 0.37 kcal/mol (reasonable prior)
            new.bias[1]   = torch.tensor(-2.0)

        mlp[last_idx] = new

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, data, **kwargs) -> dict[str, Tensor]:
        """
        Returns dict with 'mu', 'log_var', 'sigma'.

        The raw base model output is [B, 2]; we split and clamp.
        """
        raw = self.base(data, **kwargs)

        # Handle different return types:
        #   OptionA   : Tensor [B, 2]
        #   OptionB   : (Tensor [B, 2], attn_list)
        #   OptionC   : (Tensor [B, 2], frag_contrib)
        #   OptionD   : dict  (special — handled separately)
        if isinstance(raw, dict):
            # OptionD: the collagen head now outputs [B, 2]
            out2 = raw['collagen']
        elif isinstance(raw, tuple):
            out2 = raw[0]
        else:
            out2 = raw

        mu      = out2[:, 0:1]                              # [B, 1]
        log_var = out2[:, 1:2].clamp(self.min_lv, self.max_lv)  # [B, 1]
        sigma   = (0.5 * log_var).exp()                     # [B, 1]

        return {'mu': mu, 'log_var': log_var, 'sigma': sigma}

    # ── Convenience ──────────────────────────────────────────────────────────

    def predict(self, data, **kwargs) -> tuple[Tensor, Tensor]:
        """Return (μ̂, σ̂) — drop-in replacement for point-estimate models."""
        out = self.forward(data, **kwargs)
        return out['mu'], out['sigma']


# ── Gaussian NLL loss ────────────────────────────────────────────────────────

class GaussianNLLLoss(nn.Module):
    """
    𝓛 = ½ exp(−log_σ²) · (y − μ̂)²  +  ½ log_σ²

    Equivalent to -log 𝒩(y | μ̂, σ̂²) up to constant ½ log(2π).

    Parameters
    ----------
    reduction : 'mean' | 'sum' | 'none'
    eps       : added inside exp for numerical safety (default 1e-6)
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-6) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps       = eps

    def forward(
        self,
        mu:      Tensor,     # [B, 1]
        log_var: Tensor,     # [B, 1]
        target:  Tensor,     # [B, 1]
    ) -> Tensor:
        """Compute heteroscedastic Gaussian NLL."""
        # Precision = 1/σ² = exp(−log_σ²)
        precision = torch.exp(-log_var) + self.eps
        residual  = (target - mu) ** 2

        nll = 0.5 * precision * residual + 0.5 * log_var

        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        return nll


# ── Calibration utilities ────────────────────────────────────────────────────

def calibration_error(
    mu:     Tensor,
    sigma:  Tensor,
    target: Tensor,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute Expected Calibration Error for a Gaussian predictive distribution.

    For each quantile q ∈ {0.1, 0.2, ..., 0.9} we check what fraction of
    targets fall within the predicted q-quantile interval.  ECE is the mean
    absolute difference between expected and observed coverage.

    Returns dict with 'ece', 'coverages' (list), 'expected' (list).
    """
    import numpy as np
    from scipy.stats import norm as _norm

    mu_np  = mu.detach().cpu().numpy().ravel()
    sig_np = sigma.detach().cpu().numpy().ravel()
    tgt_np = target.detach().cpu().numpy().ravel()

    expected   = []
    observed   = []
    quantiles  = np.linspace(0.1, 0.9, n_bins)

    for q in quantiles:
        lo = _norm.ppf((1 - q) / 2, loc=mu_np, scale=sig_np)
        hi = _norm.ppf((1 + q) / 2, loc=mu_np, scale=sig_np)
        frac = ((tgt_np >= lo) & (tgt_np <= hi)).mean()
        expected.append(float(q))
        observed.append(float(frac))

    ece = float(np.mean(np.abs(np.array(expected) - np.array(observed))))
    return {'ece': ece, 'coverages': observed, 'expected': expected}
