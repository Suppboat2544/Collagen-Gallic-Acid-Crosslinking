"""
Graph_model.model.option_d
============================
Option D — Multi-Task Selectivity GNN
======================================

Architecture
------------
  Shared Encoder  : 4-layer GATv2 on ligand graph + condition encoding
                    (identical backbone to Option A)

  Three Prediction Heads (independent MLPs):
    head_collagen : ΔG_collagen  [B, 1]   (6 156 training samples)
    head_mmp1     : ΔG_MMP1      [B, 1]   (40 training samples)
    head_si       : SI logit     [B, 1]   → exp(·) gives SI = exp(ΔΔG / RT)

  Loss (Kendall et al. 2018 uncertainty weighting)
  ─────────────────────────────────────────────────
    Learnable log-variance s_i = log(σ_i²) per task.
    L_total = Σ_i  exp(−s_i) * L_i   +   s_i

    • This is equivalent to maximising a Gaussian likelihood with learnable σ.
    • exp(−s_i) → automatically down-weights noisy tasks.
    • + s_i → regularises σ from going to ∞.

  Class-weighting for MMP-1
  ──────────────────────────
    mmp1_weight = 10.0  (OptionDConfig default)
    L_mmp1 = mmp1_weight * MSE(pred_mmp1, tgt_mmp1)   [only non-NaN rows]

  Selectivity Index
  ─────────────────
    SI_pred = exp( (ΔG_MMP1 − ΔG_collagen) / k_BT )

    In log-space (numerically stable):
      logSI_pred = (ΔG_mmp1 − ΔG_collagen) / kT
    where kT = 0.592 kcal/mol at 298 K.

    SI_target must be provided as log(SI) ground-truth (NaN for collagen-only rows).

  Returns
  -------
  dict with keys:
    'collagen'     : Tensor [B, 1]
    'mmp1'         : Tensor [B, 1]
    'si'           : Tensor [B, 1]  (log SI)
    'log_var'      : Tensor [3]     (learned log-variances, detached)

  compute_loss() method
  ─────────────────────
    Accepts targets dict {'collagen', 'mmp1', 'si'} with NaN for missing.
    Returns scalar Tensor  (backprop-ready).

References
----------
Kendall A., Gal Y., Cipolla R.
  "Multi-Task Learning Using Uncertainty to Weigh Losses for
   Scene Geometry and Semantics."  CVPR 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .config import OptionDConfig
from .option_a import _scalar_to_batch, _encode_raw, _GATv2Block


# Physical constant (kcal/mol at 298 K)
_KT_298 = 0.592


# ── Shared Encoder ────────────────────────────────────────────────────────────

class _SharedEncoder(nn.Module):
    """
    4-layer GATv2 on the LIGAND graph, identical to Option A backbone.
    Outputs:  h_mol [B, hidden_dim]  (global mean pooled molecule embedding).
    """

    def __init__(self, cfg: OptionDConfig) -> None:
        super().__init__()
        H  = cfg.hidden_dim          # 128
        nh = cfg.gat_heads           # 4
        hd = cfg.gat_head_dim        # 32  (nh × hd = H)
        dr = cfg.dropout

        self.inp_proj = nn.Linear(cfg.ligand_node_dim, H)
        self.edge_proj = nn.Linear(cfg.ligand_edge_dim, cfg.ligand_edge_dim)

        # 4 blocks: first 3 multi-head concat, last averages
        blocks = []
        for i in range(cfg.n_layers):
            concat = (i < cfg.n_layers - 1)
            in_ch  = H if i == 0 else (nh * hd if i > 0 else H)
            # After concat: nh*hd out dim; after average: hd out dim
            blocks.append(_GATv2Block(
                in_channels  = in_ch,
                out_channels = hd,
                heads        = nh,
                edge_dim     = cfg.ligand_edge_dim,
                concat       = concat,
                dropout      = dr,
            ))
        self.blocks = nn.ModuleList(blocks)

        # After last block (concat=False) output is hd=32 → project to H
        self.post_proj = nn.Linear(hd, H)

    def forward(self, data) -> Tensor:
        x  = data['ligand'].x
        ei = data['ligand', 'bond', 'ligand'].edge_index
        ea = data['ligand', 'bond', 'ligand'].edge_attr
        bt = data['ligand'].batch

        x  = self.inp_proj(x)
        ea = self.edge_proj(ea)

        for blk in self.blocks:
            x = blk(x, ei, ea)

        x = self.post_proj(x)                     # [N_atoms, H]
        h = global_mean_pool(x, bt)               # [B, H]
        return h


# ── Task Heads ────────────────────────────────────────────────────────────────

class _RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ── Main Model ────────────────────────────────────────────────────────────────

class OptionD(nn.Module):
    """
    Multi-Task Selectivity GNN — shared encoder, three prediction heads
    with Kendall-style uncertainty-weighted multi-task loss.

    Example usage
    -------------
    model = OptionD()
    out = model(data)
    # out['collagen']  [B, 1]
    # out['mmp1']      [B, 1]
    # out['si']        [B, 1]  (log SI)

    # Compute loss:
    targets = {
        'collagen': torch.randn(B, 1),
        'mmp1':     torch.full((B, 1), float('nan')),   # masked for collagen rows
        'si':       torch.full((B, 1), float('nan')),
    }
    loss = model.compute_loss(out, targets)
    loss.backward()
    """

    def __init__(self, cfg: OptionDConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OptionDConfig()
        self.cfg = cfg
        H  = cfg.hidden_dim        # 128
        HM = cfg.mlp_hidden        # 256
        dr = cfg.dropout

        # ── Shared backbone ───────────────────────────────────────────────────
        self.encoder = _SharedEncoder(cfg)

        # ── Condition encoder ─────────────────────────────────────────────────
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, 64), nn.ReLU(), nn.Linear(64, 32),
        )

        combined = H + 32    # 128 + 32 = 160

        # ── Three task heads ──────────────────────────────────────────────────
        self.head_collagen = _RegressionHead(combined, HM, dr)
        self.head_mmp1     = _RegressionHead(combined, HM, dr)
        self.head_si       = _RegressionHead(combined, HM // 2, dr)

        # ── Kendall log-variance parameters: [log_σ²_coll, log_σ²_mmp1, log_σ²_SI]
        if cfg.learn_log_var:
            self.log_var = nn.Parameter(torch.zeros(cfg.n_tasks))
        else:
            self.register_buffer('log_var', torch.zeros(cfg.n_tasks))

        self._mmp1_w = cfg.mmp1_weight

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, data) -> dict[str, Tensor]:
        device = data['ligand'].x.device
        B      = int(data['ligand'].batch.max().item()) + 1

        # Shared molecule representation
        h_mol  = self.encoder(data)                           # [B, H]

        # Condition
        cond   = self._encode_condition(data, device, B)     # [B, COND_DIM]
        h_cond = self.cond_proj(cond)                        # [B, 32]

        h = torch.cat([h_mol, h_cond], dim=-1)              # [B, 160]

        # Task predictions
        dg_coll = self.head_collagen(h)                      # [B, 1]
        dg_mmp1 = self.head_mmp1(h)                          # [B, 1]

        # Selectivity Index in log-space: logSI = (ΔG_mmp1 − ΔG_coll) / kT
        # Use a separate learnable SI correction head that takes the
        # molecule embedding directly (lets the model learn receptor-specific
        # features beyond simple difference).
        log_si = self.head_si(h) + (dg_mmp1 - dg_coll).detach() / _KT_298

        return {
            'collagen': dg_coll,
            'mmp1':     dg_mmp1,
            'si':       log_si,
            'log_var':  self.log_var.detach(),
        }

    # ── Loss ──────────────────────────────────────────────────────────────────

    def compute_loss(
        self,
        predictions: dict[str, Tensor],
        targets:     dict[str, Tensor],
    ) -> Tensor:
        """
        Compute Kendall-weighted multi-task loss with NaN masking.

        Parameters
        ----------
        predictions : output of forward()
        targets     : dict with keys 'collagen', 'mmp1', 'si'.
                      NaN values are treated as missing and excluded from loss.

        Returns
        -------
        Tensor scalar  (differentiable; contains learnable log_var contributions)
        """
        losses = []

        # ── Task 0: ΔG_collagen ───────────────────────────────────────────────
        L_coll = _masked_mse(predictions['collagen'], targets['collagen'])
        losses.append(L_coll)

        # ── Task 1: ΔG_MMP1 with up-weighting ────────────────────────────────
        L_mmp1 = _masked_mse(
            predictions['mmp1'],
            targets['mmp1'],
            weight = self._mmp1_w,
        )
        losses.append(L_mmp1)

        # ── Task 2: log(SI) ───────────────────────────────────────────────────
        L_si = _masked_mse(predictions['si'], targets['si'])
        losses.append(L_si)

        # ── Kendall weighting ─────────────────────────────────────────────────
        # L_total = Σ_i  exp(−s_i) * L_i  +  s_i
        # where s_i = log_var[i]
        s = self.log_var
        total = torch.zeros(1, device=s.device)
        for i, L in enumerate(losses):
            if L is not None:
                total = total + torch.exp(-s[i]) * L + s[i]

        return total.squeeze(0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_condition(self, data, device: torch.device, B: int) -> Tensor:
        if hasattr(data, 'ph_enc'):
            ph_enc   = _scalar_to_batch(data.ph_enc,        B, device)
            temp_enc = _scalar_to_batch(data.temp_enc,       B, device)
            box_idx  = _scalar_to_batch(data.box_idx,        B, device, long=True)
            rec_flag = _scalar_to_batch(data.receptor_flag,  B, device)
        else:
            ph_enc, temp_enc, box_idx, rec_flag = _encode_raw(data, B, device)
        box_emb = self.box_embed(box_idx)
        cont    = torch.stack([ph_enc, temp_enc, rec_flag], dim=-1)
        return torch.cat([cont, box_emb], dim=-1)


# ── Utility ───────────────────────────────────────────────────────────────────

def _masked_mse(
    pred:   Tensor,
    target: Tensor,
    weight: float = 1.0,
) -> Tensor | None:
    """
    MSE loss ignoring NaN targets.
    Returns None if there are no valid targets (avoid divide-by-zero).
    """
    if target is None:
        return None

    # NaN mask — valid rows only
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return None

    p = pred[mask]
    t = target[mask]
    return weight * F.mse_loss(p, t)
