"""
Graph_model.model.option_a
============================
Option A — Baseline: Condition-Aware GCN/GAT
=============================================

Architecture
------------
  Input  : Level-1 ligand molecular graph  (node_dim=35, edge_dim=13)
  GNN    : 4-layer GATv2Conv  (4 heads × 32 dim = 128 per node)
  Readout: global mean pool → h_lig  [B, 128]
  Cond   : box_idx → Embedding(8, 16);  concat [ph, temp, rec] → Linear→32
  Output : MLP([h_lig ‖ cond], 256, 1) → ΔG  (kcal/mol)

Why GATv2 over vanilla GAT
--------------------------
GATv2 (Brody et al. 2022) fixes the "static attention" problem in GAT by
computing attention coefficients from a dynamic, non-linear combination of
source and target node features.  It is strictly more expressive and drops in
as a direct replacement with identical API.

Expected performance : RMSE ~0.4–0.6 kcal/mol on LOLO-CV (Baseline 1).

References
----------
Brody S., Alon U., Yahav E. "How Attentive are Graph Attention Networks?"
ICLR 2022.  arXiv:2105.14491
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv, global_mean_pool

from .config import OptionAConfig


class _GATv2Block(nn.Module):
    """Single GATv2 layer with residual, BatchNorm, Dropout."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        heads:        int,
        edge_dim:     int,
        dropout:      float,
        concat:       bool = True,
    ) -> None:
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=concat,
        )
        out_total = out_channels * heads if concat else out_channels
        self.norm = nn.BatchNorm1d(out_total)
        self.drop = nn.Dropout(dropout)

        # skip connection: project input if shapes differ
        self.skip: nn.Module
        if in_channels != out_total:
            self.skip = nn.Linear(in_channels, out_total, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = F.elu(out)
        out = self.drop(out)
        return out + self.skip(x)


class OptionA(nn.Module):
    """
    Condition-Aware 4-layer GATv2 with global mean pooling.

    Parameters
    ----------
    cfg : OptionAConfig
        Hyperparameter bundle.

    Forward
    -------
    data : torch_geometric.data.HeteroData
        Must have:
          data['ligand'].x           [total_atoms, 35]
          data['ligand'].edge_index  [2, total_bonds]
          data['ligand'].edge_attr   [total_bonds, 13]
          data['ligand'].batch       [total_atoms]
          data.ph_enc                float32  [B]
          data.temp_enc              float32  [B]
          data.box_idx               int64    [B]
          data.receptor_flag         float32  [B]

    Returns
    -------
    delta_g : Tensor  [B, 1]
    """

    def __init__(self, cfg: OptionAConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OptionAConfig()
        self.cfg = cfg

        hidden = cfg.gat_head_dim * cfg.gat_heads   # 32 × 4 = 128

        # ── GNN input projection ─────────────────────────────────────────────
        self.input_proj = nn.Linear(cfg.ligand_node_dim, hidden)
        self.edge_proj  = nn.Linear(cfg.ligand_edge_dim, cfg.ligand_edge_dim)  # kept for interface

        # ── 4-layer GATv2 ────────────────────────────────────────────────────
        self.gat_layers = nn.ModuleList()
        for i in range(cfg.n_layers):
            out_concat = (i < cfg.n_layers - 1)  # last layer: average heads
            self.gat_layers.append(
                _GATv2Block(
                    in_channels  = hidden,
                    out_channels = cfg.gat_head_dim,
                    heads        = cfg.gat_heads,
                    edge_dim     = cfg.ligand_edge_dim,
                    dropout      = cfg.dropout,
                    concat       = out_concat,
                )
            )

        # ── Post-GNN projection (last block outputs gat_head_dim, not hidden) ──
        # last block: concat=False → output is gat_head_dim=32, not hidden=128
        self.post_gnn = nn.Linear(cfg.gat_head_dim, hidden)

        # ── Condition encoder ────────────────────────────────────────────────
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # ── Readout MLP ──────────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 32, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data) -> Tensor:
        x        = data['ligand'].x
        ei       = data['ligand', 'bond', 'ligand'].edge_index
        ea       = data['ligand', 'bond', 'ligand'].edge_attr
        batch    = data['ligand'].batch

        # Input projection
        x = self.input_proj(x)

        # Message passing
        for layer in self.gat_layers:
            x = layer(x, ei, ea)

        # Project final GATv2 output back to hidden dim before pooling
        x = self.post_gnn(x)                    # [N_atoms, hidden]

        # Global readout
        h_lig = global_mean_pool(x, batch)      # [B, hidden]

        # Condition vector
        cond = self._encode_condition(data, h_lig.device)
        h_cond = self.cond_proj(cond)            # [B, 32]

        # Predict
        h = torch.cat([h_lig, h_cond], dim=-1)  # [B, 128+32]
        return self.mlp(h)                       # [B, 1]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_condition(self, data, device: torch.device) -> Tensor:
        """
        Build [B, COND_DIM] condition tensor from per-graph scalars.

        Expects data to have one of:
          a) data.ph_enc, data.temp_enc, data.box_idx, data.receptor_flag
          b) data.ph, data.temp_c (raw; will be encoded on-the-fly)
        """
        B = data['ligand'].batch.max().item() + 1

        if hasattr(data, 'ph_enc'):
            ph_enc   = _scalar_to_batch(data.ph_enc,       B, device)
            temp_enc = _scalar_to_batch(data.temp_enc,      B, device)
            box_idx  = _scalar_to_batch(data.box_idx,       B, device, long=True)
            rec_flag = _scalar_to_batch(data.receptor_flag, B, device)
        else:
            # Raw scalars — encode on-the-fly with ConditionEncoder
            ph_enc, temp_enc, box_idx, rec_flag = _encode_raw(data, B, device)

        # Embed box type: [B] int → [B, box_embed_dim]
        box_emb = self.box_embed(box_idx)        # [B, 16]

        # Continuous condition: [ph_enc, temp_enc, rec_flag] → [B, 3]
        cont = torch.stack([ph_enc, temp_enc, rec_flag], dim=-1)   # [B, 3]

        return torch.cat([cont, box_emb], dim=-1)  # [B, 19]


# ── Batch utilities ───────────────────────────────────────────────────────────

def _scalar_to_batch(
    val,
    B: int,
    device: torch.device,
    long: bool = False,
) -> Tensor:
    """Ensure a scalar/0-d/1-d tensor becomes shape [B]."""
    if not isinstance(val, Tensor):
        val = torch.tensor(val)
    val = val.to(device)
    if val.ndim == 0:
        val = val.expand(B)
    elif val.shape[0] == 1:
        val = val.expand(B)
    return val.long() if long else val.float()


def _encode_raw(data, B: int, device: torch.device):
    """Fall back to ConditionEncoder when pre-encoded fields are absent."""
    from Graph_model.data.features.conditions import ConditionEncoder
    enc = ConditionEncoder(strict=False)
    ph_list, te_list, bi_list, rf_list = [], [], [], []

    # Extract per-graph attributes (may be tensors of len B after batching)
    raw_ph  = getattr(data, 'ph',   7.0)
    raw_tc  = getattr(data, 'temp_c', 25.0)
    raw_box = getattr(data, 'docking_box', 'global_blind')
    raw_rec = getattr(data, 'receptor', 'collagen')

    for g in range(B):
        # Index into batched attributes
        if isinstance(raw_ph, torch.Tensor) and raw_ph.numel() > 1:
            ph = float(raw_ph[g])
        elif isinstance(raw_ph, torch.Tensor):
            ph = float(raw_ph)
        else:
            ph = float(raw_ph)

        if isinstance(raw_tc, torch.Tensor) and raw_tc.numel() > 1:
            temp = float(raw_tc[g])
        elif isinstance(raw_tc, torch.Tensor):
            temp = float(raw_tc)
        else:
            temp = float(raw_tc)

        if isinstance(raw_box, (list, tuple)):
            box = str(raw_box[g])
        elif isinstance(raw_box, torch.Tensor) and raw_box.numel() > 1:
            box = str(raw_box[g].item())
        else:
            box = str(raw_box)

        if isinstance(raw_rec, (list, tuple)):
            rec = str(raw_rec[g])
        elif isinstance(raw_rec, torch.Tensor) and raw_rec.numel() > 1:
            rec = str(raw_rec[g].item())
        else:
            rec = str(raw_rec)

        v    = enc.encode(ph, temp, box, rec)
        ph_list.append(v[0]); te_list.append(v[1])
        bi_list.append(int(v[2])); rf_list.append(v[3])
    mk = lambda lst, long=False: torch.tensor(
        lst, dtype=torch.long if long else torch.float32, device=device)
    return mk(ph_list), mk(te_list), mk(bi_list, long=True), mk(rf_list)
