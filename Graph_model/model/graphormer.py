"""
Graph_model.model.graphormer
================================
Option I — Graphormer (Graph Transformer)

Architecture
------------
Graphormer (Ying et al. 2021) applies full self-attention to molecular
graphs with three structural encodings:

1. **Centrality encoding:** Each node's degree is encoded via a learnable
   embedding and added to input features.
2. **Spatial encoding:** Shortest path distance between all node pairs
   biases attention logits.
3. **Edge encoding:** Edge features along shortest paths are aggregated
   and used as attention bias.

Adapted for docking with condition-aware [CLS] token:
  - A virtual [CLS] node is added, connected to all atoms
  - The CLS token is initialised from the condition vector
  - After L Graphormer layers, the CLS representation → MLP → ΔG

  ┌──────────────────────────────────────────────────────────────┐
  │  Centrality encoding:  x_i += Embed(degree_in) + Embed(out) │
  │  [CLS] token ← condition vector projection                  │
  ├──────────────────────────────────────────────────────────────┤
  │  L × GraphormerLayer:                                        │
  │    • Multi-head self-attention with spatial/edge bias        │
  │    • Pre-norm (LayerNorm) + FFN + Dropout                   │
  ├──────────────────────────────────────────────────────────────┤
  │  h_cls → MLP → ΔG                                           │
  └──────────────────────────────────────────────────────────────┘

References
----------
• Ying C. et al., "Do Transformers Really Perform Bad for Graph
  Representation?" NeurIPS 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import to_dense_adj, to_dense_batch, degree

from .config import (
    ModelConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    COND_DIM,
    N_BOX_TYPES,
    BOX_EMBEDDING_DIM,
)


@dataclass
class GraphormerConfig(ModelConfig):
    """Graphormer configuration."""
    hidden_dim:       int   = 128
    n_layers:         int   = 4
    n_heads:          int   = 8
    ffn_dim:          int   = 512
    max_degree:       int   = 10    # max node degree for centrality encoding
    max_sp_dist:      int   = 10    # max shortest path distance (clipped)
    n_edge_types:     int   = 4     # distinct edge types along shortest paths
    attention_dropout: float = 0.1
    pre_norm:          bool  = True


# ── Structural Encodings ─────────────────────────────────────────────────────

class _CentralityEncoding(nn.Module):
    """Learnable degree-based centrality encoding."""

    def __init__(self, max_degree: int, hidden_dim: int):
        super().__init__()
        self.in_embed  = nn.Embedding(max_degree + 1, hidden_dim)
        self.out_embed = nn.Embedding(max_degree + 1, hidden_dim)
        self.max_degree = max_degree

    def forward(self, edge_index: Tensor, n_nodes: int) -> Tensor:
        """Returns centrality encoding [N, hidden_dim]."""
        src, tgt = edge_index[0], edge_index[1]
        in_deg  = degree(tgt, num_nodes=n_nodes).long().clamp(max=self.max_degree)
        out_deg = degree(src, num_nodes=n_nodes).long().clamp(max=self.max_degree)
        return self.in_embed(in_deg) + self.out_embed(out_deg)


class _SpatialEncoding(nn.Module):
    """Shortest path distance → attention bias."""

    def __init__(self, max_sp_dist: int, n_heads: int):
        super().__init__()
        # Learnable bias per distance per head
        self.bias = nn.Embedding(max_sp_dist + 2, n_heads)  # +1 for self, +1 for disconnected
        self.max_dist = max_sp_dist

    def forward(self, sp_dist: Tensor) -> Tensor:
        """
        sp_dist : [B, N+1, N+1] long tensor of shortest path distances
                  (-1 for disconnected, clipped to max_sp_dist)
        Returns : [B, n_heads, N+1, N+1] attention bias
        """
        dist_clipped = sp_dist.clamp(0, self.max_dist + 1)
        bias = self.bias(dist_clipped)  # [B, N+1, N+1, n_heads]
        return bias.permute(0, 3, 1, 2)  # [B, n_heads, N+1, N+1]


# ── Graphormer Multi-Head Attention ──────────────────────────────────────────

class _GraphormerAttention(nn.Module):
    """Multi-head attention with spatial and edge bias."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        assert self.head_dim * n_heads == hidden_dim, \
            f"hidden_dim={hidden_dim} not divisible by n_heads={n_heads}"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: Tensor,              # [B, N+1, hidden]
        attn_bias: Tensor,      # [B, n_heads, N+1, N+1]
        mask: Tensor = None,    # [B, N+1] bool — True for valid nodes
    ) -> tuple[Tensor, Tensor]:
        """Returns (output, attn_weights)."""
        B, L, H = x.shape

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        attn = attn + attn_bias  # Add structural bias

        # Apply mask
        if mask is not None:
            # mask: [B, L] → expand
            mask_2d = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(~mask_2d, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L, self.n_heads * self.head_dim)
        out = self.o_proj(out)

        return out, attn_weights


# ── Graphormer Layer ─────────────────────────────────────────────────────────

class _GraphormerLayer(nn.Module):
    """Pre-norm Graphormer Transformer layer."""

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.15,
        attention_dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn  = _GraphormerAttention(hidden_dim, n_heads, attention_dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # Self-attention
        if self.pre_norm:
            h = self.norm1(x)
            h, attn_w = self.attn(h, attn_bias, mask)
            x = x + self.drop1(h)
            h = self.norm2(x)
            x = x + self.ffn(h)
        else:
            h, attn_w = self.attn(x, attn_bias, mask)
            x = self.norm1(x + self.drop1(h))
            x = self.norm2(x + self.ffn(x))

        return x, attn_w


# ── Shortest Path Computation ────────────────────────────────────────────────

def _compute_shortest_paths(edge_index: Tensor, n_nodes: int, max_dist: int) -> Tensor:
    """
    BFS-based shortest path distances for a single graph.

    Returns
    -------
    sp_dist : [n_nodes, n_nodes] long tensor (disconnected = max_dist + 1)
    """
    device = edge_index.device
    sp = torch.full((n_nodes, n_nodes), max_dist + 1, dtype=torch.long, device=device)
    sp.fill_diagonal_(0)

    # Build adjacency
    src, tgt = edge_index[0], edge_index[1]
    adj = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=device)
    adj[src, tgt] = True

    # BFS via matrix power (efficient for small graphs)
    current = adj.float()
    for d in range(1, max_dist + 1):
        mask = (sp == max_dist + 1) & (current > 0)
        sp[mask] = d
        current = current @ adj.float()
        current = (current > 0).float()

    return sp


def _batch_shortest_paths(
    edge_index: Tensor,
    batch: Tensor,
    max_dist: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """
    Compute shortest paths for a batch of graphs.

    Returns (sp_batch, mask, max_n) where:
        sp_batch : [B, max_n+1, max_n+1] — +1 for CLS token
        mask     : [B, max_n+1] bool
        max_n    : max node count
    """
    B = batch.max().item() + 1
    sizes = torch.zeros(B, dtype=torch.long, device=device)
    for i in range(B):
        sizes[i] = (batch == i).sum()
    max_n = sizes.max().item()

    # +1 for CLS node
    sp_batch = torch.full((B, max_n + 1, max_n + 1), max_dist + 1,
                          dtype=torch.long, device=device)
    mask = torch.zeros(B, max_n + 1, dtype=torch.bool, device=device)

    # CLS is at position 0; real nodes at positions 1..n
    offset = 0
    for g in range(B):
        n = sizes[g].item()
        mask[g, :n + 1] = True  # CLS + n atoms
        sp_batch[g, 0, 0] = 0   # CLS self-distance

        # CLS to all atoms: distance 1
        sp_batch[g, 0, 1:n + 1] = 1
        sp_batch[g, 1:n + 1, 0] = 1

        # Extract subgraph edges
        node_mask = (batch == g)
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        sub_ei = edge_index[:, edge_mask] - offset

        if sub_ei.numel() > 0 and n > 0:
            sp_g = _compute_shortest_paths(sub_ei, n, max_dist)
            sp_batch[g, 1:n + 1, 1:n + 1] = sp_g

        offset += n

    return sp_batch, mask, max_n


# ── Main Model ───────────────────────────────────────────────────────────────

class Graphormer(nn.Module):
    """
    Graphormer with condition-aware [CLS] token for docking energy prediction.

    Parameters
    ----------
    cfg : GraphormerConfig

    Returns
    -------
    delta_g : Tensor [B, 1]
    """

    def __init__(self, cfg: GraphormerConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GraphormerConfig()
        self.cfg = cfg
        hidden = cfg.hidden_dim

        # Node input projection
        self.input_proj = nn.Linear(cfg.ligand_node_dim, hidden)

        # Centrality encoding
        self.centrality = _CentralityEncoding(cfg.max_degree, hidden)

        # Spatial encoding (attention bias)
        self.spatial = _SpatialEncoding(cfg.max_sp_dist, cfg.n_heads)

        # CLS token from condition
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cls_proj  = nn.Linear(cfg.cond_dim, hidden)

        # Transformer layers
        self.layers = nn.ModuleList([
            _GraphormerLayer(
                hidden_dim=hidden,
                n_heads=cfg.n_heads,
                ffn_dim=cfg.ffn_dim,
                dropout=cfg.dropout,
                attention_dropout=cfg.attention_dropout,
                pre_norm=cfg.pre_norm,
            )
            for _ in range(cfg.n_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden)

        # Readout MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Store attention weights for interpretability
        self.last_attn_weights: list[Tensor] = []

    def forward(self, data) -> Tensor:
        x     = data['ligand'].x
        ei    = data['ligand', 'bond', 'ligand'].edge_index
        batch = data['ligand'].batch
        device = x.device
        B = batch.max().item() + 1

        # Node features + centrality encoding
        h = self.input_proj(x) + self.centrality(ei, x.size(0))

        # Compute shortest path distances + build dense batch
        sp_batch, mask, max_n = _batch_shortest_paths(
            ei, batch, self.cfg.max_sp_dist, device
        )

        # Spatial attention bias
        attn_bias = self.spatial(sp_batch)  # [B, n_heads, max_n+1, max_n+1]

        # Build dense batch (pad graphs to max_n)
        # Convert sparse to dense
        dense_h = torch.zeros(B, max_n, self.cfg.hidden_dim, device=device)
        offset = 0
        sizes = torch.zeros(B, dtype=torch.long, device=device)
        for g in range(B):
            n = (batch == g).sum().item()
            sizes[g] = n
            dense_h[g, :n] = h[offset:offset + n]
            offset += n

        # CLS token from condition
        cond = self._encode_condition(data, device)   # [B, cond_dim]
        cls_token = self.cls_proj(cond).unsqueeze(1)  # [B, 1, hidden]

        # Prepend CLS to dense features
        x_dense = torch.cat([cls_token, dense_h], dim=1)  # [B, max_n+1, hidden]

        # Apply Graphormer layers
        self.last_attn_weights = []
        for layer in self.layers:
            x_dense, attn_w = layer(x_dense, attn_bias, mask)
            self.last_attn_weights.append(attn_w.detach())

        # Final norm
        x_dense = self.final_norm(x_dense)

        # CLS readout
        cls_out = x_dense[:, 0, :]  # [B, hidden]

        return self.mlp(cls_out)  # [B, 1]

    def _encode_condition(self, data, device: torch.device) -> Tensor:
        from .option_a import _scalar_to_batch, _encode_raw
        B = data['ligand'].batch.max().item() + 1
        if hasattr(data, 'ph_enc'):
            ph_enc   = _scalar_to_batch(data.ph_enc, B, device)
            temp_enc = _scalar_to_batch(data.temp_enc, B, device)
            box_idx  = _scalar_to_batch(data.box_idx, B, device, long=True)
            rec_flag = _scalar_to_batch(data.receptor_flag, B, device)
        else:
            ph_enc, temp_enc, box_idx, rec_flag = _encode_raw(data, B, device)
        box_emb = self.box_embed(box_idx)
        cont = torch.stack([ph_enc, temp_enc, rec_flag], dim=-1)
        return torch.cat([cont, box_emb], dim=-1)
