"""
Graph_model.model.ggnn_sequential
====================================
Option H — Gated Graph Neural Network with Sequential Binding States

Architecture
------------
GGNN (Li et al. 2016) uses GRU-based message passing, enabling the network
to model sequential state transitions. For docking, this captures:

1. **Pre-binding state:** ligand conformation in solution
2. **Approach state:** initial contact with protein surface
3. **Bound state:** final docked configuration

Each state uses a separate GGNN propagation step, sharing weights but using
the previous state's hidden representations as initialisation.

  Input:  HeteroData with ligand nodes [N, 35], bond edges [2E, 13]
          + condition encoding [B, 19]

  ┌──────────────────────────────────────────────────────────┐
  │  State 1 (Free): K steps GGNN propagation               │
  │  State 2 (Approach): K steps, init from State 1 output  │
  │  State 3 (Bound): K steps, with condition modulation     │
  ├──────────────────────────────────────────────────────────┤
  │  Gated readout (attention over all 3 states)             │
  │  → condition concat → MLP → ΔG                          │
  └──────────────────────────────────────────────────────────┘

References
----------
• Li Y. et al., "Gated Graph Sequence Neural Networks," ICLR 2016.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool, GatedGraphConv

from .config import (
    ModelConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    COND_DIM,
    N_BOX_TYPES,
    BOX_EMBEDDING_DIM,
)


@dataclass
class GGNNSeqConfig(ModelConfig):
    """GGNN Sequential Binding configuration."""
    hidden_dim:     int   = 128
    n_steps:        int   = 4     # GRU propagation steps per state
    n_states:       int   = 3     # number of sequential binding states
    edge_proj_dim:  int   = 16    # projected edge feature dimension
    aggr:           str   = "add" # aggregation for GGNN


class _GRUPropagation(nn.Module):
    """GRU-based message-passing propagation (core of GGNN)."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        n_steps: int,
        aggr: str = "add",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.hidden_dim = hidden_dim

        # Edge-conditioned message linear
        self.msg_nn = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU cell for node update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self.aggr = aggr

    def forward(
        self,
        h: Tensor,              # [N, hidden]
        edge_index: Tensor,     # [2, E]
        edge_attr: Tensor,      # [E, edge_dim]
    ) -> Tensor:
        """
        Run n_steps of GRU propagation.
        Returns updated node features [N, hidden].
        """
        i, j = edge_index[0], edge_index[1]

        for _ in range(self.n_steps):
            # Compute messages
            msg = self.msg_nn(edge_attr) * h[j]  # [E, hidden]

            # Aggregate messages to target nodes
            agg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
            if self.aggr == "add":
                agg.scatter_add_(0, i.unsqueeze(-1).expand_as(msg), msg)
            elif self.aggr == "mean":
                count = torch.zeros(h.size(0), 1, device=h.device)
                agg.scatter_add_(0, i.unsqueeze(-1).expand_as(msg), msg)
                count.scatter_add_(0, i.unsqueeze(-1), torch.ones_like(i, dtype=torch.float).unsqueeze(-1))
                agg = agg / (count + 1e-8)

            # GRU update
            h = self.gru(agg, h)

        return h


class _StateGate(nn.Module):
    """Gated attention aggregation over sequential states."""

    def __init__(self, hidden_dim: int, n_states: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * n_states, n_states),
            nn.Softmax(dim=-1),
        )
        self.n_states = n_states

    def forward(self, states: list[Tensor]) -> Tensor:
        """
        states: list of [N, hidden] tensors (one per state)
        Returns [N, hidden] gated combination.
        """
        stacked = torch.stack(states, dim=-1)  # [N, hidden, n_states]
        concat  = torch.cat(states, dim=-1)     # [N, hidden * n_states]
        weights = self.gate(concat)             # [N, n_states]
        # Weighted sum
        out = (stacked * weights.unsqueeze(1)).sum(dim=-1)  # [N, hidden]
        return out


class GGNNSequential(nn.Module):
    """
    GGNN with sequential binding state modelling.

    Parameters
    ----------
    cfg : GGNNSeqConfig

    Returns
    -------
    delta_g : Tensor [B, 1]
    """

    def __init__(self, cfg: GGNNSeqConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or GGNNSeqConfig()
        self.cfg = cfg
        hidden = cfg.hidden_dim

        # Input projection
        self.input_proj = nn.Linear(cfg.ligand_node_dim, hidden)
        self.edge_proj  = nn.Linear(cfg.ligand_edge_dim, cfg.edge_proj_dim)

        # Sequential binding states
        self.propagators = nn.ModuleList([
            _GRUPropagation(
                hidden_dim=hidden,
                edge_dim=cfg.edge_proj_dim,
                n_steps=cfg.n_steps,
                aggr=cfg.aggr,
            )
            for _ in range(cfg.n_states)
        ])

        # Per-state layer norms
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(cfg.n_states)
        ])

        # State-dependent condition modulation for last state
        self.cond_gate = nn.Sequential(
            nn.Linear(cfg.cond_dim, hidden),
            nn.Sigmoid(),
        )

        # Gated state aggregation
        self.state_gate = _StateGate(hidden, cfg.n_states)

        # Condition encoder
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 32, cfg.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data) -> Tensor:
        x     = data['ligand'].x
        ei    = data['ligand', 'bond', 'ligand'].edge_index
        ea    = data['ligand', 'bond', 'ligand'].edge_attr
        batch = data['ligand'].batch

        # Input projection
        h = self.input_proj(x)  # [N, hidden]
        edge_feat = self.edge_proj(ea)  # [E, edge_proj_dim]

        # Condition vector
        cond_full = self._encode_condition_full(data, h.device)  # [B, cond_dim]

        # Sequential state propagation
        states = []
        for s, (prop, norm) in enumerate(zip(self.propagators, self.norms)):
            h = prop(h, ei, edge_feat)
            h = norm(h)

            # Apply condition modulation to the last state (bound)
            if s == len(self.propagators) - 1:
                gate = self.cond_gate(cond_full)  # [B, hidden]
                gate_n = gate[batch]  # [N, hidden]
                h = h * gate_n

            states.append(h)

        # Gated aggregation over states
        h_combined = self.state_gate(states)  # [N, hidden]

        # Global pool
        h_graph = global_mean_pool(h_combined, batch)  # [B, hidden]

        # Condition for MLP
        cond = self._encode_condition(data, h_graph.device)
        h_cond = self.cond_proj(cond)  # [B, 32]

        out = torch.cat([h_graph, h_cond], dim=-1)
        return self.mlp(out)  # [B, 1]

    def _encode_condition_full(self, data, device: torch.device) -> Tensor:
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

    def _encode_condition(self, data, device: torch.device) -> Tensor:
        return self._encode_condition_full(data, device)
