"""
Graph_model.model.egnn
========================
Option G — E(n) Equivariant Graph Neural Network (EGNN)

Architecture
------------
EGNN (Satorras et al. 2021) updates both node features AND coordinates
equivariantly under rotations and translations, without requiring
higher-order representations (unlike SE(3)-Transformers or SchNet).

Key equations per layer:
  m_ij   = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)        # message
  x_i'   = x_i + C Σ_j (x_i - x_j) · φ_x(m_ij)         # coordinate update
  m_i    = Σ_j m_ij                                        # aggregate
  h_i'   = φ_h(h_i, m_i)                                  # node update

For molecules without 3D coordinates, we initialise positions from a
random 3D embedding and let the network learn pseudo-coordinates.

Adapted with condition integration (FiLM-style) for pH/temperature.

References
----------
• Satorras V.G., Hoogeboom E., Welling M.
  "E(n) Equivariant Graph Neural Networks," ICML 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter

from .config import (
    ModelConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    COND_DIM,
    N_BOX_TYPES,
    BOX_EMBEDDING_DIM,
)


@dataclass
class EGNNConfig(ModelConfig):
    """EGNN configuration."""
    hidden_dim:      int   = 128
    n_layers:        int   = 4
    coord_dim:       int   = 3      # coordinate space dimension
    update_coords:   bool  = True   # whether to update coordinates
    coord_scale:     float = 1.0    # scale factor for coordinate updates
    use_edge_attr:   bool  = True   # use bond features in messages
    normalise_coord: bool  = True   # normalise coordinate differences
    use_film:        bool  = True   # FiLM conditioning at each layer


class _EGNNLayer(nn.Module):
    """Single E(n) Equivariant GNN layer."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        coord_dim: int = 3,
        update_coords: bool = True,
        coord_scale: float = 1.0,
        normalise: bool = True,
        use_edge_attr: bool = True,
        use_film: bool = True,
        cond_dim: int = COND_DIM,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.coord_dim     = coord_dim
        self.update_coords = update_coords
        self.coord_scale   = coord_scale
        self.normalise     = normalise
        self.use_edge_attr = use_edge_attr
        self.use_film      = use_film

        # Message MLP: φ_e(h_i, h_j, ||x_i-x_j||², e_ij)
        msg_in = 2 * hidden_dim + 1  # h_i, h_j, dist²
        if use_edge_attr:
            msg_in += edge_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Coordinate update MLP: φ_x(m_ij) → scalar weight
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            # Initialize near zero for stable training
            nn.init.zeros_(self.coord_mlp[-1].weight)

        # Node update MLP: φ_h(h_i, m_i)
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM conditioning
        if use_film:
            self.film_proj = nn.Linear(cond_dim, 2 * hidden_dim)  # γ, β

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,              # [N, hidden]
        pos: Tensor,            # [N, coord_dim]
        edge_index: Tensor,     # [2, E]
        edge_attr: Tensor,      # [E, edge_dim]
        cond: Tensor = None,    # [B, cond_dim] (per graph)
        batch: Tensor = None,   # [N]
    ) -> tuple[Tensor, Tensor]:
        """
        Returns (h_new, pos_new).
        """
        i, j = edge_index[0], edge_index[1]

        # Relative positions
        rel_pos = pos[i] - pos[j]  # [E, coord_dim]
        dist_sq = (rel_pos ** 2).sum(dim=-1, keepdim=True)  # [E, 1]

        # Build message input
        msg_input = [h[i], h[j], dist_sq]
        if self.use_edge_attr and edge_attr is not None:
            msg_input.append(edge_attr)
        msg_input = torch.cat(msg_input, dim=-1)

        # Compute messages
        m_ij = self.msg_mlp(msg_input)  # [E, hidden]

        # Update coordinates equivariantly
        if self.update_coords:
            coord_weights = self.coord_mlp(m_ij)  # [E, 1]
            if self.normalise:
                norm = torch.sqrt(dist_sq + 1e-8)
                rel_pos_norm = rel_pos / norm
            else:
                rel_pos_norm = rel_pos

            # Weighted sum of normalised direction vectors
            coord_update = rel_pos_norm * coord_weights  # [E, coord_dim]
            agg_coord = torch.zeros_like(pos)
            agg_coord.scatter_add_(0, i.unsqueeze(-1).expand(-1, self.coord_dim), coord_update)
            pos_new = pos + self.coord_scale * agg_coord
        else:
            pos_new = pos

        # Aggregate messages
        agg_msg = torch.zeros(h.size(0), self.hidden_dim, device=h.device)
        agg_msg.scatter_add_(0, i.unsqueeze(-1).expand(-1, self.hidden_dim), m_ij)

        # Node update
        h_new = self.node_mlp(torch.cat([h, agg_msg], dim=-1))

        # FiLM conditioning
        if self.use_film and cond is not None and batch is not None:
            film_params = self.film_proj(cond)  # [B, 2*hidden]
            gamma, beta = film_params.chunk(2, dim=-1)  # [B, hidden] each
            # Expand to node level
            gamma_n = gamma[batch]  # [N, hidden]
            beta_n  = beta[batch]   # [N, hidden]
            h_new   = gamma_n * h_new + beta_n

        # Residual + norm
        h_new = self.norm(h + h_new)

        return h_new, pos_new


class EGNN(nn.Module):
    """
    E(n) Equivariant GNN for condition-aware docking energy prediction.

    Parameters
    ----------
    cfg : EGNNConfig

    Forward: data : HeteroData → delta_g [B, 1]
    """

    def __init__(self, cfg: EGNNConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or EGNNConfig()
        self.cfg = cfg
        hidden = cfg.hidden_dim

        # Node input projection
        self.input_proj = nn.Linear(cfg.ligand_node_dim, hidden)

        # Initial coordinate embedding (learnable if no 3D coords)
        self.coord_init = nn.Linear(cfg.ligand_node_dim, cfg.coord_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            _EGNNLayer(
                hidden_dim=hidden,
                edge_dim=cfg.ligand_edge_dim,
                coord_dim=cfg.coord_dim,
                update_coords=cfg.update_coords,
                coord_scale=cfg.coord_scale,
                normalise=cfg.normalise_coord,
                use_edge_attr=cfg.use_edge_attr,
                use_film=cfg.use_film,
                cond_dim=cfg.cond_dim,
            )
            for _ in range(cfg.n_layers)
        ])

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

        # Initial node features
        h = self.input_proj(x)  # [N, hidden]

        # Initial coordinates (from features → 3D)
        pos = self.coord_init(x)  # [N, 3]

        # Condition vector (full, for FiLM)
        cond_full = self._encode_condition_full(data, h.device)  # [B, cond_dim]

        # Message passing
        for layer in self.layers:
            h, pos = layer(h, pos, ei, ea, cond=cond_full, batch=batch)

        # Pool
        h_graph = global_mean_pool(h, batch)  # [B, hidden]

        # Condition for MLP
        cond = self._encode_condition(data, h_graph.device)
        h_cond = self.cond_proj(cond)  # [B, 32]

        out = torch.cat([h_graph, h_cond], dim=-1)
        return self.mlp(out)  # [B, 1]

    def _encode_condition_full(self, data, device: torch.device) -> Tensor:
        """Full condition vector [B, cond_dim=19] for FiLM."""
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
