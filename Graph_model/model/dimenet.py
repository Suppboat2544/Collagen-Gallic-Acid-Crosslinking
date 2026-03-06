"""
Graph_model.model.dimenet
============================
Option F — DimeNet++ (Directional Message Passing Neural Network)

Architecture
------------
DimeNet++ uses directional information (bond angles) to build richer
message-passing updates. It operates on:
  • Atom embeddings (nodes)
  • Edge messages that incorporate distances AND angles between triplets

For our docking task, we adapt DimeNet++ to work with the existing
Level-1 ligand graph features + condition encoding:

  Input:  HeteroData with ligand nodes [N, 35], bond edges [2E, 13]
          + condition encoding [B, 19]

  ┌──────────────────────────────────────────────────────────┐
  │  RBF distance expansion (if 3D coords) / edge projection │
  │  SBF angle expansion / edge triplet features              │
  ├──────────────────────────────────────────────────────────┤
  │  L × InteractionBlock (Bessel basis + bilinear updates)  │
  │  Per-block output blocks with skip connections           │
  ├──────────────────────────────────────────────────────────┤
  │  Global sum pool → condition concat → MLP → ΔG          │
  └──────────────────────────────────────────────────────────┘

References
----------
• Gasteiger J. et al., "Directional Message Passing for Molecular Graphs,"
  ICLR 2020.
• Gasteiger J. et al., "Fast and Uncertainty-Aware Directional Message
  Passing for Non-Equilibrium Molecules," NeurIPS 2020 ML4Molecules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool, global_mean_pool

from .config import (
    ModelConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    COND_DIM,
    N_BOX_TYPES,
    BOX_EMBEDDING_DIM,
)


@dataclass
class DimeNetConfig(ModelConfig):
    """DimeNet++ configuration."""
    hidden_dim:       int   = 128
    n_layers:         int   = 4     # number of interaction blocks
    n_rbf:            int   = 16    # radial basis functions
    n_sbf:            int   = 7     # spherical basis functions
    envelope_exp:     int   = 5     # envelope polynomial exponent
    n_bilinear:       int   = 8     # bilinear triplet interaction rank
    output_blocks:    int   = 3     # layers in output block MLP
    activation:       str   = "swish"


# ── Building blocks ──────────────────────────────────────────────────────────

class _SinusoidalRBF(nn.Module):
    """Sinusoidal radial basis functions for distance expansion."""

    def __init__(self, n_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.n_rbf  = n_rbf
        self.cutoff = cutoff
        # Frequencies: k*π/cutoff for k=1..n_rbf
        freqs = torch.arange(1, n_rbf + 1, dtype=torch.float) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, dist: Tensor) -> Tensor:
        """dist: [E] → [E, n_rbf]"""
        dist = dist.unsqueeze(-1)  # [E, 1]
        return torch.sin(self.freqs * dist) / (dist + 1e-8)  # [E, n_rbf]


class _Envelope(nn.Module):
    """Polynomial envelope function for smooth cutoff."""

    def __init__(self, exponent: int = 5):
        super().__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: Tensor) -> Tensor:
        """x: normalised distance in [0, 1]"""
        p = self.p
        x_pow_p_minus_1 = x.pow(p - 1)
        x_pow_p = x_pow_p_minus_1 * x
        x_pow_p_plus_1 = x_pow_p * x
        env = 1.0 + self.a * x_pow_p + self.b * x_pow_p_plus_1 + self.c * x_pow_p_plus_1 * x
        return env * (x < 1.0).float()


class _ResidualLayer(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, dim: int, act: nn.Module):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act  = act

    def forward(self, x: Tensor) -> Tensor:
        return x + self.lin2(self.act(self.lin1(self.act(x))))


class _EmbeddingBlock(nn.Module):
    """Initial node + edge embedding for DimeNet++."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden: int,
        n_rbf: int,
        act: nn.Module,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.rbf_proj  = nn.Linear(n_rbf, hidden)
        self.act = act
        self.lin = nn.Linear(hidden * 3, hidden)

    def forward(
        self,
        x: Tensor,              # [N, node_dim]
        edge_attr: Tensor,      # [E, edge_dim]
        rbf: Tensor,            # [E, n_rbf]
        edge_index: Tensor,     # [2, E]
    ) -> Tensor:
        """Returns edge messages [E, hidden]."""
        i, j = edge_index[0], edge_index[1]
        xi = self.node_proj(x[i])           # [E, hidden]
        xj = self.node_proj(x[j])           # [E, hidden]
        e   = self.edge_proj(edge_attr)      # [E, hidden]
        r   = self.rbf_proj(rbf)             # [E, hidden]

        return self.act(self.lin(torch.cat([xi * xj, e, r], dim=-1)))


class _InteractionBlock(nn.Module):
    """DimeNet++ interaction block with bilinear triplet aggregation."""

    def __init__(
        self,
        hidden: int,
        n_rbf: int,
        n_sbf: int,
        n_bilinear: int,
        act: nn.Module,
    ):
        super().__init__()
        self.hidden = hidden

        # Down-project for bilinear
        self.lin_rbf = nn.Linear(n_rbf, hidden, bias=False)
        self.lin_sbf = nn.Linear(n_sbf, n_bilinear, bias=False)

        # Bilinear interaction
        self.lin_kj = nn.Linear(hidden, hidden)
        self.lin_ji = nn.Linear(hidden, hidden)

        # Bilinear weight: [hidden, n_bilinear, hidden]
        self.W = nn.Parameter(torch.randn(hidden, n_bilinear, hidden) * 0.01)

        # Post layers
        self.res1 = _ResidualLayer(hidden, act)
        self.res2 = _ResidualLayer(hidden, act)
        self.lin_out = nn.Linear(hidden, hidden)
        self.act = act

    def forward(
        self,
        m: Tensor,              # [E, hidden] edge messages
        rbf: Tensor,            # [E, n_rbf]
        sbf: Tensor,            # [T, n_sbf]  (triplet features)
        edge_index: Tensor,     # [2, E]
        triplet_index: Tensor,  # [2, T]  (kj, ji) indices into edges
    ) -> Tensor:
        """Update edge messages via triplet interactions. Returns [E, hidden]."""
        # Direct edge update
        x_ji = self.act(self.lin_ji(m))  # [E, hidden]

        # Triplet update
        kj_idx, ji_idx = triplet_index[0], triplet_index[1]
        x_kj = self.act(self.lin_kj(m[kj_idx]))  # [T, hidden]

        # RBF gate on kj edges
        rbf_gate = self.lin_rbf(rbf[kj_idx])  # [T, hidden]
        x_kj = x_kj * rbf_gate

        # Spherical basis interaction (simplified bilinear)
        sbf_proj = self.lin_sbf(sbf)  # [T, n_bilinear]

        # Bilinear: x_kj @ W @ sbf → [T, hidden]
        # Efficient: (x_kj.unsqueeze(2) * W.unsqueeze(0)) summed with sbf
        x_kj_expanded = x_kj.unsqueeze(2)  # [T, hidden, 1]
        w_contracted = torch.einsum("hbk,tb->thk", self.W, sbf_proj)  # [T, hidden, hidden]
        triplet_msg = torch.einsum("thk,th->tk", w_contracted, x_kj)  # [T, hidden]

        # Scatter-add triplet messages to target edges
        aggr = torch.zeros_like(m)
        aggr.scatter_add_(0, ji_idx.unsqueeze(-1).expand(-1, self.hidden), triplet_msg)

        # Combine
        m_new = x_ji + aggr
        m_new = self.res1(m_new)
        m_new = self.res2(m_new)
        m_new = self.act(self.lin_out(m_new))

        return m + m_new  # Residual


class _OutputBlock(nn.Module):
    """Per-block output with aggregation to node level."""

    def __init__(self, hidden: int, n_layers: int, act: nn.Module):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(_ResidualLayer(hidden, act))
        self.residuals = nn.ModuleList(layers)
        self.lin = nn.Linear(hidden, hidden)
        self.act = act

    def forward(
        self,
        m: Tensor,              # [E, hidden]
        edge_index: Tensor,     # [2, E]
        n_nodes: int,
    ) -> Tensor:
        """Aggregate edge messages to nodes. Returns [N, hidden]."""
        for res in self.residuals:
            m = res(m)
        m = self.act(self.lin(m))

        # Scatter-add to target nodes
        target = edge_index[1]
        out = torch.zeros(n_nodes, m.size(-1), device=m.device)
        out.scatter_add_(0, target.unsqueeze(-1).expand_as(m), m)
        return out


# ── Triplet computation ──────────────────────────────────────────────────────

def _build_triplets(edge_index: Tensor) -> tuple[Tensor, Tensor]:
    """
    Build triplet indices (k→j→i) from edge_index.

    For each pair of edges (k,j) and (j,i), create a triplet.

    Returns
    -------
    triplet_index : [2, T]  where [0]=kj edge idx, [1]=ji edge idx
    angles        : None (computed separately if 3D coords available)
    """
    n_edges = edge_index.size(1)
    device  = edge_index.device

    # Build adjacency: for each node j, find all edges ending at j
    # and all edges starting at j
    src, tgt = edge_index[0], edge_index[1]

    # For each edge ji (index e_ji), find edges kj where k→j
    # i.e., edges whose target is the source of ji
    kj_list = []
    ji_list = []

    # Group edges by target node
    max_node = max(src.max(), tgt.max()) + 1
    # edges_to_node[j] = list of edge indices where tgt == j
    # We need edges where src[e_kj] == some k and tgt[e_kj] == j (source of ji)

    for e_ji in range(n_edges):
        j_node = src[e_ji].item()   # edge ji goes from j to i
        i_node = tgt[e_ji].item()
        # Find all edges where tgt == j_node (those are k→j edges)
        # and src != i_node (avoid self-triplet)
        mask = (tgt == j_node) & (src != i_node)
        kj_edges = mask.nonzero(as_tuple=True)[0]
        for e_kj in kj_edges:
            kj_list.append(e_kj.item())
            ji_list.append(e_ji)

    if kj_list:
        triplet_index = torch.tensor([kj_list, ji_list], dtype=torch.long, device=device)
    else:
        triplet_index = torch.zeros((2, 0), dtype=torch.long, device=device)

    return triplet_index


def _compute_sbf(
    edge_index: Tensor,
    edge_attr: Tensor,
    triplet_index: Tensor,
    n_sbf: int,
) -> Tensor:
    """
    Compute spherical basis features for triplets.
    Without 3D coordinates, we use edge feature similarity as a proxy.

    Returns
    -------
    sbf : [T, n_sbf]
    """
    n_triplets = triplet_index.size(1)
    device = edge_attr.device

    if n_triplets == 0:
        return torch.zeros(0, n_sbf, device=device)

    kj_idx = triplet_index[0]
    ji_idx = triplet_index[1]

    # Use edge feature dot-product as angle proxy
    e_kj = edge_attr[kj_idx]  # [T, edge_dim]
    e_ji = edge_attr[ji_idx]  # [T, edge_dim]

    # Compute pseudo-angle from edge feature similarity
    cos_sim = F.cosine_similarity(e_kj, e_ji, dim=-1)  # [T]

    # Expand to n_sbf via sinusoidal basis
    freqs = torch.arange(1, n_sbf + 1, device=device, dtype=torch.float).unsqueeze(0)  # [1, n_sbf]
    angles = torch.acos(cos_sim.clamp(-0.999, 0.999)).unsqueeze(-1)  # [T, 1]
    sbf = torch.sin(freqs * angles)  # [T, n_sbf]

    return sbf


# ── Activations ──────────────────────────────────────────────────────────────

class _Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


def _get_activation(name: str) -> nn.Module:
    if name == "swish":
        return _Swish()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        return _Swish()


# ── Main model ───────────────────────────────────────────────────────────────

class DimeNet(nn.Module):
    """
    DimeNet++ adapted for condition-aware docking energy prediction.

    Parameters
    ----------
    cfg : DimeNetConfig

    Forward
    -------
    data : HeteroData with standard ligand + condition fields.

    Returns
    -------
    delta_g : Tensor [B, 1]
    """

    def __init__(self, cfg: DimeNetConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or DimeNetConfig()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        act = _get_activation(cfg.activation)

        # RBF expansion
        self.rbf = _SinusoidalRBF(cfg.n_rbf, cutoff=5.0)

        # Embedding
        self.emb_block = _EmbeddingBlock(
            node_dim=cfg.ligand_node_dim,
            edge_dim=cfg.ligand_edge_dim,
            hidden=hidden,
            n_rbf=cfg.n_rbf,
            act=act,
        )

        # Interaction blocks
        self.int_blocks = nn.ModuleList([
            _InteractionBlock(
                hidden=hidden,
                n_rbf=cfg.n_rbf,
                n_sbf=cfg.n_sbf,
                n_bilinear=cfg.n_bilinear,
                act=act,
            )
            for _ in range(cfg.n_layers)
        ])

        # Output blocks (one per interaction block + one for embedding)
        self.out_blocks = nn.ModuleList([
            _OutputBlock(hidden, cfg.output_blocks, act)
            for _ in range(cfg.n_layers + 1)
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
        x      = data['ligand'].x
        ei     = data['ligand', 'bond', 'ligand'].edge_index
        ea     = data['ligand', 'bond', 'ligand'].edge_attr
        batch  = data['ligand'].batch
        n_nodes = x.size(0)

        # Compute pseudo-distances from edge features (norm of edge_attr)
        dist = ea.norm(dim=-1)  # [E]

        # RBF expansion
        rbf = self.rbf(dist)  # [E, n_rbf]

        # Build triplets
        triplet_index = _build_triplets(ei)

        # SBF features
        sbf = _compute_sbf(ei, ea, triplet_index, self.cfg.n_sbf)

        # Initial edge messages
        m = self.emb_block(x, ea, rbf, ei)

        # First output block
        h = self.out_blocks[0](m, ei, n_nodes)

        # Interaction + output blocks
        for i, (int_block, out_block) in enumerate(
            zip(self.int_blocks, self.out_blocks[1:])
        ):
            m = int_block(m, rbf, sbf, ei, triplet_index)
            h = h + out_block(m, ei, n_nodes)

        # Global pool
        h_graph = global_add_pool(h, batch)  # [B, hidden]

        # Condition
        cond = self._encode_condition(data, h_graph.device)
        h_cond = self.cond_proj(cond)  # [B, 32]

        # Predict
        out = torch.cat([h_graph, h_cond], dim=-1)
        return self.mlp(out)  # [B, 1]

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
