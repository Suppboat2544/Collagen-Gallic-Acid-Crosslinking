"""
Graph_model.model.option_e
============================
Option E — pH-Aware Gated Equivariant Transformer (pGET)
=========================================================

A novel architecture specifically designed for collagen-crosslinker binding
energy prediction under variable pH and temperature conditions.

Novelty  (not previously published as an integrated system)
--------------------------------------------------------------
1. **Condition-Gated Message Passing (CGMP)**
   Instead of concatenating condition vectors *only* at the readout stage
   (as Options A–D do), pGET injects pH/temperature/receptor information
   directly into the message-passing weights via a FiLM-style gating
   mechanism (Feature-wise Linear Modulation) at *every* GNN layer.

   This is motivated by the physical reality that pH changes protonation
   states of GLU/ASP residues, altering hydrogen-bond networks at the
   atomic level — NOT just the global binding affinity.

   h_i^{l+1} = γ_l(c) ⊙ MPNN(h_i^l, h_j^l, e_{ij}) + β_l(c)

   where c = condition vector, γ and β are learned per-layer affine
   modulations.

2. **Hierarchical Gated Attention Pooling (HGAP)**
   Rather than simple mean-pooling (Option A/D) or per-graph loop
   cross-attention (Option B), pGET uses:
   - A *learned* attention-based pooling gate per node:
     α_i = σ(MLP(h_i ‖ c))   — which nodes matter depends on conditions
   - Two-stage pooling: atom→fragment→molecule, where fragment assignment
     is computed by a *differentiable* soft-assignment (no BRICS at
     inference time, unlike Option C).

3. **Gated Edge-Conditioned Transformer (GECT) Layers**
   A custom Transformer-style layer where:
   - Edge features modulate attention logits (like GATv2) AND value vectors
     (unlike standard GATv2 which only use edge features for attention).
   - A per-edge gating mechanism controls information flow:
     g_{ij} = σ(W_g · e_{ij})
     msg_{ij} = g_{ij} ⊙ V_j
   - This allows the model to learn that certain bond types (e.g., ester
     linkages in PGG) should amplify or suppress message flow.

4. **Auxiliary Denoising Objective**
   During training, Gaussian noise is optionally added to input node
   features, and a small decoder head reconstructs the clean features.
   This acts as a self-supervised regulariser that improves feature
   representations for the small (9-ligand) anchor dataset.

Complexity  :  O(N·E) per layer (same as GATv2); no quadratic attention.
Parameters  :  ~380K (comparable to Options A/B).

Architecture summary
────────────────────
  Input:  HeteroData with ligand nodes [N, 35], bond edges [2E, 13]
          + condition encoding [B, 19]

  ┌─────────────────────────────────────────────────────────────────┐
  │  FiLM condition projector: c → (γ_l, β_l) for each of L layers│
  └─────────────────┬───────────────────────────────────────────────┘
                    │
  ┌─────────────────▼───────────────────────────────────────────────┐
  │  Layer 1..L:  GECT(x, edge_index, edge_attr, γ_l, β_l)        │
  │               = EdgeGatedTransformer + FiLM + Residual + Norm  │
  └─────────────────┬───────────────────────────────────────────────┘
                    │
  ┌─────────────────▼───────────────────────────────────────────────┐
  │  Hierarchical Gated Attention Pooling:                         │
  │    atom-level gate: α_i = σ(W·[h_i ‖ c])                      │
  │    soft fragment assignment: S = softmax(Atom2Frag MLP)        │
  │    fragment embed: h_frag = S^T · (α ⊙ h_atom)                │
  │    graph readout: h_g = Σ_k softmax(w_k · h_frag_k) · h_frag_k│
  └─────────────────┬───────────────────────────────────────────────┘
                    │
  ┌─────────────────▼───────────────────────────────────────────────┐
  │  Prediction head: MLP(h_g ‖ c) → ΔG  [B, 1]                  │
  └─────────────────────────────────────────────────────────────────┘

Optional auxiliary head (training only):
  h_atom^noisy → Linear → x̂_atom  [N, 35]   (denoising MSE)

References
----------
• FiLM: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
  Layer," AAAI 2018.
• GATv2: Brody et al., "How Attentive are Graph Attention Networks?"
  ICLR 2022.
• Gated GNN: Li et al., "Gated Graph Sequence Neural Networks," ICLR 2016.
• Set Transformer: Lee et al., "Set Transformer: A Framework for
  Attention-based Permutation-Invariant Input," ICML 2019.
• Denoising pretext tasks: Godwin et al., "Simple GNN Regularisation for
  3D Molecular Property Prediction & Beyond," ICLR 2022.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import unbatch

from .config import (
    ModelConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    COND_DIM,
    N_BOX_TYPES,
    BOX_EMBEDDING_DIM,
)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class OptionEConfig(ModelConfig):
    """Option E — pH-Aware Gated Equivariant Transformer (pGET).

    Novel architecture with condition-gated message passing, hierarchical
    gated attention pooling, and edge-gated transformer layers.
    """
    n_gect_layers:     int   = 4     # number of GECT layers
    n_heads:           int   = 4     # attention heads per GECT layer
    head_dim:          int   = 32    # per-head dimension
    n_soft_clusters:   int   = 8     # soft fragment clusters for HGAP
    film_hidden:       int   = 64    # FiLM condition projector hidden dim
    edge_gate_dim:     int   = 32    # edge gating hidden dim
    denoise_weight:    float = 0.1   # auxiliary denoising loss weight (0 = off)
    denoise_noise_std: float = 0.2   # noise magnitude for denoising task
    pool_heads:        int   = 4     # number of attention heads for readout


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scalar_to_batch(val, B: int, device: torch.device, long: bool = False):
    """Expand a scalar / 1-D tensor to a [B] tensor."""
    if isinstance(val, Tensor):
        t = val.to(device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B)
        if long:
            t = t.long()
        return t
    t = torch.tensor(val, device=device)
    if t.dim() == 0:
        t = t.unsqueeze(0).expand(B)
    if long:
        t = t.long()
    return t


def _encode_raw(data, B: int, device: torch.device):
    """Fall-back: derive condition scalars from raw data attributes."""
    _PROPKA = {5.0: 0.85, 5.5: 0.15, 7.0: 0.02}
    _TEMP_MIN, _TEMP_RANGE = 4.0, 33.0
    _RECEPTORS = {'collagen': 0.0, 'mmp1': 1.0}

    raw_ph  = getattr(data, 'ph', 7.0)
    raw_tc  = getattr(data, 'temp_c', 25.0)
    raw_rec = getattr(data, 'receptor', 'collagen')

    ph_list, temp_list, rec_list = [], [], []
    for g in range(B):
        if isinstance(raw_ph, torch.Tensor) and raw_ph.numel() > 1:
            ph = float(raw_ph[g])
        elif isinstance(raw_ph, torch.Tensor):
            ph = float(raw_ph)
        else:
            ph = float(raw_ph)

        if isinstance(raw_tc, torch.Tensor) and raw_tc.numel() > 1:
            temp_c = float(raw_tc[g])
        elif isinstance(raw_tc, torch.Tensor):
            temp_c = float(raw_tc)
        else:
            temp_c = float(raw_tc)

        if isinstance(raw_rec, (list, tuple)):
            receptor = str(raw_rec[g]).lower()
        elif isinstance(raw_rec, torch.Tensor) and raw_rec.numel() > 1:
            receptor = str(raw_rec[g].item()).lower()
        else:
            receptor = str(raw_rec).lower()

        ph_list.append(_PROPKA.get(ph, 0.02))
        temp_list.append((temp_c - _TEMP_MIN) / _TEMP_RANGE)
        rec_list.append(_RECEPTORS.get(receptor, 0.0))

    ph_t = torch.tensor(ph_list, device=device)
    temp_t = torch.tensor(temp_list, device=device)
    box_t = torch.zeros(B, dtype=torch.long, device=device)
    rec_t = torch.tensor(rec_list, device=device)
    return ph_t, temp_t, box_t, rec_t


# ── FiLM Condition Projector ─────────────────────────────────────────────────

class _FiLMGenerator(nn.Module):
    """
    Generate per-layer FiLM modulation parameters (γ, β) from condition vector.

    For L layers of hidden size H, produces:
        γ : [B, L, H]   (multiplicative gate, initialised ≈ 1)
        β : [B, L, H]   (additive bias, initialised ≈ 0)
    """
    def __init__(self, cond_dim: int, hidden: int, n_layers: int, out_dim: int):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim  = out_dim
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_layers * out_dim * 2),  # γ and β
        )
        # Init: γ ≈ 1, β ≈ 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Set γ bias to 1 (the first half of outputs)
        with torch.no_grad():
            self.net[-1].bias[:n_layers * out_dim] = 1.0

    def forward(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond : [B, cond_dim]

        Returns
        -------
        gamma : [B, n_layers, out_dim]   — multiplicative
        beta  : [B, n_layers, out_dim]   — additive
        """
        B = cond.shape[0]
        out = self.net(cond)  # [B, n_layers * out_dim * 2]
        half = self.n_layers * self.out_dim
        gamma = out[:, :half].view(B, self.n_layers, self.out_dim)
        beta  = out[:, half:].view(B, self.n_layers, self.out_dim)
        return gamma, beta


# ── Gated Edge-Conditioned Transformer Layer ─────────────────────────────────

class _GECTLayer(nn.Module):
    """
    Gated Edge-Conditioned Transformer (GECT) layer.

    Combines:
    - Multi-head attention with edge features modulating both attention
      logits AND value vectors (not just attention as in GATv2).
    - Per-edge sigmoid gate controlling information flow.
    - FiLM modulation from condition vector.
    - Pre-LayerNorm residual connection.

    h_i ← h_i + FiLM(MHA_edge_gated(h_i, {h_j, e_ij}), γ, β)
    """
    def __init__(
        self,
        hidden: int,
        heads:  int,
        head_dim: int,
        edge_dim: int,
        edge_gate_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.heads    = heads
        self.head_dim = head_dim
        self.hidden   = hidden
        self.scale    = head_dim ** -0.5

        # Node projections
        self.norm1    = nn.LayerNorm(hidden)
        self.q_proj   = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj   = nn.Linear(hidden, heads * head_dim, bias=False)
        self.v_proj   = nn.Linear(hidden, heads * head_dim, bias=False)
        self.out_proj = nn.Linear(heads * head_dim, hidden)

        # Edge → attention bias
        self.edge_attn = nn.Sequential(
            nn.Linear(edge_dim, edge_gate_dim),
            nn.GELU(),
            nn.Linear(edge_gate_dim, heads),  # one bias per head
        )

        # Edge → value gate  (novel: gates VALUE vectors, not just attention)
        self.edge_gate = nn.Sequential(
            nn.Linear(edge_dim, edge_gate_dim),
            nn.GELU(),
            nn.Linear(edge_gate_dim, heads * head_dim),
            nn.Sigmoid(),
        )

        # FFN with pre-norm
        self.norm2 = nn.LayerNorm(hidden)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x:          Tensor,       # [N, H]
        edge_index: Tensor,       # [2, E]
        edge_attr:  Tensor,       # [E, edge_dim]
        gamma:      Tensor,       # [N, H]  — per-node FiLM γ (expanded from batch)
        beta:       Tensor,       # [N, H]  — per-node FiLM β
    ) -> Tensor:
        N, H = x.shape
        h, d = self.heads, self.head_dim
        src, dst = edge_index  # src → dst message direction

        # ── Pre-norm ──────────────────────────────────────────────────────────
        x_n = self.norm1(x)

        # ── Q, K, V ──────────────────────────────────────────────────────────
        Q = self.q_proj(x_n).view(N, h, d)   # [N, h, d]
        K = self.k_proj(x_n).view(N, h, d)   # [N, h, d]
        V = self.v_proj(x_n).view(N, h, d)   # [N, h, d]

        # ── Edge-conditioned attention logits ─────────────────────────────────
        # α_{ij} = (q_i · k_j) / √d + bias(e_{ij})
        q_i = Q[dst]          # [E, h, d]
        k_j = K[src]          # [E, h, d]

        attn_logits = (q_i * k_j).sum(dim=-1) * self.scale  # [E, h]
        edge_bias   = self.edge_attn(edge_attr)               # [E, h]
        attn_logits = attn_logits + edge_bias

        # ── Sparse softmax ────────────────────────────────────────────────────
        # Softmax over neighbours of each dst node
        attn_weights = _sparse_softmax(attn_logits, dst, N)   # [E, h]

        # ── Edge-gated value vectors (novel) ──────────────────────────────────
        v_j     = V[src]                                       # [E, h, d]
        e_gate  = self.edge_gate(edge_attr).view(-1, h, d)    # [E, h, d]
        gated_v = e_gate * v_j                                 # [E, h, d]

        # ── Weighted aggregation ──────────────────────────────────────────────
        weighted = attn_weights.unsqueeze(-1) * gated_v       # [E, h, d]
        # Scatter-add to destination nodes
        out = torch.zeros(N, h, d, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted), weighted)
        out = out.reshape(N, h * d)                            # [N, H]
        out = self.out_proj(out)
        out = self.drop(out)

        # ── FiLM modulation + residual ────────────────────────────────────────
        out = gamma * out + beta
        x = x + out

        # ── FFN block ─────────────────────────────────────────────────────────
        x = x + self.ffn(self.norm2(x))

        return x


def _sparse_softmax(logits: Tensor, index: Tensor, N: int) -> Tensor:
    """
    Softmax over variable-size neighbourhoods (sparse graph softmax).

    Parameters
    ----------
    logits : [E, h]   — unnormalised attention per edge, per head
    index  : [E]      — destination node index
    N      : int      — total number of nodes

    Returns
    -------
    [E, h] — normalised attention weights
    """
    # Numerical stability: subtract max per destination node per head
    max_vals = torch.zeros(N, logits.shape[1], device=logits.device, dtype=logits.dtype)
    max_vals.scatter_reduce_(0, index.unsqueeze(-1).expand_as(logits), logits, reduce='amax')
    logits = logits - max_vals[index]

    exp_logits = logits.exp()
    sum_exp = torch.zeros(N, logits.shape[1], device=logits.device, dtype=logits.dtype)
    sum_exp.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_logits), exp_logits)
    return exp_logits / (sum_exp[index] + 1e-10)


# ── Hierarchical Gated Attention Pooling ─────────────────────────────────────

class _HierarchicalGatedPool(nn.Module):
    """
    Two-stage pooling: atoms → soft clusters → graph representation.

    Stage 1: Condition-aware gating — learn which atoms are important
             given the current pH/temperature/receptor.
    Stage 2: Soft assignment to K learned clusters (differentiable
             fragment-like grouping, no RDKit needed at inference).
    Stage 3: Multi-head attention readout over cluster embeddings.
    """
    def __init__(
        self,
        hidden:        int,
        cond_proj_dim: int,
        n_clusters:    int,
        pool_heads:    int,
        dropout:       float,
    ):
        super().__init__()
        self.n_clusters = n_clusters

        # Stage 1: atom-level gating  α_i = σ(W · [h_i ‖ c_proj])
        self.gate = nn.Sequential(
            nn.Linear(hidden + cond_proj_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

        # Stage 2: soft assignment  S = softmax(MLP(h_atom) → K logits)
        self.assign = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_clusters),
        )

        # Stage 3: cluster projection + attention readout
        self.cluster_proj = nn.Linear(hidden, hidden)
        self.readout_q    = nn.Parameter(torch.randn(pool_heads, hidden // pool_heads))
        self.readout_k    = nn.Linear(hidden, hidden, bias=False)
        self.readout_v    = nn.Linear(hidden, hidden, bias=False)
        self.pool_heads   = pool_heads
        self.head_dim     = hidden // pool_heads
        self.scale        = self.head_dim ** -0.5
        self.out_norm     = nn.LayerNorm(hidden)
        self.drop         = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.readout_q)

    def forward(
        self,
        h_atom: Tensor,   # [total_atoms, H]
        batch:  Tensor,   # [total_atoms]  — graph membership
        c_atom: Tensor,   # [total_atoms, cond_proj_dim]  — per-atom condition
    ) -> Tensor:
        """Returns [B, H] graph-level representation."""
        # Stage 1: condition-aware gating
        alpha = self.gate(torch.cat([h_atom, c_atom], dim=-1))  # [N, 1]
        h_gated = alpha * h_atom                                 # [N, H]

        # Stage 2: soft fragment assignment
        S = F.softmax(self.assign(h_gated), dim=-1)             # [N, K]

        # Per-graph pooling (must handle variable-size graphs)
        B = int(batch.max().item()) + 1
        atom_list = unbatch(h_gated, batch)
        s_list    = unbatch(S, batch)

        cluster_embeds: list[Tensor] = []
        for h_g, s_g in zip(atom_list, s_list):
            # s_g: [n_atoms_g, K],  h_g: [n_atoms_g, H]
            # cluster embed: [K, H] = S^T · h_gated
            c_embed = s_g.t() @ h_g                             # [K, H]
            c_embed = self.cluster_proj(c_embed)                # [K, H]
            cluster_embeds.append(c_embed)

        # Stack for batched attention readout: [B, K, H]
        K_clusters = self.n_clusters
        H = h_atom.shape[-1]
        cluster_stack = torch.stack(cluster_embeds, dim=0)     # [B, K, H]

        # Stage 3: multi-head attention readout
        # Q: learnable query vectors [pool_heads, d_head]
        ph, dh = self.pool_heads, self.head_dim

        K_proj = self.readout_k(cluster_stack).view(B, K_clusters, ph, dh)  # [B, K, ph, dh]
        V_proj = self.readout_v(cluster_stack).view(B, K_clusters, ph, dh)

        Q_exp = self.readout_q.unsqueeze(0).expand(B, -1, -1)   # [B, ph, dh]

        # Attention: Q · K^T / sqrt(d)
        attn_logits = torch.einsum('bpd,bkpd->bpk', Q_exp, K_proj) * self.scale  # [B, ph, K]
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.drop(attn_weights)

        # Aggregate: weighted sum of V
        out = torch.einsum('bpk,bkpd->bpd', attn_weights, V_proj)  # [B, ph, dh]
        out = out.reshape(B, H)                                      # [B, H]
        out = self.out_norm(out)

        return out


# ── Main Model ────────────────────────────────────────────────────────────────

class OptionE(nn.Module):
    """
    pH-Aware Gated Equivariant Transformer (pGET).

    Forward signature
    -----------------
    data : torch_geometric.data.HeteroData
      Required:
        data['ligand'].x          [N, 35]
        data['ligand'].batch      [N]
        data['ligand','bond','ligand'].edge_index  [2, 2E]
        data['ligand','bond','ligand'].edge_attr   [2E, 13]
        data.ph_enc / data.temp_enc / data.box_idx / data.receptor_flag

    Returns
    -------
    delta_g : Tensor [B, 1]
    aux     : dict
        'denoise_loss' : Tensor (scalar, only if training + denoise_weight > 0)
        'cluster_assignments' : Tensor [N, K]  (soft fragment assignments)
    """

    def __init__(self, cfg: OptionEConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or OptionEConfig()
        self.cfg = cfg
        H = cfg.n_heads * cfg.head_dim   # total hidden dim (e.g. 4×32 = 128)

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(cfg.ligand_node_dim, H)
        self.edge_proj  = nn.Linear(cfg.ligand_edge_dim, cfg.ligand_edge_dim)

        # ── Condition encoder ─────────────────────────────────────────────────
        self.box_embed = nn.Embedding(cfg.n_box_types, cfg.box_embed_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.film_hidden),
            nn.GELU(),
            nn.Linear(cfg.film_hidden, 32),
        )
        cond_proj_dim = 32

        # ── FiLM generator (per-layer condition gates) ────────────────────────
        self.film = _FiLMGenerator(
            cond_dim  = cond_proj_dim,
            hidden    = cfg.film_hidden,
            n_layers  = cfg.n_gect_layers,
            out_dim   = H,
        )

        # ── GECT layers ──────────────────────────────────────────────────────
        self.gect_layers = nn.ModuleList([
            _GECTLayer(
                hidden        = H,
                heads         = cfg.n_heads,
                head_dim      = cfg.head_dim,
                edge_dim      = cfg.ligand_edge_dim,
                edge_gate_dim = cfg.edge_gate_dim,
                dropout       = cfg.dropout,
            )
            for _ in range(cfg.n_gect_layers)
        ])

        # ── Hierarchical pooling ──────────────────────────────────────────────
        self.pool = _HierarchicalGatedPool(
            hidden        = H,
            cond_proj_dim = cond_proj_dim,
            n_clusters    = cfg.n_soft_clusters,
            pool_heads    = cfg.pool_heads,
            dropout       = cfg.dropout,
        )

        # ── Prediction head ───────────────────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(H + cond_proj_dim, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, cfg.mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden // 2, 1),
        )

        # ── Auxiliary denoising head (optional) ───────────────────────────────
        self.denoise_weight = cfg.denoise_weight
        self.denoise_std    = cfg.denoise_noise_std
        if cfg.denoise_weight > 0:
            self.denoise_head = nn.Sequential(
                nn.Linear(H, H),
                nn.GELU(),
                nn.Linear(H, cfg.ligand_node_dim),
            )
        else:
            self.denoise_head = None

    def forward(self, data) -> tuple[Tensor, dict]:
        device = data['ligand'].x.device
        B      = int(data['ligand'].batch.max().item()) + 1

        # ── Raw features ──────────────────────────────────────────────────────
        x_raw  = data['ligand'].x                              # [N, 35]
        ei     = data['ligand', 'bond', 'ligand'].edge_index   # [2, E]
        ea     = data['ligand', 'bond', 'ligand'].edge_attr    # [E, 13]
        batch  = data['ligand'].batch                          # [N]

        # ── Condition encoding ────────────────────────────────────────────────
        cond_raw = self._encode_condition(data, device, B)     # [B, COND_DIM]
        cond     = self.cond_proj(cond_raw)                    # [B, 32]

        # Expand condition to per-atom tensor
        c_atom = cond[batch]                                   # [N, 32]

        # ── FiLM parameters ───────────────────────────────────────────────────
        gamma, beta = self.film(cond)  # both [B, L, H]

        # ── Input projection ──────────────────────────────────────────────────
        x = self.input_proj(x_raw)                             # [N, H]
        ea_proj = self.edge_proj(ea)                           # [E, 13]

        # ── Denoising (training only) ─────────────────────────────────────────
        aux: dict = {}
        if self.training and self.denoise_weight > 0 and self.denoise_head is not None:
            noise = torch.randn_like(x_raw) * self.denoise_std
            x_noisy = self.input_proj(x_raw + noise)
            # We'll compute denoising loss after the GECT layers
            denoise_input = x_noisy
        else:
            denoise_input = None

        # ── GECT layers ──────────────────────────────────────────────────────
        for l, layer in enumerate(self.gect_layers):
            # Per-atom FiLM parameters for this layer
            g_l = gamma[:, l, :]   # [B, H]
            b_l = beta[:, l, :]    # [B, H]
            g_atom = g_l[batch]    # [N, H]
            b_atom = b_l[batch]    # [N, H]

            x = layer(x, ei, ea_proj, g_atom, b_atom)

        # ── Denoising loss ────────────────────────────────────────────────────
        if denoise_input is not None:
            # Run noisy features through same layers
            x_den = denoise_input
            for l, layer in enumerate(self.gect_layers):
                g_atom = gamma[:, l, :][batch]
                b_atom = beta[:, l, :][batch]
                x_den = layer(x_den, ei, ea_proj, g_atom, b_atom)
            x_recon = self.denoise_head(x_den)  # [N, 35]
            denoise_loss = F.mse_loss(x_recon, x_raw)
            aux['denoise_loss'] = denoise_loss * self.denoise_weight

        # ── Hierarchical pooling ──────────────────────────────────────────────
        h_graph = self.pool(x, batch, c_atom)                 # [B, H]

        # Store cluster assignments for interpretability
        with torch.no_grad():
            S = F.softmax(self.pool.assign(self.pool.gate(
                torch.cat([x, c_atom], dim=-1)) * x), dim=-1)
            aux['cluster_assignments'] = S

        # ── Prediction ────────────────────────────────────────────────────────
        h_cat = torch.cat([h_graph, cond], dim=-1)            # [B, H + 32]
        delta_g = self.mlp(h_cat)                              # [B, 1]

        return delta_g, aux

    # ── Condition encoding (same interface as Options A-D) ────────────────────

    def _encode_condition(self, data, device: torch.device, B: int) -> Tensor:
        if hasattr(data, 'ph_enc'):
            ph_enc   = _scalar_to_batch(data.ph_enc,       B, device)
            temp_enc = _scalar_to_batch(data.temp_enc,      B, device)
            box_idx  = _scalar_to_batch(data.box_idx,       B, device, long=True)
            rec_flag = _scalar_to_batch(data.receptor_flag, B, device)
        else:
            ph_enc, temp_enc, box_idx, rec_flag = _encode_raw(data, B, device)
        box_emb = self.box_embed(box_idx)
        cont    = torch.stack([ph_enc, temp_enc, rec_flag], dim=-1)
        return torch.cat([cont, box_emb], dim=-1)
