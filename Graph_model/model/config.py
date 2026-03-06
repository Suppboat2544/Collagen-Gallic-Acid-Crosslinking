"""
Graph_model.model.config
==========================
Shared hyperparameter defaults and dimension constants for all four model options.

Dimensions at a glance
----------------------
  LIGAND_NODE_DIM    = 35   (level1_ligand)
  LIGAND_EDGE_DIM    = 13
  PROTEIN_NODE_DIM   = 30   (level2_protein)
  PROTEIN_EDGE_DIM   = 4
  BIPARTITE_EDGE_DIM = 8    (level3_bipartite)

  COND_CONTINUOUS    = 3    (ph_enc, temp_enc, receptor_flag)
  BOX_EMBED_DIM      = 16   (nn.Embedding on box_idx)
  COND_DIM           = 19   (total continuous condition vector after embedding)

  HIDDEN_DIM         = 128  (default GNN hidden width)
  N_LAYERS           = 4    (default GNN depth)
  MLP_HIDDEN         = 256
  DROPOUT            = 0.15
"""

from __future__ import annotations
from dataclasses import dataclass, field

from Graph_model.graph import (
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    PROTEIN_NODE_DIM,
    PROTEIN_EDGE_DIM,
    BIPARTITE_EDGE_DIM,
)
from Graph_model.data.config import N_BOX_TYPES, BOX_EMBEDDING_DIM

# ── Condition encoding ────────────────────────────────────────────────────────
COND_CONTINUOUS: int = 3        # ph_enc, temp_enc, receptor_flag
COND_DIM: int = COND_CONTINUOUS + BOX_EMBEDDING_DIM   # 3 + 16 = 19

# ── Re-export graph dims for convenience ─────────────────────────────────────
__all__ = [
    "LIGAND_NODE_DIM", "LIGAND_EDGE_DIM",
    "PROTEIN_NODE_DIM", "PROTEIN_EDGE_DIM",
    "BIPARTITE_EDGE_DIM",
    "COND_DIM", "COND_CONTINUOUS",
    "N_BOX_TYPES", "BOX_EMBEDDING_DIM",
    "ModelConfig",
    "OptionAConfig", "OptionBConfig", "OptionCConfig", "OptionDConfig",
    "OptionEConfig",
]


@dataclass
class ModelConfig:
    """Common hyperparameters shared by all four options."""
    hidden_dim:     int   = 128
    n_layers:       int   = 4
    mlp_hidden:     int   = 256
    dropout:        float = 0.15
    # input dims (read from graph constants; override only for testing)
    ligand_node_dim: int  = LIGAND_NODE_DIM
    ligand_edge_dim: int  = LIGAND_EDGE_DIM
    protein_node_dim: int = PROTEIN_NODE_DIM
    protein_edge_dim: int = PROTEIN_EDGE_DIM
    bipartite_edge_dim: int = BIPARTITE_EDGE_DIM
    cond_dim:       int   = COND_DIM
    n_box_types:    int   = N_BOX_TYPES
    box_embed_dim:  int   = BOX_EMBEDDING_DIM
    # GAT hyperparams shared by A, B, D
    gat_heads:      int   = 4
    gat_head_dim:   int   = 32    # per-head; total hidden = gat_heads * gat_head_dim


@dataclass
class OptionAConfig(ModelConfig):
    """Option A — Baseline: Condition-Aware GCN/GAT.
    4-layer GATv2 on ligand only → global mean pool → MLP(concat cond) → ΔG."""
    gat_heads:      int   = 4      # multi-head attention heads per GAT layer
    gat_head_dim:   int   = 32     # per-head dim  (total = heads × head_dim = 128)
    residual:       bool  = True


@dataclass
class OptionBConfig(ModelConfig):
    """Option B — Dual Encoder with Cross-Attention.
    GNN_L + GNN_P → cross-attention matrix → global readout → MLP → ΔG."""
    attn_heads:     int   = 4      # cross-attention heads
    attn_dropout:   float = 0.1
    n_cross_layers: int   = 2      # stacked cross-attention rounds
    use_bipartite:  bool  = True   # optionally bias attention with Level-3 edges


@dataclass
class OptionCConfig(ModelConfig):
    """Option C — Fragment-Aware Hierarchical MPNN.
    Atom MPNN within fragments → fragment MPNN → per-fragment ΔG → sum."""
    frag_hidden:    int   = 64     # fragment-level embedding dim
    max_fragments:  int   = 10     # safety cap on BRICS fragments per molecule


@dataclass
class OptionDConfig(ModelConfig):
    """Option D — Multi-Task Selectivity GNN.
    Shared encoder → head_collagen, head_mmp1, head_selectivity.
    Loss uses Kendall et al. 2018 uncertainty weighting."""
    n_tasks:        int   = 3      # collagen ΔG | MMP-1 ΔG | selectivity index
    mmp1_weight:    float = 10.0   # upweight sparse MMP-1 task (40 vs 6156 pts)
    learn_log_var:  bool  = True   # learnable log-variance for uncertainty weighting


@dataclass
class OptionEConfig(ModelConfig):
    """Option E — pH-Aware Gated Equivariant Transformer (pGET).

    Novel architecture with condition-gated message passing (FiLM),
    edge-gated transformer layers, and hierarchical gated attention pooling.
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
