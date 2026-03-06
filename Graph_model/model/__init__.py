"""
Graph_model.model
=================
Four GNN model architectures for docking affinity prediction.

Exports
-------
OptionA       : Baseline Condition-Aware GATv2 (global mean pool → MLP)
OptionB       : Dual Encoder with Stacked Cross-Attention (interpretable)
OptionC       : Fragment-Aware Hierarchical MPNN (per-fragment ΔG contributions)
OptionD       : Multi-Task Selectivity GNN (Kendall uncertainty weighting)

OptionAConfig : Hyperparams for Option A
OptionBConfig : Hyperparams for Option B
OptionCConfig : Hyperparams for Option C
OptionDConfig : Hyperparams for Option D

Dimensions (imported from config for convenient access)
---
LIGAND_NODE_DIM  = 35
LIGAND_EDGE_DIM  = 13
PROTEIN_NODE_DIM = 30
PROTEIN_EDGE_DIM = 4
BIPARTITE_EDGE_DIM = 8
COND_DIM         = 19   (3 continuous + 16-dim box embedding)
"""

from .option_a import OptionA
from .option_b import OptionB
from .option_c import OptionC
from .option_d import OptionD, _masked_mse as masked_mse   # re-export for training loop
from .option_e import OptionE
from .uncertainty import HeteroscedasticWrapper, GaussianNLLLoss, calibration_error
from .ensemble import DeepEnsemble, train_ensemble

# New architectures (Phase 8 improvements)
from .dimenet import DimeNet, DimeNetConfig
from .egnn import EGNN, EGNNConfig
from .ggnn_sequential import GGNNSequential, GGNNSeqConfig
from .graphormer import Graphormer, GraphormerConfig

# Advanced loss functions
from .losses import (
    ListMLELoss,
    MonotonicityLoss,
    PairwiseRankLoss,
    CombinedDockingLoss,
)

from .config import (
    ModelConfig,
    OptionAConfig,
    OptionBConfig,
    OptionCConfig,
    OptionDConfig,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
    PROTEIN_NODE_DIM,
    PROTEIN_EDGE_DIM,
    BIPARTITE_EDGE_DIM,
    COND_DIM,
    COND_CONTINUOUS,
    BOX_EMBEDDING_DIM,
    N_BOX_TYPES,
)
from .option_e import OptionEConfig

__all__ = [
    # Models A–E
    "OptionA", "OptionB", "OptionC", "OptionD", "OptionE",
    # New architectures (F–I)
    "DimeNet", "EGNN", "GGNNSequential", "Graphormer",
    # Configs
    "ModelConfig", "OptionAConfig", "OptionBConfig", "OptionCConfig", "OptionDConfig",
    "OptionEConfig",
    "DimeNetConfig", "EGNNConfig", "GGNNSeqConfig", "GraphormerConfig",
    # Dims
    "LIGAND_NODE_DIM", "LIGAND_EDGE_DIM",
    "PROTEIN_NODE_DIM", "PROTEIN_EDGE_DIM", "BIPARTITE_EDGE_DIM",
    "COND_DIM", "COND_CONTINUOUS", "BOX_EMBEDDING_DIM", "N_BOX_TYPES",
    # Utils
    "masked_mse",
    # Losses
    "ListMLELoss", "MonotonicityLoss", "PairwiseRankLoss", "CombinedDockingLoss",
    # Phase 5 — Uncertainty
    "HeteroscedasticWrapper", "GaussianNLLLoss", "calibration_error",
    "DeepEnsemble", "train_ensemble",
]
