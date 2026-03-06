"""
Graph_model.interpret
=====================
Interpretability toolkit for GNN docking models (Phase 6 + Phase 8).

Modules
-------
integrated_gradients : Atom importance via Integrated Gradients (all models)
attention_viz        : Cross-attention binding-site rediscovery (Option B)
fragment_contrib     : Per-fragment ΔG contribution ranking (Option C)
gradcam              : Graph Grad-CAM for atom importance (all models)
probing              : Linear probes on latent embeddings
attention_rollout    : Multi-layer attention flow analysis (Graphormer / pGET)
"""

from .integrated_gradients import (
    integrated_gradients,
    atom_importance_ranking,
)
from .attention_viz import (
    extract_attention,
    rank_residues,
    attention_heatmap_data,
    validate_binding_site_rediscovery,
)
from .fragment_contrib import (
    extract_fragment_contributions,
    label_fragments,
    rank_fragments,
    summarise_pgg_arms,
)
from .gradcam import (
    graph_gradcam,
    gradcam_vs_plip,
    batch_gradcam,
)
from .probing import (
    ProbingClassifier,
    extract_embeddings,
    run_all_probes,
    label_ligand_group,
    label_galloyl_units,
    label_box_type,
)
from .attention_rollout import (
    attention_rollout,
    extract_attention_from_model,
    atom_importance_from_rollout,
    compare_attribution_methods,
)

__all__ = [
    # Integrated Gradients
    "integrated_gradients",
    "atom_importance_ranking",
    # Attention visualization
    "extract_attention",
    "rank_residues",
    "attention_heatmap_data",
    "validate_binding_site_rediscovery",
    # Fragment contributions
    "extract_fragment_contributions",
    "label_fragments",
    "rank_fragments",
    "summarise_pgg_arms",
    # Grad-CAM (Phase 8)
    "graph_gradcam",
    "gradcam_vs_plip",
    "batch_gradcam",
    # Probing classifiers (Phase 8)
    "ProbingClassifier",
    "extract_embeddings",
    "run_all_probes",
    "label_ligand_group",
    "label_galloyl_units",
    "label_box_type",
    # Attention rollout (Phase 8)
    "attention_rollout",
    "extract_attention_from_model",
    "atom_importance_from_rollout",
    "compare_attribution_methods",
]
