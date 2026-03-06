"""
Graph_model.graph
==================
Three-level heterogeneous graph construction for docking affinity prediction.

Levels
------
1. Ligand Molecular Graph       — atom nodes, bond edges
2. Protein Binding-Site Graph   — residue nodes, Cα–Cα contact edges
3. Bipartite Interaction Graph  — ligand-atom ↔ residue proximity edges

Public API
----------
>>> from Graph_model.graph import (
...     ThreeLevelGraphBuilder,
...     mol_to_ligand_graph,
...     BoxProteinGraph,
...     build_bipartite_graph,
...     LIGAND_NODE_DIM,
...     LIGAND_EDGE_DIM,
...     PROTEIN_NODE_DIM,
...     PROTEIN_EDGE_DIM,
...     BIPARTITE_EDGE_DIM,
...     GRAPH_DIM_SUMMARY,
... )

Quick-start
-----------
>>> builder = ThreeLevelGraphBuilder(pdb_dir="Phukhao/collagen_gallic_results")
>>> data = builder.build(record)
>>> data.node_types          # ['ligand', 'residue']
>>> data.edge_types          # [('ligand','bond','ligand'), ...]
"""

from .level1_ligand import (
    mol_to_ligand_graph,
    LIGAND_NODE_DIM,
    LIGAND_EDGE_DIM,
)
from .level2_protein import (
    BoxProteinGraph,
    pdb_for_ph,
    PROTEIN_NODE_DIM,
    PROTEIN_EDGE_DIM,
)
from .level3_bipartite import (
    build_bipartite_graph,
    BIPARTITE_EDGE_DIM,
)
from .builder import (
    ThreeLevelGraphBuilder,
    GRAPH_DIM_SUMMARY,
    print_dim_summary,
)

__all__ = [
    # Level 1
    "mol_to_ligand_graph",
    "LIGAND_NODE_DIM",
    "LIGAND_EDGE_DIM",
    # Level 2
    "BoxProteinGraph",
    "pdb_for_ph",
    "PROTEIN_NODE_DIM",
    "PROTEIN_EDGE_DIM",
    # Level 3
    "build_bipartite_graph",
    "BIPARTITE_EDGE_DIM",
    # Builder
    "ThreeLevelGraphBuilder",
    "GRAPH_DIM_SUMMARY",
    "print_dim_summary",
]
