"""
Graph_model.data.features
==========================
Atomic/bond feature vectors, galloyl fragment detection, and condition encoding.

Public API
----------
atom_features(mol)          → np.ndarray  [N_atoms, ATOM_FEAT_DIM]
bond_features(mol)          → np.ndarray  [N_bonds, BOND_FEAT_DIM]
GalloylFragmentDetector     → fragment counts + subgraph node lists
ConditionEncoder            → 4-vector [ph_enc, temp_enc, box_idx, receptor_flag]
ATOM_FEAT_DIM               → 54  (see atom.py for breakdown)
BOND_FEAT_DIM               → 12

Node-feature dimensions in this repo — there are TWO featurisers
----------------------------------------------------------------
  features/atom.py       ATOM_FEAT_DIM   = 54, BOND_FEAT_DIM   = 12
      used by: data/dataset.py (homogeneous `Data`, currently unused by any
      model — every model indexes data['ligand'], which dataset.py never sets)
  graph/level1_ligand.py LIGAND_NODE_DIM = 35, LIGAND_EDGE_DIM = 13
      used by: graph/builder.py -> data/hetero_dataset.py -> ALL models

35 is therefore the number that describes the trained models. Both constants
above are verified against the featurisers at runtime by tests/test_features.py.

Phase 8 additions:
conformer_3d                → 3D conformer-aware features
ecfp                        → ECFP auxiliary node features
box_residues                → Protein box residue composition (20-dim)
"""

from .atom       import atom_features, bond_features, ATOM_FEAT_DIM, BOND_FEAT_DIM
from .galloyl    import GalloylFragmentDetector
from .conditions import ConditionEncoder

# Phase 8 — new feature modules
from .conformer_3d import (
    generate_conformer,
    conformer_node_features,
    conformer_edge_features,
    augment_ligand_graph_3d,
    CONFORMER_NODE_DIM,
    CONFORMER_EDGE_DIM,
)
from .ecfp import (
    ecfp_node_features,
    ecfp_mol_features,
    augment_nodes_with_ecfp,
    ECFP_NODE_DIM,
    ECFP_MOL_DIM,
)
from .box_residues import (
    residue_composition,
    normalised_residue_composition,
    box_residue_condition_vector,
    BOX_RESIDUE_DIM,
    AMINO_ACIDS_20,
)

__all__ = [
    "atom_features", "bond_features",
    "ATOM_FEAT_DIM", "BOND_FEAT_DIM",
    "GalloylFragmentDetector",
    "ConditionEncoder",
    # Phase 8 — conformer
    "generate_conformer", "conformer_node_features",
    "conformer_edge_features", "augment_ligand_graph_3d",
    "CONFORMER_NODE_DIM", "CONFORMER_EDGE_DIM",
    # Phase 8 — ECFP
    "ecfp_node_features", "ecfp_mol_features",
    "augment_nodes_with_ecfp", "ECFP_NODE_DIM", "ECFP_MOL_DIM",
    # Phase 8 — box residues
    "residue_composition", "normalised_residue_composition",
    "box_residue_condition_vector", "BOX_RESIDUE_DIM", "AMINO_ACIDS_20",
]
