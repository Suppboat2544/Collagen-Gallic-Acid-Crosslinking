"""
Graph_model.graph.level1_ligand
=================================
Standard ligand molecular graph from a SMILES string.

Node features  —  32 dimensions
---------------------------------
Segment                   Dims   Notes
─────────────────────── ──────  ─────────────────────────────────────────
element one-hot + other    11    C N O S P F Cl Br I B + UNK
hybridisation              5     SP SP2 SP3 OTHER + UNK
formal charge              5     −2 −1 0 +1 +2
H-count                    5     0 1 2 3 4
aromaticity                1     bool
ring membership            3     is_in_3 / is_in_5 / is_in_6
chirality                  2     CW / CCW
mass_normed                1     atom mass / 100
────────────────────────────── ──────
                              33  (LIGAND_NODE_DIM)

Edge features  —  13 dimensions
---------------------------------
bond type one-hot          4     SINGLE DOUBLE TRIPLE AROMATIC
is_conjugated              1
is_in_ring                 1
stereo one-hot             4     NONE E Z OTHER
ring_size_hint             3     shared_3 / shared_5 / shared_6
────────────────────────────── ──────
                              13  (LIGAND_EDGE_DIM)

Public API
----------
mol_to_ligand_graph(smiles_or_mol) → (node_feat, edge_index, edge_feat)
LIGAND_NODE_DIM, LIGAND_EDGE_DIM
"""

from __future__ import annotations

from typing import Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, Descriptors

# ── Feature dimensions ────────────────────────────────────────────────────────
# Feature count: element(11) + hybrid(5) + charge(6) + H-count(6) + arom(1) + ring(3) + chiral(2) + mass(1) = 35
LIGAND_NODE_DIM: int = 35
LIGAND_EDGE_DIM: int = 13

# ── Element set ───────────────────────────────────────────────────────────────
_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]   # + UNK = 11

# ── Hybridisation ─────────────────────────────────────────────────────────────
_HYBRID = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
]   # + OTHER + UNK = 5 total → represented as 4 with other flag

# ── Stereo ────────────────────────────────────────────────────────────────────
_STEREO_KEEP = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOZ,
]   # + OTHER = 4

# ── Bond types ────────────────────────────────────────────────────────────────
_BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]   # 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ohe(value, choices: list) -> list[float]:
    """One-hot with trailing 'other' bin."""
    v = [float(value == c) for c in choices]
    v.append(float(value not in choices))
    return v


def _ring_size_bits(atom: rdchem.Atom) -> list[float]:
    """3-bit membership: [in_3, in_5, in_6]."""
    ri = atom.GetOwningMol().GetRingInfo()
    in3 = in5 = in6 = False
    for ring in ri.AtomRings():
        if atom.GetIdx() in ring:
            sz = len(ring)
            if sz == 3: in3 = True
            if sz == 5: in5 = True
            if sz == 6: in6 = True
    return [float(in3), float(in5), float(in6)]


def _bond_ring_size_bits(bond: rdchem.Bond) -> list[float]:
    """3-bit: edge shared in 3/5/6-membered ring."""
    mol  = bond.GetOwningMol()
    ri   = mol.GetRingInfo()
    ai   = bond.GetBeginAtomIdx()
    bj   = bond.GetEndAtomIdx()
    in3 = in5 = in6 = False
    for ring in ri.AtomRings():
        ring_s = set(ring)
        if ai in ring_s and bj in ring_s:
            sz = len(ring)
            if sz == 3: in3 = True
            if sz == 5: in5 = True
            if sz == 6: in6 = True
    return [float(in3), float(in5), float(in6)]


# ── Node feature vector ───────────────────────────────────────────────────────

def _atom_feat(atom: rdchem.Atom) -> list[float]:
    """32-dim atom feature vector."""
    v: list[float] = []
    # element (11)
    v += _ohe(atom.GetSymbol(), _ELEMENTS)
    # hybridisation (5 → 4 + other)
    v += _ohe(atom.GetHybridization(),
              [rdchem.HybridizationType.SP,
               rdchem.HybridizationType.SP2,
               rdchem.HybridizationType.SP3,
               rdchem.HybridizationType.SP3D])       # 5
    # formal charge (5)
    v += _ohe(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
    # H-count (5)
    v += _ohe(min(atom.GetTotalNumHs(), 4), [0, 1, 2, 3, 4])
    # aromaticity (1)
    v.append(float(atom.GetIsAromatic()))
    # ring size (3)
    v += _ring_size_bits(atom)
    # chirality (2): CW / CCW
    tag = atom.GetChiralTag()
    v.append(float(tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW))
    v.append(float(tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW))
    # normalised mass (1)
    v.append(atom.GetMass() / 100.0)
    assert len(v) == LIGAND_NODE_DIM, f"node feat len={len(v)} expected {LIGAND_NODE_DIM}"
    return v


# ── Edge feature vector ───────────────────────────────────────────────────────

def _bond_feat(bond: rdchem.Bond) -> list[float]:
    """13-dim bond feature vector."""
    v: list[float] = []
    # bond type (4)
    v += _ohe(bond.GetBondType(), _BOND_TYPES[:-1])    # 3 choices + other = 4
    # conjugated / ring (2)
    v.append(float(bond.GetIsConjugated()))
    v.append(float(bond.IsInRing()))
    # stereo (4)
    v += _ohe(bond.GetStereo(), _STEREO_KEEP)
    # ring size hint (3)
    v += _bond_ring_size_bits(bond)
    assert len(v) == LIGAND_EDGE_DIM, f"edge feat len={len(v)} expected {LIGAND_EDGE_DIM}"
    return v


# ── Public API ────────────────────────────────────────────────────────────────

def mol_to_ligand_graph(
    smiles_or_mol: Union[str, rdchem.Mol]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a SMILES string or RDKit mol to (node_feat, edge_index, edge_feat).

    Each bond is added in both directions (PyG convention).

    Parameters
    ----------
    smiles_or_mol : str | rdkit.Chem.Mol

    Returns
    -------
    node_feat  : float32  [N_atoms, LIGAND_NODE_DIM]
    edge_index : int64    [2, 2*N_bonds]   (undirected)
    edge_feat  : float32  [2*N_bonds, LIGAND_EDGE_DIM]

    Raises
    ------
    ValueError  if SMILES cannot be parsed.
    """
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {smiles_or_mol!r}")
    else:
        mol = smiles_or_mol
        if mol is None:
            raise ValueError("Received None as mol argument")

    n = mol.GetNumAtoms()

    # Node matrix
    node_feat = np.array([_atom_feat(a) for a in mol.GetAtoms()], dtype=np.float32)

    # Edge index and edge features (undirected)
    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat  = _bond_feat(bond)
        rows += [i, j];  cols += [j, i]
        edge_feats += [feat, feat]

    if rows:
        edge_index = np.array([rows, cols], dtype=np.int64)
        edge_feat  = np.array(edge_feats,   dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feat  = np.zeros((0, LIGAND_EDGE_DIM), dtype=np.float32)

    return node_feat, edge_index, edge_feat
