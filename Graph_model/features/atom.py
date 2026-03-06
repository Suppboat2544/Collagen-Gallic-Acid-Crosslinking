"""
Graph_model.data.features.atom
================================
Node (atom) and edge (bond) feature vectors for PyG graph construction.

Atom feature vector — 74 dimensions total
------------------------------------------
Segment                     Dims  Description
─────────────────────────── ──── ─────────────────────────────────────
element one-hot (+ Other)    11   C N O S P F Cl Br I B + other
hybridisation one-hot         6   SP SP2 SP3 SP3D SP3D2 OTHER
degree (0-6, + other)         8   local connectivity count
formal charge one-hot         5   -2 -1 0 +1 +2
num_Hs (0-4)                  5   implicit + explicit H count
implicit_valence (0-5)        6   inferred valence
is_aromatic                   1   bool
is_in_ring                    1   bool
is_ring_size_3|4|5|6|7|8|>8  7   each as bool flag
chirality                     4   no_chiral, CHI_CW, CHI_CCW, OTHER
─────────────────────────── ──── 
ATOM_FEAT_DIM  =  54  (see ATOM_FEAT_DIM constant)

Bond feature vector — 12 dimensions total
------------------------------------------
bond_type one-hot             4   SINGLE DOUBLE TRIPLE AROMATIC
is_conjugated                 1
is_in_ring                    1
stereo one-hot                6   STEREONONE E Z CIS TRANS ANY
BOND_FEAT_DIM = 12
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

# ── Element set ─────────────────────────────────────────────────────────────
_ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B"]   # +other = 11

# ── Hybridisation set ────────────────────────────────────────────────────────
_HYBRID = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]   # + OTHER = 6

# ── Formal charges ───────────────────────────────────────────────────────────
_FCHARGES = [-2, -1, 0, 1, 2]   # = 5

# ── Stereo types ─────────────────────────────────────────────────────────────
_STEREO = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]   # = 6

# ── Dimensionality constants ─────────────────────────────────────────────────
ATOM_FEAT_DIM: int = (
    len(_ELEMENTS) + 1     # element   11
    + len(_HYBRID) + 1     # hybrid     6
    + 8                    # degree     8  (0..6 + other)
    + len(_FCHARGES)       # fcharge    5
    + 5                    # nH         5  (0..4)
    + 6                    # impl_val   6  (0..5)
    + 1                    # aromatic   1
    + 1                    # in_ring    1
    + 7                    # ring_size  7  (3,4,5,6,7,8,>8)
    + 4                    # chirality  4
)  # total: 11+6+8+5+5+6+1+1+7+4 = 54

BOND_FEAT_DIM: int = 4 + 1 + 1 + 6   # 12


# ── Helpers ──────────────────────────────────────────────────────────────────

def _one_hot(value, choices: list, include_other: bool = True) -> list[float]:
    """One-hot encode *value* against *choices*; append 'other' bin if requested."""
    enc = [float(value == c) for c in choices]
    if include_other:
        enc.append(float(value not in choices))
    return enc


def _ring_size_bits(atom: rdchem.Atom) -> list[float]:
    """7-bit indicator for membership in rings of size 3–8 and >8."""
    ring_info = atom.GetOwningMol().GetRingInfo()
    bits = [0.0] * 7
    for ring in ring_info.AtomRings():
        if atom.GetIdx() in ring:
            sz = len(ring)
            if 3 <= sz <= 8:
                bits[sz - 3] = 1.0
            else:
                bits[6] = 1.0
    return bits


def _chirality_bits(atom: rdchem.Atom) -> list[float]:
    """4-bit chirality: [no_chiral, CW, CCW, other]."""
    tag = atom.GetChiralTag()
    return [
        float(tag == rdchem.ChiralType.CHI_UNSPECIFIED),
        float(tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CW),
        float(tag == rdchem.ChiralType.CHI_TETRAHEDRAL_CCW),
        float(tag not in (
            rdchem.ChiralType.CHI_UNSPECIFIED,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        )),
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def atom_features(mol: rdchem.Mol) -> np.ndarray:
    """
    Return atom feature matrix of shape [N_atoms, ATOM_FEAT_DIM].

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Must have explicit Hs or *Chem.AddHs* called already (for nH accuracy).
        For our pipeline, implicit Hs are read from rdkit so Chem.AddHs is optional.

    Returns
    -------
    np.ndarray of float32, shape [N, ATOM_FEAT_DIM]
    """
    feats = []
    for atom in mol.GetAtoms():
        v: list[float] = []
        # element
        v += _one_hot(atom.GetSymbol(), _ELEMENTS, include_other=True)          # 11
        # hybridisation
        v += _one_hot(atom.GetHybridization(), _HYBRID, include_other=True)     # 6
        # degree (0-6, + other)
        v += _one_hot(min(atom.GetDegree(), 7), list(range(8)),
                      include_other=False)                                        # 8
        # formal charge
        v += _one_hot(atom.GetFormalCharge(), _FCHARGES, include_other=False)   # 5
        # num Hs (0-4)
        v += _one_hot(min(atom.GetTotalNumHs(), 4), list(range(5)),
                      include_other=False)                                        # 5
        # implicit valence (0-5)
        v += _one_hot(min(atom.GetImplicitValence(), 5), list(range(6)),
                      include_other=False)                                        # 6
        # boolean flags
        v.append(float(atom.GetIsAromatic()))                                    # 1
        v.append(float(atom.IsInRing()))                                         # 1
        # ring size membership
        v += _ring_size_bits(atom)                                               # 7
        # chirality
        v += _chirality_bits(atom)                                               # 4
        feats.append(v)

    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[1] == ATOM_FEAT_DIM, (
        f"ATOM_FEAT_DIM mismatch: expected {ATOM_FEAT_DIM}, got {arr.shape[1]}"
    )
    return arr


def bond_features(mol: rdchem.Mol) -> np.ndarray:
    """
    Return bond feature matrix of shape [N_bonds, BOND_FEAT_DIM].

    Note: for PyG, each bond is represented twice (both directions),
    so edge_index and edge_attr must be built accordingly by the caller.

    Returns
    -------
    np.ndarray of float32, shape [N_bonds, BOND_FEAT_DIM]
    """
    _BTYPES = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]

    feats = []
    for bond in mol.GetBonds():
        v: list[float] = []
        # bond type one-hot
        v += _one_hot(bond.GetBondType(), _BTYPES, include_other=False)         # 4
        # conjugated / ring
        v.append(float(bond.GetIsConjugated()))                                  # 1
        v.append(float(bond.IsInRing()))                                         # 1
        # stereo one-hot
        v += _one_hot(bond.GetStereo(), _STEREO, include_other=False)           # 6
        feats.append(v)

    if not feats:
        return np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)

    arr = np.array(feats, dtype=np.float32)
    assert arr.shape[1] == BOND_FEAT_DIM, (
        f"BOND_FEAT_DIM mismatch: expected {BOND_FEAT_DIM}, got {arr.shape[1]}"
    )
    return arr


def mol_to_edge_index_and_attr(mol: rdchem.Mol):
    """
    Convert RDKit mol to undirected PyG-style edge_index and edge_attr.

    Each bond i→j and j→i are both stored (standard PyG convention).

    Returns
    -------
    edge_index : np.ndarray  int64  [2, 2*N_bonds]
    edge_attr  : np.ndarray  float32 [2*N_bonds, BOND_FEAT_DIM]
    """
    rows, cols, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features_single(bond)
        rows += [i, j]
        cols += [j, i]
        attrs += [feat, feat]

    if not rows:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr  = np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)
        return edge_index, edge_attr

    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_attr  = np.array(attrs,        dtype=np.float32)
    return edge_index, edge_attr


def bond_features_single(bond: rdchem.Bond) -> list[float]:
    """Feature vector for one bond (used internally by mol_to_edge_index_and_attr)."""
    _BTYPES = [
        rdchem.BondType.SINGLE,
        rdchem.BondType.DOUBLE,
        rdchem.BondType.TRIPLE,
        rdchem.BondType.AROMATIC,
    ]
    v: list[float] = []
    v += _one_hot(bond.GetBondType(), _BTYPES, include_other=False)
    v.append(float(bond.GetIsConjugated()))
    v.append(float(bond.IsInRing()))
    v += _one_hot(bond.GetStereo(), _STEREO, include_other=False)
    return v
