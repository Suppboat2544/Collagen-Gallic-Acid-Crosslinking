"""
Graph_model.data.features.ecfp
================================
ECFP (Extended-Connectivity Fingerprints) as auxiliary node features.

Problem 1b — Morgan fingerprints capture substructure environment around
each atom. Instead of using ECFP only as a molecule-level descriptor,
we compute per-atom bit contributions and compress them to a dense vector
via learned or PCA projection.

Strategy
--------
1. Compute Morgan fingerprint radius-2 (ECFP4) with bit info tracking
2. For each atom, extract which bits it contributed to → sparse binary vector
3. Compress to ECFP_NODE_DIM dimensions via truncated SVD (offline) or
   random projection (online)

Features produced
-----------------
  ECFP_NODE_DIM   = 32   (compressed per-atom ECFP contribution vector)
  ECFP_MOL_DIM    = 128  (molecule-level ECFP for auxiliary input)

Public API
----------
  ecfp_node_features(mol, dim=32, n_bits=1024) → np.ndarray [N, ECFP_NODE_DIM]
  ecfp_mol_features(mol, n_bits=128) → np.ndarray [ECFP_MOL_DIM]

References
----------
• Rogers D., Hahn M. "Extended-Connectivity Fingerprints." JCIM 2010.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Dimension constants ───────────────────────────────────────────────────────
ECFP_NODE_DIM: int = 32    # compressed per-atom ECFP features
ECFP_MOL_DIM: int  = 128   # molecule-level fingerprint

__all__ = [
    "ECFP_NODE_DIM",
    "ECFP_MOL_DIM",
    "ecfp_node_features",
    "ecfp_mol_features",
]

# ── Random projection matrix (deterministic) ────────────────────────────────

_N_BITS_FULL: int = 1024

@lru_cache(maxsize=16)
def _random_projection_matrix(
    input_dim: int = _N_BITS_FULL,
    output_dim: int = ECFP_NODE_DIM,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a random Gaussian projection matrix for dimensionality reduction.
    Johnson–Lindenstrauss lemma guarantees approximate distance preservation.

    Returns
    -------
    proj : [input_dim, output_dim] float32
    """
    rng = np.random.RandomState(seed)
    proj = rng.randn(input_dim, output_dim).astype(np.float32)
    # Scale for variance preservation
    proj /= np.sqrt(output_dim)
    return proj


# ── Per-atom ECFP features ───────────────────────────────────────────────────

def ecfp_node_features(
    mol: Chem.Mol,
    dim: int = ECFP_NODE_DIM,
    n_bits: int = _N_BITS_FULL,
    radius: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Compute per-atom ECFP contribution features.

    For each atom, we identify which Morgan fingerprint bits it contributes
    to, build a sparse binary vector, then project to a dense `dim`-dimensional
    vector via random Gaussian projection.

    Parameters
    ----------
    mol    : RDKit Mol
    dim    : output dimension per atom (default 32)
    n_bits : number of fingerprint bits before projection
    radius : Morgan fingerprint radius (2 = ECFP4)
    seed   : random seed for projection matrix

    Returns
    -------
    feats : np.ndarray [N_atoms, dim]
    """
    if mol is None:
        raise ValueError("mol is None")

    n_atoms = mol.GetNumAtoms()

    # Get bit info: maps bit_idx → list of (atom_idx, radius) tuples
    bit_info: dict = {}
    AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits, bitInfo=bit_info
    )

    # Build sparse per-atom fingerprint matrix [N_atoms, n_bits]
    atom_fp = np.zeros((n_atoms, n_bits), dtype=np.float32)

    for bit_idx, atom_envs in bit_info.items():
        for atom_idx, _r in atom_envs:
            if 0 <= atom_idx < n_atoms:
                atom_fp[atom_idx, bit_idx] = 1.0

    # Random projection to dim dimensions
    proj = _random_projection_matrix(n_bits, dim, seed)
    feats = atom_fp @ proj  # [N_atoms, dim]

    return feats.astype(np.float32)


# ── Molecule-level ECFP ──────────────────────────────────────────────────────

def ecfp_mol_features(
    mol: Chem.Mol,
    n_bits: int = ECFP_MOL_DIM,
    radius: int = 2,
) -> np.ndarray:
    """
    Compute molecule-level Morgan fingerprint as a dense binary vector.

    Parameters
    ----------
    mol    : RDKit Mol
    n_bits : number of bits (default 128)
    radius : Morgan radius (2 = ECFP4)

    Returns
    -------
    fp : np.ndarray [n_bits]  binary {0, 1}
    """
    if mol is None:
        raise ValueError("mol is None")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


# ── Convenience: augment existing node features ──────────────────────────────

def augment_nodes_with_ecfp(
    smiles: str,
    node_feat: np.ndarray,
    dim: int = ECFP_NODE_DIM,
) -> np.ndarray:
    """
    Append per-atom ECFP features to existing node feature matrix.

    Parameters
    ----------
    smiles    : SMILES string
    node_feat : [N, D] existing node features
    dim       : ECFP dimension to append

    Returns
    -------
    aug_feat : [N, D + dim]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Return padded zeros
        warnings.warn(f"Cannot parse SMILES: {smiles}; padding with zeros")
        ecfp = np.zeros((node_feat.shape[0], dim), dtype=np.float32)
    else:
        ecfp = ecfp_node_features(mol, dim=dim)
        if ecfp.shape[0] != node_feat.shape[0]:
            warnings.warn(
                f"Atom count mismatch: existing={node_feat.shape[0]} vs "
                f"ECFP={ecfp.shape[0]}. Padding with zeros."
            )
            ecfp = np.zeros((node_feat.shape[0], dim), dtype=np.float32)

    return np.concatenate([node_feat, ecfp], axis=-1)
