"""
Graph_model.data.features.box_residues
========================================
Protein binding box as a residue composition vector.

Problem 1c — Instead of encoding the box type as a single integer (→ embedding),
also provide a 20-dimensional amino acid composition vector:

  [count_ALA, count_ARG, count_ASN, count_ASP, count_CYS, ...]

Each element = count of that amino acid type within the binding box,
optionally normalised by box volume or total residue count.

This gives the model a *continuous* representation of the protein pocket
chemistry that generalises across box types.

Features produced
-----------------
  BOX_RESIDUE_DIM = 20   (one dim per standard amino acid)

Public API
----------
  residue_composition(residue_names) → np.ndarray [20]
  normalised_residue_composition(residue_names) → np.ndarray [20]
  encode_box_residues(residue_names_list, normalize=True) → np.ndarray [B, 20]

References
----------
• Standard 20-letter amino acid alphabet (IUPAC)
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Standard 20 amino acids in alphabetical 3-letter code order
AMINO_ACIDS_20: list[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS_20)}

BOX_RESIDUE_DIM: int = 20

__all__ = [
    "BOX_RESIDUE_DIM",
    "AMINO_ACIDS_20",
    "AA_TO_IDX",
    "residue_composition",
    "normalised_residue_composition",
    "encode_box_residues",
]


# ── Core functions ────────────────────────────────────────────────────────────

def residue_composition(residue_names: Sequence[str]) -> np.ndarray:
    """
    Compute raw amino acid counts for a list of residue names.

    Parameters
    ----------
    residue_names : sequence of 3-letter amino acid codes (e.g. ["GLU", "LYS", "GLU"])

    Returns
    -------
    counts : np.ndarray [20]  integer counts per standard amino acid
    """
    counts = np.zeros(BOX_RESIDUE_DIM, dtype=np.float32)
    counter = Counter(residue_names)

    for aa, cnt in counter.items():
        aa_upper = aa.upper().strip()
        if aa_upper in AA_TO_IDX:
            counts[AA_TO_IDX[aa_upper]] = float(cnt)

    return counts


def normalised_residue_composition(
    residue_names: Sequence[str],
    method: str = "l1",
) -> np.ndarray:
    """
    Compute normalised amino acid composition vector.

    Parameters
    ----------
    residue_names : sequence of 3-letter AA codes
    method        : 'l1' (sum-to-1 fractions) or 'l2' (unit-norm) or 'count'

    Returns
    -------
    comp : np.ndarray [20]
    """
    counts = residue_composition(residue_names)

    if method == "count":
        return counts
    elif method == "l1":
        total = counts.sum()
        if total > 0:
            return counts / total
        return counts
    elif method == "l2":
        norm = np.linalg.norm(counts)
        if norm > 0:
            return counts / norm
        return counts
    else:
        raise ValueError(f"Unknown normalisation method: {method}")


def encode_box_residues(
    residue_names_list: Sequence[Sequence[str]],
    normalize: bool = True,
    method: str = "l1",
) -> np.ndarray:
    """
    Batch-encode multiple boxes' residue compositions.

    Parameters
    ----------
    residue_names_list : list of lists of residue names, one per graph in batch
    normalize          : whether to normalise
    method             : normalisation method if normalize=True

    Returns
    -------
    compositions : np.ndarray [B, 20]
    """
    if normalize:
        fn = lambda rn: normalised_residue_composition(rn, method)
    else:
        fn = residue_composition

    compositions = np.stack([fn(rn) for rn in residue_names_list], axis=0)
    return compositions.astype(np.float32)


# ── Utility: extract residue names from HeteroData ───────────────────────────

def extract_residue_names_from_heterodata(data) -> list[str]:
    """
    Extract residue name list from a PyG HeteroData object.

    Looks for data['residue'].residue_names or data.residue_names.

    Returns
    -------
    names : list of str (3-letter codes)
    """
    # Try HeteroData residue store
    if hasattr(data, 'residue') and hasattr(data['residue'], 'residue_names'):
        names = data['residue'].residue_names
    elif hasattr(data, 'residue_names'):
        names = data.residue_names
    else:
        return []

    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    # Could be a tensor or other format
    return [str(n) for n in names]


# ── Convenience: build condition vector extension ─────────────────────────────

def box_residue_condition_vector(
    data,
    normalize: bool = True,
) -> np.ndarray:
    """
    Build a [20]-dim residue composition vector from a HeteroData object.

    This vector is designed to be **concatenated** with the existing
    19-dim condition vector (ph_enc, temp_enc, rec_flag, box_embed[16])
    to create a 39-dim extended condition.

    Parameters
    ----------
    data      : PyG HeteroData
    normalize : whether to L1-normalise the composition

    Returns
    -------
    comp : np.ndarray [20]
    """
    res_names = extract_residue_names_from_heterodata(data)

    if not res_names:
        return np.zeros(BOX_RESIDUE_DIM, dtype=np.float32)

    if normalize:
        return normalised_residue_composition(res_names, method="l1")
    return residue_composition(res_names)
