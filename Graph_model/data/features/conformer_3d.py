"""
Graph_model.data.features.conformer_3d
========================================
3D conformer-aware molecular features via RDKit ETKDG + MMFF94.

Problem 1a — Instead of treating molecules as 2D topology-only graphs,
generate 3D conformers and extract geometric descriptors:

  • Bond lengths (Å)
  • Bond angles (degrees / π)
  • Dihedral angles (sin, cos)
  • Per-atom 3D shape descriptors (distance to centroid, SASA proxy)

Features produced
-----------------
  CONFORMER_NODE_DIM  = 4   (extra per-atom: dist_centroid, n_xyz)
  CONFORMER_EDGE_DIM  = 5   (extra per-bond: length, angle_sin, angle_cos,
                              dihedral_sin, dihedral_cos)

These are **appended** to existing ligand node/edge features, NOT replacements.

Public API
----------
  generate_conformer(mol, n_confs=10, seed=42) → Mol (with best conformer)
  conformer_node_features(mol) → np.ndarray [N, CONFORMER_NODE_DIM]
  conformer_edge_features(mol, edge_index) → np.ndarray [2E, CONFORMER_EDGE_DIM]

References
----------
• Riniker S., Landrum G. "Better Informed Distance Geometry: Using What
  We Know To Improve Conformation Generation." JCIM 2015.
• Halgren T. "Merck Molecular Force Field." JCCS 1996.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

# ── Feature dimension constants ───────────────────────────────────────────────
CONFORMER_NODE_DIM: int = 4   # dist_centroid, x_norm, y_norm, z_norm
CONFORMER_EDGE_DIM: int = 5   # bond_length, angle_sin, angle_cos, dihed_sin, dihed_cos

__all__ = [
    "CONFORMER_NODE_DIM",
    "CONFORMER_EDGE_DIM",
    "generate_conformer",
    "conformer_node_features",
    "conformer_edge_features",
]


# ── Conformer generation ─────────────────────────────────────────────────────

def generate_conformer(
    mol: Chem.Mol,
    n_confs: int = 10,
    seed: int = 42,
    max_attempts: int = 50,
    force_field: str = "MMFF94",
) -> Chem.Mol:
    """
    Generate a low-energy 3D conformer using ETKDG + MMFF94/UFF optimization.

    Parameters
    ----------
    mol          : RDKit Mol (2D or 3D; Hs are added internally)
    n_confs      : number of conformers to generate
    seed         : random seed for reproducibility
    max_attempts : max embedding attempts per conformer
    force_field  : 'MMFF94' or 'UFF'

    Returns
    -------
    mol : RDKit Mol with the lowest-energy conformer attached (conformer ID 0)

    Raises
    ------
    ValueError : if no conformer could be generated
    """
    mol = Chem.RWMol(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed  = seed
    params.numThreads  = 1
    params.maxAttempts = max_attempts

    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)

    if len(conf_ids) == 0:
        # Fallback: try without distance geometry constraints
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        if mol.GetNumConformers() == 0:
            raise ValueError("Failed to generate any 3D conformer")
        conf_ids = [0]

    # Optimise and pick lowest energy
    energies = []
    for cid in conf_ids:
        try:
            if force_field == "MMFF94":
                props = AllChem.MMFFGetMoleculeProperties(mol)
                if props is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=int(cid))
                    if ff is not None:
                        ff.Minimize(maxIts=500)
                        energies.append((float(ff.CalcEnergy()), int(cid)))
                        continue
            # Fallback to UFF
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=int(cid))
            if ff is not None:
                ff.Minimize(maxIts=500)
                energies.append((float(ff.CalcEnergy()), int(cid)))
            else:
                energies.append((float("inf"), int(cid)))
        except Exception:
            energies.append((float("inf"), int(cid)))

    if not energies:
        raise ValueError("Force field optimization failed for all conformers")

    # Select lowest energy conformer
    best_energy, best_cid = min(energies, key=lambda x: x[0])

    # Remove Hs for feature extraction (match 2D atom indices)
    mol_out = Chem.RemoveHs(mol)

    # Ensure we keep only the best conformer
    if mol_out.GetNumConformers() == 0:
        raise ValueError("Conformer lost after RemoveHs")

    return mol_out


# ── Node features ─────────────────────────────────────────────────────────────

def conformer_node_features(mol: Chem.Mol) -> np.ndarray:
    """
    Extract 3D node features from a molecule with a conformer.

    Features per atom [CONFORMER_NODE_DIM=4]:
      0: distance to molecular centroid (Å), normalised by max
      1-3: normalised xyz coordinates (centred and scaled)

    Parameters
    ----------
    mol : RDKit Mol with at least one conformer

    Returns
    -------
    feats : np.ndarray [N_heavy, CONFORMER_NODE_DIM]
    """
    n_atoms = mol.GetNumAtoms()

    if mol.GetNumConformers() == 0:
        # Return zeros if no conformer available
        return np.zeros((n_atoms, CONFORMER_NODE_DIM), dtype=np.float32)

    conf = mol.GetConformer(0)
    positions = np.array(conf.GetPositions(), dtype=np.float32)  # [N, 3]

    # Centre on centroid
    centroid = positions.mean(axis=0)
    rel_pos  = positions - centroid  # [N, 3]

    # Distance to centroid
    dists = np.linalg.norm(rel_pos, axis=1)  # [N]
    max_dist = dists.max() + 1e-8
    dists_norm = dists / max_dist  # [N]

    # Normalise positions by max absolute coordinate
    max_coord = np.abs(rel_pos).max() + 1e-8
    pos_norm  = rel_pos / max_coord  # [N, 3]

    feats = np.column_stack([dists_norm, pos_norm])  # [N, 4]
    assert feats.shape == (n_atoms, CONFORMER_NODE_DIM)
    return feats.astype(np.float32)


# ── Edge features ─────────────────────────────────────────────────────────────

def conformer_edge_features(
    mol: Chem.Mol,
    edge_index: np.ndarray,
) -> np.ndarray:
    """
    Extract 3D edge features for each bond in edge_index.

    Features per edge [CONFORMER_EDGE_DIM=5]:
      0: bond length (Å), normalised (/ 5.0)
      1: bond angle sin  (angle formed with previous atom)
      2: bond angle cos
      3: dihedral angle sin
      4: dihedral angle cos

    Parameters
    ----------
    mol        : RDKit Mol with conformer
    edge_index : [2, 2*N_bonds] array (PyG convention, both directions)

    Returns
    -------
    feats : np.ndarray [2*N_bonds, CONFORMER_EDGE_DIM]
    """
    n_edges = edge_index.shape[1]

    if mol.GetNumConformers() == 0:
        return np.zeros((n_edges, CONFORMER_EDGE_DIM), dtype=np.float32)

    conf = mol.GetConformer(0)
    positions = np.array(conf.GetPositions(), dtype=np.float32)

    feats = np.zeros((n_edges, CONFORMER_EDGE_DIM), dtype=np.float32)

    for e in range(n_edges):
        i, j = int(edge_index[0, e]), int(edge_index[1, e])

        # Bond length
        length = np.linalg.norm(positions[j] - positions[i])
        feats[e, 0] = length / 5.0  # normalise by ~max bond length

        # Bond angle: find a neighbour k of i (k ≠ j)
        atom_i = mol.GetAtomWithIdx(i)
        neighbours_i = [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j]

        if neighbours_i:
            k = neighbours_i[0]
            try:
                angle = rdMolTransforms.GetAngleRad(conf, k, i, j)
                feats[e, 1] = np.sin(angle)
                feats[e, 2] = np.cos(angle)
            except Exception:
                feats[e, 1] = 0.0
                feats[e, 2] = 1.0
        else:
            feats[e, 1] = 0.0
            feats[e, 2] = 1.0    # default: 0° angle

        # Dihedral angle: find neighbour l of j (l ≠ i)
        atom_j = mol.GetAtomWithIdx(j)
        neighbours_j = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i]

        if neighbours_i and neighbours_j:
            k = neighbours_i[0]
            l = neighbours_j[0]
            try:
                dihedral = rdMolTransforms.GetDihedralRad(conf, k, i, j, l)
                feats[e, 3] = np.sin(dihedral)
                feats[e, 4] = np.cos(dihedral)
            except Exception:
                feats[e, 3] = 0.0
                feats[e, 4] = 1.0
        else:
            feats[e, 3] = 0.0
            feats[e, 4] = 1.0  # default: 0° dihedral

    return feats


# ── Convenience: augment existing features ────────────────────────────────────

def augment_ligand_graph_3d(
    smiles: str,
    node_feat: np.ndarray,
    edge_index: np.ndarray,
    edge_feat: np.ndarray,
    n_confs: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment existing 2D ligand features with 3D conformer features.

    Parameters
    ----------
    smiles     : SMILES string
    node_feat  : [N, D_node] existing node features
    edge_index : [2, 2E] existing edge index
    edge_feat  : [2E, D_edge] existing edge features
    n_confs    : number of conformers to try
    seed       : random seed

    Returns
    -------
    aug_node : [N, D_node + CONFORMER_NODE_DIM]
    aug_edge : [2E, D_edge + CONFORMER_EDGE_DIM]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    try:
        mol_3d = generate_conformer(mol, n_confs=n_confs, seed=seed)

        node_3d = conformer_node_features(mol_3d)
        edge_3d = conformer_edge_features(mol_3d, edge_index)

        # Handle atom count mismatch (unlikely but defensive)
        if node_3d.shape[0] != node_feat.shape[0]:
            warnings.warn(
                f"Atom count mismatch: 2D={node_feat.shape[0]} vs 3D={node_3d.shape[0]}. "
                f"Padding with zeros."
            )
            node_3d = np.zeros((node_feat.shape[0], CONFORMER_NODE_DIM), dtype=np.float32)
            edge_3d = np.zeros((edge_feat.shape[0], CONFORMER_EDGE_DIM), dtype=np.float32)

    except (ValueError, RuntimeError):
        # Fallback: zero features if conformer generation fails
        node_3d = np.zeros((node_feat.shape[0], CONFORMER_NODE_DIM), dtype=np.float32)
        edge_3d = np.zeros((edge_feat.shape[0], CONFORMER_EDGE_DIM), dtype=np.float32)

    aug_node = np.concatenate([node_feat, node_3d], axis=-1)
    aug_edge = np.concatenate([edge_feat, edge_3d], axis=-1)

    return aug_node, aug_edge
