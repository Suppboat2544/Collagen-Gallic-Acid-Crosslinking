"""
Graph_model.graph.level3_bipartite
=====================================
Level 3: Ligand–Protein Bipartite Interaction Graph

Constructs edges between ligand ATOMS and protein RESIDUES based on
3-D proximity in the best docking pose.

Edge criterion : min heavy-atom distance between ligand atom i and
                 any heavy atom of residue j  <  CONTACT_CUTOFF_A (4.5 Å)

Edge features  —  8 dimensions
---------------------------------
Segment                         Dims   Range / encoding
────────────────────────────── ──── ──────────────────────────────────────────
dist_norm                        1   min_dist / CONTACT_CUTOFF_A  ∈ [0, 1]
hb_donor_atom                    1   {0,1}  ligand atom is HBD
hb_acceptor_atom                 1   {0,1}  ligand atom is HBA
hb_complementarity               1   {0,1}  HBD↔HBA complementary with residue
charge_product                   1   ∈ [−1, 1]  q_atom × q_res (normalised)
hydrophob_product                1   ∈ [0, 1]   H_atom × H_res
is_aromatic_contact              1   {0,1}  ligand atom in aromatic ring
vdw_clash                        1   {0,1}  dist < 2.8 Å (close contact)
────────────────────────────── ────
BIPARTITE_EDGE_DIM               8

API
---
build_bipartite_graph(
    sdf_path: str | Path,
    ca_coords: np.ndarray,     # [N_res, 3] — from Level 2
    residue_names: list[str],  # e.g. ['GLU27A', 'PRO28A']
    ph: float,
    mol: rdkit.Chem.Mol,       # from Level 1 (already parsed)
) → (edge_index, edge_feat)

Where:
  edge_index  : int64   [2, E_bip]   row = atom idx, col = residue idx
  edge_feat   : float32 [E_bip, 8]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .residue_data import (
    charge_at_ph,
    hydrophobicity,
    is_hbond_donor,
    is_hbond_acceptor,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTACT_CUTOFF_A: float = 4.5     # Å  standard "close contact"
VDW_CLASH_A:      float = 2.8     # Å  van-der-Waals clash threshold
BIPARTITE_EDGE_DIM: int = 8

# SMARTS patterns for HBD/HBA (Ertl 2007 / RDKit convention, heavy atoms only)
# HBD: N or O carrying at least one H
_SMARTS_HBD = Chem.MolFromSmarts('[#7,#8;H1,H2,H3,H4]')
# HBA: oxygen/sulfur lone-pair acceptors + neutral sp3/sp2 nitrogen (simplified
#      Ertl 2007 without rare N edge cases yet capturing most drug-like motifs)
_SMARTS_HBA = Chem.MolFromSmarts(
    '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),'
    '$([N;v3;H0;!$([N;v3;H0]-*=!@[O,N,P,S])]),$([n;H0;+0])]'
)

# Gasteiger charge normalisation:  q_norm = clip(q * 2, -1, 1)
# (Gasteiger charges are typically in −0.6 … +0.45 → *2 maps to ~[-1.0, 0.9])
_GASTEIGER_SCALE: float = 2.0

# Crippen per-atom logP normalisation: hydrophob = clip((c + 2) / 3.5, 0, 1)
# Crippen contributions range roughly −2 … +1.5 for drug-like atoms
_CRIPPEN_SHIFT: float = 2.0
_CRIPPEN_RANGE: float = 3.5


# ── Per-molecule real property computation ────────────────────────────────────

def _precompute_atom_props(
    mol,
) -> tuple[np.ndarray, np.ndarray, frozenset, frozenset]:
    """
    Compute per heavy-atom properties using real quantum/empirical methods.

    Parameters
    ----------
    mol : rdkit.Chem.Mol  (heavy atoms, no explicit H added yet)

    Returns
    -------
    gasteiger_norm : float32 [N_heavy]  Gasteiger partial charge in [−1, 1]
        Method: AllChem.ComputeGasteigerCharges on H-added mol
        Ref:    Gasteiger & Marsili, Tetrahedron 36 (1980) 3219–3228
    crippen_hydrophob : float32 [N_heavy]  per-atom logP contribution in [0, 1]
        Method: rdMolDescriptors._CalcCrippenContribs (Wildman & Crippen 1999)
        Normalised: clip((logP_contrib + 2.0) / 3.5, 0, 1)
    hbd_atoms : frozenset[int]  heavy-atom indices that are H-bond donors
        Method: SMARTS [#7,#8;H1,H2,H3,H4]
    hba_atoms : frozenset[int]  heavy-atom indices that are H-bond acceptors
        Method: Ertl 2007 simplified SMARTS
    """
    n_heavy = mol.GetNumAtoms()

    # ── 1. Gasteiger partial charges ─────────────────────────────────────────
    mol_h = Chem.AddHs(mol)          # Gasteiger requires explicit H
    AllChem.ComputeGasteigerCharges(mol_h)
    gasteiger_raw = np.array(
        [mol_h.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
         for i in range(n_heavy)],    # first n_heavy atoms are the original heavy atoms
        dtype=np.float64,
    )
    # Replace NaN (can occur for unusual valences) with 0
    gasteiger_raw = np.where(np.isfinite(gasteiger_raw), gasteiger_raw, 0.0)
    gasteiger_norm = np.clip(gasteiger_raw * _GASTEIGER_SCALE, -1.0, 1.0).astype(np.float32)

    # ── 2. Crippen per-atom logP contributions (atom hydrophobicity) ──────────
    crippen_contribs = rdMolDescriptors._CalcCrippenContribs(mol)  # [(logP, MR), ...]
    logp_raw = np.array([c[0] for c in crippen_contribs], dtype=np.float64)
    crippen_hydrophob = np.clip(
        (logp_raw + _CRIPPEN_SHIFT) / _CRIPPEN_RANGE, 0.0, 1.0
    ).astype(np.float32)

    # ── 3. SMARTS-based H-bond donor / acceptor sets ──────────────────────────
    hbd_atoms: frozenset[int] = frozenset(
        idx
        for match in mol.GetSubstructMatches(_SMARTS_HBD)
        for idx in match
    ) if _SMARTS_HBD is not None else frozenset()

    hba_atoms: frozenset[int] = frozenset(
        idx
        for match in mol.GetSubstructMatches(_SMARTS_HBA)
        for idx in match
    ) if _SMARTS_HBA is not None else frozenset()

    return gasteiger_norm, crippen_hydrophob, hbd_atoms, hba_atoms


# ── SDF loader ────────────────────────────────────────────────────────────────

def _load_sdf_mol(sdf_path: str | Path):
    """
    Load first conformer from SDF (docked pose).
    Returns rdkit.Chem.Mol with 3D coordinates or None.
    """
    try:
        sup = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        for mol in sup:
            if mol is not None:
                return mol
    except Exception:
        pass
    return None


def _mol_atom_coords(mol) -> np.ndarray:
    """Return heavy-atom 3D coordinates as [N_atoms, 3] float32."""
    conf = mol.GetConformer(0)
    coords = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:   # skip explicit H positions
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
    return np.array(coords, dtype=np.float32)


def _mol_heavy_atoms(mol):
    """Yield heavy atoms (skip H) in order."""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            yield atom


# ── Binding-site residue heavy atom coords ───────────────────────────────────

def _residue_heavy_coords(structure, resname_key: str) -> np.ndarray:
    """
    Find residue by label ('GLU27A') in BioPython structure and return
    all heavy-atom coordinates as [K, 3].

    Falls back to empty array if not found.
    """
    # parse resname_key: up to final char = chain, before = resname+seqnum
    # e.g. 'GLU27A' → seqnum=27, chain='A'
    chain_id  = resname_key[-1]
    body      = resname_key[:-1]          # e.g. 'GLU27'
    try:
        # split at first digit
        for i, ch in enumerate(body):
            if ch.isdigit():
                resname = body[:i]
                seq_num = int(body[i:])
                break
        else:
            return np.zeros((0, 3), dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)

    coords = []
    try:
        for model in structure:
            chain = model[chain_id]
            for res in chain:
                if (res.get_resname().strip() == resname and
                        res.get_id()[1] == seq_num):
                    for atom in res.get_atoms():
                        if atom.element != "H":
                            coords.append(atom.get_vector().get_array())
                    break
    except Exception:
        pass

    if coords:
        return np.array(coords, dtype=np.float32)
    return np.zeros((0, 3), dtype=np.float32)


# ── Main public function ──────────────────────────────────────────────────────

def build_bipartite_graph(
    sdf_path:       str | Path,
    ca_coords:      np.ndarray,          # [N_res, 3] from Level 2
    residue_names:  list[str],           # ['GLU27A', ...]  len = N_res
    ph:             float,
    mol=None,                            # rdkit Mol (pre-parsed from Level 1)
    structure=None,                      # BioPython structure for heavy-atom coords
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build bipartite edge set between ligand atoms and protein residues.

    When `structure` is None the Cα coordinate is used as residue position
    (lower accuracy but always available).  When `structure` is provided,
    the actual minimum heavy-atom distance is used.

    Returns
    -------
    edge_index  : int64   [2, E_bip]   row = atom idx, col = residue idx
    edge_feat   : float32 [E_bip, BIPARTITE_EDGE_DIM]
    """
    # 1. Get ligand atoms & coords
    if mol is None:
        mol = _load_sdf_mol(sdf_path)
    if mol is None:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, BIPARTITE_EDGE_DIM), dtype=np.float32),
        )

    # Heavy atoms only
    heavy_atoms   = list(_mol_heavy_atoms(mol))
    conf          = mol.GetConformer(0)
    lig_coords    = np.array(
        [[conf.GetAtomPosition(a.GetIdx()).x,
          conf.GetAtomPosition(a.GetIdx()).y,
          conf.GetAtomPosition(a.GetIdx()).z]
         for a in heavy_atoms],
        dtype=np.float32)

    N_atoms = len(heavy_atoms)
    N_res   = len(residue_names)

    if N_atoms == 0 or N_res == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0, BIPARTITE_EDGE_DIM), dtype=np.float32),
        )

    # 2. Residue heavy-atom coords (fallback to Cα)
    res_heavy: list[np.ndarray] = []
    for j, resname in enumerate(residue_names):
        if structure is not None:
            hc = _residue_heavy_coords(structure, resname)
            res_heavy.append(hc if len(hc) > 0 else ca_coords[j:j+1])
        else:
            res_heavy.append(ca_coords[j:j+1])

    # 3. Pre-compute residue chemistry
    res_charge = np.array(
        [charge_at_ph(rn[:-1] if rn[-1].isalpha() else rn, ph)
         for rn in residue_names], dtype=np.float32)
    res_hydrophob = np.array(
        [hydrophobicity(rn[:-1] if rn[-1].isalpha() else rn)
         for rn in residue_names], dtype=np.float32)
    res_donor = np.array(
        [is_hbond_donor(rn[:-1] if rn[-1].isalpha() else rn)
         for rn in residue_names], dtype=np.float32)
    res_acceptor = np.array(
        [is_hbond_acceptor(rn[:-1] if rn[-1].isalpha() else rn)
         for rn in residue_names], dtype=np.float32)

    # Helper to strip trailing chain char from residue label
    def _res3(rname: str) -> str:
        return rname[:-1] if (rname and rname[-1].isalpha()) else rname

    # 4. Precompute per-atom real chemistry (once per molecule)
    #    Gasteiger partial charges, Crippen logP contributions, SMARTS HBD/HBA
    lig_q, lig_h, hbd_set, hba_set = _precompute_atom_props(mol)

    # 5. Build edges
    atom_indices, res_indices, edge_rows = [], [], []

    for i, atom in enumerate(heavy_atoms):
        atom_xyz = lig_coords[i]
        rdkit_idx = atom.GetIdx()   # same as i when mol has no explicit H

        # atom features from precomputed arrays
        hbd_a  = float(rdkit_idx in hbd_set)
        hba_a  = float(rdkit_idx in hba_set)
        q_atom = float(lig_q[rdkit_idx])
        h_atom = float(lig_h[rdkit_idx])
        arom_a = float(atom.GetIsAromatic())

        for j in range(N_res):
            rh = res_heavy[j]           # [K_j, 3]
            # min distance from this atom to residue heavy atoms
            diffs     = rh - atom_xyz[np.newaxis, :]  # [K_j, 3]
            dists     = np.linalg.norm(diffs, axis=1)
            min_dist  = float(dists.min()) if len(dists) > 0 else 999.0

            if min_dist > CONTACT_CUTOFF_A:
                continue

            # edge features
            dist_norm = min_dist / CONTACT_CUTOFF_A
            hb_compl  = float(
                (hbd_a > 0.5 and res_acceptor[j] > 0.5) or
                (hba_a > 0.5 and res_donor[j] > 0.5)
            )
            q_res    = float(res_charge[j])
            q_prod   = float(np.clip(q_atom * q_res, -1.0, 1.0))
            h_prod   = float(h_atom * res_hydrophob[j])
            is_clash = float(min_dist < VDW_CLASH_A)

            efeat = [
                dist_norm,             # 1  distance normalised
                hbd_a,                 # 1  HBD atom
                hba_a,                 # 1  HBA atom
                hb_compl,             # 1  complementarity
                q_prod,                # 1  charge product
                h_prod,                # 1  hydrophob product
                arom_a,               # 1  aromatic contact
                is_clash,             # 1  VdW close contact
            ]
            atom_indices.append(i)
            res_indices.append(j)
            edge_rows.append(efeat)

    if atom_indices:
        edge_index = np.array([atom_indices, res_indices], dtype=np.int64)
        edge_feat  = np.array(edge_rows, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_feat  = np.zeros((0, BIPARTITE_EDGE_DIM), dtype=np.float32)

    return edge_index, edge_feat
