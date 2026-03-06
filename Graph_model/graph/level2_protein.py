"""
Graph_model.graph.level2_protein
===================================
Level 2: Protein Binding Site Context Graph

For each of the 76 docking boxes, build a residue-level undirected graph:
  Nodes  = residues whose Cα lies within (box_size/2 + MARGIN) Å of box centre
  Edges  = spatial contact: Cα–Cα distance < CONTACT_CUTOFF_A

Node features  —  30 dimensions
---------------------------------
Segment                     Dims  Range
─────────────────────────── ──── ──────────────────────────────────────────────
AA type one-hot             20   {0,1}
pKa_normed                   1   [0,1]  (pKa−3)/12
charge_at_pH                 1   [−1,1] Henderson–Hasselbalch
hydrophobicity               1   [0,1]  Kyte–Doolittle normalised
phi (sin)                    1   [−1,1]
phi (cos)                    1   [−1,1]
psi (sin)                    1   [−1,1]
psi (cos)                    1   [−1,1]
has_backbone                 1   {0,1}  (N+CA+C all present)
is_hbond_donor               1   {0,1}
is_hbond_acceptor            1   {0,1}
─────────────────────────── ────
PROTEIN_NODE_DIM            30

Edge features  —  4 dimensions
---------------------------------
ca_ca_dist_norm              1   Cα–Cα / CONTACT_CUTOFF_A   ∈ [0,1]
seq_dist_norm                1   |i−j| / 20 clipped to [0,1]
same_chain                   1   both residues on same chain
contact_type_one_hot         2   (seq_neighbor, spatial_only)
─────────────────────────── ────
PROTEIN_EDGE_DIM              4

BoxProteinGraph (cached)
    .get(box_label, box_center, box_size, ph) → (node_feat, edge_index, edge_feat,
                                                   residue_names, ca_coords)

Usage
-----
>>> parser = BoxProteinGraph(pdb_path)
>>> nf, ei, ea, resnames, ca = parser.get("GLU_cluster22", center, 20.0, ph=5.0)
>>> print(nf.shape)   # [N_res, 30]
"""

from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .residue_data import (
    aa_one_hot,
    pka_normed,
    charge_at_ph,
    hydrophobicity,
    is_hbond_donor,
    is_hbond_acceptor,
)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTACT_CUTOFF_A: float = 8.0     # Cα–Cα contact threshold (Å)
BOX_MARGIN_A:     float = 3.0     # extra radius beyond box_size/2
MIN_RESIDUES:     int   = 2       # minimum to form a graph (else empty)
MAX_RESIDUES:     int   = 30      # cap to prevent very large subgraphs

PROTEIN_NODE_DIM: int = 30
PROTEIN_EDGE_DIM: int = 4

_BACKBONE_ATOMS = {"N", "CA", "C", "O"}


# ── PDB parsing (bio-python) ──────────────────────────────────────────────────

def _load_structure(pdb_path: str | Path):
    """Parse PDB with BioPython. Returns Bio.PDB.Structure or None."""
    try:
        from Bio.PDB import PDBParser
    except ImportError:
        raise ImportError(
            "biopython is required for Level 2 graphs.\n"
            "Install with:  pip install biopython"
        )
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", str(pdb_path))


# ── Dihedral angle helper ─────────────────────────────────────────────────────

def _dihedral(p0, p1, p2, p3) -> float:
    """
    Praxeolitic formula — compute dihedral angle (radians) from 4 Cα positions.
    Returns 0.0 on degenerate geometry.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= (np.linalg.norm(b1) + 1e-12)
    v  = b0 - np.dot(b0, b1) * b1
    w  = b2 - np.dot(b2, b1) * b1
    x  = np.dot(v, w)
    y  = np.dot(np.cross(b1, v), w)
    return math.atan2(y, x + 1e-12)


# ── Residue atom collector ────────────────────────────────────────────────────

def _backbone_atoms(residue) -> dict[str, np.ndarray]:
    """Return dict name→coord for backbone atoms (N, CA, C, O)."""
    out = {}
    for atom in residue.get_atoms():
        name = atom.get_name().strip()
        if name in _BACKBONE_ATOMS:
            out[name] = atom.get_vector().get_array()
    return out


def _ca_coord(residue) -> Optional[np.ndarray]:
    """Return Cα coordinate or None."""
    for atom in residue.get_atoms():
        if atom.get_name().strip() == "CA":
            return atom.get_vector().get_array()
    return None


# ── Backbone angle computation ────────────────────────────────────────────────

def _compute_backbone_angles(residues: list, indices: dict) -> dict[int, tuple[float, float]]:
    """
    Compute φ/ψ for each residue (by idx in residues list).
    Returns dict: residue_idx → (phi_rad, psi_rad), default (0, 0) for termini.
    """
    n = len(residues)
    backbone = [_backbone_atoms(r) for r in residues]
    angles: dict[int, tuple[float, float]] = {}

    for i in range(n):
        phi = psi = 0.0
        bb  = backbone[i]
        has_all = ("N" in bb and "CA" in bb and "C" in bb)

        if has_all and i > 0:
            bb_prev = backbone[i - 1]
            if "C" in bb_prev:
                phi = _dihedral(
                    bb_prev["C"], bb["N"], bb["CA"], bb["C"]
                )

        if has_all and i < n - 1:
            bb_next = backbone[i + 1]
            if "N" in bb_next:
                psi = _dihedral(
                    bb["N"], bb["CA"], bb["C"], bb_next["N"]
                )

        angles[i] = (phi, psi)

    return angles


# ── Main graph builder ────────────────────────────────────────────────────────

class BoxProteinGraph:
    """
    Build a residue contact graph for a given docking box.

    Parameters
    ----------
    pdb_path : str | Path
        Pre-processed, H-fixed PDB for a specific receptor / pH
        (e.g. pig_collagen_I_alpha2_pH5_H_fix_bindingsite_clean.pdb)
    ph : float
        pH used to compute protonation-dependent features.
    """

    def __init__(self, pdb_path: str | Path, ph: float = 7.0) -> None:
        self.pdb_path = Path(pdb_path)
        self.ph       = ph
        self._structure = None   # lazy-loaded

    @property
    def structure(self):
        if self._structure is None:
            self._structure = _load_structure(self.pdb_path)
        return self._structure

    # ── Public ────────────────────────────────────────────────────────────────

    def get(
        self,
        box_center: tuple[float, float, float],
        box_size:   float,
        target_residues: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
        """
        Build the residue context graph for one docking box.

        Parameters
        ----------
        box_center       : (x, y, z) in Å
        box_size         : edge length of the cubic docking box in Å
        target_residues  : optional list from 'target_residues' CSV column
                           (e.g. ['GLU27A', 'LYS723A']).  When provided,
                           these residues are always included in the graph.

        Returns
        -------
        node_feat   : float32  [N_res, PROTEIN_NODE_DIM]
        edge_index  : int64    [2, N_edges]  (undirected, duplicated)
        edge_feat   : float32  [N_edges, PROTEIN_EDGE_DIM]
        resnames    : List[str]  e.g. ['GLU27A', 'PRO28A', ...]
        ca_coords   : float32  [N_res, 3]  Cα coordinates
        """
        center = np.array(box_center, dtype=np.float64)
        radius = box_size / 2.0 + BOX_MARGIN_A

        # Collect all residues first; use all if PDB is a small binding-site extract
        all_res = list(self._all_residues())
        if len(all_res) <= MAX_RESIDUES:
            # Pre-extracted binding-site PDB — use every residue directly
            residues = all_res
            resids   = [r.get_id() for r in residues]
        else:
            residues, resids = self._residues_in_radius(center, radius)

        # Optionally force-include target residues even if slightly outside
        if target_residues:
            full_ids = {self._res_full_id(r) for r in residues}
            for r in self._all_residues():
                fid = self._res_full_id(r)
                # match "GLU27A"-style string
                label = self._res_label(r)
                if any(t.strip() in (label, fid) for t in target_residues):
                    if fid not in full_ids:
                        residues.append(r)
                        full_ids.add(fid)

        # Cap size
        residues = residues[:MAX_RESIDUES]
        n = len(residues)

        if n < MIN_RESIDUES:
            return (
                np.zeros((0, PROTEIN_NODE_DIM), dtype=np.float32),
                np.zeros((2, 0), dtype=np.int64),
                np.zeros((0, PROTEIN_EDGE_DIM), dtype=np.float32),
                [],
                np.zeros((0, 3), dtype=np.float32),
            )

        # Collect Cα coords
        ca_list = []
        for r in residues:
            ca = _ca_coord(r)
            ca_list.append(ca if ca is not None else center)
        ca_coords = np.array(ca_list, dtype=np.float32)

        # Backbone angles
        angles = _compute_backbone_angles(residues, {})

        # Node features
        node_rows = []
        resnames  = []
        for i, res in enumerate(residues):
            resname = res.get_resname().strip()
            resnames.append(self._res_label(res))
            bb = _backbone_atoms(res)
            has_bb = float("N" in bb and "CA" in bb and "C" in bb)
            phi, psi = angles.get(i, (0.0, 0.0))

            feat = (
                aa_one_hot(resname)                   +  # 20
                [
                    pka_normed(resname),               #  1  pKa norm
                    charge_at_ph(resname, self.ph),    #  1  charge
                    hydrophobicity(resname),            #  1  KD hydrophob
                    math.sin(phi),                     #  1  φ sin
                    math.cos(phi),                     #  1  φ cos
                    math.sin(psi),                     #  1  ψ sin
                    math.cos(psi),                     #  1  ψ cos
                    has_bb,                            #  1  backbone complete
                    is_hbond_donor(resname),           #  1  HBD
                    is_hbond_acceptor(resname),        #  1  HBA
                ]                                       # = 30
            )
            node_rows.append(feat)

        node_feat = np.array(node_rows, dtype=np.float32)
        assert node_feat.shape[1] == PROTEIN_NODE_DIM

        # Edge construction: Cα–Cα < CONTACT_CUTOFF_A
        rows, cols, edge_rows = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                dist = float(np.linalg.norm(ca_coords[i] - ca_coords[j]))
                if dist < CONTACT_CUTOFF_A:
                    seq_d = abs(i - j)
                    dist_norm   = dist / CONTACT_CUTOFF_A
                    seq_norm    = min(seq_d / 20.0, 1.0)
                    same_chain  = float(
                        residues[i].get_parent().id ==
                        residues[j].get_parent().id
                    )
                    is_seq_nb   = float(seq_d == 1)
                    is_spatial  = float(seq_d > 1)
                    efeat = [dist_norm, seq_norm, same_chain,
                             is_seq_nb]  # 4 dims

                    for a, b in [(i, j), (j, i)]:
                        rows.append(a); cols.append(b)
                        edge_rows.append(efeat)

        if rows:
            edge_index = np.array([rows, cols], dtype=np.int64)
            edge_feat  = np.array(edge_rows,   dtype=np.float32)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_feat  = np.zeros((0, PROTEIN_EDGE_DIM), dtype=np.float32)

        return node_feat, edge_index, edge_feat, resnames, ca_coords

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _all_residues(self):
        """Iterate all residues in the structure (skip HETATM with no Cα)."""
        for model in self.structure:
            for chain in model:
                for res in chain:
                    if _ca_coord(res) is not None:
                        yield res

    def _residues_in_radius(
        self, center: np.ndarray, radius: float
    ) -> tuple[list, list]:
        """Return residues whose Cα lies within radius Å of center."""
        found, ids = [], []
        for res in self._all_residues():
            ca = _ca_coord(res)
            if ca is not None and np.linalg.norm(ca - center) <= radius:
                found.append(res)
                ids.append(res.get_id())
        return found, ids

    @staticmethod
    def _res_label(residue) -> str:
        """Produce label like 'GLU27A'."""
        resname  = residue.get_resname().strip()
        seq_num  = residue.get_id()[1]
        chain_id = residue.get_parent().id
        return f"{resname}{seq_num}{chain_id}"

    @staticmethod
    def _res_full_id(residue) -> str:
        return str(residue.get_full_id())


# ── PDB path selector ─────────────────────────────────────────────────────────

def pdb_for_ph(base_dir: str | Path, ph: float, receptor: str = "collagen") -> Path:
    """
    Return the appropriate PDB file path for the given pH.

    Available pH variants: 5.0, 5.5, 7.0
    Falls back to pH 7.0 if exact match not found.
    """
    base = Path(base_dir)
    if receptor == "mmp1":
        candidates = [
            base / "porcine_MMP1_966C.pdb",
            base / f"porcine_MMP1_pH{ph}_H.pdb",
        ]
        for p in candidates:
            if p.exists():
                return p
        # any existing MMP-1 PDB
        matches = sorted(base.glob("porcine_MMP1*.pdb"))
        if matches:
            return matches[0]
    else:
        # collagen — use pH-specific protonated PDB
        ph_str_map = {5.0: "pH5", 5.5: "pH5.5", 7.0: "pH7"}
        ph_str = ph_str_map.get(ph, "pH7")
        candidates = [
            base / f"pig_collagen_I_alpha2_{ph_str}_H_fix_bindingsite_clean.pdb",
            base / f"pig_collagen_I_alpha2_{ph_str}_H_fix_bindingsite.pdb",
            base / f"pig_collagen_I_alpha2_{ph_str}_H_fix.pdb",
        ]
        for p in candidates:
            if p.exists():
                return p
    # Last resort
    fallback = sorted(base.glob("pig_collagen*.pdb"))
    if fallback:
        return fallback[0]
    raise FileNotFoundError(
        f"No suitable PDB found in {base} for receptor={receptor}, pH={ph}"
    )
