"""
Graph_model.graph.builder
===========================
Assembles all three graph levels into a single PyTorch Geometric HeteroData.

Node types
----------
  'ligand'   — heavy atoms of the docked molecule
  'residue'  — amino-acid residues in the binding site

Edge types
----------
  ('ligand',  'bond',     'ligand' )  — covalent bonds (Level 1)
  ('residue', 'contact',  'residue')  — Cα–Cα spatial contacts (Level 2)
  ('ligand',  'interacts','residue')  — proximity edges (Level 3)

Labels / auxiliary tensors
--------------------------
  data.y         : float32  [1, 1]   binding ΔG (kcal/mol)
  data.delta_g   : float32  scalar
  data.ph        : float32  scalar
  data.temp_c    : float32  scalar
  data.smiles    : str
  data.sample_id : str

Usage
-----
>>> from Graph_model.graph.builder import ThreeLevelGraphBuilder
>>> builder = ThreeLevelGraphBuilder(pdb_dir="/path/to/Phukhao/collagen_gallic_results")
>>> from Graph_model.data.config import GallicDockingConfig
>>> cfg = GallicDockingConfig()
>>> data = builder.build(record)   # record is a dict from AnchorDataset.__getitem__
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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

try:
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError(
        "torch_geometric is required.\n"
        "Install:  pip install torch-geometric"
    )

# ── PDB directory structure ───────────────────────────────────────────────────
_COLLAGEN_KEY = "collagen"
_MMP1_KEY     = "mmp1"


class ThreeLevelGraphBuilder:
    """
    Builds a three-level heterogeneous graph from a docking record.

    Parameters
    ----------
    pdb_dir : str | Path
        Directory containing the pH-specific PDB files
        (e.g. Phukhao/collagen_gallic_results/).
    add_bipartite : bool
        Whether to include Level-3 bipartite edges (requires SDF file).
        Default True — set False for quick structure-only tests.
    """

    def __init__(
        self,
        pdb_dir:        str | Path,
        add_bipartite:  bool = True,
    ) -> None:
        self.pdb_dir       = Path(pdb_dir)
        self.add_bipartite = add_bipartite
        # BoxProteinGraph is cached per (pdb_path, ph)
        self._protein_graphs: dict[tuple, BoxProteinGraph] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, record: dict) -> HeteroData:
        """
        Build HeteroData from a single docking record.

        Expected record keys (matches AnchorDataset row schema):
            smiles           : SMILES string
            ph               : float
            temperature_c    : float
            docking_box      : str  e.g. "GLU_cluster22"
            box_center_x/y/z : float
            box_size_A       : float
            target_residues  : str  e.g. "GLU27A, GLU30A"
            sdf_file         : str  relative or absolute path to docked .sdf
            delta_g          : float  (may be NaN for test set)
            receptor         : str  'collagen' or 'mmp1'  (optional)
        """
        smiles      = record["smiles"]
        ph          = float(record.get("ph", record.get("pH", 7.0)))
        temp_c      = float(record.get("temperature_c", record.get("temperature_C", 25.0)))
        box_center  = (
            float(record["box_center_x"]),
            float(record["box_center_y"]),
            float(record["box_center_z"]),
        )
        box_size    = float(record.get("box_size_A", 20.0))
        target_res  = [r.strip() for r in str(record.get("target_residues", "")).split(",") if r.strip()]
        sdf_file    = record.get("sdf_file", "")
        receptor    = str(record.get("receptor", _COLLAGEN_KEY)).lower()
        delta_g     = float(record.get("delta_g", float("nan")))
        sample_id   = str(record.get("sample_id", record.get("id", "")))

        # ── Level 1: Ligand molecular graph ──────────────────────────────────
        try:
            lig_x, lig_ei, lig_ea = mol_to_ligand_graph(smiles)
        except Exception as exc:
            warnings.warn(f"Level 1 failed for smiles={smiles}: {exc}")
            lig_x  = np.zeros((1, LIGAND_NODE_DIM), dtype=np.float32)
            lig_ei = np.zeros((2, 0), dtype=np.int64)
            lig_ea = np.zeros((0, LIGAND_EDGE_DIM), dtype=np.float32)

        # ── Level 2: Protein binding site graph ──────────────────────────────
        box_graph = self._get_box_graph(ph, receptor)
        try:
            prot_x, prot_ei, prot_ea, resnames, ca_coords = box_graph.get(
                box_center=box_center,
                box_size=box_size,
                target_residues=target_res if target_res else None,
            )
        except Exception as exc:
            warnings.warn(f"Level 2 failed (pH={ph}, receptor={receptor}): {exc}")
            prot_x    = np.zeros((1, PROTEIN_NODE_DIM), dtype=np.float32)
            prot_ei   = np.zeros((2, 0), dtype=np.int64)
            prot_ea   = np.zeros((0, PROTEIN_EDGE_DIM), dtype=np.float32)
            resnames  = []
            ca_coords = np.zeros((1, 3), dtype=np.float32)

        # ── Level 3: Bipartite interaction graph ─────────────────────────────
        bip_ei  = np.zeros((2, 0), dtype=np.int64)
        bip_ea  = np.zeros((0, BIPARTITE_EDGE_DIM), dtype=np.float32)

        if self.add_bipartite and sdf_file:
            sdf_path = Path(sdf_file)
            if not sdf_path.is_absolute():
                sdf_path = self.pdb_dir / sdf_path
            if sdf_path.exists():
                try:
                    # Optionally pass the BioPython structure for better accuracy
                    structure = box_graph.structure if len(resnames) > 0 else None
                    bip_ei, bip_ea = build_bipartite_graph(
                        sdf_path      = sdf_path,
                        ca_coords     = ca_coords,
                        residue_names = resnames,
                        ph            = ph,
                        structure     = structure,
                    )
                except Exception as exc:
                    warnings.warn(f"Level 3 failed for {sdf_path}: {exc}")
            else:
                warnings.warn(f"SDF not found: {sdf_path}")

        # ── Assemble HeteroData ───────────────────────────────────────────────
        data = HeteroData()

        data["ligand"].x = torch.tensor(lig_x, dtype=torch.float32)
        data["residue"].x = torch.tensor(prot_x, dtype=torch.float32)

        data["ligand", "bond", "ligand"].edge_index  = torch.tensor(lig_ei,  dtype=torch.long)
        data["ligand", "bond", "ligand"].edge_attr   = torch.tensor(lig_ea,  dtype=torch.float32)

        data["residue", "contact", "residue"].edge_index = torch.tensor(prot_ei, dtype=torch.long)
        data["residue", "contact", "residue"].edge_attr  = torch.tensor(prot_ea, dtype=torch.float32)

        data["ligand", "interacts", "residue"].edge_index = torch.tensor(bip_ei, dtype=torch.long)
        data["ligand", "interacts", "residue"].edge_attr  = torch.tensor(bip_ea, dtype=torch.float32)

        # Labels & metadata
        data.y        = torch.tensor([[delta_g]], dtype=torch.float32)
        data.delta_g  = torch.tensor(delta_g,   dtype=torch.float32)
        data.ph       = torch.tensor(ph,         dtype=torch.float32)
        data.temp_c   = torch.tensor(temp_c,     dtype=torch.float32)
        data.smiles   = smiles
        data.sample_id = sample_id
        data.resnames = resnames   # non-tensor, for debugging

        return data

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_box_graph(self, ph: float, receptor: str) -> BoxProteinGraph:
        """Return a (cached) BoxProteinGraph for a given pH and receptor."""
        key = (ph, receptor)
        if key not in self._protein_graphs:
            pdb_path = pdb_for_ph(self.pdb_dir, ph, receptor)
            self._protein_graphs[key] = BoxProteinGraph(pdb_path, ph=ph)
        return self._protein_graphs[key]


# ── Dimension summary ─────────────────────────────────────────────────────────
GRAPH_DIM_SUMMARY = {
    "ligand_node_dim":       LIGAND_NODE_DIM,
    "ligand_edge_dim":       LIGAND_EDGE_DIM,
    "protein_node_dim":      PROTEIN_NODE_DIM,
    "protein_edge_dim":      PROTEIN_EDGE_DIM,
    "bipartite_edge_dim":    BIPARTITE_EDGE_DIM,
}


def print_dim_summary() -> None:
    """Print a summary of all graph tensor dimensions."""
    print("Three-Level Graph Dimensions")
    print("────────────────────────────")
    for k, v in GRAPH_DIM_SUMMARY.items():
        print(f"  {k:<26}  {v}")
