"""
Graph_model.train.pdbbind_dataset
===================================
PyG-compatible dataset adapter for PDBbind v2020 general set.

Directory layout expected (from Graph_model/external_dataset/):
    P-L/
        <year-range>/
            <pdb_code>/
                <pdb_code>_ligand.sdf      ← RDKit-parseable ligand
                <pdb_code>_pocket.pdb      ← protein binding pocket
    index/
        INDEX_general_PL.2020R1.lst        ← affinity index

ΔG conversion
-------------
The index file gives Kd / Ki / IC50 values in mixed units.
We convert to ΔG (kcal/mol) via:
    ΔG = RT · ln(Kd)  at T = 298.15 K
       = 0.592 · ln(Kd_M)

Notes:
  - IC50 values are included with a 2× correction (IC50 ≈ 2·Kd at 50% inhibition)
    but flagged as lower-quality (``ic50_corrected=True``).
  - Entries with inequality signs (<, >, ~) are excluded (uncertain affinity).
  - Entries marked "incomplete ligand structure" are excluded.

Graph schema
------------
Same HeteroData schema as Phase 2 builder:
    data['ligand'].x / edge_index / edge_attr / batch
    data['residue'].x / edge_index / edge_attr / batch
    data['ligand','interacts','residue'].edge_index / edge_attr

Condition encoding for PDBbind records:
    ph_enc        = 0.02   (neutral pH 7.0, the standard PDBbind condition)
    temp_enc      = 0.64   (25°C normalised → (25−4)/33 ≈ 0.636)
    box_idx       = 0      (generic — no specific collagen box)
    receptor_flag = 0.0    (generic, not MMP-1)

Usage
-----
    >>> from Graph_model.train.pdbbind_dataset import PDBbindGraphDataset
    >>> ds = PDBbindGraphDataset()
    >>> print(f"{len(ds)} entries loaded")
    >>> data = ds[0]
    >>> data['ligand'].x.shape    # [N_atoms, LIGAND_NODE_DIM]
    >>> data.y                    # ΔG (kcal/mol)
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from pathlib import Path
from typing import Generator, List, Optional

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ── Physical constants ─────────────────────────────────────────────────────────
_R  = 1.987e-3   # kcal mol⁻¹ K⁻¹
_T  = 298.15
_RT = _R * _T    # ≈ 0.5921 kcal/mol

# ── Paths ─────────────────────────────────────────────────────────────────────
_EXT_ROOT   = Path(__file__).parents[1] / "external_dataset"
_PL_ROOT    = _EXT_ROOT / "P-L"
_INDEX_FILE = _EXT_ROOT / "index" / "INDEX_general_PL.2020R1.lst"

# ── Default condition vector for PDBbind (pH 7.0, 25°C, generic box) ──────────
_PDBBIND_PH_ENC   = 0.02    # neutral (from ConditionEncoder at pH=7.0)
_PDBBIND_TEMP_ENC = float((25 - 4) / 33)   # ≈ 0.6364
_PDBBIND_BOX_IDX  = 0       # generic binding site
_PDBBIND_REC_FLAG = 0.0     # not MMP-1


# ── Index parser ──────────────────────────────────────────────────────────────

_UNIT_TO_M: dict[str, float] = {
    'fM': 1e-15, 'pM': 1e-12, 'nM': 1e-9,
    'uM': 1e-6,  'mM': 1e-3,  'M': 1.0,
}

_BINDING_RE = re.compile(
    r'(Kd|Ki|IC50)\s*=\s*([~<>]?)\s*([\d.]+(?:e[+-]?\d+)?)\s*'
    r'(fM|pM|nM|uM|mM|M)\b',
    re.IGNORECASE,
)


def _parse_affinity(affinity_str: str) -> tuple[float, str] | tuple[None, None]:
    """
    Parse a PDBbind affinity string like 'Kd=49uM', 'Ki=0.43uM', 'IC50=1nM'.

    Returns (delta_g_kcal_per_mol, type_str)  or  (None, None) if unparseable.

    Inequalities (<, >, ~) are excluded (uncertain boundary conditions).
    IC50 values are converted via Kd ≈ IC50/2 (Cheng-Prusoff at 50% inhibition).
    """
    m = _BINDING_RE.search(affinity_str)
    if m is None:
        return None, None

    btype    = m.group(1).upper()     # Kd | Ki | IC50
    inequality = m.group(2).strip()
    value_str  = m.group(3)
    unit       = m.group(4)

    # Exclude inequality bounds
    if inequality in ('<', '>', '~'):
        return None, None

    value_m = float(value_str) * _UNIT_TO_M.get(unit, 1.0)
    if value_m <= 0:
        return None, None

    # IC50 correction: Kd_eff ≈ IC50 / 2
    if btype == 'IC50':
        value_m = value_m / 2.0

    # ΔG = RT · ln(Kd)
    delta_g = _RT * math.log(value_m)

    return delta_g, btype


def parse_pdbbind_index(
    index_file:   Path = _INDEX_FILE,
    exclude_ic50: bool = False,
) -> dict[str, float]:
    """
    Parse the PDBbind INDEX file and return {pdb_code: delta_g} dict.

    Parameters
    ----------
    index_file   : path to INDEX_general_PL.2020R1.lst (or similar)
    exclude_ic50 : if True, entries with IC50-derived ΔG are omitted

    Returns
    -------
    dict mapping 4-char PDB code (lower-case) → ΔG (kcal/mol)
    """
    result: dict[str, float] = {}
    n_skip = 0
    n_ineq = 0
    n_ok   = 0

    with open(index_file, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Skip entries flagged as incomplete ligand / peptide
            if 'incomplete' in line.lower():
                n_skip += 1
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            pdb_code = parts[0].lower()
            if len(pdb_code) != 4:
                continue

            # Find affinity column (Kd=... / Ki=... / IC50=...)
            affinity_part = parts[3]
            dg, btype = _parse_affinity(affinity_part)

            if dg is None:
                n_ineq += 1
                continue
            if exclude_ic50 and btype == 'IC50':
                n_skip += 1
                continue

            result[pdb_code] = dg
            n_ok += 1

    logger.info(
        "parse_pdbbind_index: %d valid | %d inequalities skipped | %d other skipped",
        n_ok, n_ineq, n_skip
    )
    return result


# ── PDB-entry locator ─────────────────────────────────────────────────────────

def _find_entry_dir(pdb_code: str, pl_root: Path = _PL_ROOT) -> Optional[Path]:
    """
    Find the directory containing <pdb_code>_ligand.sdf.
    Searches all year-range sub-directories under pl_root.
    Returns None if not found.
    """
    for year_dir in pl_root.iterdir():
        if not year_dir.is_dir():
            continue
        entry = year_dir / pdb_code
        if entry.is_dir():
            return entry
    return None


# ── Graph construction helper ──────────────────────────────────────────────────

def _build_heterodata(
    pdb_code:  str,
    entry_dir: Path,
    delta_g:   float,
) -> Optional[HeteroData]:
    """
    Build a HeteroData object from one PDBbind P-L entry.
    Returns None if the ligand SDF or pocket PDB cannot be parsed.
    """
    from rdkit import Chem

    sdf_path    = entry_dir / f"{pdb_code}_ligand.sdf"
    pocket_path = entry_dir / f"{pdb_code}_pocket.pdb"

    if not sdf_path.exists() or not pocket_path.exists():
        return None

    # ── Ligand graph ──────────────────────────────────────────────────────────
    try:
        from Graph_model.graph.level1_ligand import mol_to_ligand_graph

        mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True), None)
        if mol is None:
            return None

        smiles   = Chem.MolToSmiles(mol)
        lig_node_feat, lig_edge_index, lig_edge_feat = mol_to_ligand_graph(mol)
    except Exception as exc:
        logger.debug("Ligand parse failed for %s: %s", pdb_code, exc)
        return None

    # ── Protein pocket graph ──────────────────────────────────────────────────
    prot_node_feat  = None
    prot_edge_index = None
    prot_edge_attr  = None
    try:
        from Graph_model.graph.level2_protein import BoxProteinGraph
        # Pocket PDB is a pre-extracted binding-site file → BoxProteinGraph
        # uses all residues when len(residues) <= MAX_RESIDUES.
        # Pass a large box to ensure nothing is clipped.
        builder = BoxProteinGraph(pocket_path)
        result  = builder.get(box_center=(0.0, 0.0, 0.0), box_size=10000.0)
        # Returns (node_feat, edge_index, edge_feat, resnames, ca_coords)
        prot_node_feat, prot_edge_index, prot_edge_attr = result[0], result[1], result[2]
    except Exception as exc:
        logger.debug("Pocket parse failed for %s: %s", pdb_code, exc)

    # ── Assemble HeteroData ───────────────────────────────────────────────────
    data = HeteroData()

    # Ligand
    data['ligand'].x          = torch.from_numpy(lig_node_feat)
    data['ligand', 'bond', 'ligand'].edge_index = torch.from_numpy(lig_edge_index).long()
    data['ligand', 'bond', 'ligand'].edge_attr  = torch.from_numpy(lig_edge_feat)

    # Residue (protein pocket)
    if prot_node_feat is not None and prot_node_feat.shape[0] > 0:
        data['residue'].x = torch.from_numpy(prot_node_feat)
        data['residue', 'contact', 'residue'].edge_index = torch.from_numpy(prot_edge_index).long()
        data['residue', 'contact', 'residue'].edge_attr  = torch.from_numpy(prot_edge_attr)
    else:
        from Graph_model.model import PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM
        data['residue'].x = torch.zeros((0, PROTEIN_NODE_DIM), dtype=torch.float32)
        data['residue', 'contact', 'residue'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['residue', 'contact', 'residue'].edge_attr  = torch.zeros((0, PROTEIN_EDGE_DIM), dtype=torch.float32)

    # Bipartite (empty for PDBbind — no Vinardo-derived proximity)
    from Graph_model.model import BIPARTITE_EDGE_DIM
    data['ligand', 'interacts', 'residue'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data['ligand', 'interacts', 'residue'].edge_attr  = torch.zeros((0, BIPARTITE_EDGE_DIM), dtype=torch.float32)

    # ── Metadata — exact same schema as anchor HeteroData ────────────────────
    # (from ThreeLevelGraphBuilder.build() + HeteroDockingDataset._row_to_heterodata)
    data.y           = torch.tensor([[delta_g]], dtype=torch.float32)
    data.delta_g     = torch.tensor(delta_g,     dtype=torch.float32)
    data.ph          = torch.tensor(7.0,         dtype=torch.float32)   # standard PDBbind
    data.temp_c      = torch.tensor(25.0,        dtype=torch.float32)   # 25°C
    data.smiles      = smiles                                            # top-level str
    data.sample_id   = pdb_code
    data.resnames    = []                                                # no specific residues
    data.docking_box = "pdbbind_generic"
    data.receptor    = "pdbbind"
    data.ligand_name = pdb_code
    data.tier        = 1                                                 # int, matching anchor (0)

    return data


# ── Dataset class ──────────────────────────────────────────────────────────────

class PDBbindGraphDataset:
    """
    Lazy-loading PyG-compatible dataset for PDBbind general set.

    Parameters
    ----------
    pl_root      : Path to the P-L/ directory.  Defaults to external_dataset/P-L/
    index_file   : Path to the affinity index.  Defaults to external_dataset/index/INDEX_general_PL.2020R1.lst
    exclude_ic50 : bool, default False
    max_entries  : int or None — cap on dataset size (for smoke tests / dev)
    """

    def __init__(
        self,
        pl_root:      Path = _PL_ROOT,
        index_file:   Path = _INDEX_FILE,
        exclude_ic50: bool = False,
        max_entries:  Optional[int] = None,
    ) -> None:
        self.pl_root      = Path(pl_root)
        self.index_file   = Path(index_file)
        self.exclude_ic50 = exclude_ic50
        self.max_entries  = max_entries
        self._data_list:  Optional[list[HeteroData]] = None
        self._affinity:   Optional[dict[str, float]] = None

    # ── Index loading ──────────────────────────────────────────────────────────

    def get_affinity_index(self) -> dict[str, float]:
        """Parse the index file (cached after first call)."""
        if self._affinity is None:
            if not self.index_file.exists():
                raise FileNotFoundError(
                    f"PDBbind index not found: {self.index_file}\n"
                    f"Expected location: {_INDEX_FILE}"
                )
            self._affinity = parse_pdbbind_index(self.index_file, self.exclude_ic50)
        return self._affinity

    # ── Dataset access ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Build all HeteroData objects eagerly.
        Call once; subsequent access uses the cached list.
        """
        if self._data_list is not None:
            return

        # Suppress noisy RDKit "2D tagged as 3D" warnings during bulk loading
        try:
            from rdkit import RDLogger
            RDLogger.logger().setLevel(RDLogger.ERROR)
        except Exception:
            pass

        affinity = self.get_affinity_index()
        self._data_list = []
        n_no_structure = 0
        n_parse_err    = 0

        for pdb_code, delta_g in affinity.items():
            if self.max_entries is not None and len(self._data_list) >= self.max_entries:
                break
            entry_dir = _find_entry_dir(pdb_code, self.pl_root)
            if entry_dir is None:
                n_no_structure += 1
                continue
            d = _build_heterodata(pdb_code, entry_dir, delta_g)
            if d is None:
                n_parse_err += 1
                continue
            self._data_list.append(d)

        # Restore RDKit logging
        try:
            from rdkit import RDLogger
            RDLogger.logger().setLevel(RDLogger.WARNING)
        except Exception:
            pass

        n_index = len(affinity)
        logger.info(
            "PDBbindGraphDataset: %d graphs built from %d index entries "
            "(%d without local structures, %d parse errors)",
            len(self._data_list), n_index, n_no_structure, n_parse_err,
        )

    def __len__(self) -> int:
        if self._data_list is None:
            self.load()
        return len(self._data_list)        # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> HeteroData:
        if self._data_list is None:
            self.load()
        return self._data_list[idx]        # type: ignore[index]

    def __iter__(self) -> Generator[HeteroData, None, None]:
        if self._data_list is None:
            self.load()
        yield from self._data_list         # type: ignore[union-attr]

    def is_available(self) -> bool:
        """Quick check that the index file and at least one entry folder exist."""
        return self.index_file.exists() and self.pl_root.exists()

    def delta_g_stats(self) -> dict[str, float]:
        """Return basic statistics of ΔG values in the loaded dataset."""
        import numpy as np
        if self._data_list is None:
            self.load()
        vals = [float(d.y.item()) for d in self._data_list]   # type: ignore
        arr  = np.array(vals)
        return {
            'n':    len(arr),
            'mean': float(arr.mean()),
            'std':  float(arr.std()),
            'min':  float(arr.min()),
            'max':  float(arr.max()),
        }
