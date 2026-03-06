"""
Graph_model.data.transfer
===========================
Tier 2 data loader — PDBbind v2020R1 general set for GNN pre-training.

The PDBbind dataset provides ~19,037 protein-ligand complexes with
experimentally measured binding affinities (Kd / Ki / IC50), converted
to ΔG via ΔG = RT · ln(Kd).  Pre-training on this diverse set prevents
the GNN from overfitting to the 9-ligand anchor set.

Data location
-------------
    Graph_model/external_dataset/
        index/INDEX_general_PL.2020R1.lst     ← affinity index
        P-L/<year-range>/<pdb_code>/          ← structure files
            <pdb_code>_ligand.sdf
            <pdb_code>_ligand.mol2
            <pdb_code>_pocket.pdb
            <pdb_code>_protein.pdb

Currently 385 entries have structure files (in P-L/1981-2000/).
SMILES are resolved from the SDF files via RDKit.

Record schema (returned by to_records)
--------------------------------------
{
    "sample_id":    str,   # PDB ID (4-char)
    "smiles":       str,   # from SDF file (when available)
    "ligand_name":  str,   # ligand HET code
    "delta_g":      float, # kcal/mol  (from Kd/Ki via ΔG = RT·ln(Kd))
    "pH":           float, # NaN (not known for PDBbind; masked during training)
    "temperature_C": float,# NaN
    "docking_box":  str,   # "pdbbind_generic"
    "receptor":     str,   # "pdbbind"
    "tier":         str,   # "transfer"
}
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import PDBBIND_ROOT, PDBBIND_INDEX, PDBBIND_PL_DIR

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
_R  = 1.987e-3   # kcal mol⁻¹ K⁻¹
_T  = 298.15     # 25 °C in Kelvin → standard PDBbind condition
_RT = _R * _T    # ≈ 0.5921 kcal/mol

# ── Affinity parser ───────────────────────────────────────────────────────────
_UNIT_TO_M: dict[str, float] = {
    'fM': 1e-15, 'pM': 1e-12, 'nM': 1e-9,
    'uM': 1e-6,  'mM': 1e-3,  'M': 1.0,
}

_BINDING_RE = re.compile(
    r'(Kd|Ki|IC50)\s*=\s*([~<>]?)\s*([\d.]+(?:e[+-]?\d+)?)\s*'
    r'(fM|pM|nM|uM|mM|M)\b',
    re.IGNORECASE,
)


def _parse_affinity_str(affinity_str: str) -> Optional[float]:
    """Parse 'Kd=49uM' → ΔG (kcal/mol). Returns None on failure."""
    m = _BINDING_RE.search(affinity_str)
    if m is None:
        return None
    inequality = m.group(2).strip()
    if inequality in ('<', '>', '~'):
        return None
    btype = m.group(1).upper()
    value_m = float(m.group(3)) * _UNIT_TO_M.get(m.group(4), 1.0)
    if value_m <= 0:
        return None
    if btype == 'IC50':
        value_m = value_m / 2.0
    return _RT * math.log(value_m)


def _find_entry_dir(pdb_code: str, pl_root: Path) -> Optional[Path]:
    """Find PDB entry directory under year-range sub-dirs."""
    for year_dir in pl_root.iterdir():
        if not year_dir.is_dir():
            continue
        entry = year_dir / pdb_code
        if entry.is_dir():
            return entry
    return None


def _smiles_from_sdf(sdf_path: Path) -> str:
    """Extract SMILES from a ligand SDF file via RDKit."""
    try:
        from rdkit import Chem
        mol = next(Chem.SDMolSupplier(str(sdf_path), removeHs=True, sanitize=True), None)
        if mol is not None:
            return Chem.MolToSmiles(mol)
    except Exception:
        pass
    return ""


class PDBbindLoader:
    """
    Lazy loader for PDBbind v2020R1 general set.

    Reads INDEX_general_PL.2020R1.lst and structure files from P-L/<year>/.

    Parameters
    ----------
    pdbbind_root : Path-like, optional
        Override default PDBBIND_ROOT from config.
    max_records  : int or None
        If set, only the first N records are returned (useful for debugging).
    resolve_smiles : bool, default True
        Read SMILES from ligand SDF files (slower but needed for models).
    """

    EXPECTED_RECORDS: int = 19_037

    def __init__(
        self,
        pdbbind_root:   Optional[Path] = None,
        max_records:    Optional[int]  = None,
        resolve_smiles: bool           = True,
    ) -> None:
        self.root           = Path(pdbbind_root or PDBBIND_ROOT)
        self.index_file     = PDBBIND_INDEX
        self.pl_dir         = PDBBIND_PL_DIR
        self.max_records    = max_records
        self.resolve_smiles = resolve_smiles
        self._df: pd.DataFrame | None = None

    # ── Availability check ─────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the index file exists and is readable."""
        return self.index_file.exists() and self.index_file.stat().st_size > 0

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        if not self.is_available():
            logger.warning(
                "PDBbind index not found at %s. "
                "Returning empty DataFrame; Tier-2 pre-training will be skipped.",
                self.index_file,
            )
            self._df = pd.DataFrame(columns=["pdb_id", "delta_g", "smiles",
                                              "ligand_name"])
            return self._df

        records = []
        with open(self.index_file, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "incomplete" in line.lower():
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue

                pdb_id = parts[0].lower()
                if len(pdb_id) != 4:
                    continue

                # Parse affinity string (e.g. "Kd=49uM")
                delta_g = _parse_affinity_str(parts[3])
                if delta_g is None:
                    continue

                # Extract ligand name from "// ref.pdf (HET)" or "(HET)"
                ligand_name = ""
                paren_match = re.search(r'\(([A-Za-z0-9]{2,5})\)', line)
                if paren_match:
                    ligand_name = paren_match.group(1)

                # Resolve SMILES from SDF if structure files exist
                smiles = ""
                entry_dir = None
                if self.resolve_smiles:
                    entry_dir = _find_entry_dir(pdb_id, self.pl_dir)
                    if entry_dir is not None:
                        sdf_path = entry_dir / f"{pdb_id}_ligand.sdf"
                        if sdf_path.exists():
                            smiles = _smiles_from_sdf(sdf_path)

                records.append({
                    "pdb_id":      pdb_id,
                    "delta_g":     delta_g,
                    "smiles":      smiles,
                    "ligand_name": ligand_name,
                    "has_structure": entry_dir is not None,
                })

                if self.max_records and len(records) >= self.max_records:
                    break

        self._df = pd.DataFrame(records)
        n_with_smiles = (self._df["smiles"].str.len() > 0).sum() if len(self._df) > 0 else 0
        n_with_struct = self._df["has_structure"].sum() if len(self._df) > 0 else 0
        logger.info(
            "PDBbind: loaded %d records (%d with SMILES, %d with structures) from %s",
            len(self._df), n_with_smiles, n_with_struct, self.index_file,
        )
        return self._df

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_dataframe(self) -> pd.DataFrame:
        return self._load()

    def to_records(self) -> List[dict]:
        """
        Return Tier-2 records in the universal anchor-compatible schema.
        Missing fields (pH, temperature) are encoded as NaN to be masked
        by the GNN's condition head.
        """
        df = self._load()
        records: List[dict] = []
        for _, row in df.iterrows():
            smiles = str(row.get("smiles", "")).strip()
            if not smiles or smiles.lower() in ("nan", "//", ""):
                continue
            records.append({
                "sample_id":            str(row.get("pdb_id", "")),
                "smiles":               smiles,
                "ligand_name":          str(row.get("ligand_name", "")),
                "delta_g":              float(row.get("delta_g", math.nan)),
                "pH":                   math.nan,      # not known
                "temperature_C":        math.nan,
                "docking_box":          "pdbbind_generic",
                "receptor":             "pdbbind",
                "n_interactions":       0,
                "interacting_residues": "",
                "tier":                 "transfer",
            })
        return records

    def is_empty(self) -> bool:
        return len(self._load()) == 0


# ── Legacy parser removed — affinity parsing is now inline in _load() ──────────
