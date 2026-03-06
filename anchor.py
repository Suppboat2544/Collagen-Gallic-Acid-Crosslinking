"""
Graph_model.data.anchor
=========================
Tier 1 data loader — reads the 6,156-row collagen-crosslinker docking CSV
(+ optional 40 MMP-1 records) into a unified list of record dicts.

Record schema
-------------
{
    "sample_id":          str,
    "smiles":             str,
    "ligand_name":        str,
    "delta_g":            float,   # best_energy_kcalmol
    "pH":                 float,
    "temperature_C":      int,
    "docking_box":        str,     # raw column value
    "receptor":           str,     # "collagen" or "mmp1"
    "n_interactions":     int,
    "interacting_residues": str,   # raw CSV string
    "tier":               str,     # always "anchor"
}

Usage
-----
>>> from Graph_model.data.anchor import AnchorLoader
>>> loader = AnchorLoader(include_mmp1=True)
>>> records = loader.to_records()
>>> len(records)
6196
>>> df = loader.get_dataframe()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from .config import (
    COLLAGEN_CSV,
    MMP1_CSV,
    LIGAND_CATALOGUE,
)

logger = logging.getLogger(__name__)

# ── Column name constants (from CSV header inspection) ────────────────────────
_COL_SAMPLE_ID     = "sample_id"
_COL_PH            = "pH"
_COL_LIGAND        = "ligand"
_COL_SMILES        = "ligand_smiles"
_COL_TEMP          = "temperature_C"
_COL_BOX           = "docking_box"
_COL_DELTA_G       = "best_energy_kcalmol"
_COL_N_INTERACT    = "n_interactions"
_COL_RESIDUES      = "interacting_residues"

# Fallback SMILES from catalogue when CSV field is empty/NaN
_CATALOGUE_SMILES: dict[str, str] = {
    name: info["smiles"] for name, info in LIGAND_CATALOGUE.items()
}


class AnchorLoader:
    """
    Load Tier-1 anchor records from on-disk CSV files.

    Parameters
    ----------
    include_mmp1 : bool, default True
        If True and MMP1_CSV exists, appends 40 MMP-1 records.
    drop_missing_smiles : bool, default True
        Drop rows where SMILES cannot be resolved (CSV + catalogue both empty).
    """

    def __init__(
        self,
        include_mmp1: bool = True,
        drop_missing_smiles: bool = True,
    ) -> None:
        self.include_mmp1       = include_mmp1
        self.drop_missing_smiles = drop_missing_smiles
        self._df: pd.DataFrame | None = None

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        collagen_path = Path(COLLAGEN_CSV)
        if not collagen_path.exists():
            raise FileNotFoundError(
                f"Collagen CSV not found at {collagen_path}. "
                "Run run_casein_docking.py or run_phukhao_docking_viz.py first."
            )

        df_col = pd.read_csv(collagen_path)
        df_col["receptor"] = "collagen"
        logger.info("Loaded %d collagen records from %s", len(df_col), collagen_path)

        frames = [df_col]

        if self.include_mmp1:
            mmp1_path = Path(MMP1_CSV)
            if mmp1_path.exists():
                df_mmp = pd.read_csv(mmp1_path)
                df_mmp["receptor"] = "mmp1"
                frames.append(df_mmp)
                logger.info("Loaded %d MMP-1 records from %s",
                            len(df_mmp), mmp1_path)
            else:
                logger.warning("MMP-1 CSV not found at %s — skipping.", mmp1_path)

        df = pd.concat(frames, ignore_index=True)

        # Resolve missing SMILES from catalogue
        if _COL_SMILES in df.columns:
            df[_COL_SMILES] = df.apply(_resolve_smiles, axis=1)
        else:
            df[_COL_SMILES] = df[_COL_LIGAND].map(_CATALOGUE_SMILES)

        # Drop rows with unresolvable SMILES
        if self.drop_missing_smiles:
            before = len(df)
            df = df.dropna(subset=[_COL_SMILES])
            df = df[df[_COL_SMILES].str.strip() != ""]
            dropped = before - len(df)
            if dropped:
                logger.warning("Dropped %d rows with missing SMILES.", dropped)

        # Coerce types
        df[_COL_PH]         = pd.to_numeric(df[_COL_PH],     errors="coerce")
        df[_COL_TEMP]       = pd.to_numeric(df[_COL_TEMP],   errors="coerce")
        df[_COL_DELTA_G]    = pd.to_numeric(df[_COL_DELTA_G],errors="coerce")
        df[_COL_N_INTERACT] = pd.to_numeric(df.get(_COL_N_INTERACT, 0),
                                             errors="coerce").fillna(0).astype(int)

        df["tier"] = "anchor"
        self._df = df
        return self._df

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_dataframe(self) -> pd.DataFrame:
        """Return the full merged DataFrame (lazy-loaded on first call)."""
        return self._load()

    def to_records(self) -> List[dict]:
        """
        Return all records as a list of lightweight dicts.

        Only the fields needed by CollagenDockingDataset are included.
        """
        df = self._load()
        records: List[dict] = []
        for _, row in df.iterrows():
            records.append({
                "sample_id":            str(row.get(_COL_SAMPLE_ID, "")),
                "smiles":               str(row[_COL_SMILES]),
                "ligand_name":          str(row.get(_COL_LIGAND, "")),
                "delta_g":              float(row[_COL_DELTA_G]),
                "pH":                   float(row[_COL_PH]),
                "temperature_C":        int(row[_COL_TEMP]),
                "docking_box":          str(row.get(_COL_BOX, "global_blind")),
                "receptor":             str(row.get("receptor", "collagen")),
                "n_interactions":       int(row.get(_COL_N_INTERACT, 0)),
                "interacting_residues": str(row.get(_COL_RESIDUES, "")),
                "tier":                 "anchor",
            })
        return records

    def summary(self) -> dict:
        """Print a quick data summary (ligand / pH / temp / box type counts)."""
        df = self._load()
        return {
            "total_records": len(df),
            "ligands": df[_COL_LIGAND].value_counts().to_dict(),
            "pH_counts": df[_COL_PH].value_counts().to_dict(),
            "temp_counts": df[_COL_TEMP].value_counts().to_dict(),
            "receptor_counts": df["receptor"].value_counts().to_dict(),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_smiles(row: pd.Series) -> str:
    """Fill SMILES from CSV; fall back to ligand catalogue if blank/NaN."""
    csv_val = str(row.get(_COL_SMILES, "")).strip()
    if csv_val and csv_val.lower() != "nan":
        return csv_val
    ligand_name = str(row.get(_COL_LIGAND, "")).strip()
    return _CATALOGUE_SMILES.get(ligand_name, "")
