"""
Graph_model.data.hetero_dataset
=================================
Full three-level HeteroData dataset for model training.

Reads the anchor CSV, builds HeteroData via ThreeLevelGraphBuilder,
and adds all metadata attributes needed by the model condition encoders.

Each HeteroData object has:
  - data['ligand'].x           : [N_atoms, 35]   ligand node features
  - data['ligand','bond','ligand'].edge_index/attr
  - data['residue'].x          : [N_res, 30]     protein residue features
  - data['residue','contact','residue'].edge_index/attr
  - data['ligand','interacts','residue'].edge_index/attr  (bipartite)
  - data.y                     : [1,1]  ΔG kcal/mol
  - data.ph, data.temp_c       : float scalars
  - data.docking_box, data.receptor, data.ligand_name : str metadata
  - data.tier                  : int (0=anchor)

Usage
-----
>>> from Graph_model.data.hetero_dataset import HeteroDockingDataset
>>> ds = HeteroDockingDataset()
>>> ds.load()
>>> print(len(ds))
>>> data = ds[0]
"""

from __future__ import annotations

import logging
import math
import pickle
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    _HAS_TORCH = False

from .config import (
    REPO_ROOT,
    ANCHOR_DIR,
    COLLAGEN_CSV,
    MMP1_CSV,
    PROCESSED_DIR,
    LIGAND_CATALOGUE,
)

try:
    from torch_geometric.data import HeteroData
    _HAS_PYG = True
except ImportError:
    HeteroData = object  # type: ignore
    _HAS_PYG = False

# Cache file
_CACHE_FILE = PROCESSED_DIR / "hetero_docking_dataset_cache.pkl"

# SMILES fallback
_CATALOGUE_SMILES = {k: v["smiles"] for k, v in LIGAND_CATALOGUE.items()}


class HeteroDockingDataset:
    """
    Three-level heterogeneous graph dataset for collagen-crosslinker docking.

    Builds HeteroData objects via ThreeLevelGraphBuilder from the full CSV
    (including box coordinates, target residues, and SDF poses).

    Parameters
    ----------
    include_mmp1 : bool
        Include MMP-1 docking records.
    include_bipartite : bool
        Include Level-3 bipartite (ligand↔residue) edges.
        Set False for faster building if you only need ligand+protein graphs.
    use_cache : bool
        Cache built HeteroData to disk for fast reload.
    force_reload : bool
        Ignore cache and rebuild from scratch.
    max_records : int, optional
        Limit number of records (for testing/debugging). None = all.
    """

    def __init__(
        self,
        include_mmp1: bool = True,
        include_bipartite: bool = True,
        use_cache: bool = True,
        force_reload: bool = False,
        max_records: Optional[int] = None,
    ) -> None:
        self.include_mmp1 = include_mmp1
        self.include_bipartite = include_bipartite
        self.use_cache = use_cache
        self.force_reload = force_reload
        self.max_records = max_records
        self._data_list: List = []
        self._loaded = False

    def load(self, verbose: bool = True) -> "HeteroDockingDataset":
        """Build HeteroData objects. Returns self for chaining."""
        if self._loaded:
            return self

        # Try cache first
        if self.use_cache and not self.force_reload and _CACHE_FILE.exists():
            logger.info("Loading HeteroData dataset from cache: %s", _CACHE_FILE)
            try:
                with open(_CACHE_FILE, "rb") as f:
                    self._data_list = pickle.load(f)
                self._loaded = True
                if verbose:
                    print(f"[HeteroDataset] Loaded {len(self._data_list)} graphs from cache.")
                return self
            except Exception as exc:
                logger.warning("Cache load failed (%s), rebuilding...", exc)

        # Build from scratch
        self._build(verbose=verbose)
        self._loaded = True

        # Save cache
        if self.use_cache and self._data_list:
            try:
                _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(_CACHE_FILE, "wb") as f:
                    pickle.dump(self._data_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                if verbose:
                    print(f"[HeteroDataset] Cached {len(self._data_list)} graphs → {_CACHE_FILE}")
            except Exception as exc:
                logger.warning("Cache save failed: %s", exc)

        return self

    def _build(self, verbose: bool = True) -> None:
        """Read CSV and build HeteroData via ThreeLevelGraphBuilder."""
        from ..graph.builder import ThreeLevelGraphBuilder

        # Read CSVs
        frames = []
        if COLLAGEN_CSV.exists():
            df_col = pd.read_csv(COLLAGEN_CSV)
            df_col["receptor"] = "collagen"
            frames.append(df_col)
            if verbose:
                print(f"[HeteroDataset] Collagen CSV: {len(df_col)} rows")
        else:
            raise FileNotFoundError(f"Collagen CSV not found: {COLLAGEN_CSV}")

        if self.include_mmp1 and MMP1_CSV.exists():
            df_mmp = pd.read_csv(MMP1_CSV)
            df_mmp["receptor"] = "mmp1"
            frames.append(df_mmp)
            if verbose:
                print(f"[HeteroDataset] MMP-1 CSV: {len(df_mmp)} rows")

        df = pd.concat(frames, ignore_index=True)

        if self.max_records is not None:
            df = df.head(self.max_records)

        # Builder — pdb_dir points to where PDB files live
        builder = ThreeLevelGraphBuilder(
            pdb_dir=ANCHOR_DIR,
            add_bipartite=self.include_bipartite,
        )

        failed = 0
        total = len(df)

        if verbose:
            try:
                from tqdm.auto import tqdm as tqdm_func
                iterator = tqdm_func(df.iterrows(), total=total, desc="Building graphs")
            except ImportError:
                iterator = df.iterrows()
        else:
            iterator = df.iterrows()

        for idx, row in iterator:
            try:
                data = self._row_to_heterodata(row, builder)
                if data is not None:
                    self._data_list.append(data)
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                if failed <= 5:
                    logger.debug("Row %d failed: %s", idx, exc)

        if verbose:
            print(f"[HeteroDataset] Built {len(self._data_list)} graphs "
                  f"({failed} failed, skipped).")

    def _row_to_heterodata(self, row: pd.Series, builder) -> Optional[HeteroData]:
        """Convert one CSV row to a HeteroData via ThreeLevelGraphBuilder."""
        smiles = str(row.get("ligand_smiles", "")).strip()
        ligand_name = str(row.get("ligand", "")).strip()

        # Resolve SMILES from catalogue if missing
        if not smiles or smiles.lower() == "nan":
            smiles = _CATALOGUE_SMILES.get(ligand_name, "")
        if not smiles:
            return None

        # Parse numeric fields
        ph = _safe_float(row.get("pH", 7.0), 7.0)
        temp_c = _safe_float(row.get("temperature_C", 25.0), 25.0)
        delta_g = _safe_float(row.get("best_energy_kcalmol", 0.0), 0.0)
        box_cx = _safe_float(row.get("box_center_x", 0.0), 0.0)
        box_cy = _safe_float(row.get("box_center_y", 0.0), 0.0)
        box_cz = _safe_float(row.get("box_center_z", 0.0), 0.0)
        box_size = _safe_float(row.get("box_size_A", 20.0), 20.0)
        docking_box = str(row.get("docking_box", "global_blind"))
        receptor = str(row.get("receptor", "collagen")).lower()
        sample_id = str(row.get("sample_id", ""))

        # Target residues
        target_res_raw = str(row.get("target_residues", ""))
        target_residues = target_res_raw if target_res_raw.lower() != "nan" else ""

        # SDF file — resolve to absolute path
        sdf_file_raw = str(row.get("sdf_file", "")).strip()
        if sdf_file_raw and sdf_file_raw.lower() != "nan":
            # CSV paths start with "collagen_gallic_results/..."
            # Resolve from Phukhao/ since that's the parent
            sdf_abs = REPO_ROOT / "Phukhao" / sdf_file_raw
            sdf_file = str(sdf_abs) if sdf_abs.exists() else ""
        else:
            sdf_file = ""

        # Build record for ThreeLevelGraphBuilder
        record = {
            "smiles":        smiles,
            "pH":            ph,
            "ph":            ph,
            "temperature_c": temp_c,
            "temperature_C": temp_c,
            "box_center_x":  box_cx,
            "box_center_y":  box_cy,
            "box_center_z":  box_cz,
            "box_size_A":    box_size,
            "target_residues": target_residues,
            "sdf_file":      sdf_file,
            "receptor":      receptor,
            "delta_g":       delta_g,
            "sample_id":     sample_id,
        }

        # Build HeteroData
        data = builder.build(record)

        # Add metadata attributes needed by models
        data.docking_box = docking_box
        data.receptor = receptor
        data.ligand_name = ligand_name
        data.tier = 0  # anchor

        return data

    # ── List-like interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx):
        if not self._loaded:
            self.load(verbose=False)
        return self._data_list[idx]

    def len(self) -> int:
        return len(self)

    def get(self, idx: int):
        return self[idx]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        from ..graph import LIGAND_NODE_DIM
        return LIGAND_NODE_DIM

    @property
    def num_edge_features(self) -> int:
        from ..graph import LIGAND_EDGE_DIM
        return LIGAND_EDGE_DIM

    # ── Convenience ───────────────────────────────────────────────────────────

    def ligand_names(self) -> List[str]:
        """Return unique ligand names in the dataset."""
        names = set()
        for d in self._data_list:
            ln = getattr(d, "ligand_name", None)
            if ln:
                names.add(ln)
        return sorted(names)

    def summary(self) -> dict:
        """Quick dataset summary."""
        by_ligand: dict[str, int] = {}
        for d in self._data_list:
            ln = getattr(d, "ligand_name", "unknown")
            by_ligand[ln] = by_ligand.get(ln, 0) + 1
        return {
            "total": len(self),
            "by_ligand": by_ligand,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val, default: float) -> float:
    """Safely convert to float, returning default on failure."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default
