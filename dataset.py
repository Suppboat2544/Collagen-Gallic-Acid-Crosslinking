"""
Graph_model.data.dataset
==========================
Three-tier PyTorch Geometric Dataset for collagen-crosslinker binding energy
prediction.

Each Data object has
--------------------
Attribute       Shape                   Description
─────────────── ─────────────────────── ────────────────────────────────────────
x               [N_atoms, 54]           atom feature matrix
edge_index      [2, 2·N_bonds]          undirected bond connectivity
edge_attr       [2·N_bonds, 12]         bond feature matrix
fragment_x      [N_frag, 6]             per-fragment summary features
fragment_map    [N_atoms]               atom→fragment index (-1 = not in frag)
cond            [4]                     [ph_enc, temp_enc, box_idx, receptor]
y               [1]                     ΔG  (kcal/mol)
ligand_name     str
sample_id       str
tier            int                     0=anchor, 1=transfer, 2=augment

Fragment features (fragment_x)  — 6 dims per fragment
    [galloyl_weighted, catechol_count, pyrogallol_count, n_oh, ring_size, frag_idx_norm]

Usage
-----
>>> from Graph_model.data.dataset import CollagenDockingDataset
>>> ds = CollagenDockingDataset(include_transfer=False, include_augment=False)
>>> ds.load()
>>> print(len(ds))        # 6196 anchor records (collagen + MMP-1)
>>> data = ds[0]
>>> data.x.shape          # torch.Size([N_atoms, 54])
>>> data.cond             # tensor([0.8500, 0.6364, 3.0000, 0.0000])
"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

import numpy as np
from rdkit import Chem

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    logger.warning("torch not installed; CollagenDockingDataset cannot build tensors.")
    torch = None          # type: ignore[assignment]
    _HAS_TORCH = False

from .anchor    import AnchorLoader
from .augment   import PhenolicAugmentLoader
from .config    import PROCESSED_DIR
from .features  import (
    ATOM_FEAT_DIM, BOND_FEAT_DIM,
    atom_features, GalloylFragmentDetector,
    ConditionEncoder,
)
from .features.atom import mol_to_edge_index_and_attr
from .transfer  import PDBbindLoader

# PyG import — optional at module level so the file can be imported for
# inspection even without torch_geometric installed.
try:
    from torch_geometric.data import Data, Dataset
    _HAS_PYG = True
except ImportError:
    logger.warning(
        "torch_geometric not installed; CollagenDockingDataset cannot be used "
        "for model training.  Install with:  pip install torch_geometric"
    )
    Data    = object
    Dataset = object
    _HAS_PYG = False

# Tier codes ──────────────────────────────────────────────────────────────────
TIER_CODE = {"anchor": 0, "transfer": 1, "augment": 2}

# Fragment feature dimension
FRAG_FEAT_DIM = 6

# Cache file name
_CACHE_FILE = PROCESSED_DIR / "collagen_docking_dataset_cache.pkl"


class CollagenDockingDataset:
    """
    Three-tier PyG-compatible dataset.

    This class acts as both a plain Python container (indexable list of
    torch_geometric.data.Data objects) and a thin wrapper exposing a
    PyG-like interface (.len() / .get() / .num_features).

    For full PyG Dataset compatibility (with automatic disk caching), use
    CollagenDockingDataset.as_pyg_dataset() (returns a PyGWrapper).

    Parameters
    ----------
    include_transfer : bool, default True
        Include PDBbind Tier-2 records (skipped silently if unavailable).
    include_augment  : bool, default True
        Include phenolic augmentation Tier-3 records.
    include_mmp1     : bool, default True
        Include MMP-1 docking records in the anchor tier.
    use_cache        : bool, default True
        Persist processed Data objects to disk (speeds up repeated loads).
    force_reload     : bool, default False
        Ignore existing cache and re-process everything.
    condition_encoder : ConditionEncoder, optional
        Custom encoder; defaults to ConditionEncoder(strict=False).
    """

    def __init__(
        self,
        include_transfer:  bool = True,
        include_augment:   bool = True,
        include_mmp1:      bool = True,
        use_cache:         bool = True,
        force_reload:      bool = False,
        condition_encoder: Optional[ConditionEncoder] = None,
    ) -> None:
        self.include_transfer  = include_transfer
        self.include_augment   = include_augment
        self.include_mmp1      = include_mmp1
        self.use_cache         = use_cache
        self.force_reload      = force_reload
        self.cond_enc          = condition_encoder or ConditionEncoder(strict=False)
        self._data_list: List[Data] = []
        self._loaded = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, verbose: bool = True) -> "CollagenDockingDataset":
        """
        Build Data objects from all tiers.  Returns self for chaining.

        If use_cache=True and a valid cache file exists, loads from disk.
        """
        if self._loaded:
            return self

        if self.use_cache and not self.force_reload and _CACHE_FILE.exists():
            logger.info("Loading dataset from cache: %s", _CACHE_FILE)
            with open(_CACHE_FILE, "rb") as f:
                self._data_list = pickle.load(f)
            self._loaded = True
            if verbose:
                print(f"[Dataset] Loaded {len(self._data_list)} records from cache.")
            return self

        records: List[dict] = []

        # Tier 1 — Anchor
        anchor = AnchorLoader(include_mmp1=self.include_mmp1)
        records += anchor.to_records()
        if verbose:
            print(f"[Dataset] Tier-1 anchor: {sum(1 for r in records if r['tier']=='anchor')} records")

        # Tier 2 — Transfer
        if self.include_transfer:
            transfer = PDBbindLoader()
            if transfer.is_available():
                t2 = transfer.to_records()
                records += t2
                if verbose:
                    print(f"[Dataset] Tier-2 transfer: {len(t2)} records")
            else:
                if verbose:
                    print("[Dataset] Tier-2 transfer: PDBbind not available, skipping.")

        # Tier 3 — Augment
        if self.include_augment:
            aug = PhenolicAugmentLoader()
            t3 = aug.to_records()
            records += t3
            if verbose:
                print(f"[Dataset] Tier-3 augment: {len(t3)} records")

        # Convert records → Data objects
        failed = 0
        for rec in records:
            data = self._record_to_data(rec)
            if data is not None:
                self._data_list.append(data)
            else:
                failed += 1

        if verbose:
            print(f"[Dataset] Processed {len(self._data_list)} graphs "
                  f"({failed} failed SMILES → skipped).")

        # Cache to disk
        if self.use_cache:
            with open(_CACHE_FILE, "wb") as f:
                pickle.dump(self._data_list, f)
            logger.info("Dataset cached to %s", _CACHE_FILE)

        self._loaded = True
        return self

    # ── Core converter ────────────────────────────────────────────────────────

    def _record_to_data(self, rec: dict) -> Optional[Data]:
        """Convert one record dict to a torch_geometric.data.Data object."""
        if not _HAS_PYG or not _HAS_TORCH:
            raise RuntimeError(
                "torch and torch_geometric are required to build Data tensors. "
                "Install with:  pip install torch torch_geometric"
            )

        smiles = rec.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("Could not parse SMILES for %s: %s",
                         rec.get("sample_id", "?"), smiles)
            return None

        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return None

        # ── Atom features ─────────────────────────────────────────────────────
        x = torch.tensor(atom_features(mol), dtype=torch.float)   # [N, 54]

        # ── Bond features ─────────────────────────────────────────────────────
        edge_index_np, edge_attr_np = mol_to_edge_index_and_attr(mol)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)   # [2, 2E]
        edge_attr  = torch.tensor(edge_attr_np,  dtype=torch.float)  # [2E, 12]

        # ── Galloyl fragment features ─────────────────────────────────────────
        counts  = GalloylFragmentDetector.count_fragments(mol)
        sublists = GalloylFragmentDetector.fragment_subgraph_nodes(mol)
        frag_map_np = _build_fragment_map(mol, sublists)
        fragment_map = torch.tensor(frag_map_np, dtype=torch.long)   # [N]

        fragment_x = _build_fragment_features(mol, sublists, counts)  # [F, 6]

        # ── Condition vector ──────────────────────────────────────────────────
        ph   = rec.get("pH", 7.0)
        temp = rec.get("temperature_C", 25.0)
        box  = rec.get("docking_box", "global_blind")
        rcpt = rec.get("receptor", "collagen")

        if isinstance(ph, float) and math.isnan(ph):
            ph = 7.0    # PDBbind: assume neutral pH
        if isinstance(temp, float) and math.isnan(temp):
            temp = 25.0

        cond_np = self.cond_enc.encode(ph, temp, box, rcpt)
        cond = torch.tensor(cond_np, dtype=torch.float)              # [4]

        # ── Label ─────────────────────────────────────────────────────────────
        dg = rec.get("delta_g", 0.0)
        if isinstance(dg, float) and math.isnan(dg):
            dg = 0.0
        y = torch.tensor([[float(dg)]], dtype=torch.float)            # [1, 1]

        # ── Tier code ─────────────────────────────────────────────────────────
        tier_code = TIER_CODE.get(rec.get("tier", "anchor"), 0)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            fragment_x=fragment_x,
            fragment_map=fragment_map,
            cond=cond,
            y=y,
            ligand_name=rec.get("ligand_name", ""),
            sample_id=rec.get("sample_id", ""),
            tier=tier_code,
        )
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

    def get(self, idx: int) -> Data:
        return self[idx]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        return ATOM_FEAT_DIM

    @property
    def num_edge_features(self) -> int:
        return BOND_FEAT_DIM

    @property
    def num_fragment_features(self) -> int:
        return FRAG_FEAT_DIM

    # ── Convenience ───────────────────────────────────────────────────────────

    def anchor_indices(self) -> List[int]:
        return [i for i, d in enumerate(self._data_list) if d.tier == 0]

    def transfer_indices(self) -> List[int]:
        return [i for i, d in enumerate(self._data_list) if d.tier == 1]

    def augment_indices(self) -> List[int]:
        return [i for i, d in enumerate(self._data_list) if d.tier == 2]

    def summary(self) -> dict:
        return {
            "total":    len(self),
            "anchor":   len(self.anchor_indices()),
            "transfer": len(self.transfer_indices()),
            "augment":  len(self.augment_indices()),
        }


# ── Fragment helpers ──────────────────────────────────────────────────────────

def _build_fragment_map(mol: Chem.Mol, sublists: list) -> np.ndarray:
    """Return per-atom fragment index array (−1 = not in any fragment)."""
    n = mol.GetNumAtoms()
    fmap = np.full(n, -1, dtype=np.int32)
    for fi, group in enumerate(sublists):
        for ai in group:
            if 0 <= ai < n:
                fmap[ai] = fi
    return fmap


def _build_fragment_features(
    mol: Chem.Mol,
    sublists: list,
    global_counts: dict,
) -> torch.Tensor:
    """
    Build fragment-level feature matrix.

    Each fragment (galloyl unit) gets a 6-dim vector:
        0: galloyl_weighted score from global_counts
        1: catechol subunit count
        2: pyrogallol subunit count
        3: # aromatic OH in this fragment
        4: ring size of the aromatic ring (6 for all galloyl units → norm to 1)
        5: fragment index, normalised by (max_frag − 1)  ∈ [0, 1]
    """
    if not sublists:
        return torch.zeros((0, FRAG_FEAT_DIM), dtype=torch.float)

    n_frags = len(sublists)
    feats = []
    for fi, group in enumerate(sublists):
        oh_count = sum(
            1 for ai in group
            if mol.GetAtomWithIdx(ai).GetSymbol() == "O"
            and mol.GetAtomWithIdx(ai).GetDegree() == 1
        )
        ring_size_norm = 1.0     # all detected units are 6-membered aromatic
        frag_idx_norm  = fi / max(n_frags - 1, 1)
        row = [
            global_counts.get("galloyl_weighted", 0.0) / max(n_frags, 1),
            float(global_counts.get("catechol", 0)),
            float(global_counts.get("pyrogallol", 0)),
            float(oh_count),
            ring_size_norm,
            frag_idx_norm,
        ]
        feats.append(row)

    return torch.tensor(feats, dtype=torch.float)
