"""
Graph_model.train.scaffold_split
===================================
Scaffold-based train/test splitting — Problem 6b.

Alternative to LOLO-CV: groups molecules by Murcko scaffold to test whether
the model generalises to unseen chemical scaffolds.

With only 9 unique ligands this produces fewer splits than traditional
scaffold splitting in large datasets, but it validates the molecular
representation quality: if the model performs well on held-out scaffolds,
the learned features capture more than scaffold identity.

Usage
-----
    >>> from Graph_model.train.scaffold_split import ScaffoldSplitter
    >>> splitter = ScaffoldSplitter(seed=42)
    >>> for fold in splitter.split(dataset):
    ...     print(fold.scaffold_group, fold.n_train, fold.n_test)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _get_murcko_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES (canonical). Falls back to full SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core, canonical=True)
    except Exception:
        return smiles


def _generic_scaffold(smiles: str) -> str:
    """Return generic Murcko scaffold (all atoms → C, all bonds → single)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        core = MurckoScaffold.GetScaffoldForMol(mol)
        generic = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(generic, canonical=True)
    except Exception:
        return smiles


@dataclass
class ScaffoldFold:
    """Single scaffold-split fold."""
    fold:             int
    scaffold_group:   str          # scaffold SMILES defining the held-out set
    held_out_ligands: List[str]    # ligand names in held-out scaffold group
    train_idx:        List[int]
    val_idx:          List[int]
    test_idx:         List[int]

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_val(self) -> int:
        return len(self.val_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)

    def __repr__(self) -> str:
        return (
            f"ScaffoldFold(fold={self.fold}, scaffold='{self.scaffold_group[:30]}', "
            f"held_out={self.held_out_ligands}, "
            f"train={self.n_train}, val={self.n_val}, test={self.n_test})"
        )


class ScaffoldSplitter:
    """
    Leave-One-Scaffold-Out cross-validation.

    Groups the 9 ligands by Murcko scaffold (or generic scaffold).
    Each fold holds out all ligands sharing one scaffold group.

    Parameters
    ----------
    val_ratio : float
        Fraction of non-held-out anchor records for validation (default 0.15).
    seed : int
        Randomness for train/val partition.
    use_generic : bool
        If True, use generic scaffold (all C, single bonds).
        Groups more aggressively — fewer, larger folds.
    """

    def __init__(
        self,
        val_ratio:   float = 0.15,
        seed:        int   = 42,
        use_generic: bool  = False,
    ) -> None:
        self.val_ratio   = val_ratio
        self.seed        = seed
        self.use_generic = use_generic

    def get_scaffold_groups(
        self,
        ligand_smiles: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """
        Map ligands to scaffold groups.

        Parameters
        ----------
        ligand_smiles : {ligand_name: SMILES}

        Returns
        -------
        {scaffold_smiles: [ligand_name, ...]}
        """
        scaffold_fn = _generic_scaffold if self.use_generic else _get_murcko_scaffold
        groups: Dict[str, List[str]] = defaultdict(list)
        for name, smi in ligand_smiles.items():
            scaf = scaffold_fn(smi)
            groups[scaf].append(name)
        logger.info(
            "ScaffoldSplitter: %d ligands → %d scaffold groups",
            len(ligand_smiles), len(groups),
        )
        for scaf, ligs in sorted(groups.items()):
            logger.debug("  scaffold '%s' → %s", scaf[:40], ligs)
        return dict(groups)

    def split(
        self,
        dataset,
        ligand_smiles: Optional[Dict[str, str]] = None,
    ) -> Iterator[ScaffoldFold]:
        """
        Yield ScaffoldFold objects, one per scaffold group.

        Parameters
        ----------
        dataset : indexable PyG dataset
            Each item must have .ligand_name (str), .tier (int).
        ligand_smiles : {name → SMILES}. If None, uses LIGAND_CATALOGUE.
        """
        if ligand_smiles is None:
            try:
                from Graph_model.data.config import LIGAND_CATALOGUE
                ligand_smiles = {k: v["smiles"] for k, v in LIGAND_CATALOGUE.items()}
            except ImportError:
                raise ValueError("Must provide ligand_smiles or install Graph_model.data.config")

        scaffold_groups = self.get_scaffold_groups(ligand_smiles)

        # Build ligand → indices map from dataset
        ligand_indices: Dict[str, List[int]] = defaultdict(list)
        non_anchor: List[int] = []
        for i in range(len(dataset)):
            d = dataset[i]
            tier = int(d.tier.item()) if hasattr(d.tier, "item") else int(d.tier)
            if tier != 0:
                non_anchor.append(i)
                continue
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            ligand_indices[name].append(i)

        # Yield one fold per scaffold group
        sorted_scaffolds = sorted(scaffold_groups.keys())
        for fold_idx, held_scaffold in enumerate(sorted_scaffolds):
            rng = random.Random(self.seed + fold_idx)
            held_ligands = scaffold_groups[held_scaffold]

            # Test: all records for held-out scaffold's ligands
            test_idx: List[int] = []
            for lig in held_ligands:
                test_idx.extend(ligand_indices.get(lig, []))
            rng.shuffle(test_idx)

            # Train/Val: remaining ligands
            train_idx: List[int] = []
            val_idx:   List[int] = []
            held_set = set(held_ligands)

            for scaf, scaf_ligands in scaffold_groups.items():
                if scaf == held_scaffold:
                    continue
                for lig in scaf_ligands:
                    idxs = list(ligand_indices.get(lig, []))
                    rng.shuffle(idxs)
                    n = len(idxs)
                    n_val = max(1, round(n * self.val_ratio))
                    if n <= 1:
                        train_idx.extend(idxs)
                    else:
                        val_idx.extend(idxs[:n_val])
                        train_idx.extend(idxs[n_val:])

            # Non-anchor → always train
            train_idx.extend(non_anchor)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)

            fold = ScaffoldFold(
                fold=fold_idx,
                scaffold_group=held_scaffold,
                held_out_ligands=held_ligands,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
            logger.debug("  %r", fold)
            yield fold

    @staticmethod
    def verify_no_leakage(fold: ScaffoldFold, dataset) -> bool:
        """
        Verify no scaffold leakage: test ligands must not appear in train/val.
        """
        held_set = set(fold.held_out_ligands)
        for idx in fold.train_idx:
            d = dataset[idx]
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            tier = int(d.tier.item()) if hasattr(d.tier, "item") else int(d.tier)
            if tier == 0:
                assert name not in held_set, \
                    f"SCAFFOLD LEAKAGE: {name} in train but should be held out"

        for idx in fold.val_idx:
            d = dataset[idx]
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            assert name not in held_set, \
                f"SCAFFOLD LEAKAGE: {name} in val but should be held out"

        return True
