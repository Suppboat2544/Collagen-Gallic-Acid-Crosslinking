"""
Graph_model.data.features.galloyl
===================================
Fragment detection and subgraph labelling for galloyl / polyphenolic units.

Three SMARTS patterns are defined for the three phenolic core types present
in the nine-ligand set:

  Pattern                Ligands                         galloyl_units
  ─────────────────────  ──────────────────────────────  ─────────────
  trihydroxyphenyl (GA)  gallic_acid, pyrogallol, PGG    1 per match
  catechol (CA)          protocatechuic_acid              0 (partial)
  ellagyl (EL)           ellagic_acid                    2 (2 fused)

Workflow
--------
1.  GalloylFragmentDetector.count_fragments(mol)
        → {"galloyl_strict": int, "catechol": int, "pyrogallol": int,
            "total_aromatic_oh": int, "galloyl_weighted": float}

2.  GalloylFragmentDetector.fragment_subgraph_nodes(mol)
        → List[List[int]]  — atom-index sublists for each galloyl unit
        → Used to build a fragment-level node x_frag in the GNN.

atom_to_fragment_map(mol, sublists)
        → np.ndarray int  [N_atoms]  — index of enclosing fragment or -1
"""

from __future__ import annotations

from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# ── SMARTS definitions ────────────────────────────────────────────────────────

# 3,4,5-trihydroxyphenyl ring — core galloyl unit (gallic acid, PGG arms, pyrogallol)
_SMARTS_GALLOYL  = "Oc1cc(O)c(O)cc1"      # 3,4,5 pattern (symmetric)
# alternative pyrogallol orientation (1,2,3-OH in pyrogallol ring)
_SMARTS_PYROGALL = "Oc1cccc(O)c1O"
# 3,4-catechol — ortho-dihydroxyphenyl (protocatechuic acid, dopamine-like)
# Use `Oc1ccccc1O`: OHs on the two atoms joined by the ring-closure bond (adjacent)
_SMARTS_CATECHOL = "Oc1ccccc1O"
# Generic aromatic OH count (broad phenol)
_SMARTS_PHENOL   = "Oc1ccccc1"

# Compiled patterns (singleton per module)
_PAT_GALLOYL  = Chem.MolFromSmarts(_SMARTS_GALLOYL)
_PAT_PYROGALL = Chem.MolFromSmarts(_SMARTS_PYROGALL)
_PAT_CATECHOL = Chem.MolFromSmarts(_SMARTS_CATECHOL)
_PAT_PHENOL   = Chem.MolFromSmarts(_SMARTS_PHENOL)

# Weighted contribution to galloyl_weighted score
_GALLOYL_WEIGHTS = {
    "galloyl_strict": 1.00,   # full galloyl unit
    "pyrogallol":     1.00,   # isomeric but equivalent H-bonding capacity
    "catechol":       0.67,   # 2/3 of OH → partial contribution
}


class GalloylFragmentDetector:
    """
    Detect and label polyphenolic substructures in an RDKit molecule.

    Instantiate once; call methods on any molecule.
    """

    # ── Fragment counting ─────────────────────────────────────────────────────

    @staticmethod
    def count_fragments(mol: Chem.Mol) -> dict:
        """
        Count galloyl / catechol / pyrogallol fragment occurrences.

        Parameters
        ----------
        mol : rdkit.Chem.Mol

        Returns
        -------
        dict with keys:
            galloyl_strict   (int) — 3,4,5-triOH phenyl matches
            catechol         (int) — 3,4-diOH phenyl matches
            pyrogallol       (int) — 1,2,3-triOH phenyl matches (pyrogallol)
            total_aromatic_oh(int) — total aromatic OH groups
            galloyl_weighted (float) — weighted sum per _GALLOYL_WEIGHTS
        """
        if mol is None:
            return {k: 0 for k in
                    ("galloyl_strict", "catechol", "pyrogallol",
                     "total_aromatic_oh", "galloyl_weighted")}

        g  = len(mol.GetSubstructMatches(_PAT_GALLOYL))
        py = len(mol.GetSubstructMatches(_PAT_PYROGALL))
        ca = len(mol.GetSubstructMatches(_PAT_CATECHOL))
        # subtract catechol matches that overlap with galloyl/pyrogallol rings
        # to avoid double-counting (catechol ⊂ galloyl):
        ca_net = max(0, ca - g - py)
        # total aromatic OH — count [OH] directly attached to aromatic ring
        ph = len(mol.GetSubstructMatches(_PAT_PHENOL))
        # aromatic OH atoms
        aro_oh = _count_aromatic_oh(mol)

        weighted = (
            g  * _GALLOYL_WEIGHTS["galloyl_strict"]
            + py * _GALLOYL_WEIGHTS["pyrogallol"]
            + ca_net * _GALLOYL_WEIGHTS["catechol"]
        )

        return {
            "galloyl_strict":    g,
            "catechol":          ca_net,
            "pyrogallol":        py,
            "total_aromatic_oh": aro_oh,
            "galloyl_weighted":  round(weighted, 3),
        }

    # ── Subgraph node lists ───────────────────────────────────────────────────

    @staticmethod
    def fragment_subgraph_nodes(mol: Chem.Mol) -> List[List[int]]:
        """
        Return a list of atom-index groups, one per detected galloyl unit.

        Each group contains the 6 ring atoms + attached OH oxygen atoms.

        For pentagalloylglucose → 5 groups (one per galloyl arm).
        For gallic_acid         → 1 group.
        For EDC                 → [] (empty list).

        Parameters
        ----------
        mol : rdkit.Chem.Mol

        Returns
        -------
        List[List[int]]  — atom indices for each fragment subgraph
        """
        if mol is None:
            return []

        fragments: List[List[int]] = []
        seen_atoms: set[int] = set()

        def _extract_matches(pattern: Chem.Mol, extend_oh: bool = True) -> None:
            for match in mol.GetSubstructMatches(pattern):
                match_set = set(match)
                # skip if majority of ring atoms already claimed
                if len(match_set & seen_atoms) > 2:
                    continue
                group = list(match_set)
                if extend_oh:
                    # extend with directly attached OH oxygens
                    for idx in list(match_set):
                        atom = mol.GetAtomWithIdx(idx)
                        for nb in atom.GetNeighbors():
                            if (nb.GetSymbol() == "O"
                                    and nb.GetDegree() == 1
                                    and nb.GetIdx() not in seen_atoms):
                                group.append(nb.GetIdx())
                seen_atoms.update(group)
                fragments.append(group)

        _extract_matches(_PAT_GALLOYL,  extend_oh=True)
        _extract_matches(_PAT_PYROGALL, extend_oh=True)
        # catechol only if no galloyl match consumed those atoms
        _extract_matches(_PAT_CATECHOL, extend_oh=True)

        return fragments

    # ── Convenience wrapper ───────────────────────────────────────────────────

    @staticmethod
    def from_smiles(smiles: str) -> tuple[dict, List[List[int]]]:
        """
        Parse *smiles* with RDKit and return (count_dict, subgraph_node_lists).

        Returns ({…}, []) on parse failure.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return GalloylFragmentDetector.count_fragments(None), []
        counts = GalloylFragmentDetector.count_fragments(mol)
        nodes  = GalloylFragmentDetector.fragment_subgraph_nodes(mol)
        return counts, nodes


# ── atom → fragment index map ─────────────────────────────────────────────────

def atom_to_fragment_map(mol: Chem.Mol,
                         subgraph_nodes: List[List[int]]) -> np.ndarray:
    """
    Build a per-atom array mapping each atom to its fragment index (0-based).

    Atoms not in any fragment receive index -1.

    Parameters
    ----------
    mol             : rdkit.Chem.Mol
    subgraph_nodes  : List[List[int]]  from GalloylFragmentDetector.fragment_subgraph_nodes

    Returns
    -------
    np.ndarray  int32  shape [N_atoms]
    """
    n = mol.GetNumAtoms()
    fmap = np.full(n, fill_value=-1, dtype=np.int32)
    for frag_idx, atom_indices in enumerate(subgraph_nodes):
        for ai in atom_indices:
            if 0 <= ai < n:
                fmap[ai] = frag_idx
    return fmap


# ── Internal helpers ──────────────────────────────────────────────────────────

def _count_aromatic_oh(mol: Chem.Mol) -> int:
    """Count oxygen atoms bonded to an aromatic ring carbon (phenolic OH)."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O":
            continue
        # single bond, degree 1 (bare OH) or degree 2 (ether; skip)
        if atom.GetDegree() != 1 and atom.GetDegree() != 2:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic() and nb.GetSymbol() == "C":
                count += 1
                break
    return count
