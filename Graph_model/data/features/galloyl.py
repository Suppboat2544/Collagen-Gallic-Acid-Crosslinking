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

# Feature-schema version. Bump whenever a change here alters the numeric value
# of any feature, so that a results file can be traced to the code that made it.
#   1 : original patterns (galloyl_strict identically 0; every trihydroxy ring
#       double-counted as pyrogallol + catechol -> 1.67 per ring).
#   2 : corrected patterns, ring-level disjoint accounting (current).
FEATURE_SCHEMA_VERSION = 2

# ── Corrected 2026-08 (schema v2) ────────────────────────────────────────────
# The previous pattern "Oc1cc(O)c(O)cc1" places hydroxyls at ring positions
# 1,3,4 (hydroxyquinol). Those are not three-adjacent, so it matched no ligand
# in the catalogue -- galloyl_strict was identically zero for all nine, gallic
# acid included, and its 1.00 weight never contributed. `galloyl_weighted` was
# then carried by the pyrogallol and catechol patterns, which BOTH match every
# trihydroxyphenyl ring, scoring each ring 1.00 + 0.67 = 1.67 instead of 1.00.
#
# Nomenclature used below (IUPAC / standard phytochemistry):
#   galloyl        = 3,4,5-trihydroxybenzoyl -- a trihydroxyphenyl ring bearing
#                    an acyl carbon. This is the unit in gallate esters such as
#                    1,2,3,4,6-penta-O-galloyl-beta-D-glucose.
#   pyrogallol     = benzene-1,2,3-triol -- three adjacent OH, no acyl.
#   catechol       = benzene-1,2-diol -- two adjacent OH.
# The three are strictly nested (galloyl subset of trihydroxyphenyl subset of
# catechol-bearing), so they are counted at RING level and made disjoint below;
# see _count_fragment_rings.
_SMARTS_GALLOYL  = "[OX2H]c1cc(cc([OX2H])c1[OX2H])[CX3]=[OX1]"  # 3,4,5-triOH benzoyl
_SMARTS_PYROGALL = "[OX2H]c1cccc([OX2H])c1[OX2H]"               # benzene-1,2,3-triol
_SMARTS_CATECHOL = "[OX2H]c1ccccc1[OX2H]"                       # benzene-1,2-diol
# Generic aromatic OH count (broad phenol)
_SMARTS_PHENOL   = "[OX2H]c1ccccc1"

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

        # Count at RING level, then make the three classes disjoint. Matching
        # per-SMARTS-hit and subtracting (the previous approach) is wrong
        # because a single trihydroxyphenyl ring yields TWO catechol hits, so
        # `ca - g - py` left one spurious catechol behind on every such ring.
        g_rings  = _matched_rings(mol, _PAT_GALLOYL)
        tri_rings = _matched_rings(mol, _PAT_PYROGALL)
        ca_rings = _matched_rings(mol, _PAT_CATECHOL)

        # galloyl ⊂ trihydroxyphenyl ⊂ catechol-bearing
        py_net_rings = tri_rings - g_rings
        ca_net_rings = ca_rings - tri_rings

        g      = len(g_rings)
        py     = len(py_net_rings)
        ca_net = len(ca_net_rings)
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

def _matched_rings(mol: Chem.Mol, pattern: Chem.Mol) -> set[frozenset]:
    """
    Set of aromatic rings hit by *pattern*, keyed by the ring's atom indices.

    Collapsing matches onto their parent ring is what makes the galloyl /
    pyrogallol / catechol classes comparable: the same physical ring produces a
    different number of raw SMARTS hits per pattern (a benzene-1,2,3-triol
    matches the catechol pattern twice), so raw hit counts cannot be subtracted
    from one another.
    """
    rings = [frozenset(r) for r in mol.GetRingInfo().AtomRings()]
    out: set[frozenset] = set()
    for match in mol.GetSubstructMatches(pattern, uniquify=True):
        aromatic = {i for i in match if mol.GetAtomWithIdx(i).GetIsAromatic()}
        if not aromatic:
            continue
        for ring in rings:
            if aromatic <= ring:
                out.add(ring)
                break
    return out


def _count_aromatic_oh(mol: Chem.Mol) -> int:
    """
    Count phenolic hydroxyls: an oxygen bearing a hydrogen, bonded to an
    aromatic carbon.

    Corrected 2026-08 (schema v2). The previous version accepted any oxygen of
    degree 1 or 2 attached to an aromatic carbon, with no hydrogen requirement.
    RDKit perceives the fused lactone rings of ellagic acid as aromatic, so its
    two ester oxygens and two lactone carbonyl oxygens were all scored as
    phenolic OH -- 8 reported against 4 actual. Requiring a hydrogen also
    excludes aryl ethers, which have no donor and do not belong in a
    hydrogen-bond-capacity feature.
    """
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O" or atom.GetTotalNumHs() < 1:
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIsAromatic() and nb.GetSymbol() == "C":
                count += 1
                break
    return count
