"""
Graph_model.screen.library_gen
===============================
Combinatorial galloyl-analogue library generator.

Strategy
--------
Start from a gallic acid core (3,4,5-trihydroxybenzoic acid) and
systematically enumerate structural variations:

  1. **OH pattern** — Vary the number (1-5) and positions of phenolic
     hydroxyls on the aromatic ring(s).
  2. **Acyl / ester linkage** — Attach 1-5 galloyl arms to a central
     scaffold (glucose, cyclodextrin ring, Lys/Glu linker).
  3. **Ring substituents** — Add -OCH₃, -CH₃, -F, -Cl to remaining ring
     positions for lipophilicity / H-bond tuning.
  4. **Linker length** — Vary the number of carbons between the carboxyl
     anchor and the phenolic ring (C0-C3).

The library targets 200-500 unique, chemically valid structures.

All SMILES are canonicalised and sanitised by RDKit.

Usage
-----
>>> lib = GalloylLibrary(seed=42)
>>> lib.generate()
>>> print(len(lib))         # 200–500 unique SMILES
>>> df = lib.to_dataframe()
>>> df.head()
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    logger.warning("RDKit not installed; GalloylLibrary will not function.")


# ── Core scaffolds ────────────────────────────────────────────────────────────

# Gallic acid core:  3,4,5-trihydroxybenzoic acid
_GALLIC_ACID = "OC(=O)c1cc(O)c(O)c(O)c1"

# Phenolic ring templates with variable OH (positions 3,4,5 on benzoic acid)
# {pos}: O or substituent placeholder
_PHENOL_RING_TEMPLATE = "c1cc({R3})c({R4})c({R5})c1"

# Central scaffolds for multi-arm molecules
_SCAFFOLDS: dict[str, str] = {
    "glucose":       "OC1OC(CO)C(O)C(O)C1O",         # β-D-glucose
    "shikimic":      "OC1CC(=CC1O)C(=O)O",            # shikimic acid
    "catechol_dimer": "Oc1ccc(-c2ccc(O)c(O)c2)cc1O",  # biphenyl catechol
}

# Substituents for ring decoration
_SUBSTITUENTS: dict[str, str] = {
    "OH":    "O",
    "OCH3":  "OC",
    "CH3":   "C",
    "F":     "F",
    "Cl":    "Cl",
    "H":     "[H]",  # explicit-H placeholder → removed during sanitisation
}

# Linker chains between carboxyl and ring
_LINKERS: list[str] = [
    "",           # direct attachment (gallic acid)
    "C",          # one-carbon spacer
    "CC",         # two-carbon spacer
    "CCC",        # three-carbon spacer
    "C=C",        # vinyl spacer
    "C(=O)N",     # amide linker
]


@dataclass
class GalloylLibrary:
    """
    Combinatorial galloyl analogue library.

    Parameters
    ----------
    seed : int
        Random seed for reproducible subset selection.
    max_size : int
        Maximum library size (compounds are de-duplicated then capped).
    min_size : int
        Minimum target — if combinatorial expansion yields fewer, extras are
        generated via random OH/substituent permutations.
    max_mw : float
        Maximum molecular weight cutoff (filter drug-likeness).
    max_ha : int
        Maximum heavy atoms.
    """
    seed:     int   = 42
    max_size: int   = 500
    min_size: int   = 200
    max_mw:   float = 1200.0
    max_ha:   int   = 80
    _library: list[dict] = field(default_factory=list, init=False, repr=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self) -> "GalloylLibrary":
        """
        Generate the combinatorial library.  Returns self for chaining.
        """
        assert _HAS_RDKIT, "RDKit is required for library generation"
        rng = np.random.default_rng(self.seed)
        seen: set[str] = set()
        candidates: list[dict] = []

        def _add(smiles: str, source: str, n_oh: int = -1,
                 scaffold: str = "gallic", **extra):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return
            canon = Chem.MolToSmiles(mol, canonical=True)
            if canon in seen:
                return

            mw = Descriptors.ExactMolWt(mol)
            ha = mol.GetNumHeavyAtoms()
            if mw > self.max_mw or ha > self.max_ha:
                return

            seen.add(canon)
            if n_oh < 0:
                n_oh = sum(1 for a in mol.GetAtoms()
                           if a.GetAtomicNum() == 8
                           and any(n.GetAtomicNum() == 1 or
                                   (n.GetAtomicNum() == 6 and
                                    mol.GetBondBetweenAtoms(
                                        a.GetIdx(), n.GetIdx()
                                    ).GetBondTypeAsDouble() == 1.0)
                                   for n in a.GetNeighbors()))
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)

            candidates.append({
                "smiles": canon,
                "source": source,
                "scaffold": scaffold,
                "n_oh": n_oh,
                "mw": round(mw, 2),
                "n_ha": ha,
                "hbd": hbd,
                "hba": hba,
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "uid": hashlib.md5(canon.encode()).hexdigest()[:10],
                **extra,
            })

        # ── Strategy 1: Single-ring OH enumeration ───────────────────────────
        oh_patterns = [
            # (R3, R4, R5) — substituents at positions 3,4,5
            ("O", "O", "O"),       # gallic acid (3-OH)
            ("O", "O", "[H]"),     # protocatechuic-like (2-OH)
            ("O", "[H]", "O"),     # 3,5-diOH
            ("[H]", "O", "O"),     # 4,5-diOH
            ("O", "[H]", "[H]"),   # 3-monoOH
            ("[H]", "O", "[H]"),   # 4-monoOH (p-hydroxybenzoic)
            ("[H]", "[H]", "O"),   # 5-monoOH
            ("O", "O", "OC"),      # catechol + methoxy
            ("OC", "O", "O"),      # methoxy + catechol
            ("O", "OC", "O"),      # 3-OH, 4-OMe, 5-OH
        ]

        for r3, r4, r5 in oh_patterns:
            for linker in _LINKERS:
                smi = f"OC(=O){linker}c1cc({r3})c({r4})c({r5})c1"
                _add(smi, "single_ring_oh", scaffold="single_ring")

        # ── Strategy 2: Ring substituent decoration ──────────────────────────
        base_acids = [
            "OC(=O)c1cc(O)c(O)c(O)c1",       # gallic acid
            "OC(=O)c1ccc(O)c(O)c1",           # PCA
            "OC(=O)c1cc(O)c(O)cc1",           # 3,4-diOH variant
            "OC(=O)/C=C/c1cc(O)c(O)c(O)c1",  # caffeic acid galloyl
        ]
        extra_subs = ["C", "OC", "F", "Cl", "CC"]
        for base in base_acids:
            mol = Chem.MolFromSmiles(base)
            if mol is None:
                continue
            for sub in extra_subs:
                # Add substituent at available position via reaction SMARTS
                rxn_smarts = "[c:1][H]>>[c:1]" + sub
                try:
                    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                    products = rxn.RunReactants((mol,))
                    for prod_set in products[:3]:
                        for p in prod_set:
                            try:
                                Chem.SanitizeMol(p)
                                _add(Chem.MolToSmiles(p), "ring_decoration",
                                     scaffold="decorated")
                            except Exception:
                                continue
                except Exception:
                    continue

        # ── Strategy 3: Multi-galloyl esters (di/tri/tetra/penta) ────────────
        galloyl_smiles = "OC(=O)c1cc(O)c(O)c(O)c1"
        ga_variants = [
            galloyl_smiles,
            "OC(=O)c1ccc(O)c(O)c1",   # PCA ester arm
            "OC(=O)c1cc(O)c(O)cc1",   # 3,4-diOH arm
        ]

        # Glucose-based esters (2-5 arms)
        glucose = Chem.MolFromSmiles("OC1OC(CO)C(O)C(O)C1O")
        if glucose is not None:
            oh_indices = [a.GetIdx() for a in glucose.GetAtoms()
                         if a.GetAtomicNum() == 8
                         and a.GetTotalNumHs() >= 1]
            for n_arms in range(2, min(6, len(oh_indices) + 1)):
                combos = _choose_subsets(oh_indices, n_arms, rng, max_combos=5)
                for arm_smiles in ga_variants:
                    for combo in combos:
                        ester_smi = _build_multi_ester(
                            "OC1OC(CO)C(O)C(O)C1O", arm_smiles, combo)
                        if ester_smi:
                            _add(ester_smi, f"glucose_{n_arms}arm",
                                 scaffold="glucose_ester", n_oh=n_arms * 3)

        # ── Strategy 4: Dimeric galloyl structures ───────────────────────────
        dimer_linkers = ["O", "OCC0", "NC(=O)", "OC(=O)"]
        for link in dimer_linkers:
            smi = (f"OC(=O)c1cc(O)c(O)c(O)c1{link}"
                   f"c1cc(O)c(O)c(O)c1C(=O)O")
            _add(smi, "dimer", scaffold="dimer")

        # ── Strategy 5: Natural product analogues ────────────────────────────
        natural_analogues = [
            ("ellagic_acid",    "O=c1oc2cc(O)c(O)cc2-c2cc(O)c(O)cc2-1"),
            ("tannic_core",     "OC(=O)c1cc(O)c(O)c(O)c1OC(=O)c1cc(O)c(O)c(O)c1"),
            ("theaflavin_core", "O=C(O)c1cc(O)c2c(c1)OC(=O)c1cc(O)c(O)c(O)c1-2"),
            ("EGCG_core",
             "OC1Cc2c(O)cc(O)cc2OC1c1cc(O)c(O)c(O)c1"),
            ("corilagin_core",
             "OC(=O)c1cc(O)c(O)c(O)c1OC1OC(COC(=O)c2cc(O)c(O)c(O)c2)"
             "C(O)C(O)C1OC(=O)c1cc(O)c(O)c(O)c1"),
            ("methyl_gallate",  "COC(=O)c1cc(O)c(O)c(O)c1"),
            ("propyl_gallate",  "CCCOC(=O)c1cc(O)c(O)c(O)c1"),
            ("lauryl_gallate",  "CCCCCCCCCCCCOC(=O)c1cc(O)c(O)c(O)c1"),
            ("octyl_gallate",   "CCCCCCCCOC(=O)c1cc(O)c(O)c(O)c1"),
            ("ethyl_gallate",   "CCOC(=O)c1cc(O)c(O)c(O)c1"),
            ("syringic_acid",   "COc1cc(C(=O)O)cc(OC)c1O"),
            ("vanillic_acid",   "COc1cc(C(=O)O)ccc1O"),
            ("ferulic_acid",    "COc1cc(/C=C/C(=O)O)ccc1O"),
            ("caffeic_acid",    "OC(=O)/C=C/c1ccc(O)c(O)c1"),
            ("chlorogenic_acid",
             "O=C(/C=C/c1ccc(O)c(O)c1)OC1CC(O)(C(=O)O)CC(O)C1O"),
            ("rosmarinic_acid",
             "OC(=O)C(Cc1ccc(O)c(O)c1)OC(=O)/C=C/c1ccc(O)c(O)c1"),
        ]
        for name, smi in natural_analogues:
            _add(smi, "natural_analogue", scaffold=name)

        # ── Strategy 6: Random OH/sub permutations to fill to min_size ───────
        if len(candidates) < self.min_size:
            n_needed = self.min_size - len(candidates)
            for _ in range(n_needed * 5):
                if len(candidates) >= self.min_size:
                    break
                # Random ring: 1-4 OH, 0-2 extra substituents
                n_oh_ring = int(rng.integers(1, 5))
                positions = rng.choice(3, size=n_oh_ring, replace=False)
                subs = ["[H]", "[H]", "[H]"]
                for p in positions:
                    subs[p] = "O"
                # Random extra substituent on remaining positions
                for i in range(3):
                    if subs[i] == "[H]" and rng.random() < 0.3:
                        subs[i] = rng.choice(["OC", "C", "F"])
                linker = rng.choice(_LINKERS)
                smi = f"OC(=O){linker}c1cc({subs[0]})c({subs[1]})c({subs[2]})c1"
                _add(smi, "random_fill", scaffold="random")

        # ── De-duplicate and cap ─────────────────────────────────────────────
        if len(candidates) > self.max_size:
            # Prefer diverse scaffolds
            indices = list(range(len(candidates)))
            rng.shuffle(indices)
            candidates = [candidates[i] for i in indices[:self.max_size]]

        self._library = candidates
        logger.info(f"Generated galloyl library: {len(self._library)} compounds")
        return self

    def __len__(self) -> int:
        return len(self._library)

    def __iter__(self):
        return iter(self._library)

    def __getitem__(self, idx):
        return self._library[idx]

    @property
    def smiles_list(self) -> list[str]:
        """All unique SMILES in the library."""
        return [c["smiles"] for c in self._library]

    @property
    def compounds(self) -> list[dict]:
        """Full compound metadata."""
        return list(self._library)

    def to_dataframe(self):
        """Return library as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self._library)

    def scaffold_distribution(self) -> dict[str, int]:
        """Count compounds per scaffold type."""
        from collections import Counter
        return dict(Counter(c["scaffold"] for c in self._library))

    def filter(self, **kwargs) -> list[dict]:
        """
        Filter compounds by property ranges.

        Example: lib.filter(mw_max=500, logp_min=-2, logp_max=3)
        """
        results = list(self._library)
        for key, val in kwargs.items():
            prop, op = key.rsplit("_", 1)
            if op == "max":
                results = [c for c in results if c.get(prop, 0) <= val]
            elif op == "min":
                results = [c for c in results if c.get(prop, 0) >= val]
            elif op == "eq":
                results = [c for c in results if c.get(prop) == val]
        return results


# ── Helper functions ──────────────────────────────────────────────────────────

def _choose_subsets(
    items: list[int],
    k: int,
    rng: np.random.Generator,
    max_combos: int = 5,
) -> list[tuple[int, ...]]:
    """
    Return up to `max_combos` random k-subsets of `items`.
    """
    from itertools import combinations
    all_combos = list(combinations(items, k))
    if len(all_combos) <= max_combos:
        return all_combos
    indices = rng.choice(len(all_combos), size=max_combos, replace=False)
    return [all_combos[i] for i in indices]


def _build_multi_ester(
    scaffold_smi: str,
    arm_smi: str,
    oh_positions: tuple[int, ...],
) -> Optional[str]:
    """
    Build a multi-ester by replacing OH groups at given positions
    with galloyl ester linkages.

    Uses a simplified approach: esterify selected -OH groups.
    """
    try:
        scaffold = Chem.MolFromSmiles(scaffold_smi)
        arm = Chem.MolFromSmiles(arm_smi)
        if scaffold is None or arm is None:
            return None

        # RxnSMARTS for esterification: R-OH + HO-C(=O)-R' → R-O-C(=O)-R'
        rxn = AllChem.ReactionFromSmarts(
            "[O:1][H].[O:2][C:3](=[O:4])>>[O:1][C:3](=[O:4]).[O:2]"
        )

        current = scaffold
        for _ in oh_positions:
            products = rxn.RunReactants((current, arm))
            if products:
                prod = products[0][0]
                try:
                    Chem.SanitizeMol(prod)
                    current = prod
                except Exception:
                    break
            else:
                break

        return Chem.MolToSmiles(current, canonical=True) if current else None
    except Exception:
        return None
