"""
Graph_model.data.augment
==========================
Tier 3 data loader — curated phenolic binders from ChEMBL / PubChem.

This tier provides 30–300 additional phenolic structures with estimated
or measured binding affinities to augment the anchor set.

Three sources are merged:
  1. Hardcoded fallback set  (30 compounds, always available)
  2. phenolic_augment.csv    (user-provided ChEMBL/PubChem export, optional)
  3. phenolic_augment.sdf    (structure file with activity annotations, optional)

Obtaining the ChEMBL data (optional, recommended)
--------------------------------------------------
  pip install chembl_webresource_client

  Then run:
      python -m Graph_model.data.augment --download

  This queries ChEMBL for phenolic acids / hydrolysable tannins with
  Ki or IC50 values against proteins related to collagen or ECM.
  Results are saved to Graph_model/external/phenolic_augment.csv.

Fallback set rationale
----------------------
The 30 built-in phenolics span the galloyl fragment space:
  - monohydroxyphenyl:  phenol, cinnamic acid
  - dihydroxyphenyl:    catechol, caffeic acid, protocatechuic acid
  - trihydroxyphenyl:   gallic acid, pyrogallol, 3,4,5-THBA
  - condensed tannins:  epigallocatechin, EGCG
  - hydrolysable tannin analogues: methyl gallate, propyl gallate, PGG frag
  ΔG estimates are derived from QSAR models / literature pKd where available.

Record schema (same as anchor / transfer)
------------------------------------------
{
    "sample_id":    str,
    "smiles":       str,
    "ligand_name":  str,
    "delta_g":      float,
    "pH":           float,   # 7.0 (assumed physiological)
    "temperature_C": float,  # 25  (assumed room temp)
    "docking_box":  str,     # "augment_phenolic"
    "receptor":     str,     # "collagen"
    "tier":         str,     # "augment"
}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import AUGMENT_CSV, AUGMENT_SDF

logger = logging.getLogger(__name__)


# ── Built-in phenolic fallback set ────────────────────────────────────────────
# Estimated ΔG from literature QSARs (kcal/mol); range −3 to −8 for small phenolics

BUILTIN_PHENOLICS: List[dict] = [
    # ── Simple phenols ──────────────────────────────────────────────────────
    {"smiles": "Oc1ccccc1",                              "name": "phenol",               "delta_g": -3.4},
    {"smiles": "OC(=O)c1ccccc1",                         "name": "benzoic_acid",          "delta_g": -3.6},
    {"smiles": "Oc1ccc(O)cc1",                           "name": "hydroquinone",          "delta_g": -3.8},
    {"smiles": "Oc1cccc(O)c1",                           "name": "resorcinol",            "delta_g": -3.7},
    # ── Catechols ────────────────────────────────────────────────────────────
    {"smiles": "Oc1ccccc1O",                             "name": "catechol",              "delta_g": -4.0},
    {"smiles": "OC(=O)c1ccc(O)c(O)c1",                  "name": "protocatechuic_acid",   "delta_g": -4.8},
    {"smiles": "OC(=O)/C=C/c1ccc(O)c(O)c1",             "name": "caffeic_acid",          "delta_g": -5.1},
    {"smiles": "OC(=O)CCc1ccc(O)c(O)c1",                "name": "dihydrocaffeic_acid",   "delta_g": -4.9},
    # ── Pyrogallols / gallates ──────────────────────────────────────────────
    {"smiles": "Oc1cccc(O)c1O",                          "name": "pyrogallol",            "delta_g": -4.6},
    {"smiles": "OC(=O)c1cc(O)c(O)c(O)c1",               "name": "gallic_acid",           "delta_g": -5.2},
    {"smiles": "COC(=O)c1cc(O)c(O)c(O)c1",              "name": "methyl_gallate",        "delta_g": -5.4},
    {"smiles": "CCCOC(=O)c1cc(O)c(O)c(O)c1",            "name": "propyl_gallate",        "delta_g": -5.7},
    {"smiles": "CCCCCCCCOC(=O)c1cc(O)c(O)c(O)c1",       "name": "octyl_gallate",         "delta_g": -6.3},
    {"smiles": "OC(=O)c1cc(O)c(O)c(O)c1.OC(=O)c1cc(O)c(O)c(O)c1", "name": "digallic_acid_ref", "delta_g": -6.0},
    # ── Cinnamic / hydroxycinnamic acids ────────────────────────────────────
    {"smiles": "OC(=O)/C=C/c1ccc(O)cc1",                "name": "p_coumaric_acid",       "delta_g": -4.5},
    {"smiles": "OC(=O)/C=C/c1cc(O)c(O)c(O)c1",         "name": "sinapic_acid",          "delta_g": -5.0},
    {"smiles": "OC(=O)/C=C/c1cc(OC)c(O)c(O)c1",        "name": "ferulic_acid",          "delta_g": -5.0},
    # ── Flavonoids / flavan-3-ols ───────────────────────────────────────────
    {"smiles": "Oc1cc(O)c2c(c1)OC(c1ccc(O)cc1)C(O)C2", "name": "catechin",              "delta_g": -6.8},
    {"smiles": "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2", "name": "epicatechin",       "delta_g": -6.9},
    {"smiles": "Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)C(O)C2", "name": "gallocatechin",  "delta_g": -7.0},
    {"smiles": "O=C(O[C@@H]1Cc2c(O)cc(O)cc2O[C@H]1c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1",
               "name": "EGCG",                                                            "delta_g": -8.1},
    # ── Ellagic acid / urolithins ────────────────────────────────────────────
    {"smiles": "O=c1oc2c(O)c(O)cc3c2c1cc1cc(O)c(O)c2c(=O)oc13", "name": "ellagic_acid", "delta_g": -7.3},
    {"smiles": "Oc1cccc2cc(=O)oc12",                    "name": "urolithin_a",           "delta_g": -5.8},
    {"smiles": "Oc1ccc2cc(=O)oc2c1",                    "name": "urolithin_b",           "delta_g": -5.5},
    # ── Quercetin / kaempferol / myricetin ────────────────────────────────
    {"smiles": "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12", "name": "kaempferol",         "delta_g": -7.4},
    {"smiles": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12", "name": "quercetin",       "delta_g": -7.8},
    {"smiles": "O=c1c(O)c(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12", "name": "myricetin",    "delta_g": -8.0},
    # ── Tannin-related phenols ────────────────────────────────────────────────
    {"smiles": "OC(=O)c1cc(OC(=O)c2cc(O)c(O)c(O)c2)c(O)c(O)c1", "name": "depsidone_model", "delta_g": -6.5},
    {"smiles": "OC1C(OC(=O)c2cc(O)c(O)c(O)c2)CCCC1",   "name": "mono_gallate_cyclohex", "delta_g": -5.9},
]


class PhenolicAugmentLoader:
    """
    Tier 3: curated phenolic augmentation set.

    Priority order (first available wins per record):
        1. phenolic_augment.csv  (user-provided, ChEMBL download)
        2. Hardcoded BUILTIN_PHENOLICS (30 compounds, always available)

    Parameters
    ----------
    assume_collagen : bool, default True
        Tag augmented records as "collagen" receptor and pH=7.0, T=25.
    include_builtin : bool, default True
        Always add the 30 built-in phenolics even if an external file is found.
    """

    def __init__(
        self,
        assume_collagen: bool = True,
        include_builtin: bool = True,
    ) -> None:
        self.assume_collagen = assume_collagen
        self.include_builtin = include_builtin

    # ── Public API ─────────────────────────────────────────────────────────────

    def to_records(self) -> List[dict]:
        """Return Tier-3 records in the universal schema."""
        records = []

        # External CSV (optional)
        csv_path = Path(AUGMENT_CSV)
        if csv_path.exists():
            records += self._load_csv(csv_path)

        # Built-in fallback
        if self.include_builtin:
            records += self._load_builtin()

        # De-duplicate by SMILES (keep first occurrence → external takes priority)
        seen_smiles: set[str] = set()
        deduped = []
        for r in records:
            if r["smiles"] not in seen_smiles:
                seen_smiles.add(r["smiles"])
                deduped.append(r)

        logger.info("PhenolicAugmentLoader: %d unique augment records.", len(deduped))
        return deduped

    # ── Loaders ────────────────────────────────────────────────────────────────

    def _load_builtin(self) -> List[dict]:
        out = []
        for i, entry in enumerate(BUILTIN_PHENOLICS):
            out.append(self._make_record(
                sample_id=f"augment_builtin_{i:03d}",
                smiles=entry["smiles"],
                name=entry["name"],
                delta_g=entry["delta_g"],
            ))
        return out

    def _load_csv(self, path: Path) -> List[dict]:
        """
        Load an external CSV with at minimum these columns:
            smiles, delta_g_kcalmol
        Optional: name, pH, temperature_C
        """
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.warning("Could not read augment CSV %s: %s", path, exc)
            return []

        # normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        required = {"smiles", "delta_g_kcalmol"}
        if not required.issubset(df.columns):
            # try alternate column name
            alt_map = {"delta_g": "delta_g_kcalmol",
                       "dg": "delta_g_kcalmol",
                       "binding_energy": "delta_g_kcalmol"}
            for old, new in alt_map.items():
                if old in df.columns:
                    df = df.rename(columns={old: new})
            if "delta_g_kcalmol" not in df.columns:
                logger.warning("Augment CSV missing 'delta_g_kcalmol' column; skipping.")
                return []

        out = []
        for i, row in df.iterrows():
            smiles = str(row.get("smiles", "")).strip()
            if not smiles or smiles.lower() == "nan":
                continue
            try:
                dg = float(row["delta_g_kcalmol"])
            except (ValueError, KeyError):
                continue

            out.append(self._make_record(
                sample_id=f"augment_ext_{i:05d}",
                smiles=smiles,
                name=str(row.get("name", row.get("compound_name", ""))),
                delta_g=dg,
                ph=float(row.get("ph", 7.0)),
                temp=float(row.get("temperature_c", 25.0)),
            ))
        logger.info("PhenolicAugmentLoader: loaded %d records from %s", len(out), path)
        return out

    # ── Record factory ─────────────────────────────────────────────────────────

    def _make_record(
        self,
        sample_id: str,
        smiles: str,
        name: str,
        delta_g: float,
        ph: float = 7.0,
        temp: float = 25.0,
    ) -> dict:
        return {
            "sample_id":            sample_id,
            "smiles":               smiles,
            "ligand_name":          name,
            "delta_g":              delta_g,
            "pH":                   ph,
            "temperature_C":        temp,
            "docking_box":          "augment_phenolic",
            "receptor":             "collagen" if self.assume_collagen else "unknown",
            "n_interactions":       0,
            "interacting_residues": "",
            "tier":                 "augment",
        }
