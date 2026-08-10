"""
Graph_model.data.config
=======================
Ground truth for every ligand, fixed condition vocabulary, and filesystem paths.
All downstream modules import from here — do not hard-code strings elsewhere.
"""

from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem roots
# ---------------------------------------------------------------------------
# Resolution order:
#   1. $COLLAGEN_DATA_ROOT           (set this to point at your data checkout)
#   2. the repository root itself    (parents[2] of this file)
# Never hard-code an absolute path here again — it makes the package
# unimportable on every machine but one.
_ENV_ROOT = os.environ.get("COLLAGEN_DATA_ROOT")
REPO_ROOT   = Path(_ENV_ROOT).expanduser() if _ENV_ROOT else Path(__file__).resolve().parents[2]
ANCHOR_DIR  = REPO_ROOT / "Phukhao" / "collagen_gallic_results"

# Primary CSV files (Tier 1 — Anchor)
COLLAGEN_CSV   = ANCHOR_DIR / "collagen_crosslinking_docking_results.csv"
MMP1_CSV       = ANCHOR_DIR / "mmp1_collagenase_docking_results.csv"
REDOCK_CSV     = ANCHOR_DIR / "redocking_validation_results.csv"
FULL_JSON      = ANCHOR_DIR / "full_results.json"

# Tier 2 — Transfer  (PDBbind v2020R1 from external_dataset/)
PDBBIND_ROOT   = REPO_ROOT / "Graph_model" / "external_dataset"
PDBBIND_INDEX  = PDBBIND_ROOT / "index" / "INDEX_general_PL.2020R1.lst"
PDBBIND_PL_DIR = PDBBIND_ROOT / "P-L"                      # structure files

# Tier 3 — Augment  (ChEMBL/PubChem phenolic exports)
AUGMENT_SDF    = REPO_ROOT / "Graph_model" / "external" / "phenolic_augment.sdf"
AUGMENT_CSV    = REPO_ROOT / "Graph_model" / "external" / "phenolic_augment.csv"

# Processed / cached artifacts
PROCESSED_DIR  = REPO_ROOT / "Graph_model" / "data" / "processed"


def ensure_dirs() -> None:
    """
    Create the writable output directories.

    Call this explicitly from an entry point — importing a config module must
    never touch the filesystem, or `import Graph_model.data` silently creates
    directories on whatever machine happens to run it.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def check_data_present() -> list[str]:
    """Return a list of human-readable problems with the current data root."""
    problems: list[str] = []
    if not ANCHOR_DIR.exists():
        problems.append(
            f"Anchor directory not found: {ANCHOR_DIR}\n"
            f"  Set COLLAGEN_DATA_ROOT to the checkout that contains "
            f"Phukhao/collagen_gallic_results/, or place the CSVs there."
        )
    # Only the collagen CSV is required; MMP-1 is opt-in (include_mmp1=False).
    if not COLLAGEN_CSV.exists():
        problems.append(f"Missing required collagen CSV: {COLLAGEN_CSV}")
    return problems


def optional_data_warnings() -> list[str]:
    """Non-fatal gaps worth reporting at startup."""
    warnings: list[str] = []
    if not MMP1_CSV.exists():
        warnings.append(
            f"MMP-1 CSV not found ({MMP1_CSV}); running collagen-only. "
            f"Selectivity / multi-task results are unavailable."
        )
    return warnings


# ---------------------------------------------------------------------------
# Nine-ligand catalogue
# ---------------------------------------------------------------------------
# Each entry:  name → {smiles, pubchem_cid, mw, n_ha, role, group,
#                       galloyl_units, prop_ka_pka_catechol}
LIGAND_CATALOGUE: dict[str, dict] = {
    # ── Primary crosslinkers ─────────────────────────────────────────────────
    "gallic_acid": {
        "smiles":            "OC(=O)c1cc(O)c(O)c(O)c1",
        "pubchem_cid":       370,
        "mw":                170.12,
        "n_ha":              12,
        "role":              "primary",
        "group":             "primary",
        "galloyl_units":     1,          # one 3,4,5-trihydroxyphenyl ring
        "catechol_oh":       3,          # aromatic OH count
        "propka_pka_oh":     [9.2, 9.8, 10.4],  # per-OH PropKa estimates (pH 5–7 range)
    },
    "EDC": {
        "smiles":            "CCN=C=NCCCN(C)C",
        "pubchem_cid":       2723761,
        # Free base, matching the SMILES above (C8H17N3). The commonly quoted
        # 191.70 is the hydrochloride salt (EDC.HCl) — a different species.
        "mw":                155.24,
        "n_ha":              11,
        "role":              "primary",
        "group":             "primary",
        "galloyl_units":     0,
        "catechol_oh":       0,
        "propka_pka_oh":     [],
    },
    "NHS": {
        "smiles":            "O=C1CCC(=O)NO1",
        "pubchem_cid":       80180,
        "mw":                115.09,
        "n_ha":              8,
        "role":              "primary",
        "group":             "primary",
        "galloyl_units":     0,
        "catechol_oh":       0,
        "propka_pka_oh":     [],
    },
    # ── Reaction intermediates ───────────────────────────────────────────────
    "EDC_Oacylisourea": {
        # Stand-in: O-acylisourea of propionic acid + EDC, not of gallic acid.
        "smiles":            "CCC(=O)OC(=NCC)NCCCN(C)C",
        "pubchem_cid":       2723762,
        "mw":                229.32,          # RDKit-verified for the SMILES above
        "n_ha":              16,              # was 13 — miscounted
        "role":              "intermediate",
        "group":             "intermediate",
        "galloyl_units":     0,
        "catechol_oh":       0,
        "propka_pka_oh":     [],
    },
    "NHS_ester_intermediate": {
        # FIXME(chemistry): this entry was internally inconsistent — the stated
        # mw (285.22) and n_ha (10) matched neither each other nor the SMILES,
        # which RDKit reads as C10H16N2O4, MW 228.25, 16 heavy atoms (the NHS
        # ester of 4-(dimethylamino)butanoic acid). mw/n_ha are set to the
        # SMILES below so the catalogue is at least self-consistent, but the
        # intended species is ambiguous: if what was docked is the NHS ester of
        # *gallic acid* (C11H9NO7, MW 267.19, 19 heavy atoms), replace the
        # SMILES and these two numbers together. Confirm against the docking
        # inputs before reporting this ligand.
        "smiles":            "O=C1CCC(=O)N1OC(=O)CCCN(C)C",
        "pubchem_cid":       2723763,
        "mw":                228.25,
        "n_ha":              16,
        "role":              "intermediate",
        "group":             "intermediate",
        "galloyl_units":     0,
        "catechol_oh":       0,
        "propka_pka_oh":     [],
    },
    # ── Galloyl analogues ────────────────────────────────────────────────────
    "protocatechuic_acid": {
        "smiles":            "OC(=O)c1ccc(O)c(O)c1",
        "pubchem_cid":       72,
        "mw":                154.12,
        "n_ha":              11,
        "role":              "galloyl_analogue",
        "group":             "GA_analogue",
        "galloyl_units":     0,          # catechol (2-OH), not strict galloyl (3-OH)
        "catechol_oh":       2,
        "propka_pka_oh":     [9.4, 11.1],
    },
    "pyrogallol": {
        "smiles":            "Oc1cccc(O)c1O",
        "pubchem_cid":       1057,
        "mw":                126.11,
        "n_ha":              9,
        "role":              "galloyl_analogue",
        "group":             "GA_analogue",
        "galloyl_units":     1,          # 1,2,3-trihydroxybenzene ~ galloyl core
        "catechol_oh":       3,
        "propka_pka_oh":     [9.0, 9.7, 11.2],
    },
    "ellagic_acid": {
        # PubChem CID 5281855. RDKit-verified: 22 heavy atoms, MW 302.19, C14H6O8.
        #
        # NOTE: the previous string was a *mono*lactone (C13H8O6, MW 260.20,
        # 19 heavy atoms) — it was missing one of the two lactone bridges, so it
        # did not match this entry's own mw/n_ha or the "dilactone" comment below.
        "smiles":            "C1=C2C3=C(C(=C1O)O)OC(=O)C4=CC(=C(C(=C43)OC2=O)O)O",
        "pubchem_cid":       5281855,
        "mw":                302.19,
        "n_ha":              22,
        "role":              "galloyl_analogue",
        "group":             "GA_analogue",
        "galloyl_units":     2,          # two galloyl rings fused into dilactone
        "catechol_oh":       4,
        "propka_pka_oh":     [8.5, 9.2, 9.8, 10.3],
    },
    "pentagalloylglucose": {
        # PGG — 1,2,3,4,6-penta-O-galloyl-beta-D-glucopyranose, PubChem CID 65238.
        # RDKit-verified: 1 fragment, 67 heavy atoms, MW 940.68, C41H32O26,
        # 5 galloyl-ring SMARTS matches (== galloyl_units below).
        #
        # NOTE: the previous value here was a '.'-separated string of five free
        # gallic acids plus a cyclopentane core — RDKit parsed it as 6
        # disconnected fragments, 114 heavy atoms, MW 1609.15, and the galloyl
        # detector counted 9 units. Every model consuming it saw a molecule
        # roughly 1.7x too large. Do not "simplify" this string again.
        "smiles":            (
            "O=C(OC[C@H]1O[C@@H](OC(=O)c2cc(O)c(O)c(O)c2)"
            "[C@H](OC(=O)c2cc(O)c(O)c(O)c2)"
            "[C@@H](OC(=O)c2cc(O)c(O)c(O)c2)"
            "[C@@H]1OC(=O)c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
        ),
        "pubchem_cid":       65238,
        "mw":                940.68,
        "n_ha":              67,
        "role":              "galloyl_analogue",
        "group":             "GA_analogue",
        "galloyl_units":     5,
        "catechol_oh":       15,         # 3 OH × 5 galloyl arms
        "propka_pka_oh":     [8.3, 8.6, 9.0, 9.2, 9.5,
                               9.7, 9.9, 10.0, 10.1, 10.1,
                               10.2, 10.3, 10.4, 10.5, 10.6],
    },
}

# Backwards-compatible alias. This used to hold a *different* (and also wrong)
# PGG string — a cyclopentane pentagallate with no ring oxygen and no C6
# hydroxymethyl: C40H30O25, MW 910.66, 65 heavy atoms. It was read by nothing.
# It now simply points at the corrected catalogue entry so the two cannot drift.
PGG_SMILES_RDKIT = LIGAND_CATALOGUE["pentagalloylglucose"]["smiles"]
LIGAND_CATALOGUE["pentagalloylglucose"]["smiles_rdkit"] = PGG_SMILES_RDKIT


# ---------------------------------------------------------------------------
# Condition vocabularies
# ---------------------------------------------------------------------------

# pH levels tested
PH_VALUES: list[float] = [5.0, 5.5, 7.0]

# Temperature levels (°C)
TEMP_VALUES: list[int] = [4, 25, 37]

# Receptors
RECEPTORS: dict[str, int] = {
    "collagen":  0,   # Sus scrofa Collagen I α-2 (AlphaFold F1SFA7)
    "mmp1":      1,   # Porcine MMP-1, PDB 966C
}

# Box type taxonomy  →  integer ID for embedding lookup
# 8 types  →  embed into 16-dim vector (trainable)
BOX_TYPE_VOCAB: dict[str, int] = {
    "GLU_cluster":         0,
    "LYS_cluster":         1,
    "ASP_cluster":         2,
    "GLU_LYS_cluster":     3,
    "ASP_GLU_cluster":     4,
    "ASP_LYS_cluster":     5,
    "ASP_GLU_LYS_cluster": 6,
    "global_blind":        7,
}
BOX_EMBEDDING_DIM: int = 16
N_BOX_TYPES: int = len(BOX_TYPE_VOCAB)   # 8

# PropKa-derived fractional protonation of GLU/ASP at each pH
# Encodes the "physics" of pH instead of raw pH value.
# Source: SI-1.1 — GLU724/GLU741 Glu_cluster22 vicinity.
#   pH 5.0 → protonation prob > 0.85  → f_protonated ~ 0.85
#   pH 5.5 → just deprotonated        → f_protonated ~ 0.15
#   pH 7.0 → fully deprotonated       → f_protonated ~ 0.02
# FIXME(physics): these three values are not reachable from any single pKa.
# Henderson-Hasselbalch with f=0.85 at pH 5.0 implies pKa 5.75, which predicts
# f=0.64 at pH 5.5 and f=0.05 at pH 7.0 — not 0.15 and 0.02. An 0.85 -> 0.15
# drop across 0.5 pH units is thermodynamically impossible for one titratable
# site. Graph_model/graph/residue_data.py already implements proper HH; this
# table is a second, contradictory pH model, and it is the one that reaches the
# network. Left unchanged here on purpose — fixing it changes every condition
# vector and therefore every reported number, so it needs a deliberate re-run.
PROPKA_PROTONATION: dict[float, float] = {
    5.0: 0.85,
    5.5: 0.15,
    7.0: 0.02,
}

# Ligand groups (for stratified splitting and evaluation)
LIGAND_GROUPS: dict[str, str] = {
    k: v["group"] for k, v in LIGAND_CATALOGUE.items()
}

# All 9 ligand names in canonical order
LIGAND_NAMES: list[str] = list(LIGAND_CATALOGUE.keys())

# Galloyl unit counts per ligand (for the fragment-graph layer)
GALLOYL_UNIT_COUNTS: dict[str, int] = {
    k: v["galloyl_units"] for k, v in LIGAND_CATALOGUE.items()
}
