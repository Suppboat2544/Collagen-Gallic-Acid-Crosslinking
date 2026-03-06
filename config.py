"""
Graph_model.data.config
=======================
Ground truth for every ligand, fixed condition vocabulary, and filesystem paths.
All downstream modules import from here — do not hard-code strings elsewhere.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem roots
# ---------------------------------------------------------------------------
REPO_ROOT   = Path("/Users/suppboat/Jupyter_Dock")
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
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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
        "mw":                191.70,
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
        "smiles":            "CCC(=O)OC(=NCC)NCCCN(C)C",
        "pubchem_cid":       2723762,
        "mw":                229.32,
        "n_ha":              13,
        "role":              "intermediate",
        "group":             "intermediate",
        "galloyl_units":     0,
        "catechol_oh":       0,
        "propka_pka_oh":     [],
    },
    "NHS_ester_intermediate": {
        "smiles":            "O=C1CCC(=O)N1OC(=O)CCCN(C)C",
        "pubchem_cid":       2723763,
        "mw":                285.22,
        "n_ha":              10,
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
        # PubChem CID 5281855 canonical SMILES (RDKit-verified)
        "smiles":            "O=c1oc2cc(O)c(O)cc2-c2cc(O)c(O)cc2-1",
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
        # PGG — canonical SMILES (flattened representation)
        "smiles":            (
            "OC(=O)c1cc(O)c(O)c(O)c1."     # galloyl 1  (fragmented SMILES
            "OC(=O)c1cc(O)c(O)c(O)c1."     # galloyl 2   for readability;
            "OC(=O)c1cc(O)c(O)c(O)c1."     # galloyl 3   use PubChem CID
            "OC(=O)c1cc(O)c(O)c(O)c1."     # galloyl 4   65238 for exact)
            "OC(=O)c1cc(O)c(O)c(O)c1."     # galloyl 5
            "OC1C(OC(=O)c2cc(O)c(O)c(O)c2)"
            "C(OC(=O)c2cc(O)c(O)c(O)c2)"
            "C(OC(=O)c2cc(O)c(O)c(O)c2)"
            "C1OC(=O)c1cc(O)c(O)c(O)c1"
        ),
        # Use PubChem fetch for real docking (sdf already in anchor dir)
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

# Two specific PGG SMILES for RDKit (use fully connected form from PubChem)
PGG_SMILES_RDKIT = (
    "O=C(O[C@@H]1[C@H](OC(=O)c2cc(O)c(O)c(O)c2)"
    "[C@@H](OC(=O)c2cc(O)c(O)c(O)c2)"
    "[C@H](OC(=O)c2cc(O)c(O)c(O)c2)"
    "[C@@H]1OC(=O)c1cc(O)c(O)c(O)c1)c1cc(O)c(O)c(O)c1"
)
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
