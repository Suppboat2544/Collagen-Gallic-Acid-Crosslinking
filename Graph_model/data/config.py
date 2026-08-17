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
        # Resolved 2026-08 against the shipped docking inputs.
        #
        # The original entry was internally inconsistent: mw 285.22 and n_ha 10
        # matched neither each other nor its SMILES (C10H16N2O4, MW 228.25, 16
        # heavy atoms -- the NHS ester of 4-(dimethylamino)butanoic acid).
        #
        # What the campaign actually modelled is now settled by reading
        # <data_root>/Phukhao/collagen_gallic_results/*.sdf. The EDC/NHS series
        # there is built on PROPIONIC acid as a small stand-in for the carboxylic
        # acid partner, not on gallic acid: EDC_Oacylisourea is the
        # O-propanoylisourea (C11H23N3O2), which the catalogue and the structure
        # file agree on. The consistent NHS-stage species for that series is
        # therefore N-succinimidyl propionate, set below.
        #
        # The shipped structure is NOT that molecule. It reads as C6H7NO4 with a
        # FOUR-membered ring -- exactly one CH2 short of succinimidyl propionate,
        # and NHS.sdf is short by exactly the same CH2 (C3H3NO3 against C4H5NO3).
        # A succinimide ring has four ring carbons; both files were built with
        # three. This is one systematic structure-building error propagated
        # through every NHS-containing species, not two independent typos.
        #
        # The catalogue records the correct chemistry. The docking scores are
        # left untouched -- they belong to the CH2-deficient molecules and
        # cannot be corrected by editing this file. Re-docking is required
        # before any NHS-stage affinity is reported.
        # Graph_model.data.provenance reports the discrepancy on every run.
        #
        # If the intent was instead the mechanistic gallic-acid adduct, the
        # species is N-succinimidyl gallate, C11H9NO7, MW 267.19, 19 heavy atoms,
        # SMILES "O=C(ON1C(=O)CCC1=O)c1cc(O)c(O)c(O)c1" -- but that is a
        # different experiment from the one on disk, so it is not asserted here.
        "smiles":            "CCC(=O)ON1C(=O)CCC1=O",   # N-succinimidyl propionate
        "pubchem_cid":       None,   # not asserted; the previous CID 2723763 was
                                     # for the dimethylaminobutanoate ester above
        "mw":                171.15,
        "n_ha":              12,
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

# ── Fractional protonation of the GLU carboxylate vs pH ──────────────────────
# Corrected 2026-08. Previously a hardcoded table:
#     {5.0: 0.85, 5.5: 0.15, 7.0: 0.02}
# Those three numbers are not reachable from any single pKa. Henderson-
# Hasselbalch with f = 0.85 at pH 5.0 implies pKa 5.75, which then predicts
# f = 0.64 at pH 5.5 and f = 0.05 at pH 7.0 -- not 0.15 and 0.02. A drop from
# 0.85 to 0.15 across 0.5 pH units is thermodynamically impossible for one
# titratable site: HH bounds any single site to at most ~0.76 -> 0.24 over half
# a pH unit, and only when centred exactly on its pKa. The table was also a
# SECOND pH model contradicting the Henderson-Hasselbalch implementation in
# Graph_model/graph/residue_data.py, and it was the one reaching the network.
#
# The fractions are now DERIVED from a single pKa via Henderson-Hasselbalch,
# so there is one pH model in the codebase and the table cannot drift out of
# thermodynamic consistency again.
#
#     f_protonated(pH) = 1 / (1 + 10^(pH - pKa))
#
# pKa source: Olsson M.H.M. et al., "PROPKA3: Consistent Treatment of Internal
# and Surface Residues in Empirical pKa Predictions", J. Chem. Theory Comput.
# 7(2):525-537, 2011 -- the same reference already used for the residue-level
# table in Graph_model/graph/residue_data.py, which lists GLU at 4.07.
#
# NOTE FOR THE AUTHORS: if SI-1.1 reports a PropKa-shifted pKa for the specific
# residues in question (GLU724 / GLU741 near GLU_cluster22 are buried and may
# titrate above the 4.07 generic value), replace GLU_PKA below with that single
# number. All three fractions then follow consistently. Do not re-enter
# per-pH fractions by hand -- that is what produced the impossible table.
GLU_PKA: float = 4.07     # Olsson et al. 2011; matches residue_data._PKA_LOOKUP


def protonation_fraction(ph: float, pka: float = GLU_PKA) -> float:
    """
    Henderson-Hasselbalch fraction of the acid in its PROTONATED (neutral) form.

        f = 1 / (1 + 10^(pH - pKa))

    Monotonically decreasing in pH, as any single titratable site must be.
    """
    return 1.0 / (1.0 + 10.0 ** (ph - pka))


# Experimental pH values in the study. Fractions are computed, never typed in.
STUDY_PH_VALUES: tuple[float, ...] = (5.0, 5.5, 7.0)

PROPKA_PROTONATION: dict[float, float] = {
    ph: round(protonation_fraction(ph), 4) for ph in STUDY_PH_VALUES
}
# -> {5.0: 0.1051, 5.5: 0.0358, 7.0: 0.0012}

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
