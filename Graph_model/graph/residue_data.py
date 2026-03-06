"""
Graph_model.graph.residue_data
================================
Amino acid property tables referenced by Level 2 (protein context graph).

Includes:
  AA_INDEX          20-AA canonical ordering  (index for one-hot)
  PROPKA_PKA        residue-type pKa from PropKa defaults
  KYTE_DOOLITTLE    hydrophobicity scale
  CHARGE_AT_PH      callable: charge(resname, pH) → {-1, 0, +1}
  HBOND_DONOR       residue types that act as H-bond donors
  HBOND_ACCEPTOR    residue types that act as H-bond acceptors
"""

from __future__ import annotations

import math
from typing import Union

# ── 20 standard amino acids ──────────────────────────────────────────────────
AA_INDEX: dict[str, int] = {
    "ALA": 0,  "ARG": 1,  "ASN": 2,  "ASP": 3,  "CYS": 4,
    "GLN": 5,  "GLU": 6,  "GLY": 7,  "HIS": 8,  "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
N_AA: int = 20   # one-hot vector length

# Non-standard mappings (HIS tautomers, selenocysteine …)
_AA_ALIAS: dict[str, str] = {
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS",
    "CYX": "CYS", "CYM": "CYS",
    "GLH": "GLU", "ASH": "ASP",
    "LYN": "LYS",
    "SEC": "CYS",  # selenocysteine
    "MSE": "MET",  # selenomethionine
}


def aa_to_index(resname: str) -> int:
    """Return AA index 0-19, or 19 (VAL) if unrecognised (safe fallback)."""
    key = resname.strip().upper()
    key = _AA_ALIAS.get(key, key)
    return AA_INDEX.get(key, N_AA - 1)  # unknown → last slot


def aa_one_hot(resname: str) -> list[float]:
    """20-dim one-hot vector."""
    idx = aa_to_index(resname)
    v = [0.0] * N_AA
    v[idx] = 1.0
    return v


# ── PropKa-derived intrinsic pKa values ──────────────────────────────────────
# Source: Olsson et al., J. Chem. Theory Comput. 2011, with collagen-specific
# adjustments from SI-1.1 (GLU residues in GLU_cluster22 vicinity).
#
# Normalised to [0, 1] range:  pKa_norm = (pKa − 3) / 12  (range 3–15)
_PKA_LOOKUP: dict[str, float] = {
    "ASP":  3.80,   "GLU":  4.07,   "HIS":  6.50,
    "CYS":  8.30,   "LYS": 10.50,   "TYR": 10.46,
    "ARG": 12.50,   "SER": 13.60,   "THR": 14.15,
    "N_TERM": 8.0,  "C_TERM": 3.2,
}


def pka(resname: str, default: float = 14.0) -> float:
    """Intrinsic pKa for ionisable residues; 14.0 for neutral/unknown."""
    return _PKA_LOOKUP.get(resname.upper(), default)


def pka_normed(resname: str) -> float:
    """pKa normalised to [0, 1] using 3–15 range."""
    raw = pka(resname)
    return float((raw - 3.0) / 12.0)


def protonation_fraction(resname: str, ph: float) -> float:
    """
    Henderson–Hasselbalch protonation fraction (fraction in protonated form).
    p = 1 / (1 + 10^(pH − pKa))
    """
    k = _PKA_LOOKUP.get(resname.upper())
    if k is None:
        return 0.0
    return 1.0 / (1.0 + 10.0 ** (ph - k))


# ── Charge at given pH ────────────────────────────────────────────────────────
_CHARGE_SIGN: dict[str, int] = {
    "ASP": -1, "GLU": -1, "CYS": -1, "TYR": -1,   # lose proton → negative
    "LYS": +1, "ARG": +1, "HIS": +1,                 # gain proton  → positive
}


def charge_at_ph(resname: str, ph: float) -> float:
    """
    Fractional charge at given pH.
    Returns ~-1 (deprotonated acid) / ~+1 (protonated base) / 0 (neutral).
    """
    sign = _CHARGE_SIGN.get(resname.upper(), 0)
    if sign == 0:
        return 0.0
    frac_prot = protonation_fraction(resname, ph)
    if sign == -1:
        # acid: fully charged (−1) when deprotonated
        return -(1.0 - frac_prot)
    else:
        # base: fully charged (+1) when protonated
        return frac_prot


# ── Kyte–Doolittle hydrophobicity ─────────────────────────────────────────────
# Normalised from original KD scale (−4.5 to +4.5) to [0, 1]
_KD_RAW: dict[str, float] = {
    "ILE": 4.5,  "VAL": 4.2,  "LEU": 3.8,  "PHE": 2.8,  "CYS": 2.5,
    "MET": 1.9,  "ALA": 1.8,  "GLY": -0.4, "THR": -0.7, "SER": -0.8,
    "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
    "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5,
}
_KD_MIN, _KD_MAX = -4.5, 4.5
_KD_RANGE = _KD_MAX - _KD_MIN


def hydrophobicity(resname: str) -> float:
    """Kyte–Doolittle hydrophobicity normalised to [0, 1]."""
    raw = _KD_RAW.get(resname.upper(), 0.0)
    return (raw - _KD_MIN) / _KD_RANGE


# ── H-bond donor / acceptor classification ────────────────────────────────────
HBOND_DONOR: set[str] = {
    "ARG", "LYS", "ASN", "GLN", "HIS", "SER", "THR", "TRP", "TYR", "CYS",
}
HBOND_ACCEPTOR: set[str] = {
    "ASP", "GLU", "HIS", "ASN", "GLN", "SER", "THR", "TYR",
    "MET", "CYS",
}


def is_hbond_donor(resname: str) -> float:
    return float(resname.upper() in HBOND_DONOR)


def is_hbond_acceptor(resname: str) -> float:
    return float(resname.upper() in HBOND_ACCEPTOR)


# ── Summary vector (for use in Level 3 edge features) ─────────────────────────

def residue_feature_vector(resname: str, ph: float = 7.0) -> list[float]:
    """
    Return a 5-dim residue property vector:
        [pka_normed, charge_at_ph, hydrophobicity, is_hbond_donor, is_hbond_acceptor]
    """
    return [
        pka_normed(resname),
        charge_at_ph(resname, ph),
        hydrophobicity(resname),
        is_hbond_donor(resname),
        is_hbond_acceptor(resname),
    ]
