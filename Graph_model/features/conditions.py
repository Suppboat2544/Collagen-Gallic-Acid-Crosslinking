"""
Graph_model.data.features.conditions
======================================
Condition vector encoding for the three physicochemical axes:

    pH          → PropKa-derived neutral-GLU protonation fraction
    Temperature → normalised (T − 4) / 33   ∈ [0, 1]
    Box type    → integer index for nn.Embedding(8, 16)
    Receptor    → 0 (collagen) or 1 (MMP-1)

The combined condition tensor is shape [4]:
    [ph_enc, temp_enc, box_idx (float), receptor_flag]

Only box_idx is categorical (feeds an Embedding in the GNN model).
The other three are continuous scalars.

Usage
-----
>>> enc = ConditionEncoder()
>>> vec = enc.encode(ph=5.0, temp_C=25, box_label="GLU_cluster22", receptor="collagen")
>>> vec
array([0.85  , 0.636 , 0.    , 0.    ], dtype=float32)

>>> idx = enc.parse_box_type("ASP_GLU_cluster14")
>>> idx
4
"""

from __future__ import annotations

import re
from typing import Union

import numpy as np

from ..config import (
    PROPKA_PROTONATION,
    BOX_TYPE_VOCAB,
    PH_VALUES,
    TEMP_VALUES,
    RECEPTORS,
)

# ── Temperature normalisation constants ──────────────────────────────────────
_TEMP_MIN: float = float(min(TEMP_VALUES))   # 4 °C  → 0.0
_TEMP_MAX: float = float(max(TEMP_VALUES))   # 37 °C → 1.0
_TEMP_RANGE: float = _TEMP_MAX - _TEMP_MIN   # 33


class ConditionEncoder:
    """
    Encode (pH, temperature, docking_box, receptor) into a 4-element float32
    vector suitable for concatenation with graph-level embeddings.

    Parameters
    ----------
    strict : bool, default True
        If True, raise ValueError on unknown pH / box type / receptor.
        If False, fall back to nearest known value / index 7 (global_blind) / 0.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode(
        self,
        ph: Union[float, str],
        temp_C: Union[float, int],
        box_label: str,
        receptor: str = "collagen",
    ) -> np.ndarray:
        """
        Return a float32 array [ph_enc, temp_enc, box_idx, receptor_flag].

        Parameters
        ----------
        ph          : float — one of 5.0, 5.5, 7.0
        temp_C      : float or int — one of 4, 25, 37
        box_label   : str  — raw 'docking_box' column value
                             e.g. "GLU_LYS_cluster12" or "global_blind"
        receptor    : str  — "collagen" (default) or "mmp1"

        Returns
        -------
        np.ndarray float32 shape [4]
        """
        ph_enc       = self._encode_ph(float(ph))
        temp_enc     = self._encode_temp(float(temp_C))
        box_idx      = float(self.parse_box_type(box_label))
        rec_flag     = float(self._encode_receptor(receptor))
        return np.array([ph_enc, temp_enc, box_idx, rec_flag], dtype=np.float32)

    def encode_batch(self, records: list[dict]) -> np.ndarray:
        """
        Vectorised encode for a list of dicts with keys:
        'pH', 'temperature_C', 'docking_box', and optionally 'receptor'.

        Returns np.ndarray float32 [N, 4].
        """
        rows = []
        for r in records:
            receptor = r.get("receptor", "collagen")
            rows.append(self.encode(r["pH"], r["temperature_C"],
                                    r["docking_box"], receptor))
        return np.vstack(rows) if rows else np.zeros((0, 4), dtype=np.float32)

    # ── pH encoding ───────────────────────────────────────────────────────────

    def _encode_ph(self, ph: float) -> float:
        """
        Map pH → neutral-GLU protonation fraction using PropKa values.

        pH 5.0 → 0.85 (mostly protonated / neutral)
        pH 5.5 → 0.15 (mostly deprotonated)
        pH 7.0 → 0.02 (fully deprotonated)
        """
        if ph in PROPKA_PROTONATION:
            return PROPKA_PROTONATION[ph]
        if self.strict:
            raise ValueError(
                f"Unknown pH {ph}. Known values: {list(PROPKA_PROTONATION)}"
            )
        # nearest-neighbour fallback
        closest = min(PROPKA_PROTONATION, key=lambda x: abs(x - ph))
        return PROPKA_PROTONATION[closest]

    # ── Temperature encoding ─────────────────────────────────────────────────

    @staticmethod
    def _encode_temp(temp: float) -> float:
        """
        Normalise temperature to [0, 1] using min-max over {4, 25, 37}.

        T=4  → 0.000
        T=25 → 0.636
        T=37 → 1.000

        Clamps to [0, 1] for out-of-range values.
        """
        val = (temp - _TEMP_MIN) / _TEMP_RANGE
        return float(np.clip(val, 0.0, 1.0))

    # ── Box type encoding ─────────────────────────────────────────────────────

    def parse_box_type(self, box_label: str) -> int:
        """
        Parse the raw 'docking_box' column value to a BOX_TYPE_VOCAB integer.

        Strips trailing cluster numbers:  "GLU_LYS_cluster12" → "GLU_LYS_cluster"
        Handles "global_blind" directly.

        Parameters
        ----------
        box_label : str  e.g. "ASP_GLU_LYS_cluster5" or "global_blind"

        Returns
        -------
        int  in range [0, 7]
        """
        label = box_label.strip()
        # global blind has no number suffix
        if label == "global_blind":
            return BOX_TYPE_VOCAB["global_blind"]

        # strip trailing digits from cluster labels
        # "GLU_LYS_cluster15" → "GLU_LYS_cluster"
        canonical = re.sub(r"\d+$", "", label)

        if canonical in BOX_TYPE_VOCAB:
            return BOX_TYPE_VOCAB[canonical]

        if self.strict:
            raise ValueError(
                f"Unknown box label '{box_label}' → canonical '{canonical}'. "
                f"Known: {list(BOX_TYPE_VOCAB)}"
            )
        # unknown → global_blind fallback
        return BOX_TYPE_VOCAB["global_blind"]

    # ── Receptor encoding ─────────────────────────────────────────────────────

    def _encode_receptor(self, receptor: str) -> int:
        """
        Map receptor name to binary flag:  collagen → 0,  mmp1 → 1.
        """
        key = receptor.strip().lower()
        if key in RECEPTORS:
            return RECEPTORS[key]
        if self.strict:
            raise ValueError(
                f"Unknown receptor '{receptor}'. Known: {list(RECEPTORS)}"
            )
        return 0  # default to collagen

    # ── Utility: decode back ──────────────────────────────────────────────────

    @staticmethod
    def decode_ph(ph_enc: float) -> float:
        """Reverse lookup: protonation fraction → nearest pH value."""
        best = min(PROPKA_PROTONATION, key=lambda x: abs(PROPKA_PROTONATION[x] - ph_enc))
        return best

    @staticmethod
    def decode_temp(temp_enc: float) -> float:
        """Reverse normalisation: [0,1] → °C."""
        return temp_enc * _TEMP_RANGE + _TEMP_MIN
