"""
Protonation-physics tests.

Regression context: the pH axis was encoded from a hardcoded table

    {5.0: 0.85, 5.5: 0.15, 7.0: 0.02}

which no single pKa can produce. Henderson-Hasselbalch with f = 0.85 at pH 5.0
implies pKa 5.75, which then predicts 0.64 at pH 5.5 and 0.05 at pH 7.0. The
table was also a second, contradictory pH model competing with the correct
Henderson-Hasselbalch implementation in Graph_model/graph/residue_data.py --
and the table was the one reaching the network.

These tests assert the thermodynamics rather than the specific numbers, so they
keep holding if the pKa is later replaced with a PropKa-derived value.
"""
from __future__ import annotations

import math

import pytest

from Graph_model.data.config import (
    GLU_PKA, PROPKA_PROTONATION, STUDY_PH_VALUES, protonation_fraction,
)


# ── Henderson-Hasselbalch invariants ─────────────────────────────────────────

def test_fraction_is_monotonically_decreasing_in_ph():
    """A single titratable site deprotonates as pH rises. Always."""
    fractions = [protonation_fraction(ph) for ph in
                 [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 9.0]]
    for a, b in zip(fractions, fractions[1:]):
        assert b < a, f"protonation increased with pH: {fractions}"


def test_fraction_is_one_half_at_the_pka():
    assert protonation_fraction(GLU_PKA) == pytest.approx(0.5)


def test_fraction_is_bounded():
    for ph in (-2.0, 0.0, 7.0, 14.0, 20.0):
        f = protonation_fraction(ph)
        assert 0.0 <= f <= 1.0, f"pH {ph} gave f={f}"


def test_study_table_is_derived_not_typed():
    """
    Every tabulated fraction must equal the Henderson-Hasselbalch value for the
    declared pKa. This is what stops a hand-entered number drifting back in.
    """
    for ph in STUDY_PH_VALUES:
        assert PROPKA_PROTONATION[ph] == pytest.approx(
            protonation_fraction(ph), abs=1e-4), (
            f"pH {ph}: table says {PROPKA_PROTONATION[ph]}, "
            f"HH says {protonation_fraction(ph):.4f}"
        )


def test_the_old_hardcoded_table_is_thermodynamically_impossible():
    """
    Pins the reasoning, so nobody restores the old numbers believing they were
    merely imprecise.

    Over half a pH unit, HH bounds any single site to at most 0.76 -> 0.24, and
    only when centred on its pKa. The old table claimed 0.85 -> 0.15.
    """
    best_possible_drop = 0.0
    for pka_candidate in [x / 100 for x in range(300, 900)]:
        hi = 1.0 / (1.0 + 10.0 ** (5.0 - pka_candidate))
        lo = 1.0 / (1.0 + 10.0 ** (5.5 - pka_candidate))
        best_possible_drop = max(best_possible_drop, hi - lo)

    assert best_possible_drop < (0.85 - 0.15), (
        "the old table's pH 5.0 -> 5.5 drop should exceed what any single pKa "
        f"can produce, but the maximum achievable drop is {best_possible_drop:.3f}"
    )

    # And no single pKa reproduces all three old values at once.
    for pka_candidate in [x / 100 for x in range(300, 900)]:
        f = lambda ph: 1.0 / (1.0 + 10.0 ** (ph - pka_candidate))  # noqa: E731
        if (abs(f(5.0) - 0.85) < 0.05 and abs(f(5.5) - 0.15) < 0.05
                and abs(f(7.0) - 0.02) < 0.05):
            pytest.fail(f"pKa {pka_candidate} reproduces the old table")


# ── One pH model, not two ────────────────────────────────────────────────────

def test_config_agrees_with_the_residue_level_implementation():
    """
    Graph_model/graph/residue_data.py carries the residue-level Henderson-
    Hasselbalch. The condition encoder must not disagree with it for GLU.
    """
    from Graph_model.graph.residue_data import protonation_fraction as residue_hh

    for ph in (4.0, 5.0, 5.5, 6.0, 7.0):
        assert protonation_fraction(ph) == pytest.approx(
            residue_hh("GLU", ph), abs=1e-9), (
            f"pH {ph}: config and residue_data give different answers for GLU"
        )


def test_glu_pka_matches_the_residue_table():
    from Graph_model.graph.residue_data import pka

    assert GLU_PKA == pytest.approx(pka("GLU")), (
        "GLU_PKA must stay tied to the referenced pKa table, or the two pH "
        "models drift apart again"
    )


# ── Encoder behaviour ────────────────────────────────────────────────────────

def test_encoder_uses_the_titration_curve_off_study():
    """
    Off-study pH used to snap to the nearest tabulated point, making the
    feature a step function where the physics is smooth.
    """
    from Graph_model.data.features.conditions import ConditionEncoder

    enc = ConditionEncoder(strict=False)
    a, b, c = enc._encode_ph(5.6), enc._encode_ph(6.0), enc._encode_ph(6.4)
    assert a > b > c, f"encoding not monotone off-study: {a}, {b}, {c}"
    assert len({a, b, c}) == 3, "distinct pH values collapsed to the same code"


def test_encoder_round_trips_through_decode():
    from Graph_model.data.features.conditions import ConditionEncoder

    enc = ConditionEncoder(strict=False)
    for ph in (4.5, 5.0, 5.5, 6.0, 6.5):
        assert enc.decode_ph(enc._encode_ph(ph)) == pytest.approx(ph, abs=0.02)


def test_strict_encoder_still_rejects_unknown_ph():
    from Graph_model.data.features.conditions import ConditionEncoder

    with pytest.raises(ValueError, match="Unknown pH"):
        ConditionEncoder(strict=True)._encode_ph(6.3)
