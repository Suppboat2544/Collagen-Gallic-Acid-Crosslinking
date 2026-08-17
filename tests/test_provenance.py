"""
Ligand-identity and results-provenance tests.

Regression context: nothing ever compared the molecule named in the catalogue
against the molecule the docking campaign actually read from disk. Three
disagreements were sitting in the released data -- most consequentially
`pentagalloylglucose`, whose input .sdf holds four disconnected fragments so
that docking scored only the largest of them, a tri-O-galloylglucose with three
galloyl units rather than the named five.

The structure files live in the data root, not the repository, so the
data-dependent tests skip when COLLAGEN_DATA_ROOT is not pointed at them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from Graph_model.data.config import ANCHOR_DIR, LIGAND_CATALOGUE  # noqa: E402
from Graph_model.data.provenance import (  # noqa: E402
    KNOWN_MISMATCHES, check_structures, format_report,
)


requires_structures = pytest.mark.skipif(
    not any((ANCHOR_DIR / f"{n}.sdf").exists() for n in LIGAND_CATALOGUE),
    reason="ligand structure files not present; set COLLAGEN_DATA_ROOT",
)


# ── The catalogue must at least agree with itself ────────────────────────────

def test_catalogue_is_internally_consistent():
    """
    Declared mw / n_ha must match what RDKit reads from the declared SMILES.

    Regression: NHS_ester_intermediate declared mw 285.22 and n_ha 10, which
    matched neither each other nor its own SMILES (228.25 / 16).
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    for name, rec in LIGAND_CATALOGUE.items():
        mol = Chem.MolFromSmiles(rec["smiles"])
        assert mol is not None, f"{name}: SMILES does not parse"
        assert Descriptors.MolWt(mol) == pytest.approx(rec["mw"], abs=0.05), (
            f"{name}: declared mw {rec['mw']}, SMILES gives "
            f"{Descriptors.MolWt(mol):.2f}"
        )
        assert mol.GetNumHeavyAtoms() == rec["n_ha"], (
            f"{name}: declared n_ha {rec['n_ha']}, SMILES gives "
            f"{mol.GetNumHeavyAtoms()}"
        )


def test_every_catalogue_ligand_is_a_single_connected_molecule():
    """
    A disconnected SMILES silently becomes "whichever fragment docking kept".
    That is exactly how pentagalloylglucose became tri-O-galloylglucose.
    """
    from rdkit import Chem

    for name, rec in LIGAND_CATALOGUE.items():
        mol = Chem.MolFromSmiles(rec["smiles"])
        n_frags = len(Chem.GetMolFrags(mol))
        assert n_frags == 1, f"{name}: catalogue SMILES has {n_frags} fragments"


# ── Catalogue vs what was docked ─────────────────────────────────────────────

@requires_structures
def test_identity_check_runs_and_covers_every_ligand():
    checks = check_structures()
    assert {c.ligand for c in checks} == set(LIGAND_CATALOGUE)
    assert format_report(checks).strip()


@requires_structures
def test_known_mismatches_are_still_detected():
    """
    These are live defects in the released data, not historical ones. If a
    re-dock fixes them, this test fails and KNOWN_MISMATCHES should be pruned
    -- which is the point.
    """
    checks = {c.ligand: c for c in check_structures()}
    for name in KNOWN_MISMATCHES:
        c = checks[name]
        if c.status in ("missing", "unreadable"):
            pytest.skip(f"{name} structure unavailable")
        assert c.status == "mismatch", (
            f"{name} now matches its docked structure -- if it was re-docked, "
            f"remove it from KNOWN_MISMATCHES"
        )


@requires_structures
def test_no_new_mismatches_beyond_the_known_three():
    """A mismatch outside KNOWN_MISMATCHES is a fresh defect."""
    unexpected = {
        c.ligand: c.detail for c in check_structures()
        if c.status == "mismatch" and c.ligand not in KNOWN_MISMATCHES
    }
    assert not unexpected, f"new ligand-identity mismatches: {unexpected}"


@requires_structures
def test_pgg_docked_structure_is_short_two_galloyl_units():
    """
    Pins the specific consequence: every affinity labelled
    'pentagalloylglucose' belongs to a molecule with three galloyl units.
    """
    from rdkit import Chem

    path = ANCHOR_DIR / "pentagalloylglucose.sdf"
    if not path.exists():
        pytest.skip("pentagalloylglucose.sdf not available")

    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)), None)
    assert mol is not None
    frags = Chem.GetMolFrags(mol, asMols=True)
    assert len(frags) > 1, "input is connected -- has it been re-docked?"

    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    galloyl = Chem.MolFromSmarts("[OX2H]c1cc(cc([OX2H])c1[OX2H])[CX3]=[OX1]")
    assert len(largest.GetSubstructMatches(galloyl)) == 3, (
        "the docked fragment should carry three galloyl units, not the five "
        "its label claims"
    )


def test_missing_structures_are_reported_not_silently_passed():
    """An absent file must never read as 'match'."""
    checks = check_structures(structure_dir=Path("/nonexistent-structure-dir"))
    assert checks and all(c.status == "missing" for c in checks)
    assert not any(c.ok for c in checks)


# ── Results provenance ───────────────────────────────────────────────────────

def test_run_provenance_records_what_is_needed_to_trust_a_number():
    torch = pytest.importorskip("torch")  # noqa: F841
    from Graph_model.train.run_training import run_provenance

    prov = run_provenance()
    for key in ("python", "git_commit", "feature_schema_version",
                "ligand_identity"):
        assert key in prov, f"provenance is missing {key!r}"
    assert isinstance(prov["feature_schema_version"], int)
