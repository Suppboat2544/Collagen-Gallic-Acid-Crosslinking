"""
Chemistry invariants for the ligand catalogue.

These require only rdkit + numpy — deliberately not torch — so they run in CI
without a GPU stack and without the docking CSVs.

Every one of these assertions failed against the catalogue as committed before
2026-08-11: PGG was 6 disconnected fragments (114 heavy atoms instead of 67),
ellagic acid was a monolactone (19 instead of 22), and three n_ha/mw values
disagreed with their own SMILES.
"""
from __future__ import annotations

import pytest

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger                      # noqa: E402
from rdkit.Chem import Descriptors, rdMolDescriptors  # noqa: E402

from Graph_model.data.config import LIGAND_CATALOGUE, LIGAND_NAMES  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ALL = sorted(LIGAND_CATALOGUE.items())
IDS = [k for k, _ in ALL]


@pytest.fixture(scope="module", params=ALL, ids=IDS)
def entry(request):
    name, meta = request.param
    mol = Chem.MolFromSmiles(meta["smiles"])
    assert mol is not None, f"{name}: SMILES does not parse"
    return name, meta, mol


def test_catalogue_has_nine_ligands():
    assert len(LIGAND_CATALOGUE) == 9
    assert len(LIGAND_NAMES) == 9


def test_smiles_is_a_single_connected_molecule(entry):
    """A '.'-separated SMILES is several molecules, not one ligand."""
    name, _, mol = entry
    frags = Chem.GetMolFrags(mol)
    assert len(frags) == 1, (
        f"{name}: SMILES parses as {len(frags)} disconnected fragments. "
        f"Every model would see these as one 'molecule' with no bonds between "
        f"the pieces."
    )


def test_heavy_atom_count_matches_metadata(entry):
    name, meta, mol = entry
    assert mol.GetNumHeavyAtoms() == meta["n_ha"], (
        f"{name}: SMILES has {mol.GetNumHeavyAtoms()} heavy atoms but "
        f"n_ha says {meta['n_ha']}"
    )


def test_molecular_weight_matches_metadata(entry):
    name, meta, mol = entry
    mw = Descriptors.MolWt(mol)
    assert abs(mw - meta["mw"]) < 0.6, (
        f"{name}: SMILES MW is {mw:.2f} but mw says {meta['mw']:.2f} "
        f"(formula {rdMolDescriptors.CalcMolFormula(mol)})"
    )


def test_pgg_is_the_real_molecule():
    """
    Pin PGG specifically — it is the headline ligand and it has been wrong
    twice, in two different ways.

    Fragment count alone is NOT a sufficient check: the old `PGG_SMILES_RDKIT`
    was a single connected fragment but a cyclopentane pentagallate, missing
    the pyranose ring oxygen and the C6 hydroxymethyl (C40H30O25, 910.66).
    """
    meta = LIGAND_CATALOGUE["pentagalloylglucose"]
    mol = Chem.MolFromSmiles(meta["smiles"])
    assert len(Chem.GetMolFrags(mol)) == 1
    assert mol.GetNumHeavyAtoms() == 67
    assert rdMolDescriptors.CalcMolFormula(mol) == "C41H32O26"
    assert abs(Descriptors.MolWt(mol) - 940.68) < 0.1

    # a pyranose ring: 5 C + 1 O, and it must actually be there
    ring_o = [a for a in mol.GetAtoms()
              if a.GetSymbol() == "O" and a.IsInRingSize(6)]
    assert ring_o, "PGG has no 6-membered ring oxygen — that is not a glucopyranose"


def test_ellagic_acid_is_a_dilactone():
    """The catalogue comment says 'dilactone'; the SMILES must agree."""
    meta = LIGAND_CATALOGUE["ellagic_acid"]
    mol = Chem.MolFromSmiles(meta["smiles"])
    lactone = Chem.MolFromSmarts("[#6](=O)[#8][#6]")
    assert len(mol.GetSubstructMatches(lactone)) >= 2, (
        "ellagic acid must contain two lactone bridges")
    assert rdMolDescriptors.CalcMolFormula(mol) == "C14H6O8"


def test_galloyl_unit_count_matches_smarts_detection():
    """
    `galloyl_units` drives the fragment graph. It must equal what the detector
    actually finds, or the fragment layer is built on a lie.

    With the old disconnected PGG string this found 9 units against a declared
    5 -- the five free gallic acids were each counted separately.
    """
    from Graph_model.features.galloyl import GalloylFragmentDetector

    # The pyrogallol pattern is what actually identifies galloyl rings today
    # (see the FIXME in features/galloyl.py: _SMARTS_GALLOYL is malformed and
    # matches nothing). These counts DO track galloyl_units, so pin them —
    # with the old disconnected PGG string this read 9 instead of 5.
    expected_rings = {
        "pentagalloylglucose": 5,
        "gallic_acid": 1,
        "pyrogallol": 1,
        "protocatechuic_acid": 0,   # catechol only, 2 OH — not a galloyl ring
    }
    for name, want in expected_rings.items():
        mol = Chem.MolFromSmiles(LIGAND_CATALOGUE[name]["smiles"])
        counts = GalloylFragmentDetector.count_fragments(mol)
        assert counts["pyrogallol"] == want, (
            f"{name}: detector found {counts['pyrogallol']} trihydroxyphenyl "
            f"rings, expected {want}"
        )


@pytest.mark.xfail(
    reason="_SMARTS_GALLOYL is malformed (1,3,4-trihydroxy, not 3,4,5) so it "
           "matches nothing — see FIXME in Graph_model/features/galloyl.py. "
           "Remove this xfail when the pattern is corrected and models re-run.",
    strict=True,
)
def test_strict_galloyl_pattern_matches_gallic_acid():
    """
    Gallic acid IS a galloyl unit. If the strict detector cannot find one in
    gallic acid, the `galloyl_strict` feature is dead and its weight in
    `galloyl_weighted` never fires.
    """
    from Graph_model.features.galloyl import GalloylFragmentDetector

    mol = Chem.MolFromSmiles(LIGAND_CATALOGUE["gallic_acid"]["smiles"])
    assert GalloylFragmentDetector.count_fragments(mol)["galloyl_strict"] == 1


def test_detector_returns_finite_counts_for_every_ligand():
    """The fragment layer consumes these; a KeyError or NaN here is silent."""
    from Graph_model.features.galloyl import GalloylFragmentDetector

    keys = ("galloyl_strict", "catechol", "pyrogallol",
            "total_aromatic_oh", "galloyl_weighted")
    for name, meta in LIGAND_CATALOGUE.items():
        counts = GalloylFragmentDetector.count_fragments(
            Chem.MolFromSmiles(meta["smiles"]))
        for k in keys:
            assert k in counts, f"{name}: detector did not return {k!r}"
            assert counts[k] == counts[k], f"{name}: {k} is NaN"
            assert counts[k] >= 0, f"{name}: {k} is negative"
