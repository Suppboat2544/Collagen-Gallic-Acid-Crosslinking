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

    History: with the old disconnected PGG string this read 9 against a
    declared 5 -- the five free gallic acids were each counted separately.
    """
    from Graph_model.data.features.galloyl import GalloylFragmentDetector

    # A galloyl unit is a 3,4,5-trihydroxybenzoyl group: a trihydroxyphenyl
    # ring bearing an acyl carbon. Gallic acid is one; pyrogallol is NOT (no
    # acyl); protocatechuic acid is not (only two OH).
    expected = {
        "pentagalloylglucose": 5,
        "gallic_acid": 1,
        "pyrogallol": 0,
        "protocatechuic_acid": 0,
    }
    for name, want in expected.items():
        mol = Chem.MolFromSmiles(LIGAND_CATALOGUE[name]["smiles"])
        counts = GalloylFragmentDetector.count_fragments(mol)
        assert counts["galloyl_strict"] == want, (
            f"{name}: detector found {counts['galloyl_strict']} galloyl units, "
            f"expected {want}"
        )

    # And the declared catalogue value must agree with the detector for the
    # ligands whose galloyl_units is a genuine galloyl count.
    mol = Chem.MolFromSmiles(LIGAND_CATALOGUE["pentagalloylglucose"]["smiles"])
    assert (GalloylFragmentDetector.count_fragments(mol)["galloyl_strict"]
            == LIGAND_CATALOGUE["pentagalloylglucose"]["galloyl_units"] == 5)


def test_strict_galloyl_pattern_matches_gallic_acid():
    """
    Gallic acid IS a galloyl unit. When this regressed, `galloyl_strict` was
    identically zero across the whole catalogue and its 1.00 weight in
    `galloyl_weighted` never fired.
    """
    from Graph_model.data.features.galloyl import GalloylFragmentDetector

    mol = Chem.MolFromSmiles(LIGAND_CATALOGUE["gallic_acid"]["smiles"])
    assert GalloylFragmentDetector.count_fragments(mol)["galloyl_strict"] == 1


def test_galloyl_classes_are_disjoint_and_not_double_counted():
    """
    galloyl / pyrogallol / catechol are strictly nested substructures, so a
    ring must be counted in exactly ONE of them.

    Regression: matches were counted per SMARTS hit and subtracted
    (`ca - g - py`). A benzene-1,2,3-triol yields TWO catechol hits, so one
    spurious catechol survived on every trihydroxy ring and each scored
    1.00 + 0.67 = 1.67 instead of 1.00 -- gallic acid 1.67, PGG 8.35.
    """
    from Graph_model.data.features.galloyl import GalloylFragmentDetector

    expected_weight = {
        "gallic_acid":         1.00,
        "pyrogallol":          1.00,
        "protocatechuic_acid": 0.67,
        "ellagic_acid":        1.34,   # two catechol rings
        "pentagalloylglucose": 5.00,
        "EDC":                 0.00,
    }
    for name, want in expected_weight.items():
        # Keep a reference to the Mol: RingInfo is invalidated once the parent
        # molecule is collected, so building it inline reports zero rings.
        mol = Chem.MolFromSmiles(LIGAND_CATALOGUE[name]["smiles"])
        counts = GalloylFragmentDetector.count_fragments(mol)
        assert counts["galloyl_weighted"] == pytest.approx(want, abs=1e-6), (
            f"{name}: galloyl_weighted={counts['galloyl_weighted']}, "
            f"expected {want}"
        )
        # No ring may be classified into more than one of the three nested
        # categories, so the classifications cannot outnumber the rings.
        total = counts["galloyl_strict"] + counts["pyrogallol"] + counts["catechol"]
        n_aromatic_rings = sum(
            1 for ring in mol.GetRingInfo().AtomRings()
            if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
        )
        assert total <= n_aromatic_rings, (
            f"{name}: {total} fragment classifications over "
            f"{n_aromatic_rings} aromatic rings -- a ring is counted twice"
        )


def test_phenolic_oh_counts_match_reference_chemistry():
    """
    `total_aromatic_oh` is a hydrogen-bond-donor capacity feature, so it must
    count hydroxyls -- not every oxygen touching an aromatic carbon.

    Regression: ellagic acid scored 8 against 4 actual, because RDKit perceives
    its fused lactone rings as aromatic and the counter accepted the two ester
    oxygens and two lactone carbonyls as phenolic OH.
    """
    from Graph_model.data.features.galloyl import GalloylFragmentDetector

    reference = {           # phenolic OH per molecule
        "gallic_acid":         3,
        "pyrogallol":          3,
        "protocatechuic_acid": 2,
        "ellagic_acid":        4,
        "pentagalloylglucose": 15,   # 5 rings x 3 OH
        "EDC":                 0,
        "NHS":                 0,
    }
    for name, want in reference.items():
        counts = GalloylFragmentDetector.count_fragments(
            Chem.MolFromSmiles(LIGAND_CATALOGUE[name]["smiles"]))
        assert counts["total_aromatic_oh"] == want, (
            f"{name}: total_aromatic_oh={counts['total_aromatic_oh']}, "
            f"reference value is {want}"
        )


def test_detector_returns_finite_counts_for_every_ligand():
    """The fragment layer consumes these; a KeyError or NaN here is silent."""
    from Graph_model.data.features.galloyl import GalloylFragmentDetector

    keys = ("galloyl_strict", "catechol", "pyrogallol",
            "total_aromatic_oh", "galloyl_weighted")
    for name, meta in LIGAND_CATALOGUE.items():
        counts = GalloylFragmentDetector.count_fragments(
            Chem.MolFromSmiles(meta["smiles"]))
        for k in keys:
            assert k in counts, f"{name}: detector did not return {k!r}"
            assert counts[k] == counts[k], f"{name}: {k} is NaN"
            assert counts[k] >= 0, f"{name}: {k} is negative"
