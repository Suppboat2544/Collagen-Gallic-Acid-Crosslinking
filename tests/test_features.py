"""
Feature-dimension invariants.

The repo has carried five different values for "the node feature dimension"
(35, 54, 74, 79, and a docstring claiming 32/33). Two featurisers genuinely
exist and they genuinely differ; the fix is to pin each constant to what its
own code produces, so a stale docstring can never again be mistaken for a
specification.
"""
from __future__ import annotations

import pytest

pytest.importorskip("rdkit")
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BENZENE = "c1ccccc1"
GALLIC = "OC(=O)c1cc(O)c(O)c(O)c1"


def test_level1_node_dim_is_35():
    """graph/level1_ligand.py — the featuriser every model consumes."""
    from Graph_model.graph.level1_ligand import (
        mol_to_ligand_graph, LIGAND_NODE_DIM, LIGAND_EDGE_DIM)

    assert LIGAND_NODE_DIM == 35
    assert LIGAND_EDGE_DIM == 13

    nodes, edge_index, edges = mol_to_ligand_graph(BENZENE)
    assert nodes.shape == (6, LIGAND_NODE_DIM)
    assert edge_index.shape[0] == 2
    assert edges.shape[1] == LIGAND_EDGE_DIM


def test_atom_featuriser_dim_is_54():
    """features/atom.py — the level-0 featuriser used only by data/dataset.py."""
    from Graph_model.data.features.atom import (
        atom_features, bond_features, ATOM_FEAT_DIM, BOND_FEAT_DIM)

    assert ATOM_FEAT_DIM == 54
    assert BOND_FEAT_DIM == 12

    mol = Chem.MolFromSmiles(BENZENE)
    assert atom_features(mol).shape == (6, ATOM_FEAT_DIM)
    assert bond_features(mol).shape[1] == BOND_FEAT_DIM


def test_the_two_featurisers_are_known_to_differ():
    """
    Guard against someone 'fixing' the mismatch by aligning the numbers without
    aligning the code. If these ever become equal, the docstrings in
    features/__init__.py must be rewritten too.
    """
    from Graph_model.graph.level1_ligand import LIGAND_NODE_DIM
    from Graph_model.data.features.atom import ATOM_FEAT_DIM

    assert LIGAND_NODE_DIM != ATOM_FEAT_DIM, (
        "featurisers now agree — update the dimension table in "
        "Graph_model/features/__init__.py")


def test_node_count_equals_heavy_atom_count():
    from Graph_model.graph.level1_ligand import mol_to_ligand_graph

    nodes, _, _ = mol_to_ligand_graph(GALLIC)
    assert nodes.shape[0] == Chem.MolFromSmiles(GALLIC).GetNumHeavyAtoms() == 12


def test_features_are_finite():
    """NaNs here propagate silently all the way to the loss."""
    import numpy as np
    from Graph_model.graph.level1_ligand import mol_to_ligand_graph
    from Graph_model.data.config import LIGAND_CATALOGUE

    for name, meta in LIGAND_CATALOGUE.items():
        nodes, _, edges = mol_to_ligand_graph(meta["smiles"])
        assert np.isfinite(nodes).all(), f"{name}: non-finite node features"
        assert np.isfinite(edges).all(), f"{name}: non-finite edge features"
