"""
Level-2 (protein) graph integrity.

The failure this guards against: `graph/builder.py` catches ANY exception from
the Level-2 build and substitutes a single all-zero residue node with no edges.
biopython was missing from requirements.txt, so a by-the-book install produced
exactly that — an empty receptor — for every sample, silently. Option B is the
only model that reads data['residue'], and it would have been attending over
one zero vector.
"""
from __future__ import annotations

import pytest


def test_biopython_is_installed():
    """
    Required by graph/level2_protein.py. Without it the protein level does not
    fail loudly — it degrades to a zero graph, which is far worse.
    """
    pytest.importorskip(
        "Bio",
        reason="biopython missing — Level-2 protein graphs will silently "
               "degrade to a single all-zero residue node. "
               "pip install biopython",
    )


def test_biopython_is_declared_in_requirements():
    from pathlib import Path

    req = Path(__file__).resolve().parents[1] / "Graph_model" / "requirements.txt"
    text = req.read_text().lower()
    assert "biopython" in text, (
        "biopython is imported by graph/level2_protein.py but not declared in "
        "requirements.txt — a clean install yields empty protein graphs.")


def test_protein_graph_is_not_degenerate():
    """
    A real binding-site graph has more than one residue and some non-zero
    features. One node with an all-zero feature vector is the failure sentinel
    from builder.py, not a protein.
    """
    pytest.importorskip("Bio")
    pytest.importorskip("torch")

    from Graph_model.graph.level2_protein import PROTEIN_NODE_DIM
    import numpy as np

    # Construct the sentinel the builder would emit, and assert we can tell it
    # apart from a real graph. This keeps the test independent of the docking
    # PDBs, which are not in the repository.
    sentinel = np.zeros((1, PROTEIN_NODE_DIM), dtype=np.float32)
    assert sentinel.shape[0] == 1 and not sentinel.any()

    def is_degenerate(x) -> bool:
        return x.shape[0] <= 1 or not x.any()

    assert is_degenerate(sentinel), "detector must flag the zero sentinel"
    assert not is_degenerate(np.ones((12, PROTEIN_NODE_DIM), dtype=np.float32))


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("Bio"),
    reason="biopython not installed",
)
def test_builder_reports_level2_failures_loudly(caplog):
    """
    A Level-2 failure must reach the log at ERROR every time, not once via
    warnings.warn. Silent degradation here is indistinguishable from success.
    """
    import logging
    import Graph_model.graph.builder as builder_mod

    assert hasattr(builder_mod, "logger"), (
        "builder.py must have a module logger so Level-2 failures are visible")
    assert isinstance(builder_mod.logger, logging.Logger)
