"""
Split-integrity tests.

LOLO-CV is the only honest protocol for a 9-molecule dataset. These tests
assert that property directly on synthetic groups, so they run without torch
and without the docking CSVs.
"""
from __future__ import annotations

import pytest

from Graph_model.train.lolo_cv import LOLOCVSplitter


def _fake_groups(n_per_ligand: int = 20) -> dict[str, list[int]]:
    """9 ligands x n rows each, mimicking the real record layout."""
    from Graph_model.data.config import LIGAND_NAMES
    groups, idx = {}, 0
    for name in LIGAND_NAMES:
        groups[name] = list(range(idx, idx + n_per_ligand))
        idx += n_per_ligand
    return groups


def test_nine_folds_one_per_ligand():
    groups = _fake_groups()
    folds = list(LOLOCVSplitter(seed=0).split_from_groups(groups, []))
    assert len(folds) == 9
    assert {f.held_out_ligand for f in folds} == set(groups)


def test_no_index_leaks_between_splits():
    groups = _fake_groups()
    for fold in LOLOCVSplitter(seed=0).split_from_groups(groups, []):
        assert LOLOCVSplitter.verify_no_leakage(fold) is True


def test_held_out_ligand_is_entirely_absent_from_training():
    """
    The point of LOLO: not one row of the test ligand may appear in train or
    val, at any pH, temperature or box. Index-level disjointness is necessary
    but not sufficient — this checks ligand identity.
    """
    groups = _fake_groups()
    for fold in LOLOCVSplitter(seed=0).split_from_groups(groups, []):
        held = set(groups[fold.held_out_ligand])
        assert set(fold.test_idx) == held
        assert not (set(fold.train_idx) & held)
        assert not (set(fold.val_idx) & held)


def test_every_row_is_used_exactly_once_per_fold():
    groups = _fake_groups()
    total = sum(len(v) for v in groups.values())
    for fold in LOLOCVSplitter(seed=0).split_from_groups(groups, []):
        allocated = list(fold.train_idx) + list(fold.val_idx) + list(fold.test_idx)
        assert len(allocated) == len(set(allocated)), "a row appears twice"
        assert len(allocated) == total, "rows silently dropped from the fold"


def test_splits_are_deterministic_given_a_seed():
    groups = _fake_groups()
    a = [f.train_idx for f in LOLOCVSplitter(seed=7).split_from_groups(groups, [])]
    b = [f.train_idx for f in LOLOCVSplitter(seed=7).split_from_groups(groups, [])]
    assert a == b


def test_verify_no_leakage_actually_rejects_a_leaky_fold():
    """A guard that cannot fail is not a guard."""
    from Graph_model.train.lolo_cv import LOLOFold

    leaky = LOLOFold(
        fold=0,
        held_out_ligand="gallic_acid",
        train_idx=[1, 2, 3],
        val_idx=[4, 5],
        test_idx=[3, 6],          # 3 is also in train
    )
    with pytest.raises(AssertionError, match="LEAKAGE"):
        LOLOCVSplitter.verify_no_leakage(leaky)
