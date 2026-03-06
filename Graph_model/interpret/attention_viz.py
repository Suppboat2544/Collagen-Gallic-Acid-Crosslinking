"""
Graph_model.interpret.attention_viz
=====================================
Cross-attention binding-site rediscovery (Option B).

After Option B's forward pass, ``model.last_attn_weights`` holds per-graph
cross-attention matrices [N_atoms × N_residues].  This module extracts,
normalises, and ranks residue attention to identify the binding-site residues
that the model deems most important — *without* explicit residue identity
supervision.

Scientific hypothesis
---------------------
For gallic acid at GLU_cluster22, the top-attended residues should be
ARG496, ARG499, GLY497, LYS723, GLU724 — the same six residues identified
from docking PLIP analysis (Figure 2).

If reproduced, this is a powerful independent cross-validation result.

Usage
-----
    from Graph_model.interpret.attention_viz import (
        extract_attention, rank_residues, attention_heatmap_data
    )
    model = OptionB(cfg)
    model.eval()
    out, attn_list = model(data)
    attn = extract_attention(model)
    ranked = rank_residues(attn[0], residue_labels)
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from torch import Tensor


def extract_attention(
    model,
    graph_idx: Optional[int] = None,
) -> list[dict]:
    """
    Extract cross-attention weights from an OptionB model after forward().

    Parameters
    ----------
    model     : OptionB instance (must have .last_attn_weights populated)
    graph_idx : if specified, return only that graph's attention; else all

    Returns
    -------
    list of dicts, each with:
        'attn_weights'  : Tensor [N_atoms, N_residues]
        'graph_idx'     : int
    """
    if not hasattr(model, 'last_attn_weights'):
        # If wrapped in HeteroscedasticWrapper, dig into .base
        base = getattr(model, 'base', model)
        if not hasattr(base, 'last_attn_weights'):
            raise AttributeError(
                "Model has no 'last_attn_weights'. "
                "Did you run model.forward() with an OptionB model?"
            )
        attn_list = base.last_attn_weights
    else:
        attn_list = model.last_attn_weights

    if graph_idx is not None:
        return [a for a in attn_list if a.get('graph_idx') == graph_idx]
    return attn_list


def rank_residues(
    attn_entry:     dict,
    residue_labels: list[str],
    top_k:          int = 10,
    aggregate:      str = 'max',
) -> list[dict]:
    """
    Rank binding-site residues by attention weight.

    Parameters
    ----------
    attn_entry      : single dict from extract_attention()
    residue_labels  : list of residue labels (e.g. ['ARG496A', 'GLY497A', ...])
    top_k           : number of top residues to return
    aggregate       : how to reduce atom→residue attention matrix to per-residue
                      score.  'max' = max over atoms; 'mean' = mean over atoms.

    Returns
    -------
    list of dicts: [{'rank', 'residue', 'score', 'top_atoms'}, ...]
    """
    weights = attn_entry.get('attn_weights')
    if weights is None:
        return []

    if isinstance(weights, Tensor):
        W = weights.detach().cpu().numpy()
    else:
        W = np.asarray(weights)

    N_atoms, N_res = W.shape
    n_res = min(N_res, len(residue_labels))

    # Aggregate over atoms
    if aggregate == 'max':
        scores = W[:, :n_res].max(axis=0)     # [N_res]
    elif aggregate == 'mean':
        scores = W[:, :n_res].mean(axis=0)
    else:
        scores = W[:, :n_res].sum(axis=0)

    ranked = sorted(range(n_res), key=lambda i: -scores[i])[:top_k]

    results = []
    for rank, idx in enumerate(ranked):
        # Which atoms attend most to this residue?
        atom_scores = W[:, idx]
        top_atoms = sorted(range(N_atoms), key=lambda a: -atom_scores[a])[:3]
        results.append({
            'rank':      rank + 1,
            'residue':   residue_labels[idx],
            'score':     float(scores[idx]),
            'top_atoms': top_atoms,
        })
    return results


def attention_heatmap_data(
    attn_entry:     dict,
    atom_labels:    Optional[list[str]] = None,
    residue_labels: Optional[list[str]] = None,
) -> dict:
    """
    Prepare attention weights for heatmap visualization.

    Returns dict with 'matrix' (numpy), 'atom_labels', 'residue_labels',
    ready for seaborn.heatmap or matplotlib.imshow.
    """
    weights = attn_entry.get('attn_weights')
    if weights is None:
        return {'matrix': np.zeros((0, 0)), 'atom_labels': [], 'residue_labels': []}

    if isinstance(weights, Tensor):
        W = weights.detach().cpu().numpy()
    else:
        W = np.asarray(weights)

    N_atoms, N_res = W.shape
    if atom_labels is None:
        atom_labels = [f"atom_{i}" for i in range(N_atoms)]
    if residue_labels is None:
        residue_labels = [f"res_{j}" for j in range(N_res)]

    return {
        'matrix':         W,
        'atom_labels':    atom_labels[:N_atoms],
        'residue_labels': residue_labels[:N_res],
    }


def validate_binding_site_rediscovery(
    ranked:            list[dict],
    expected_residues: list[str],
    top_k:             int = 6,
) -> dict:
    """
    Check if the model's top-attended residues match the expected binding site.

    Parameters
    ----------
    ranked            : output from rank_residues()
    expected_residues : ground-truth binding-site residues from PLIP / Figure 2
    top_k             : how many model-predicted residues to check

    Returns
    -------
    dict with:
        'recall_at_k'     : fraction of expected residues found in top-k
        'precision_at_k'  : fraction of top-k that are expected residues
        'matched'         : list of matched residue names
        'missed'          : list of expected residues not in top-k
    """
    predicted = {r['residue'] for r in ranked[:top_k]}
    expected  = set(expected_residues)

    matched = predicted & expected
    missed  = expected - predicted

    recall    = len(matched) / max(len(expected), 1)
    precision = len(matched) / max(len(predicted), 1)

    return {
        'recall_at_k':    recall,
        'precision_at_k': precision,
        'matched':        sorted(matched),
        'missed':         sorted(missed),
    }
