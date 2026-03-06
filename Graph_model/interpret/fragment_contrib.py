"""
Graph_model.interpret.fragment_contrib
========================================
Fragment Contribution Ranking for Option C.

After Option C's forward pass, ``frag_contrib`` contains per-fragment ΔG
contributions δ_i for each graph in the batch.  This module extracts, labels,
and ranks fragments by their binding energy contribution.

Scientific application
----------------------
For PGG (penta-galloyl glucose), rank the five galloyl arms by δ_i.
This answers: which arms can be removed to create a smaller, more diffusible
PGG analogue without major affinity loss?  This is a direct drug/biomaterial
design output.

Expected result:
    "Arm 3 (equatorial galloyl) contributes −2.1 ± 0.3 kcal/mol to GLU-site
    binding vs −1.4 ± 0.4 kcal/mol from the remaining four arms combined."

Usage
-----
    from Graph_model.interpret.fragment_contrib import (
        extract_fragment_contributions, label_fragments, rank_fragments
    )
    model = OptionC(cfg)
    delta_g, frag_contribs = model(data, smiles_list=smiles)
    ranked = rank_fragments(frag_contribs[0], smiles[0])
"""

from __future__ import annotations

from typing import Optional

import torch
import numpy as np
from torch import Tensor


def extract_fragment_contributions(
    frag_contribs: list[Tensor],
    graph_idx:     int = 0,
) -> Tensor:
    """
    Extract per-fragment ΔG contributions for one graph from batch output.

    Parameters
    ----------
    frag_contribs : list of [N_frag_i] tensors (from OptionC forward)
    graph_idx     : which graph in the batch

    Returns
    -------
    Tensor [N_frag] — per-fragment ΔG contribution (kcal/mol)
    """
    if graph_idx >= len(frag_contribs):
        raise IndexError(
            f"graph_idx={graph_idx} out of range (batch has {len(frag_contribs)} graphs)"
        )
    contrib = frag_contribs[graph_idx]
    if isinstance(contrib, Tensor):
        return contrib.detach().cpu()
    return torch.tensor(contrib)


def label_fragments(
    smiles:      str,
    max_frags:   int = 10,
) -> list[dict]:
    """
    Decompose a SMILES into BRICS fragments and return labels.

    Parameters
    ----------
    smiles    : SMILES string
    max_frags : maximum fragments (matches OptionC.cfg.max_fragments)

    Returns
    -------
    list of dicts: [{'frag_idx', 'smiles', 'n_atoms', 'atom_indices'}, ...]
    """
    from rdkit import Chem
    from rdkit.Chem.BRICS import FindBRICSBonds

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [{'frag_idx': 0, 'smiles': smiles, 'n_atoms': 0, 'atom_indices': []}]

    n_heavy = mol.GetNumAtoms()
    brics_bonds = list(FindBRICSBonds(mol))

    if not brics_bonds:
        return [{
            'frag_idx':     0,
            'smiles':       smiles,
            'n_atoms':      n_heavy,
            'atom_indices': list(range(n_heavy)),
        }]

    # Flood-fill fragmentation (same algorithm as option_c._brics_fragments)
    cut_set = {frozenset([b[0][0], b[0][1]]) for b in brics_bonds}
    visited = [-1] * n_heavy
    frag_id = 0

    for start in range(n_heavy):
        if visited[start] != -1:
            continue
        queue = [start]
        visited[start] = frag_id
        while queue:
            node = queue.pop()
            for nbr in mol.GetAtomWithIdx(node).GetNeighbors():
                j = nbr.GetIdx()
                if visited[j] == -1 and frozenset([node, j]) not in cut_set:
                    visited[j] = frag_id
                    queue.append(j)
        frag_id += 1

    n_frags = min(frag_id, max_frags)
    fragments = []

    for f in range(n_frags):
        atom_idx = [i for i, v in enumerate(visited) if v == f]
        # Generate fragment SMILES
        try:
            frag_smi = Chem.MolFragmentToSmiles(mol, atom_idx)
        except Exception:
            frag_smi = "?"
        fragments.append({
            'frag_idx':     f,
            'smiles':       frag_smi,
            'n_atoms':      len(atom_idx),
            'atom_indices': atom_idx,
        })

    return fragments


def rank_fragments(
    contributions: Tensor,
    smiles:        str,
    max_frags:     int = 10,
) -> list[dict]:
    """
    Rank fragments by ΔG contribution (most stabilising first).

    Parameters
    ----------
    contributions : [N_frag] tensor of δ_i values from OptionC
    smiles        : SMILES of the molecule
    max_frags     : cap on number of fragments

    Returns
    -------
    list of dicts sorted by δ_i (most negative = most stabilising):
        [{'rank', 'frag_idx', 'smiles', 'delta_g', 'pct_total', 'atom_indices'}, ...]
    """
    frags  = label_fragments(smiles, max_frags)
    deltas = contributions.cpu().numpy()

    n = min(len(frags), len(deltas))
    total = float(np.sum(deltas[:n]))

    results = []
    for i in range(n):
        pct = (float(deltas[i]) / total * 100) if abs(total) > 1e-9 else 0.0
        results.append({
            'frag_idx':     frags[i]['frag_idx'],
            'smiles':       frags[i]['smiles'],
            'delta_g':      float(deltas[i]),
            'pct_total':    pct,
            'atom_indices': frags[i]['atom_indices'],
        })

    # Sort by δ_i (most negative first = most stabilising)
    results.sort(key=lambda x: x['delta_g'])
    for rank, r in enumerate(results):
        r['rank'] = rank + 1

    return results


def summarise_pgg_arms(
    contributions: Tensor,
    smiles:        str = "OC(=O)c1cc(O)c(O)c(O)c1OC1OC(COC(=O)c2cc(O)c(O)c(O)c2)"
                         "C(OC(=O)c2cc(O)c(O)c(O)c2)C(OC(=O)c2cc(O)c(O)c(O)c2)"
                         "C1OC(=O)c1cc(O)c(O)c(O)c1",
) -> dict:
    """
    Specialised summary for PGG: identify glucose core vs galloyl arms.

    Returns dict with 'core_delta_g', 'arm_delta_gs' (list), 'dominant_arm_idx'.
    """
    ranked = rank_fragments(contributions, smiles)

    if len(ranked) <= 1:
        return {
            'core_delta_g':     ranked[0]['delta_g'] if ranked else 0.0,
            'arm_delta_gs':     [],
            'dominant_arm_idx': None,
        }

    # Heuristic: fragment with fewest atoms containing only C/O is the core
    core_idx  = None
    arms      = []
    for r in ranked:
        r_smi = r['smiles']
        n_atoms = len(r['atom_indices'])
        # fragments containing 'c1cc(O)c(O)c(O)c1' pattern are galloyl arms
        if 'c1cc' in r_smi.lower() or ('O' in r_smi and n_atoms >= 6):
            arms.append(r)
        else:
            if core_idx is None or n_atoms > len(ranked[core_idx - 1].get('atom_indices', [])):
                core_idx = r['frag_idx']

    arm_dgs = [a['delta_g'] for a in arms]
    dominant = min(range(len(arm_dgs)), key=lambda i: arm_dgs[i]) if arm_dgs else None

    return {
        'core_delta_g':     next((r['delta_g'] for r in ranked if r['frag_idx'] == core_idx), 0.0),
        'arm_delta_gs':     arm_dgs,
        'dominant_arm_idx': dominant,
        'n_fragments':      len(ranked),
    }
