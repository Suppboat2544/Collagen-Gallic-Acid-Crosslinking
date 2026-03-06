"""
Graph_model.interpret.integrated_gradients
============================================
Atom-level importance maps via Integrated Gradients (Sundararajan et al. 2017).

Given a model and an input molecular graph, compute the attribution of each
atom's feature vector to the predicted ΔG.  The result is a per-atom scalar
importance score that can be overlaid on the 2D molecular structure.

Expected scientific results
----------------------------
  Gallic acid  : three OH groups of pyrogallol ring → highest positive importance
  EDC          : carbodiimide N=C=N group → highest importance
  PGG          : galloyl arms → high importance; glucose core → near-zero

Cross-validation opportunity
-----------------------------
If atom importance reproduces the PLIP interaction fingerprint from SMINA runs
(ARG496, ARG499, LYS723 contacts → OH groups), this is a novel cross-validation
between docking-derived interaction data and GNN gradient attribution.

Reference
---------
Sundararajan M., Taly A., Yan Q.
  "Axiomatic Attribution for Deep Networks."  ICML 2017.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData, Batch


def integrated_gradients(
    model:       nn.Module,
    data:        HeteroData,
    target_idx:  int = 0,
    n_steps:     int = 50,
    baseline:    Optional[Tensor] = None,
    internal_batch_size: int = 10,
) -> dict[str, Tensor]:
    """
    Compute Integrated Gradients for ligand atom features → predicted ΔG.

    Parameters
    ----------
    model       : trained model (OptionA/B/C/D or HeteroscedasticWrapper)
    data        : single HeteroData (unbatched; batch dim will be added)
    target_idx  : which output scalar to attribute (0 = ΔG / mu)
    n_steps     : number of interpolation steps (higher = more precise)
    baseline    : baseline atom features [N_atoms, 35]; default = zeros
    internal_batch_size : how many interpolation steps to batch at once

    Returns
    -------
    dict with:
        'atom_attr'       : [N_atoms]        per-atom scalar attribution (L2 norm)
        'atom_attr_raw'   : [N_atoms, 35]    per-atom per-feature attribution
        'atom_importance'  : [N_atoms]        absolute attribution (for ranking)
        'convergence_delta': float            completeness check (should be ~0)
    """
    model.eval()

    # Ensure we have a single graph (no batch dim)
    if not hasattr(data['ligand'], 'batch') or data['ligand'].batch is None:
        data = Batch.from_data_list([data])

    x_input = data['ligand'].x.clone().detach().float()   # [N, 35]
    N, D    = x_input.shape

    if baseline is None:
        x_base = torch.zeros_like(x_input)
    else:
        x_base = baseline.clone().detach().float()

    delta = x_input - x_base   # [N, D]

    # Accumulate gradients over interpolation steps
    total_grad = torch.zeros_like(x_input)   # [N, D]

    # Process in mini-batches of steps for memory efficiency
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x_input.device)

    for start in range(0, n_steps + 1, internal_batch_size):
        end = min(start + internal_batch_size, n_steps + 1)
        batch_alphas = alphas[start:end]
        k = len(batch_alphas)

        # Create k copies of the data, each with interpolated ligand features
        data_list = []
        for alpha in batch_alphas:
            d_copy = data.clone()
            d_copy['ligand'].x = x_base + alpha * delta
            data_list.append(d_copy)

        batch_data = Batch.from_data_list(data_list)
        batch_data['ligand'].x.requires_grad_(True)

        # Forward
        raw = model(batch_data)
        if isinstance(raw, dict):
            out = raw.get('mu', raw.get('collagen'))
        elif isinstance(raw, tuple):
            out = raw[0]
        else:
            out = raw

        # Select target output for each graph in the batch
        # out is [k, 1] or [k, n_outputs]
        if out.dim() == 2 and out.shape[1] > 1:
            target = out[:, target_idx].sum()
        else:
            target = out.sum()

        target.backward()

        grad = batch_data['ligand'].x.grad   # [k*N, D]
        if grad is not None:
            grad_per_step = grad.view(k, N, D)
            total_grad += grad_per_step.sum(dim=0)

    # Riemann approximation (trapezoidal would be: average consecutive pairs)
    avg_grad = total_grad / (n_steps + 1)

    # Attribution = delta * avg_gradient
    attr = delta * avg_grad   # [N, D]

    # Per-atom scalar: L2 norm across feature dimension
    atom_attr = attr.norm(dim=-1)            # [N]

    # Convergence delta: sum of attributions should ≈ F(x) - F(baseline)
    with torch.no_grad():
        data_input = data.clone()
        data_input['ligand'].x = x_input
        raw_i = model(data_input)
        f_input = _extract_scalar(raw_i, target_idx)

        data_base = data.clone()
        data_base['ligand'].x = x_base
        raw_b = model(data_base)
        f_base = _extract_scalar(raw_b, target_idx)

    convergence = (f_input - f_base).item() - attr.sum().item()

    return {
        'atom_attr':         atom_attr.detach(),
        'atom_attr_raw':     attr.detach(),
        'atom_importance':   atom_attr.abs().detach(),
        'convergence_delta': convergence,
    }


def atom_importance_ranking(
    attr: dict[str, Tensor],
    smiles: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Rank atoms by importance and annotate with chemical identity.

    Parameters
    ----------
    attr    : output from integrated_gradients()
    smiles  : SMILES string of the molecule
    top_k   : number of top atoms to return

    Returns
    -------
    list of dicts: [{'rank', 'atom_idx', 'symbol', 'importance', 'neighbors'}, ...]
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")

    importance = attr['atom_importance'].cpu().numpy()
    n = min(len(importance), mol.GetNumAtoms())

    ranked = sorted(range(n), key=lambda i: -importance[i])[:top_k]

    results = []
    for rank, idx in enumerate(ranked):
        atom = mol.GetAtomWithIdx(idx)
        nbrs = [mol.GetAtomWithIdx(nb.GetIdx()).GetSymbol()
                for nb in atom.GetNeighbors()]
        results.append({
            'rank':       rank + 1,
            'atom_idx':   idx,
            'symbol':     atom.GetSymbol(),
            'importance':  float(importance[idx]),
            'neighbors':  nbrs,
        })
    return results


def _extract_scalar(raw, target_idx: int = 0) -> Tensor:
    """Extract a single scalar from model output."""
    if isinstance(raw, dict):
        out = raw.get('mu', raw.get('collagen'))
    elif isinstance(raw, tuple):
        out = raw[0]
    else:
        out = raw
    if out.dim() == 2 and out.shape[1] > 1:
        return out[:, target_idx].sum()
    return out.sum()
