"""
Graph_model.interpret.gradcam
================================
Graph-adapted Grad-CAM for atom-level importance — Problem 7a.

Registers a backward hook on the last GNN layer to capture gradients of the
prediction with respect to node activations, then weights the activations by
mean gradient (Selvaraju et al. 2017, adapted for graphs).

Scientific validation
---------------------
Compare Grad-CAM atom importance to PLIP interaction contacts from SMINA runs:
if atoms with highest Grad-CAM weight correspond to atoms forming H-bonds
or hydrophobic contacts in PLIP, the GNN has learned physically meaningful
representations (highest scientific value per user priority).

Usage
-----
    >>> from Graph_model.interpret.gradcam import graph_gradcam, gradcam_vs_plip
    >>> importance = graph_gradcam(model, data)
    >>> overlap = gradcam_vs_plip(importance['atom_cam'], plip_atom_indices)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData, Batch

logger = logging.getLogger(__name__)


def _find_last_gnn_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """
    Find the last GNN/message-passing layer in the model.

    Strategy: walk named_modules in reverse, pick the first that looks like
    a GNN conv or attention layer.
    """
    candidates = []
    gnn_keywords = {"conv", "gat", "gect", "interaction", "propagation",
                    "egnn", "graphormer", "layer"}

    for name, module in model.named_modules():
        mod_type = type(module).__name__.lower()
        if any(kw in mod_type or kw in name.lower() for kw in gnn_keywords):
            # Skip containers
            if isinstance(module, (nn.ModuleList, nn.Sequential)):
                continue
            candidates.append((name, module))

    if not candidates:
        # Fallback: last non-trivial module
        for name, module in model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                candidates.append((name, module))

    if not candidates:
        raise ValueError("Could not find a GNN layer to hook for Grad-CAM")

    return candidates[-1]


def graph_gradcam(
    model:       nn.Module,
    data:        HeteroData,
    target_idx:  int = 0,
    layer_name:  Optional[str] = None,
    relu_cam:    bool = True,
) -> Dict[str, Any]:
    """
    Compute Graph Grad-CAM for ligand atoms → predicted ΔG.

    Parameters
    ----------
    model      : trained model (any architecture)
    data       : single HeteroData graph
    target_idx : output index to attribute (0 = ΔG / mu)
    layer_name : specific layer name to hook (if None, auto-detect last GNN)
    relu_cam   : apply ReLU to CAM scores (only positive attribution)

    Returns
    -------
    dict:
        'atom_cam'       : [N_atoms]  per-atom Grad-CAM importance
        'atom_cam_norm'  : [N_atoms]  min-max normalized to [0, 1]
        'activations'    : [N_atoms, D]  activations at hooked layer
        'gradients'      : [N_atoms, D]  gradients at hooked layer
        'layer_name'     : str  name of hooked layer
    """
    model.eval()

    # Ensure batched
    if not hasattr(data['ligand'], 'batch') or data['ligand'].batch is None:
        data = Batch.from_data_list([data])

    # Find layer to hook
    if layer_name is not None:
        target_layer = dict(model.named_modules())[layer_name]
        hook_name = layer_name
    else:
        hook_name, target_layer = _find_last_gnn_layer(model)

    # Storage for hook outputs
    activations: list[Tensor] = []
    gradients:   list[Tensor] = []

    def forward_hook(module, inp, out):
        # out might be a tuple (GATv2Conv returns tuple)
        if isinstance(out, tuple):
            out = out[0]
        activations.append(out.detach())

    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, tuple):
            g = grad_output[0]
        else:
            g = grad_output
        if g is not None:
            gradients.append(g.detach())

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward
        data_device = next(model.parameters()).device
        data = data.to(data_device)

        # Enable grad for backward
        for p in model.parameters():
            p.requires_grad_(True)

        out = model(data)

        # Handle different return types
        if isinstance(out, dict):
            pred = out.get("collagen", out.get("pred", list(out.values())[0]))
        elif isinstance(out, (tuple, list)):
            pred = out[0]
        else:
            pred = out

        # Select target scalar
        if pred.dim() > 1:
            target = pred[0, target_idx]
        else:
            target = pred[target_idx]

        # Backward
        model.zero_grad()
        target.backward(retain_graph=True)

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    if not activations or not gradients:
        logger.warning("Grad-CAM hooks captured no activations/gradients. "
                       "Layer '%s' may not be on the computation path.", hook_name)
        n_atoms = data['ligand'].x.shape[0]
        return {
            "atom_cam":      torch.zeros(n_atoms),
            "atom_cam_norm": torch.zeros(n_atoms),
            "activations":   torch.zeros(n_atoms, 1),
            "gradients":     torch.zeros(n_atoms, 1),
            "layer_name":    hook_name,
        }

    act  = activations[0]    # [N_atoms, D]
    grad = gradients[0]      # [N_atoms, D]

    # Ensure same shape
    if act.shape != grad.shape:
        min_d = min(act.shape[-1], grad.shape[-1])
        act  = act[..., :min_d]
        grad = grad[..., :min_d]

    # Grad-CAM: weight = mean gradient per channel, then weighted sum
    # α_k = (1/N) Σ_i ∂y/∂A_{i,k}    (global average pooling of grads)
    weights = grad.mean(dim=0)            # [D]
    cam = (act * weights.unsqueeze(0)).sum(dim=-1)   # [N_atoms]

    if relu_cam:
        cam = torch.relu(cam)

    # Normalize to [0, 1]
    cam_min = cam.min()
    cam_max = cam.max()
    if cam_max - cam_min > 1e-8:
        cam_norm = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam_norm = torch.zeros_like(cam)

    return {
        "atom_cam":      cam.cpu(),
        "atom_cam_norm": cam_norm.cpu(),
        "activations":   act.cpu(),
        "gradients":     grad.cpu(),
        "layer_name":    hook_name,
    }


def gradcam_vs_plip(
    atom_cam:         Tensor,
    plip_atom_indices: Sequence[int],
    top_k:            int = 10,
) -> Dict[str, float]:
    """
    Compare Grad-CAM importance to PLIP interaction contacts.

    Parameters
    ----------
    atom_cam          : [N_atoms] Grad-CAM scores
    plip_atom_indices : list of atom indices involved in PLIP contacts
    top_k             : how many top-CAM atoms to consider

    Returns
    -------
    dict:
        'precision_at_k'  : fraction of top-k CAM atoms that are PLIP contacts
        'recall_at_k'     : fraction of PLIP contacts found in top-k
        'jaccard'         : Jaccard similarity between top-k and PLIP set
        'mean_cam_plip'   : mean CAM score at PLIP contact atoms
        'mean_cam_other'  : mean CAM score at non-PLIP atoms
        'enrichment_ratio': mean_cam_plip / mean_cam_other
    """
    cam = atom_cam.cpu().numpy() if isinstance(atom_cam, Tensor) else atom_cam
    n_atoms = len(cam)
    plip_set = set(plip_atom_indices)

    # Top-k atoms by CAM score
    top_k = min(top_k, n_atoms)
    top_indices = set(cam.argsort()[-top_k:].tolist())

    # Metrics
    tp = len(top_indices & plip_set)
    precision = tp / max(len(top_indices), 1)
    recall    = tp / max(len(plip_set), 1)
    union     = len(top_indices | plip_set)
    jaccard   = tp / max(union, 1)

    # Mean CAM at PLIP vs non-PLIP
    plip_indices = [i for i in plip_set if i < n_atoms]
    other_indices = [i for i in range(n_atoms) if i not in plip_set]

    mean_plip  = float(cam[plip_indices].mean()) if plip_indices else 0.0
    mean_other = float(cam[other_indices].mean()) if other_indices else 0.0
    enrichment = mean_plip / max(mean_other, 1e-8)

    return {
        "precision_at_k":   precision,
        "recall_at_k":      recall,
        "jaccard":          jaccard,
        "mean_cam_plip":    mean_plip,
        "mean_cam_other":   mean_other,
        "enrichment_ratio": enrichment,
        "top_k":            top_k,
        "n_plip_contacts":  len(plip_set),
    }


def batch_gradcam(
    model:   nn.Module,
    dataset: Sequence[HeteroData],
    top_k:   int = 10,
) -> List[Dict[str, Any]]:
    """
    Run Grad-CAM on multiple graphs and return per-graph results.

    Parameters
    ----------
    model   : trained model
    dataset : sequence of HeteroData objects
    top_k   : number of top atoms to report per graph

    Returns
    -------
    List of dicts, each with:
        'graph_idx', 'atom_cam', 'top_atoms' (indices), 'top_scores'
    """
    results = []
    for idx, data in enumerate(dataset):
        try:
            cam_result = graph_gradcam(model, data)
            cam = cam_result["atom_cam"]
            k = min(top_k, len(cam))
            top_vals, top_idx = torch.topk(cam, k)
            results.append({
                "graph_idx":  idx,
                "atom_cam":   cam,
                "top_atoms":  top_idx.tolist(),
                "top_scores": top_vals.tolist(),
                "layer_name": cam_result["layer_name"],
            })
        except Exception as e:
            logger.warning("Grad-CAM failed for graph %d: %s", idx, e)
            results.append({
                "graph_idx": idx,
                "atom_cam":  torch.zeros(1),
                "top_atoms": [],
                "top_scores": [],
                "error": str(e),
            })
    return results
