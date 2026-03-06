"""
Graph_model.interpret.attention_rollout
==========================================
Multi-layer attention flow analysis — Problem 7c.

Implements the Attention Rollout algorithm (Abnar & Zuidema 2020) for
multi-layer transformer/attention-based GNNs. Computes how attention
flows from input atoms to the [CLS] or readout token through all layers,
accounting for residual connections.

Particularly useful for:
  • Graphormer (stores last_attn_weights)
  • Option E (pGET) with attention-based GECT layers
  • Any model that stores per-layer attention matrices

Usage
-----
    >>> from Graph_model.interpret.attention_rollout import attention_rollout
    >>> flow = attention_rollout(attn_matrices)  # list of [N, N] tensors
    >>> atom_importance = flow[0, 1:]  # CLS → all atoms
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def attention_rollout(
    attention_matrices: Sequence[Tensor],
    add_residual:       bool = True,
    head_reduction:     str  = "mean",
) -> Tensor:
    """
    Compute attention rollout across multiple layers.

    Parameters
    ----------
    attention_matrices : list of [H, N, N] or [N, N] attention matrices
        One per layer, in order from input to output.
        H = number of heads, N = number of tokens/nodes.
    add_residual : bool
        If True, add identity matrix to each attention map
        (accounts for residual connections in transformer layers).
    head_reduction : str
        How to reduce multiple heads: 'mean', 'max', 'min'.

    Returns
    -------
    rollout : [N, N] tensor
        Entry [i, j] = total attention flow from token i to token j
        through all layers.
    """
    if not attention_matrices:
        raise ValueError("Must provide at least one attention matrix")

    rollout = None

    for layer_idx, attn in enumerate(attention_matrices):
        attn = attn.detach().float()

        # Reduce heads
        if attn.dim() == 3:  # [H, N, N]
            if head_reduction == "mean":
                attn = attn.mean(dim=0)
            elif head_reduction == "max":
                attn = attn.max(dim=0).values
            elif head_reduction == "min":
                attn = attn.min(dim=0).values
            else:
                raise ValueError(f"Unknown head_reduction: {head_reduction}")

        assert attn.dim() == 2, f"Expected 2D attention, got {attn.dim()}D"
        N = attn.shape[0]

        # Add residual connection (identity)
        if add_residual:
            attn = 0.5 * attn + 0.5 * torch.eye(N, device=attn.device)

        # Re-normalize rows to sum to 1
        row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / row_sums

        # Accumulate: rollout = attn @ previous_rollout
        if rollout is None:
            rollout = attn
        else:
            # If dimensions changed between layers, pad
            if attn.shape[0] != rollout.shape[0]:
                new_n = max(attn.shape[0], rollout.shape[0])
                if attn.shape[0] < new_n:
                    pad = torch.zeros(new_n, new_n, device=attn.device)
                    pad[:attn.shape[0], :attn.shape[1]] = attn
                    attn = pad
                if rollout.shape[0] < new_n:
                    pad = torch.zeros(new_n, new_n, device=rollout.device)
                    pad[:rollout.shape[0], :rollout.shape[1]] = rollout
                    rollout = pad
            rollout = attn @ rollout

    return rollout


def extract_attention_from_model(
    model:  torch.nn.Module,
    data,
    n_layers: Optional[int] = None,
) -> List[Tensor]:
    """
    Extract per-layer attention matrices from a model during forward pass.

    Supports:
    - Models with stored `last_attn_weights` (Graphormer)
    - Models with GATv2Conv layers (Option A/B/D/E)
    - Custom attention storage

    Parameters
    ----------
    model    : trained model
    data     : HeteroData input
    n_layers : expected number of layers (for validation)

    Returns
    -------
    list of [N, N] or [H, N, N] attention tensors per layer
    """
    from torch_geometric.data import Batch

    model.eval()
    device = next(model.parameters()).device

    # Ensure batched
    if hasattr(data, '__getitem__') and hasattr(data['ligand'], 'x'):
        if not hasattr(data['ligand'], 'batch') or data['ligand'].batch is None:
            data = Batch.from_data_list([data])
    data = data.to(device)

    attention_matrices: List[Tensor] = []
    hooks = []

    # Strategy 1: hook GATv2Conv layers
    from torch_geometric.nn import GATv2Conv
    for name, module in model.named_modules():
        if isinstance(module, GATv2Conv):
            original_return = module.return_attention_weights

            def make_hook(mod):
                def hook_fn(m, inp, out):
                    if isinstance(out, tuple) and len(out) >= 2:
                        # (edge_index, attn_weights)
                        attn_w = out[1]  # [E, H]
                        attention_matrices.append(attn_w.detach())
                return hook_fn

            h = module.register_forward_hook(make_hook(module))
            hooks.append(h)

    # Forward pass
    with torch.no_grad():
        _ = model(data)

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # Strategy 2: check for stored attention (Graphormer)
    if not attention_matrices:
        if hasattr(model, "last_attn_weights"):
            stored = model.last_attn_weights
            if isinstance(stored, list):
                attention_matrices = [a.detach() for a in stored]
            elif isinstance(stored, Tensor):
                attention_matrices = [stored.detach()]

    # Strategy 3: check layers for stored attention
    if not attention_matrices:
        for name, module in model.named_modules():
            if hasattr(module, "last_attn_weights") and module.last_attn_weights is not None:
                w = module.last_attn_weights
                if isinstance(w, Tensor):
                    attention_matrices.append(w.detach())

    return attention_matrices


def atom_importance_from_rollout(
    rollout:   Tensor,
    cls_index: int = 0,
) -> Tensor:
    """
    Extract per-atom importance from rollout matrix.

    Parameters
    ----------
    rollout   : [N, N] attention rollout matrix
    cls_index : index of the [CLS]/readout token (default 0)

    Returns
    -------
    importance : [N-1] tensor (excluding CLS) or [N] if cls_index is invalid
    """
    if cls_index >= rollout.shape[0]:
        # No CLS: return diagonal (self-attention)
        return rollout.diag()

    importance = rollout[cls_index]  # [N]

    # Remove CLS token from result
    mask = torch.ones(importance.shape[0], dtype=torch.bool)
    mask[cls_index] = False
    return importance[mask]


def compare_attribution_methods(
    model:            torch.nn.Module,
    data,
    plip_contacts:    Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """
    Run multiple attribution methods and compare rankings.

    Compares:
    - Attention rollout (this module)
    - Grad-CAM (gradcam module)
    - Integrated Gradients (integrated_gradients module)

    Parameters
    ----------
    model          : trained model
    data           : single HeteroData
    plip_contacts  : optional PLIP atom indices for validation

    Returns
    -------
    dict with per-method atom rankings and their correlations
    """
    results: Dict[str, Any] = {}

    # 1. Attention rollout
    try:
        attn_mats = extract_attention_from_model(model, data)
        if attn_mats:
            rollout = attention_rollout(attn_mats)
            importance = atom_importance_from_rollout(rollout)
            results["attention_rollout"] = {
                "importance": importance.cpu(),
                "n_layers": len(attn_mats),
            }
    except Exception as e:
        logger.warning("Attention rollout failed: %s", e)

    # 2. Grad-CAM
    try:
        from .gradcam import graph_gradcam
        cam_result = graph_gradcam(model, data)
        results["gradcam"] = {
            "importance": cam_result["atom_cam"],
            "layer_name": cam_result["layer_name"],
        }
    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e)

    # 3. Integrated Gradients
    try:
        from .integrated_gradients import integrated_gradients
        ig_result = integrated_gradients(model, data)
        results["integrated_gradients"] = {
            "importance": ig_result["atom_importance"],
        }
    except Exception as e:
        logger.warning("IG failed: %s", e)

    # 4. Rank correlation between methods
    method_names = list(results.keys())
    if len(method_names) >= 2:
        from scipy.stats import spearmanr
        correlations = {}
        for i, m1 in enumerate(method_names):
            for m2 in method_names[i + 1:]:
                imp1 = results[m1]["importance"].numpy()
                imp2 = results[m2]["importance"].numpy()
                # Truncate to same length if needed
                min_len = min(len(imp1), len(imp2))
                if min_len >= 3:
                    r, p = spearmanr(imp1[:min_len], imp2[:min_len])
                    correlations[f"{m1}_vs_{m2}"] = {
                        "spearman_r": float(r),
                        "p_value": float(p),
                    }
        results["rank_correlations"] = correlations

    # 5. PLIP comparison
    if plip_contacts:
        from .gradcam import gradcam_vs_plip
        for method_name in method_names:
            if "importance" in results[method_name]:
                imp = results[method_name]["importance"]
                plip_metrics = gradcam_vs_plip(imp, plip_contacts)
                results[method_name]["plip_validation"] = plip_metrics

    return results
