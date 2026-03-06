"""
Graph_model.interpret.probing
================================
Probing classifiers for latent representation analysis — Problem 7b.

Trains lightweight linear classifiers on frozen GNN embeddings to test what
information the learned representations encode:
  • Ligand group (primary / intermediate / GA_analogue)
  • Galloyl unit count (0 / 1 / 2 / 5)  — ordinal regression probe
  • Selectivity index sign (selective / non-selective) — binary probe
  • Box type (8-class)

If probes achieve high accuracy, the GNN has captured domain-relevant
chemical semantics beyond simple docking score regression.

Usage
-----
    >>> from Graph_model.interpret.probing import ProbingClassifier, extract_embeddings
    >>> embeddings, labels = extract_embeddings(model, dataset, label_fn)
    >>> probe = ProbingClassifier(embed_dim=128, n_classes=3)
    >>> results = probe.train_and_evaluate(embeddings, labels)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ProbingClassifier(nn.Module):
    """
    Linear probing classifier for GNN embedding analysis.

    Parameters
    ----------
    embed_dim  : int  — dimension of the GNN embeddings
    n_classes  : int  — number of output classes
    probe_type : str  — 'linear' or 'mlp' (1-hidden-layer)
    hidden_dim : int  — hidden dim for MLP probe (only if probe_type='mlp')
    """

    def __init__(
        self,
        embed_dim:  int,
        n_classes:  int,
        probe_type: str = "linear",
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.n_classes  = n_classes
        self.probe_type = probe_type

        if probe_type == "linear":
            self.probe = nn.Linear(embed_dim, n_classes)
        elif probe_type == "mlp":
            self.probe = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, n_classes),
            )
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}")

    def forward(self, x: Tensor) -> Tensor:
        return self.probe(x)

    def train_and_evaluate(
        self,
        embeddings: Tensor,
        labels:     Tensor,
        train_ratio: float = 0.8,
        n_epochs:    int   = 100,
        lr:          float = 1e-3,
        seed:        int   = 42,
    ) -> Dict[str, Any]:
        """
        Train probe on embeddings and return metrics.

        Parameters
        ----------
        embeddings : [N, D] float tensor
        labels     : [N] long tensor (class indices)
        train_ratio: train/test split ratio
        n_epochs   : training epochs
        lr         : learning rate

        Returns
        -------
        dict with 'train_acc', 'test_acc', 'per_class_acc', 'loss_history'
        """
        device = embeddings.device
        self.to(device)

        N = len(embeddings)
        torch.manual_seed(seed)
        perm = torch.randperm(N, device=device)
        n_train = int(N * train_ratio)

        train_idx = perm[:n_train]
        test_idx  = perm[n_train:]

        X_train, y_train = embeddings[train_idx], labels[train_idx]
        X_test,  y_test  = embeddings[test_idx],  labels[test_idx]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_history: List[float] = []

        self.train()
        for epoch in range(n_epochs):
            logits = self(X_train)
            loss = criterion(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

        # Evaluate
        self.eval()
        with torch.no_grad():
            train_logits = self(X_train)
            train_preds  = train_logits.argmax(dim=-1)
            train_acc    = (train_preds == y_train).float().mean().item()

            test_logits = self(X_test)
            test_preds  = test_logits.argmax(dim=-1)
            test_acc    = (test_preds == y_test).float().mean().item()

            # Per-class accuracy
            per_class_acc = {}
            for c in range(self.n_classes):
                mask = y_test == c
                if mask.sum() > 0:
                    per_class_acc[c] = (test_preds[mask] == c).float().mean().item()
                else:
                    per_class_acc[c] = float("nan")

        return {
            "train_acc":      train_acc,
            "test_acc":       test_acc,
            "per_class_acc":  per_class_acc,
            "loss_history":   loss_history,
            "n_train":        n_train,
            "n_test":         N - n_train,
        }


def extract_embeddings(
    model:    nn.Module,
    dataset:  Sequence,
    hook_layer_name: Optional[str] = None,
) -> Tensor:
    """
    Extract graph-level embeddings from a trained model.

    Uses a forward hook on the layer just before the final MLP head.
    Falls back to hooking on `post_gnn`, `pool`, or `mlp.0`.

    Parameters
    ----------
    model           : trained GNN model
    dataset         : sequence of HeteroData graphs
    hook_layer_name : specific layer to hook (optional)

    Returns
    -------
    embeddings : [N_graphs, D] tensor
    """
    from torch_geometric.data import Batch

    model.eval()
    device = next(model.parameters()).device

    # Find layer to hook
    if hook_layer_name:
        target = dict(model.named_modules())[hook_layer_name]
    else:
        target = _find_embedding_layer(model)

    embeddings: List[Tensor] = []
    captured: List[Tensor] = []

    def hook_fn(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        captured.append(out.detach())

    handle = target.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            for data in dataset:
                captured.clear()
                if not hasattr(data['ligand'], 'batch') or data['ligand'].batch is None:
                    data = Batch.from_data_list([data])
                data = data.to(device)
                _ = model(data)
                if captured:
                    emb = captured[-1]
                    # If node-level, pool to graph level
                    if emb.dim() == 2 and emb.shape[0] > 1:
                        if hasattr(data['ligand'], 'batch'):
                            from torch_geometric.nn import global_mean_pool
                            emb = global_mean_pool(emb, data['ligand'].batch)
                    embeddings.append(emb.cpu())
    finally:
        handle.remove()

    if not embeddings:
        raise RuntimeError("No embeddings captured. Check model and hook layer.")

    return torch.cat(embeddings, dim=0)


def _find_embedding_layer(model: nn.Module) -> nn.Module:
    """Find the layer producing graph-level embeddings (before final head)."""
    # Try common names
    for name in ["post_gnn", "graph_pool", "pool", "readout"]:
        if hasattr(model, name):
            return getattr(model, name)

    # Try first sub-module of MLP
    if hasattr(model, "mlp"):
        mlp = model.mlp
        if isinstance(mlp, nn.Sequential) and len(mlp) > 0:
            return mlp[0]

    # Fallback: second-to-last module with parameters
    modules_with_params = [(n, m) for n, m in model.named_modules()
                          if hasattr(m, 'weight') and m.weight is not None]
    if len(modules_with_params) >= 2:
        return modules_with_params[-2][1]

    raise ValueError("Could not auto-detect embedding layer. Provide hook_layer_name.")


# ── Pre-defined label functions ──────────────────────────────────────────────

def label_ligand_group(data) -> int:
    """Map ligand to group: primary=0, intermediate=1, GA_analogue=2."""
    try:
        from Graph_model.data.config import LIGAND_GROUPS
        name = data.ligand_name if isinstance(data.ligand_name, str) else str(data.ligand_name)
        group = LIGAND_GROUPS.get(name, "unknown")
        mapping = {"primary": 0, "intermediate": 1, "GA_analogue": 2}
        return mapping.get(group, 0)
    except ImportError:
        return 0


def label_galloyl_units(data) -> int:
    """Map ligand to galloyl unit class: 0→0, 1→1, 2→2, 5→3."""
    try:
        from Graph_model.data.config import GALLOYL_UNIT_COUNTS
        name = data.ligand_name if isinstance(data.ligand_name, str) else str(data.ligand_name)
        g = GALLOYL_UNIT_COUNTS.get(name, 0)
        # Map to class index
        mapping = {0: 0, 1: 1, 2: 2, 5: 3}
        return mapping.get(g, 0)
    except ImportError:
        return 0


def label_box_type(data) -> int:
    """Map box_idx to class label (0–7)."""
    box_idx = getattr(data, "box_idx", 0)
    return int(box_idx.item()) if hasattr(box_idx, "item") else int(box_idx)


def run_all_probes(
    model:   nn.Module,
    dataset: Sequence,
    probe_type: str = "linear",
) -> Dict[str, Dict[str, Any]]:
    """
    Run all standard probes on a trained model.

    Returns
    -------
    dict: {probe_name → probe_results_dict}
    """
    logger.info("Extracting embeddings for probing analysis...")
    embeddings = extract_embeddings(model, dataset)
    device = embeddings.device
    N = len(embeddings)
    embed_dim = embeddings.shape[-1]

    results = {}

    # 1. Ligand group probe (3 classes)
    logger.info("Running ligand group probe (3 classes)...")
    group_labels = torch.tensor(
        [label_ligand_group(dataset[i]) for i in range(N)],
        dtype=torch.long, device=device
    )
    probe_group = ProbingClassifier(embed_dim, n_classes=3, probe_type=probe_type)
    results["ligand_group"] = probe_group.train_and_evaluate(embeddings, group_labels)

    # 2. Galloyl unit probe (4 classes)
    logger.info("Running galloyl unit probe (4 classes)...")
    gal_labels = torch.tensor(
        [label_galloyl_units(dataset[i]) for i in range(N)],
        dtype=torch.long, device=device
    )
    probe_gal = ProbingClassifier(embed_dim, n_classes=4, probe_type=probe_type)
    results["galloyl_units"] = probe_gal.train_and_evaluate(embeddings, gal_labels)

    # 3. Box type probe (8 classes)
    logger.info("Running box type probe (8 classes)...")
    box_labels = torch.tensor(
        [label_box_type(dataset[i]) for i in range(N)],
        dtype=torch.long, device=device
    )
    probe_box = ProbingClassifier(embed_dim, n_classes=8, probe_type=probe_type)
    results["box_type"] = probe_box.train_and_evaluate(embeddings, box_labels)

    logger.info("Probing complete. Ligand_group=%.3f, Galloyl=%.3f, Box=%.3f",
                results["ligand_group"]["test_acc"],
                results["galloyl_units"]["test_acc"],
                results["box_type"]["test_acc"])

    return results
