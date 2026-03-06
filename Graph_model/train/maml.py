"""
Graph_model.train.maml
========================
Model-Agnostic Meta-Learning (MAML) for few-shot molecule generalisation.

Problem 5a — With only 9 ligands, leaving one out means the test molecule
is entirely unseen. MAML learns an initialisation that can rapidly adapt
to a new molecule with just a few gradient steps.

Algorithm (First-Order MAML / Reptile variant)
----------------------------------------------
1. Sample a task = (support set of 1 ligand, query set of another)
2. Clone model parameters θ
3. Inner loop: K gradient steps on support set → θ'
4. Outer loop: compute loss on query set with θ', update θ

For our 9-ligand setup:
  - Each task: support = records for 1 ligand, query = records for another
  - This forces the model to learn features that transfer across molecules

Public API
----------
  MAMLTrainer(model, inner_lr, outer_lr, n_inner_steps, first_order)
  maml_train(model, dataset, ...)

References
----------
• Finn C. et al., "Model-Agnostic Meta-Learning for Fast Adaptation
  of Deep Networks," ICML 2017.
• Nichol A. et al., "On First-Order Meta-Learning Algorithms," arXiv 2018.
"""

from __future__ import annotations

import copy
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """MAML training configuration."""
    inner_lr:         float = 0.01    # learning rate for inner (task-specific) loop
    outer_lr:         float = 1e-3    # learning rate for outer (meta) loop
    n_inner_steps:    int   = 5       # gradient steps per task
    n_tasks_per_epoch: int  = 9       # tasks sampled per meta-epoch
    n_meta_epochs:    int   = 50      # total meta-training epochs
    support_size:     int   = 16      # samples per support set
    query_size:       int   = 16      # samples per query set
    first_order:      bool  = True    # first-order approximation (faster)
    batch_size:       int   = 16
    device:           str   = "cpu"


class MAMLTrainer:
    """
    MAML meta-learning trainer for docking GNNs.

    Learns a parameter initialisation from which the model can quickly
    adapt to a new (unseen) ligand.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: MAMLConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg   = cfg or MAMLConfig()
        self.meta_optim = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.outer_lr,
        )

    def _group_by_ligand(self, dataset) -> dict[str, list[int]]:
        """Group dataset indices by ligand name."""
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            d = dataset[i]
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            groups[name].append(i)
        return dict(groups)

    def _sample_task(
        self,
        groups: dict[str, list[int]],
        dataset,
        rng: random.Random,
    ) -> tuple[PyGDataLoader, PyGDataLoader]:
        """
        Sample a MAML task: support from one ligand, query from another.

        Returns (support_loader, query_loader).
        """
        ligands = list(groups.keys())
        support_lig, query_lig = rng.sample(ligands, 2)

        s_idx = rng.sample(groups[support_lig], min(self.cfg.support_size, len(groups[support_lig])))
        q_idx = rng.sample(groups[query_lig],   min(self.cfg.query_size,   len(groups[query_lig])))

        support_ds = Subset(dataset, s_idx)
        query_ds   = Subset(dataset, q_idx)

        s_loader = PyGDataLoader(support_ds, batch_size=self.cfg.batch_size, shuffle=True)
        q_loader = PyGDataLoader(query_ds,   batch_size=self.cfg.batch_size, shuffle=False)

        return s_loader, q_loader

    def _inner_loop(
        self,
        support_loader: PyGDataLoader,
        device: torch.device,
    ) -> nn.Module:
        """
        Perform K inner gradient steps on support set.
        Returns adapted model (clone).
        """
        # Clone model
        adapted = copy.deepcopy(self.model)
        adapted.train()
        inner_optim = torch.optim.SGD(adapted.parameters(), lr=self.cfg.inner_lr)

        for _step in range(self.cfg.n_inner_steps):
            for batch in support_loader:
                batch = batch.to(device)
                inner_optim.zero_grad()

                pred = self._forward(adapted, batch)
                target = batch.y.view(-1, 1).to(device)
                loss = nn.functional.mse_loss(pred, target)

                loss.backward()
                inner_optim.step()

        return adapted

    def _compute_query_loss(
        self,
        adapted_model: nn.Module,
        query_loader: PyGDataLoader,
        device: torch.device,
    ) -> Tensor:
        """Compute loss on query set using adapted parameters."""
        total_loss = torch.tensor(0.0, device=device)
        n = 0

        for batch in query_loader:
            batch = batch.to(device)
            pred = self._forward(adapted_model, batch)
            target = batch.y.view(-1, 1).to(device)
            loss = nn.functional.mse_loss(pred, target, reduction='sum')
            total_loss = total_loss + loss
            n += target.size(0)

        return total_loss / max(n, 1)

    def _forward(self, model: nn.Module, batch) -> Tensor:
        """Model-agnostic forward pass."""
        out = model(batch)
        if isinstance(out, dict):
            return out.get('collagen', out.get('mu', list(out.values())[0]))
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def train(
        self,
        dataset,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Run MAML meta-training.

        Returns history dict with 'meta_loss' per epoch.
        """
        device = torch.device(self.cfg.device)
        self.model.to(device)

        groups = self._group_by_ligand(dataset)
        rng    = random.Random(seed)

        if len(groups) < 2:
            logger.warning("MAML requires at least 2 ligand groups; skipping")
            return {'meta_loss': []}

        history = {'meta_loss': []}

        for epoch in range(self.cfg.n_meta_epochs):
            epoch_losses = []

            for _task in range(self.cfg.n_tasks_per_epoch):
                # Sample task
                s_loader, q_loader = self._sample_task(groups, dataset, rng)

                # Inner loop: adapt
                adapted = self._inner_loop(s_loader, device)

                # Outer loop: compute query loss
                query_loss = self._compute_query_loss(adapted, q_loader, device)
                epoch_losses.append(query_loss.item())

                if self.cfg.first_order:
                    # First-order MAML (Reptile-style)
                    # Move original parameters toward adapted parameters
                    self.meta_optim.zero_grad()
                    with torch.no_grad():
                        for p_orig, p_adapted in zip(
                            self.model.parameters(),
                            adapted.parameters(),
                        ):
                            if p_orig.requires_grad:
                                p_orig.grad = p_orig.data - p_adapted.data
                    self.meta_optim.step()
                else:
                    # Full MAML (second-order)
                    self.meta_optim.zero_grad()
                    query_loss.backward()
                    self.meta_optim.step()

                # Clean up adapted model
                del adapted

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            history['meta_loss'].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(f"MAML epoch {epoch+1}/{self.cfg.n_meta_epochs}: "
                           f"meta_loss={avg_loss:.4f}")

        return history


def maml_train(
    model: nn.Module,
    dataset,
    cfg: MAMLConfig | None = None,
    seed: int = 42,
) -> tuple[nn.Module, dict]:
    """
    Convenience function for MAML training.

    Returns (trained_model, history).
    """
    cfg = cfg or MAMLConfig()
    trainer = MAMLTrainer(model, cfg)
    history = trainer.train(dataset, seed=seed)
    return trainer.model, history
