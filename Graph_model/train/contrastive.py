"""
Graph_model.train.contrastive
================================
Contrastive pre-training for binding energy models.

Problem 5b — With limited labelled data, self-supervised contrastive
pre-training learns a molecular representation by maximising agreement
between different "views" of the same molecule and minimising agreement
between different molecules.

Strategy for docking
--------------------
• **Positive pairs:** Same molecule, different conditions (pH/box/temp)
    → The model learns condition-invariant molecular identity
• **Negative pairs:** Different molecules, same condition
    → Teaches the model to distinguish molecules within a condition

Loss: NT-Xent (Normalised Temperature-scaled Cross-Entropy)
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))

Public API
----------
  ContrastivePretrainer(model, proj_dim, temperature)
  contrastive_pretrain(model, dataset, ...)

References
----------
• Chen T. et al., "A Simple Framework for Contrastive Learning of
  Visual Representations (SimCLR)," ICML 2020.
• You Y. et al., "Graph Contrastive Learning with Augmentations,"
  NeurIPS 2020.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """Contrastive pre-training configuration."""
    proj_dim:        int   = 64     # projection head output dimension
    temperature:     float = 0.1    # NT-Xent temperature τ
    lr:              float = 1e-3
    n_epochs:        int   = 30
    batch_size:      int   = 32
    augment_noise:   float = 0.05   # Gaussian noise σ for feature augmentation
    hidden_dim:      int   = 128    # projection head hidden dim
    device:          str   = "cpu"


class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross-Entropy Loss.

    Given N positive pairs in a batch of 2N samples, computes contrastive
    loss where the positive pair is pulled together and all other pairs
    are pushed apart.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        z_i, z_j : [N, D] normalised embeddings of positive pairs.
        Returns scalar loss.
        """
        N = z_i.size(0)
        if N == 0:
            return torch.tensor(0.0, device=z_i.device, requires_grad=True)

        # Concatenate all embeddings: [2N, D]
        z = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Cosine similarity matrix: [2N, 2N]
        sim = F.cosine_similarity(z.unsqueeze(0), z.unsqueeze(1), dim=-1)
        sim = sim / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Labels: positive pair for i is at i+N, for i+N is at i
        labels = torch.cat([
            torch.arange(N, 2 * N, device=sim.device),
            torch.arange(0, N, device=sim.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss


class _ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ContrastivePretrainer:
    """
    Contrastive pre-training wrapper for any docking GNN.

    Extracts graph-level representations, projects to embedding space,
    and trains with NT-Xent loss on positive/negative pairs.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ContrastiveConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg   = cfg or ContrastiveConfig()

        # Extract hidden dim from model
        hidden = self._get_hidden_dim()

        # Projection head
        self.proj = _ProjectionHead(hidden, self.cfg.hidden_dim, self.cfg.proj_dim)
        self.loss_fn = NTXentLoss(self.cfg.temperature)

    def _get_hidden_dim(self) -> int:
        """Infer the model's graph-level representation dimension."""
        cfg = getattr(self.model, 'cfg', None)
        if cfg:
            return getattr(cfg, 'hidden_dim', 128)
        return 128

    def _extract_embedding(self, model: nn.Module, batch) -> Tensor:
        """
        Extract graph-level embedding before the final MLP.

        This hooks into the model's pooling output.
        """
        embeddings = []

        def hook_fn(module, input, output):
            if isinstance(output, Tensor) and output.ndim == 2:
                embeddings.append(output)

        # Register hook on the MLP's first layer input
        handles = []
        mlp = getattr(model, 'mlp', None)
        if mlp is not None and hasattr(mlp, '__getitem__'):
            h = mlp[0].register_forward_hook(
                lambda mod, inp, out: embeddings.append(inp[0])
            )
            handles.append(h)

        # Forward
        _ = model(batch)

        # Remove hooks
        for h in handles:
            h.remove()

        if embeddings:
            # Take the last captured embedding
            emb = embeddings[-1]
            # Slice to hidden_dim if it's wider (due to condition concat)
            hidden = self._get_hidden_dim()
            return emb[:, :hidden]

        # Fallback: use model output
        out = model(batch)
        if isinstance(out, dict):
            out = list(out.values())[0]
        elif isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def _augment_batch(self, batch, noise_std: float):
        """Add Gaussian noise to node features for data augmentation."""
        batch = batch.clone()
        if hasattr(batch, 'ligand') and hasattr(batch['ligand'], 'x'):
            batch['ligand'].x = batch['ligand'].x + torch.randn_like(
                batch['ligand'].x
            ) * noise_std
        elif hasattr(batch, 'x'):
            batch.x = batch.x + torch.randn_like(batch.x) * noise_std
        return batch

    def _group_by_ligand(self, dataset) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(dataset)):
            d = dataset[i]
            name = d.ligand_name if isinstance(d.ligand_name, str) else str(d.ligand_name)
            groups[name].append(i)
        return dict(groups)

    def train(
        self,
        dataset,
        seed: int = 42,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        """
        Run contrastive pre-training.

        Pairs are formed by: same molecule, different condition records.

        Returns history dict.
        """
        device = torch.device(self.cfg.device)
        self.model.to(device)
        self.proj.to(device)

        self.model.train()
        self.proj.train()

        # Joint optimizer
        params = list(self.model.parameters()) + list(self.proj.parameters())
        optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

        groups = self._group_by_ligand(dataset)
        rng    = random.Random(seed)

        # Build positive pairs (same mol, different records)
        pair_indices = []
        for lig, idxs in groups.items():
            if len(idxs) >= 2:
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        pair_indices.append((idxs[i], idxs[j]))

        if not pair_indices:
            logger.warning("Not enough pairs for contrastive pre-training")
            return {'contrastive_loss': []}

        history = {'contrastive_loss': []}

        for epoch in range(self.cfg.n_epochs):
            rng.shuffle(pair_indices)
            epoch_losses = []

            # Process pairs in batches
            for start in range(0, len(pair_indices), self.cfg.batch_size):
                batch_pairs = pair_indices[start:start + self.cfg.batch_size]
                if len(batch_pairs) < 2:
                    continue

                idx_i = [p[0] for p in batch_pairs]
                idx_j = [p[1] for p in batch_pairs]

                # Load batches
                loader_i = PyGDataLoader(Subset(dataset, idx_i), batch_size=len(idx_i))
                loader_j = PyGDataLoader(Subset(dataset, idx_j), batch_size=len(idx_j))

                for bi, bj in zip(loader_i, loader_j):
                    bi = bi.to(device)
                    bj = bj.to(device)

                    # Augment view 2
                    bj_aug = self._augment_batch(bj, self.cfg.augment_noise)

                    # Extract embeddings
                    optimizer.zero_grad()

                    z_i = self.proj(self._extract_embedding(self.model, bi))
                    z_j = self.proj(self._extract_embedding(self.model, bj_aug))

                    # Ensure same batch size
                    min_n = min(z_i.size(0), z_j.size(0))
                    z_i = z_i[:min_n]
                    z_j = z_j[:min_n]

                    loss = self.loss_fn(z_i, z_j)
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            history['contrastive_loss'].append(avg_loss)

            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Contrastive epoch {epoch+1}/{self.cfg.n_epochs}: "
                    f"loss={avg_loss:.4f}"
                )

        return history


def contrastive_pretrain(
    model: nn.Module,
    dataset,
    cfg: ContrastiveConfig | None = None,
    seed: int = 42,
) -> tuple[nn.Module, dict]:
    """
    Convenience function for contrastive pre-training.

    Returns (pretrained_model, history).
    """
    cfg = cfg or ContrastiveConfig()
    trainer = ContrastivePretrainer(model, cfg)
    history = trainer.train(dataset, seed=seed)
    return trainer.model, history
