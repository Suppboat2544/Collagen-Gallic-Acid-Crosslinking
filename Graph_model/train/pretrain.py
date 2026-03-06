"""
Graph_model.train.pretrain
============================
Stage 1 — Pre-training on PDBbind general set.

Objective
---------
Learn generic protein-ligand binding geometry from ~19k diverse complexes,
so the GNN backbone encodes hydrogen-bond patterns and van der Waals contacts
before seeing the narrow 9-ligand anchor set.

Architecture compatibility
---------------------------
Accepts any model that:
  • takes a HeteroData batch as input
  • returns either  Tensor [B, 1]  (Options A, C)
                or  (Tensor [B, 1], ...)  (Options B — ignoring attn_weights)

Stage scheduler
---------------
The learning-rate is decayed via a cosine-decay-with-warmup schedule.
Early stopping is based on validation MSE with patience.

Checkpoint format
-----------------
  {
    'epoch':        int,
    'state_dict':   model.state_dict(),
    'optimizer':    optimizer.state_dict(),
    'best_val_mse': float,
    'train_losses': List[float],
    'val_losses':   List[float],
  }

Usage
-----
    >>> from Graph_model.train.pretrain import pretrain
    >>> from Graph_model.model import OptionA, OptionAConfig
    >>> model = OptionA()
    >>> model, history = pretrain(
    ...     model       = model,
    ...     pdbbind_ds  = PDBbindGraphDataset(),
    ...     max_epochs  = 50,
    ...     save_path   = "checkpoints/pretrained_option_a.pt",
    ... )
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = logging.getLogger(__name__)

# ── Cosine warmup schedule ────────────────────────────────────────────────────

def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs:  int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return epoch / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Single forward pass (model-type agnostic) ─────────────────────────────────

def _forward(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """
    Run model forward and always return a Tensor [B, 1].
    Works for Options A, B, C, D.
    """
    batch = batch.to(device)
    out = model(batch)
    if isinstance(out, dict):
        # Option D — use collagen head for PDBbind pre-training
        return out['collagen']
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


# ── Main pretrain function ────────────────────────────────────────────────────

def pretrain(
    model:          nn.Module,
    pdbbind_ds,
    max_epochs:     int   = 50,
    batch_size:     int   = 32,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    warmup_epochs:  int   = 5,
    patience:       int   = 10,
    val_fraction:   float = 0.1,
    device:         Optional[torch.device] = None,
    save_path:      Optional[str | Path]   = None,
    verbose:        bool  = True,
) -> tuple[nn.Module, dict]:
    """
    Pre-train *model* on the PDBbind dataset (Stage 1).

    Parameters
    ----------
    model        : any OptionX model (A, B, C, D)
    pdbbind_ds   : PDBbindGraphDataset (or any indexable HeteroData container)
    max_epochs   : maximum training epochs
    batch_size   : mini-batch size
    lr           : initial learning rate
    weight_decay : L2 regularisation
    warmup_epochs: linear LR warmup
    patience     : early stopping patience (epochs without val improvement)
    val_fraction : fraction of PDBbind data to hold out for early stopping
    device       : torch.device (auto-detected if None)
    save_path    : path to save best checkpoint (.pt)
    verbose      : print epoch summaries

    Returns
    -------
    model   : model with best pre-trained weights loaded
    history : dict with 'train_losses', 'val_losses', 'best_epoch'
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    logger.info("Pretrain: device=%s, max_epochs=%d, lr=%.1e", device, max_epochs, lr)

    # Lazy-load the dataset
    if hasattr(pdbbind_ds, 'load'):
        pdbbind_ds.load()

    n_total = len(pdbbind_ds)
    if n_total == 0:
        raise RuntimeError("PDBbindGraphDataset returned 0 entries — check data path.")

    # Train / val split (random, since PDBbind affinity is used for pre-training only)
    n_val   = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        pdbbind_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = PyGDataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _cosine_schedule(optimizer, warmup_epochs, max_epochs)
    criterion = nn.MSELoss()

    best_val_mse  = float('inf')
    best_state    = None
    best_epoch    = 0
    patience_cnt  = 0
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(max_epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            optimizer.zero_grad()
            try:
                pred = _forward(model, batch, device)
                target = batch.y.to(device).view(-1, 1)
                loss = criterion(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1
            except Exception as exc:
                logger.warning("Batch error in pretrain epoch %d: %s", epoch, exc)
                continue

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        n_val_b   = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    pred   = _forward(model, batch, device)
                    target = batch.y.to(device).view(-1, 1)
                    val_loss += criterion(pred, target).item()
                    n_val_b  += 1
                except Exception:
                    continue
        avg_val = val_loss / max(n_val_b, 1)
        val_losses.append(avg_val)

        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == max_epochs - 1):
            logger.info(
                "Pretrain epoch %4d/%d  train_MSE=%.4f  val_MSE=%.4f  "
                "RMSE_val=%.4f kcal/mol  LR=%.2e",
                epoch + 1, max_epochs,
                avg_train, avg_val, avg_val ** 0.5,
                optimizer.param_groups[0]['lr'],
            )

        # ── Early stopping ─────────────────────────────────────────────────────
        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, patience)
                break

    # ── Load best weights ──────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
        logger.info(
            "Pretrain complete. Best epoch=%d  val_RMSE=%.4f kcal/mol",
            best_epoch + 1, best_val_mse ** 0.5
        )

    history = {
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'best_epoch':   best_epoch,
        'best_val_mse': best_val_mse,
        'best_val_rmse': best_val_mse ** 0.5,
    }

    # ── Save checkpoint ────────────────────────────────────────────────────────
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch':        best_epoch,
            'state_dict':   model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'best_val_mse': best_val_mse,
            'train_losses': train_losses,
            'val_losses':   val_losses,
        }
        torch.save(checkpoint, save_path)
        logger.info("Checkpoint saved → %s", save_path)

    return model, history
