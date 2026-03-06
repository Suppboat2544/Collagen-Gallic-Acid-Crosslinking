"""
Graph_model.train.finetune
============================
Stage 2 + 3 — Fine-tuning with LOLO-CV and curriculum learning.

Transfer learning protocol
---------------------------
  Stage 2  :  Freeze GNN backbone layers 1–2 (input projection + blocks 0,1);
              fine-tune layers 3–4 (blocks 2,3), condition encoder, and MLP head.
              LR = 1e-3 (higher — frozen backbone suppresses gradient noise).
              Curriculum: Weeks 1–3 active.

  Stage 3  :  Unfreeze ALL layers for full fine-tuning.
              LR = 1e-5  (very low — preserves pre-trained geometry).
              Optional — run only if Stage-2 RMSE < 0.5 kcal/mol.

LOLO-CV evaluation
-------------------
Each fold is an independent fine-tuning run.  The pre-trained weights are
reloaded at the start of each fold so no ligand's holdout contaminates another.

Results are reported as mean ± std across all 9 folds.

Layer freeze map
----------------
The freeze utility targets named-parameter groups:
  layers 1–2:  'input_proj', 'edge_proj', 'gat_layers.0', 'gat_layers.1'
               (Option A notation; auto-detected for Options B, C, D)

Usage
-----
    >>> results = finetune(
    ...     model           = pretrained_model,
    ...     anchor_dataset  = collagen_dataset,
    ...     non_anchor_ds   = pdbbind_dataset,   # added to training folds
    ...     stage           = 2,
    ...     run_lolo_cv     = True,
    ...     curriculum_schedule = [10, 20],
    ...     save_dir        = "checkpoints/",
    ... )
    >>> print_lolo_report(results['folds'], model_name="OptionA Stage2")
"""

from __future__ import annotations

import copy
import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset

from .lolo_cv    import LOLOCVSplitter, LOLOFold
from .curriculum import CurriculumSampler
from .metrics    import FoldMetrics, aggregate_folds, regression_metrics
from .pretrain   import _forward, _cosine_schedule

logger = logging.getLogger(__name__)


# ── Freeze utilities ──────────────────────────────────────────────────────────

# Parameter name prefixes that belong to "early backbone layers 1–2".
# These are frozen in Stage 2 and unfrozen in Stage 3.
_EARLY_LAYER_PREFIXES = [
    # Option A / B / D (GATv2-based)
    'input_proj',
    'edge_proj',
    'gat_layers.0',
    'gat_layers.1',
    # Option B (dual encoder)
    'gnn_l.proj', 'gnn_l.layers.0', 'gnn_l.layers.1',
    'gnn_p.proj', 'gnn_p.layers.0', 'gnn_p.layers.1',
    # Option C (GINEStack)
    'atom_gnn.proj', 'atom_gnn.layers.0', 'atom_gnn.layers.1',
    # Shared encoder in Option D
    'encoder.proj', 'encoder.layers.0', 'encoder.layers.1',
    'encoder.inp_proj', 'encoder.edge_proj',
    'encoder.blocks.0', 'encoder.blocks.1',
]


def freeze_backbone_stages(
    model: nn.Module,
    freeze_early: bool = True,
) -> tuple[int, int]:
    """
    Freeze / unfreeze backbone parameters.

    Parameters
    ----------
    model        : any OptionX model
    freeze_early : True  → freeze layers 1–2  (Stage 2)
                   False → unfreeze ALL layers (Stage 3)

    Returns
    -------
    (n_frozen, n_total) — count of frozen and total parameters
    """
    n_frozen = 0
    n_total  = 0

    for name, param in model.named_parameters():
        n_total += 1
        is_early = any(name.startswith(pfx) for pfx in _EARLY_LAYER_PREFIXES)
        if freeze_early and is_early:
            param.requires_grad_(False)
            n_frozen += 1
        else:
            param.requires_grad_(True)

    logger.info(
        "freeze_backbone_stages(freeze_early=%s): %d/%d params frozen",
        freeze_early, n_frozen, n_total
    )
    return n_frozen, n_total


# ── Per-fold training ─────────────────────────────────────────────────────────

def _train_one_fold(
    model:              nn.Module,
    fold:               LOLOFold,
    dataset,
    stage:              int,
    max_epochs:         int,
    batch_size:         int,
    lr:                 float,
    weight_decay:       float,
    warmup_epochs:      int,
    patience:           int,
    curriculum_schedule: list[int],
    device:             torch.device,
    verbose:            bool,
) -> tuple[nn.Module, FoldMetrics]:
    """Fine-tune model on one LOLO fold and return metrics on held-out ligand."""

    # Freeze stage
    if stage == 2:
        freeze_backbone_stages(model, freeze_early=True)
    elif stage == 3:
        freeze_backbone_stages(model, freeze_early=False)

    # Build subsets
    train_subset = Subset(dataset, fold.train_idx)
    val_subset   = Subset(dataset, fold.val_idx)
    test_subset  = Subset(dataset, fold.test_idx)

    # Curriculum sampler for training loader
    curriculum = CurriculumSampler(
        dataset          = train_subset,
        stage_schedule   = curriculum_schedule,
        non_anchor_weight= 0.3,
        seed             = fold.fold * 100,
    )
    curriculum.set_epoch(0)
    train_sampler = curriculum.get_sampler(num_samples=len(train_subset))

    train_loader = PyGDataLoader(
        train_subset,
        batch_size  = batch_size,
        sampler     = train_sampler,
        num_workers = 0,
        drop_last   = False,
    )
    val_loader = PyGDataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer (only over trainable parameters)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer  = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    scheduler  = _cosine_schedule(optimizer, warmup_epochs, max_epochs)
    criterion  = nn.MSELoss()

    best_val_mse = float('inf')
    best_state   = None
    best_epoch   = 0
    patience_cnt = 0

    for epoch in range(max_epochs):
        curriculum.set_epoch(epoch)
        # Rebuild sampler with updated weights each epoch
        train_sampler = curriculum.get_sampler(num_samples=len(train_subset))
        train_loader  = PyGDataLoader(
            train_subset,
            batch_size  = batch_size,
            sampler     = train_sampler,
            num_workers = 0,
        )

        # Train epoch
        model.train()
        ep_loss  = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            try:
                pred   = _forward(model, batch, device)
                target = batch.y.to(device).view(-1, 1)
                loss   = criterion(pred, target)
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                ep_loss   += loss.item()
                n_batches += 1
            except Exception as exc:
                logger.debug("Batch error fold %d epoch %d: %s", fold.fold, epoch, exc)
        avg_train = ep_loss / max(n_batches, 1)

        # Val epoch
        model.eval()
        vl = 0.0; nvb = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    pred   = _forward(model, batch, device)
                    target = batch.y.to(device).view(-1, 1)
                    vl    += criterion(pred, target).item()
                    nvb   += 1
                except Exception:
                    continue
        avg_val = vl / max(nvb, 1)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            logger.info(
                "  Fold %d | epoch %3d | stage %d | train_MSE=%.4f val_MSE=%.4f "
                "RMSE_val=%.4f",
                fold.fold, epoch + 1, stage,
                avg_train, avg_val, avg_val ** 0.5,
            )

        if avg_val < best_val_mse:
            best_val_mse = avg_val
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch   = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("  Fold %d: early stop epoch %d", fold.fold, epoch + 1)
                break

    # Load best val weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Evaluate on held-out test set
    model.eval()
    all_preds:   list[float] = []
    all_targets: list[float] = []

    test_loader = PyGDataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in test_loader:
            try:
                pred   = _forward(model, batch, device)
                target = batch.y.view(-1)
                all_preds.extend(pred.cpu().view(-1).tolist())
                all_targets.extend(target.cpu().tolist())
            except Exception as exc:
                logger.debug("Test batch error: %s", exc)

    m = regression_metrics(all_preds, all_targets)
    fold_m = FoldMetrics(
        fold        = fold.fold,
        held_out    = fold.held_out_ligand,
        n_test      = fold.n_test,
        rmse        = m['rmse'],
        mae         = m['mae'],
        pearson_r   = m['pearson_r'],
        spearman_r  = m['spearman_r'],
        n_train     = fold.n_train,
        n_val       = fold.n_val,
        best_epoch  = best_epoch,
        val_rmse    = best_val_mse ** 0.5,
    )
    logger.info("  → %s", fold_m)
    return model, fold_m


# ── Main fine-tune function ───────────────────────────────────────────────────

def finetune(
    model:              nn.Module,
    anchor_dataset,
    non_anchor_ds       = None,
    stage:              int   = 2,
    run_lolo_cv:        bool  = True,
    max_epochs:         int   = 50,
    batch_size:         int   = 32,
    lr:                 float = 1e-3,
    weight_decay:       float = 1e-4,
    warmup_epochs:      int   = 5,
    patience:           int   = 10,
    curriculum_schedule: list[int] = None,
    val_ratio:          float = 0.15,
    device:             Optional[torch.device] = None,
    seed:               int   = 42,
    save_dir:           Optional[str | Path]   = None,
    verbose:            bool  = True,
) -> dict:
    """
    Fine-tune *model* on the anchor dataset using LOLO-CV.

    Parameters
    ----------
    model            : pre-trained model (Stage-1 weights)
    anchor_dataset   : CollagenDockingDataset (tier=0 records)
    non_anchor_ds    : optional PDBbindGraphDataset to add to training folds
    stage            : 2 → freeze early layers; 3 → unfreeze all
    run_lolo_cv      : if False, train on all anchor data (no CV) — for final model
    max_epochs       : max epochs per fold
    batch_size       : mini-batch size
    lr               : learning rate
    weight_decay     : L2 regulariser
    warmup_epochs    : cosine warmup length
    patience         : early stopping patience
    curriculum_schedule : [stage1_end, stage2_end] epochs; default [10, 20]
    val_ratio        : fraction of non-held-out records used for val
    device           : torch device (auto-detected if None)
    seed             : random seed
    save_dir         : directory to save per-fold checkpoints
    verbose          : print epoch summaries

    Returns
    -------
    dict with:
      'folds'     : List[FoldMetrics]   per-fold results
      'aggregate' : dict                mean/std across folds
      'models'    : List[nn.Module]     one trained model per fold
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    if curriculum_schedule is None:
        curriculum_schedule = [10, 20]

    # Merge datasets: anchor + optional non-anchor (PDBbind)
    if non_anchor_ds is not None:
        from torch.utils.data import ConcatDataset
        full_dataset = ConcatDataset([anchor_dataset, non_anchor_ds])
    else:
        full_dataset = anchor_dataset

    # Capture pre-trained state for resetting between folds
    pretrained_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    splitter = LOLOCVSplitter(val_ratio=val_ratio, seed=seed)
    fold_list:  list[LOLOFold]    = list(splitter.split(full_dataset))
    fold_metrics: list[FoldMetrics] = []
    trained_models: list[nn.Module]  = []

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    folds_to_run = fold_list if run_lolo_cv else fold_list[:1]

    for fold in folds_to_run:
        logger.info(
            "\n── Fold %d/%d  held-out: '%s'  "
            "(train=%d, val=%d, test=%d) ──",
            fold.fold + 1, len(folds_to_run),
            fold.held_out_ligand, fold.n_train, fold.n_val, fold.n_test,
        )
        LOLOCVSplitter.verify_no_leakage(fold)

        # Reset to pre-trained weights for a clean start
        fold_model = copy.deepcopy(model)
        fold_model.load_state_dict(
            {k: v.clone().to(device) for k, v in pretrained_state.items()}
        )
        fold_model = fold_model.to(device)

        fold_model, fm = _train_one_fold(
            model               = fold_model,
            fold                = fold,
            dataset             = full_dataset,
            stage               = stage,
            max_epochs          = max_epochs,
            batch_size          = batch_size,
            lr                  = lr,
            weight_decay        = weight_decay,
            warmup_epochs       = warmup_epochs,
            patience            = patience,
            curriculum_schedule = curriculum_schedule,
            device              = device,
            verbose             = verbose,
        )
        fold_metrics.append(fm)
        trained_models.append(fold_model.cpu())

        if save_dir is not None:
            ckpt_path = save_dir / f"fold_{fold.fold:02d}_{fold.held_out_ligand}.pt"
            torch.save({'state_dict': fold_model.state_dict(), 'metrics': fm.to_dict()},
                       ckpt_path)
            logger.info("   Saved → %s", ckpt_path)

    agg = aggregate_folds(fold_metrics)
    logger.info(
        "\nLOLO-CV RMSE: %.4f ± %.4f kcal/mol  "
        "MAE: %.4f ± %.4f  r: %.4f ± %.4f",
        agg.get('rmse_mean', float('nan')), agg.get('rmse_std', float('nan')),
        agg.get('mae_mean',  float('nan')), agg.get('mae_std',  float('nan')),
        agg.get('pearson_r_mean', float('nan')), agg.get('pearson_r_std', float('nan')),
    )

    return {
        'folds':     fold_metrics,
        'aggregate': agg,
        'models':    trained_models,
    }
