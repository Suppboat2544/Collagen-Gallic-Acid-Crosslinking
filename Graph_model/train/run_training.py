"""
Graph_model.train.run_training
=================================
Full training pipeline with tqdm progress bars and per-epoch JSON tracking.

Trains all five model options (A–E) with identical data splits, records
every metric at every epoch, and writes JSON result files for downstream
visualization.

Usage
-----
    >>> from Graph_model.train.run_training import train_all_models
    >>> results = train_all_models(dataset, results_dir="results/")

    # Or train a single model:
    >>> from Graph_model.train.run_training import train_single_model
    >>> res = train_single_model("E", dataset, results_dir="results/")
"""

from __future__ import annotations

import copy
import gc
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Subset

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from ..model import (
    OptionA, OptionB, OptionC, OptionD, OptionE,
    OptionAConfig, OptionBConfig, OptionCConfig, OptionDConfig,
    DimeNet, EGNN, GGNNSequential, Graphormer,
    DimeNetConfig, EGNNConfig, GGNNSeqConfig, GraphormerConfig,
)
from ..model.option_e import OptionEConfig
from .metrics import regression_metrics, FoldMetrics, aggregate_folds, print_lolo_report
from .pretrain import _forward, _cosine_schedule
from .finetune import freeze_backbone_stages

logger = logging.getLogger(__name__)

# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "A": {"cls": OptionA, "cfg_cls": OptionAConfig, "name": "Option A — Baseline GATv2",       "max_batch_size": 16},
    "B": {"cls": OptionB, "cfg_cls": OptionBConfig, "name": "Option B — Dual Encoder CrossAttn", "max_batch_size": 16},
    "C": {"cls": OptionC, "cfg_cls": OptionCConfig, "name": "Option C — Fragment MPNN",          "max_batch_size": 16},
    "D": {"cls": OptionD, "cfg_cls": OptionDConfig, "name": "Option D — Multi-Task Selectivity",  "max_batch_size": 16},
    "E": {"cls": OptionE, "cfg_cls": OptionEConfig, "name": "Option E — pGET (novel)",            "max_batch_size": 16},
    # Phase 8 — new architectures (smaller batch sizes for memory-heavy models)
    "F": {"cls": DimeNet,       "cfg_cls": DimeNetConfig,    "name": "Option F — DimeNet++",       "max_batch_size": 8},
    "G": {"cls": EGNN,          "cfg_cls": EGNNConfig,       "name": "Option G — EGNN",            "max_batch_size": 16},
    "H": {"cls": GGNNSequential,"cfg_cls": GGNNSeqConfig,    "name": "Option H — GGNN Sequential", "max_batch_size": 16},
    "I": {"cls": Graphormer,    "cfg_cls": GraphormerConfig,  "name": "Option I — Graphormer",      "max_batch_size": 16},
}


def _build_model(key: str, cfg_overrides: dict | None = None) -> nn.Module:
    """Instantiate a model by its registry key."""
    entry = MODEL_REGISTRY[key]
    cfg = entry["cfg_cls"](**(cfg_overrides or {}))
    return entry["cls"](cfg)


# ── Forward wrapper that handles Option E aux dict ─────────────────────────────

def _forward_any(model: nn.Module, batch, device: torch.device) -> tuple[torch.Tensor, dict]:
    """
    Unified forward: always returns (pred [B,1], aux_dict).
    Handles all model return signatures.
    """
    batch = batch.to(device)
    out = model(batch)

    if isinstance(out, dict):
        # Option D multi-task head
        return out['collagen'], {'all_heads': out}
    if isinstance(out, (tuple, list)):
        pred = out[0]
        if len(out) <= 1:
            return pred, {}
        secondary = out[1]
        if isinstance(secondary, dict):
            # Option E returns (pred, {'denoise_loss': ...})
            return pred, secondary
        if isinstance(secondary, list):
            # Option B returns (pred, [attn_dict, ...])
            # Option C returns (pred, [frag_contribs])
            return pred, {'attn_summaries': secondary}
        # Scalar or tensor auxiliary output
        return pred, {'aux': secondary}
    return out, {}


# ── Per-epoch JSON logger ──────────────────────────────────────────────────────

class EpochLogger:
    """Logs per-epoch metrics to a JSON file incrementally."""

    def __init__(self, path: Path, model_name: str, meta: dict | None = None):
        self.path = path
        self.data = {
            "model_name":  model_name,
            "created_at":  datetime.now().isoformat(),
            "meta":        meta or {},
            "epochs":      [],
        }
        self._save()

    def log_epoch(self, epoch: int, metrics: dict):
        entry = {"epoch": epoch, "timestamp": datetime.now().isoformat()}
        entry.update(metrics)
        self.data["epochs"].append(entry)
        self._save()

    def finalise(self, summary: dict):
        self.data["summary"] = summary
        self.data["finished_at"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2, default=_json_safe)


def _json_safe(obj):
    """JSON serialiser for numpy/torch types."""
    if isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return str(obj)


# ── Detect device ──────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ── Single model training (with tqdm + JSON tracking) ──────────────────────────

def train_single_model(
    model_key:    str,
    dataset,
    *,
    results_dir:  str | Path       = "results",
    max_epochs:   int              = 100,
    batch_size:   int              = 32,
    lr:           float            = 1e-3,
    weight_decay: float            = 1e-4,
    warmup_epochs: int             = 5,
    patience:     int              = 15,
    val_ratio:    float            = 0.15,
    device:       torch.device | None = None,
    seed:         int              = 42,
    cfg_overrides: dict | None     = None,
    disable_tqdm: bool             = False,
    fold_indices: list[dict] | None = None,
) -> dict:
    """
    Train a single model with full tqdm progress bars and per-epoch JSON logging.

    Parameters
    ----------
    model_key     : one of 'A', 'B', 'C', 'D', 'E'
    dataset       : PyG dataset (CollagenDockingDataset or similar)
    results_dir   : directory where JSON epoch logs are saved
    max_epochs    : maximum training epochs
    batch_size    : mini-batch size
    lr            : learning rate
    weight_decay  : L2 regularisation
    warmup_epochs : cosine LR warmup
    patience      : early stopping patience (epochs without val improvement)
    val_ratio     : fraction to use as validation
    device        : torch device (auto-detected if None)
    seed          : random seed
    cfg_overrides : optional dict of config overrides
    disable_tqdm  : set True to suppress progress bars (e.g., in tests)
    fold_indices  : optional pre-computed train/val/test splits as list of dicts

    Returns
    -------
    dict with:
      'model'        : trained nn.Module
      'train_losses' : list per epoch
      'val_losses'   : list per epoch
      'val_metrics'  : list of dicts per epoch (rmse, mae, pearson_r, spearman_r)
      'best_epoch'   : int
      'best_val_rmse': float
      'epoch_log_path': Path to JSON log file
      'wall_time_s'  : total training wall time
    """
    device = device or _auto_device()
    if isinstance(device, str):
        device = torch.device(device)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODEL_REGISTRY[model_key]
    model_name = model_info["name"]

    # Build model
    model = _build_model(model_key, cfg_overrides)
    model = model.to(device)

    # Train/val split
    torch.manual_seed(seed)
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:].tolist()

    if fold_indices:
        train_idx = fold_indices[0].get('train', train_idx)
        val_idx = fold_indices[0].get('val', val_idx)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Optimizer + scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _cosine_schedule(optimizer, warmup_epochs, max_epochs)
    scheduler._step_count = 1  # suppress PyTorch "step before optimizer" warning
    criterion = nn.MSELoss()

    # JSON logger
    log_path = results_dir / f"option_{model_key.lower()}_training.json"
    epoch_logger = EpochLogger(log_path, model_name, meta={
        "max_epochs": max_epochs, "batch_size": batch_size,
        "lr": lr, "weight_decay": weight_decay, "warmup_epochs": warmup_epochs,
        "patience": patience, "n_train": n_train, "n_val": n_val,
        "device": str(device), "seed": seed,
    })

    best_val_mse = float('inf')
    best_state = None
    best_epoch = 0
    patience_cnt = 0
    RSS_LIMIT_MB = 20_000   # graceful stop before macOS OOM-kills the process

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_metrics_list: list[dict] = []

    start_time = time.time()

    # ── Training loop with tqdm ────────────────────────────────────────────────
    epoch_bar = tqdm(
        range(max_epochs),
        desc=f"Training {model_name}",
        unit="epoch",
        disable=disable_tqdm,
        leave=True,
    )

    for epoch in epoch_bar:
        # ── Train phase ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        all_train_preds = []
        all_train_targets = []

        batch_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{max_epochs} [train]",
            unit="batch",
            leave=False,
            disable=disable_tqdm,
        )

        pred = target = loss = aux = None  # pre-declare for cleanup

        for batch in batch_bar:
            optimizer.zero_grad()
            try:
                pred, aux = _forward_any(model, batch, device)
                target = batch.y.to(device).view(-1, 1)
                loss = criterion(pred, target)

                # Add auxiliary denoising loss for Option E
                if 'denoise_loss' in aux:
                    loss = loss + aux['denoise_loss']

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                all_train_preds.extend(pred.detach().cpu().view(-1).tolist())
                all_train_targets.extend(target.detach().cpu().view(-1).tolist())

                running_avg = epoch_loss / n_batches
                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    avg=f"{running_avg:.4f}",
                )
            except Exception as exc:
                import traceback as _tb
                logger.warning("Train batch error epoch %d: %s", epoch, exc)
                logger.debug("Train batch traceback:\n%s", _tb.format_exc())
                if n_batches == 0:
                    raise

            # Free MPS/CUDA tensors promptly to prevent memory leak
            del pred, target, loss, aux
            pred = target = loss = aux = None

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        train_m = regression_metrics(all_train_preds, all_train_targets)

        # ── Validation phase ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val_b = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                try:
                    pred, aux = _forward_any(model, batch, device)
                    target = batch.y.to(device).view(-1, 1)
                    vl = criterion(pred, target)
                    val_loss += vl.item()
                    n_val_b += 1

                    all_val_preds.extend(pred.cpu().view(-1).tolist())
                    all_val_targets.extend(target.cpu().view(-1).tolist())
                except Exception as exc:
                    logger.warning("Val batch error epoch %d: %s", epoch, exc)
                    if n_val_b == 0:
                        raise
                    continue

        avg_val_loss = val_loss / max(n_val_b, 1)
        val_losses.append(avg_val_loss)

        val_m = regression_metrics(all_val_preds, all_val_targets)
        val_metrics_list.append(val_m)

        # ── Per-epoch memory cleanup (prevents OOM on long MPS runs) ──────
        del all_train_preds, all_train_targets
        del all_val_preds, all_val_targets
        gc.collect()
        if device.type == 'mps':
            torch.mps.synchronize()   # wait for async MPS ops to finish
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # ── Log epoch to JSON ──────────────────────────────────────────────────
        epoch_data = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "train_rmse": train_m['rmse'],
            "train_mae": train_m['mae'],
            "train_pearson_r": train_m['pearson_r'],
            "train_spearman_r": train_m['spearman_r'],
            "val_rmse": val_m['rmse'],
            "val_mae": val_m['mae'],
            "val_pearson_r": val_m['pearson_r'],
            "val_spearman_r": val_m['spearman_r'],
            "learning_rate": current_lr,
            "wall_time_s": time.time() - start_time,
        }
        epoch_logger.log_epoch(epoch, epoch_data)

        # ── Print epoch summary ────────────────────────────────────────────
        val_r_str = f"{val_m['pearson_r']:.4f}" if not math.isnan(val_m['pearson_r']) else "N/A"
        val_sr_str = f"{val_m['spearman_r']:.4f}" if not math.isnan(val_m['spearman_r']) else "N/A"

        # Memory monitoring (RSS + MPS GPU)
        mem_str = ""
        try:
            import resource
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            if device.type == 'mps':
                mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                drv_mb = torch.mps.driver_allocated_memory() / (1024 * 1024)
                mem_str = f"  rss={rss_mb:.0f}MB  mps={mps_mb:.0f}/{drv_mb:.0f}MB"
            else:
                mem_str = f"  mem={rss_mb:.0f}MB"
        except Exception:
            pass

        tqdm.write(
            f"  Epoch {epoch+1:3d}/{max_epochs} │ "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_RMSE={val_m['rmse']:.4f}  "
            f"val_MAE={val_m['mae']:.4f}  "
            f"val_r={val_r_str}  "
            f"val_ρ={val_sr_str}  "
            f"lr={current_lr:.2e}"
            f"{mem_str}"
        )

        # ── Update tqdm bar ────────────────────────────────────────────────────
        epoch_bar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",
            "val_RMSE": f"{val_m['rmse']:.4f}",
            "val_r": val_r_str,
            "lr": f"{current_lr:.2e}",
        })

        # ── RSS memory safety check (macOS OOM-killer prevention) ─────────
        try:
            import resource
            current_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
            if current_rss_mb > RSS_LIMIT_MB:
                logger.warning(
                    "RSS %.0fMB exceeds limit %dMB — stopping early to avoid OOM kill",
                    current_rss_mb, RSS_LIMIT_MB,
                )
                break
        except Exception:
            pass

        # ── Early stopping ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_mse:
            best_val_mse = avg_val_loss
            old_state = best_state
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            del old_state  # free previous best checkpoint
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    wall_time = time.time() - start_time

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Finalise JSON log (use train_m / val_m from last epoch — lists are freed per-epoch)
    summary = {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_rmse": best_val_mse ** 0.5,
        "total_epochs": len(train_losses),
        "wall_time_s": wall_time,
        "final_train_metrics": train_m,
        "final_val_metrics": val_m,
    }
    epoch_logger.finalise(summary)

    return {
        "model": model,
        "model_key": model_key,
        "model_name": model_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_metrics": val_metrics_list,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_mse ** 0.5,
        "epoch_log_path": log_path,
        "wall_time_s": wall_time,
    }


# ── Train all models ───────────────────────────────────────────────────────────

def train_all_models(
    dataset,
    *,
    model_keys:   list[str] | None = None,
    results_dir:  str | Path       = "results",
    max_epochs:   int              = 100,
    batch_size:   int              = 32,
    lr:           float            = 1e-3,
    weight_decay: float            = 1e-4,
    warmup_epochs: int             = 5,
    patience:     int              = 15,
    val_ratio:    float            = 0.15,
    device:       torch.device | None = None,
    seed:         int              = 42,
    disable_tqdm: bool             = False,
) -> dict[str, dict]:
    """
    Train all (or selected) models and produce per-epoch JSON logs.

    Parameters
    ----------
    dataset      : shared dataset for all models
    model_keys   : list like ['A','B','C','D','E'] or None for all
    results_dir  : directory for JSON output
    ...          : training hyperparameters (same for all models)

    Returns
    -------
    dict mapping model_key → result dict from train_single_model
    """
    if model_keys is None:
        model_keys = list(MODEL_REGISTRY.keys())

    device = device or _auto_device()
    results_dir = Path(results_dir)
    all_results: dict[str, dict] = {}

    # Use same train/val split for all models (fair comparison)
    n = len(dataset)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train:].tolist()
    fold_indices = [{'train': train_idx, 'val': val_idx}]

    overall_bar = tqdm(
        model_keys,
        desc="Training all models",
        unit="model",
        disable=disable_tqdm,
    )

    for key in overall_bar:
        overall_bar.set_description(f"Training {MODEL_REGISTRY[key]['name']}")

        # Per-model batch size limit (for memory-heavy models like DimeNet++)
        model_bs = batch_size
        max_bs = MODEL_REGISTRY[key].get('max_batch_size')
        if max_bs is not None and batch_size > max_bs:
            model_bs = max_bs
            logger.info("Model %s: reducing batch_size %d → %d (memory limit)",
                        key, batch_size, model_bs)

        result = train_single_model(
            model_key     = key,
            dataset       = dataset,
            results_dir   = results_dir,
            max_epochs    = max_epochs,
            batch_size    = model_bs,
            lr            = lr,
            weight_decay  = weight_decay,
            warmup_epochs = warmup_epochs,
            patience      = patience,
            val_ratio     = val_ratio,
            device        = device,
            seed          = seed,
            disable_tqdm  = disable_tqdm,
            fold_indices  = fold_indices,
        )
        all_results[key] = result
        logger.info(
            "Model %s done: best_val_RMSE=%.4f at epoch %d (%.1f s)",
            key, result['best_val_rmse'], result['best_epoch'], result['wall_time_s']
        )

        # Free GPU/MPS memory between models
        import gc
        del result['model']
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Write comparison summary ───────────────────────────────────────────────
    comparison = {
        "created_at": datetime.now().isoformat(),
        "models": {},
    }
    for key, res in all_results.items():
        comparison["models"][key] = {
            "name": res["model_name"],
            "best_epoch": res["best_epoch"],
            "best_val_rmse": res["best_val_rmse"],
            "total_epochs": len(res["train_losses"]),
            "wall_time_s": res["wall_time_s"],
            "final_val_metrics": res["val_metrics"][-1] if res["val_metrics"] else {},
        }

    comp_path = results_dir / "comparison_summary.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=_json_safe)

    logger.info("Comparison summary saved → %s", comp_path)
    return all_results


# ── LOLO-CV training with tqdm + JSON ──────────────────────────────────────────

def train_lolo_cv(
    model_key:     str,
    dataset,
    *,
    results_dir:   str | Path       = "results",
    max_epochs:    int              = 50,
    batch_size:    int              = 32,
    lr:            float            = 1e-3,
    weight_decay:  float            = 1e-4,
    warmup_epochs: int              = 5,
    patience:      int              = 10,
    val_ratio:     float            = 0.15,
    device:        torch.device | None = None,
    seed:          int              = 42,
    cfg_overrides: dict | None      = None,
    disable_tqdm:  bool             = False,
) -> dict:
    """
    Train a model using LOLO-CV with full tracking.

    Each fold trains independently, with per-epoch JSON logging and tqdm.

    Returns
    -------
    dict with:
      'folds'       : list of FoldMetrics
      'aggregate'   : dict of mean/std metrics
      'fold_logs'   : list of epoch log file paths
      'wall_time_s' : total wall time
    """
    from .lolo_cv import LOLOCVSplitter

    device = device or _auto_device()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODEL_REGISTRY[model_key]
    model_name = model_info["name"]

    # Build base model
    base_model = _build_model(model_key, cfg_overrides)
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    # Get folds
    splitter = LOLOCVSplitter(val_ratio=val_ratio, seed=seed)
    fold_list = list(splitter.split(dataset))

    fold_metrics: list[FoldMetrics] = []
    fold_logs: list[Path] = []
    start_time = time.time()

    fold_bar = tqdm(
        fold_list,
        desc=f"LOLO-CV {model_name}",
        unit="fold",
        disable=disable_tqdm,
    )

    for fold in fold_bar:
        fold_bar.set_description(
            f"LOLO-CV fold {fold.fold+1}/{len(fold_list)} [{fold.held_out_ligand}]"
        )

        # Reset model to initial state
        fold_model = copy.deepcopy(base_model)
        fold_model.load_state_dict(
            {k: v.clone() for k, v in base_state.items()}
        )
        fold_model = fold_model.to(device)

        # Setup data loaders
        train_ds = Subset(dataset, fold.train_idx)
        val_ds = Subset(dataset, fold.val_idx)
        test_ds = Subset(dataset, fold.test_idx)

        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = PyGDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(fold_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = _cosine_schedule(optimizer, warmup_epochs, max_epochs)
        criterion = nn.MSELoss()

        # Per-fold JSON logger
        fold_log_path = results_dir / f"option_{model_key.lower()}_fold_{fold.fold:02d}.json"
        fold_logger = EpochLogger(fold_log_path, model_name, meta={
            "fold": fold.fold, "held_out": fold.held_out_ligand,
            "n_train": fold.n_train, "n_val": fold.n_val, "n_test": fold.n_test,
        })
        fold_logs.append(fold_log_path)

        best_val_mse = float('inf')
        best_state = None
        best_epoch = 0
        patience_cnt = 0

        epoch_bar = tqdm(
            range(max_epochs),
            desc=f"  Fold {fold.fold+1}",
            unit="ep",
            leave=False,
            disable=disable_tqdm,
        )

        for epoch in epoch_bar:
            # Train
            fold_model.train()
            ep_loss = 0.0
            nb = 0
            for batch in train_loader:
                optimizer.zero_grad()
                try:
                    pred, aux = _forward_any(fold_model, batch, device)
                    target = batch.y.to(device).view(-1, 1)
                    loss = criterion(pred, target)
                    if 'denoise_loss' in aux:
                        loss = loss + aux['denoise_loss']
                    loss.backward()
                    nn.utils.clip_grad_norm_(fold_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    ep_loss += loss.item()
                    nb += 1
                except Exception:
                    continue

            avg_train = ep_loss / max(nb, 1)

            # Val
            fold_model.eval()
            vl = 0.0
            nvb = 0
            val_preds, val_tgts = [], []
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        pred, _ = _forward_any(fold_model, batch, device)
                        target = batch.y.to(device).view(-1, 1)
                        vl += criterion(pred, target).item()
                        nvb += 1
                        val_preds.extend(pred.cpu().view(-1).tolist())
                        val_tgts.extend(target.cpu().view(-1).tolist())
                    except Exception:
                        continue

            avg_val = vl / max(nvb, 1)
            val_m = regression_metrics(val_preds, val_tgts)
            scheduler.step()

            fold_logger.log_epoch(epoch, {
                "train_loss": avg_train, "val_loss": avg_val,
                "val_rmse": val_m['rmse'], "val_mae": val_m['mae'],
                "val_pearson_r": val_m['pearson_r'],
            })

            epoch_bar.set_postfix(val_RMSE=f"{val_m['rmse']:.4f}")

            if avg_val < best_val_mse:
                best_val_mse = avg_val
                best_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        # Load best state and evaluate on test
        if best_state is not None:
            fold_model.load_state_dict(best_state)
            fold_model = fold_model.to(device)

        fold_model.eval()
        test_preds, test_tgts = [], []
        with torch.no_grad():
            for batch in test_loader:
                try:
                    pred, _ = _forward_any(fold_model, batch, device)
                    test_preds.extend(pred.cpu().view(-1).tolist())
                    test_tgts.extend(batch.y.view(-1).tolist())
                except Exception:
                    continue

        tm = regression_metrics(test_preds, test_tgts)
        fm = FoldMetrics(
            fold=fold.fold, held_out=fold.held_out_ligand, n_test=fold.n_test,
            rmse=tm['rmse'], mae=tm['mae'],
            pearson_r=tm['pearson_r'], spearman_r=tm['spearman_r'],
            n_train=fold.n_train, n_val=fold.n_val,
            best_epoch=best_epoch, val_rmse=best_val_mse ** 0.5,
        )
        fold_metrics.append(fm)

        fold_logger.finalise({
            "test_rmse": tm['rmse'], "test_mae": tm['mae'],
            "test_pearson_r": tm['pearson_r'], "best_epoch": best_epoch,
        })

    agg = aggregate_folds(fold_metrics)
    wall_time = time.time() - start_time

    # Write overall LOLO summary
    lolo_summary = {
        "model_key": model_key,
        "model_name": model_name,
        "aggregate": agg,
        "folds": [fm.to_dict() for fm in fold_metrics],
        "wall_time_s": wall_time,
    }
    summary_path = results_dir / f"option_{model_key.lower()}_lolo_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(lolo_summary, f, indent=2, default=_json_safe)

    return {
        "folds": fold_metrics,
        "aggregate": agg,
        "fold_logs": fold_logs,
        "wall_time_s": wall_time,
    }
