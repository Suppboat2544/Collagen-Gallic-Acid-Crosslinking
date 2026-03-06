"""
Graph_model.train.hpo
========================
Bayesian Hyperparameter Optimisation with Optuna.

Problem 5c — Rather than hand-tuning hyperparameters, use Tree-structured
Parzen Estimator (TPE) to efficiently search the hyperparameter space.

Search space
------------
• hidden_dim      : [64, 128, 256]
• n_layers        : [2, 3, 4, 5, 6]
• dropout         : [0.05, 0.30]
• lr              : [1e-4, 1e-2]  (log-uniform)
• weight_decay    : [1e-6, 1e-3]  (log-uniform)
• gat_heads       : [2, 4, 8]
• batch_size      : [16, 32, 64]
• loss_alpha      : [0.5, 1.5]    (MSE weight)
• loss_beta       : [0.0, 0.5]    (rank loss weight)
• loss_gamma      : [0.0, 0.2]    (monotonicity weight)

Objective: Minimize mean validation RMSE across 3-fold LOLO-CV
         (3 folds for speed during HPO; full 9-fold for final eval)

Public API
----------
  OptunaTuner(model_key, dataset, n_trials, ...)
  run_hpo(model_key, dataset, ...)

References
----------
• Akiba T. et al., "Optuna: A Next-generation Hyperparameter Optimization
  Framework," KDD 2019.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

logger = logging.getLogger(__name__)


@dataclass
class HPOConfig:
    """Optuna HPO configuration."""
    n_trials:            int   = 50    # number of Optuna trials
    n_cv_folds:          int   = 3     # CV folds for HPO (subset for speed)
    max_epochs_per_trial: int  = 30    # max epochs per trial
    early_stop_patience: int   = 5     # early stopping patience
    study_name:          str   = "docking_hpo"
    direction:           str   = "minimize"  # minimize RMSE
    pruner:              str   = "median"    # Optuna pruner type
    device:              str   = "cpu"
    results_dir:         str   = "hpo_results"


def _suggest_hyperparams(trial) -> dict[str, Any]:
    """Suggest hyperparameters using Optuna trial."""
    return {
        "hidden_dim":    trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "n_layers":      trial.suggest_int("n_layers", 2, 6),
        "dropout":       trial.suggest_float("dropout", 0.05, 0.30),
        "lr":            trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "gat_heads":     trial.suggest_categorical("gat_heads", [2, 4, 8]),
        "batch_size":    trial.suggest_categorical("batch_size", [16, 32, 64]),
        "loss_alpha":    trial.suggest_float("loss_alpha", 0.5, 1.5),
        "loss_beta":     trial.suggest_float("loss_beta", 0.0, 0.5),
        "loss_gamma":    trial.suggest_float("loss_gamma", 0.0, 0.2),
    }


def _train_and_evaluate(
    model: nn.Module,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: torch.device,
    trial=None,
) -> float:
    """
    Train model and return best validation RMSE.
    Supports Optuna pruning.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )

    best_val_rmse = float("inf")
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            if isinstance(out, dict):
                pred = out.get('collagen', list(out.values())[0])
            elif isinstance(out, (tuple, list)):
                pred = out[0]
            else:
                pred = out

            target = batch.y.view(-1, 1).to(device)
            loss = nn.functional.mse_loss(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            n_train += target.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                if isinstance(out, dict):
                    pred = out.get('collagen', list(out.values())[0])
                elif isinstance(out, (tuple, list)):
                    pred = out[0]
                else:
                    pred = out
                target = batch.y.view(-1, 1).to(device)
                val_preds.append(pred.cpu())
                val_targets.append(target.cpu())

        val_preds   = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse    = torch.sqrt(nn.functional.mse_loss(val_preds, val_targets)).item()

        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Optuna pruning
        if trial is not None:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise _get_optuna().exceptions.TrialPruned()

    return best_val_rmse


def _get_optuna():
    """Lazy import of optuna."""
    try:
        import optuna
        return optuna
    except ImportError:
        raise ImportError(
            "optuna is required for HPO. Install with: pip install optuna"
        )


class OptunaTuner:
    """
    Bayesian hyperparameter optimisation using Optuna.

    Uses Tree-structured Parzen Estimator (TPE) with optional median
    pruning for efficient search.
    """

    def __init__(
        self,
        model_key: str,
        dataset,
        cfg: HPOConfig | None = None,
    ) -> None:
        self.model_key = model_key
        self.dataset   = dataset
        self.cfg       = cfg or HPOConfig()

    def _build_model(self, hparams: dict) -> nn.Module:
        """Build model with suggested hyperparameters."""
        # Import model registry
        from .run_training import MODEL_REGISTRY
        entry = MODEL_REGISTRY[self.model_key]
        cfg_cls = entry["cfg_cls"]

        # Map HPO params to model config
        cfg_kwargs = {}
        if "hidden_dim" in hparams:
            cfg_kwargs["hidden_dim"] = hparams["hidden_dim"]
            cfg_kwargs["mlp_hidden"] = hparams["hidden_dim"] * 2
        if "n_layers" in hparams:
            cfg_kwargs["n_layers"] = hparams["n_layers"]
        if "dropout" in hparams:
            cfg_kwargs["dropout"] = hparams["dropout"]
        if "gat_heads" in hparams:
            cfg_kwargs["gat_heads"] = hparams["gat_heads"]
            cfg_kwargs["gat_head_dim"] = hparams["hidden_dim"] // hparams["gat_heads"]

        cfg = cfg_cls(**cfg_kwargs)
        return entry["cls"](cfg)

    def _get_cv_folds(self) -> list:
        """Get cross-validation folds (subset for speed)."""
        from .lolo_cv import LOLOCVSplitter
        splitter = LOLOCVSplitter(val_ratio=0.15, seed=42)
        folds = list(splitter.split(self.dataset))
        # Take only first n_cv_folds for speed during HPO
        return folds[:self.cfg.n_cv_folds]

    def _objective(self, trial) -> float:
        """Optuna objective function."""
        hparams = _suggest_hyperparams(trial)
        device  = torch.device(self.cfg.device)

        folds  = self._get_cv_folds()
        rmses  = []

        for fold in folds:
            model = self._build_model(hparams)

            train_ds = Subset(self.dataset, fold.train_idx)
            val_ds   = Subset(self.dataset, fold.test_idx)

            bs = hparams.get("batch_size", 32)
            train_loader = PyGDataLoader(train_ds, batch_size=bs, shuffle=True)
            val_loader   = PyGDataLoader(val_ds,   batch_size=bs, shuffle=False)

            rmse = _train_and_evaluate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=hparams["lr"],
                weight_decay=hparams["weight_decay"],
                max_epochs=self.cfg.max_epochs_per_trial,
                patience=self.cfg.early_stop_patience,
                device=device,
                trial=trial,
            )
            rmses.append(rmse)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        mean_rmse = sum(rmses) / len(rmses) if rmses else float("inf")
        return mean_rmse

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """
        Run Optuna HPO study.

        Returns dict with 'best_params', 'best_value', 'study'.
        """
        optuna = _get_optuna()

        # Create pruner
        if self.cfg.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        elif self.cfg.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        study = optuna.create_study(
            study_name=self.cfg.study_name,
            direction=self.cfg.direction,
            pruner=pruner,
        )

        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study.optimize(
            self._objective,
            n_trials=self.cfg.n_trials,
            show_progress_bar=verbose,
        )

        result = {
            "best_params": study.best_params,
            "best_value":  study.best_value,
            "n_trials":    len(study.trials),
            "study":       study,
        }

        logger.info(f"HPO complete: best RMSE={study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return result


def run_hpo(
    model_key: str,
    dataset,
    cfg: HPOConfig | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convenience function for Optuna HPO.

    Returns dict with best hyperparameters and study results.
    """
    cfg = cfg or HPOConfig()
    tuner = OptunaTuner(model_key, dataset, cfg)
    return tuner.run(verbose=verbose)
