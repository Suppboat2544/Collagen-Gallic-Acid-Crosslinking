"""
Graph_model.train
==================
Phase 4 — Training Strategy + Phase 8 Improvements

Contains:
  LOLOCVSplitter      Leave-One-Ligand-Out 9-fold cross-validation
  CurriculumSampler   Complexity-ordered batch sampler (3 difficulty stages)
  PDBbindGraphDataset PyG dataset built from Graph_model/external_dataset/
  Trainer             Two-stage transfer learning controller
  FoldMetrics         Per-fold RMSE / Pearson-r / Spearman-r report
  aggregate_folds     Summary statistics (mean ± std) across LOLO-CV folds
  train_single_model  Per-model training with tqdm + JSON epoch logging
  train_all_models    Train all 5 models in sequence with shared splits

  Phase 8 additions:
  MAMLTrainer         Model-Agnostic Meta-Learning for few-shot ligands
  ContrastivePretrainer  NT-Xent contrastive pre-training
  OptunaTuner         Bayesian hyperparameter optimization
  StratifiedEvaluator Fine-grained evaluation breakdowns
  ScaffoldSplitter    Murcko scaffold-based train/test splitting
"""

from .lolo_cv     import LOLOCVSplitter
from .curriculum  import CurriculumSampler, STAGE_LIGANDS, STAGE_WEIGHTS
from .pdbbind_dataset import PDBbindGraphDataset, parse_pdbbind_index
from .pretrain    import pretrain
from .finetune    import finetune, freeze_backbone_stages
from .metrics     import FoldMetrics, aggregate_folds, regression_metrics
from .run_training import (
    train_single_model,
    train_all_models,
    train_lolo_cv,
    MODEL_REGISTRY,
    EpochLogger,
)

# Phase 8 — new training modules
from .maml import MAMLTrainer, MAMLConfig, maml_train
from .contrastive import ContrastivePretrainer, ContrastiveConfig, contrastive_pretrain
from .hpo import OptunaTuner, HPOConfig, run_hpo
from .stratified_eval import StratifiedEvaluator, StratifiedReport, StrataMetrics
from .scaffold_split import ScaffoldSplitter, ScaffoldFold

__all__ = [
    "LOLOCVSplitter",
    "CurriculumSampler", "STAGE_LIGANDS", "STAGE_WEIGHTS",
    "PDBbindGraphDataset", "parse_pdbbind_index",
    "pretrain",
    "finetune", "freeze_backbone_stages",
    "FoldMetrics", "aggregate_folds", "regression_metrics",
    # Full training pipeline
    "train_single_model", "train_all_models", "train_lolo_cv",
    "MODEL_REGISTRY", "EpochLogger",
    # Phase 8 — MAML
    "MAMLTrainer", "MAMLConfig", "maml_train",
    # Phase 8 — Contrastive
    "ContrastivePretrainer", "ContrastiveConfig", "contrastive_pretrain",
    # Phase 8 — HPO
    "OptunaTuner", "HPOConfig", "run_hpo",
    # Phase 8 — Stratified evaluation
    "StratifiedEvaluator", "StratifiedReport", "StrataMetrics",
    # Phase 8 — Scaffold splitting
    "ScaffoldSplitter", "ScaffoldFold",
]
