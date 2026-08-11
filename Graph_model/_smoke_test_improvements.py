"""
Graph_model._smoke_test_improvements
========================================
Comprehensive smoke tests for all Phase 8 improvements.

Tests 18 new components across 7 categories:
  1. Input features:  conformer_3d, ecfp, box_residues
  2. Architectures:   DimeNet++, EGNN, GGNNSequential, Graphormer
  3. Loss functions:  ListMLE, Monotonicity, Pairwise, Combined
  4. Training:        MAML, Contrastive, HPO
  5. Evaluation:      StratifiedEvaluator, ScaffoldSplitter
  6. Interpretability: GradCAM, Probing, AttentionRollout
  7. Registry:        MODEL_REGISTRY updated, __init__.py imports

Run:
    python -m Graph_model._smoke_test_improvements
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

import torch
import numpy as np

PASS = 0
FAIL = 0
RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = "") -> None:
    global PASS, FAIL
    if passed:
        PASS += 1
        RESULTS.append((name, True, detail))
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        RESULTS.append((name, False, detail))
        print(f"  ❌ {name}  —  {detail}")


def _make_dummy_heterodata(n_atoms: int = 12, n_bonds: int = 24,
                            batch_size: int = 2) -> Any:
    """Create a dummy HeteroData batch for smoke testing."""
    from torch_geometric.data import HeteroData, Batch

    data_list = []
    for _ in range(batch_size):
        d = HeteroData()
        d['ligand'].x = torch.randn(n_atoms, 35)
        ei = torch.randint(0, n_atoms, (2, n_bonds))
        d['ligand', 'bond', 'ligand'].edge_index = ei
        d['ligand', 'bond', 'ligand'].edge_attr = torch.randn(n_bonds, 13)
        d.ph_enc = torch.tensor(0.85)
        d.temp_enc = torch.tensor(0.5)
        d.box_idx = torch.tensor(0, dtype=torch.long)
        d.receptor_flag = torch.tensor(0.0)
        d.ligand_name = "gallic_acid"
        d.tier = torch.tensor(0)
        d.delta_g = torch.tensor(-5.0)
        data_list.append(d)

    return Batch.from_data_list(data_list)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INPUT FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

def test_conformer_3d() -> None:
    """Test 3D conformer feature generation."""
    print("\n── 1a. Conformer 3D Features ──")
    try:
        from Graph_model.data.features.conformer_3d import (
            generate_conformer, conformer_node_features,
            conformer_edge_features, augment_ligand_graph_3d,
            CONFORMER_NODE_DIM, CONFORMER_EDGE_DIM,
        )
        _record("Import conformer_3d", True)
    except Exception as e:
        _record("Import conformer_3d", False, str(e))
        return

    _record("CONFORMER_NODE_DIM == 4", CONFORMER_NODE_DIM == 4)
    _record("CONFORMER_EDGE_DIM == 5", CONFORMER_EDGE_DIM == 5)

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("OC(=O)c1cc(O)c(O)c(O)c1")  # gallic acid
        mol = Chem.AddHs(mol)
        conf_mol = generate_conformer(mol, n_confs=1)
        _record("generate_conformer", conf_mol.GetNumConformers() > 0)

        node_feat = conformer_node_features(conf_mol)
        _record("conformer_node_features shape", node_feat.shape[1] == CONFORMER_NODE_DIM,
                f"got {node_feat.shape}")

        # Edge features need edge_index
        n_atoms = conf_mol.GetNumAtoms()
        edge_index = torch.randint(0, n_atoms, (2, 20))
        edge_feat = conformer_edge_features(conf_mol, edge_index)
        _record("conformer_edge_features shape", edge_feat.shape[1] == CONFORMER_EDGE_DIM,
                f"got {edge_feat.shape}")
    except Exception as e:
        _record("conformer_3d functions", False, str(e))


def test_ecfp() -> None:
    """Test ECFP auxiliary features."""
    print("\n── 1b. ECFP Auxiliary Features ──")
    try:
        from Graph_model.data.features.ecfp import (
            ecfp_node_features, ecfp_mol_features,
            augment_nodes_with_ecfp,
            ECFP_NODE_DIM, ECFP_MOL_DIM,
        )
        _record("Import ecfp", True)
    except Exception as e:
        _record("Import ecfp", False, str(e))
        return

    _record("ECFP_NODE_DIM == 32", ECFP_NODE_DIM == 32)
    _record("ECFP_MOL_DIM == 128", ECFP_MOL_DIM == 128)

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles("OC(=O)c1cc(O)c(O)c(O)c1")
        node_feat = ecfp_node_features(mol, dim=32)
        _record("ecfp_node_features shape", node_feat.shape[1] == 32,
                f"got {node_feat.shape}")

        mol_feat = ecfp_mol_features(mol, n_bits=128)
        _record("ecfp_mol_features shape", mol_feat.shape[-1] == 128,
                f"got {mol_feat.shape}")
    except Exception as e:
        _record("ecfp functions", False, str(e))


def test_box_residues() -> None:
    """Test box residue composition vectors."""
    print("\n── 1c. Box Residue Composition ──")
    try:
        from Graph_model.data.features.box_residues import (
            residue_composition, normalised_residue_composition,
            box_residue_condition_vector,
            BOX_RESIDUE_DIM, AMINO_ACIDS_20,
        )
        _record("Import box_residues", True)
    except Exception as e:
        _record("Import box_residues", False, str(e))
        return

    _record("BOX_RESIDUE_DIM == 20", BOX_RESIDUE_DIM == 20)
    _record("AMINO_ACIDS_20 has 20 entries", len(AMINO_ACIDS_20) == 20)

    try:
        residues = ["ALA", "GLY", "GLU", "GLU", "LYS", "ASP"]
        comp = residue_composition(residues)
        _record("residue_composition shape", comp.shape == (20,), f"got {comp.shape}")
        _record("residue_composition sum", float(comp.sum()) == 6.0,
                f"got {comp.sum()}")

        norm_comp = normalised_residue_composition(residues, method="l1")
        _record("normalised sum ≈ 1", abs(float(norm_comp.sum()) - 1.0) < 1e-5)
    except Exception as e:
        _record("box_residues functions", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════

def test_dimenet() -> None:
    """Test DimeNet++ architecture."""
    print("\n── 2a. DimeNet++ ──")
    try:
        from Graph_model.model.dimenet import DimeNet, DimeNetConfig
        _record("Import DimeNet", True)
    except Exception as e:
        _record("Import DimeNet", False, str(e))
        return

    try:
        cfg = DimeNetConfig(hidden_dim=64, n_layers=2, n_rbf=8, n_bilinear=4)
        model = DimeNet(cfg)
        _record("DimeNet instantiation", True)

        n_params = sum(p.numel() for p in model.parameters())
        _record(f"DimeNet params ({n_params:,})", n_params > 0)

        batch = _make_dummy_heterodata()
        out = model(batch)
        _record("DimeNet forward shape", out.shape == (2, 1), f"got {out.shape}")
        _record("DimeNet output finite", torch.isfinite(out).all().item())
    except Exception as e:
        _record("DimeNet forward", False, str(e))


def test_egnn() -> None:
    """Test EGNN architecture."""
    print("\n── 2b. EGNN ──")
    try:
        from Graph_model.model.egnn import EGNN, EGNNConfig
        _record("Import EGNN", True)
    except Exception as e:
        _record("Import EGNN", False, str(e))
        return

    try:
        cfg = EGNNConfig(hidden_dim=64, n_layers=2)
        model = EGNN(cfg)
        _record("EGNN instantiation", True)

        batch = _make_dummy_heterodata()
        out = model(batch)
        _record("EGNN forward shape", out.shape == (2, 1), f"got {out.shape}")
        _record("EGNN output finite", torch.isfinite(out).all().item())
    except Exception as e:
        _record("EGNN forward", False, str(e))


def test_ggnn_sequential() -> None:
    """Test GGNN Sequential Binding."""
    print("\n── 2c. GGNN Sequential ──")
    try:
        from Graph_model.model.ggnn_sequential import GGNNSequential, GGNNSeqConfig
        _record("Import GGNNSequential", True)
    except Exception as e:
        _record("Import GGNNSequential", False, str(e))
        return

    try:
        cfg = GGNNSeqConfig(hidden_dim=64, n_steps=2, n_states=3)
        model = GGNNSequential(cfg)
        _record("GGNNSeq instantiation", True)

        batch = _make_dummy_heterodata()
        out = model(batch)
        _record("GGNNSeq forward shape", out.shape == (2, 1), f"got {out.shape}")
        _record("GGNNSeq output finite", torch.isfinite(out).all().item())
    except Exception as e:
        _record("GGNNSeq forward", False, str(e))


def test_graphormer() -> None:
    """Test Graphormer architecture."""
    print("\n── 2d. Graphormer ──")
    try:
        from Graph_model.model.graphormer import Graphormer, GraphormerConfig
        _record("Import Graphormer", True)
    except Exception as e:
        _record("Import Graphormer", False, str(e))
        return

    try:
        cfg = GraphormerConfig(hidden_dim=64, n_layers=2, n_heads=4, ffn_dim=128)
        model = Graphormer(cfg)
        _record("Graphormer instantiation", True)

        batch = _make_dummy_heterodata()
        out = model(batch)
        _record("Graphormer forward shape", out.shape == (2, 1), f"got {out.shape}")
        _record("Graphormer output finite", torch.isfinite(out).all().item())
    except Exception as e:
        _record("Graphormer forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_losses() -> None:
    """Test advanced loss functions."""
    print("\n── 3. Loss Functions ──")
    try:
        from Graph_model.model.losses import (
            ListMLELoss, MonotonicityLoss, PairwiseRankLoss, CombinedDockingLoss,
        )
        _record("Import losses", True)
    except Exception as e:
        _record("Import losses", False, str(e))
        return

    B = 8
    preds = torch.randn(B, 1)
    targets = torch.randn(B, 1)

    # ListMLE
    try:
        lm = ListMLELoss()
        loss_lm = lm(preds, targets)
        _record("ListMLELoss scalar", loss_lm.dim() == 0)
        _record("ListMLELoss finite", torch.isfinite(loss_lm).item())
    except Exception as e:
        _record("ListMLELoss", False, str(e))

    # Monotonicity
    try:
        ml = MonotonicityLoss(margin=0.1)
        galloyl = torch.tensor([0, 1, 0, 2, 1, 5, 0, 1], dtype=torch.long)
        loss_ml = ml(preds, galloyl)
        _record("MonotonicityLoss scalar", loss_ml.dim() == 0)
        _record("MonotonicityLoss ≥ 0", loss_ml.item() >= 0)
    except Exception as e:
        _record("MonotonicityLoss", False, str(e))

    # Pairwise
    try:
        pr = PairwiseRankLoss()
        loss_pr = pr(preds, targets)
        _record("PairwiseRankLoss scalar", loss_pr.dim() == 0)
    except Exception as e:
        _record("PairwiseRankLoss", False, str(e))

    # Combined
    try:
        cd = CombinedDockingLoss(alpha=1.0, beta=0.1, gamma=0.05)
        loss_dict = cd(preds, targets, galloyl_units=galloyl)
        _record("CombinedDockingLoss returns dict", isinstance(loss_dict, dict))
        _record("CombinedDockingLoss has 'total'", "total" in loss_dict)
        _record("CombinedDockingLoss total finite",
                torch.isfinite(loss_dict["total"]).item())
    except Exception as e:
        _record("CombinedDockingLoss", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING MODULES
# ═══════════════════════════════════════════════════════════════════════════════

def test_maml() -> None:
    """Test MAML meta-learning."""
    print("\n── 4a. MAML ──")
    try:
        from Graph_model.train.maml import MAMLTrainer, MAMLConfig
        _record("Import MAML", True)
    except Exception as e:
        _record("Import MAML", False, str(e))
        return

    try:
        cfg = MAMLConfig(inner_lr=0.01, n_inner_steps=2, n_tasks_per_epoch=2, n_meta_epochs=1)
        _record("MAMLConfig creation", True)
        _record("MAMLConfig fields", cfg.inner_lr == 0.01 and cfg.first_order is True)
    except Exception as e:
        _record("MAMLConfig", False, str(e))


def test_contrastive() -> None:
    """Test contrastive pre-training."""
    print("\n── 4b. Contrastive ──")
    try:
        from Graph_model.train.contrastive import (
            ContrastivePretrainer, ContrastiveConfig, NTXentLoss,
        )
        _record("Import contrastive", True)
    except Exception as e:
        _record("Import contrastive", False, str(e))
        return

    try:
        # Test NT-Xent loss directly
        loss_fn = NTXentLoss(temperature=0.1)
        z1 = torch.randn(4, 64)
        z2 = torch.randn(4, 64)
        loss = loss_fn(z1, z2)
        _record("NTXentLoss scalar", loss.dim() == 0)
        _record("NTXentLoss finite", torch.isfinite(loss).item())
        _record("NTXentLoss > 0", loss.item() > 0)
    except Exception as e:
        _record("NTXentLoss", False, str(e))


def test_hpo() -> None:
    """Test Optuna HPO config."""
    print("\n── 4c. Optuna HPO ──")
    try:
        from Graph_model.train.hpo import OptunaTuner, HPOConfig
        _record("Import hpo", True)
    except Exception as e:
        _record("Import hpo", False, str(e))
        return

    try:
        cfg = HPOConfig(n_trials=5, n_cv_folds=2, max_epochs_per_trial=3)
        _record("HPOConfig creation", True)
        _record("HPOConfig fields", cfg.n_trials == 5 and cfg.n_cv_folds == 2)
    except Exception as e:
        _record("HPOConfig", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_stratified_eval() -> None:
    """Test stratified LOLO-CV evaluation."""
    print("\n── 5a. Stratified Evaluator ──")
    try:
        from Graph_model.train.stratified_eval import (
            StratifiedEvaluator, StratifiedReport, StrataMetrics,
        )
        _record("Import stratified_eval", True)
    except Exception as e:
        _record("Import stratified_eval", False, str(e))
        return

    try:
        evaluator = StratifiedEvaluator()
        preds = np.random.randn(20)
        targets = np.random.randn(20)
        metadata = [
            {"box_type": "GLU_cluster", "receptor": "collagen",
             "ligand_group": "primary", "galloyl_units": 1}
        ] * 10 + [
            {"box_type": "LYS_cluster", "receptor": "mmp1",
             "ligand_group": "GA_analogue", "galloyl_units": 2}
        ] * 10

        report = evaluator.evaluate_fold(
            test_indices=list(range(20)),
            predictions=preds,
            targets=targets,
            fold_idx=0,
            held_out="gallic_acid",
            metadata=metadata,
        )
        _record("StratifiedReport type", isinstance(report, StratifiedReport))
        _record("StratifiedReport has box_type", len(report.by_box_type) == 2)
        _record("StratifiedReport has receptor", len(report.by_receptor) == 2)
        _record("StratifiedReport as_dict", "fold" in report.as_dict())

        # Aggregate
        agg = evaluator.aggregate_reports([report, report])
        _record("aggregate_reports has overall", "overall" in agg)
    except Exception as e:
        _record("StratifiedEvaluator", False, str(e))


def test_scaffold_split() -> None:
    """Test scaffold-based splitting."""
    print("\n── 5b. Scaffold Splitter ──")
    try:
        from Graph_model.train.scaffold_split import ScaffoldSplitter, ScaffoldFold
        _record("Import scaffold_split", True)
    except Exception as e:
        _record("Import scaffold_split", False, str(e))
        return

    try:
        splitter = ScaffoldSplitter(seed=42)
        # Test scaffold grouping
        ligand_smiles = {
            "gallic_acid": "OC(=O)c1cc(O)c(O)c(O)c1",
            "pyrogallol": "Oc1cccc(O)c1O",
            "EDC": "CCN=C=NCCCN(C)C",
        }
        groups = splitter.get_scaffold_groups(ligand_smiles)
        _record("get_scaffold_groups", len(groups) >= 1, f"got {len(groups)} groups")

        # Verify all ligands accounted for
        all_ligands = []
        for ligs in groups.values():
            all_ligands.extend(ligs)
        _record("All ligands in groups", set(all_ligands) == set(ligand_smiles.keys()))
    except Exception as e:
        _record("ScaffoldSplitter", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTERPRETABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def test_gradcam() -> None:
    """Test Graph Grad-CAM."""
    print("\n── 6a. Graph Grad-CAM ──")
    try:
        from Graph_model.interpret.gradcam import graph_gradcam, gradcam_vs_plip
        _record("Import gradcam", True)
    except Exception as e:
        _record("Import gradcam", False, str(e))
        return

    try:
        from Graph_model.model.option_a import OptionA
        from Graph_model.model.config import OptionAConfig

        model = OptionA(OptionAConfig(hidden_dim=64, n_layers=2, gat_heads=2, gat_head_dim=32))
        batch = _make_dummy_heterodata(n_atoms=8, n_bonds=16, batch_size=1)

        result = graph_gradcam(model, batch)
        _record("gradcam returns dict", isinstance(result, dict))
        _record("gradcam has atom_cam", "atom_cam" in result)
        _record("gradcam atom_cam shape", result["atom_cam"].shape[0] == 8,
                f"got {result['atom_cam'].shape}")
        _record("gradcam has layer_name", "layer_name" in result)

        # PLIP comparison
        plip_atoms = [0, 2, 5]
        plip_result = gradcam_vs_plip(result["atom_cam"], plip_atoms, top_k=3)
        _record("gradcam_vs_plip returns dict", "precision_at_k" in plip_result)
        _record("gradcam_vs_plip enrichment", "enrichment_ratio" in plip_result)
    except Exception as e:
        _record("gradcam functions", False, str(e))


def test_probing() -> None:
    """Test probing classifiers."""
    print("\n── 6b. Probing Classifiers ──")
    try:
        from Graph_model.interpret.probing import ProbingClassifier
        _record("Import probing", True)
    except Exception as e:
        _record("Import probing", False, str(e))
        return

    try:
        # Test probe training directly
        probe = ProbingClassifier(embed_dim=64, n_classes=3, probe_type="linear")
        embeddings = torch.randn(50, 64)
        labels = torch.randint(0, 3, (50,))

        results = probe.train_and_evaluate(embeddings, labels, n_epochs=20)
        _record("probe train_acc exists", "train_acc" in results)
        _record("probe test_acc exists", "test_acc" in results)
        _record("probe train_acc > 0", results["train_acc"] > 0)
        _record("probe loss_history", len(results["loss_history"]) == 20)

        # MLP probe
        probe_mlp = ProbingClassifier(embed_dim=64, n_classes=3, probe_type="mlp")
        results_mlp = probe_mlp.train_and_evaluate(embeddings, labels, n_epochs=10)
        _record("MLP probe works", results_mlp["test_acc"] >= 0)
    except Exception as e:
        _record("ProbingClassifier", False, str(e))


def test_attention_rollout() -> None:
    """Test attention rollout."""
    print("\n── 6c. Attention Rollout ──")
    try:
        from Graph_model.interpret.attention_rollout import (
            attention_rollout, atom_importance_from_rollout,
        )
        _record("Import attention_rollout", True)
    except Exception as e:
        _record("Import attention_rollout", False, str(e))
        return

    try:
        # Create dummy attention matrices (3 layers, 10 tokens)
        attn_mats = [torch.softmax(torch.randn(10, 10), dim=-1) for _ in range(3)]

        rollout = attention_rollout(attn_mats, add_residual=True)
        _record("rollout shape", rollout.shape == (10, 10), f"got {rollout.shape}")
        _record("rollout rows sum ≈ 1",
                (rollout.sum(dim=-1) - 1.0).abs().max().item() < 0.1)

        # With multi-head attention
        attn_mats_mh = [torch.softmax(torch.randn(4, 10, 10), dim=-1) for _ in range(3)]
        rollout_mh = attention_rollout(attn_mats_mh, head_reduction="mean")
        _record("multi-head rollout shape", rollout_mh.shape == (10, 10))

        # Atom importance
        importance = atom_importance_from_rollout(rollout, cls_index=0)
        _record("atom_importance shape", importance.shape[0] == 9,
                f"got {importance.shape}")
    except Exception as e:
        _record("attention_rollout functions", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REGISTRY & INIT IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_model_registry() -> None:
    """Test that MODEL_REGISTRY includes new architectures."""
    print("\n── 7a. MODEL_REGISTRY ──")
    try:
        from Graph_model.train.run_training import MODEL_REGISTRY
        _record("Import MODEL_REGISTRY", True)

        expected_keys = {"A", "B", "C", "D", "E", "F", "G", "H", "I"}
        actual_keys = set(MODEL_REGISTRY.keys())
        _record("Registry has 9 models", actual_keys == expected_keys,
                f"got {actual_keys}")

        for key in ["F", "G", "H", "I"]:
            entry = MODEL_REGISTRY.get(key)
            _record(f"Registry['{key}'] has cls", entry is not None and "cls" in entry)
    except Exception as e:
        _record("MODEL_REGISTRY", False, str(e))


def test_init_imports() -> None:
    """Test that all __init__.py exports work."""
    print("\n── 7b. Init Imports ──")

    # model/__init__.py
    try:
        from Graph_model.model import (
            DimeNet, EGNN, GGNNSequential, Graphormer,
            DimeNetConfig, EGNNConfig, GGNNSeqConfig, GraphormerConfig,
            ListMLELoss, MonotonicityLoss, CombinedDockingLoss,
        )
        _record("model.__init__ new exports", True)
    except Exception as e:
        _record("model.__init__ new exports", False, str(e))

    # train/__init__.py
    try:
        from Graph_model.train import (
            MAMLTrainer, MAMLConfig, maml_train,
            ContrastivePretrainer, ContrastiveConfig, contrastive_pretrain,
            OptunaTuner, HPOConfig, run_hpo,
            StratifiedEvaluator, StratifiedReport,
            ScaffoldSplitter, ScaffoldFold,
        )
        _record("train.__init__ new exports", True)
    except Exception as e:
        _record("train.__init__ new exports", False, str(e))

    # interpret/__init__.py
    try:
        from Graph_model.interpret import (
            graph_gradcam, gradcam_vs_plip, batch_gradcam,
            ProbingClassifier, extract_embeddings, run_all_probes,
            attention_rollout, extract_attention_from_model,
            atom_importance_from_rollout, compare_attribution_methods,
        )
        _record("interpret.__init__ new exports", True)
    except Exception as e:
        _record("interpret.__init__ new exports", False, str(e))

    # data/features/__init__.py
    try:
        from Graph_model.data.features import (
            generate_conformer, conformer_node_features,
            CONFORMER_NODE_DIM, CONFORMER_EDGE_DIM,
            ecfp_node_features, ecfp_mol_features,
            ECFP_NODE_DIM, ECFP_MOL_DIM,
            residue_composition, BOX_RESIDUE_DIM, AMINO_ACIDS_20,
        )
        _record("data.features.__init__ new exports", True)
    except Exception as e:
        _record("data.features.__init__ new exports", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. BUILD ALL NEW MODELS FROM REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_all_new_models() -> None:
    """Test building and running all new models via registry."""
    print("\n── 8. Build & Forward All New Models ──")
    try:
        from Graph_model.train.run_training import MODEL_REGISTRY, _build_model, _forward_any
        _record("Import build helpers", True)
    except Exception as e:
        _record("Import build helpers", False, str(e))
        return

    batch = _make_dummy_heterodata()
    device = torch.device("cpu")

    for key in ["F", "G", "H", "I"]:
        try:
            # Use smaller configs for speed
            overrides = {"hidden_dim": 32, "n_layers": 2}
            if key == "I":
                overrides.update({"n_heads": 2, "ffn_dim": 64})
            model = _build_model(key, overrides)
            model.eval()

            pred, aux = _forward_any(model, batch, device)
            _record(f"Model['{key}'] forward → [{pred.shape}]",
                    pred.shape == (2, 1), f"got {pred.shape}")
        except Exception as e:
            _record(f"Model['{key}'] build+forward", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 70)
    print("  Phase 8 Improvement Smoke Tests")
    print("=" * 70)

    test_conformer_3d()
    test_ecfp()
    test_box_residues()
    test_dimenet()
    test_egnn()
    test_ggnn_sequential()
    test_graphormer()
    test_losses()
    test_maml()
    test_contrastive()
    test_hpo()
    test_stratified_eval()
    test_scaffold_split()
    test_gradcam()
    test_probing()
    test_attention_rollout()
    test_model_registry()
    test_init_imports()
    test_build_all_new_models()

    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} total")
    print("=" * 70)

    if FAIL > 0:
        print("\nFailed tests:")
        for name, passed, detail in RESULTS:
            if not passed:
                print(f"  ❌ {name}: {detail}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
