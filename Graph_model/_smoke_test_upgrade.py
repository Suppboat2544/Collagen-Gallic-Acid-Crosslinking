"""
_smoke_test_upgrade.py — Upgrade Smoke Tests
==============================================
Tests for:
  1. Option E (pGET) — novel model forward pass + aux outputs
  2. Option E — with denoising off
  3. Option E — FiLM parameter generation
  4. Option E — Hierarchical Gated Pooling shapes
  5. Option E — GECT layer standalone
  6. OptionC detach bug fix — gradients flow through fragment sum
  7. Training pipeline — train_single_model with synthetic data
  8. Training pipeline — JSON epoch log produced
  9. Visualization — individual plot generation from JSON
  10. Visualization — comparison plots from multiple JSONs
  11. EpochLogger — incremental JSON writing
  12. Model registry — all 5 models listed
  13. Option E config — defaults are reasonable
  14. Forward wrapper — handles all return types

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_upgrade.py
"""

from __future__ import annotations
import sys, math, warnings, json, tempfile, shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.data import HeteroData, Batch

warnings.filterwarnings('ignore')

# ── Test harness ──────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

def ok(n):
    global PASS; PASS += 1; print(f"  [PASS] {n}")

def fail(n, r=""):
    global FAIL; FAIL += 1; print(f"  [FAIL] {n}" + (f" — {r}" if r else ""))

def check(n, c, r=""):
    ok(n) if c else fail(n, r)

# ── Synthetic HeteroData builder ──────────────────────────────────────────────
def _make_batch(B=2, n_lig=6, n_res=8):
    items = []
    for g in range(B):
        d = HeteroData()
        d['ligand'].x = torch.randn(n_lig, 35)
        d['ligand', 'bond', 'ligand'].edge_index = torch.randint(0, n_lig, (2, 10))
        d['ligand', 'bond', 'ligand'].edge_attr = torch.randn(10, 13)
        d['residue'].x = torch.randn(n_res, 30)
        d['residue', 'contact', 'residue'].edge_index = torch.randint(0, n_res, (2, 12))
        d['residue', 'contact', 'residue'].edge_attr = torch.randn(12, 4)
        n_bp = n_lig * n_res
        d['ligand', 'interacts', 'residue'].edge_index = torch.stack([
            torch.randint(0, n_lig, (n_bp,)),
            torch.randint(0, n_res, (n_bp,)),
        ])
        d['ligand', 'interacts', 'residue'].edge_attr = torch.randn(n_bp, 8)
        d.y = torch.tensor([[-5.0 + g * 0.5]])
        d.ph_enc = torch.tensor([0.5])
        d.temp_enc = torch.tensor([0.5])
        d.box_idx = torch.tensor([0])
        d.receptor_flag = torch.tensor([0.0])
        d.tier = 0
        d.ligand_name = f"lig_{g}"
        items.append(d)
    return Batch.from_data_list(items)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Option E (pGET) Forward Pass
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Option E (pGET) — Forward Pass")
from Graph_model.model.option_e import OptionE, OptionEConfig

try:
    cfg = OptionEConfig()
    model = OptionE(cfg)
    batch = _make_batch(B=3)

    model.train()
    out, aux = model(batch)
    check("returns tensor + dict", isinstance(out, torch.Tensor) and isinstance(aux, dict))
    check("output shape [B,1]", out.shape == (3, 1))
    check("output is finite", torch.isfinite(out).all().item())
    check("has denoise_loss (train)", 'denoise_loss' in aux)
    check("denoise_loss scalar", aux['denoise_loss'].dim() == 0)
    check("has cluster_assignments", 'cluster_assignments' in aux)
    check("cluster shape [N, K]", aux['cluster_assignments'].shape == (18, cfg.n_soft_clusters),
          f"got {aux['cluster_assignments'].shape}")
except Exception as e:
    for t in ["returns tensor + dict", "output shape [B,1]", "output is finite",
              "has denoise_loss (train)", "denoise_loss scalar",
              "has cluster_assignments", "cluster shape [N, K]"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — Option E with Denoising Off
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Option E — Denoising Off")
try:
    cfg_no_dn = OptionEConfig(denoise_weight=0.0)
    model_no_dn = OptionE(cfg_no_dn)
    model_no_dn.train()
    batch = _make_batch(B=2)
    out, aux = model_no_dn(batch)
    check("no denoise head created", model_no_dn.denoise_head is None)
    check("no denoise_loss in aux", 'denoise_loss' not in aux)
    check("still produces valid output", out.shape == (2, 1) and torch.isfinite(out).all().item())
except Exception as e:
    for t in ["no denoise head created", "no denoise_loss in aux", "still produces valid output"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — Option E Eval Mode
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Option E — Eval Mode (no denoising)")
try:
    cfg = OptionEConfig()
    model = OptionE(cfg)
    model.eval()
    batch = _make_batch(B=2)
    with torch.no_grad():
        out, aux = model(batch)
    check("eval output shape [B,1]", out.shape == (2, 1))
    check("no denoise_loss in eval", 'denoise_loss' not in aux)
    check("cluster assignments present", 'cluster_assignments' in aux)
except Exception as e:
    for t in ["eval output shape [B,1]", "no denoise_loss in eval", "cluster assignments present"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — FiLM Generator
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] FiLM Generator — shapes & initialisation")
from Graph_model.model.option_e import _FiLMGenerator
try:
    film = _FiLMGenerator(cond_dim=32, hidden=64, n_layers=4, out_dim=128)
    cond = torch.randn(3, 32)
    gamma, beta = film(cond)
    check("gamma shape [B, L, H]", gamma.shape == (3, 4, 128))
    check("beta shape [B, L, H]", beta.shape == (3, 4, 128))
    check("gamma init ≈ 1", (gamma.mean() - 1.0).abs() < 0.5,
          f"mean={gamma.mean():.3f}")
    check("beta init ≈ 0", beta.mean().abs() < 0.5,
          f"mean={beta.mean():.3f}")
except Exception as e:
    for t in ["gamma shape [B, L, H]", "beta shape [B, L, H]",
              "gamma init ≈ 1", "beta init ≈ 0"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — GECT Layer standalone
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] GECT Layer — standalone forward")
from Graph_model.model.option_e import _GECTLayer
try:
    layer = _GECTLayer(hidden=128, heads=4, head_dim=32, edge_dim=13,
                        edge_gate_dim=32, dropout=0.1)
    x = torch.randn(10, 128)
    ei = torch.randint(0, 10, (2, 20))
    ea = torch.randn(20, 13)
    gamma = torch.ones(10, 128)
    beta = torch.zeros(10, 128)
    out = layer(x, ei, ea, gamma, beta)
    check("GECT output shape", out.shape == (10, 128))
    check("GECT output finite", torch.isfinite(out).all().item())
    # Check skip connection is effective (output ≠ 0)
    check("GECT non-trivial output", out.abs().mean() > 0.01, f"mean={out.abs().mean():.5f}")
except Exception as e:
    for t in ["GECT output shape", "GECT output finite", "GECT non-trivial output"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — Hierarchical Gated Pooling
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Hierarchical Gated Pooling — shapes")
from Graph_model.model.option_e import _HierarchicalGatedPool
try:
    pool = _HierarchicalGatedPool(
        hidden=128, cond_proj_dim=32, n_clusters=8, pool_heads=4, dropout=0.1)
    h_atom = torch.randn(15, 128)
    batch_idx = torch.tensor([0]*5 + [1]*5 + [2]*5)
    c_atom = torch.randn(15, 32)
    h_graph = pool(h_atom, batch_idx, c_atom)
    check("pool output shape [B, H]", h_graph.shape == (3, 128))
    check("pool output finite", torch.isfinite(h_graph).all().item())
except Exception as e:
    for t in ["pool output shape [B, H]", "pool output finite"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — Option E Gradient Flow
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Option E — Gradient flow (backprop)")
try:
    cfg = OptionEConfig()
    model = OptionE(cfg)
    model.train()
    batch = _make_batch(B=2)
    out, aux = model(batch)
    loss = out.sum()
    if 'denoise_loss' in aux:
        loss = loss + aux['denoise_loss']
    loss.backward()

    # Check FiLM has gradients
    film_grad = model.film.net[-1].weight.grad
    check("FiLM gradients exist", film_grad is not None)
    check("FiLM gradients nonzero", film_grad is not None and film_grad.abs().sum() > 0)

    # Check GECT layers have gradients
    gect_grad = model.gect_layers[0].q_proj.weight.grad
    check("GECT grad exists", gect_grad is not None)

    # Check pooling has gradients
    pool_grad = model.pool.gate[0].weight.grad
    check("Pool gate grad exists", pool_grad is not None)
except Exception as e:
    for t in ["FiLM gradients exist", "FiLM gradients nonzero",
              "GECT grad exists", "Pool gate grad exists"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 8 — OptionC Bug Fix (detach removed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] OptionC Bug Fix — fragment gradients flow")
from Graph_model.model import OptionC, OptionCConfig
try:
    cfg_c = OptionCConfig()
    model_c = OptionC(cfg_c)
    model_c.train()
    batch_c = _make_batch(B=2)
    out_c, frag_list = model_c(batch_c)

    # Check that frag_contribs are NOT detached
    has_grad_fn = all(f.requires_grad or f.grad_fn is not None for f in frag_list)
    check("frag_contribs have grad_fn (not detached)", has_grad_fn,
          f"grad_fns: {[f.grad_fn for f in frag_list]}")

    loss_c = out_c.sum()
    loss_c.backward()
    frag_head_grad = model_c.frag_out[0].weight.grad
    check("frag_out receives gradients", frag_head_grad is not None and frag_head_grad.abs().sum() > 0)
except Exception as e:
    for t in ["frag_contribs have grad_fn (not detached)", "frag_out receives gradients"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 9 — OptionE Config defaults
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9] OptionE Config — defaults")
try:
    cfg = OptionEConfig()
    check("n_gect_layers default 4", cfg.n_gect_layers == 4)
    check("n_heads default 4", cfg.n_heads == 4)
    check("head_dim default 32", cfg.head_dim == 32)
    check("n_soft_clusters default 8", cfg.n_soft_clusters == 8)
    check("denoise_weight default 0.1", cfg.denoise_weight == 0.1)
    check("pool_heads default 4", cfg.pool_heads == 4)
    # Base config inherited
    check("ligand_node_dim = 35", cfg.ligand_node_dim == 35)
    check("ligand_edge_dim = 13", cfg.ligand_edge_dim == 13)
    check("cond_dim = 19", cfg.cond_dim == 19)
except Exception as e:
    for t in ["n_gect_layers default 4", "n_heads default 4", "head_dim default 32",
              "n_soft_clusters default 8", "denoise_weight default 0.1",
              "pool_heads default 4", "ligand_node_dim = 35",
              "ligand_edge_dim = 13", "cond_dim = 19"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 10 — Model Registry
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10] Model Registry — all 5 models")
from Graph_model.train.run_training import MODEL_REGISTRY
try:
    check("5 models registered", len(MODEL_REGISTRY) == 5)
    check("A in registry", 'A' in MODEL_REGISTRY)
    check("B in registry", 'B' in MODEL_REGISTRY)
    check("C in registry", 'C' in MODEL_REGISTRY)
    check("D in registry", 'D' in MODEL_REGISTRY)
    check("E in registry", 'E' in MODEL_REGISTRY)
    check("E is pGET", 'pGET' in MODEL_REGISTRY['E']['name'])
except Exception as e:
    for t in ["5 models registered", "A in registry", "B in registry",
              "C in registry", "D in registry", "E in registry", "E is pGET"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 11 — EpochLogger
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11] EpochLogger — JSON writing")
from Graph_model.train.run_training import EpochLogger
try:
    tmp = Path(tempfile.mkdtemp())
    log_file = tmp / "test_log.json"
    logger = EpochLogger(log_file, "TestModel", meta={"test": True})
    check("log file created", log_file.exists())

    logger.log_epoch(0, {"train_loss": 0.5, "val_loss": 0.4})
    logger.log_epoch(1, {"train_loss": 0.3, "val_loss": 0.25})

    with open(log_file) as f:
        data = json.load(f)
    check("2 epochs logged", len(data['epochs']) == 2)
    check("model_name correct", data['model_name'] == 'TestModel')
    check("meta preserved", data['meta']['test'] == True)

    logger.finalise({"best_val_rmse": 0.5})
    with open(log_file) as f:
        data = json.load(f)
    check("summary added", 'summary' in data)
    check("finished_at added", 'finished_at' in data)

    shutil.rmtree(tmp)
except Exception as e:
    for t in ["log file created", "2 epochs logged", "model_name correct",
              "meta preserved", "summary added", "finished_at added"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 12 — Forward Wrapper
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12] Forward wrapper — _forward_any")
from Graph_model.train.run_training import _forward_any
from Graph_model.model import OptionA, OptionAConfig
try:
    model_a = OptionA(OptionAConfig())
    batch = _make_batch(B=2)
    pred, aux = _forward_any(model_a, batch, torch.device('cpu'))
    check("OptionA via _forward_any shape", pred.shape == (2, 1))
    check("OptionA aux is dict", isinstance(aux, dict))

    model_e = OptionE(OptionEConfig())
    pred_e, aux_e = _forward_any(model_e, batch, torch.device('cpu'))
    check("OptionE via _forward_any shape", pred_e.shape == (2, 1))
    check("OptionE has cluster_assignments", 'cluster_assignments' in aux_e)
except Exception as e:
    for t in ["OptionA via _forward_any shape", "OptionA aux is dict",
              "OptionE via _forward_any shape", "OptionE has cluster_assignments"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 13 — Visualization imports
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13] Visualization — imports")
try:
    from Graph_model.viz import (
        plot_training_loss, plot_validation_loss,
        plot_rmse_per_epoch, plot_mae_per_epoch,
        plot_pearson_per_epoch, plot_spearman_per_epoch,
        plot_learning_rate, plot_pred_vs_true, plot_residuals,
        plot_all_individual,
        plot_rmse_comparison_bar, plot_mae_comparison_bar,
        plot_pearson_comparison_bar, plot_loss_overlay,
        plot_rmse_overlay, plot_wall_time_comparison,
        plot_all_comparisons,
    )
    check("all viz functions importable", True)
except Exception as e:
    fail("all viz functions importable", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 14 — Visualization from synthetic JSON
# ══════════════════════════════════════════════════════════════════════════════
print("\n[14] Visualization — generate plots from synthetic JSON")
try:
    import matplotlib
    matplotlib.use('Agg')

    tmp = Path(tempfile.mkdtemp())
    # Create a synthetic JSON log
    synthetic_log = {
        "model_name": "Test Model",
        "epochs": [
            {
                "epoch": i,
                "train_loss": 1.0 / (i + 1),
                "val_loss": 1.2 / (i + 1),
                "train_rmse": (1.0 / (i + 1)) ** 0.5,
                "val_rmse": (1.2 / (i + 1)) ** 0.5,
                "train_mae": 0.8 / (i + 1),
                "val_mae": 0.9 / (i + 1),
                "train_pearson_r": min(0.99, 0.2 + i * 0.05),
                "val_pearson_r": min(0.95, 0.15 + i * 0.05),
                "train_spearman_r": min(0.98, 0.18 + i * 0.05),
                "val_spearman_r": min(0.93, 0.13 + i * 0.05),
                "learning_rate": 1e-3 * (0.95 ** i),
                "wall_time_s": i * 2.5,
            }
            for i in range(20)
        ],
        "summary": {"best_epoch": 15, "best_val_rmse": 0.25},
    }
    json_path = tmp / "synthetic_log.json"
    with open(json_path, 'w') as f:
        json.dump(synthetic_log, f)

    fig_dir = tmp / "figs"
    from Graph_model.viz.plot_results import plot_all_individual
    paths = plot_all_individual(json_path, out_dir=fig_dir)
    check("7 individual plots created", len(paths) == 7, f"got {len(paths)}")
    check("all PNG files exist", all(p.exists() for p in paths))

    # Also test pred_vs_true and residuals
    from Graph_model.viz.plot_results import plot_pred_vs_true, plot_residuals
    preds = [float(i) * 0.1 for i in range(20)]
    targets = [float(i) * 0.1 + 0.05 for i in range(20)]
    p1 = plot_pred_vs_true(preds, targets, "Test Model", fig_dir)
    p2 = plot_residuals(preds, targets, "Test Model", fig_dir)
    check("pred_vs_true PNG exists", p1.exists())
    check("residuals PNG exists", p2.exists())

    shutil.rmtree(tmp)
except Exception as e:
    for t in ["7 individual plots created", "all PNG files exist",
              "pred_vs_true PNG exists", "residuals PNG exists"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 15 — Comparison visualizations
# ══════════════════════════════════════════════════════════════════════════════
print("\n[15] Comparison visualizations — from synthetic JSONs")
try:
    import matplotlib
    matplotlib.use('Agg')

    tmp = Path(tempfile.mkdtemp())

    # Create multiple model training logs
    for key in ['a', 'b', 'c', 'd', 'e']:
        log = {
            "model_name": f"Option {key.upper()} — Test",
            "epochs": [
                {
                    "epoch": i,
                    "train_loss": (1.0 + ord(key) * 0.1) / (i + 1),
                    "val_loss": (1.2 + ord(key) * 0.1) / (i + 1),
                    "val_rmse": ((1.2 + ord(key) * 0.1) / (i + 1)) ** 0.5,
                    "val_mae": (0.9 + ord(key) * 0.05) / (i + 1),
                    "val_pearson_r": min(0.95, 0.1 + i * 0.05 + ord(key) * 0.01),
                    "learning_rate": 1e-3,
                    "wall_time_s": i * 2.0 + ord(key),
                }
                for i in range(15)
            ],
        }
        with open(tmp / f"option_{key}_training.json", 'w') as f:
            json.dump(log, f)

    # Also create comparison_summary.json
    comp = {
        "models": {
            k.upper(): {
                "name": f"Option {k.upper()} — Test",
                "best_epoch": 10,
                "best_val_rmse": 0.3 + ord(k) * 0.01,
                "total_epochs": 15,
                "wall_time_s": 30 + ord(k),
            }
            for k in ['a', 'b', 'c', 'd', 'e']
        }
    }
    with open(tmp / "comparison_summary.json", 'w') as f:
        json.dump(comp, f)

    fig_dir = tmp / "figs"
    from Graph_model.viz.compare_models import plot_all_comparisons
    paths = plot_all_comparisons(tmp, out_dir=fig_dir)
    check("7 comparison plots created", len(paths) == 7, f"got {len(paths)}")
    check("all comparison PNGs exist", all(p.exists() for p in paths))

    # Check specific plots
    existing = [p.name for p in paths]
    check("RMSE bar chart exists", "comparison_rmse_bar.png" in existing)
    check("loss overlay exists", "comparison_val_loss_overlay.png" in existing)
    check("RMSE overlay exists", "comparison_rmse_overlay.png" in existing)

    shutil.rmtree(tmp)
except Exception as e:
    for t in ["7 comparison plots created", "all comparison PNGs exist",
              "RMSE bar chart exists", "loss overlay exists", "RMSE overlay exists"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 16 — Option E parameter count
# ══════════════════════════════════════════════════════════════════════════════
print("\n[16] Option E — parameter count")
try:
    cfg = OptionEConfig()
    model = OptionE(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check("param count reasonable (100K-1M)", 100_000 < n_params < 1_000_000,
          f"got {n_params:,}")
    check("all params trainable", n_params == n_trainable)
    print(f"       Total parameters: {n_params:,}")
except Exception as e:
    for t in ["param count reasonable (100K-1M)", "all params trainable"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 17 — Option E with various batch sizes
# ══════════════════════════════════════════════════════════════════════════════
print("\n[17] Option E — batch size robustness")
try:
    cfg = OptionEConfig()
    model = OptionE(cfg)
    model.eval()
    for B in [1, 2, 4, 8]:
        with torch.no_grad():
            batch = _make_batch(B=B)
            out, _ = model(batch)
            ok(f"B={B}: shape={out.shape}")
except Exception as e:
    fail(f"batch size robustness", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 18 — Model __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
print("\n[18] Model __init__ exports OptionE + Config")
try:
    from Graph_model.model import OptionE as OE, OptionEConfig as OEC
    check("OptionE importable from model", OE is not None)
    check("OptionEConfig importable from model", OEC is not None)
    check("OptionE in __all__", 'OptionE' in __import__('Graph_model.model', fromlist=['__all__']).__all__)
    check("OptionEConfig in __all__", 'OptionEConfig' in __import__('Graph_model.model', fromlist=['__all__']).__all__)
except Exception as e:
    for t in ["OptionE importable from model", "OptionEConfig importable from model",
              "OptionE in __all__", "OptionEConfig in __all__"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = PASS + FAIL
print(f" UPGRADE SMOKE TESTS: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print(" ALL TESTS PASSED")
else:
    print(f" {FAIL} TESTS FAILED")
print("=" * 60)
sys.exit(FAIL)
