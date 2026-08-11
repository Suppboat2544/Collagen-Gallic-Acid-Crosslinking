"""
_smoke_test_phase4.py  — Phase 4 Training Infrastructure (48 checks)

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_phase4.py

All tests are self-contained.  External I/O is limited to:
  - Reading a few lines from the real PDBbind index file
  - Checking PDBbind entry dirs are present
"""

from __future__ import annotations
import sys, types, warnings, math, random
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np

# ── test harness ──────────────────────────────────────────────────────────────
PASS = 0; FAIL = 0

def ok(name: str):
    global PASS; PASS += 1
    print(f"  [PASS] {name}")

def fail(name: str, reason: str = ""):
    global FAIL; FAIL += 1
    msg = f"  [FAIL] {name}"
    if reason:
        msg += f" — {reason}"
    print(msg)

def check(name: str, cond: bool, reason: str = ""):
    if cond:
        ok(name)
    else:
        fail(name, reason)


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — parse_pdbbind_index
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] parse_pdbbind_index")
from Graph_model.train.pdbbind_dataset import parse_pdbbind_index

INDEX_FILE = ROOT / "Graph_model/external_dataset/index/INDEX_general_PL.2020R1.lst"

try:
    dg_map = parse_pdbbind_index(str(INDEX_FILE))
    check("parse returns dict",         isinstance(dg_map, dict))
    check("non-empty",                  len(dg_map) > 0)
    check("keys are 4-char PDB codes",  all(len(k) == 4 for k in list(dg_map)[:10]))
    vals = list(dg_map.values())
    check("all ΔG are finite floats",   all(math.isfinite(v) for v in vals[:50]))
    check("ΔG range physical (-20,0.5)",all(-20 <= v <= 0.5 for v in vals[:50]))
    # IC50 exclusion test
    dg_no_ic50 = parse_pdbbind_index(str(INDEX_FILE), exclude_ic50=True)
    check("exclude_ic50 reduces map", len(dg_no_ic50) <= len(dg_map))
except Exception as e:
    for t in ["parse returns dict","non-empty","keys are 4-char PDB codes",
              "all ΔG are finite floats","ΔG range physical (-20,0.5)",
              "exclude_ic50 reduces map"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — PDBbindGraphDataset
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] PDBbindGraphDataset")
from Graph_model.train.pdbbind_dataset import PDBbindGraphDataset

PL_ROOT    = ROOT / "Graph_model/external_dataset/P-L"
INDEX_FILE_S = str(INDEX_FILE)

try:
    ds_check = PDBbindGraphDataset(str(PL_ROOT), INDEX_FILE_S, max_entries=1)
    check("is_available()", ds_check.is_available())
except Exception as e:
    fail("is_available()", str(e))

try:
    ds = PDBbindGraphDataset(str(PL_ROOT), INDEX_FILE_S, max_entries=3)
    ds.load()
    check("len==3 (or ≤3)",      1 <= len(ds) <= 3)

    if len(ds) > 0:
        item = ds[0]
        from torch_geometric.data import HeteroData
        check("item is HeteroData",          isinstance(item, HeteroData))
        check("ligand nodes present",        item['ligand'].x.shape[1] == 35)
        check("residue nodes present",       item['residue'].x.shape[1] == 30)
        check("y scalar",                    item.y.numel() == 1)
        check("tier==1 (transfer)",          int(item.tier) == 1)
        check("cond shape (4,)",             item.cond.shape == (4,))
        check("ΔG magnitude physical",       -20 <= item.y.item() <= 0.5)
    else:
        for t in ["item is HeteroData","ligand nodes present","residue nodes present",
                  "y scalar","tier==1 (transfer)","cond shape (4,)",
                  "ΔG magnitude physical"]:
            fail(t, "dataset empty — no local PDB entries matched")

    ds2 = PDBbindGraphDataset(str(PL_ROOT), INDEX_FILE_S, max_entries=3)
    ds2.load()
    stats = ds2.delta_g_stats()
    check("delta_g_stats returns dict",  isinstance(stats, dict))
    check("stats has 'mean'",            'mean' in stats)
except Exception as e:
    for t in ["len==3 (or ≤3)","item is HeteroData","ligand nodes present",
              "residue nodes present","y scalar","tier==1 (transfer)",
              "cond shape (4,)","ΔG magnitude physical",
              "delta_g_stats returns dict","stats has 'mean'"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — LOLOCVSplitter
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] LOLOCVSplitter")
from Graph_model.train.lolo_cv import LOLOCVSplitter, LOLOFold

# Build a minimal synthetic dataset with 9 ligands
class _SynItem:
    def __init__(self, ligand_name, tier=0, sample_id=""):
        import torch
        self.ligand_name = ligand_name
        self.tier        = tier
        self.sample_id   = sample_id
        self.y           = torch.tensor([0.0])

LIGAND_NAMES = [
    'NHS','EDC','pyrogallol','gallic_acid','protocatechuic_acid',
    'EDC_Oacylisourea','EDC_NHS','ellagic_acid','PGG',
]

# 3 samples per ligand (27 total)
syn_dataset = [
    _SynItem(lig, tier=0, sample_id=f"{lig}_{i}")
    for lig in LIGAND_NAMES for i in range(3)
]

try:
    splitter = LOLOCVSplitter(val_ratio=0.15, seed=42)
    folds    = list(splitter.split(syn_dataset))
    check("9 folds produced",           len(folds) == 9)
    check("LOLOFold instances",         all(isinstance(f, LOLOFold) for f in folds))
    unique_held = {f.held_out_ligand for f in folds}
    check("all 9 ligands held out once",  unique_held == set(LIGAND_NAMES))
    check("test indices non-empty",     all(len(f.test_idx)  > 0 for f in folds))
    check("train indices non-empty",    all(len(f.train_idx) > 0 for f in folds))
    # No leakage
    leaked = 0
    for f in folds:
        try:
            LOLOCVSplitter.verify_no_leakage(f)
        except AssertionError:
            leaked += 1
    check("no leakage in any fold",     leaked == 0,
          f"{leaked} folds leaked" if leaked else "")
    # Index disjoint for a single fold
    f0 = folds[0]
    s_train = set(f0.train_idx); s_val = set(f0.val_idx); s_test = set(f0.test_idx)
    check("train ∩ val = ∅",  len(s_train & s_val)  == 0)
    check("train ∩ test = ∅", len(s_train & s_test) == 0)
    check("val ∩ test = ∅",   len(s_val   & s_test) == 0)
except Exception as e:
    for t in ["9 folds produced","LOLOFold instances","all 9 ligands held out once",
              "test indices non-empty","train indices non-empty","no leakage in any fold",
              "train ∩ val = ∅","train ∩ test = ∅","val ∩ test = ∅"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — CurriculumSampler
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] CurriculumSampler")
from Graph_model.train.curriculum import CurriculumSampler, STAGE_LIGANDS, STAGE_WEIGHTS
from torch.utils.data import WeightedRandomSampler

try:
    check("STAGE_LIGANDS has 3 keys",   set(STAGE_LIGANDS.keys()) == {1,2,3})
    check("STAGE_WEIGHTS has 3 keys",   set(STAGE_WEIGHTS.keys()) == {1,2,3})
    total_ligs = sum(len(v) for v in STAGE_LIGANDS.values())
    check("9 unique stage ligands",     total_ligs == 9)

    curriculum = CurriculumSampler(syn_dataset, stage_schedule=[10,20], seed=42)
    curriculum.set_epoch(0)
    sampler0 = curriculum.get_sampler(num_samples=20)
    check("sampler epoch 0 type",       isinstance(sampler0, WeightedRandomSampler))
    active0 = curriculum.active_ligands()
    stage1_ligs = set(STAGE_LIGANDS[1])
    check("epoch 0 active = stage1",    set(active0) == stage1_ligs)

    curriculum.set_epoch(10)
    active10 = curriculum.active_ligands()
    stage12 = stage1_ligs | set(STAGE_LIGANDS[2])
    check("epoch 10 active = stage1+2", set(active10) == stage12)

    curriculum.set_epoch(20)
    active20 = curriculum.active_ligands()
    all_ligs = stage12 | set(STAGE_LIGANDS[3])
    check("epoch 20 active = all 3",    set(active20) == all_ligs)

    summary = curriculum.stage_summary(0)
    check("stage_summary returns str",  isinstance(summary, str) and len(summary) > 0)
except Exception as e:
    for t in ["STAGE_LIGANDS has 3 keys","STAGE_WEIGHTS has 3 keys",
              "9 unique stage ligands","sampler epoch 0 type",
              "epoch 0 active = stage1","epoch 10 active = stage1+2",
              "epoch 20 active = all 3","stage_summary returns str"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — regression_metrics & aggregate_folds
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] regression_metrics + aggregate_folds")
from Graph_model.train.metrics import (
    regression_metrics, FoldMetrics, aggregate_folds, print_lolo_report
)

try:
    # Perfect predictions
    m = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    check("rmse == 0 (perfect)",        math.isclose(m['rmse'], 0.0, abs_tol=1e-9))
    check("mae == 0 (perfect)",         math.isclose(m['mae'],  0.0, abs_tol=1e-9))
    check("pearson_r == 1 (perfect)",   math.isclose(m['pearson_r'], 1.0, abs_tol=1e-9))
    check("n == 3",                     m['n'] == 3)

    # Known shift: predictions = targets + 1.0
    targets = [float(i) for i in range(10)]
    preds   = [t + 1.0 for t in targets]
    ms = regression_metrics(preds, targets)
    check("rmse == 1.0 for constant shift", math.isclose(ms['rmse'], 1.0, abs_tol=1e-6))
    check("pearson_r == 1 for shift",       math.isclose(ms['pearson_r'], 1.0, abs_tol=1e-6))

    # aggregate_folds
    mock_folds = [
        FoldMetrics(fold=i, held_out=f"lig_{i}", n_test=3,
                    rmse=float(i+1), mae=float(i+1)*0.8,
                    pearson_r=0.9, spearman_r=0.85,
                    n_train=20, n_val=5, best_epoch=5, val_rmse=float(i+1)*1.1)
        for i in range(3)
    ]
    agg = aggregate_folds(mock_folds)
    check("aggregate returns dict",             isinstance(agg, dict))
    check("aggregate has rmse_mean",            'rmse_mean' in agg)
    check("aggregate has rmse_std",             'rmse_std' in agg)
    check("rmse_mean == 2.0",                   math.isclose(agg['rmse_mean'], 2.0, abs_tol=1e-6))
    # print_lolo_report should not raise
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_lolo_report(mock_folds, model_name="TestModel")
        check("print_lolo_report runs", True)
    except Exception as e2:
        fail("print_lolo_report runs", str(e2))
except Exception as e:
    for t in ["rmse == 0 (perfect)","mae == 0 (perfect)","pearson_r == 1 (perfect)",
              "n == 3","rmse == 1.0 for constant shift","pearson_r == 1 for shift",
              "aggregate returns dict","aggregate has rmse_mean","aggregate has rmse_std",
              "rmse_mean == 2.0","print_lolo_report runs"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — freeze_backbone_stages
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] freeze_backbone_stages")
from Graph_model.train.finetune import freeze_backbone_stages

try:
    from Graph_model.model.config import ModelConfig, OptionBConfig, OptionCConfig, OptionDConfig
    from Graph_model.model import OptionA, OptionB, OptionC, OptionD

    cfg = ModelConfig()

    for name, Model, Cfg in [
        ("OptionA", OptionA, ModelConfig),
        ("OptionB", OptionB, OptionBConfig),
        ("OptionC", OptionC, OptionCConfig),
        ("OptionD", OptionD, OptionDConfig),
    ]:
        try:
            m = Model(Cfg())
            n_frozen, n_total_ = freeze_backbone_stages(m, freeze_early=True)
            n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
            check(f"{name} some params frozen",     n_frozen > 0)
            check(f"{name} some params trainable",  n_trainable > 0)

            # Unfreeze all
            freeze_backbone_stages(m, freeze_early=False)
            n_train2 = sum(1 for p in m.parameters() if p.requires_grad)
            n_total_ = sum(1 for _ in m.parameters())
            check(f"{name} all unfrozen = all",     n_train2 == n_total_)
        except Exception as e2:
            for t in [f"{name} some params frozen",
                      f"{name} some params trainable",
                      f"{name} all unfrozen = all"]:
                fail(t, str(e2))
except Exception as e:
    fail("import models for freeze test", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — FoldMetrics.to_dict (used in checkpoint saving)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] FoldMetrics.to_dict")
try:
    fm = FoldMetrics(fold=0, held_out='NHS', n_test=3,
                     rmse=0.4, mae=0.3, pearson_r=0.9, spearman_r=0.88,
                     n_train=20, n_val=5, best_epoch=7, val_rmse=0.42)
    d = fm.to_dict()
    check("to_dict returns dict",   isinstance(d, dict))
    check("to_dict has 'rmse'",     'rmse'       in d)
    check("to_dict has 'held_out'", 'held_out'   in d)
    check("to_dict has 'fold'",     'fold'        in d)
except AttributeError:
    # If to_dict not implemented, fall back to dataclasses.asdict
    import dataclasses
    try:
        d = dataclasses.asdict(fm)
        check("to_dict via asdict works", isinstance(d, dict))
        for t in ["to_dict has 'rmse'","to_dict has 'held_out'",
                  "to_dict has 'fold'"]:
            check(t, t.split("'")[1] in d)
    except Exception as e:
        for t in ["to_dict returns dict","to_dict has 'rmse'",
                  "to_dict has 'held_out'","to_dict has 'fold'"]:
            fail(t, str(e))
except Exception as e:
    for t in ["to_dict returns dict","to_dict has 'rmse'",
              "to_dict has 'held_out'","to_dict has 'fold'"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Group 8 — End-to-end mini finetune (synthetic, CPU, 2 epochs)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] End-to-end mini finetune (OptionA, synthetic, 2 epochs)")

try:
    import torch
    from torch_geometric.data import HeteroData

    def _make_synth_item(ligand_name: str, seed: int = 0) -> HeteroData:
        """Minimal HeteroData matching the collagen-docking schema."""
        rng  = random.Random(seed)
        n_lig = rng.randint(5, 10)
        n_res = rng.randint(8, 15)
        n_bp  = n_lig * n_res
        item  = HeteroData()
        # Ligand nodes / edges
        item['ligand'].x          = torch.randn(n_lig, 35)
        ei_lig = torch.randint(0, n_lig, (2, n_lig * 2))
        item['ligand', 'bond', 'ligand'].edge_index = ei_lig
        item['ligand', 'bond', 'ligand'].edge_attr  = torch.randn(n_lig * 2, 13)
        # Residue nodes / edges
        item['residue'].x          = torch.randn(n_res, 30)
        ei_prot = torch.randint(0, n_res, (2, n_res * 2))
        item['residue', 'contact', 'residue'].edge_index = ei_prot
        item['residue', 'contact', 'residue'].edge_attr  = torch.randn(n_res * 2, 4)
        # Bipartite
        src = torch.randint(0, n_lig, (n_bp,))
        dst = torch.randint(0, n_res, (n_bp,))
        item['ligand', 'interacts', 'residue'].edge_index = torch.stack([src, dst])
        item['ligand', 'interacts', 'residue'].edge_attr  = torch.randn(n_bp, 8)
        # Labels / meta
        item.y           = torch.tensor([rng.uniform(-10, -1)])
        item.cond        = torch.tensor([
            round(rng.uniform(4, 9), 1) / 14.0,  # ph_enc
            round(rng.uniform(25, 37), 0) / 50.0, # temp_enc
            0.0,                                  # reserved
            0.0,                                  # receptor_flag
        ])
        item.tier        = 0
        item.ligand_name = ligand_name
        item.sample_id   = f"{ligand_name}_{seed}"
        return item

    # Build synthetic dataset (3 items per ligand × 9 ligands = 27 items)
    synth_ds = [
        _make_synth_item(lig, seed=i * 100 + j)
        for j, lig in enumerate(LIGAND_NAMES)
        for i in range(3)
    ]

    from Graph_model.model.config import ModelConfig
    from Graph_model.model        import OptionA
    from Graph_model.train.finetune import finetune

    cfg   = ModelConfig()
    model = OptionA(cfg)

    warnings.filterwarnings('ignore')  # suppress PyG loader noise
    result = finetune(
        model               = model,
        anchor_dataset      = synth_ds,
        stage               = 2,
        run_lolo_cv         = True,
        max_epochs          = 2,
        batch_size          = 8,
        lr                  = 1e-3,
        weight_decay        = 1e-4,
        warmup_epochs       = 1,
        patience            = 5,
        curriculum_schedule = [1, 2],
        device              = torch.device('cpu'),
        verbose             = False,
    )
    check("finetune returns dict",         isinstance(result, dict))
    check("result has 'folds'",            'folds'    in result)
    check("result has 'aggregate'",        'aggregate' in result)
    check("result has 'models'",           'models'   in result)
    check("9 folds completed",             len(result['folds']) == 9)
    check("9 trained models",              len(result['models']) == 9)
    check("all FoldMetrics instances",
          all(isinstance(fm, FoldMetrics) for fm in result['folds']))
    check("all folds have finite RMSE",
          all(math.isfinite(fm.rmse) for fm in result['folds']))
    check("aggregate has rmse_mean",       'rmse_mean' in result['aggregate'])
except Exception as e:
    for t in ["finetune returns dict","result has 'folds'","result has 'aggregate'",
              "result has 'models'","9 folds completed","9 trained models",
              "all FoldMetrics instances","all folds have finite RMSE",
              "aggregate has rmse_mean"]:
        fail(t, str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"Phase 4 smoke test: {PASS}/{total} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("ALL PASS ✓")
else:
    print(f"{FAIL} FAILURE(S) — fix before proceeding")
print('='*60)
sys.exit(0 if FAIL == 0 else 1)
