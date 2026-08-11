"""
_smoke_test_phase6.py  — Interpretability (Phase 6)

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_phase6.py
"""

from __future__ import annotations
import sys, math, warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
from torch_geometric.data import HeteroData, Batch

warnings.filterwarnings('ignore')

PASS = 0; FAIL = 0
def ok(n):  global PASS; PASS += 1; print(f"  [PASS] {n}")
def fail(n, r=""):  global FAIL; FAIL += 1; print(f"  [FAIL] {n}" + (f" — {r}" if r else ""))
def check(n, c, r=""): ok(n) if c else fail(n, r)

# ── Synthetic data builder ───────────────────────────────────────────────────
def _make_single(n_lig=8, n_res=10):
    d = HeteroData()
    d['ligand'].x = torch.randn(n_lig, 35)
    d['ligand', 'bond', 'ligand'].edge_index = torch.randint(0, n_lig, (2, 14))
    d['ligand', 'bond', 'ligand'].edge_attr  = torch.randn(14, 13)
    d['residue'].x = torch.randn(n_res, 30)
    d['residue', 'contact', 'residue'].edge_index = torch.randint(0, n_res, (2, 16))
    d['residue', 'contact', 'residue'].edge_attr  = torch.randn(16, 4)
    n_bp = n_lig * n_res
    d['ligand', 'interacts', 'residue'].edge_index = torch.stack([
        torch.randint(0, n_lig, (n_bp,)), torch.randint(0, n_res, (n_bp,))])
    d['ligand', 'interacts', 'residue'].edge_attr = torch.randn(n_bp, 8)
    d.y = torch.tensor([[-5.0]])
    d.ph_enc = torch.tensor([0.5])
    d.temp_enc = torch.tensor([0.5])
    d.box_idx = torch.tensor([0])
    d.receptor_flag = torch.tensor([0.0])
    d.tier = 0
    d.ligand_name = "gallic_acid"
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — Integrated Gradients on OptionA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Integrated Gradients — OptionA")
from Graph_model.model import OptionA, OptionAConfig
from Graph_model.interpret.integrated_gradients import (
    integrated_gradients, atom_importance_ranking
)

try:
    model_a = OptionA(OptionAConfig())
    model_a.eval()
    data = _make_single(n_lig=8)

    attr = integrated_gradients(model_a, data, n_steps=10)
    check("returns dict",              isinstance(attr, dict))
    check("has atom_attr",             'atom_attr' in attr)
    check("has atom_importance",       'atom_importance' in attr)
    check("atom_attr shape [N]",       attr['atom_attr'].shape == (8,))
    check("atom_attr_raw shape [N,35]",attr['atom_attr_raw'].shape == (8, 35))
    check("convergence_delta finite",  math.isfinite(attr['convergence_delta']))
    check("convergence_delta small",   abs(attr['convergence_delta']) < 5.0)
    check("importance ≥ 0",           (attr['atom_importance'] >= 0).all().item())
except Exception as e:
    for t in ["returns dict","has atom_attr","has atom_importance",
              "atom_attr shape [N]","atom_attr_raw shape [N,35]",
              "convergence_delta finite","convergence_delta small",
              "importance ≥ 0"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — atom_importance_ranking with real SMILES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Atom importance ranking — gallic acid")

try:
    gallic_smiles = "OC(=O)c1cc(O)c(O)c(O)c1"
    from rdkit import Chem
    mol = Chem.MolFromSmiles(gallic_smiles)
    n_atoms = mol.GetNumAtoms()

    # Build data matching atom count
    data_ga = _make_single(n_lig=n_atoms)
    attr_ga = integrated_gradients(model_a, data_ga, n_steps=10)

    ranked = atom_importance_ranking(attr_ga, gallic_smiles, top_k=5)
    check("ranking returns list",      isinstance(ranked, list))
    check("ranking has entries",        len(ranked) > 0)
    check("each has 'symbol'",         all('symbol' in r for r in ranked))
    check("each has 'importance'",     all('importance' in r for r in ranked))
    check("rank order correct",
          all(ranked[i]['importance'] >= ranked[i+1]['importance']
              for i in range(len(ranked)-1)))
except Exception as e:
    for t in ["ranking returns list","ranking has entries",
              "each has 'symbol'","each has 'importance'","rank order correct"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — Cross-attention visualization (OptionB)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] Cross-attention — OptionB")
from Graph_model.model import OptionB, OptionBConfig
from Graph_model.interpret.attention_viz import (
    extract_attention, rank_residues, attention_heatmap_data,
    validate_binding_site_rediscovery,
)

try:
    model_b = OptionB(OptionBConfig())
    model_b.eval()
    data_b = Batch.from_data_list([_make_single(n_lig=6, n_res=10)])

    _ = model_b(data_b)
    attn_list = extract_attention(model_b)
    check("extract_attention returns list",  isinstance(attn_list, list))
    check("has entries",                     len(attn_list) > 0)

    if attn_list and attn_list[0]:
        entry = attn_list[0]
        check("entry has attn_weights", 'attn_weights' in entry)
        check("attn_weights shape",
              entry['attn_weights'].shape[0] == 6 and entry['attn_weights'].shape[1] == 10)

        res_labels = [f"RES_{i}" for i in range(10)]
        ranked_r = rank_residues(entry, res_labels, top_k=5)
        check("rank_residues returns list",  isinstance(ranked_r, list))
        check("rank_residues has entries",   len(ranked_r) > 0)
        check("each has 'residue'",          all('residue' in r for r in ranked_r))
        check("each has 'score'",            all('score' in r for r in ranked_r))

        hm = attention_heatmap_data(entry, residue_labels=res_labels)
        check("heatmap has matrix",          'matrix' in hm)
        check("heatmap matrix shape",        hm['matrix'].shape == (6, 10))

        val = validate_binding_site_rediscovery(
            ranked_r, ['RES_0', 'RES_1', 'RES_2'], top_k=5)
        check("validation has recall",       'recall_at_k' in val)
        check("recall is float [0,1]",       0.0 <= val['recall_at_k'] <= 1.0)
    else:
        for t in ["entry has attn_weights","attn_weights shape",
                  "rank_residues returns list","rank_residues has entries",
                  "each has 'residue'","each has 'score'","heatmap has matrix",
                  "heatmap matrix shape","validation has recall","recall is float [0,1]"]:
            fail(t, "no attention entries returned")
except Exception as e:
    for t in ["extract_attention returns list","has entries",
              "entry has attn_weights","attn_weights shape",
              "rank_residues returns list","rank_residues has entries",
              "each has 'residue'","each has 'score'","heatmap has matrix",
              "heatmap matrix shape","validation has recall","recall is float [0,1]"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — Fragment contribution (OptionC)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] Fragment contributions — OptionC")
from Graph_model.model import OptionC, OptionCConfig
from Graph_model.interpret.fragment_contrib import (
    extract_fragment_contributions, label_fragments, rank_fragments,
)

try:
    model_c = OptionC(OptionCConfig())
    model_c.eval()

    # Use gallic acid (has meaningful BRICS fragments)
    gallic_smiles = "OC(=O)c1cc(O)c(O)c(O)c1"
    mol_ga = Chem.MolFromSmiles(gallic_smiles)
    n_ga = mol_ga.GetNumAtoms()

    data_c = _make_single(n_lig=n_ga)
    data_c.smiles = gallic_smiles
    batch_c = Batch.from_data_list([data_c])

    delta_g, frag_contribs = model_c(batch_c, smiles_list=[gallic_smiles])
    check("OptionC returns tuple",      isinstance(frag_contribs, list))
    check("frag_contribs non-empty",    len(frag_contribs) > 0)

    contrib = extract_fragment_contributions(frag_contribs, graph_idx=0)
    check("extract returns Tensor",     isinstance(contrib, torch.Tensor))
    check("contrib has values",         contrib.numel() > 0)

    frags = label_fragments(gallic_smiles)
    check("label_fragments returns list", isinstance(frags, list))
    check("label_fragments non-empty",    len(frags) > 0)
    check("each frag has smiles",         all('smiles' in f for f in frags))
    check("each frag has atom_indices",   all('atom_indices' in f for f in frags))

    ranked_f = rank_fragments(contrib, gallic_smiles)
    check("rank_fragments returns list",  isinstance(ranked_f, list))
    check("rank_fragments has rank field",
          all('rank' in r for r in ranked_f))
    check("rank_fragments has delta_g",
          all('delta_g' in r for r in ranked_f))
    check("rank_fragments has pct_total",
          all('pct_total' in r for r in ranked_f))
except Exception as e:
    for t in ["OptionC returns tuple","frag_contribs non-empty",
              "extract returns Tensor","contrib has values",
              "label_fragments returns list","label_fragments non-empty",
              "each frag has smiles","each frag has atom_indices",
              "rank_fragments returns list","rank_fragments has rank field",
              "rank_fragments has delta_g","rank_fragments has pct_total"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — Integrated Gradients on HeteroscedasticWrapper
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Integrated Gradients — HeteroscedasticWrapper")
from Graph_model.model.uncertainty import HeteroscedasticWrapper

try:
    het = HeteroscedasticWrapper(OptionA(OptionAConfig()))
    het.eval()
    data_h = _make_single(n_lig=8)
    attr_h = integrated_gradients(het, data_h, n_steps=10)
    check("het wrapper attr shape",   attr_h['atom_attr'].shape == (8,))
    check("het wrapper convergence",  math.isfinite(attr_h['convergence_delta']))
except Exception as e:
    for t in ["het wrapper attr shape","het wrapper convergence"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — Module imports from interpret/__init__
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] interpret package imports")

try:
    from Graph_model.interpret import (
        integrated_gradients as _ig,
        atom_importance_ranking as _air,
        extract_attention as _ea,
        rank_residues as _rr,
        attention_heatmap_data as _ahd,
        validate_binding_site_rediscovery as _vbsr,
        extract_fragment_contributions as _efc,
        label_fragments as _lf,
        rank_fragments as _rf,
        summarise_pgg_arms as _spa,
    )
    check("all 10 public functions importable", True)
except ImportError as e:
    fail("all 10 public functions importable", str(e))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"Phase 6 smoke test: {PASS}/{total} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("ALL PASS ✓")
else:
    print(f"{FAIL} FAILURE(S) — fix before proceeding")
print('='*60)
sys.exit(0 if FAIL == 0 else 1)
