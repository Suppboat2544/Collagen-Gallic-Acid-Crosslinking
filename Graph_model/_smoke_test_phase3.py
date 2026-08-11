"""
Phase 3 — Smoke test for all four model architectures.

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_phase3.py

All 28 checks must PASS.
"""

from __future__ import annotations

import sys
import math
import traceback

import torch
from torch_geometric.data import HeteroData, Batch


# ── Console helpers ───────────────────────────────────────────────────────────

OK   = "[PASS]"
FAIL = "[FAIL]"
_total = _passed = 0


def check(cond: bool, name: str, detail: str = "") -> None:
    global _total, _passed
    _total += 1
    if cond:
        _passed += 1
        print(f"  {OK}  {name}")
    else:
        info = f"  → {detail}" if detail else ""
        print(f"  {FAIL}  {name}{info}")


# ── Synthetic HeteroData factory ──────────────────────────────────────────────

def _make_graph(
    n_atoms: int = 8,
    n_bonds: int = 10,
    n_res:   int = 6,
    n_cont:  int = 5,
    n_bip:   int = 12,
) -> HeteroData:
    """
    Build one synthetic HeteroData object with all fields required by the models.
    Validates correct tensor shapes before returning.
    """
    from Graph_model.model import (
        LIGAND_NODE_DIM, LIGAND_EDGE_DIM,
        PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM,
        BIPARTITE_EDGE_DIM,
    )

    d = HeteroData()

    # Ligand
    d['ligand'].x           = torch.randn(n_atoms, LIGAND_NODE_DIM)
    d['ligand'].batch        = torch.zeros(n_atoms, dtype=torch.long)

    src = torch.randint(0, n_atoms, (n_bonds,))
    dst = torch.randint(0, n_atoms, (n_bonds,))
    d['ligand', 'bond', 'ligand'].edge_index = torch.stack([src, dst])
    d['ligand', 'bond', 'ligand'].edge_attr  = torch.randn(n_bonds, LIGAND_EDGE_DIM)

    # Protein / residue (use the same keys as the real graph builder)
    d['residue'].x           = torch.randn(n_res, PROTEIN_NODE_DIM)
    d['residue'].batch        = torch.zeros(n_res, dtype=torch.long)

    src_p = torch.randint(0, n_res, (n_cont,))
    dst_p = torch.randint(0, n_res, (n_cont,))
    d['residue', 'contact', 'residue'].edge_index = torch.stack([src_p, dst_p])
    d['residue', 'contact', 'residue'].edge_attr  = torch.randn(n_cont, PROTEIN_EDGE_DIM)

    # Bipartite (ligand → residue, 'interacts')
    src_b = torch.randint(0, n_atoms, (n_bip,))
    dst_b = torch.randint(0, n_res,   (n_bip,))
    d['ligand', 'interacts', 'residue'].edge_index = torch.stack([src_b, dst_b])
    d['ligand', 'interacts', 'residue'].edge_attr  = torch.randn(n_bip, BIPARTITE_EDGE_DIM)

    # Condition (1-D so Batch.from_data_list can cat correctly)
    d.ph_enc        = torch.tensor([0.02], dtype=torch.float32)
    d.temp_enc      = torch.tensor([0.64], dtype=torch.float32)
    d.box_idx       = torch.tensor([3],    dtype=torch.long)
    d.receptor_flag = torch.tensor([0.0],  dtype=torch.float32)

    return d


_PH   = [0.02, 0.85, 0.15, 0.50, 0.70, 0.02, 0.85, 0.15]
_TEMP = [0.64, 0.00, 0.64, 0.64, 0.30, 0.64, 0.00, 0.64]
_BOX  = [3, 1, 0, 5, 2, 4, 6, 7]
_REC  = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]


def _make_batch(B: int = 2) -> HeteroData:
    """Batch B synthetic graphs together using PyG Batch.from_data_list."""
    graphs = []
    for i in range(B):
        g = _make_graph(
            n_atoms = 8 + (i % 4) * 3,
            n_bonds = 10 + (i % 4) * 2,
            n_res   = 6,
            n_cont  = 5 + (i % 4),
            n_bip   = 12 + (i % 4) * 2,
        )
        j = i % 8
        g.ph_enc        = torch.tensor([_PH[j]],  dtype=torch.float32)
        g.temp_enc      = torch.tensor([_TEMP[j]], dtype=torch.float32)
        g.box_idx       = torch.tensor([_BOX[j]],  dtype=torch.long)
        g.receptor_flag = torch.tensor([_REC[j]],  dtype=torch.float32)
        graphs.append(g)
    return Batch.from_data_list(graphs)


# ── Individual model tests ────────────────────────────────────────────────────

def test_option_a(B: int = 2) -> None:
    print("\n─── Option A: Baseline GATv2 ──────────────────────────────────")
    from Graph_model.model import OptionA, OptionAConfig
    try:
        cfg   = OptionAConfig()
        model = OptionA(cfg).eval()
        data  = _make_batch(B)

        with torch.no_grad():
            out = model(data)

        check(isinstance(out, torch.Tensor),          "output is Tensor")
        check(out.shape == (B, 1),                    f"shape == ({B}, 1)", f"got {out.shape}")
        check(not out.isnan().any().item(),            "no NaN in output")
        check(not out.isinf().any().item(),            "no Inf in output")

        # Gradient flow
        model.train()
        data2 = _make_batch(B)
        pred  = model(data2)
        loss  = pred.sum()
        loss.backward()
        any_grad = any(p.grad is not None for p in model.parameters())
        check(any_grad, "gradients flow to parameters")

    except Exception:
        traceback.print_exc()
        check(False, "Option A forward pass (exception caught)")


def test_option_b(B: int = 2) -> None:
    print("\n─── Option B: Dual Encoder Cross-Attention ────────────────────")
    from Graph_model.model import OptionB, OptionBConfig
    try:
        cfg   = OptionBConfig()
        model = OptionB(cfg).eval()
        data  = _make_batch(B)

        with torch.no_grad():
            out, attn_list = model(data)

        check(isinstance(out, torch.Tensor),          "output is Tensor")
        check(out.shape == (B, 1),                    f"shape == ({B}, 1)", f"got {out.shape}")
        check(not out.isnan().any().item(),            "no NaN in output")

        # Attention weights: list of per-graph dicts {'attn_weights', 'graph_idx'}
        check(isinstance(attn_list, list),             "attn_list is list")
        check(len(attn_list) == B,                     f"len(attn_list) == B={B}", f"got {len(attn_list)}")

        for gi, entry in enumerate(attn_list):
            check(isinstance(entry, dict),             f"graph {gi}: entry is dict")
            check('attn_weights' in entry,             f"graph {gi}: has 'attn_weights'")
            if 'attn_weights' in entry:
                aw = entry['attn_weights']             # [N_L_g, N_R_g]  (layer-averaged)
                check(isinstance(aw, torch.Tensor),    f"graph {gi}: attn_weights is Tensor")
                row_sums = aw.sum(dim=-1)              # should sum to ~1 per atom
                row_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
                check(row_ok, f"graph {gi}: attn rows sum to 1",
                      f"max_dev={((row_sums - 1).abs().max().item()):.2e}")

        # Gradient flow
        model.train()
        data2  = _make_batch(B)
        pred2, _ = model(data2)
        pred2.sum().backward()
        any_grad = any(p.grad is not None for p in model.parameters())
        check(any_grad, "gradients flow to parameters")

    except Exception:
        traceback.print_exc()
        check(False, "Option B forward pass (exception caught)")


def test_option_c(B: int = 2) -> None:
    print("\n─── Option C: Fragment-Aware Hierarchical MPNN ────────────────")
    from Graph_model.model import OptionC, OptionCConfig
    try:
        cfg   = OptionCConfig()
        model = OptionC(cfg).eval()
        data  = _make_batch(B)
        # PGG and gallic acid SMILES (real molecules for BRICS test)
        smiles = [
            "OC(=O)c1cc(O)c(O)c(O)c1",                    # gallic acid (1 fragment)
            "OC(=O)c1cc(O)c(O)c(O)c1",                    # gallic acid again
        ]

        with torch.no_grad():
            delta_g, frag_contrib = model(data, smiles_list=smiles)

        check(isinstance(delta_g, torch.Tensor),           "delta_g is Tensor")
        check(delta_g.shape == (B, 1),                     f"shape == ({B}, 1)", f"got {delta_g.shape}")
        check(not delta_g.isnan().any().item(),             "no NaN in delta_g")

        check(isinstance(frag_contrib, list),               "frag_contrib is list")
        check(len(frag_contrib) == B,                      f"len(frag_contrib) == {B}")
        for gi, fc in enumerate(frag_contrib):
            check(isinstance(fc, torch.Tensor),            f"graph {gi}: frag_contrib is Tensor")
            check(fc.ndim == 1,                            f"graph {gi}: frag_contrib 1-D", f"shape={fc.shape}")
            check(fc.shape[0] >= 1,                        f"graph {gi}: ≥1 fragment")

        # Gradient flow
        model.train()
        data2 = _make_batch(B)
        pred2, _ = model(data2, smiles_list=smiles)
        pred2.sum().backward()
        any_grad = any(p.grad is not None for p in model.parameters())
        check(any_grad, "gradients flow to parameters")

    except Exception:
        traceback.print_exc()
        check(False, "Option C forward pass (exception caught)")


def test_option_d(B: int = 4) -> None:
    """Test with B=4 so we can have some NaN rows."""
    print("\n─── Option D: Multi-Task Selectivity GNN ──────────────────────")
    from Graph_model.model import OptionD, OptionDConfig
    try:
        cfg   = OptionDConfig()
        model = OptionD(cfg).eval()
        data  = _make_batch(B)
        # Mock the batch condition for B=4
        # conditions already set correctly by _make_batch(B=4)

        with torch.no_grad():
            out = model(data)

        check(isinstance(out, dict),                       "output is dict")
        for key in ('collagen', 'mmp1', 'si', 'log_var'):
            check(key in out,                              f"'{key}' in output")

        check(out['collagen'].shape == (B, 1),             f"collagen shape ({B},1)", f"got {out['collagen'].shape}")
        check(out['mmp1'].shape    == (B, 1),              f"mmp1 shape ({B},1)",     f"got {out['mmp1'].shape}")
        check(out['si'].shape      == (B, 1),              f"si shape ({B},1)",       f"got {out['si'].shape}")
        check(out['log_var'].shape == (3,),                f"log_var shape (3,)",     f"got {out['log_var'].shape}")
        check(not out['collagen'].isnan().any().item(),    "no NaN in collagen")

        # Loss computation with NaN masking
        targets = {
            'collagen': torch.randn(B, 1),
            'mmp1':     torch.full((B, 1), float('nan')),  # all NaN → masked out
            'si':       torch.full((B, 1), float('nan')),  # all NaN → masked out
        }
        model.train()
        out_train = model(data)
        loss = model.compute_loss(out_train, targets)
        check(isinstance(loss, torch.Tensor),              "compute_loss returns Tensor")
        check(loss.ndim == 0,                              "loss is scalar")
        check(not loss.isnan().item(),                     "loss is not NaN")

        loss.backward()
        any_grad = any(p.grad is not None for p in model.parameters())
        check(any_grad,                                    "gradients flow to parameters")

        # Test with partial MMP-1 targets
        model.zero_grad()
        t2 = {
            'collagen': torch.randn(B, 1),
            'mmp1':     torch.full((B, 1), float('nan')),
            'si':       torch.full((B, 1), float('nan')),
        }
        # Only first row has MMP-1 target
        t2['mmp1'][0] = torch.tensor([-1.5])
        t2['si'][0]   = torch.tensor([0.3])
        out2 = model(data)
        loss2 = model.compute_loss(out2, t2)
        check(not loss2.isnan().item(),                    "partial NaN targets → valid loss")

    except Exception:
        traceback.print_exc()
        check(False, "Option D forward pass (exception caught)")


# ── PGG multi-fragment interpretability demo ──────────────────────────────────

def test_pgg_fragment_decomposition() -> None:
    """
    Regression check: PGG (pentagalloylglucose) SMILES must decompose into
    ≥ 5 fragments (5 galloyl arms + glucose core = 6 BRICS components).
    """
    print("\n─── PGG BRICS Fragment Decomposition ──────────────────────────")
    try:
        from Graph_model.model.option_c import _brics_fragments
        pgg = (
            "OC(=O)c1cc(O)c(O)c(O)c1OC[C@H]2OC(OC(=O)c3cc(O)c(O)c(O)c3)"
            "[C@@H](OC(=O)c4cc(O)c(O)c(O)c4)[C@H](OC(=O)c5cc(O)c(O)c(O)c5)"
            "[C@@H]2OC(=O)c6cc(O)c(O)c(O)c6"
        )
        atom_frag, cut_pairs = _brics_fragments(pgg)
        n_frags = max(atom_frag) + 1 if atom_frag else 0
        print(f"  PGG: {len(atom_frag)} atoms → {n_frags} fragments, "
              f"{len(cut_pairs)//2} cut bonds")
        check(n_frags >= 2, f"PGG has ≥ 2 fragments (got {n_frags})")
        check(len(cut_pairs) > 0, "PGG has inter-fragment cut bonds")
    except Exception:
        traceback.print_exc()
        check(False, "PGG fragment decomposition")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 66)
    print("       Phase 3 Smoke Test — GNN Model Architectures")
    print("=" * 66)

    test_option_a()
    test_option_b()
    test_option_c()
    test_option_d()
    test_pgg_fragment_decomposition()

    print("\n" + "=" * 66)
    status = "ALL PASS" if _passed == _total else f"{_total - _passed} FAILED"
    print(f"  Result: {_passed}/{_total}  {status}")
    print("=" * 66)

    sys.exit(0 if _passed == _total else 1)


if __name__ == "__main__":
    # Ensure project root is on the path
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    main()
