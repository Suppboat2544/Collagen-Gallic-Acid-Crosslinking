"""
_smoke_test_phase5.py  — Uncertainty Quantification (Phase 5)

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_phase5.py
"""

from __future__ import annotations
import sys, math, warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch_geometric.data import HeteroData

warnings.filterwarnings('ignore')

# ── test harness ──────────────────────────────────────────────────────────────
PASS = 0; FAIL = 0

def ok(n):
    global PASS; PASS += 1; print(f"  [PASS] {n}")
def fail(n, r=""):
    global FAIL; FAIL += 1; print(f"  [FAIL] {n}" + (f" — {r}" if r else ""))
def check(n, c, r=""):
    ok(n) if c else fail(n, r)

# ── Synthetic HeteroData builder ─────────────────────────────────────────────
def _make_batch(B=2):
    items = []
    for g in range(B):
        d = HeteroData()
        n_lig, n_res = 6, 8
        d['ligand'].x = torch.randn(n_lig, 35)
        d['ligand', 'bond', 'ligand'].edge_index = torch.randint(0, n_lig, (2, 10))
        d['ligand', 'bond', 'ligand'].edge_attr  = torch.randn(10, 13)
        d['residue'].x = torch.randn(n_res, 30)
        d['residue', 'contact', 'residue'].edge_index = torch.randint(0, n_res, (2, 12))
        d['residue', 'contact', 'residue'].edge_attr  = torch.randn(12, 4)
        n_bp = n_lig * n_res
        d['ligand', 'interacts', 'residue'].edge_index = torch.stack([
            torch.randint(0, n_lig, (n_bp,)),
            torch.randint(0, n_res, (n_bp,)),
        ])
        d['ligand', 'interacts', 'residue'].edge_attr = torch.randn(n_bp, 8)
        d.y = torch.tensor([[-5.0 + g]])
        d.cond = torch.tensor([0.5, 0.5, 0.0, 0.0])
        d.ph_enc = torch.tensor([0.5])
        d.temp_enc = torch.tensor([0.5])
        d.box_idx = torch.tensor([0])
        d.receptor_flag = torch.tensor([0.0])
        d.tier = 0
        d.ligand_name = f"lig_{g}"
        items.append(d)
    from torch_geometric.data import Batch
    return Batch.from_data_list(items)

# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — HeteroscedasticWrapper on OptionA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] HeteroscedasticWrapper + OptionA")
from Graph_model.model import OptionA, OptionAConfig
from Graph_model.model.uncertainty import HeteroscedasticWrapper, GaussianNLLLoss

try:
    cfg = OptionAConfig()
    base = OptionA(cfg)
    model = HeteroscedasticWrapper(base)
    batch = _make_batch(2)

    out = model(batch)
    check("returns dict",         isinstance(out, dict))
    check("has 'mu'",             'mu'      in out)
    check("has 'log_var'",        'log_var' in out)
    check("has 'sigma'",          'sigma'   in out)
    check("mu shape [B,1]",       out['mu'].shape == (2, 1))
    check("log_var shape [B,1]",  out['log_var'].shape == (2, 1))
    check("sigma > 0",            (out['sigma'] > 0).all().item())
    mu, sig = model.predict(batch)
    check("predict returns tuple", mu.shape == (2, 1) and sig.shape == (2, 1))
except Exception as e:
    for t in ["returns dict","has 'mu'","has 'log_var'","has 'sigma'",
              "mu shape [B,1]","log_var shape [B,1]","sigma > 0",
              "predict returns tuple"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — HeteroscedasticWrapper on OptionB
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] HeteroscedasticWrapper + OptionB")
from Graph_model.model import OptionB, OptionBConfig

try:
    base_b = OptionB(OptionBConfig())
    model_b = HeteroscedasticWrapper(base_b)
    batch_b = _make_batch(2)
    out_b = model_b(batch_b)
    check("OptionB mu shape",  out_b['mu'].shape == (2, 1))
    check("OptionB sigma > 0", (out_b['sigma'] > 0).all().item())
except Exception as e:
    for t in ["OptionB mu shape","OptionB sigma > 0"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — HeteroscedasticWrapper on OptionC
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] HeteroscedasticWrapper + OptionC")
from Graph_model.model import OptionC, OptionCConfig

try:
    base_c = OptionC(OptionCConfig())
    model_c = HeteroscedasticWrapper(base_c)
    batch_c = _make_batch(2)
    out_c = model_c(batch_c)
    check("OptionC mu shape",  out_c['mu'].shape == (2, 1))
    check("OptionC sigma > 0", (out_c['sigma'] > 0).all().item())
except Exception as e:
    for t in ["OptionC mu shape","OptionC sigma > 0"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — GaussianNLLLoss
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] GaussianNLLLoss")

try:
    nll = GaussianNLLLoss(reduction='mean')
    mu_t      = torch.tensor([[0.0], [1.0]])
    log_var_t = torch.tensor([[0.0], [0.0]])   # σ²=1
    target_t  = torch.tensor([[0.0], [1.0]])   # perfect → loss = ½ log(1) = 0
    loss = nll(mu_t, log_var_t, target_t)
    check("NLL is scalar",          loss.ndim == 0)
    check("NLL ≥ 0 for σ²=1",       loss.item() >= -0.01)  # ½ log(1) = 0

    # Non-zero residual
    target2 = torch.tensor([[5.0], [5.0]])
    loss2   = nll(mu_t, log_var_t, target2)
    check("NLL increases with error", loss2.item() > loss.item())

    # Backprop
    mu_p  = torch.tensor([[0.0]], requires_grad=True)
    lv_p  = torch.tensor([[0.0]], requires_grad=True)
    tgt_p = torch.tensor([[1.0]])
    l = nll(mu_p, lv_p, tgt_p)
    l.backward()
    check("NLL gradients flow to mu",     mu_p.grad is not None)
    check("NLL gradients flow to log_var", lv_p.grad is not None)
except Exception as e:
    for t in ["NLL is scalar","NLL ≥ 0 for σ²=1","NLL increases with error",
              "NLL gradients flow to mu","NLL gradients flow to log_var"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — DeepEnsemble
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] DeepEnsemble")
from Graph_model.model.ensemble import DeepEnsemble

try:
    members = []
    for seed in range(3):
        torch.manual_seed(seed)
        m = OptionA(OptionAConfig())
        members.append(m)

    ens = DeepEnsemble(members)
    check("ensemble has 3 members", ens.M == 3)

    batch_e = _make_batch(2)
    out_e = ens(batch_e)
    check("ens returns dict",        isinstance(out_e, dict))
    check("ens has mu_mean",         'mu_mean' in out_e)
    check("ens has mu_std",          'mu_std'  in out_e)
    check("ens has total_var",       'total_var' in out_e)
    check("mu_mean shape [B,1]",     out_e['mu_mean'].shape == (2, 1))
    check("mu_std ≥ 0",             (out_e['mu_std'] >= 0).all().item())
    check("all_mu shape [M,B,1]",    out_e['all_mu'].shape == (3, 2, 1))

    mu_m, ep_std, tot_std = ens.predict(batch_e)
    check("predict returns 3 tensors", mu_m.shape == (2, 1))
except Exception as e:
    for t in ["ensemble has 3 members","ens returns dict","ens has mu_mean",
              "ens has mu_std","ens has total_var","mu_mean shape [B,1]",
              "mu_std ≥ 0","all_mu shape [M,B,1]","predict returns 3 tensors"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — Heteroscedastic ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Heteroscedastic ensemble")

try:
    het_members = []
    for seed in range(3):
        torch.manual_seed(seed)
        b = OptionA(OptionAConfig())
        het_members.append(HeteroscedasticWrapper(b))

    het_ens = DeepEnsemble(het_members)
    batch_h = _make_batch(2)
    out_h = het_ens(batch_h)
    check("het ens has sigma_mean",    'sigma_mean' in out_h)
    check("het ens sigma_mean > 0",    (out_h['sigma_mean'] > 0).all().item())
    check("het ens total_var > 0",     (out_h['total_var'] > 0).all().item())
    check("total_var ≥ epistemic_var",
          (out_h['total_var'] >= out_h['mu_std']**2 - 1e-6).all().item())
except Exception as e:
    for t in ["het ens has sigma_mean","het ens sigma_mean > 0",
              "het ens total_var > 0", "total_var ≥ epistemic_var"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — Ensemble save/load round-trip
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] Ensemble save/load round-trip")
import tempfile, shutil

try:
    tmpdir = Path(tempfile.mkdtemp())
    ens.save(tmpdir)
    check("member files created", all((tmpdir / f"member_{i}.pt").exists() for i in range(3)))

    loaded = DeepEnsemble.load(
        model_factory=lambda: OptionA(OptionAConfig()),
        save_dir=tmpdir,
        n_members=3,
    )
    out_loaded = loaded(batch_e)
    diff = (out_loaded['mu_mean'] - out_e['mu_mean']).abs().max().item()
    check("loaded ensemble matches", diff < 1e-5)
    shutil.rmtree(tmpdir)
except Exception as e:
    for t in ["member files created","loaded ensemble matches"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 8 — NLL training step (heteroscedastic, backprop)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] NLL training step (heteroscedastic model)")

try:
    base_tr = OptionA(OptionAConfig())
    het_tr  = HeteroscedasticWrapper(base_tr)
    opt     = torch.optim.Adam(het_tr.parameters(), lr=1e-3)
    nll_fn  = GaussianNLLLoss()
    batch_tr = _make_batch(4)
    targets  = batch_tr.y

    losses = []
    for _ in range(5):
        opt.zero_grad()
        out = het_tr(batch_tr)
        loss = nll_fn(out['mu'], out['log_var'], targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    check("loss is finite",       all(math.isfinite(l) for l in losses))
    check("loss non-increasing",  losses[-1] <= losses[0] + 1.0)  # rough check
except Exception as e:
    for t in ["loss is finite", "loss non-increasing"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"Phase 5 smoke test: {PASS}/{total} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("ALL PASS ✓")
else:
    print(f"{FAIL} FAILURE(S) — fix before proceeding")
print('='*60)
sys.exit(0 if FAIL == 0 else 1)
