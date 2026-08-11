"""
_smoke_test_phase7.py  — Virtual Screening (Phase 7)

Run:
    cd /Users/suppboat/Jupyter_Dock
    .venv/bin/python Graph_model/_smoke_test_phase7.py
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


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — GalloylLibrary generation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] GalloylLibrary generation")
from Graph_model.screen.library_gen import GalloylLibrary

try:
    lib = GalloylLibrary(seed=42, min_size=50, max_size=300)
    lib.generate()
    check("library generates",         True)
    check("has compounds",             len(lib) > 0)
    check("len ≥ min_size (50)",       len(lib) >= 50)
    check("len ≤ max_size (300)",      len(lib) <= 300)
    check("smiles_list returns list",  isinstance(lib.smiles_list, list))
    check("all SMILES are strings",    all(isinstance(s, str) for s in lib.smiles_list))

    # Check compound metadata
    c0 = lib[0]
    check("compound has 'smiles'",     'smiles' in c0)
    check("compound has 'mw'",         'mw' in c0)
    check("compound has 'n_ha'",       'n_ha' in c0)
    check("compound has 'logp'",       'logp' in c0)
    check("compound has 'source'",     'source' in c0)
    check("compound has 'uid'",        'uid' in c0)

    # Scaffold distribution
    dist = lib.scaffold_distribution()
    check("scaffold_distribution dict", isinstance(dist, dict))
    check("multiple scaffolds",         len(dist) > 1)
except Exception as e:
    for t in ["library generates","has compounds","len ≥ min_size (50)",
              "len ≤ max_size (300)","smiles_list returns list",
              "all SMILES are strings","compound has 'smiles'",
              "compound has 'mw'","compound has 'n_ha'","compound has 'logp'",
              "compound has 'source'","compound has 'uid'",
              "scaffold_distribution dict","multiple scaffolds"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — Library filtering
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2] Library filtering")
try:
    filtered = lib.filter(mw_max=400)
    check("filter returns list",        isinstance(filtered, list))
    check("filtered ≤ original",        len(filtered) <= len(lib))
    check("all pass MW filter",
          all(c['mw'] <= 400 for c in filtered))

    # DataFrame export
    df = lib.to_dataframe()
    check("to_dataframe works",         df is not None)
    check("df has smiles column",       'smiles' in df.columns)
    check("df length matches",          len(df) == len(lib))
except Exception as e:
    for t in ["filter returns list","filtered ≤ original",
              "all pass MW filter","to_dataframe works",
              "df has smiles column","df length matches"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — Screening inference (_smiles_to_screen_graph)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3] SMILES → screening graph")
from Graph_model.screen.predict import _smiles_to_screen_graph

try:
    g = _smiles_to_screen_graph("OC(=O)c1cc(O)c(O)c(O)c1", ph=5.0, temp_c=25)
    check("graph created",            g is not None)
    check("has ligand.x",             hasattr(g['ligand'], 'x'))
    check("has residue.x (dummy)",    hasattr(g['residue'], 'x'))
    check("has ph_enc",               hasattr(g, 'ph_enc'))
    check("has temp_enc",             hasattr(g, 'temp_enc'))
    check("has box_idx",              hasattr(g, 'box_idx'))
    check("ph_enc value correct",     abs(g.ph_enc.item() - 0.85) < 0.01)

    # Invalid SMILES returns None
    g_bad = _smiles_to_screen_graph("not_a_smiles")
    check("invalid SMILES → None",    g_bad is None)
except Exception as e:
    for t in ["graph created","has ligand.x","has residue.x (dummy)",
              "has ph_enc","has temp_enc","has box_idx",
              "ph_enc value correct","invalid SMILES → None"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — screen_candidates with a mock ensemble
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4] screen_candidates (mock ensemble)")
from Graph_model.screen.predict import screen_candidates
from Graph_model.model import OptionA, OptionAConfig
from Graph_model.model.uncertainty import HeteroscedasticWrapper
from Graph_model.model.ensemble import DeepEnsemble

try:
    # Build a 2-member ensemble of HeteroscedasticWrapper(OptionA)
    m1 = HeteroscedasticWrapper(OptionA(OptionAConfig()))
    m2 = HeteroscedasticWrapper(OptionA(OptionAConfig()))
    ens = DeepEnsemble([m1, m2])

    test_smiles = [
        "OC(=O)c1cc(O)c(O)c(O)c1",         # gallic acid
        "OC(=O)c1ccc(O)c(O)c1",             # PCA
        "Oc1cccc(O)c1O",                     # pyrogallol
    ]

    results = screen_candidates(ens, test_smiles, ph=5.0, temp_c=25)
    check("returns list",               isinstance(results, list))
    check("3 results",                  len(results) == 3)
    check("each has delta_g_mean",      all('delta_g_mean' in r for r in results))
    check("each has delta_g_std",       all('delta_g_std' in r for r in results))
    check("each has total_std",         all('total_std' in r for r in results))
    check("delta_g is finite",
          all(math.isfinite(r['delta_g_mean']) for r in results))
    check("std ≥ 0",
          all(r['delta_g_std'] >= 0 for r in results))
except Exception as e:
    for t in ["returns list","3 results","each has delta_g_mean",
              "each has delta_g_std","each has total_std",
              "delta_g is finite","std ≥ 0"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — Pareto front
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5] Pareto front identification")
from Graph_model.screen.pareto import pareto_front

try:
    # Synthetic two-objective data
    synth = [
        {"smiles": "A", "delta_g_mean": -8.0, "si_mean": 1.0},
        {"smiles": "B", "delta_g_mean": -6.0, "si_mean": 3.0},
        {"smiles": "C", "delta_g_mean": -7.0, "si_mean": 2.0},  # dominated by A+B
        {"smiles": "D", "delta_g_mean": -5.0, "si_mean": 4.0},
        {"smiles": "E", "delta_g_mean": -9.0, "si_mean": 0.5},
    ]
    front = pareto_front(synth)
    front_smiles = {r["smiles"] for r in front}

    check("pareto returns list",        isinstance(front, list))
    check("pareto non-empty",           len(front) > 0)
    # E has best ΔG, D has best SI — both should be on front
    check("E on front (best ΔG)",       "E" in front_smiles)
    check("D on front (best SI)",       "D" in front_smiles)
    # C is dominated by some combination
    check("front ≤ original",          len(front) <= len(synth))

    # Empty input
    check("empty input → empty",       pareto_front([]) == [])
except Exception as e:
    for t in ["pareto returns list","pareto non-empty",
              "E on front (best ΔG)","D on front (best SI)",
              "front ≤ original","empty input → empty"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 6 — Uncertainty filter
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6] Uncertainty filter")
from Graph_model.screen.pareto import filter_by_uncertainty

try:
    data_filt = [
        {"smiles": "A", "delta_g_std": 0.3},
        {"smiles": "B", "delta_g_std": 0.6},
        {"smiles": "C", "delta_g_std": 0.1},
        {"smiles": "D", "delta_g_std": 0.5},
        {"smiles": "E", "delta_g_std": 1.2},
    ]
    kept = filter_by_uncertainty(data_filt, max_std=0.5)
    kept_smiles = {r["smiles"] for r in kept}

    check("keeps low-uncertainty",      "A" in kept_smiles and "C" in kept_smiles)
    check("D at boundary kept",         "D" in kept_smiles)  # 0.5 ≤ 0.5
    check("removes high-uncertainty",   "B" not in kept_smiles and "E" not in kept_smiles)
    check("correct count (3)",          len(kept) == 3)
except Exception as e:
    for t in ["keeps low-uncertainty","D at boundary kept",
              "removes high-uncertainty","correct count (3)"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 7 — Shortlist candidates (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7] shortlist_candidates")
from Graph_model.screen.pareto import shortlist_candidates

try:
    # Use screening results from Group 4 (add fake SI data)
    for r in results:
        r["si_mean"] = np.random.uniform(0.5, 3.0)
        r["delta_g_std"] = 0.3  # all reliable

    top = shortlist_candidates(results, top_k=2, max_std=0.5,
                               diversity_filter=False)
    check("shortlist returns list",     isinstance(top, list))
    check("shortlist ≤ top_k",         len(top) <= 2)
    check("each has rank",             all('rank' in r for r in top))
    check("rank starts at 1",          top[0].get('rank') == 1 if top else True)

    # All filtered out
    bad = [{"smiles": "X", "delta_g_mean": -5, "delta_g_std": 2.0}]
    empty = shortlist_candidates(bad, top_k=5, max_std=0.5)
    check("all filtered → empty",      len(empty) == 0)
except Exception as e:
    for t in ["shortlist returns list","shortlist ≤ top_k",
              "each has rank","rank starts at 1","all filtered → empty"]:
        fail(t, str(e))

# ══════════════════════════════════════════════════════════════════════════════
# Group 8 — Package imports
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8] screen package imports")
try:
    from Graph_model.screen import (
        GalloylLibrary as _GL,
        screen_candidates as _sc,
        pareto_front as _pf,
        filter_by_uncertainty as _fu,
        shortlist_candidates as _slc,
    )
    check("all 5 public functions importable", True)
except ImportError as e:
    fail("all 5 public functions importable", str(e))

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
total = PASS + FAIL
print(f"Phase 7 smoke test: {PASS}/{total} PASS  {FAIL} FAIL")
if FAIL == 0:
    print("ALL PASS ✓")
else:
    print(f"{FAIL} FAILURE(S) — fix before proceeding")
print('='*60)
sys.exit(0 if FAIL == 0 else 1)
