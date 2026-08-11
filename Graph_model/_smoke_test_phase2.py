#!/usr/bin/env python3
"""
Graph_model/_smoke_test_phase2.py
===================================
Smoke test for Phase 2: Three-Level Graph Construction.

Tests:
  A. Level 1  — mol_to_ligand_graph() on gallic acid SMILES
  B. Level 2  — BoxProteinGraph.get() on a real pH5 PDB + known box
  C. Level 3  — build_bipartite_graph() on a real docked SDF
  D. Builder  — ThreeLevelGraphBuilder.build() end-to-end → HeteroData

Run with:
  /Users/suppboat/Jupyter_Dock/.venv/bin/python Graph_model/_smoke_test_phase2.py
"""

import sys
from pathlib import Path

# ── make project importable ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
INFO = "\033[36mINFO\033[0m"

errors = []

def check(name, expr, *, expected=None, got=None):
    if expr:
        print(f"  [{PASS}] {name}")
    else:
        tag = f"  [{FAIL}] {name}"
        if expected is not None:
            tag += f"  expected={expected}  got={got}"
        print(tag)
        errors.append(name)

# ─────────────────────────────────────────────────────────────────────────────
# A. Level 1: Ligand Molecular Graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== A. Level 1: Ligand Molecular Graph ===")
try:
    from Graph_model.graph import mol_to_ligand_graph, LIGAND_NODE_DIM, LIGAND_EDGE_DIM
    import numpy as np

    GALLIC_SMILES = "OC(=O)c1cc(O)c(O)c(O)c1"  # gallic acid
    nf, ei, ea = mol_to_ligand_graph(GALLIC_SMILES)

    print(f"  {INFO} node_feat: {nf.shape}  edge_index: {ei.shape}  edge_feat: {ea.shape}")
    check("node_feat shape col", nf.shape[1] == LIGAND_NODE_DIM,
          expected=LIGAND_NODE_DIM, got=nf.shape[1])
    check("edge_feat shape col", ea.shape[1] == LIGAND_EDGE_DIM,
          expected=LIGAND_EDGE_DIM, got=ea.shape[1])
    check("edge_index shape rows", ei.shape[0] == 2)
    check("atoms > 0", nf.shape[0] > 0)
    check("bonds > 0", ei.shape[1] > 0)
    check("dtype float32", nf.dtype == np.float32)
    check("no NaN in node_feat", not np.isnan(nf).any())
    check("undirected (bonds paired)", ei.shape[1] % 2 == 0)

except Exception as exc:
    print(f"  [{FAIL}] Level 1 crashed: {exc}")
    import traceback; traceback.print_exc()
    errors.append("Level 1 crash")

# ─────────────────────────────────────────────────────────────────────────────
# B. Level 2: Protein Binding Site Context Graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== B. Level 2: Protein Binding Site Context Graph ===")
PDB_DIR = ROOT / "Phukhao" / "collagen_gallic_results"

try:
    from Graph_model.graph import BoxProteinGraph, pdb_for_ph, PROTEIN_NODE_DIM, PROTEIN_EDGE_DIM

    # Use pH 5.0 PDB
    try:
        pdb_path = pdb_for_ph(PDB_DIR, ph=5.0)
        print(f"  {INFO} PDB: {pdb_path.name}")
        pdb_ok = True
    except FileNotFoundError as fnf:
        print(f"  [{FAIL}] PDB not found: {fnf}")
        pdb_ok = False
        errors.append("Level 2 PDB not found")

    if pdb_ok:
        bg = BoxProteinGraph(pdb_path, ph=5.0)

        # Use a known box from the dataset  (GLU_cluster22 center from CSV)
        box_center = (56.0, 59.6, 54.3)   # approximate from CSV first row
        box_size   = 20.0
        target_residues = ["GLU27A", "GLU30A"]

        nf2, ei2, ea2, resnames, ca = bg.get(
            box_center=box_center,
            box_size=box_size,
            target_residues=target_residues,
        )

        print(f"  {INFO} node_feat: {nf2.shape}  edge_index: {ei2.shape}")
        print(f"  {INFO} residues: {resnames[:6]}{'...' if len(resnames)>6 else ''}")
        check("node_feat col dim", nf2.shape[1] == PROTEIN_NODE_DIM,
              expected=PROTEIN_NODE_DIM, got=nf2.shape[1])
        check("at least 1 residue", nf2.shape[0] >= 1)
        check("edge_feat col dim",
              ea2.shape[1] == PROTEIN_EDGE_DIM or ei2.shape[1] == 0,
              expected=PROTEIN_EDGE_DIM, got=ea2.shape[1] if ea2.ndim==2 else "empty")
        check("ca_coords shape", ca.shape == (nf2.shape[0], 3))
        check("no NaN node", not np.isnan(nf2).any())

except Exception as exc:
    print(f"  [{FAIL}] Level 2 crashed: {exc}")
    import traceback; traceback.print_exc()
    errors.append("Level 2 crash")

# ─────────────────────────────────────────────────────────────────────────────
# C. Level 3: Bipartite Interaction Graph
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== C. Level 3: Bipartite Interaction Graph ===")

try:
    from Graph_model.graph import build_bipartite_graph, BIPARTITE_EDGE_DIM

    # Find a real SDF in the dataset
    sdf_candidates = sorted(PDB_DIR.glob("*/docked_gallic_acid.sdf"))
    if not sdf_candidates:
        sdf_candidates = sorted(PDB_DIR.glob("**/*.sdf"))

    if not sdf_candidates:
        print(f"  {INFO} No SDF found in {PDB_DIR}, skipping Level 3")
    else:
        sdf_path = sdf_candidates[0]
        print(f"  {INFO} SDF: {sdf_path.name}  (dir: {sdf_path.parent.name})")

        # Use residues from Level 2 result (if available)
        dummy_ca = np.array([[56.0, 59.6, 54.3]], dtype=np.float32)
        dummy_res = ["GLU27A"]

        # Use proper Level 2 residues if they were computed
        try:
            bip_ca    = ca
            bip_res   = resnames if resnames else dummy_res
            bip_ca_use = bip_ca if len(bip_res) == len(bip_ca) else dummy_ca
        except NameError:
            bip_ca_use = dummy_ca
            bip_res    = dummy_res

        ei3, ea3 = build_bipartite_graph(
            sdf_path       = sdf_path,
            ca_coords      = bip_ca_use,
            residue_names  = bip_res,
            ph             = 5.0,
        )

        print(f"  {INFO} edge_index: {ei3.shape}  edge_feat: {ea3.shape}")
        check("bipartite edge_index rows = 2", ei3.shape[0] == 2)
        check("bipartite edge_feat col dim",
              ea3.shape[1] == BIPARTITE_EDGE_DIM or ei3.shape[1] == 0,
              expected=BIPARTITE_EDGE_DIM, got=ea3.shape[1] if ea3.ndim==2 else "empty")
        check("no NaN edge_feat", not np.isnan(ea3).any())
        check("atom indices in range",
              ei3.shape[1] == 0 or bool(ei3[0].max() < 100))  # gallic acid has ~11 heavy atoms

except Exception as exc:
    print(f"  [{FAIL}] Level 3 crashed: {exc}")
    import traceback; traceback.print_exc()
    errors.append("Level 3 crash")

# ─────────────────────────────────────────────────────────────────────────────
# D. Builder: end-to-end HeteroData
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== D. Builder: End-to-End HeteroData ===")

try:
    from Graph_model.graph import ThreeLevelGraphBuilder, GRAPH_DIM_SUMMARY
    import torch

    # Find a real SDF
    sdf_candidates = sorted(PDB_DIR.glob("*/docked_*.sdf"))
    sdf_rel = str(sdf_candidates[0].relative_to(PDB_DIR)) if sdf_candidates else ""

    record = {
        "smiles"        : "OC(=O)c1cc(O)c(O)c(O)c1",
        "ph"            : 5.0,
        "temperature_c" : 4.0,
        "docking_box"   : "GLU_cluster22",
        "box_center_x"  : 56.0,
        "box_center_y"  : 59.6,
        "box_center_z"  : 54.3,
        "box_size_A"    : 20.0,
        "target_residues": "GLU27A, GLU30A",
        "sdf_file"      : sdf_rel,
        "delta_g"       : -6.5,
        "receptor"      : "collagen",
        "sample_id"     : "test_001",
    }

    builder = ThreeLevelGraphBuilder(pdb_dir=PDB_DIR)
    data    = builder.build(record)

    print(f"  {INFO} node_types  : {data.node_types}")
    print(f"  {INFO} edge_types  : {data.edge_types}")
    print(f"  {INFO} ligand nodes: {data['ligand'].x.shape}")
    print(f"  {INFO} residue nodes: {data['residue'].x.shape}")
    print(f"  {INFO} bond edges  : {data['ligand','bond','ligand'].edge_index.shape}")
    print(f"  {INFO} contact edges: {data['residue','contact','residue'].edge_index.shape}")
    print(f"  {INFO} interact edges: {data['ligand','interacts','residue'].edge_index.shape}")
    print(f"  {INFO} y (ΔG): {data.y}")

    check("HeteroData type", str(type(data).__name__) == "HeteroData")
    check("ligand node type present",  "ligand"  in data.node_types)
    check("residue node type present", "residue" in data.node_types)
    check("ligand node dim",
          data["ligand"].x.shape[1] == GRAPH_DIM_SUMMARY["ligand_node_dim"],
          expected=GRAPH_DIM_SUMMARY["ligand_node_dim"],
          got=data["ligand"].x.shape[1])
    check("residue node dim",
          data["residue"].x.shape[1] == GRAPH_DIM_SUMMARY["protein_node_dim"],
          expected=GRAPH_DIM_SUMMARY["protein_node_dim"],
          got=data["residue"].x.shape[1])
    check("y shape",    data.y.shape == torch.Size([1, 1]))
    check("y correct",  abs(float(data.y) - (-6.5)) < 1e-5)
    check("ph stored",  abs(float(data.ph) - 5.0) < 1e-5)
    check("no NaN ligand x",  not torch.isnan(data["ligand"].x).any())
    check("no NaN residue x", not torch.isnan(data["residue"].x).any())

    # Print dimension summary
    print()
    from Graph_model.graph import print_dim_summary
    print_dim_summary()

except Exception as exc:
    print(f"  [{FAIL}] Builder crashed: {exc}")
    import traceback; traceback.print_exc()
    errors.append("Builder crash")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═" * 52)
if errors:
    print(f"  [{FAIL}] {len(errors)} check(s) failed: {errors}")
    sys.exit(1)
else:
    print(f"  [{PASS}] All Phase 2 graph checks passed ✓")
    sys.exit(0)
