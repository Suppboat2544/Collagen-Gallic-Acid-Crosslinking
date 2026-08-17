#!/usr/bin/env python3
"""
Train Model D (Multi-Task Selectivity GNN) and run interpretability analysis.

Steps:
1. Load the HeteroData dataset (anchor only — fast)
2. Train Model D for up to 50 epochs with early stopping
3. Save best checkpoint to results/model_d_best.pt
4. Run Integrated Gradients on representative ligands
5. Run GradCAM for cross-validation with PLIP contacts
6. Print SHAP-style feature importance summary

Usage:
    cd /Users/suppboat/Jupyter_Dock
    python -m Graph_model.run_model_d_interpret
"""

from __future__ import annotations
import json
import os
import sys
import time
import gc
from pathlib import Path

import torch
import numpy as np

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main():
    print("=" * 70)
    print("  Model D — Multi-Task Selectivity GNN: Train + Interpret")
    print("=" * 70)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    print("\n[1/6] Loading HeteroData dataset...")
    from Graph_model.data.hetero_dataset import HeteroDockingDataset

    ds = HeteroDockingDataset(
        include_mmp1=True,
        include_bipartite=True,
        use_cache=True,
        force_reload=False,
    )
    ds.load(verbose=True)
    print(f"  Loaded {len(ds)} graphs")

    if len(ds) == 0:
        print("ERROR: Dataset is empty!")
        return

    # ── 2. Train Model D ─────────────────────────────────────────────────
    print("\n[2/6] Training Model D...")
    from Graph_model.train.run_training import train_single_model

    # Was MPS-or-CPU only, so this entry point silently ran on CPU on any
    # CUDA machine. Now shares the package-wide resolver (CUDA -> MPS -> CPU).
    from Graph_model.train.device import resolve_device, describe
    device = resolve_device("auto")
    print(f"  Device: {device}   ({describe()})")

    result = train_single_model(
        model_key="D",
        dataset=ds,
        results_dir="Graph_model/results",
        max_epochs=50,
        batch_size=16,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_epochs=5,
        patience=15,
        val_ratio=0.15,
        device=device,
        seed=42,
    )

    model = result["model"]
    print(f"  Best epoch: {result['best_epoch']}")
    print(f"  Best val RMSE: {result['best_val_rmse']:.4f}")

    # ── 3. Save checkpoint ───────────────────────────────────────────────
    print("\n[3/6] Saving checkpoint...")
    ckpt_path = Path("Graph_model/results/model_d_best.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "best_epoch": result["best_epoch"],
        "best_val_rmse": result["best_val_rmse"],
    }, ckpt_path)
    print(f"  Saved to {ckpt_path}")

    # Move model to CPU for interpretation
    model = model.cpu()
    model.eval()
    gc.collect()

    # ── 4. Select representative ligands ─────────────────────────────────
    print("\n[4/6] Selecting representative ligands for interpretation...")

    # Group graphs by ligand
    ligand_groups = {}
    for i in range(len(ds)):
        data = ds[i]
        # Try to get ligand name from metadata
        lig_name = getattr(data, 'ligand_name', None)
        if lig_name is None and hasattr(data, 'smiles'):
            lig_name = data.smiles[:30]
        if lig_name is None:
            lig_name = f"ligand_{i}"
        if lig_name not in ligand_groups:
            ligand_groups[lig_name] = []
        ligand_groups[lig_name].append(i)

    print(f"  Found {len(ligand_groups)} unique ligands:")
    for name, indices in sorted(ligand_groups.items()):
        print(f"    {name}: {len(indices)} graphs")

    # Pick one representative graph per ligand (first occurrence)
    # Focus on galloyl compounds for interpretation
    priority_ligands = [
        "gallic_acid", "gallic acid",
        "pgg", "PGG", "penta-O-galloyl",
        "ellagic_acid", "ellagic acid",
        "protocatechuic_acid", "protocatechuic acid",
        "pyrogallol",
        "EDC", "edc",
        "NHS", "nhs",
    ]

    representatives = {}
    for lig_name, indices in ligand_groups.items():
        representatives[lig_name] = indices[0]

    # If we have very few ligands (all same name), just pick evenly spaced
    if len(representatives) <= 1:
        n_samples = min(9, len(ds))
        step = max(1, len(ds) // n_samples)
        representatives = {f"sample_{i}": i for i in range(0, len(ds), step)[:n_samples]}
        print(f"  Using {len(representatives)} evenly spaced samples")

    # ── 5. Integrated Gradients ──────────────────────────────────────────
    print("\n[5/6] Running Integrated Gradients (atom-level attribution)...")
    from Graph_model.interpret.integrated_gradients import (
        integrated_gradients, atom_importance_ranking
    )

    ig_results = {}
    for lig_name, idx in sorted(representatives.items())[:9]:  # Max 9 ligands
        data = ds[idx]
        print(f"\n  Ligand: {lig_name} (graph #{idx})")

        try:
            attr = integrated_gradients(
                model=model,
                data=data,
                target_idx=0,  # collagen ΔG head
                n_steps=50,
                internal_batch_size=10,
            )
            n_atoms = attr['atom_importance'].shape[0]
            top_k = min(10, n_atoms)
            importance = attr['atom_importance'].cpu().numpy()

            # Sort atoms by importance
            ranked_idx = np.argsort(-importance)[:top_k]

            print(f"    Convergence delta: {attr['convergence_delta']:.4f}")
            print(f"    Total atoms: {n_atoms}")
            print(f"    Top-{top_k} atoms by importance:")
            for rank, ai in enumerate(ranked_idx):
                print(f"      #{rank+1}: atom {ai} — importance {importance[ai]:.4f}")

            ig_results[lig_name] = {
                "n_atoms": n_atoms,
                "convergence_delta": float(attr['convergence_delta']),
                "top_atoms": [
                    {"atom_idx": int(ai), "importance": float(importance[ai])}
                    for ai in ranked_idx
                ],
                "mean_importance": float(importance.mean()),
                "max_importance": float(importance.max()),
                "std_importance": float(importance.std()),
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            ig_results[lig_name] = {"error": str(e)}

    # ── 6. GradCAM ──────────────────────────────────────────────────────
    print("\n[6/6] Running GradCAM (layer activation mapping)...")
    from Graph_model.interpret.gradcam import graph_gradcam

    gcam_results = {}
    for lig_name, idx in sorted(representatives.items())[:5]:  # Top 5
        data = ds[idx]
        print(f"\n  Ligand: {lig_name} (graph #{idx})")

        try:
            gcam = graph_gradcam(
                model=model,
                data=data,
                target_idx=0,
            )
            cam = gcam['atom_cam'].cpu().numpy()
            n_atoms = len(cam)
            top_k = min(5, n_atoms)
            ranked_idx = np.argsort(-cam)[:top_k]

            print(f"    Top-{top_k} GradCAM atoms:")
            for rank, ai in enumerate(ranked_idx):
                print(f"      #{rank+1}: atom {ai} — CAM {cam[ai]:.4f}")

            gcam_results[lig_name] = {
                "n_atoms": n_atoms,
                "top_atoms": [
                    {"atom_idx": int(ai), "cam_score": float(cam[ai])}
                    for ai in ranked_idx
                ],
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            gcam_results[lig_name] = {"error": str(e)}

    # ── Save all interpretation results ──────────────────────────────────
    print("\n" + "=" * 70)
    print("  Saving interpretation results...")
    interp_path = Path("Graph_model/results/model_d_interpretation.json")
    interp_data = {
        "model": "Option D — Multi-Task Selectivity GNN",
        "best_epoch": result["best_epoch"],
        "best_val_rmse": float(result["best_val_rmse"]),
        "integrated_gradients": ig_results,
        "gradcam": gcam_results,
    }

    with open(interp_path, "w") as f:
        json.dump(interp_data, f, indent=2)
    print(f"  Saved to {interp_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INTERPRETATION SUMMARY — Model D")
    print("=" * 70)
    print(f"  Training:  best epoch {result['best_epoch']}, "
          f"val RMSE = {result['best_val_rmse']:.3f} kcal/mol")
    print(f"  IG results for {len(ig_results)} ligands")
    print(f"  GradCAM results for {len(gcam_results)} ligands")

    # Cross-validate IG vs GradCAM for overlapping ligands
    common = set(ig_results.keys()) & set(gcam_results.keys())
    for lig in sorted(common):
        ig = ig_results[lig]
        gc_ = gcam_results[lig]
        if "error" in ig or "error" in gc_:
            continue
        ig_top = set(a["atom_idx"] for a in ig["top_atoms"][:5])
        gc_top = set(a["atom_idx"] for a in gc_["top_atoms"][:5])
        overlap = ig_top & gc_top
        print(f"  {lig}: IG/GradCAM top-5 overlap = {len(overlap)}/5 "
              f"atoms ({overlap})")

    print("\nDone!")


if __name__ == "__main__":
    main()
