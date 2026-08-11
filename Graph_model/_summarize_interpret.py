#!/usr/bin/env python3
"""Summarize Model D interpretation results."""
import json

with open("results/model_d_interpretation.json") as f:
    d = json.load(f)

ig = d["integrated_gradients"]

print("=== MODEL D INTERPRETATION SUMMARY ===")
print(f"Best epoch: {d['best_epoch']}, val_RMSE: {d['best_val_rmse']:.4f}")
print()

for name in ["gallic_acid", "pyrogallol", "protocatechuic_acid",
             "ellagic_acid", "pentagalloylglucose", "EDC"]:
    r = ig[name]
    total = r["mean_importance"] * r["n_atoms"]
    top3 = sum(a["importance"] for a in r["top_atoms"][:3])
    print(f"{name}: {r['n_atoms']} atoms, conv_delta={r['convergence_delta']:.3f}")
    print(f"  top-3 sum={top3:.3f} / total={total:.3f} = {100*top3/total:.1f}%")
    print(f"  max={r['max_importance']:.3f}, mean={r['mean_importance']:.3f}")
    top3_atoms = [(a["atom_idx"], a["importance"]) for a in r["top_atoms"][:3]]
    print(f"  top-3 atoms: {top3_atoms}")
    print()

# PGG galloyl enrichment
pgg = ig["pentagalloylglucose"]
top10_sum = sum(a["importance"] for a in pgg["top_atoms"][:10])
total_pgg = pgg["mean_importance"] * pgg["n_atoms"]
top10_mean = top10_sum / 10
print(f"PGG top-10 mean importance: {top10_mean:.3f} vs overall mean: {pgg['mean_importance']:.3f}")
print(f"PGG enrichment ratio: {top10_mean/pgg['mean_importance']:.2f}x")
print(f"PGG top-10 fraction: {100*top10_sum/total_pgg:.1f}% from 12.3% of atoms")

# Gallic acid: OH-bearing C atoms (5, 9, 7)
ga = ig["gallic_acid"]
oh_atoms = [a for a in ga["top_atoms"] if a["atom_idx"] in [5, 7, 9]]
oh_sum = sum(a["importance"] for a in oh_atoms)
total_ga = ga["mean_importance"] * ga["n_atoms"]
print(f"\nGallic acid OH-bearing C (5,7,9): {oh_sum:.3f} / {total_ga:.3f} = {100*oh_sum/total_ga:.1f}%")
