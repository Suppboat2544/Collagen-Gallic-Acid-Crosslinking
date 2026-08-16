#!/usr/bin/env python3
"""
Global SHAP-like Feature Attribution for Model D
=================================================
Runs Integrated Gradients on a large sample of the dataset and aggregates
atom-level, feature-level, and condition-level attributions to produce
global explanations analogous to SHAP summary plots.

Outputs (saved to Graph_model/results/figs/):

  1. global_feature_importance.png    — Mean |IG| per feature (bar chart)
  2. global_shap_beeswarm.png         — Beeswarm: feature value vs attribution
  3. global_shap_violin.png           — Violin: attribution distribution per feature
  4. global_element_importance.png     — Mean importance by atom element type
  5. global_ligand_comparison.png      — Per-ligand mean attribution heatmap
  6. global_feature_interaction.png    — Feature×feature interaction heatmap
  7. global_condition_sensitivity.png  — Condition (pH/temp/box) sensitivity
  8. global_attribution_dist.png       — Distribution of total attribution per sample
  9. global_topk_atoms.png             — Which atom positions matter most (positional)
 10. global_shap_dependence.png        — Dependence plots for top features
 11. model_d_global_shap.json          — Raw results for downstream use

Usage:
    cd /Users/suppboat/Jupyter_Dock
    source .venv/bin/activate
    python -m Graph_model.viz.run_global_shap [--n-samples 200] [--n-steps 30]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn

# ── Feature name catalogue ────────────────────────────────────────────────────
# 35 ligand node features (from level1_ligand.py)

LIGAND_FEATURE_NAMES = [
    # Element one-hot (0-10)
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Elem_UNK",
    # Hybridisation (11-15)
    "SP", "SP2", "SP3", "SP3D", "Hyb_OTHER",
    # Formal charge (16-21)
    "Charge_-2", "Charge_-1", "Charge_0", "Charge_+1", "Charge_+2", "Charge_OTHER",
    # H-count (22-27)
    "H0", "H1", "H2", "H3", "H4", "H_OTHER",
    # Single flags (28-33)
    "is_aromatic",
    "in_3ring", "in_5ring", "in_6ring",
    "chiral_CW", "chiral_CCW",
    # Mass (34)
    "mass_norm",
]

# Feature groups for higher-level summary
FEATURE_GROUPS = {
    "Element":       list(range(0, 11)),
    "Hybridisation": list(range(11, 16)),
    "Formal Charge": list(range(16, 22)),
    "H-count":       list(range(22, 28)),
    "Aromaticity":   [28],
    "Ring":          [29, 30, 31],
    "Chirality":     [32, 33],
    "Mass":          [34],
}

assert len(LIGAND_FEATURE_NAMES) == 35


# ── Matplotlib setup ──────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

_DPI = 200


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# ── Load Model D ──────────────────────────────────────────────────────────────

def load_model_d(ckpt_path: Path = Path("Graph_model/results/model_d_best.pt")):
    """Load trained Model D from checkpoint."""
    from Graph_model.model.option_d import OptionD
    from Graph_model.model.config import OptionDConfig

    model = OptionD(OptionDConfig())
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded Model D: best_epoch={ckpt.get('best_epoch')}, "
          f"best_val_rmse={ckpt.get('best_val_rmse', '?'):.4f}")
    return model


# ── Run IG on many samples ────────────────────────────────────────────────────

def run_global_ig(model, dataset, n_samples: int = 200, n_steps: int = 30):
    """
    Run Integrated Gradients on n_samples from the dataset.
    Returns list of dicts with per-sample attributions.
    """
    from Graph_model.interpret.integrated_gradients import integrated_gradients

    n = len(dataset)
    # Stratify: pick evenly spaced indices
    if n_samples >= n:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, n_samples, dtype=int).tolist()
        indices = sorted(set(indices))  # deduplicate

    print(f"  Running IG on {len(indices)} samples (n_steps={n_steps})...")

    results = []
    t0 = time.time()

    for count, idx in enumerate(indices):
        data = dataset[idx]

        try:
            attr = integrated_gradients(
                model=model,
                data=data,
                target_idx=0,   # collagen ΔG head
                n_steps=n_steps,
                internal_batch_size=min(10, n_steps),
            )

            # Get metadata
            lig_name = getattr(data, "ligand_name", f"sample_{idx}")
            ph = getattr(data, "ph", None)
            temp_c = getattr(data, "temp_c", None)
            box = getattr(data, "docking_box", None)
            receptor = getattr(data, "receptor", None)
            y = float(data.y.item()) if hasattr(data, "y") and data.y is not None else None

            # Extract raw per-feature attributions [N_atoms, 35]
            raw = attr["atom_attr_raw"].cpu().numpy()  # [N_atoms, 35]
            importance = attr["atom_importance"].cpu().numpy()  # [N_atoms]

            # Get ligand node features for SHAP-dependence
            x_feat = data["ligand"].x.cpu().numpy()  # [N_atoms, 35]

            results.append({
                "idx": int(idx),
                "ligand_name": str(lig_name),
                "ph": float(ph) if ph is not None else None,
                "temp_c": float(temp_c) if temp_c is not None else None,
                "docking_box": str(box) if box is not None else None,
                "receptor": str(receptor) if receptor is not None else None,
                "y": y,
                "n_atoms": raw.shape[0],
                "convergence_delta": float(attr["convergence_delta"]),
                "atom_attr_raw": raw,      # [N_atoms, 35]
                "atom_importance": importance,  # [N_atoms]
                "x_features": x_feat,       # [N_atoms, 35]
            })

        except Exception as e:
            print(f"    [WARN] Sample {idx} failed: {e}")
            continue

        if (count + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (count + 1) / elapsed
            eta = (len(indices) - count - 1) / rate
            print(f"    [{count+1}/{len(indices)}] {rate:.1f} samples/s, ETA {eta:.0f}s")

        # Aggressive cleanup
        del attr
        if count % 50 == 0:
            gc.collect()

    elapsed = time.time() - t0
    print(f"  Done: {len(results)} samples in {elapsed:.1f}s "
          f"({len(results)/max(elapsed,1):.1f} samples/s)")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1: Global Feature Importance (SHAP summary bar)
# ══════════════════════════════════════════════════════════════════════════════

def plot_global_feature_importance(results: list, out_dir: Path) -> Path:
    """Mean |IG attribution| per feature across all atoms and samples."""

    # Aggregate: for each feature, collect mean |attribution| per sample
    all_attr = np.vstack([r["atom_attr_raw"] for r in results])  # [total_atoms, 35]
    mean_abs = np.mean(np.abs(all_attr), axis=0)  # [35]

    # Sort descending
    order = np.argsort(-mean_abs)

    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(35)
    colors = plt.cm.RdYlBu_r(mean_abs[order] / max(mean_abs.max(), 1e-8))
    ax.barh(y, mean_abs[order], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([LIGAND_FEATURE_NAMES[i] for i in order], fontsize=9)
    ax.set_xlabel("Mean |IG Attribution|", fontsize=11)
    ax.set_title("Global Feature Importance (SHAP-style)\nModel D — Integrated Gradients",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values on top bars
    for i in range(min(10, 35)):
        val = mean_abs[order[i]]
        ax.text(val + mean_abs.max() * 0.01, i, f"{val:.4f}", fontsize=8, va="center")

    return _save(fig, out_dir / "global_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2: SHAP Beeswarm
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_beeswarm(results: list, out_dir: Path) -> Path:
    """SHAP-style beeswarm: color = feature value, x = attribution."""

    all_attr = np.vstack([r["atom_attr_raw"] for r in results])   # [A, 35]
    all_feat = np.vstack([r["x_features"] for r in results])      # [A, 35]

    # Subsample if too many atoms
    max_pts = 3000
    rng = np.random.default_rng(42)
    if all_attr.shape[0] > max_pts:
        idx_sub = rng.choice(all_attr.shape[0], max_pts, replace=False)
        all_attr = all_attr[idx_sub]
        all_feat = all_feat[idx_sub]

    # Sort features by mean |attr|
    mean_abs = np.mean(np.abs(all_attr), axis=0)
    order = np.argsort(-mean_abs)[:20]  # Top 20 features

    fig, ax = plt.subplots(figsize=(10, 10))

    for row_i, feat_i in enumerate(order):
        attrs = all_attr[:, feat_i]
        feats = all_feat[:, feat_i]

        # Normalise feature for color (0-1)
        fmin, fmax = feats.min(), feats.max()
        if fmax > fmin:
            feat_norm = (feats - fmin) / (fmax - fmin)
        else:
            feat_norm = np.zeros_like(feats)

        # Jitter y
        jitter = rng.normal(0, 0.15, len(attrs))
        y_vals = row_i + jitter

        scatter = ax.scatter(attrs, y_vals, c=feat_norm, cmap="RdYlBu_r",
                             s=8, alpha=0.5, edgecolors="none",
                             vmin=0, vmax=1)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LIGAND_FEATURE_NAMES[i] for i in order], fontsize=9)
    ax.set_xlabel("IG Attribution (impact on ΔG)", fontsize=11)
    ax.set_title("SHAP-style Beeswarm — Top 20 Features\nModel D — Integrated Gradients",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    cb = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label("Feature value (normalised)", fontsize=9)
    return _save(fig, out_dir / "global_shap_beeswarm.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3: SHAP Violin
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_violin(results: list, out_dir: Path) -> Path:
    """Violin plot of IG attribution distributions per feature (top 15)."""

    all_attr = np.vstack([r["atom_attr_raw"] for r in results])
    mean_abs = np.mean(np.abs(all_attr), axis=0)
    order = np.argsort(-mean_abs)[:15]

    fig, ax = plt.subplots(figsize=(10, 8))

    data_for_violin = [all_attr[:, i] for i in order]
    parts = ax.violinplot(data_for_violin, positions=range(len(order)),
                          showmeans=True, showmedians=True,
                          vert=False)

    # Colour violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(order)))
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.7)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LIGAND_FEATURE_NAMES[i] for i in order], fontsize=9)
    ax.set_xlabel("IG Attribution", fontsize=11)
    ax.set_title("Feature Attribution Distributions (Violin)\nModel D — Top 15 Features",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.15, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    return _save(fig, out_dir / "global_shap_violin.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Element-wise importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_element_importance(results: list, out_dir: Path) -> Path:
    """Bar chart: mean atom importance grouped by element type."""

    elem_names = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "UNK"]
    elem_importance = {e: [] for e in elem_names}

    for r in results:
        x = r["x_features"]       # [N, 35]
        imp = r["atom_importance"]  # [N]
        elem_idx = np.argmax(x[:, :11], axis=1)  # which element

        for atom_i in range(len(imp)):
            eidx = elem_idx[atom_i]
            if eidx < len(elem_names):
                elem_importance[elem_names[eidx]].append(imp[atom_i])

    # Compute stats
    elems_used = [(e, np.mean(v), np.std(v), len(v))
                  for e, v in elem_importance.items() if len(v) > 0]
    elems_used.sort(key=lambda x: -x[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    names = [e[0] for e in elems_used]
    means = [e[1] for e in elems_used]
    stds = [e[2] for e in elems_used]
    counts = [e[3] for e in elems_used]

    bars = ax.bar(range(len(names)), means, yerr=stds, color="#5C6BC0",
                  edgecolor="white", capsize=4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Atom Importance (IG)", fontsize=11)
    ax.set_title("Element-wise Atom Importance\nModel D — Integrated Gradients",
                 fontsize=13, fontweight="bold")

    # Annotate counts
    for i, (bar, c) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.005,
                f"n={c}", ha="center", fontsize=8, color="#666")

    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / "global_element_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Per-ligand comparison heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_ligand_feature_heatmap(results: list, out_dir: Path) -> Path:
    """Heatmap: ligand × feature group — mean |attribution|."""

    # Per-ligand: aggregate by feature group
    lig_data = {}
    for r in results:
        lig = r["ligand_name"]
        raw = r["atom_attr_raw"]  # [N, 35]
        if lig not in lig_data:
            lig_data[lig] = []
        lig_data[lig].append(np.mean(np.abs(raw), axis=0))  # [35]

    # Average across samples of same ligand
    lig_means = {}
    for lig, arrs in lig_data.items():
        lig_means[lig] = np.mean(arrs, axis=0)  # [35]

    ligands = sorted(lig_means.keys())
    groups = list(FEATURE_GROUPS.keys())

    mat = np.zeros((len(ligands), len(groups)))
    for i, lig in enumerate(ligands):
        feat_vec = lig_means[lig]
        for j, grp in enumerate(groups):
            idxs = FEATURE_GROUPS[grp]
            mat[i, j] = np.mean(feat_vec[idxs])

    fig, ax = plt.subplots(figsize=(10, max(4, len(ligands) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ligands)))
    ax.set_yticklabels([_short_lig(l) for l in ligands], fontsize=9)
    ax.set_title("Feature Group Attribution by Ligand\nModel D — Integrated Gradients",
                 fontsize=13, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Mean |IG Attribution|", fontsize=9)

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=7, color="white" if mat[i, j] > mat.max() * 0.6 else "black")

    return _save(fig, out_dir / "global_ligand_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Feature-feature interaction heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_feature_interaction(results: list, out_dir: Path) -> Path:
    """Correlation of |IG attributions| between feature pairs (group-level)."""

    all_attr = np.vstack([r["atom_attr_raw"] for r in results])  # [A, 35]
    abs_attr = np.abs(all_attr)

    groups = list(FEATURE_GROUPS.keys())
    n_groups = len(groups)
    group_attr = np.zeros((abs_attr.shape[0], n_groups))
    for j, grp in enumerate(groups):
        idxs = FEATURE_GROUPS[grp]
        group_attr[:, j] = abs_attr[:, idxs].mean(axis=1)

    # Correlation
    corr = np.corrcoef(group_attr.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(groups, fontsize=9)

    for i in range(n_groups):
        for j in range(n_groups):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else "black")

    ax.set_title("Feature Group Attribution Correlation\nModel D",
                 fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Pearson r", fontsize=9)
    return _save(fig, out_dir / "global_feature_interaction.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Condition sensitivity
# ══════════════════════════════════════════════════════════════════════════════

def plot_condition_sensitivity(results: list, out_dir: Path) -> Path:
    """How does mean |attribution| vary with pH, temperature, receptor type."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Collect per-sample total importance
    sample_data = []
    for r in results:
        total_imp = float(np.mean(r["atom_importance"]))
        sample_data.append({
            "ph": r["ph"],
            "temp_c": r["temp_c"],
            "receptor": r["receptor"],
            "box": r["docking_box"],
            "mean_importance": total_imp,
            "y": r["y"],
        })

    # pH
    ax = axes[0]
    phs = sorted(set(s["ph"] for s in sample_data if s["ph"] is not None))
    ph_groups = {p: [s["mean_importance"] for s in sample_data if s["ph"] == p] for p in phs}

    if len(ph_groups) > 1:
        box_data = [ph_groups[p] for p in phs]
        bp = ax.boxplot(box_data, labels=[f"pH {p}" for p in phs], patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(phs)))):
            patch.set_facecolor(color)
    else:
        ax.text(0.5, 0.5, "Single pH value", transform=ax.transAxes, ha="center")

    ax.set_ylabel("Mean Atom Importance", fontsize=10)
    ax.set_title("pH Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    # Temperature
    ax = axes[1]
    temps = sorted(set(s["temp_c"] for s in sample_data if s["temp_c"] is not None))
    if len(temps) > 3:
        # Bin temperatures
        temp_vals = [s["temp_c"] for s in sample_data if s["temp_c"] is not None]
        imp_vals = [s["mean_importance"] for s in sample_data if s["temp_c"] is not None]
        ax.scatter(temp_vals, imp_vals, alpha=0.4, s=20, color="#F44336")
        # Trend line
        if len(temp_vals) > 5:
            z = np.polyfit(temp_vals, imp_vals, 1)
            x_line = np.linspace(min(temp_vals), max(temp_vals), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Temperature (°C)", fontsize=10)
    else:
        temp_groups = {t: [s["mean_importance"] for s in sample_data if s["temp_c"] == t] for t in temps}
        box_data = [temp_groups[t] for t in temps]
        bp = ax.boxplot(box_data, labels=[f"{t}°C" for t in temps], patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(temps)))):
            patch.set_facecolor(color)

    ax.set_ylabel("Mean Atom Importance", fontsize=10)
    ax.set_title("Temperature Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    # Receptor type
    ax = axes[2]
    receptors = sorted(set(s["receptor"] for s in sample_data if s["receptor"] is not None))
    if len(receptors) > 1:
        rec_groups = {rc: [s["mean_importance"] for s in sample_data if s["receptor"] == rc] for rc in receptors}
        box_data = [rec_groups[rc] for rc in receptors]
        bp = ax.boxplot(box_data, labels=[_short_lig(rc) for rc in receptors], patch_artist=True)
        for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(receptors)))):
            patch.set_facecolor(color)
    else:
        ax.text(0.5, 0.5, f"Single receptor: {receptors[0] if receptors else '?'}",
                transform=ax.transAxes, ha="center")

    ax.set_ylabel("Mean Atom Importance", fontsize=10)
    ax.set_title("Receptor Sensitivity", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")

    fig.suptitle("Condition Sensitivity — Attribution vs Experimental Conditions\nModel D",
                 fontsize=13, fontweight="bold", y=1.04)
    return _save(fig, out_dir / "global_condition_sensitivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Attribution distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_attribution_distribution(results: list, out_dir: Path) -> Path:
    """Distribution of per-sample total |attribution| and convergence delta."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Total attribution per sample
    total_attrs = [np.sum(np.abs(r["atom_attr_raw"])) for r in results]
    ax = axes[0]
    ax.hist(total_attrs, bins=30, color="#5C6BC0", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(total_attrs), color="red", linewidth=2, linestyle="--",
               label=f"mean={np.mean(total_attrs):.3f}")
    ax.set_xlabel("Total |IG Attribution|", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Total Attribution Distribution", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Convergence delta distribution
    deltas = [abs(r["convergence_delta"]) for r in results]
    ax = axes[1]
    ax.hist(deltas, bins=30, color="#4CAF50", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(deltas), color="red", linewidth=2, linestyle="--",
               label=f"mean={np.mean(deltas):.3f}")
    ax.set_xlabel("|Convergence Δ|", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("IG Convergence Check", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Attribution vs prediction
    ys = [r["y"] for r in results if r["y"] is not None]
    ta = [np.sum(np.abs(r["atom_attr_raw"])) for r in results if r["y"] is not None]
    ax = axes[2]
    if ys:
        ax.scatter(ys, ta, alpha=0.5, s=20, color="#FF9800")
        ax.set_xlabel("True ΔG (kcal/mol)", fontsize=10)
        ax.set_ylabel("Total |IG Attribution|", fontsize=10)
        ax.set_title("Attribution vs Target", fontsize=11, fontweight="bold")
        # Trend
        if len(ys) > 10:
            z = np.polyfit(ys, ta, 1)
            x_line = np.linspace(min(ys), max(ys), 100)
            ax.plot(x_line, np.polyval(z, x_line), "k--", linewidth=1.5, alpha=0.5)

    ax.grid(True, alpha=0.2, linestyle="--")
    fig.suptitle("Attribution Quality Diagnostics\nModel D — Integrated Gradients",
                 fontsize=13, fontweight="bold", y=1.04)
    return _save(fig, out_dir / "global_attribution_dist.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 9: Positional atom importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_positional_importance(results: list, out_dir: Path) -> Path:
    """Which atom *positions* (0, 1, 2, ...) tend to be most important?"""

    max_atoms = max(r["n_atoms"] for r in results)
    max_pos = min(max_atoms, 50)  # Cap at 50

    # Aggregate importance by position
    pos_data = [[] for _ in range(max_pos)]
    for r in results:
        imp = r["atom_importance"]
        for pos in range(min(len(imp), max_pos)):
            pos_data[pos].append(imp[pos])

    means = [np.mean(d) if d else 0 for d in pos_data]
    stds = [np.std(d) if d else 0 for d in pos_data]
    counts = [len(d) for d in pos_data]

    fig, ax = plt.subplots(figsize=(14, 5))

    x = range(max_pos)
    ax.bar(x, means, yerr=stds, color="#2196F3", edgecolor="white",
           linewidth=0.3, capsize=1, alpha=0.8)

    ax.set_xlabel("Atom Position Index", fontsize=11)
    ax.set_ylabel("Mean Importance ± σ", fontsize=11)
    ax.set_title("Positional Atom Importance (averaged across molecules)\nModel D",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / "global_topk_atoms.png")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 10: SHAP Dependence plots (top 4 features)
# ══════════════════════════════════════════════════════════════════════════════

def plot_shap_dependence(results: list, out_dir: Path) -> Path:
    """SHAP-style dependence plot for top-4 important features."""

    all_attr = np.vstack([r["atom_attr_raw"] for r in results])
    all_feat = np.vstack([r["x_features"] for r in results])

    # Subsample
    max_pts = 5000
    rng = np.random.default_rng(42)
    if all_attr.shape[0] > max_pts:
        idx_sub = rng.choice(all_attr.shape[0], max_pts, replace=False)
        all_attr = all_attr[idx_sub]
        all_feat = all_feat[idx_sub]

    mean_abs = np.mean(np.abs(all_attr), axis=0)
    top4 = np.argsort(-mean_abs)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, feat_i in zip(axes.flat, top4):
        x = all_feat[:, feat_i]
        y = all_attr[:, feat_i]

        # Find strongest interacting feature
        other_feats = [j for j in np.argsort(-mean_abs) if j != feat_i]
        interact_feat = other_feats[0]
        c = all_feat[:, interact_feat]

        # Normalise color
        cmin, cmax = c.min(), c.max()
        if cmax > cmin:
            c_norm = (c - cmin) / (cmax - cmin)
        else:
            c_norm = np.zeros_like(c)

        scatter = ax.scatter(x, y, c=c_norm, cmap="RdYlBu_r", s=6, alpha=0.5,
                             edgecolors="none", vmin=0, vmax=1)

        ax.set_xlabel(f"{LIGAND_FEATURE_NAMES[feat_i]} (value)", fontsize=10)
        ax.set_ylabel(f"IG Attribution", fontsize=10)
        ax.set_title(f"{LIGAND_FEATURE_NAMES[feat_i]}\n(colour = {LIGAND_FEATURE_NAMES[interact_feat]})",
                     fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.15, linestyle="--")

        cb = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(LIGAND_FEATURE_NAMES[interact_feat], fontsize=8)

    fig.suptitle("SHAP Dependence Plots — Top 4 Features\nModel D",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, out_dir / "global_shap_dependence.png")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE JSON summary
# ══════════════════════════════════════════════════════════════════════════════

def save_json_summary(results: list, out_dir: Path) -> Path:
    """Save aggregated results as JSON for downstream use."""

    all_attr = np.vstack([r["atom_attr_raw"] for r in results])
    mean_abs = np.mean(np.abs(all_attr), axis=0)

    # Feature ranking
    order = np.argsort(-mean_abs)
    feature_ranking = [
        {"rank": i + 1, "feature": LIGAND_FEATURE_NAMES[j],
         "mean_abs_attr": float(mean_abs[j]), "idx": int(j)}
        for i, j in enumerate(order)
    ]

    # Feature group ranking
    group_ranking = []
    for grp, idxs in FEATURE_GROUPS.items():
        group_ranking.append({
            "group": grp,
            "mean_abs_attr": float(np.mean(mean_abs[idxs])),
            "features": [LIGAND_FEATURE_NAMES[i] for i in idxs],
        })
    group_ranking.sort(key=lambda x: -x["mean_abs_attr"])

    # Per-ligand summary
    lig_summary = {}
    for r in results:
        lig = r["ligand_name"]
        if lig not in lig_summary:
            lig_summary[lig] = {"n_samples": 0, "mean_importance": [], "y_values": []}
        lig_summary[lig]["n_samples"] += 1
        lig_summary[lig]["mean_importance"].append(float(np.mean(r["atom_importance"])))
        if r["y"] is not None:
            lig_summary[lig]["y_values"].append(r["y"])

    for lig, v in lig_summary.items():
        v["avg_importance"] = float(np.mean(v["mean_importance"]))
        v["avg_y"] = float(np.mean(v["y_values"])) if v["y_values"] else None
        del v["mean_importance"]

    summary = {
        "model": "Option D — Multi-Task Selectivity GNN",
        "n_samples": len(results),
        "total_atoms_analysed": int(all_attr.shape[0]),
        "mean_convergence_delta": float(np.mean([abs(r["convergence_delta"]) for r in results])),
        "feature_ranking": feature_ranking,
        "feature_group_ranking": group_ranking,
        "per_ligand": lig_summary,
    }

    path = out_dir.parent / "model_d_global_shap.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    -> {path.name}")
    return path


# ══════════════════════════════════════════════════════════════════════════════

def _short_lig(name: str) -> str:
    if len(name) > 25:
        return name[:22] + "..."
    return name


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Global SHAP-like analysis for Model D")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of dataset samples to run IG on (default: 200)")
    parser.add_argument("--n-steps", type=int, default=30,
                        help="IG interpolation steps (default: 30)")
    parser.add_argument("--results-dir", type=str, default="Graph_model/results")
    args = parser.parse_args()

    # Work from workspace root
    WORKSPACE = Path(__file__).resolve().parent.parent.parent
    os.chdir(WORKSPACE)

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "figs"

    print("=" * 60)
    print("  Global SHAP-like Analysis — Model D")
    print("  (Integrated Gradients over full dataset)")
    print(f"  Working dir: {WORKSPACE}")
    print("=" * 60)

    # ── 1. Load dataset ──────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")

    from Graph_model.data.hetero_dataset import HeteroDockingDataset

    ds = HeteroDockingDataset(
        include_mmp1=True,
        include_bipartite=True,
        use_cache=True,
        force_reload=False,
    )
    ds.load(verbose=True)
    print(f"  Dataset: {len(ds)} graphs")

    # ── 2. Load Model D ──────────────────────────────────────────────
    print("\n[2/4] Loading Model D checkpoint...")
    model = load_model_d(results_dir / "model_d_best.pt")

    # ── 3. Run global IG ─────────────────────────────────────────────
    print(f"\n[3/4] Running Integrated Gradients (n={args.n_samples}, steps={args.n_steps})...")
    ig_results = run_global_ig(model, ds, n_samples=args.n_samples, n_steps=args.n_steps)

    if not ig_results:
        print("ERROR: No IG results produced!")
        return

    # ── 4. Generate plots ────────────────────────────────────────────
    print(f"\n[4/4] Generating {10} SHAP-style plots + JSON summary...")

    paths = []
    print("\n  [1/11] Global feature importance bar...")
    paths.append(plot_global_feature_importance(ig_results, out_dir))
    print("  [2/11] SHAP beeswarm...")
    paths.append(plot_shap_beeswarm(ig_results, out_dir))
    print("  [3/11] SHAP violin...")
    paths.append(plot_shap_violin(ig_results, out_dir))
    print("  [4/11] Element importance...")
    paths.append(plot_element_importance(ig_results, out_dir))
    print("  [5/11] Ligand-feature heatmap...")
    paths.append(plot_ligand_feature_heatmap(ig_results, out_dir))
    print("  [6/11] Feature interaction...")
    paths.append(plot_feature_interaction(ig_results, out_dir))
    print("  [7/11] Condition sensitivity...")
    paths.append(plot_condition_sensitivity(ig_results, out_dir))
    print("  [8/11] Attribution distribution...")
    paths.append(plot_attribution_distribution(ig_results, out_dir))
    print("  [9/11] Positional importance...")
    paths.append(plot_positional_importance(ig_results, out_dir))
    print("  [10/11] SHAP dependence plots...")
    paths.append(plot_shap_dependence(ig_results, out_dir))
    print("  [11/11] JSON summary...")
    paths.append(save_json_summary(ig_results, out_dir))

    print(f"\n{'='*60}")
    print(f"  Complete: {len(paths)} outputs generated")
    print(f"  Output dir: {out_dir}")
    print(f"{'='*60}")

    # Quick textual summary
    all_attr = np.vstack([r["atom_attr_raw"] for r in ig_results])
    mean_abs = np.mean(np.abs(all_attr), axis=0)
    order = np.argsort(-mean_abs)

    print("\n  TOP 10 GLOBAL FEATURES (by mean |IG|):")
    for i in range(10):
        j = order[i]
        print(f"    #{i+1}: {LIGAND_FEATURE_NAMES[j]:15s}  mean|IG|={mean_abs[j]:.5f}")


if __name__ == "__main__":
    main()
