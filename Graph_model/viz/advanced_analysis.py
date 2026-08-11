"""
Graph_model.viz.advanced_analysis
=====================================
In-depth model analysis and SHAP-style visualisations.

Produces publication-quality figures:
  1.  Radar chart — multi-metric comparison across all models
  2.  Convergence heatmap — epoch-by-epoch RMSE for every model
  3.  Train-vs-Val gap (overfitting) panel
  4.  Efficiency frontier — accuracy vs compute cost
  5.  Learning rate sensitivity ribbon
  6.  Model ranking table (LaTeX-ready)
  7.  Integrated Gradients atom importance (SHAP-style beeswarm)
  8.  Integrated Gradients waterfall per ligand
  9.  Grad-CAM heatmap per ligand
  10. Feature attribution summary bar chart
  11. Correlation matrix: inter-model val-RMSE trajectories
  12. Epoch-wise metric evolution (small multiples)

Usage
-----
    python -m Graph_model.viz.advanced_analysis [--results-dir Graph_model/results]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    raise SystemExit("matplotlib is required: pip install matplotlib")

# ── Constants ─────────────────────────────────────────────────────────────────

_MODEL_COLORS = {
    "A": "#2196F3", "B": "#4CAF50", "C": "#FF9800", "D": "#9C27B0",
    "E": "#F44336", "F": "#00BCD4", "G": "#795548", "H": "#607D8B",
    "I": "#E91E63",
}

_MODEL_SHORT = {
    "A": "GATv2", "B": "CrossAttn", "C": "FragMPNN", "D": "MultiTask",
    "E": "pGET", "F": "DimeNet++", "G": "EGNN", "H": "GGNN-Seq",
    "I": "Graphormer",
}

_DPI = 200
_FONT_TITLE = 13
_FONT_LABEL = 11


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {path}")
    return path


def _safe(val):
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_all(results_dir: Path):
    """Return {key: (name, epochs_list, summary_dict, meta_dict)}."""
    out = {}
    for key in "ABCDEFGHI":
        p = results_dir / f"option_{key.lower()}_training.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        out[key] = (
            d.get("model_name", f"Option {key}"),
            d.get("epochs", []),
            d.get("summary", {}),
            d.get("meta", {}),
        )
    return out


def _load_interpretation(results_dir: Path) -> dict | None:
    p = results_dir / "model_d_interpretation.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# 1. RADAR CHART — multi-metric comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_radar(results_dir: Path, out_dir: Path) -> Path:
    """Radar/spider chart comparing best metrics across all models."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())

    # Metrics: lower-is-better → invert so higher = better on radar
    metrics = ["RMSE", "MAE", "Pearson r", "Spearman ρ", "Convergence\nSpeed"]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for key in keys:
        name, epochs, summary, meta = data[key]
        fv = summary.get("final_val_metrics", {})
        best_rmse = _safe(summary.get("best_val_rmse", float("nan")))
        best_mae = min((_safe(e.get("val_mae", 99)) for e in epochs), default=float("nan"))
        best_pr = max((_safe(e.get("val_pearson_r", -1)) for e in epochs), default=float("nan"))
        best_sr = max((_safe(e.get("val_spearman_r", -1)) for e in epochs), default=float("nan"))
        best_ep = _safe(summary.get("best_epoch", len(epochs)))
        total_ep = len(epochs)

        # Normalise: RMSE/MAE inverted (lower→higher), correlations as-is, speed = 1 - best_ep/total_ep
        vals = [
            max(0, 1.0 - (best_rmse - 1.0) / 1.0),   # map [1.0, 2.0] → [1.0, 0.0]
            max(0, 1.0 - (best_mae - 0.5) / 1.0),     # map [0.5, 1.5] → [1.0, 0.0]
            max(0, best_pr),
            max(0, best_sr),
            max(0, 1.0 - best_ep / max(total_ep, 1)),  # faster convergence = higher
        ]
        vals += vals[:1]

        color = _MODEL_COLORS.get(key, "#888")
        ax.fill(angles, vals, alpha=0.08, color=color)
        ax.plot(angles, vals, linewidth=2, label=f"{key}: {_MODEL_SHORT.get(key, key)}", color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Model Comparison", fontsize=_FONT_TITLE, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    return _save(fig, out_dir / "radar_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONVERGENCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_heatmap(results_dir: Path, out_dir: Path) -> Path:
    """Heatmap: epoch × model colour-coded by val RMSE."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())
    max_ep = max(len(data[k][1]) for k in keys)

    # Build matrix (models × epochs), pad with NaN
    mat = np.full((len(keys), max_ep), np.nan)
    for i, k in enumerate(keys):
        eps = data[k][1]
        for j, e in enumerate(eps):
            mat[i, j] = _safe(e.get("val_rmse", float("nan")))

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = LinearSegmentedColormap.from_list("rmse", ["#1a9850", "#fee08b", "#d73027"])
    vmin = np.nanmin(mat)
    vmax = min(np.nanmax(mat), 3.0)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k}: {_MODEL_SHORT[k]}" for k in keys], fontsize=10)
    ax.set_xlabel("Epoch", fontsize=_FONT_LABEL)
    ax.set_title("Validation RMSE Convergence Heatmap", fontsize=_FONT_TITLE, fontweight="bold")

    # Mark best epoch for each model
    for i, k in enumerate(keys):
        best_ep = data[k][2].get("best_epoch", 0)
        ax.plot(best_ep, i, marker="*", color="white", markersize=12, markeredgecolor="black", markeredgewidth=0.5)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Validation RMSE (kcal/mol)", fontsize=10)
    return _save(fig, out_dir / "convergence_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. OVERFITTING GAP — train vs val loss
# ══════════════════════════════════════════════════════════════════════════════

def plot_overfitting_gap(results_dir: Path, out_dir: Path) -> Path:
    """Train–Val RMSE gap over epochs for each model (2×5 small multiples)."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())
    n = len(keys)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        name, epochs, summary, _ = data[key]

        xs = [e["epoch"] for e in epochs]
        train_r = [_safe(e.get("train_rmse", float("nan"))) for e in epochs]
        val_r = [_safe(e.get("val_rmse", float("nan"))) for e in epochs]

        ax.plot(xs, train_r, color="#2196F3", linewidth=1.5, label="Train", alpha=0.8)
        ax.plot(xs, val_r, color="#F44336", linewidth=1.5, label="Val", alpha=0.8)

        # Shade the gap
        ax.fill_between(xs, train_r, val_r, alpha=0.15, color="#FF9800")

        best_ep = summary.get("best_epoch", 0)
        ax.axvline(best_ep, color="#4CAF50", linestyle="--", alpha=0.6, linewidth=1)

        ax.set_title(f"{key}: {_MODEL_SHORT[key]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("RMSE", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Train–Val RMSE Gap (Overfitting Analysis)", fontsize=_FONT_TITLE, fontweight="bold", y=1.02)
    return _save(fig, out_dir / "overfitting_gap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. EFFICIENCY FRONTIER — accuracy vs compute
# ══════════════════════════════════════════════════════════════════════════════

def plot_efficiency_frontier(results_dir: Path, out_dir: Path) -> Path:
    """Scatter: best val RMSE vs training wall time. Pareto-optimal frontier."""
    data = _load_all(results_dir)
    comp_path = results_dir / "comparison_summary.json"
    wall_times = {}
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        for k, v in comp.get("models", {}).items():
            wall_times[k] = _safe(v.get("wall_time_s", 0))

    keys = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(9, 6))

    xs, ys, labels = [], [], []
    for key in keys:
        name, epochs, summary, _ = data[key]
        rmse = _safe(summary.get("best_val_rmse", float("nan")))
        wt = wall_times.get(key, _safe(summary.get("wall_time_s", 0)))
        wt_hr = wt / 3600
        xs.append(wt_hr)
        ys.append(rmse)
        labels.append(key)

        color = _MODEL_COLORS.get(key, "#888")
        ax.scatter(wt_hr, rmse, color=color, s=150, zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(f"{key}: {_MODEL_SHORT[key]}",
                    (wt_hr, rmse), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold", color=color)

    # Pareto frontier (lower RMSE + lower time = better)
    pts = sorted(zip(xs, ys, labels))
    pareto_x, pareto_y = [], []
    best_y = float("inf")
    for x, y, _ in pts:
        if y < best_y:
            pareto_x.append(x)
            pareto_y.append(y)
            best_y = y
    if len(pareto_x) > 1:
        ax.plot(pareto_x, pareto_y, "k--", alpha=0.4, linewidth=1.5, label="Pareto frontier")

    ax.set_xlabel("Training Time (hours)", fontsize=_FONT_LABEL)
    ax.set_ylabel("Best Validation RMSE (kcal/mol)", fontsize=_FONT_LABEL)
    ax.set_title("Efficiency Frontier — Accuracy vs Compute Cost", fontsize=_FONT_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if pareto_x:
        ax.legend(fontsize=10)
    return _save(fig, out_dir / "efficiency_frontier.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LEARNING RATE vs LOSS RIBBON
# ══════════════════════════════════════════════════════════════════════════════

def plot_lr_loss_ribbon(results_dir: Path, out_dir: Path) -> Path:
    """Dual-axis: LR schedule + val loss for each model."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())
    n = len(keys)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, key in enumerate(keys):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        name, epochs, summary, _ = data[key]

        xs = [e["epoch"] for e in epochs]
        lrs = [_safe(e.get("learning_rate", 1e-3)) for e in epochs]
        vl = [_safe(e.get("val_loss", float("nan"))) for e in epochs]

        ax.plot(xs, vl, color="#F44336", linewidth=1.5, label="Val Loss")
        ax.set_ylabel("Val Loss (MSE)", fontsize=8, color="#F44336")
        ax.tick_params(axis="y", labelcolor="#F44336", labelsize=7)

        ax2 = ax.twinx()
        ax2.plot(xs, lrs, color="#2196F3", linewidth=1, alpha=0.7, label="LR")
        ax2.set_ylabel("LR", fontsize=8, color="#2196F3")
        ax2.tick_params(axis="y", labelcolor="#2196F3", labelsize=7)
        ax2.set_yscale("log")

        ax.set_title(f"{key}: {_MODEL_SHORT[key]}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Learning Rate Schedule vs Validation Loss", fontsize=_FONT_TITLE, fontweight="bold", y=1.02)
    return _save(fig, out_dir / "lr_loss_ribbon.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MODEL RANKING TABLE (image)
# ══════════════════════════════════════════════════════════════════════════════

def plot_ranking_table(results_dir: Path, out_dir: Path) -> Path:
    """Publication-ready model ranking table as a figure."""
    data = _load_all(results_dir)
    comp_path = results_dir / "comparison_summary.json"
    wall_times = {}
    if comp_path.exists():
        with open(comp_path) as f:
            comp = json.load(f)
        for k, v in comp.get("models", {}).items():
            wall_times[k] = _safe(v.get("wall_time_s", 0))

    rows = []
    for key in sorted(data.keys()):
        name, epochs, summary, meta = data[key]
        best_rmse = _safe(summary.get("best_val_rmse", float("nan")))
        best_mae = min((_safe(e.get("val_mae", 99)) for e in epochs), default=float("nan"))
        best_pr = max((_safe(e.get("val_pearson_r", -1)) for e in epochs), default=float("nan"))
        best_sr = max((_safe(e.get("val_spearman_r", -1)) for e in epochs), default=float("nan"))
        best_ep = summary.get("best_epoch", "?")
        n_ep = len(epochs)
        wt = wall_times.get(key, _safe(summary.get("wall_time_s", 0)))
        n_train = meta.get("n_train", "?")
        n_val = meta.get("n_val", "?")
        rows.append([
            f"{key}", _MODEL_SHORT[key], f"{n_train}", f"{n_val}",
            f"{best_rmse:.4f}", f"{best_mae:.4f}", f"{best_pr:.4f}", f"{best_sr:.4f}",
            f"{best_ep}", f"{n_ep}", f"{wt/3600:.1f}h",
        ])

    # Sort by RMSE
    rows.sort(key=lambda r: float(r[4]))

    # Add rank
    for i, row in enumerate(rows):
        row.insert(0, f"#{i+1}")

    cols = ["Rank", "Key", "Architecture", "N_train", "N_val",
            "RMSE↓", "MAE↓", "Pearson r↑", "Spearman ρ↑",
            "Best Ep", "Total Ep", "Time"]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")
    ax.set_title("Model Performance Ranking", fontsize=14, fontweight="bold", pad=15)

    table = ax.table(
        cellText=rows, colLabels=cols, loc="center",
        cellLoc="center", colColours=["#e8eaf6"] * len(cols),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Highlight best (first row)
    for j in range(len(cols)):
        table[(1, j)].set_facecolor("#c8e6c9")
        table[(1, j)].set_text_props(fontweight="bold")

    return _save(fig, out_dir / "ranking_table.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP-STYLE BEESWARM — Integrated Gradients atom importance
# ══════════════════════════════════════════════════════════════════════════════

def plot_ig_beeswarm(results_dir: Path, out_dir: Path) -> Path:
    """SHAP-style horizontal beeswarm of IG atom importances per ligand."""
    interp = _load_interpretation(results_dir)
    if interp is None:
        print("  [SKIP] No model_d_interpretation.json found")
        return out_dir / "ig_beeswarm_SKIPPED.png"

    ig_data = interp.get("integrated_gradients", {})
    if not ig_data:
        print("  [SKIP] No integrated_gradients data")
        return out_dir / "ig_beeswarm_SKIPPED.png"

    ligands = sorted(ig_data.keys())

    fig, ax = plt.subplots(figsize=(10, max(4, len(ligands) * 0.8)))

    y_positions = []
    for i, lig in enumerate(ligands):
        entry = ig_data[lig]
        atoms = entry.get("top_atoms", [])
        importances = [a["importance"] for a in atoms]
        idxs = [a["atom_idx"] for a in atoms]

        # Beeswarm: jitter y around i
        n = len(importances)
        jitter = np.random.default_rng(42).normal(0, 0.12, n)
        y_vals = np.full(n, i) + jitter

        # Colour by importance magnitude
        imp_arr = np.array(importances)
        scatter = ax.scatter(imp_arr, y_vals, c=imp_arr, cmap="RdYlBu_r",
                             s=40, alpha=0.8, edgecolors="white", linewidth=0.5,
                             vmin=0, vmax=max(imp_arr.max(), 0.1) if len(imp_arr) > 0 else 0.1)

        y_positions.append(i)

    ax.set_yticks(range(len(ligands)))
    ax.set_yticklabels([_pretty_ligand(l) for l in ligands], fontsize=10)
    ax.set_xlabel("Atom Importance (IG Attribution)", fontsize=_FONT_LABEL)
    ax.set_title("Integrated Gradients — Atom Importance by Ligand\n(SHAP-style beeswarm)",
                 fontsize=_FONT_TITLE, fontweight="bold")
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    cb = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Attribution magnitude", fontsize=9)
    return _save(fig, out_dir / "ig_beeswarm.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. WATERFALL — IG per-atom breakdown for top ligands
# ══════════════════════════════════════════════════════════════════════════════

def plot_ig_waterfall(results_dir: Path, out_dir: Path) -> Path:
    """Waterfall chart of atom-level IG attributions for top 4 ligands."""
    interp = _load_interpretation(results_dir)
    if interp is None:
        print("  [SKIP] No interpretation data")
        return out_dir / "ig_waterfall_SKIPPED.png"

    ig_data = interp.get("integrated_gradients", {})
    if not ig_data:
        print("  [SKIP] No IG data")
        return out_dir / "ig_waterfall_SKIPPED.png"

    # Pick top 4 ligands by mean importance
    ranked = sorted(ig_data.items(), key=lambda kv: kv[1].get("mean_importance", 0), reverse=True)[:4]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (lig, entry) in enumerate(ranked):
        ax = axes[idx // 2][idx % 2]
        atoms = entry.get("top_atoms", [])
        if not atoms:
            ax.set_visible(False)
            continue

        importances = [a["importance"] for a in atoms]
        labels = [f"Atom {a['atom_idx']}" for a in atoms]

        # Sort by importance descending
        order = np.argsort(importances)[::-1]
        importances = [importances[i] for i in order]
        labels = [labels[i] for i in order]

        colors = ["#e53935" if v > 0 else "#1e88e5" for v in importances]

        # Waterfall: cumulative
        cumulative = np.cumsum(importances)
        starts = np.concatenate([[0], cumulative[:-1]])

        ax.barh(range(len(importances)), importances, left=starts, color=colors,
                edgecolor="white", linewidth=0.5, height=0.7)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Cumulative Attribution", fontsize=9)
        ax.set_title(f"{_pretty_ligand(lig)}\nmean={entry.get('mean_importance', 0):.3f}, "
                     f"Δconv={entry.get('convergence_delta', 0):.4f}",
                     fontsize=10, fontweight="bold")
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.grid(True, axis="x", alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.invert_yaxis()

    fig.suptitle("Integrated Gradients — Atom Attribution Waterfall",
                 fontsize=_FONT_TITLE, fontweight="bold", y=1.02)
    return _save(fig, out_dir / "ig_waterfall.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAD-CAM HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradcam_heatmap(results_dir: Path, out_dir: Path) -> Path:
    """Heatmap of Grad-CAM scores per ligand (ligands × atom index)."""
    interp = _load_interpretation(results_dir)
    if interp is None:
        print("  [SKIP] No interpretation data")
        return out_dir / "gradcam_heatmap_SKIPPED.png"

    gc_data = interp.get("gradcam", {})
    if not gc_data:
        print("  [SKIP] No Grad-CAM data")
        return out_dir / "gradcam_heatmap_SKIPPED.png"

    ligands = sorted(gc_data.keys())

    # Build matrix: ligand × atom (padded)
    max_atoms = max(gc_data[l].get("n_atoms", 0) for l in ligands)
    if max_atoms == 0:
        max_atoms = max(max((a["atom_idx"] for a in gc_data[l].get("top_atoms", [])), default=0) for l in ligands) + 1

    mat = np.zeros((len(ligands), max_atoms))
    for i, lig in enumerate(ligands):
        atoms = gc_data[lig].get("top_atoms", [])
        for a in atoms:
            idx = a["atom_idx"]
            if idx < max_atoms:
                mat[i, idx] = a.get("importance", a.get("cam_score", 0))

    fig, ax = plt.subplots(figsize=(12, max(3, len(ligands) * 0.8)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(len(ligands)))
    ax.set_yticklabels([_pretty_ligand(l) for l in ligands], fontsize=10)
    ax.set_xlabel("Atom Index", fontsize=_FONT_LABEL)
    ax.set_title("Grad-CAM Atom Attribution Heatmap (Model D)",
                 fontsize=_FONT_TITLE, fontweight="bold")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("CAM Score", fontsize=9)
    return _save(fig, out_dir / "gradcam_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. FEATURE ATTRIBUTION SUMMARY — bar chart per ligand
# ══════════════════════════════════════════════════════════════════════════════

def plot_attribution_summary(results_dir: Path, out_dir: Path) -> Path:
    """Bar chart: mean IG importance per ligand (ranked)."""
    interp = _load_interpretation(results_dir)
    if interp is None:
        print("  [SKIP] No interpretation data")
        return out_dir / "attribution_summary_SKIPPED.png"

    ig_data = interp.get("integrated_gradients", {})
    if not ig_data:
        print("  [SKIP] No IG data")
        return out_dir / "attribution_summary_SKIPPED.png"

    # Sort by mean importance
    items = sorted(ig_data.items(), key=lambda kv: kv[1].get("mean_importance", 0), reverse=True)
    ligands = [_pretty_ligand(k) for k, _ in items]
    means = [v.get("mean_importance", 0) for _, v in items]
    maxs = [v.get("max_importance", 0) for _, v in items]
    stds = [v.get("std_importance", 0) for _, v in items]
    n_atoms = [v.get("n_atoms", 0) for _, v in items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(ligands) * 0.6)))

    y = np.arange(len(ligands))
    bars = ax.barh(y, means, xerr=stds, color="#5C6BC0", edgecolor="white",
                   linewidth=0.5, height=0.6, capsize=3, alpha=0.85)

    # Annotate with n_atoms
    for i, (m, na) in enumerate(zip(means, n_atoms)):
        ax.text(m + stds[i] + 0.01, i, f"({na} atoms)", fontsize=8, va="center", color="#666")

    ax.set_yticks(y)
    ax.set_yticklabels(ligands, fontsize=10)
    ax.set_xlabel("Mean IG Atom Importance ± σ", fontsize=_FONT_LABEL)
    ax.set_title("Feature Attribution Summary — Integrated Gradients (Model D)",
                 fontsize=_FONT_TITLE, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    return _save(fig, out_dir / "attribution_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# 11. INTER-MODEL CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_model_correlation(results_dir: Path, out_dir: Path) -> Path:
    """Correlation matrix of val RMSE trajectories across models."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())

    # Align to same epoch range
    min_ep = min(len(data[k][1]) for k in keys)
    series = {}
    for k in keys:
        eps = data[k][1][:min_ep]
        series[k] = np.array([_safe(e.get("val_rmse", float("nan"))) for e in eps])

    # Pearson correlation
    n = len(keys)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1 = series[keys[i]]
            s2 = series[keys[j]]
            valid = ~(np.isnan(s1) | np.isnan(s2))
            if valid.sum() > 2:
                r = np.corrcoef(s1[valid], s2[valid])[0, 1]
                corr[i, j] = r
                corr[j, i] = r

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)

    labels = [f"{k}: {_MODEL_SHORT[k]}" for k in keys]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate values
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr[i, j]) > 0.7 else "black")

    ax.set_title("Inter-Model Val RMSE Correlation",
                 fontsize=_FONT_TITLE, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Pearson r", fontsize=10)
    return _save(fig, out_dir / "model_correlation.png")


# ══════════════════════════════════════════════════════════════════════════════
# 12. METRIC EVOLUTION — small multiples (4 metrics × all models)
# ══════════════════════════════════════════════════════════════════════════════

def plot_metric_evolution(results_dir: Path, out_dir: Path) -> Path:
    """2×2 grid: RMSE, MAE, Pearson r, Spearman ρ — all models overlaid."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())

    metrics_config = [
        ("val_rmse", "Validation RMSE ↓", "RMSE (kcal/mol)"),
        ("val_mae", "Validation MAE ↓", "MAE (kcal/mol)"),
        ("val_pearson_r", "Validation Pearson r ↑", "Pearson r"),
        ("val_spearman_r", "Validation Spearman ρ ↑", "Spearman ρ"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, (metric_key, title, ylabel) in zip(axes.flat, metrics_config):
        for key in keys:
            name, epochs, _, _ = data[key]
            xs = [e["epoch"] for e in epochs]
            ys = [_safe(e.get(metric_key, float("nan"))) for e in epochs]
            color = _MODEL_COLORS.get(key, "#888")
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8,
                    label=f"{key}: {_MODEL_SHORT[key]}")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Metric Evolution Across All Models", fontsize=_FONT_TITLE, fontweight="bold")
    return _save(fig, out_dir / "metric_evolution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 13. CONVERGENCE SPEED bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_convergence_speed(results_dir: Path, out_dir: Path) -> Path:
    """Stacked-bar: epochs to best + wasted epochs after best."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, key in enumerate(keys):
        name, epochs, summary, _ = data[key]
        best_ep = summary.get("best_epoch", 0)
        total_ep = len(epochs)
        wasted = total_ep - best_ep

        color = _MODEL_COLORS.get(key, "#888")
        ax.barh(i, best_ep, color=color, edgecolor="white", linewidth=0.5, label="To best" if i == 0 else "")
        ax.barh(i, wasted, left=best_ep, color=color, alpha=0.3, edgecolor="white", linewidth=0.5,
                label="Patience tail" if i == 0 else "")
        ax.text(total_ep + 1, i, f"ep {best_ep}/{total_ep}", fontsize=8, va="center")

    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{k}: {_MODEL_SHORT[k]}" for k in keys], fontsize=10)
    ax.set_xlabel("Epochs", fontsize=_FONT_LABEL)
    ax.set_title("Convergence Speed — Epochs to Best vs Patience Tail",
                 fontsize=_FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / "convergence_speed.png")


# ══════════════════════════════════════════════════════════════════════════════
# 14. FINAL METRIC COMPARISON — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_grouped_metric_bars(results_dir: Path, out_dir: Path) -> Path:
    """Grouped bar chart: RMSE, MAE, 1-Pearson, 1-Spearman for all models."""
    data = _load_all(results_dir)
    keys = sorted(data.keys())

    metric_names = ["RMSE", "MAE", "1−r", "1−ρ"]
    n_metrics = len(metric_names)
    bar_width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(14, 6))

    for j, key in enumerate(keys):
        _, epochs, summary, _ = data[key]
        best_rmse = _safe(summary.get("best_val_rmse", float("nan")))
        best_mae = min((_safe(e.get("val_mae", 99)) for e in epochs), default=float("nan"))
        best_pr = max((_safe(e.get("val_pearson_r", -1)) for e in epochs), default=float("nan"))
        best_sr = max((_safe(e.get("val_spearman_r", -1)) for e in epochs), default=float("nan"))

        vals = [best_rmse, best_mae, 1 - best_pr, 1 - best_sr]
        colors = ["#e53935", "#ff9800", "#1e88e5", "#4caf50"]

        for m, (val, col) in enumerate(zip(vals, colors)):
            x = j + m * bar_width - (n_metrics - 1) * bar_width / 2
            ax.bar(x, val, width=bar_width * 0.9, color=col, alpha=0.8,
                   edgecolor="white", linewidth=0.5,
                   label=metric_names[m] if j == 0 else "")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n{_MODEL_SHORT[k]}" for k in keys], fontsize=9)
    ax.set_ylabel("Error / Residual", fontsize=_FONT_LABEL)
    ax.set_title("Multi-Metric Error Comparison", fontsize=_FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / "grouped_metric_bars.png")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pretty_ligand(name: str) -> str:
    """Clean up ligand names for display."""
    return (name.replace("_", " ")
            .replace("Oacylisourea", "O-acylisourea")
            .replace("ester intermediate", "ester int.")
            .replace("pentagalloylglucose", "PGG")
            .replace("protocatechuic acid", "PCA")
            .title())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════════════

def generate_all(results_dir: str | Path, out_dir: str | Path | None = None) -> list[Path]:
    """Generate all advanced analysis plots."""
    results_dir = Path(results_dir)
    if out_dir is None:
        out_dir = results_dir / "figs"
    out_dir = Path(out_dir)

    print("=" * 60)
    print("Advanced Analysis — generating in-depth visualisations")
    print("=" * 60)

    paths = []

    # Cross-model analysis
    print("\n[1/14] Radar chart...")
    paths.append(plot_radar(results_dir, out_dir))
    print("[2/14] Convergence heatmap...")
    paths.append(plot_convergence_heatmap(results_dir, out_dir))
    print("[3/14] Overfitting gap...")
    paths.append(plot_overfitting_gap(results_dir, out_dir))
    print("[4/14] Efficiency frontier...")
    paths.append(plot_efficiency_frontier(results_dir, out_dir))
    print("[5/14] LR-loss ribbon...")
    paths.append(plot_lr_loss_ribbon(results_dir, out_dir))
    print("[6/14] Ranking table...")
    paths.append(plot_ranking_table(results_dir, out_dir))
    print("[7/14] Metric evolution...")
    paths.append(plot_metric_evolution(results_dir, out_dir))
    print("[8/14] Convergence speed...")
    paths.append(plot_convergence_speed(results_dir, out_dir))
    print("[9/14] Grouped metric bars...")
    paths.append(plot_grouped_metric_bars(results_dir, out_dir))
    print("[10/14] Model correlation matrix...")
    paths.append(plot_model_correlation(results_dir, out_dir))

    # Interpretation (Model D)
    print("\n[11/14] IG beeswarm (SHAP-style)...")
    paths.append(plot_ig_beeswarm(results_dir, out_dir))
    print("[12/14] IG waterfall...")
    paths.append(plot_ig_waterfall(results_dir, out_dir))
    print("[13/14] Grad-CAM heatmap...")
    paths.append(plot_gradcam_heatmap(results_dir, out_dir))
    print("[14/14] Attribution summary...")
    paths.append(plot_attribution_summary(results_dir, out_dir))

    print(f"\n{'='*60}")
    print(f"Done: {len(paths)} figures saved to {out_dir}")
    print(f"{'='*60}")
    return paths


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Advanced model analysis & SHAP-style plots")
    p.add_argument("--results-dir", default="Graph_model/results")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    generate_all(args.results_dir, args.out_dir)
