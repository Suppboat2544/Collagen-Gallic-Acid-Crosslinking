# Graph_model

Heterogeneous GNN code for collagen / MMP-1 Vinardo ΔG and CSI prediction,
aligned with **`Phukhao/Proposal_LLNL/Proposal_LLNL_bioRxiv.tex`**.

Public snapshot:
https://github.com/Suppboat2544/Collagen-Gallic-Acid-Crosslinking/tree/main/Graph_model

## Models A–I (bioRxiv Table IV, LOLO-CV)

| ID | Architecture | RMSE | MAE | ρ | CSI ρ |
|----|--------------|------|-----|---|-------|
| A | GATv2 baseline | 1.33 | 0.91 | 0.93 | 0.90 |
| B | Cross-attention | 1.47 | 1.01 | 0.91 | 0.85 |
| C | Fragment MPNN | 1.51 | 1.05 | 0.89 | 0.80 |
| D | Multi-task (3-head) | 1.39 | 0.95 | 0.92 | **0.95** |
| E | pGET | 1.42 | 0.98 | 0.91 | 0.85 |
| F | DimeNet++ | 1.55 | 1.10 | 0.88 | 0.80 |
| G | EGNN | 1.48 | 1.02 | 0.90 | 0.85 |
| H | GGNN | 1.52 | 1.06 | 0.89 | 0.80 |
| I | Graphormer | 1.45 | 1.00 | 0.91 | 0.85 |

- Primary affinity model: **A** (~1.3 kcal/mol LOLO-CV RMSE).
- Primary selectivity model: **D** (CSI Spearman ρ = 0.95).
- Targets are **Vinardo docking scores**, not experimental \(K_d\).
- Model D IG analyses may quote a **single-fold** checkpoint `val_RMSE ≈ 0.549`
  (best epoch 45); that is **not** the Table IV LOLO-CV affinity RMSE (1.39).

## Layout (local)

```
Graph_model/
  data/           # datasets, splitters; features live in data/features/
  graph/          # ligand / protein / bipartite builders (35-D / 13-D ligand)
  model/          # option_a … option_e, dimenet, egnn, ggnn, graphormer
  train/          # LOLO-CV, HPO, transfer, metrics
  interpret/      # Grad-CAM, Integrated Gradients, attention rollout
  screen/         # library scoring / Pareto
  viz/            # comparison plots (A–I)
  results/        # training logs, figures, Model D checkpoint artifacts
  train_main.py   # entry point
```

Note: the older GitHub tree kept `features/` at the package root; this tree nests
them under `data/features/` (same modules).

## Architecture constants (match bioRxiv Methods)

- Ligand atom features: **35-D**; bond features: **13-D**
- Model A/D backbone: 4-layer GATv2, **4 heads × 32 = 128**, MLP **160→256→128→1**
- Model D: three heads (collagen ΔG, MMP-1 ΔG, log-CSI) + Kendall uncertainty loss;
  MMP-1 head loss weight ×10 (120 MMP-1 vs ~2,052 collagen calculations)

## Install

From the repository root (`Collagen-Gallic-Acid-Crosslinking/`):

```bash
# recommended: editable package install (uses pyproject.toml)
pip install -e .

# or dependencies only
pip install -r Graph_model/requirements.txt

# optional conda env
conda env create -f Graph_model/environment.yml
conda activate graph-model
pip install -e .
```

Install files:
- `pyproject.toml` — package metadata + dependencies (`pip install -e .`)
- `Graph_model/requirements.txt` — pip requirements list
- `Graph_model/environment.yml` — conda environment template

## Reproduce figures / metrics

```bash
python -m Graph_model.train_main --help
python -m Graph_model.viz.compare_models   # expects results/option_{a-i}_training.json
```

`results/option_*_training.json` are **single-split** training logs used for curves.
Manuscript Table IV reports **LOLO-CV** aggregates (see `results/comparison_summary.json`
→ `biorxiv_table_iv_lolo_cv`).
