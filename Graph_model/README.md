# Graph_model

Heterogeneous GNN code for collagen / MMP-1 **Vinardo ΔG** prediction
(and dual-receptor CSI aggregation). Aligned with the remediated bioRxiv
manuscript in [`docs/manuscript/Proposal_LLNL_bioRxiv.tex`](../docs/manuscript/Proposal_LLNL_bioRxiv.tex).

Public snapshot:
https://github.com/Suppboat2544/Collagen-Gallic-Acid-Crosslinking/tree/main/Graph_model

> **Do not cite the old Table IV numbers (affinity ρ ≈ 0.93, CSI ρ ≈ 0.95).**
> They do not reproduce under LOLO on corrected docks. Use the verified
> post–B6 values below (also in the root [`readme.md`](../readme.md)).

## Verified LOLO (post–B6 schema-2)

| ID | Role | Metric | Value |
|----|------|--------|-------|
| **A** | Affinity (GATv2) | RMSE / MAE / Spearman ρ | **1.338 / 1.248 / 0.176** |
| — | Global-mean baseline | RMSE | 1.066 |
| — | Per-box-mean baseline | RMSE | **0.977** (beats Model A) |
| **D** | CSI (aggregated ΔG) | Spearman ρ vs Vinardo CSI | **−0.60** (n=5, p=0.28) |

JSON: [`docs/remediation_outputs/`](../docs/remediation_outputs/).

- Targets are **Vinardo docking scores**, not experimental \(K_d\).
- Only **Model B** reads protein residue nodes; A and C–I are ligand-only.
- Shipped training uses **MSE** (not `CombinedDockingLoss`).
- Models **F/G** lack true docked 3D geometry in the feature pipeline.
- PGG IG (Model D, 50 steps): **N = 67** HA, mean Ī ≈ 0.033
  (checkpoint: PGG LOLO fold, epoch 10, val RMSE ≈ 0.403 — not Model A
  LOLO affinity RMSE).

### Withdrawn prior Table IV (do not reuse)

| ID | Was reported | Status |
|----|--------------|--------|
| A | RMSE 1.33, MAE 0.91, ρ 0.93, CSI ρ 0.90 | affinity ρ **withdrawn** |
| D | RMSE 1.39, CSI ρ **0.95** | CSI ρ **withdrawn** |
| B–I | various ρ ≈ 0.88–0.92 | not re-verified post-B6 |

## Layout

```
Graph_model/
  data/           # datasets, splitters; features in data/features/
  graph/          # ligand / protein / bipartite builders (35-D / 13-D)
  model/          # option_a … option_e, dimenet, egnn, ggnn, graphormer
  train/          # LOLO-CV, HPO, metrics, device
  interpret/      # Grad-CAM, Integrated Gradients, attention
  screen/         # library scoring / Pareto
  viz/            # comparison plots
  train_main.py   # multi-model entry
```

## Architecture constants

- Ligand atom features: **35-D**; bond features: **13-D**
- Model A/D backbone: 4-layer GATv2, **4 heads × 32 = 128**, MLP **160→256→128→1**
- Model D defines three heads (collagen ΔG, MMP-1 ΔG, log-CSI); train loop
  currently backprops the collagen head via MSE. CSI ρ above is from
  LOLO predicted mean \|ΔG\| ratios, matching the manuscript definition.

## Install / run

From the repository root:

```bash
pip install -e .
export COLLAGEN_DATA_ROOT=/path/to/data   # CSVs live outside this repo

python scripts/run_lolo.py --model A --epochs 50 --seeds 0 --device cpu
python scripts/recompute_csi_ig.py --device cpu --batch-size 8
```

See root [`readme.md`](../readme.md) for full commands and data paths.

## Reproduce manuscript metrics

```bash
# Affinity Table IV (Model A)
python scripts/run_lolo.py --model A --epochs 50 --seeds 0 --device cpu \
  --results-dir Graph_model/results/lolo_schema2_b6

# CSI Spearman + per-ligand IG
python scripts/recompute_csi_ig.py --device cpu
```

`results/option_*_training.json` are **single-split** logs for curves only.
`Graph_model/results/comparison_summary.json` → `biorxiv_table_iv_lolo_cv`
still stores the **historical withdrawn** table for audit trail; prefer
`docs/remediation_outputs/` for current numbers.
