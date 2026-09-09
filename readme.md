# Collagen / Gallic-Acid Crosslinking — Graph Models

Graph neural network models predicting **Vinardo docking ΔG** (not experimental
\(K_d\)) for collagen crosslinking agents (gallic acid / polyphenols, EDC/NHS)
against collagen I α-2 and MMP-1.

> **Remediation status (2026-09):** blocking import/NaN bugs are fixed; NHS /
> NHS-ester / PGG were re-docked as the catalogue molecules; Model A schema-2
> LOLO and Model D CSI/IG were recomputed. **Prior Table IV affinity ρ ≈ 0.93
> and CSI ρ ≈ 0.95 do not reproduce** and must not be cited. See
> [`docs/REMEDIATION_STATUS.md`](docs/REMEDIATION_STATUS.md) and
> [`docs/issues_by_category.tex`](docs/issues_by_category.tex).

---

## Verified LOLO numbers (post–B6, schema-2)

| Predictor | RMSE | MAE | Spearman ρ | Notes |
|-----------|------|-----|------------|-------|
| **Model A** (GATv2) | **1.338** | **1.248** | **0.176** | 3/9 folds beat global-mean |
| Global-mean baseline | 1.066 | — | — | same LOLO folds |
| Per-box-mean baseline | 0.977 | — | — | stronger than Model A |
| **Model D CSI** | — | — | **−0.60** | n=5 dual-receptor ligands; p=0.28 |
| Prior claimed A | 1.33 | 0.91 | 0.93 | **withdrawn** |
| Prior claimed D CSI | — | — | 0.95 | **withdrawn** |

JSON: [`docs/remediation_outputs/`](docs/remediation_outputs/).

**Vinardo CSI** (T=25 collagen vs MMP-1, post-correction): PGG **1.21**,
ellagic **1.05**, phenols **≥1.20** — **0/5** with CSI < 1. Prior PGG
CSI ≈ 0.84 is withdrawn.

**PGG IG** (Model D, 50 steps): **67** heavy atoms, mean Ī **0.033**
(prior N=81 maps withdrawn).

Manuscript copies: [`docs/manuscript/`](docs/manuscript/).

---

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r Graph_model/requirements.txt
pip install pytest
```

Device selection is CUDA → MPS → CPU (`Graph_model.train.device`). Prefer
`--device cpu` when disk/MPS temp space is tight.

## Data

**Docking CSVs / poses are not in this repository.**

```
$COLLAGEN_DATA_ROOT/Phukhao/collagen_gallic_results/
    collagen_crosslinking_docking_results.csv
    mmp1_collagenase_docking_results.csv
```

```bash
export COLLAGEN_DATA_ROOT=/path/to/Jupyter_Dock
python scripts/run_lolo.py --check
```

## Run

```bash
# Affinity LOLO (Model A) — reuse cache; CPU recommended after B6
python scripts/run_lolo.py --model A --epochs 50 --seeds 0 \
  --device cpu --batch-size 8 \
  --results-dir Graph_model/results/lolo_schema2_b6

# CSI Spearman + PGG Integrated Gradients (Model D LOLO + IG)
python scripts/recompute_csi_ig.py --device cpu --batch-size 8 --epochs 50

# Trivial baselines only
python scripts/run_lolo.py --baselines-only --device cpu
```

Always compare LOLO RMSE to the printed baselines. A model that cannot beat
the global / per-box training mean has not demonstrated learned chemistry.

### Why LOLO-CV

There are **9 unique molecules**. A row-level split measures interpolation
across pH/box, not generalisation across chemistry. Use
`scripts/run_lolo.py` (not `StratifiedSplitter`).

## Test

```bash
pytest tests/ -q
```

## Models

| Key | Architecture | Reads protein? |
|-----|--------------|----------------|
| A | GATv2 + conditions | no |
| B | Dual encoder + cross-attention | **yes** |
| C | Fragment MPNN | no |
| D | Multi-task selectivity heads | no* |
| E–I | pGET / DimeNet++ / EGNN / GGNN / Graphormer | no |

\*Model D has collagen / MMP-1 / log-CSI heads in code, but the shipped train
loop optimises the collagen head with **MSE** only (`CombinedDockingLoss` is
unused). CSI ρ in the table above is from **aggregated dual-receptor ΔG
predictions**, not the unused CSI head loss.

## What was broken (historical)

Pre-2026-08-11: package unimportable; A–D forward raised; LOLO reported NaN;
biopython missing → empty receptors; hardcoded absolute paths; catalogue
SMILES errors; NaN targets imputed as 0. See the long form in git history /
`docs/issues_by_category.tex` §P1–P3. Those blockers are **fixed**.

## Known issues (still live in code)

- **MSE only.** Documented `CombinedDockingLoss` is never instantiated.
- **Models F/G lack true 3D geometry** (pseudo-coordinates / feature proxies).
- **`CollagenDockingDataset` is a dead path** — only `HeteroDockingDataset`
  feeds training.
- **MMP-1:** 6 reconstructed boxes remain empty of protein; original Vina
  configs still preferred over pose-centroid sidecars (~4 Å validation error).
- **Figures** in the older manuscript PDF may still annotate pre-B6 CSI
  values; text tables in `docs/manuscript/` are authoritative.

Galloyl SMARTS and catalogue SMILES were corrected (feature schema 2); do not
reuse caches built before that fix without `--force-reload`.

## Layout

```
Graph_model/   package (data, graph, model, train, interpret, …)
scripts/       run_lolo.py, recompute_csi_ig.py, redock helpers, …
docs/          REMEDIATION_STATUS.md, issues_by_category.tex, manuscript/, outputs/
tests/         pytest suite
```

## See also

- [`Graph_model/README.md`](Graph_model/README.md) — architecture notes
- [`docs/REMEDIATION_STATUS.md`](docs/REMEDIATION_STATUS.md) — checklist
- [`docs/issues_by_category.tex`](docs/issues_by_category.tex) — full issue register
- Public tree: https://github.com/Suppboat2544/Collagen-Gallic-Acid-Crosslinking
