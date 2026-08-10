# Collagen / Gallic-Acid Crosslinking — Graph Models

Graph neural network models predicting docking binding energy for collagen
crosslinking agents (gallic acid and related polyphenols, EDC/NHS chemistry)
against collagen I α-2 and MMP-1.

> **Status: research code under repair.** Until 2026-08-11 this package could
> not be imported at all, and `train_lolo_cv` reported NaN for every fold
> without failing. Both are fixed. Several further defects remain live and are
> marked `FIXME` in the source — read [Known issues](#known-issues) before
> using any number this repository produces.
>
> **No result produced before 2026-08-11 can be reproduced from this code.**
> See [What was broken](#what-was-broken).

---

## Install

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r Graph_model/requirements.txt
pip install pytest          # for the test suite
```

`requirements.txt` targets macOS/Apple Silicon and installs the MPS build of
PyTorch. Device selection is MPS → CUDA → CPU
(`Graph_model/train/run_training.py::_auto_device`), so on Apple hardware these
models train on **MPS, not CUDA** — describe it that way in any write-up.

## Data

**The docking data is not in this repository.** The models read two CSVs
produced by an upstream docking pipeline:

```
$COLLAGEN_DATA_ROOT/Phukhao/collagen_gallic_results/
    collagen_crosslinking_docking_results.csv
    mmp1_collagenase_docking_results.csv
```

Point the package at them:

```bash
export COLLAGEN_DATA_ROOT=/path/to/Jupyter_Dock
python scripts/run_lolo.py --check      # verifies deps + data, exits non-zero if not
```

If `COLLAGEN_DATA_ROOT` is unset, the repository root is used. Nothing is
created on import — call `Graph_model.data.config.ensure_dirs()` from your
entry point (`scripts/run_lolo.py` already does).

## Run

```bash
# honest protocol: leave-one-ligand-out CV, 3 seeds, with baselines
python scripts/run_lolo.py --model A --epochs 50 --seeds 0 1 2

# trivial baselines only — takes seconds, needs no training
python scripts/run_lolo.py --baselines-only

# smoke test on 200 records
python scripts/run_lolo.py --model A --epochs 2 --max-records 200
```

Results land in `results/lolo_<MODEL>.json`.

### Why LOLO-CV

There are **9 unique molecules**. A row-level split puts gallic-acid-at-pH-5 in
train and gallic-acid-at-pH-7 in test, which measures interpolation across
conditions, not generalisation across chemistry. `Graph_model/train/lolo_cv.py`
states this in its own module docstring. `StratifiedSplitter`
(`Graph_model/data/splitter.py`) implements the row-level split and is **not
wired into any training path** — do not report numbers from it.

Always report LOLO results against the baselines that `run_lolo.py` prints. A
model that cannot beat "predict the global training mean" has not learned
chemistry. With 9 molecules this is a real risk, not a formality.

## Test

```bash
pytest tests/ -q
```

43 tests covering catalogue chemistry (every SMILES parsed and checked against
its own `mw`/`n_ha`), feature dimensions, and split integrity. One `xfail`
documents the live galloyl-SMARTS defect below.

## Models

| Key | Architecture | Uses the protein? |
|-----|--------------|-------------------|
| A | GATv2 + conditions | no |
| B | Dual encoder + cross-attention | **yes** |
| C | Fragment MPNN | no |
| D | Multi-task selectivity | no |
| E | pGET | no |
| F | DimeNet++ | no |
| G | EGNN | no |
| H | GGNN sequential | no |
| I | Graphormer | no |

Only Option B reads `data['residue']` / the `interacts` edges. **The other
eight are ligand-only regressors** over 9 unique molecules — they see atom
features, bond edges, and a condition vector, and nothing about the receptor.
Describe them accordingly.

## What was broken

Defects found by actually executing the code, all now fixed and covered by
tests. Each independently invalidates any earlier result:

1. **The package could not be imported.** `data/dataset.py` imported
   `.features` (i.e. `Graph_model.data.features`, which has never existed) and
   `features/conditions.py` imported `..config` (`Graph_model.config`, also
   nonexistent). Since `data/__init__.py` imported `CollagenDockingDataset`
   eagerly, `import Graph_model.data.config` raised `ModuleNotFoundError`.
2. **Every forward pass of options A–D raised.** `option_a._encode_raw`
   imported `Graph_model.data.features.conditions` *inside* `forward()`, so it
   failed at runtime, not import time. B, C and D reuse `_encode_raw`.
3. **`train_lolo_cv` swallowed it and reported NaN.** Train, validation, and
   test batches were each wrapped in a bare `except Exception: continue` with
   no logging. Every batch failed, `val_preds`/`test_preds` stayed empty,
   `regression_metrics([], [])` returned NaN, and all 9 folds "completed".
   A full LOLO run finished in ~50 s reporting `rmse_mean: nan`.
4. **`biopython` was missing from `requirements.txt`.** It is required by
   `graph/level2_protein.py`; without it `builder.py` caught the ImportError
   and substituted a **single all-zero residue node**. A by-the-book install
   gave every sample an empty receptor — verified: `residue.x` was `[1, 30]`
   all zeros, versus `[30, 30]` with biopython installed. Option B, the only
   protein-aware model, was attending over one zero vector.
5. **Hardcoded absolute path.** `REPO_ROOT` pointed at
   `/Users/suppboat/Jupyter_Dock`, and `PROCESSED_DIR.mkdir()` ran at import
   time, creating directories on any machine that imported the package.
6. **Ligand catalogue chemistry.** PGG was a `.`-separated string of five free
   gallic acids plus a cyclopentane core — 6 disconnected fragments, 114 heavy
   atoms, MW 1609 against a true 67/940.68, with the galloyl detector counting
   9 units instead of 5. Ellagic acid was a monolactone (C13H8O6) rather than
   the dilactone its own comment described. Three `n_ha`/`mw` values disagreed
   with their own SMILES. All nine now round-trip through RDKit.
7. **NaN targets imputed as 0.0.** Against a −3.83 ± 1.07 kcal/mol
   distribution that is a fabricated +3.5σ outlier. Rows are now dropped.
8. **Silent row loss.** `hetero_dataset` counted failures behind five debug
   messages. It now logs every failure with its reason and refuses to build if
   >5 % of rows drop.

## Known issues

Live defects, each marked `FIXME` at the relevant line:

- **Galloyl SMARTS is malformed** (`features/galloyl.py`). `_SMARTS_GALLOYL`
  encodes 1,3,4-trihydroxybenzene, not a 3,4,5-galloyl ring, so
  `galloyl_strict` is **0 for every ligand including gallic acid**. Its weight
  never fires, and `galloyl_weighted` double-counts each ring via the catechol
  + pyrogallol patterns (gallic acid → 1.67, not 1.00).
- **pH encoding violates Henderson–Hasselbalch** (`data/config.py`).
  `PROPKA_PROTONATION` hardcodes 0.85 → 0.15 across 0.5 pH units, which no
  single pKa produces. `graph/residue_data.py` implements HH correctly; the
  incorrect table is the one reaching the network.
- **The documented loss is never used.** All four training loops use
  `nn.MSELoss()`. `CombinedDockingLoss` (MSE + ListMLE + monotonicity + NLL) is
  defined and exported but **instantiated nowhere**, so the ranking and
  monotonicity terms and the heteroscedastic uncertainty head are untrained.
- **Models F and G have no real geometry.** EGNN coordinates are a
  `Linear(35→3)` of atom features; DimeNet "angles" come from cosine similarity
  of edge features and "distances" from `norm(edge_attr)`. Genuine ETKDG
  conformers exist in `features/conformer_3d.py` and are imported nowhere.
  Neither model is SE(3)-equivariant as built.
- **`CollagenDockingDataset` is a dead path.** It emits a homogeneous 54-D
  `Data` object; every model indexes `data['ligand']`, which it never sets.
  Only `HeteroDockingDataset` feeds the models.
- **No seeding of `random`/`numpy`** inside the library itself — only
  `torch.manual_seed`. `scripts/run_lolo.py::set_all_seeds` covers all three
  plus deterministic algorithms; use the script, not the library, as the entry
  point.
- `maml.py`, `contrastive.py`, `hpo.py` are not exercised by any entry point.
  If HPO was used to pick hyperparameters on the validation split, reported
  validation metrics are optimistically biased.

### Node feature dimensions

Two featurisers exist and genuinely differ. Quote **35** when describing the
trained models:

| Module | Constant | Value | Consumed by |
|--------|----------|-------|-------------|
| `graph/level1_ligand.py` | `LIGAND_NODE_DIM` | **35** | `builder.py` → `hetero_dataset.py` → all models |
| `features/atom.py` | `ATOM_FEAT_DIM` | 54 | `data/dataset.py` only (dead path) |

Both are asserted against the running code in `tests/test_features.py`.

## Layout

```
Graph_model/
  data/       config (ligand catalogue, paths), loaders, datasets, splitters
  features/   atom/bond features, galloyl detection, conditions, conformers
  graph/      level-1 ligand, level-2 protein, level-3 bipartite, builder
  model/      options A–I, losses, uncertainty, ensembles
  train/      training loops, LOLO-CV, metrics, HPO, pretraining
  interpret/  integrated gradients, Grad-CAM, attention rollout, probing
  screen/     virtual screening and Pareto selection
scripts/      entry points
tests/        pytest suite
```
