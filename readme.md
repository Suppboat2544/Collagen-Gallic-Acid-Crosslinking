# Project: Collagen–Gallic Acid Crosslinking

This repository integrates **wet‑lab protocols**, **multi‑scale molecular simulations**, and an **AI‑driven predictive model** into a unified workflow for designing and validating collagen materials crosslinked with gallic acid.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Getting Started](#getting-started)
4. [Wet‑Lab Protocol](#wet-lab-protocol)
5. [Computational Simulation](#computational-simulation)
   - [Atomistic MD](#atomistic-md)
   - [Enhanced Sampling (Metadynamics & Umbrella)](#enhanced-sampling-metadynamics--umbrella)
   - [QM/MM Reaction Mechanism](#qmmm-reaction-mechanism)
   - [Coarse‑Grained Network Mechanics](#coarse-grained-network-mechanics)
6. [AI‑Driven Predictive Model](#ai-driven-predictive-model)
   - [Data Collection & Feature Engineering](#data-collection--feature-engineering)
   - [Model Architecture & Hyperparameters](#model-architecture--hyperparameters)
   - [Training Pipeline](#training-pipeline)
   - [Inference & Deployment](#inference--deployment)
7. [Scripts & Examples](#scripts--examples)
8. [CI/CD & Reproducibility](#cicd--reproducibility)
9. [Contributing](#contributing)
10. [License](#license)

---

## Overview

Collagen crosslinked with gallic acid exhibits improved thermal stability, mechanical strength, and antioxidant properties. This end‑to‑end project:

- **Designs** crosslinking conditions via combinatorial wet‑lab assays.
- **Simulates** interactions from atomistic to mesoscale.
- **Predicts** material properties using a machine learning model trained on combined datasets.
- **Validates** predictions experimentally.

## Repository Structure

```
├── wetlab/                   # Protocol docs, raw data, instrument logs
│   └── protocols.md
├── simulation/               # MD, enhanced sampling, QM/MM, CG input & analysis
│   ├── atomistic/            # GROMACS topologies, run scripts
│   ├── metadynamics/         # PLUMED input, analysis notebooks
│   ├── qmmm/                 # ORCA input templates, scripts
│   └── cg/                   # MARTINI mapping, LAMMPS/GROMACS CG scripts
├── scripts/                  # Preprocess, train.py, predict.py, utils
│   ├── preprocess.py         # Feature extraction from simulation outputs
│   ├── train.py              # Model training & cross‑validation
│   ├── predict.py            # Inference on new modifiers
│   └── evaluate.py           # Metrics & plotting routines
├── models/                   # Serialized model files (e.g. xgboost.pkl)
├── data/                     # Sample datasets (CSV), feature tables
│   └── new_modifiers.csv
├── notebooks/                # Jupyter notebooks for analysis & tutorials
│   ├── simulation_analysis.ipynb
│   └── inference_example.ipynb
├── environment.yml           # Conda environment specification
├── CI/                       # CI configuration (e.g. GitHub Actions workflows)
└── README.md                 # This file
```

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/your_org/collagen-gallic-acid.git
   cd collagen-gallic-acid
   ```

2. **Create conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate collagen-ga
   ```

3. **Install external tools**

   - **GROMACS** (≥2021) with PLUMED plugin
   - **ORCA** (≥5.0) for QM/MM
   - **LAMMPS** or **GROMACS-MARTINI** for CG simulations

4. **Directory permissions** Ensure `wetlab/raw/` is read‑only and `simulation/output/` is writeable.

## Wet‑Lab Protocol

Detailed step‑by‑step instructions in `wetlab/protocols.md`, including:

- Chemical sourcing (catalog numbers)
- Reaction scales (100 µg–10 mg collagen)
- Instrument parameters (FTIR, NMR, DSC, ITC)
- Data export conventions (CSV naming, metadata headers)

## Computational Simulation

### Atomistic MD

1. **Structure Preparation** (`simulation/atomistic/build.sh`)

   - Fetch collagen α‑chain PDB (UniProt IDs P02452, Q02252)
   - Use CHARMM‑GUI Collagen Builder or Modeller script to generate 9‑triplet helix.

2. **Parameterization**

   - `topol.top`: AMBER ff14SB for protein, GAFF2 for gallic acid (charges via RESP)
   - `oint.itp`: include gallic acid residue definition.

3. **Equilibration & Production** (`run_equil.sh` & `run_prod.sh`)

   ```bash
   gmx grompp -f minim.mdp -c start.gro -p topol.top -o em.tpr
   gmx mdrun -deffnm em
   # NVT
   gmx grompp -f nvt.mdp -c em.gro -o nvt.tpr
   gmx mdrun -deffnm nvt
   # NPT
   gmx grompp -f npt.mdp -c nvt.gro -o npt.tpr
   gmx mdrun -deffnm npt
   # Production
   gmx grompp -f md.mdp -c npt.gro -o md.tpr
   gmx mdrun -deffnm md -nt 16
   ```

4. **Analysis**

   - RMSD, RMSF: `gmx rms`, `gmx rmsf`
   - Hydrogen bonds: `gmx hbond`
   - Save time series to `simulation/atomistic/output/`

### Enhanced Sampling (Metadynamics & Umbrella)

- **PLUMED input** in `simulation/metadynamics/plumed.dat`

- **Metadynamics**

  ```bash
  gmx mdrun -plumed plumed.dat -deffnm meta  
  ```

  - Hill deposition: 1.2 kJ/mol every 500 steps, biasfactor = 15.

- **Umbrella Sampling**

  - Generate windows with `scripts/gen_windows.py` (0.2 Å spacing).
  - Run each window in parallel.
  - Combine PMF with `gmx wham -it tpr-files.dat -if pullf-files.dat`

### QM/MM Reaction Mechanism

- **ORCA inputs** in `simulation/qmmm/`
- Example `qmmm.inp`:
  ```ini
  ! B3LYP D3BJ def2-SVP QM/MM
  %qmmm
    Program           Psi4
    QMAtoms           [list of indices]
    LinkAtoms         Yes
  end
  * xyz 0 1
  ... coordinates ...
  *
  ```
- **NEB**: `neb.sh` automates 16 images and IRC checks.

### Coarse‑Grained Network Mechanics

- **Mapping script**: `simulation/cg/map_martini.py`
- **Run**:
  ```bash
  gmx grompp -f cg.mdp -c init_cg.gro -p topol_cg.top -o cg.tpr
  gmx mdrun -deffnm cg -nt 8
  ```
- **Tensile testing** in `scripts/tensile.py` (uses MD trajectories to compute stress–strain).

## AI‑Driven Predictive Model

### Data Collection & Feature Engineering

- **Input sources**:

  - `simulation/*/output/` CSVs for ΔG\_binding, RMSD peaks, H-bond lifetimes.
  - `wetlab/output/*.csv` for T\_d (DSC), ΔG\_ITC, reaction yields.

- **Feature engineering** in `scripts/preprocess.py`:

  - Physicochemical: `rdkit` for logP, TPSA, pKa.
  - Structural: mean RMSD, CG stiffness.
  - Kinetic: activation barrier from QM/MM.

### Model Architecture & Hyperparameters

- **XGBoost** with cross‑validated hyperopt search:
  - `scripts/train.py` defines:
    ```yaml
    n_estimators: 800
    max_depth: 7
    learning_rate: 0.005
    subsample: 0.8
    colsample_bytree: 0.8
    early_stopping_rounds: 50
    ```

### Training Pipeline

1. Split data: 70% train / 15% val / 15% test (stratified).
2. 5‑fold CV with `sklearn.model_selection.StratifiedKFold`.
3. Metrics: MAE, RMSE, R².
4. Save best model to `models/xgb_best.pkl`.

Usage:

```bash
python scripts/train.py --config config/train.yaml --output models/
```

### Inference & Deployment

- **Predict script** (`scripts/predict.py`):
  ```bash
  python scripts/predict.py --model models/xgb_best.pkl --input data/features_new.csv --output results/predictions.csv
  ```
- **API**: Flask app in `services/api.py`, serve `/predict` endpoint.
- **Docker**: `Dockerfile` for containerizing ML service.

## Scripts & Examples

- **Notebook tutorials** in `notebooks/`:

  - **simulation\_analysis.ipynb**: Visualize FES, PMF, RMSD plots.
  - **inference\_example.ipynb**: End‑to‑end demo from raw MD data → prediction.

- **Template configs**:

  - `environment.yml`: lists Python packages (pandas, scikit‑learn, xgboost, rdkit).
  - `CI/gitlab‑ci.yml`: runs lint, unit tests, training smoke test.

## CI/CD & Reproducibility

- **GitHub Actions** workflows in `.github/workflows/`:

  - **lint.yml**: flake8, black check.
  - **test.yml**: pytest suite on `scripts/`.
  - **train-test.yml**: quick training run (<5 min) on subset to catch regressions.

- **Data versioning** with DVC:

  - `dvc.yaml` tracks raw data & models.
  - Storage: S3 or local `dvc_storage/`.

## Contributing

1. Fork & create feature branch
2. Write tests for new functionality
3. Submit PR, ensure CI passes
4. Tag reviewers from relevant teams

Please adhere to [CONTRIBUTING.md](CONTRIBUTING.md) guidelines.

## License

MIT © 2025 IST\_Fujii\_Lab

