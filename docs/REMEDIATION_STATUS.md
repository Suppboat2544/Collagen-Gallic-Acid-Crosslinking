# Remediation status (2026-09)

## Completed
- **P4.1–P4.2**: Manuscript Table IV reconciled; prior affinity ρ ≈ 0.93 does not reproduce.
- **P1.5**: Table IV target clarified as anchor Vinardo-only (not mixed PDBbind).
- **P1.6 / P4.4 / P4.5**: Training claims match code (MSE; ligand-only except Model B; F/G pseudo-geometry; CPU not H100).
- **B4 / B5**: MMP-1 box centres reconstructed (67/67 groups); sidecar CSV.
- **B6**: NHS / NHS-ester / PGG re-docked (catalogue structures; PGG mean ΔG ≈ −6.34).
- **Schema-2 LOLO Model A** (seed 0, CPU): RMSE **1.338**, MAE **1.248**, Spearman ρ **0.176** (3/9 beat global-mean). Baselines 1.066 / 0.977. JSON: `docs/remediation_outputs/lolo_schema2_b6/lolo_A.json`.
- **CSI Spearman (Model D LOLO, post-B6)**: ρ **−0.60** (n=5, p=0.28). Prior CSI ρ ≈ 0.95 **withdrawn**. JSON: `docs/remediation_outputs/csi_ig_schema2_b6/csi_spearman_lolo_D.json`.
- **Vinardo CSI (pooled, T=25 collagen vs MMP-1)**: PGG **1.21**, ellagic **1.05**, gallic **1.20**, PCA **1.20**, pyrogallol **1.24** — **0/5** with CSI < 1. Prior PGG CSI ≈ 0.84 **withdrawn**.
- **PGG IG (Model D, 50 steps)**: **N=67** heavy atoms, mean Ī **0.033**, max 0.063 (checkpoint: PGG LOLO fold, epoch 10, val RMSE 0.403). Prior N=81 maps **withdrawn**. JSON: `docs/remediation_outputs/csi_ig_schema2_b6/model_d_interpretation.json`.
- Scripts: `scripts/recompute_csi_ig.py`, `scripts/run_lolo.py --device`, redock/harvest helpers.

## Optional next
- Original MMP-1 Vina configs for the 6 empty reconstructed boxes.
- Refresh figures that still annotate pre-B6 CSI values.

## Notes
- Docking CSVs / pose libraries / `*.pt` checkpoints live outside this code repo (`COLLAGEN_DATA_ROOT`).
