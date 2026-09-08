# Remediation status (2026-09)

## Completed
- **P4.1–P4.2**: Manuscript Table IV reconciled — LOLO Spearman ρ ≈ 0.163, RMSE 1.047 vs global-mean 0.992 / per-box 0.918 (prior ρ ≈ 0.93 does not reproduce).
- **P1.5**: Table IV target clarified as anchor Vinardo-only (not mixed PDBbind).
- **P1.6 / P4.4 / P4.5**: Training claims match code (MSE; ligand-only except Model B; F/G pseudo-geometry; CPU not H100).
- **B4 / B5**: `scripts/reconstruct_mmp1_boxes.py` recovers 67/67 MMP-1 box centres from poses (~4 Å collagen validation error); sidecar CSV + occupancy check.
- **B6 structures**: Catalogue SMILES rebuilt; provenance check passes for all nine ligands.
- **B6 NHS / NHS-ester**: Fully re-docked on full pH-protonated collagen receptors (not bindingsite extracts).
- Scripts: `redock_corrected_ligands.py`, `harvest_redock_sdfs.py`, `run_post_redock_lolo.sh`.

## In progress
- **B6 PGG**: Collapsed-T (T=25) re-dock at exhaustiveness 8; resume with `--resume`. Until complete, PGG CSI/exhaustiveness-64 and IG maps remain provisional.
- **Schema-2 LOLO**: Run `scripts/run_post_redock_lolo.sh` after PGG finishes (`--force-reload`).

## Notes
- Never dock against `*_bindingsite*.pdb` (≈80 atoms) — yields null Vinardo scores.
- Docking CSVs / pose libraries live outside this code repository (set `COLLAGEN_DATA_ROOT`).
