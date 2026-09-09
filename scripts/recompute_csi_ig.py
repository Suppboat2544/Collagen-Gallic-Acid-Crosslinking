#!/usr/bin/env python3
"""Recompute Vinardo CSI, Model-D LOLO CSI Spearman, and PGG Integrated Gradients
after B6 ligand-identity remediation.

Usage:
  export COLLAGEN_DATA_ROOT=/Users/suppboat/Jupyter_Dock
  export PYTHONPATH=$COLLAGEN_DATA_ROOT/Graph_model_repo
  python scripts/recompute_csi_ig.py --device cpu --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

log = logging.getLogger("csi_ig")


def _data_root() -> Path:
    import os
    return Path(os.environ.get("COLLAGEN_DATA_ROOT", ROOT))


def vinardo_csi() -> dict:
    """Pooled Vinardo CSI = |mean ΔG_MMP1| / |mean ΔG_collagen| per shared ligand."""
    root = _data_root() / "Phukhao" / "collagen_gallic_results"
    col = pd.read_csv(root / "collagen_crosslinking_docking_results.csv")
    mmp = pd.read_csv(root / "mmp1_collagenase_docking_results.csv")
    ligands = sorted(set(col["ligand"]) & set(mmp["ligand"]))
    rows = []
    for lig in ligands:
        c = col.loc[col.ligand == lig, "best_energy_kcalmol"].astype(float)
        m = mmp.loc[mmp.ligand == lig, "best_energy_kcalmol"].astype(float)
        # Match paper's MMP conditions: collagen T=25 subset when available
        c25 = col.loc[(col.ligand == lig) & (col.temperature_C == 25),
                      "best_energy_kcalmol"].astype(float)
        if len(c25) == 0:
            c25 = c
        mean_c, mean_m = float(c25.mean()), float(m.mean())
        csi = abs(mean_m) / abs(mean_c) if abs(mean_c) > 1e-12 else float("nan")
        rows.append({
            "ligand": lig,
            "n_collagen_T25": int(len(c25)),
            "n_mmp1": int(len(m)),
            "mean_dG_collagen": mean_c,
            "mean_dG_mmp1": mean_m,
            "csi": csi,
            "collagen_selective": bool(csi < 1.0),
            "formula": str(col.loc[col.ligand == lig, "ligand_formula"].iloc[0]),
        })
    out = {
        "definition": "CSI = |mean_dG_MMP1| / |mean_dG_collagen_T25|",
        "ligands": rows,
        "n_selective": sum(1 for r in rows if r["collagen_selective"]),
    }
    return out


def _ligand_name(g) -> str:
    for attr in ("ligand_name", "ligand", "ligand_id"):
        v = getattr(g, attr, None)
        if v is not None and str(v):
            return str(v)
    return "unknown"


def _receptor(g) -> str:
    r = getattr(g, "receptor", None)
    if r is not None:
        return str(r).lower()
    sid = str(getattr(g, "sample_id", ""))
    if sid.upper().startswith("MMP1") or "mmp1" in sid.lower():
        return "mmp1"
    return "collagen"


def _forward_delta_g(model, batch, device):
    from Graph_model.train.run_training import _forward_any
    pred, _ = _forward_any(model, batch, device)
    return pred.detach().cpu().view(-1)


def train_lolo_csi(dataset, *, device, epochs, batch_size, lr, val_ratio, seed,
                   results_dir: Path) -> dict:
    """LOLO Model D; per held-out ligand with MMP-1 rows, compute predicted CSI."""
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from Graph_model.train.lolo_cv import LOLOCVSplitter
    from Graph_model.train.device import resolve_device, loader_kwargs, seed_everything
    from Graph_model.train.run_training import _forward_any, _build_model
    from Graph_model.train.metrics import regression_metrics

    device = resolve_device(device)
    seed_everything(seed, device)
    splitter = LOLOCVSplitter(val_ratio=val_ratio, seed=seed)
    vinardo = {r["ligand"]: r["csi"] for r in vinardo_csi()["ligands"]}

    fold_rows = []
    best_pgg_state = None
    best_pgg_val = float("inf")

    for fold in splitter.split(dataset):
        held = fold.held_out_ligand
        log.info("LOLO fold %d held-out=%s train=%d val=%d test=%d",
                 fold.fold, held, fold.n_train, fold.n_val, fold.n_test)

        model = _build_model("D").to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        lk = loader_kwargs(device)
        train_loader = PyGDataLoader(
            Subset(dataset, fold.train_idx), batch_size=batch_size,
            shuffle=True, **lk)
        val_loader = PyGDataLoader(
            Subset(dataset, fold.val_idx), batch_size=batch_size,
            shuffle=False, **lk)

        best_state, best_val, patience_cnt, best_epoch = None, float("inf"), 0, 0
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                opt.zero_grad()
                pred, _ = _forward_any(model, batch, device)
                target = batch.y.to(device).view(-1, 1)
                loss = torch.nn.functional.mse_loss(pred.view(-1, 1), target)
                loss.backward()
                opt.step()

            model.eval()
            v_pred, v_tgt = [], []
            with torch.no_grad():
                for batch in val_loader:
                    pred, _ = _forward_any(model, batch, device)
                    v_pred.extend(pred.cpu().view(-1).tolist())
                    v_tgt.extend(batch.y.view(-1).tolist())
            val_rmse = float(np.sqrt(np.mean(
                (np.array(v_pred) - np.array(v_tgt)) ** 2))) if v_pred else 1e9
            if val_rmse < best_val:
                best_val = val_rmse
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                best_epoch = epoch
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= 15:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        model.to(device).eval()

        col_pred, mmp_pred = [], []
        t_pred, t_tgt = [], []
        with torch.no_grad():
            for i in fold.test_idx:
                g = dataset[i]
                batch = next(iter(PyGDataLoader(
                    Subset(dataset, [i]), batch_size=1, **lk)))
                pred = _forward_delta_g(model, batch, device).item()
                t_pred.append(pred)
                t_tgt.append(float(g.y.item()))
                if _receptor(g) == "mmp1":
                    mmp_pred.append(pred)
                else:
                    col_pred.append(pred)

        tm = regression_metrics(t_pred, t_tgt)
        row = {
            "fold": fold.fold,
            "held_out": held,
            "best_epoch": best_epoch,
            "val_rmse": best_val,
            "test_rmse": tm["rmse"],
            "n_test_collagen": len(col_pred),
            "n_test_mmp1": len(mmp_pred),
        }
        if col_pred and mmp_pred and held in vinardo:
            mean_c_s = float(np.mean(col_pred))
            mean_m_s = float(np.mean(mmp_pred))
            csi_pred = abs(mean_m_s) / abs(mean_c_s) if abs(mean_c_s) > 1e-12 else float("nan")
            row.update({
                "mean_pred_collagen": mean_c_s,
                "mean_pred_mmp1": mean_m_s,
                "csi_pred": csi_pred,
                "csi_vinardo": vinardo[held],
            })
            log.info("  CSI pred=%.4f vinardo=%.4f", csi_pred, vinardo[held])
        fold_rows.append(row)

        if held == "pentagalloylglucose" and best_val < best_pgg_val:
            best_pgg_val = best_val
            best_pgg_state = best_state
            torch.save({
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_rmse": best_val,
                "held_out": held,
                "source": "lolo_fold_pgg",
            }, results_dir / "model_d_best.pt")

        if mmp_pred and col_pred:
            torch.save({
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_rmse": best_val,
                "held_out": held,
                "source": "lolo_fold",
            }, results_dir / "model_d_lolo_last_mmp.pt")

    paired = [r for r in fold_rows if "csi_pred" in r]
    if len(paired) >= 3:
        rho, p = spearmanr([r["csi_pred"] for r in paired],
                           [r["csi_vinardo"] for r in paired])
    else:
        rho, p = float("nan"), float("nan")

    summary = {
        "model": "D",
        "protocol": "LOLO held-out predicted CSI vs Vinardo CSI",
        "n_csi_ligands": len(paired),
        "spearman_rho": float(rho) if rho == rho else None,
        "spearman_p": float(p) if p == p else None,
        "folds": fold_rows,
        "vinardo_csi": vinardo,
    }
    if best_pgg_state is None and (results_dir / "model_d_lolo_last_mmp.pt").exists():
        import shutil
        shutil.copy(results_dir / "model_d_lolo_last_mmp.pt",
                    results_dir / "model_d_best.pt")
    # Also publish under Graph_model/results/
    pub = _data_root() / "Graph_model" / "results" / "model_d_best.pt"
    if (results_dir / "model_d_best.pt").exists():
        import shutil
        shutil.copy(results_dir / "model_d_best.pt", pub)
    return summary


def run_ig(dataset, ckpt_path: Path, results_dir: Path, device: str) -> dict:
    from Graph_model.train.device import resolve_device
    from Graph_model.train.run_training import _build_model
    from Graph_model.interpret.integrated_gradients import integrated_gradients

    device = resolve_device(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _build_model("D")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    picks: dict[str, int] = {}
    for i in range(len(dataset)):
        g = dataset[i]
        if _receptor(g) != "collagen":
            continue
        name = _ligand_name(g)
        if name not in picks:
            picks[name] = i

    ig_results = {}
    for name, idx in sorted(picks.items()):
        g = dataset[idx]
        n_atoms = int(g["ligand"].x.size(0)) if "ligand" in g.node_types else -1
        try:
            attr = integrated_gradients(
                model=model, data=g, target_idx=0, n_steps=50,
                internal_batch_size=8,
            )
            importance = attr["atom_importance"].cpu().numpy()
            n_atoms = int(importance.shape[0])
            ranked = np.argsort(-importance)[: min(10, n_atoms)]
            ig_results[name] = {
                "graph_idx": idx,
                "n_atoms": n_atoms,
                "convergence_delta": float(attr["convergence_delta"]),
                "mean_importance": float(importance.mean()),
                "max_importance": float(importance.max()),
                "std_importance": float(importance.std()),
                "top_atoms": [
                    {"atom_idx": int(ai), "importance": float(importance[ai])}
                    for ai in ranked
                ],
            }
            log.info("IG %s n_atoms=%d mean_I=%.4f", name, n_atoms,
                     ig_results[name]["mean_importance"])
        except Exception as exc:
            log.exception("IG failed for %s", name)
            ig_results[name] = {"error": str(exc), "graph_idx": idx, "n_atoms": n_atoms}

    pgg_keys = [k for k in ig_results if "penta" in k.lower() or k.lower() == "pgg"]
    return {
        "checkpoint": str(ckpt_path),
        "ckpt_meta": {k: ckpt[k] for k in ("best_epoch", "best_val_rmse", "held_out", "source")
                      if k in ckpt},
        "n_steps": 50,
        "target": "collagen_head",
        "per_ligand": ig_results,
        "pgg_keys": pgg_keys,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--skip-train", action="store_true",
                    help="Only Vinardo CSI + IG from existing model_d_best.pt")
    ap.add_argument("--ig-only", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s %(name)s: %(message)s")

    results_dir = Path(args.results_dir or (
        _data_root() / "Graph_model" / "results" / "csi_ig_schema2_b6"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vinardo CSI
    v = vinardo_csi()
    (results_dir / "vinardo_csi.json").write_text(json.dumps(v, indent=2))
    print("=== Vinardo CSI (post-B6) ===")
    for r in v["ligands"]:
        print(f"  {r['ligand']:28s} CSI={r['csi']:.3f}  "
              f"col={r['mean_dG_collagen']:.3f} mmp={r['mean_dG_mmp1']:.3f}  "
              f"selective={r['collagen_selective']}")
    print(f"  n_selective={v['n_selective']}/{len(v['ligands'])}")

    if args.ig_only:
        from Graph_model.data.hetero_dataset import HeteroDockingDataset
        ds = HeteroDockingDataset(include_mmp1=True, force_reload=False).load(verbose=True)
        ckpt = _data_root() / "Graph_model" / "results" / "model_d_best.pt"
        if not ckpt.exists():
            ckpt = results_dir / "model_d_best.pt"
        ig = run_ig(ds, ckpt, results_dir, args.device)
        (results_dir / "model_d_interpretation.json").write_text(json.dumps(ig, indent=2))
        print(f"Wrote {results_dir / 'model_d_interpretation.json'}")
        return 0

    from Graph_model.data.hetero_dataset import HeteroDockingDataset
    ds = HeteroDockingDataset(include_mmp1=True, force_reload=False).load(verbose=True)
    print(f"Dataset: {len(ds)} graphs")

    if not args.skip_train:
        print("=== Model D LOLO + CSI Spearman ===")
        csi_sum = train_lolo_csi(
            ds, device=args.device, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, val_ratio=args.val_ratio, seed=args.seed,
            results_dir=results_dir,
        )
        (results_dir / "csi_spearman_lolo_D.json").write_text(
            json.dumps(csi_sum, indent=2))
        print(f"CSI Spearman ρ = {csi_sum['spearman_rho']} "
              f"(n={csi_sum['n_csi_ligands']}, p={csi_sum['spearman_p']})")

    ckpt = results_dir / "model_d_best.pt"
    if not ckpt.exists():
        ckpt = _data_root() / "Graph_model" / "results" / "model_d_best.pt"
    if ckpt.exists():
        print(f"=== IG from {ckpt} ===")
        ig = run_ig(ds, ckpt, results_dir, args.device)
        (results_dir / "model_d_interpretation.json").write_text(json.dumps(ig, indent=2))
        # also copy to Graph_model/results for compatibility
        dest = _data_root() / "Graph_model" / "results" / "model_d_interpretation.json"
        dest.write_text(json.dumps(ig, indent=2))
        print(f"Wrote {results_dir / 'model_d_interpretation.json'}")
        pgg = [ig["per_ligand"][k] for k in ig.get("pgg_keys", [])]
        if pgg:
            print(f"PGG n_atoms={pgg[0].get('n_atoms')} "
                  f"mean_I={pgg[0].get('mean_importance')}")
    else:
        print("WARNING: no model_d_best.pt — skip IG")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
