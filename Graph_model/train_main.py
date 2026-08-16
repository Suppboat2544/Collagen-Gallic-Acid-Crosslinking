#!/usr/bin/env python3
"""
Graph_model.train_main
=========================
CLI entry point for full model training.

Builds the three-level HeteroData dataset from docking CSVs + PDB/SDF files,
then trains all 9 model architectures (A–I) with identical data splits.

Examples
--------
  # Train all models (default: 100 epochs, batch 32)
  python -m Graph_model.train_main

  # Train specific models
  python -m Graph_model.train_main --models A E I

  # Quick test run
  python -m Graph_model.train_main --models A --max-epochs 5 --max-records 200

  # Full training with custom settings
  python -m Graph_model.train_main --max-epochs 200 --patience 25 --lr 5e-4

  # LOLO-CV for one model
  python -m Graph_model.train_main --models E --lolo-cv --max-epochs 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Ensure macOS OpenMP compatibility ──────────────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Graph_model docking prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model keys:
  A  GATv2 Baseline            F  DimeNet++
  B  Dual Encoder CrossAttn    G  EGNN
  C  Fragment MPNN             H  GGNN Sequential
  D  Multi-Task Selectivity    I  Graphormer
  E  pGET (novel)
""",
    )
    p.add_argument(
        "--models", nargs="*", default=None,
        help="Model keys to train (e.g. A B E). Default: all 9 models.",
    )
    p.add_argument(
        "--results-dir", type=str, default="Graph_model/results",
        help="Output directory for training logs and checkpoints (default: Graph_model/results).",
    )
    p.add_argument("--max-epochs", type=int, default=200, help="Max epochs (default: 200).")
    p.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="L2 weight decay (default: 1e-4).")
    p.add_argument("--warmup-epochs", type=int, default=5, help="LR warmup epochs (default: 5).")
    p.add_argument("--patience", type=int, default=15, help="Early stopping patience (default: 15).")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Validation fraction (default: 0.15).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")

    # Dataset options
    p.add_argument(
        "--no-mmp1", action="store_true",
        help="Exclude MMP-1 docking records.",
    )
    p.add_argument(
        "--no-bipartite", action="store_true",
        help="Skip Level-3 bipartite (ligand↔residue) edges for faster building.",
    )
    p.add_argument(
        "--max-records", type=int, default=None,
        help="Limit dataset records (for testing). Default: all.",
    )
    p.add_argument(
        "--force-reload", action="store_true",
        help="Force rebuild dataset (ignore cache).",
    )
    p.add_argument(
        "--fallback-homogeneous", action="store_true",
        help="Fall back to CollagenDockingDataset (homogeneous graphs) if HeteroData fails.",
    )
    p.add_argument(
        "--include-transfer", action="store_true",
        help="Include PDBbind v2020R1 transfer data from external_dataset/.",
    )
    p.add_argument(
        "--max-transfer", type=int, default=5000,
        help="Max PDBbind transfer entries (default: 5000). Use 0 for all ~18K.",
    )

    # Training mode
    p.add_argument(
        "--lolo-cv", action="store_true",
        help="Use leave-one-ligand-out cross-validation instead of simple split.",
    )

    # Device
    p.add_argument(
        "--device", type=str, default=None,
        choices=["cpu", "mps", "cuda"],
        help="Force device. Default: auto-detect (MPS on Apple Silicon).",
    )

    # Logging
    p.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG).",
    )

    # Hidden: subprocess mode (trains single model in-process, used internally)
    p.add_argument("--_subprocess", action="store_true", help=argparse.SUPPRESS)

    return p.parse_args()


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(args: argparse.Namespace):
    """Load dataset (HeteroData preferred, with homogeneous fallback)."""
    try:
        from Graph_model.data.hetero_dataset import HeteroDockingDataset

        print("=" * 60)
        print("Loading three-level HeteroData dataset...")
        print(f"  Source: Phukhao/collagen_gallic_results/ (anchor)")
        if args.include_transfer:
            print(f"  Source: Graph_model/external_dataset/ (PDBbind transfer)")
        print("=" * 60)

        ds = HeteroDockingDataset(
            include_mmp1=not args.no_mmp1,
            include_bipartite=not args.no_bipartite,
            use_cache=True,
            force_reload=args.force_reload,
            max_records=args.max_records,
        )
        ds.load(verbose=True)

        if len(ds) == 0:
            raise RuntimeError("HeteroData dataset is empty")

        print(f"\nAnchor dataset: {len(ds)} HeteroData graphs")
        summary = ds.summary()
        print(f"Ligands: {summary['by_ligand']}")

        # Optionally add PDBbind transfer data
        if args.include_transfer:
            max_t = args.max_transfer if args.max_transfer > 0 else None
            transfer_ds = _load_pdbbind_transfer(max_entries=max_t)
            if transfer_ds is not None and len(transfer_ds) > 0:
                combined = _CombinedDataset(ds, transfer_ds)
                print(f"\nCombined: {len(combined)} total "
                      f"({len(ds)} anchor + {len(transfer_ds)} PDBbind transfer)")
                return combined

        return ds

    except Exception as exc:
        print(f"\n[WARNING] HeteroData dataset failed: {exc}")

        if args.fallback_homogeneous:
            print("Falling back to CollagenDockingDataset (homogeneous)...")
            from Graph_model.data.dataset import CollagenDockingDataset

            ds = CollagenDockingDataset(
                include_transfer=False,
                include_augment=False,
                include_mmp1=not args.no_mmp1,
                use_cache=True,
                force_reload=args.force_reload,
            )
            ds.load(verbose=True)

            if len(ds) == 0:
                raise RuntimeError("Homogeneous dataset is also empty!")

            print(f"\n[FALLBACK] Using homogeneous dataset: {len(ds)} Data graphs")
            print("[NOTE] Models expecting HeteroData will need config overrides.")
            return ds
        else:
            raise


def _load_pdbbind_transfer(max_entries: int | None = 5000):
    """Load PDBbind transfer dataset from external_dataset/."""
    try:
        from Graph_model.train.pdbbind_dataset import PDBbindGraphDataset
        ds = PDBbindGraphDataset(max_entries=max_entries)
        if not ds.is_available():
            print("[Transfer] PDBbind index not found — skipping transfer data.")
            return None
        print("[Transfer] Loading PDBbind structures (this may take a moment)...")
        ds.load()
        n_total = len(ds.get_affinity_index())
        print(f"[Transfer] PDBbind: {len(ds)} graphs built "
              f"({n_total} in index, {n_total - len(ds)} without local structure files)")
        if len(ds) > 0:
            stats = ds.delta_g_stats()
            print(f"  ΔG range: [{stats['min']:.1f}, {stats['max']:.1f}] kcal/mol "
                  f"(mean={stats['mean']:.1f} ± {stats['std']:.1f})")
        else:
            print("  [WARNING] No PDBbind graphs could be built.")
            return None
        return ds
    except Exception as exc:
        print(f"[Transfer] PDBbind load failed: {exc}")
        return None


class _CombinedDataset:
    """Lightweight wrapper that concatenates two datasets."""

    def __init__(self, ds1, ds2):
        self._ds1 = ds1
        self._ds2 = ds2
        self._n1 = len(ds1)
        self._n2 = len(ds2)

    def __len__(self):
        return self._n1 + self._n2

    def __getitem__(self, idx):
        if idx < self._n1:
            return self._ds1[idx]
        return self._ds2[idx - self._n1]

    def len(self):
        return len(self)

    def get(self, idx):
        return self[idx]

    def summary(self):
        s1 = self._ds1.summary() if hasattr(self._ds1, 'summary') else {}
        return {"total": len(self), "anchor": self._n1, "transfer": self._n2, **s1}


def resolve_device(device_str: str | None):
    """Resolve device from CLI arg or auto-detect."""
    import torch

    if device_str:
        return torch.device(device_str)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Subprocess-based model training ───────────────────────────────────────────

def _train_single_model_entry(
    model_key: str, args: argparse.Namespace,
    results_dir: str, device_str: str,
) -> dict:
    """
    Train one model in the current process.  Called either directly in-process
    or via a subprocess wrapper.  Returns a summary dict (no nn.Module).
    """
    import torch
    from Graph_model.train.run_training import train_single_model, MODEL_REGISTRY

    device = torch.device(device_str)
    dataset = load_dataset(args)

    # Deterministic split shared across all models
    n = len(dataset)
    n_val = max(1, int(n * args.val_ratio))
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    train_idx = perm[:n_train].tolist()
    val_idx   = perm[n_train:].tolist()
    fold_indices = [{'train': train_idx, 'val': val_idx}]

    # Per-model batch size limit
    model_bs = args.batch_size
    max_bs = MODEL_REGISTRY[model_key].get('max_batch_size')
    if max_bs is not None and args.batch_size > max_bs:
        model_bs = max_bs

    result = train_single_model(
        model_key     = model_key,
        dataset       = dataset,
        results_dir   = results_dir,
        max_epochs    = args.max_epochs,
        batch_size    = model_bs,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        val_ratio     = args.val_ratio,
        device        = device,
        seed          = args.seed,
        fold_indices  = fold_indices,
    )

    # Return serialisable summary (no model object)
    return {
        "model_key":     model_key,
        "model_name":    result["model_name"],
        "best_epoch":    result["best_epoch"],
        "best_val_rmse": result["best_val_rmse"],
        "wall_time_s":   result["wall_time_s"],
        "train_losses":  result["train_losses"],
        "val_losses":    result["val_losses"],
        "val_metrics":   result["val_metrics"],
        "epoch_log_path": str(result["epoch_log_path"]),
    }


def _train_models_subprocess(
    args: argparse.Namespace,
    model_keys: list[str],
    results_dir: Path,
    device,
) -> dict[str, dict]:
    """
    Train each model in a *separate subprocess* so that MPS/Metal memory is
    fully reclaimed by the OS between models (Apple Silicon unified-memory
    issue).  Falls back to in-process training on failure.
    """
    import json, subprocess, tempfile

    all_results: dict[str, dict] = {}

    for i, key in enumerate(model_keys, 1):
        from Graph_model.train.run_training import MODEL_REGISTRY
        name = MODEL_REGISTRY[key]["name"]
        print(f"\n{'─'*60}")
        print(f"[{i}/{len(model_keys)}] Training {name}  (separate process)")
        print(f"{'─'*60}")

        # Build the subprocess command — re-invoke train_main with --models <key>
        # and a special --_subprocess flag so it trains in-process (no recursion)
        cmd = [
            sys.executable, "-m", "Graph_model.train_main",
            "--models", key,
            "--results-dir", str(results_dir),
            "--max-epochs", str(args.max_epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--weight-decay", str(args.weight_decay),
            "--warmup-epochs", str(args.warmup_epochs),
            "--patience", str(args.patience),
            "--val-ratio", str(args.val_ratio),
            "--seed", str(args.seed),
            "--_subprocess",  # signals single-model in-process mode
        ]
        if args.include_transfer:
            cmd += ["--include-transfer", "--max-transfer", str(args.max_transfer)]
        if args.no_mmp1:
            cmd.append("--no-mmp1")
        if args.no_bipartite:
            cmd.append("--no-bipartite")
        if args.max_records is not None:
            cmd += ["--max-records", str(args.max_records)]
        if args.force_reload:
            cmd.append("--force-reload")
        if args.device:
            cmd += ["--device", args.device]
        if args.verbose:
            cmd += ["-" + "v" * args.verbose]

        t0 = time.time()
        proc = subprocess.run(cmd, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"})
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"  [WARNING] Model {key} subprocess exited with code {proc.returncode}")
            all_results[key] = {
                "model_name": name, "model_key": key,
                "best_val_rmse": float("nan"), "best_epoch": -1,
                "wall_time_s": elapsed, "error": f"exit code {proc.returncode}",
            }
            continue

        # Read back the results from the JSON log file
        log_path = results_dir / f"option_{key.lower()}_training.json"
        if log_path.exists():
            with open(log_path) as f:
                log_data = json.load(f)
            summary = log_data.get("summary", {})
            all_results[key] = {
                "model_name": name, "model_key": key,
                "best_val_rmse": summary.get("best_val_rmse", float("nan")),
                "best_epoch": summary.get("best_epoch", -1),
                "wall_time_s": elapsed,
                "val_metrics": [ep for ep in log_data.get("epochs", [])],
                "train_losses": [ep.get("train_loss", 0) for ep in log_data.get("epochs", [])],
                "val_losses": [ep.get("val_loss", 0) for ep in log_data.get("epochs", [])],
            }
            print(f"  Model {key} done: best_val_RMSE={summary.get('best_val_rmse', 'N/A'):.4f}  "
                  f"best_epoch={summary.get('best_epoch', '?')}  time={elapsed:.1f}s")
        else:
            all_results[key] = {
                "model_name": name, "model_key": key,
                "best_val_rmse": float("nan"), "best_epoch": -1,
                "wall_time_s": elapsed, "error": "no log file found",
            }

    # Write comparison summary
    from datetime import datetime
    comparison = {"created_at": datetime.now().isoformat(), "models": {}}
    for key, res in all_results.items():
        comparison["models"][key] = {
            "name": res.get("model_name", key),
            "best_epoch": res.get("best_epoch", -1),
            "best_val_rmse": res.get("best_val_rmse", float("nan")),
            "wall_time_s": res.get("wall_time_s", 0),
        }
    comp_path = results_dir / "comparison_summary.json"
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)

    return all_results


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    import torch
    print(f"PyTorch {torch.__version__}")
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print()

    # ── Subprocess mode: train single model and exit ──────────────────────────
    if getattr(args, '_subprocess', False):
        t0 = time.time()
        dataset = load_dataset(args)
        dt = time.time() - t0
        print(f"Dataset loaded in {dt:.1f}s\n")

        from Graph_model.train.run_training import train_all_models
        # train_all_models handles a single key just fine
        train_all_models(
            dataset=dataset,
            model_keys=args.models,
            results_dir=str(Path(args.results_dir)),
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            val_ratio=args.val_ratio,
            device=device,
            seed=args.seed,
        )
        return

    # ── Load dataset ──────────────────────────────────────────────────────────
    t0 = time.time()
    dataset = load_dataset(args)
    dt = time.time() - t0
    print(f"Dataset loaded in {dt:.1f}s\n")

    # ── Import training functions ─────────────────────────────────────────────
    from Graph_model.train.run_training import (
        train_all_models,
        train_single_model,
        train_lolo_cv,
        MODEL_REGISTRY,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_keys = args.models
    if model_keys:
        # Validate
        valid = set(MODEL_REGISTRY.keys())
        for k in model_keys:
            if k.upper() not in valid:
                print(f"[ERROR] Unknown model key '{k}'. Valid: {sorted(valid)}")
                sys.exit(1)
        model_keys = [k.upper() for k in model_keys]
    else:
        model_keys = list(MODEL_REGISTRY.keys())

    print("=" * 60)
    print(f"Training {len(model_keys)} model(s): {model_keys}")
    print(f"  max_epochs  = {args.max_epochs}")
    print(f"  batch_size  = {args.batch_size}")
    print(f"  lr          = {args.lr}")
    print(f"  weight_decay= {args.weight_decay}")
    print(f"  warmup      = {args.warmup_epochs} epochs")
    print(f"  patience    = {args.patience} epochs")
    print(f"  val_ratio   = {args.val_ratio}")
    print(f"  seed        = {args.seed}")
    print(f"  results_dir = {results_dir}")
    print(f"  device      = {device}")
    print("=" * 60)
    print()

    t_start = time.time()

    # ── LOLO-CV mode ──────────────────────────────────────────────────────────
    if args.lolo_cv:
        for key in model_keys:
            name = MODEL_REGISTRY[key]["name"]
            print(f"\n{'─'*60}")
            print(f"LOLO-CV: {name}")
            print(f"{'─'*60}")

            result = train_lolo_cv(
                model_key=key,
                dataset=dataset,
                results_dir=str(results_dir / "lolo_cv"),
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                patience=args.patience,
                val_ratio=args.val_ratio,
                device=device,
                seed=args.seed,
            )

            agg = result["aggregate"]
            print(f"\n[LOLO-CV] {name} aggregate:")
            for metric, val in agg.items():
                print(f"  {metric}: {val}")

    # ── Standard training ─────────────────────────────────────────────────────
    else:
        # Train each model in a separate subprocess to prevent MPS memory
        # accumulation across models (Apple Silicon unified memory issue).
        all_results = _train_models_subprocess(args, model_keys, results_dir, device)

        # ── Summary ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        for key, res in all_results.items():
            name = res.get("model_name", key)
            best_rmse = res.get("best_val_rmse", float("nan"))
            best_ep = res.get("best_epoch", -1)
            wall = res.get("wall_time_s", 0.0)
            print(f"  {key} ({name}): val_RMSE={best_rmse:.4f}  "
                  f"best_epoch={best_ep}  time={wall:.1f}s")

    total_time = time.time() - t_start
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results saved to: {results_dir.resolve()}")


if __name__ == "__main__":
    main()
