#!/usr/bin/env python3
"""
Leave-One-Ligand-Out cross-validation runner.

This is the honest evaluation protocol for this dataset: with only 9 unique
molecules, any row-level split leaks molecular identity (see
Graph_model/train/lolo_cv.py). Report LOLO numbers, and report them next to
the trivial baselines below — a model that cannot beat "predict this ligand's
mean" has not learned anything about chemistry.

Usage
-----
    export COLLAGEN_DATA_ROOT=/path/to/Jupyter_Dock
    python scripts/run_lolo.py --model A --epochs 50 --seeds 0 1 2
    python scripts/run_lolo.py --baselines-only
    python scripts/run_lolo.py --check          # environment/data preflight
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ── Determinism ───────────────────────────────────────────────────────────────

def set_all_seeds(seed: int, strict: bool = True) -> None:
    """
    Seed every RNG that affects a run.

    The training code previously seeded only torch, which left numpy and the
    stdlib `random` module free-running — two runs with the same --seed were
    not the same run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:  # older torch / unsupported backend
            logging.getLogger(__name__).warning(
                "Could not enable deterministic algorithms: %s", exc)


# ── Preflight ─────────────────────────────────────────────────────────────────

def preflight(verbose: bool = True) -> int:
    """Check imports and data presence. Returns a process exit code."""
    problems: list[str] = []

    for mod in ("torch", "torch_geometric", "rdkit", "pandas", "numpy"):
        try:
            __import__(mod)
        except ImportError as exc:
            problems.append(f"Missing dependency {mod!r}: {exc}")

    try:
        from Graph_model.data.config import (
            REPO_ROOT, check_data_present, optional_data_warnings, ensure_dirs)
        if verbose:
            print(f"Data root : {REPO_ROOT}")
            print(f"  (from ${{COLLAGEN_DATA_ROOT}})"
                  if os.environ.get("COLLAGEN_DATA_ROOT")
                  else "  (defaulted to the repository root)")
        problems.extend(check_data_present())
        if verbose:
            for w in optional_data_warnings():
                print(f"  warning: {w}")
        ensure_dirs()
    except Exception as exc:
        problems.append(f"Could not import Graph_model.data.config: {exc}")

    if problems:
        print("\nPreflight FAILED:\n")
        for p in problems:
            print(f"  * {p}")
        print("\nThe docking CSVs are not in this repository. Point "
              "COLLAGEN_DATA_ROOT at the checkout that contains them.")
        return 1

    if verbose:
        print("Preflight OK.")
    return 0


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baselines(dataset, val_ratio: float, seed: int) -> dict:
    """
    Trivial predictors evaluated under the *same* LOLO folds as the models.

    Note the asymmetry that makes LOLO honest: the held-out ligand's own mean
    is unavailable at test time, so `per_ligand_mean` necessarily falls back to
    the global training mean. If a trained model does not clearly beat
    `global_mean` here, it has not generalised across molecules.
    """
    import numpy as np
    from Graph_model.train.lolo_cv import LOLOCVSplitter

    splitter = LOLOCVSplitter(val_ratio=val_ratio, seed=seed)
    rows: dict[str, list[float]] = {"global_mean": [], "per_box_mean": []}

    def _y(i):
        return float(dataset[i].y.item())

    def _box(i):
        return str(getattr(dataset[i], "docking_box", "NA"))

    for fold in splitter.split(dataset):
        LOLOCVSplitter.verify_no_leakage(fold)

        train_y = np.array([_y(i) for i in fold.train_idx])
        test_y = np.array([_y(i) for i in fold.test_idx])
        if test_y.size == 0:
            continue

        # 1. global training mean
        pred = np.full_like(test_y, train_y.mean())
        rows["global_mean"].append(float(np.sqrt(((pred - test_y) ** 2).mean())))

        # 2. per-docking-box training mean, backing off to the global mean
        box_mean: dict[str, float] = {}
        for i in fold.train_idx:
            box_mean.setdefault(_box(i), []).append(_y(i))
        box_mean = {k: float(np.mean(v)) for k, v in box_mean.items()}
        pred_b = np.array([box_mean.get(_box(i), train_y.mean())
                           for i in fold.test_idx])
        rows["per_box_mean"].append(
            float(np.sqrt(((pred_b - test_y) ** 2).mean())))

    import statistics
    return {
        name: {
            "rmse_mean": statistics.fmean(v) if v else float("nan"),
            "rmse_std": statistics.pstdev(v) if len(v) > 1 else 0.0,
            "n_folds": len(v),
        }
        for name, v in rows.items()
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run Leave-One-Ligand-Out CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="A",
                    help="Model registry key: A B C D E F G H I")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="Run once per seed and aggregate across seeds.")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--max-records", type=int, default=None,
                    help="Truncate the dataset (smoke tests only).")
    ap.add_argument("--no-mmp1", action="store_true")
    ap.add_argument("--force-reload", action="store_true",
                    help="Ignore the cached graphs and rebuild.")
    ap.add_argument("--baselines-only", action="store_true",
                    help="Compute trivial baselines and exit.")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="Run the preflight check and exit.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    rc = preflight()
    if args.check or rc:
        return rc

    from Graph_model.data.config import ensure_dirs
    from Graph_model.data.hetero_dataset import HeteroDockingDataset
    from Graph_model.train.run_training import train_lolo_cv

    ensure_dirs()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset ...")
    dataset = HeteroDockingDataset(
        include_mmp1=not args.no_mmp1,
        force_reload=args.force_reload,
        max_records=args.max_records,
    ).load(verbose=True)
    print(f"Dataset: {len(dataset)} graphs")

    report: dict = {
        "n_graphs": len(dataset),
        "model": args.model,
        "seeds": args.seeds,
        "epochs": args.epochs,
    }

    if not args.skip_baselines:
        print("\nBaselines (LOLO folds, seed %d):" % args.seeds[0])
        report["baselines"] = run_baselines(dataset, args.val_ratio, args.seeds[0])
        for name, m in report["baselines"].items():
            print(f"  {name:<16} RMSE {m['rmse_mean']:.4f} "
                  f"+/- {m['rmse_std']:.4f}  ({m['n_folds']} folds)")

    if args.baselines_only:
        (results_dir / "baselines.json").write_text(json.dumps(report, indent=2))
        return 0

    runs = []
    for seed in args.seeds:
        print(f"\n=== {args.model}  seed {seed} ===")
        set_all_seeds(seed)
        out = train_lolo_cv(
            args.model,
            dataset,
            results_dir=results_dir / f"{args.model}_seed{seed}",
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            seed=seed,
        )
        runs.append(out.get("aggregate", {}))
        print(f"seed {seed}: {out.get('aggregate', {})}")

    report["runs"] = runs
    out_path = results_dir / f"lolo_{args.model}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
