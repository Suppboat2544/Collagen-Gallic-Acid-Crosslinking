#!/usr/bin/env python3
"""Harvest completed redock SDF poses into the B6 state JSON (resume-safe).

Use when the ProcessPool parent is slow to checkpoint, or after a crash:
poses already on disk are recorded so --resume will skip them.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def affinity_from_sdf(path: Path) -> float | None:
    from rdkit import Chem
    try:
        mol = next(m for m in Chem.SDMolSupplier(str(path), removeHs=False) if m)
    except Exception:
        return None
    for key in ("minimizedAffinity", "affinity", "CNNscore"):
        if mol.HasProp(key):
            try:
                return float(mol.GetProp(key))
            except ValueError:
                pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=Path, required=True)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--since-epoch", type=float, default=None,
                    help="Only harvest SDFs with mtime >= this UNIX epoch")
    ap.add_argument("--min-heavy-atoms", type=int, default=None,
                    help="Reject poses with fewer heavy atoms (e.g. 60 for true PGG)")
    args = ap.parse_args()

    state = {"results": {}}
    if args.state.exists():
        state = json.loads(args.state.read_text())
        state.setdefault("results", {})

    n_new = 0
    for lig in args.ligands:
        for sdf in args.anchor.glob(f"*/docked_{lig}.sdf"):
            if sdf.stat().st_size < 500:
                continue
            if args.since_epoch is not None and sdf.stat().st_mtime < args.since_epoch:
                continue
            sid = sdf.parent.name
            if sid in state["results"] and state["results"][sid].get("ok"):
                continue
            e = affinity_from_sdf(sdf)
            if e is None:
                continue
            if args.min_heavy_atoms is not None:
                from rdkit import Chem
                mol = next((m for m in Chem.SDMolSupplier(str(sdf), removeHs=True) if m), None)
                if mol is None or mol.GetNumHeavyAtoms() < args.min_heavy_atoms:
                    continue
            state["results"][sid] = {
                "sample_id": sid,
                "ok": True,
                "best_energy_kcalmol": e,
                "all_energies": [e],
                "seconds": None,
                "returncode": 0,
                "stderr_tail": "harvested_from_sdf",
                "out_sdf": str(sdf),
                "rel_sdf": f"collagen_gallic_results/{sid}/docked_{lig}.sdf",
            }
            n_new += 1

    # MMP-1 PGG
    for sdf in args.anchor.glob("MMP1_*/docked_pentagalloylglucose.sdf"):
        if sdf.stat().st_size < 500:
            continue
        if args.since_epoch is not None and sdf.stat().st_mtime < args.since_epoch:
            continue
        sid = sdf.parent.name
        if sid in state["results"] and state["results"][sid].get("ok"):
            continue
        e = affinity_from_sdf(sdf)
        if e is None:
            continue
        state["results"][sid] = {
            "sample_id": sid,
            "ok": True,
            "best_energy_kcalmol": e,
            "all_energies": [e],
            "seconds": None,
            "returncode": 0,
            "stderr_tail": "harvested_from_sdf",
            "out_sdf": str(sdf),
            "rel_sdf": f"collagen_gallic_results/{sid}/docked_pentagalloylglucose.sdf",
        }
        n_new += 1

    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state["harvest_note"] = f"added {n_new} from on-disk SDFs"
    args.state.write_text(json.dumps(state, indent=2))
    ok = sum(1 for v in state["results"].values() if v.get("ok"))
    print(f"harvested +{n_new}; total ok in state={ok}; wrote {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
