#!/usr/bin/env python3
"""
Re-dock ligands whose released poses were the wrong molecules (B6.1–B6.3).

Rebuilds poses + updates collagen_crosslinking_docking_results.csv (and MMP-1
rows for pentagalloylglucose) after verifying catalogue identity via
Graph_model.data.provenance.

Usage
-----
    export COLLAGEN_DATA_ROOT=/path/to/Jupyter_Dock
    python scripts/redock_corrected_ligands.py --workers 10
    python scripts/redock_corrected_ligands.py --workers 10 --resume
    python scripts/redock_corrected_ligands.py --ligands NHS --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LIGANDS = ("pentagalloylglucose", "NHS", "NHS_ester_intermediate")


def _smina_bin(root: Path) -> Path:
    cand = [
        root / "bin" / "smina.osx",
        root / "bin" / "smina",
    ]
    if platform.system() == "Darwin":
        cand = [cand[0], cand[1]]
    else:
        cand = [cand[1], cand[0]]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError("smina binary not found under bin/")


def _receptor_for_ph(anchor: Path, ph: float) -> Path:
    # Prefer the FULL pH-protonated receptor. The *_bindingsite*.pdb files are
    # tiny PLIP extracts (~80 atoms) and must never be used for docking — they
    # yield null Vinardo scores (all affinities 0.0).
    if abs(ph - 5.5) < 1e-6:
        tag = "5.5"
    elif abs(ph - 5.0) < 1e-6:
        tag = "5"
    elif abs(ph - 7.0) < 1e-6:
        tag = "7"
    else:
        tag = f"{ph:g}"
    candidates = [
        anchor / f"pig_collagen_I_alpha2_pH{tag}_H_fix.pdb",
        anchor / f"pig_collagen_I_alpha2_pH{tag}_H.pdb",
        anchor / f"pig_collagen_I_alpha2_pH{tag}_H_fix_clean.pdb",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"No collagen receptor for pH {ph}: tried {candidates}")


def _mmp1_receptor(anchor: Path, target: str) -> Path:
    # target column values like porcine_MMP1_966C
    name = target.replace("porcine_", "") if target.startswith("porcine_") else target
    # Files are named porcine_MMP1_*.pdb in anchor
    for cand in [
        anchor / f"{target}.pdb",
        anchor / f"{target}_H_fix.pdb",
        anchor / f"porcine_{name}.pdb",
    ]:
        if cand.exists():
            return cand
    # common prepared names
    for p in sorted(anchor.glob(f"*{name}*.pdb")):
        if "bindingsite" not in p.name.lower():
            return p
    raise FileNotFoundError(f"MMP-1 receptor PDB not found for target={target}")


def _parse_smina_energies(stdout: str) -> tuple[float | None, list[float]]:
    best = None
    all_e: list[float] = []
    found = False
    for line in stdout.splitlines():
        if "-----+----" in line:
            found = True
            continue
        if found and line.strip():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    e = float(parts[1])
                except ValueError:
                    continue
                all_e.append(e)
                if best is None:
                    best = e
    return best, all_e


def _dock_one(job: dict) -> dict:
    """Worker: run one smina job. Must be top-level for pickling."""
    cmd = [
        job["smina"],
        "-r", job["receptor"],
        "-l", job["ligand_mol2"],
        "--center_x", str(job["cx"]),
        "--center_y", str(job["cy"]),
        "--center_z", str(job["cz"]),
        "--size_x", str(job["size"]),
        "--size_y", str(job["size"]),
        "--size_z", str(job["size"]),
        "-o", job["out_sdf"],
        "--exhaustiveness", str(job["exhaustiveness"]),
        "--num_modes", str(job["num_modes"]),
        "--scoring", "vinardo",
        "--cpu", "1",
        "--seed", str(job["seed"]),
    ]
    Path(job["out_sdf"]).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=job["timeout"])
        best, all_e = _parse_smina_energies(r.stdout)
        ok = r.returncode == 0 and best is not None and Path(job["out_sdf"]).exists()
        return {
            "sample_id": job["sample_id"],
            "ok": ok,
            "best_energy_kcalmol": best,
            "all_energies": all_e,
            "seconds": round(time.time() - t0, 2),
            "returncode": r.returncode,
            "stderr_tail": (r.stderr or "")[-300:],
            "out_sdf": job["out_sdf"],
            "rel_sdf": job["rel_sdf"],
        }
    except Exception as exc:
        return {
            "sample_id": job["sample_id"],
            "ok": False,
            "best_energy_kcalmol": None,
            "all_energies": [],
            "seconds": round(time.time() - t0, 2),
            "returncode": -1,
            "stderr_tail": str(exc),
            "out_sdf": job["out_sdf"],
            "rel_sdf": job["rel_sdf"],
        }


def build_jobs(args) -> list[dict]:
    import pandas as pd
    from Graph_model.data.config import ANCHOR_DIR, REPO_ROOT

    root = REPO_ROOT
    anchor = ANCHOR_DIR
    smina = str(_smina_bin(root))
    state_path = Path(args.state)
    done: set[str] = set()
    if args.resume and state_path.exists():
        prev = json.loads(state_path.read_text())
        done = {k for k, v in prev.get("results", {}).items() if v.get("ok")}

    ligands = tuple(args.ligands) if args.ligands else LIGANDS
    for lig in ligands:
        mol2 = anchor / f"{lig}.mol2"
        if not mol2.exists():
            raise FileNotFoundError(f"Corrected ligand mol2 missing: {mol2}")

    jobs: list[dict] = []

    # Collagen campaign
    col = pd.read_csv(anchor / "collagen_crosslinking_docking_results.csv")
    col = col[col["ligand"].isin(ligands)].copy()
    if args.max_jobs:
        col = col.head(args.max_jobs)

    for _, row in col.iterrows():
        sid = str(row["sample_id"])
        if sid in done:
            continue
        lig = str(row["ligand"])
        ph = float(row["pH"])
        t_c = int(row["temperature_C"])
        if getattr(args, "collapse_temperature", False):
            # One physical dock per (ligand, box, pH); replicate across T later.
            # Use T=25 as the canonical sample_id for the docked pose file.
            if int(t_c) != 25:
                continue
        # Seed from (ligand, box, pH, T) so T replicates differ stochastically
        seed = abs(hash((lig, row["docking_box"], ph, t_c))) % (2**31 - 1)
        out_dir = anchor / sid
        out_sdf = out_dir / f"docked_{lig}.sdf"
        rel = f"collagen_gallic_results/{sid}/docked_{lig}.sdf"
        exh = int(args.exhaustiveness) if args.exhaustiveness else int(row.get("exhaustiveness", 16) or 16)
        jobs.append({
            "sample_id": sid,
            "kind": "collagen",
            "smina": smina,
            "receptor": str(_receptor_for_ph(anchor, ph)),
            "ligand_mol2": str(anchor / f"{lig}.mol2"),
            "cx": float(row["box_center_x"]),
            "cy": float(row["box_center_y"]),
            "cz": float(row["box_center_z"]),
            "size": float(row.get("box_size_A", 20.0) or 20.0),
            "out_sdf": str(out_sdf),
            "rel_sdf": rel,
            "exhaustiveness": exh,
            "num_modes": int(row.get("num_modes", 9) or 9),
            "seed": seed if not args.fixed_seed else int(args.fixed_seed),
            "timeout": args.timeout,
        })

    # MMP-1 PGG rows (NHS not in MMP-1 campaign)
    mmp_csv = anchor / "mmp1_collagenase_docking_results.csv"
    if mmp_csv.exists() and "pentagalloylglucose" in ligands and not args.skip_mmp1:
        mmp = pd.read_csv(mmp_csv)
        mmp = mmp[mmp["ligand"] == "pentagalloylglucose"].copy()
        # Prefer reconstructed box sidecar if present
        side = anchor / "mmp1_box_centres_reconstructed.csv"
        box_map: dict[tuple[str, str], tuple[float, float, float]] = {}
        if side.exists():
            import csv as _csv
            with open(side, newline="") as fh:
                for r in _csv.DictReader(fh):
                    box_map[(r["receptor"], r["docking_box"])] = (
                        float(r["box_center_x_reconstructed"]),
                        float(r["box_center_y_reconstructed"]),
                        float(r["box_center_z_reconstructed"]),
                    )
        for _, row in mmp.iterrows():
            sid = str(row["sample_id"])
            if sid in done:
                continue
            target = str(row["target"])
            box = str(row["docking_box"])
            if (target, box) in box_map:
                cx, cy, cz = box_map[(target, box)]
            elif "box_center_x" in row and pd.notna(row.get("box_center_x")):
                cx, cy, cz = float(row["box_center_x"]), float(row["box_center_y"]), float(row["box_center_z"])
            else:
                # Cannot dock without a box; skip with note in state later
                continue
            out_dir = anchor / sid
            out_sdf = out_dir / f"docked_pentagalloylglucose.sdf"
            rel = f"collagen_gallic_results/{sid}/docked_pentagalloylglucose.sdf"
            seed = abs(hash((sid,))) % (2**31 - 1)
            jobs.append({
                "sample_id": sid,
                "kind": "mmp1",
                "smina": smina,
                "receptor": str(_mmp1_receptor(anchor, target)),
                "ligand_mol2": str(anchor / "pentagalloylglucose.mol2"),
                "cx": cx, "cy": cy, "cz": cz,
                "size": 20.0,
                "out_sdf": str(out_sdf),
                "rel_sdf": rel,
                "exhaustiveness": 16,
                "num_modes": 9,
                "seed": seed if not args.fixed_seed else int(args.fixed_seed),
                "timeout": args.timeout,
            })

    return jobs


def apply_results_to_csv(results: dict[str, dict], args) -> None:
    import pandas as pd
    from Graph_model.data.config import ANCHOR_DIR, LIGAND_CATALOGUE
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    anchor = ANCHOR_DIR
    ligands = tuple(args.ligands) if args.ligands else LIGANDS

    def _props(name: str) -> dict:
        smi = LIGAND_CATALOGUE[name]["smiles"]
        if isinstance(smi, (tuple, list)):
            smi = "".join(smi)
        mol = Chem.MolFromSmiles(smi)
        return {
            "ligand_smiles": smi,
            "ligand_formula": rdMolDescriptors.CalcMolFormula(mol),
            "ligand_MW": round(Descriptors.MolWt(mol), 2),
            "ligand_LogP": round(Descriptors.MolLogP(mol), 2),
            "ligand_HBD": rdMolDescriptors.CalcNumHBD(mol),
            "ligand_HBA": rdMolDescriptors.CalcNumHBA(mol),
            "ligand_TPSA": round(Descriptors.TPSA(mol), 1),
            "ligand_RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
        }

    props = {n: _props(n) for n in ligands}

    path = anchor / "collagen_crosslinking_docking_results.csv"
    df = pd.read_csv(path)

    # Map sample_id -> result; if collapse-temperature, also map sibling T rows
    # from the T=25 canonical dock via (ligand, pH, docking_box).
    lookup: dict[str, dict] = {sid: r for sid, r in results.items() if r.get("ok")}
    if getattr(args, "collapse_temperature", False):
        by_key: dict[tuple, dict] = {}
        for sid, r in lookup.items():
            # sample_id: {ligand}_pH{x}_T{t}_{box}
            parts = sid.split("_pH", 1)
            if len(parts) != 2:
                continue
            lig = parts[0]
            rest = parts[1]
            import re as _re
            m = _re.match(r"([\d.]+)_T(\d+)_(.+)$", rest)
            if not m:
                continue
            ph_s, t_s, box = m.group(1), m.group(2), m.group(3)
            if t_s == "25":
                by_key[(lig, ph_s, box)] = r
        for i, row in df.iterrows():
            lig = str(row["ligand"])
            if lig not in props:
                continue
            key = (lig, f"{float(row['pH']):g}".replace(".0", "") if float(row["pH"]) in (5.0, 7.0) else str(row["pH"]),
                   str(row["docking_box"]))
            # Normalize pH key to match sample_id convention (5, 5.5, 7)
            ph = float(row["pH"])
            ph_s = "5.5" if abs(ph - 5.5) < 1e-6 else str(int(ph)) if abs(ph - int(ph)) < 1e-6 else f"{ph:g}"
            key = (lig, ph_s, str(row["docking_box"]))
            if key in by_key:
                lookup[str(row["sample_id"])] = by_key[key]

    n_upd = 0
    for i, row in df.iterrows():
        sid = str(row["sample_id"])
        if sid not in lookup:
            continue
        lig = str(row["ligand"])
        if lig not in props:
            continue
        r = lookup[sid]
        df.at[i, "best_energy_kcalmol"] = r["best_energy_kcalmol"]
        # Keep per-row sdf path; for collapsed T siblings point at T=25 pose if present
        df.at[i, "sdf_file"] = r.get("rel_sdf", row.get("sdf_file"))
        df.at[i, "generated_at"] = datetime.now(timezone.utc).isoformat()
        for k, v in props[lig].items():
            if k in df.columns:
                df.at[i, k] = v
        n_upd += 1
    bak = path.with_suffix(".csv.pre_b6_redock")
    if not bak.exists():
        path.replace(bak)
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Updated {n_upd} collagen CSV rows -> {path}")

    # MMP-1
    mpath = anchor / "mmp1_collagenase_docking_results.csv"
    if mpath.exists():
        mdf = pd.read_csv(mpath)
        n_m = 0
        for i, row in mdf.iterrows():
            sid = str(row["sample_id"])
            if sid not in results or not results[sid].get("ok"):
                continue
            r = results[sid]
            mdf.at[i, "best_energy_kcalmol"] = r["best_energy_kcalmol"]
            mdf.at[i, "n_modes"] = len(r.get("all_energies") or [])
            mdf.at[i, "all_energies"] = str(r.get("all_energies") or [])
            mdf.at[i, "sdf_file"] = r["rel_sdf"]
            mdf.at[i, "generated_at"] = datetime.now(timezone.utc).isoformat()
            n_m += 1
        mbak = mpath.with_suffix(".csv.pre_b6_redock")
        if not mbak.exists():
            import shutil
            shutil.copy2(mpath, mbak)
        mdf.to_csv(mpath, index=False)
        print(f"Updated {n_m} MMP-1 CSV rows -> {mpath}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ligands", nargs="+", default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=3600,
                    help="Per-job smina timeout in seconds (PGG needs >>15 min)")
    ap.add_argument("--exhaustiveness", type=int, default=None,
                    help="Override CSV exhaustiveness (default: keep CSV value)")
    ap.add_argument("--collapse-temperature", action="store_true",
                    help="Dock only T=25 per (ligand,box,pH); replicate to T=4/37 on apply")
    ap.add_argument("--fixed-seed", type=int, default=None)
    ap.add_argument("--skip-mmp1", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply-only", action="store_true",
                    help="Only merge state JSON into CSVs")
    ap.add_argument("--state", default=None)
    args = ap.parse_args(argv)

    from Graph_model.data.config import ANCHOR_DIR, ensure_dirs
    ensure_dirs()
    if args.state is None:
        args.state = str(ANCHOR_DIR / "_redock_b6_state.json")

    # Require corrected structures
    from Graph_model.data.provenance import check_structures
    bad = [c for c in check_structures() if c.ligand in (args.ligands or LIGANDS) and not c.ok]
    if bad and not args.apply_only:
        print("Ligand identity check FAILED — rebuild SDF/MOL2 before docking:")
        for c in bad:
            print(f"  {c.ligand}: {c.detail or c.status}")
        return 1

    state_path = Path(args.state)
    state = {"results": {}, "started_at": datetime.now(timezone.utc).isoformat()}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.setdefault("results", {})

    if args.apply_only:
        apply_results_to_csv(state["results"], args)
        return 0

    jobs = build_jobs(args)
    print(f"Jobs pending: {len(jobs)}  (workers={args.workers})")
    if args.dry_run:
        for j in jobs[:5]:
            print(" ", j["sample_id"], j["kind"], j["receptor"])
        return 0

    if not jobs:
        print("Nothing to do.")
        apply_results_to_csv(state["results"], args)
        return 0

    ok_n = fail_n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_dock_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            state["results"][res["sample_id"]] = res
            if res["ok"]:
                ok_n += 1
            else:
                fail_n += 1
                print(f"FAIL {res['sample_id']}: {res.get('stderr_tail','')[:120]}")
            if i % 25 == 0 or i == len(jobs):
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                state["progress"] = {"done": i, "total": len(jobs), "ok": ok_n, "fail": fail_n}
                state_path.write_text(json.dumps(state, indent=2))
                print(f"[{i}/{len(jobs)}] ok={ok_n} fail={fail_n}")

    state_path.write_text(json.dumps(state, indent=2))
    apply_results_to_csv(state["results"], args)
    print(f"Finished. ok={ok_n} fail={fail_n} state={state_path}")
    return 0 if fail_n == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
