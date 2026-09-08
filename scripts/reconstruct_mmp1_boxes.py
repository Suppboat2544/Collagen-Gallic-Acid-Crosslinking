#!/usr/bin/env python3
"""
Reconstruct the missing MMP-1 docking-box centres from the docked poses.

The problem
-----------
`mmp1_collagenase_docking_results.csv` has 13 columns. The collagen CSV has 44,
including `box_center_x/y/z` and `box_size_A`. The MMP-1 file carries none of
them. `hetero_dataset.py` reads the centre with

    box_cx = _safe_float(row.get("box_center_x", 0.0), 0.0)

so every MMP-1 record gets a box at the coordinate origin. The receptor sits at
z in [13.45, 57.92], tens of angstroms away, so the 20 A box intersects nothing
and EVERY MMP-1 graph is built with an empty receptor. This is a pre-existing
defect: the authors' own shipped cache has 40/40 empty, and pristine
upstream/main@5cb24be reproduces 120/120 empty.

What this script does
---------------------
The box geometry was never written down, but the docked poses were. A pose can
only be generated inside its search box, so the poses are physical evidence of
where the box was. For each (receptor, docking_box) group this script takes the
centroid of every heavy atom across all poses in the group and reports it as an
ESTIMATE of the box centre, together with the spread that says how much to
trust it.

What it deliberately does NOT do
--------------------------------
* It does not modify the shipped CSVs. Output goes to a NEW sidecar file.
* It does not invent a value for any group with no poses. Those are reported as
  unrecoverable and left empty; there is nothing to infer them from.
* It does not guess `box_size_A`. Pose extent is a lower bound on box size, not
  a measurement of it, so the column is emitted as a lower bound under a name
  that says so.
* It is not wired into training. Reconstructed geometry is not measured
  geometry, and whether to use it is the authors' call, not a default.

Validation
----------
Run with `--validate` to apply the identical procedure to the COLLAGEN set,
where the true centres are known, and print the reconstruction error. That
number is the honest uncertainty to attach to the MMP-1 estimates.

Usage
-----
    python scripts/reconstruct_mmp1_boxes.py --poses <dir> --validate
    python scripts/reconstruct_mmp1_boxes.py --poses <dir> \
        --out Phukhao/collagen_gallic_results/mmp1_box_centres_reconstructed.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Pose loading ─────────────────────────────────────────────────────────────

# Receptor names as they appear in the CSV `target` column. Matched
# longest-first so that "MMP1_966C_apo" is never truncated to "MMP1_966C" --
# the apo and holo structures are different receptors, and a regex that splits
# on the first plausible boundary silently merges them.
KNOWN_RECEPTORS: tuple[str, ...] = (
    "porcine_MMP1_966C_apo",
    "porcine_MMP1_966C",
    "porcine_MMP1_1SU3",
)


def _pose_key(filename: str,
              receptors: tuple[str, ...] = KNOWN_RECEPTORS) -> tuple[str, str] | None:
    """
    Split a pose path/filename into (receptor, docking_box).

    Supported layouts
    -----------------
    1. Flat Drive export:
           collagen_gallic_results__<sample_id>__docked_<ligand>.sdf
    2. Native campaign layout (this checkout):
           <results_dir>/<sample_id>/docked_<ligand>.sdf
       where the sample_id is the parent directory name.

    sample_id forms:
        <receptor_stem>_<ligand>_pH<x>_T<t>_<box>   for MMP-1
        <ligand>_pH<x>_T<t>_<box>                    for collagen

    The receptor is identified by prefix match against a known list rather than
    by regex, because receptor names are not separable from ligand names by
    punctuation alone.
    """
    path = Path(filename)
    base = path.name
    sample_id: str | None = None

    m = re.match(r"^(?:.*?__)?(.+?)__docked_.*\.sdf$", base)
    if m:
        sample_id = m.group(1)
    elif base.startswith("docked_") and base.endswith(".sdf"):
        # Native layout: parent directory is the sample_id
        parent = path.parent.name
        if parent and parent not in (".", ""):
            sample_id = parent
    if sample_id is None:
        sample_id = base[:-4] if base.endswith(".sdf") else base

    for receptor in sorted(receptors, key=len, reverse=True):
        stem = receptor.removeprefix("porcine_")
        if sample_id.startswith(stem + "_"):
            rest = sample_id[len(stem) + 1:]
            m = re.match(r"^.+?_pH[\d.]+_T\d+_(.+)$", rest)
            return (receptor, m.group(1)) if m else None

    m = re.match(r"^.+?_pH[\d.]+_T\d+_(.+)$", sample_id)
    if m:
        return "collagen", m.group(1)
    return None


def _heavy_atom_coords(path: Path) -> list[tuple[float, float, float]]:
    """All heavy-atom coordinates from every pose in an SDF."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    out: list[tuple[float, float, float]] = []
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=True, sanitize=False)
    except Exception:
        return out
    for mol in supplier:
        if mol is None or mol.GetNumConformers() == 0:
            continue
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
                continue
            p = conf.GetAtomPosition(i)
            out.append((p.x, p.y, p.z))
    return out


def collect_groups(pose_dir: Path, receptor_filter: str | None = None) -> dict:
    """Map (receptor, box) -> {'coords': [...], 'n_poses': int}."""
    groups: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"coords": [], "n_poses": 0, "files": []})
    for path in sorted(pose_dir.rglob("*.sdf")):
        # Skip catalogue ligand files sitting at the results root
        if path.parent.resolve() == pose_dir.resolve() and not path.name.startswith("docked_"):
            continue
        key = _pose_key(str(path))
        if key is None:
            continue
        if receptor_filter == "mmp1" and not key[0].startswith("porcine_MMP1"):
            continue
        if receptor_filter == "collagen" and key[0] != "collagen":
            continue
        coords = _heavy_atom_coords(path)
        if not coords:
            continue
        groups[key]["coords"].extend(coords)
        groups[key]["n_poses"] += 1
        groups[key]["files"].append(str(path.relative_to(pose_dir)))
    return dict(groups)


# ── Reconstruction ───────────────────────────────────────────────────────────

def reconstruct(groups: dict) -> list[dict]:
    """Centroid and dispersion per (receptor, box) group."""
    rows = []
    for (receptor, box), g in sorted(groups.items()):
        pts = g["coords"]
        n = len(pts)
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        cz = sum(p[2] for p in pts) / n

        # Radius of gyration about the centroid: how tightly the poses cluster.
        rg = math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
                           for p in pts) / n)
        span = max(max(p[i] for p in pts) - min(p[i] for p in pts)
                   for i in range(3))
        rows.append({
            "receptor": receptor,
            "docking_box": box,
            "box_center_x_reconstructed": round(cx, 3),
            "box_center_y_reconstructed": round(cy, 3),
            "box_center_z_reconstructed": round(cz, 3),
            "n_poses": g["n_poses"],
            "n_atoms": n,
            "pose_spread_rg_A": round(rg, 3),
            "pose_span_lower_bound_A": round(span, 3),
            "method": "centroid_of_docked_pose_heavy_atoms",
            "is_measured": "false",
        })
    return rows


# ── Validation against known collagen centres ────────────────────────────────

def validate_against_collagen(pose_dir: Path, collagen_csv: Path) -> int:
    """
    Apply the same procedure where the answer is known, and print the error.

    This is the number that quantifies how far the MMP-1 estimates can be
    trusted. Without it the reconstruction is an assertion.
    """
    if not collagen_csv.exists():
        print(f"Cannot validate: {collagen_csv} not found")
        return 1

    truth: dict[str, tuple[float, float, float]] = {}
    with open(collagen_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            box = (row.get("docking_box") or "").strip()
            try:
                c = (float(row["box_center_x"]), float(row["box_center_y"]),
                     float(row["box_center_z"]))
            except (KeyError, TypeError, ValueError):
                continue
            if box:
                truth.setdefault(box, c)

    groups = collect_groups(pose_dir, receptor_filter="collagen")
    rows = reconstruct(groups)

    errors: list[tuple[str, float, int]] = []
    for r in rows:
        t = truth.get(r["docking_box"])
        if t is None:
            continue
        d = math.dist(
            (r["box_center_x_reconstructed"], r["box_center_y_reconstructed"],
             r["box_center_z_reconstructed"]), t)
        errors.append((r["docking_box"], d, r["n_poses"]))

    if not errors:
        print("Cannot validate: no collagen box matched between poses and CSV")
        return 1

    errors.sort(key=lambda e: e[1])
    ds = [e[1] for e in errors]
    mean = sum(ds) / len(ds)
    median = sorted(ds)[len(ds) // 2]

    print("\n=== Validation on collagen (true centres known) ===")
    print(f"  boxes checked      : {len(errors)}")
    print(f"  mean error         : {mean:.2f} A")
    print(f"  median error       : {median:.2f} A")
    print(f"  best / worst       : {ds[0]:.2f} / {ds[-1]:.2f} A")

    multi = [e for e in errors if e[2] > 1]
    if multi:
        m = [e[1] for e in multi]
        print(f"  boxes with >1 pose : {len(multi)}, "
              f"mean error {sum(m)/len(m):.2f} A")
    print("\n  Read this as the uncertainty on every reconstructed MMP-1 centre.")
    print("  A docking box is typically 20 A across, so an error of this size")
    print("  recovers the right binding region, NOT the exact original box.")
    return 0


# ── Does the reconstructed box actually contain protein? ─────────────────────

def _pdb_atoms(path: Path) -> list[tuple[float, float, float]]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    out.append((float(line[30:38]), float(line[38:46]),
                                float(line[46:54])))
                except ValueError:
                    pass
    return out


def check_occupancy(rows: list[dict], receptor_dir: Path,
                    box_size: float = 20.0) -> int:
    """
    Count receptor atoms inside each box, before and after reconstruction.

    This is the test that matters: the defect is not "the centre is missing",
    it is "the resulting box is empty, so the model sees no protein". A
    reconstruction that still yields an empty box has fixed nothing.
    """
    half = box_size / 2.0
    receptor_files = {
        "porcine_MMP1_1SU3": "porcine_MMP1_1SU3.pdb",
        "porcine_MMP1_966C": "porcine_MMP1_966C.pdb",
        "porcine_MMP1_966C_apo": "porcine_MMP1_966C_apo.pdb",
    }

    print(f"\n=== Receptor occupancy of a {box_size:.0f} A box ===")
    still_empty: list[tuple[str, str]] = []
    for receptor, fname in sorted(receptor_files.items()):
        path = receptor_dir / fname
        if not path.exists():
            print(f"  {receptor}: {fname} not found, skipped")
            continue
        atoms = _pdb_atoms(path)
        mine = [r for r in rows if r["receptor"] == receptor]
        if not atoms or not mine:
            continue

        at_origin = sum(1 for a in atoms if all(abs(a[i]) <= half for i in range(3)))
        counts = []
        for r in mine:
            c = (r["box_center_x_reconstructed"], r["box_center_y_reconstructed"],
                 r["box_center_z_reconstructed"])
            n = sum(1 for a in atoms
                    if all(abs(a[i] - c[i]) <= half for i in range(3)))
            counts.append(n)
            if n == 0:
                still_empty.append((receptor, r["docking_box"]))

        ordered = sorted(counts)
        print(f"  {receptor}  ({len(atoms)} atoms, {len(mine)} boxes)")
        print(f"    box at origin (current code) : {at_origin} atoms  <- the defect")
        print(f"    reconstructed centres        : median {ordered[len(ordered)//2]}, "
              f"min {ordered[0]}, max {ordered[-1]}")
        print(f"    boxes still empty            : "
              f"{sum(1 for c in counts if c == 0)}/{len(counts)}")

    if still_empty:
        print(f"\n  {len(still_empty)} reconstructed box(es) still contain no "
              f"protein. Their poses sit off the receptor surface, so the")
        print("  centroid is not a usable box centre. These need the original "
              "docking configuration:")
        for rec, box in still_empty:
            print(f"      {rec} / {box}")
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from Graph_model.data.config import ANCHOR_DIR

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--poses", type=Path, required=True,
                   help="directory containing docked pose .sdf files")
    p.add_argument("--out", type=Path, default=None,
                   help="sidecar CSV to write (never modifies the source data)")
    p.add_argument("--validate", action="store_true",
                   help="also reconstruct collagen boxes and report the error "
                        "against their known centres")
    p.add_argument("--collagen-csv", type=Path,
                   default=ANCHOR_DIR / "collagen_crosslinking_docking_results.csv")
    p.add_argument("--mmp1-csv", type=Path,
                   default=ANCHOR_DIR / "mmp1_collagenase_docking_results_FULL_335.csv")
    p.add_argument("--check-occupancy", action="store_true",
                   help="count receptor atoms inside each reconstructed box, "
                        "against the current all-zero-centre behaviour")
    p.add_argument("--receptor-dir", type=Path, default=ANCHOR_DIR,
                   help="directory holding the receptor .pdb files")
    p.add_argument("--box-size", type=float, default=20.0,
                   help="edge length in A assumed for occupancy (default 20; "
                        "the true size was never recorded)")
    args = p.parse_args(argv)

    if not args.poses.exists():
        print(f"Pose directory not found: {args.poses}")
        return 1

    if args.validate:
        validate_against_collagen(args.poses, args.collagen_csv)

    print("\n=== Reconstructing MMP-1 box centres ===")
    groups = collect_groups(args.poses, receptor_filter="mmp1")
    rows = reconstruct(groups)
    if not rows:
        print("No MMP-1 poses found; nothing to reconstruct.")
        return 1

    print(f"  recovered {len(rows)} (receptor, box) groups "
          f"from {sum(r['n_poses'] for r in rows)} poses")
    by_receptor: dict[str, int] = defaultdict(int)
    for r in rows:
        by_receptor[r["receptor"]] += 1
    for rec, n in sorted(by_receptor.items()):
        print(f"    {rec:26s} {n:3d} boxes")

    single = [r for r in rows if r["n_poses"] == 1]
    if single:
        print(f"  {len(single)} group(s) rest on a SINGLE pose -- weakest evidence")

    # Which CSV rows remain unrecoverable?
    if args.mmp1_csv.exists():
        needed = set()
        with open(args.mmp1_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                needed.add(((row.get("target") or "").strip(),
                            (row.get("docking_box") or "").strip()))
        have = {(r["receptor"], r["docking_box"]) for r in rows}
        missing = {k for k in needed if k[0] and k not in have}
        print(f"  {len(needed - missing)}/{len(needed)} "
              f"(receptor, box) pairs in the CSV recovered")
        if missing:
            print(f"  {len(missing)} pair(s) have NO pose and cannot be "
                  f"reconstructed -- these need the original docking config:")
            for k in sorted(missing)[:10]:
                print(f"      {k[0]} / {k[1]}")

    if args.check_occupancy:
        check_occupancy(rows, args.receptor_dir, args.box_size)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {args.out}")
        print("  This is a SIDECAR file of ESTIMATES (is_measured=false).")
        print("  The shipped CSVs are untouched. Do not merge these values into")
        print("  the released dataset as if they were recorded geometry.")
    else:
        print("\n  (no --out given; nothing written)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
