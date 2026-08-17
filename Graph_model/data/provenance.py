"""
Graph_model.data.provenance
===========================
Verify that the molecule named in the ligand catalogue is the molecule that was
actually docked.

Why this exists
---------------
The catalogue in `config.LIGAND_CATALOGUE` is the reference chemistry: SMILES,
molecular formula, molecular weight, heavy-atom count. The docking campaign,
however, read its own structure files from the data root, and nothing ever
compared the two. Three disagreements were sitting in the released dataset:

  * `pentagalloylglucose` -- the input .sdf holds FOUR disconnected fragments
    (a tri-O-galloylglucose plus three loose gallic acid molecules, C48H42O33).
    Docking kept only the largest fragment, so every pose scored under the name
    "pentagalloylglucose" is in fact tri-O-galloylglucose: C27H24O18,
    MW 636.47, 45 heavy atoms, THREE galloyl units. The named compound is
    C41H32O26, MW 940.68, 67 heavy atoms, FIVE galloyl units.

  * `NHS` -- docked as C3H3NO3 (MW 101.06), which is one CH2 short of
    N-hydroxysuccinimide (C4H5NO3, MW 115.09, PubChem CID 6467).

  * `NHS_ester_intermediate` -- docked with a FOUR-membered ring (C6H7NO4). A
    succinimidyl ester necessarily contains a five-membered imide ring.

A binding affinity is only meaningful next to the structure it was computed
for. This module makes that link checkable, and `check_structures` is wired
into the training preflight so the mismatch cannot pass silently again.

It deliberately does NOT repair anything. The docking scores are experimental
output and are left exactly as produced; the remedy for a mismatch is to re-dock
the correct structure, which is an author decision, not a code change.

Usage
-----
    python -m Graph_model.data.provenance                 # human-readable table
    python -m Graph_model.data.provenance --json          # machine-readable
    python -m Graph_model.data.provenance --strict        # exit 1 on mismatch

    from Graph_model.data.provenance import check_structures
    report = check_structures()          # list[StructureCheck]
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from .config import ANCHOR_DIR, LIGAND_CATALOGUE

logger = logging.getLogger(__name__)

__all__ = [
    "StructureCheck",
    "check_structures",
    "format_report",
    "KNOWN_MISMATCHES",
]

# Mismatches confirmed against the released data (both the v1 and v2 Drive
# snapshots). Recorded so that a fresh run can tell a KNOWN defect from a NEW
# one; presence here is documentation, never a licence to ignore.
KNOWN_MISMATCHES: dict[str, str] = {
    "pentagalloylglucose": (
        "input .sdf is 4 disconnected fragments; docking scored only the "
        "largest, tri-O-galloylglucose (C27H24O18, 3 galloyl units) rather "
        "than pentagalloylglucose (C41H32O26, 5 galloyl units)"
    ),
    "NHS": (
        "docked as C3H3NO3 (MW 101.06); N-hydroxysuccinimide is C4H5NO3 "
        "(MW 115.09, CID 6467) -- one CH2 short"
    ),
    "NHS_ester_intermediate": (
        "docked structure contains a 4-membered ring; a succinimidyl ester "
        "requires a 5-membered imide ring"
    ),
}


@dataclass
class StructureCheck:
    """Catalogue-vs-docked comparison for one ligand."""
    ligand: str
    structure_file: str | None
    catalogue_formula: str | None
    docked_formula: str | None
    catalogue_mw: float | None
    docked_mw: float | None
    catalogue_heavy_atoms: int | None
    docked_heavy_atoms: int | None
    docked_fragments: int | None
    catalogue_inchikey: str | None
    docked_inchikey: str | None
    status: str          # "match" | "mismatch" | "missing" | "unreadable"
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "match"


def _skeleton(inchikey: str | None) -> str | None:
    """First InChIKey block -- constitution only, ignoring stereo/protonation."""
    return inchikey.split("-")[0] if inchikey else None


def _describe(mol) -> dict:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:                       # InChI is optional in some builds
        inchikey = None
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mw": round(Descriptors.MolWt(mol), 2),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "fragments": len(Chem.GetMolFrags(mol)),
        "inchikey": inchikey,
    }


def _load_structure(path: Path):
    """Read the first molecule from an .sdf / .mol2 file, or None."""
    from rdkit import Chem

    try:
        if path.suffix.lower() == ".sdf":
            return next(iter(Chem.SDMolSupplier(str(path), removeHs=True)), None)
        if path.suffix.lower() == ".mol2":
            return Chem.MolFromMol2File(str(path), removeHs=True)
    except Exception as exc:                # pragma: no cover - malformed input
        logger.warning("Could not read %s: %s", path, exc)
    return None


def check_structures(structure_dir: Path | None = None,
                     ligands: Iterable[str] | None = None) -> list[StructureCheck]:
    """
    Compare each catalogue entry against its docking input structure.

    Parameters
    ----------
    structure_dir : Path, optional
        Directory holding `<ligand>.sdf` / `<ligand>.mol2`. Defaults to
        `config.ANCHOR_DIR`.
    ligands : iterable of str, optional
        Restrict the check to these catalogue keys.

    Returns
    -------
    list[StructureCheck], one per ligand, in catalogue order.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    structure_dir = Path(structure_dir) if structure_dir else ANCHOR_DIR
    names = list(ligands) if ligands is not None else list(LIGAND_CATALOGUE)

    out: list[StructureCheck] = []
    for name in names:
        rec = LIGAND_CATALOGUE[name]

        cat_mol = Chem.MolFromSmiles(rec["smiles"])
        cat = _describe(cat_mol) if cat_mol is not None else {}

        path = None
        for ext in (".sdf", ".mol2"):
            candidate = structure_dir / f"{name}{ext}"
            if candidate.exists():
                path = candidate
                break

        if path is None:
            out.append(StructureCheck(
                ligand=name, structure_file=None,
                catalogue_formula=cat.get("formula"), docked_formula=None,
                catalogue_mw=cat.get("mw"), docked_mw=None,
                catalogue_heavy_atoms=cat.get("heavy_atoms"), docked_heavy_atoms=None,
                docked_fragments=None,
                catalogue_inchikey=cat.get("inchikey"), docked_inchikey=None,
                status="missing",
                detail=f"no {name}.sdf or {name}.mol2 under {structure_dir}",
            ))
            continue

        mol = _load_structure(path)
        if mol is None:
            out.append(StructureCheck(
                ligand=name, structure_file=str(path),
                catalogue_formula=cat.get("formula"), docked_formula=None,
                catalogue_mw=cat.get("mw"), docked_mw=None,
                catalogue_heavy_atoms=cat.get("heavy_atoms"), docked_heavy_atoms=None,
                docked_fragments=None,
                catalogue_inchikey=cat.get("inchikey"), docked_inchikey=None,
                status="unreadable", detail="RDKit could not parse the file",
            ))
            continue

        dock = _describe(mol)

        # Constitution-level identity: formula must agree. InChIKey skeletons are
        # compared as a cross-check where InChI is available, so that a
        # coincidental formula match (an isomer) is still caught.
        same_formula = cat.get("formula") == dock["formula"]
        cat_skel, dock_skel = _skeleton(cat.get("inchikey")), _skeleton(dock["inchikey"])
        same_skeleton = (cat_skel == dock_skel) if (cat_skel and dock_skel) else None

        if same_formula and same_skeleton is not False:
            status, detail = "match", ""
        else:
            status = "mismatch"
            bits = []
            if not same_formula:
                bits.append(f"formula {cat.get('formula')} vs docked {dock['formula']}")
                bits.append(f"MW {cat.get('mw')} vs {dock['mw']}")
                bits.append(f"heavy atoms {cat.get('heavy_atoms')} vs {dock['heavy_atoms']}")
            elif same_skeleton is False:
                bits.append("same formula but different constitution (isomer)")
            if dock["fragments"] > 1:
                bits.append(f"docking input is {dock['fragments']} disconnected fragments")
            if name in KNOWN_MISMATCHES:
                bits.append(f"KNOWN: {KNOWN_MISMATCHES[name]}")
            detail = "; ".join(bits)

        out.append(StructureCheck(
            ligand=name, structure_file=str(path),
            catalogue_formula=cat.get("formula"), docked_formula=dock["formula"],
            catalogue_mw=cat.get("mw"), docked_mw=dock["mw"],
            catalogue_heavy_atoms=cat.get("heavy_atoms"),
            docked_heavy_atoms=dock["heavy_atoms"],
            docked_fragments=dock["fragments"],
            catalogue_inchikey=cat.get("inchikey"), docked_inchikey=dock["inchikey"],
            status=status, detail=detail,
        ))
    return out


def format_report(checks: list[StructureCheck]) -> str:
    """Human-readable table plus a summary line."""
    head = (f"{'ligand':26s} {'status':10s} {'catalogue':12s} {'docked':12s} "
            f"{'MW cat':>8s} {'MW dock':>8s} {'HA':>7s}")
    lines = [head, "-" * len(head)]
    for c in checks:
        ha = (f"{c.catalogue_heavy_atoms}/{c.docked_heavy_atoms}"
              if c.docked_heavy_atoms is not None else "-")
        lines.append(
            f"{c.ligand:26s} {c.status:10s} {c.catalogue_formula or '-':12s} "
            f"{c.docked_formula or '-':12s} "
            f"{(c.catalogue_mw if c.catalogue_mw is not None else 0):8.2f} "
            f"{(c.docked_mw if c.docked_mw is not None else 0):8.2f} {ha:>7s}"
        )
    bad = [c for c in checks if c.status == "mismatch"]
    for c in bad:
        lines.append(f"\n  ! {c.ligand}: {c.detail}")

    n_ok = sum(c.status == "match" for c in checks)
    lines.append(
        f"\n{n_ok}/{len(checks)} ligands match their docking input; "
        f"{len(bad)} mismatch, "
        f"{sum(c.status in ('missing', 'unreadable') for c in checks)} unavailable."
    )
    if bad:
        lines.append(
            "A mismatch means the reported affinity belongs to a different "
            "molecule than the one it is labelled with. Re-dock the correct "
            "structure; do not edit the scores."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Check that catalogue ligands match their docked structures.")
    p.add_argument("--structures", type=Path, default=None,
                   help=f"directory of ligand .sdf/.mol2 files (default: {ANCHOR_DIR})")
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    p.add_argument("--strict", action="store_true",
                   help="exit 1 if any ligand mismatches")
    args = p.parse_args(argv)

    checks = check_structures(args.structures)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(format_report(checks))

    if args.strict and any(c.status == "mismatch" for c in checks):
        return 1
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
