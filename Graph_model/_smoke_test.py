import sys
sys.path.insert(0, "/Users/suppboat/Jupyter_Dock")

from Graph_model.data.config import LIGAND_CATALOGUE, BOX_TYPE_VOCAB, GALLOYL_UNIT_COUNTS
from Graph_model.data.features.galloyl import GalloylFragmentDetector
from Graph_model.data.features.conditions import ConditionEncoder
from Graph_model.data.features.atom import ATOM_FEAT_DIM, BOND_FEAT_DIM

print("ATOM_FEAT_DIM:", ATOM_FEAT_DIM, "  BOND_FEAT_DIM:", BOND_FEAT_DIM)
print("Ligands:", list(LIGAND_CATALOGUE.keys()))
print("Galloyl units:", GALLOYL_UNIT_COUNTS)
print()

# Fragment detection
for name, info in LIGAND_CATALOGUE.items():
    smiles = info.get("smiles_rdkit", info["smiles"])
    counts, nodes = GalloylFragmentDetector.from_smiles(smiles)
    print(
        f"  {name:28s}  galloyl={counts['galloyl_strict']}"
        f"  weighted={counts['galloyl_weighted']:.2f}"
        f"  n_fragments={len(nodes)}"
    )

print()

# Condition encoding
enc = ConditionEncoder(strict=False)
for ph in [5.0, 5.5, 7.0]:
    v = enc.encode(ph, 25, "GLU_LYS_cluster12", "collagen")
    print(f"  pH={ph}: cond={v}")
v = enc.encode(7.0, 37, "global_blind", "mmp1")
print(f"  MMP-1 pH7 T37 global_blind: cond={v}")

print()
print("Smoke test: PASSED")
