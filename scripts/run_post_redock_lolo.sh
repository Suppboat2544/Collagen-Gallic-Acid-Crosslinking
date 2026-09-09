#!/usr/bin/env bash
# After B6 redock completes: force-reload hetero cache + LOLO Model A with baselines.
set -euo pipefail
ROOT="${COLLAGEN_DATA_ROOT:-/Users/suppboat/Jupyter_Dock}"
export COLLAGEN_DATA_ROOT="$ROOT"
export PYTHONPATH="$ROOT/Graph_model_repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
PY="$ROOT/.venv/bin/python"
cd "$ROOT/Graph_model_repo"

echo "=== Provenance ==="
"$PY" -m Graph_model.data.provenance

echo "=== Force-reload hetero cache (schema-2 + new docks + MMP-1 boxes) ==="
"$PY" - <<'PY'
from Graph_model.data.hetero_dataset import HeteroDockingDataset
ds = HeteroDockingDataset(include_mmp1=True, force_reload=True).load(verbose=True)
print(f"graphs={len(ds)}")
# quick emptiness check for MMP-1 protein nodes
n_mmp = n_empty = 0
for g in ds:
    if getattr(g, "receptor", None) == "mmp1" or str(getattr(g, "sample_id", "")).startswith("MMP1"):
        n_mmp += 1
        try:
            nres = int(g["residue"].x.size(0))
        except Exception:
            nres = 0
        if nres == 0:
            n_empty += 1
print(f"MMP-1 graphs={n_mmp} empty_protein={n_empty}")
PY

echo "=== LOLO Model A + baselines (CPU; reuse cache unless FORCE_RELOAD=1) ==="
RELOAD_FLAG=()
if [[ "${FORCE_RELOAD:-0}" == "1" ]]; then
  RELOAD_FLAG=(--force-reload)
fi
"$PY" scripts/run_lolo.py --model A --epochs 50 --seeds 0 \
  --device cpu --batch-size 8 \
  "${RELOAD_FLAG[@]}" \
  --results-dir "$ROOT/Graph_model/results/lolo_schema2_b6" \
  -v

echo "Wrote results under Graph_model/results/lolo_schema2_b6"
