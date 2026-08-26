#!/usr/bin/env bash
# LATTICE runs for converged_gat only -- its encoder + object_feat.npy already
# exist (the earlier attempt trained fine and crashed only while writing
# provenance.json, which has since been rebuilt from the checkpoint).
# Re-exporting would waste ~10 min and, worse, overwrite a verified artifact.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

[ -f data/lattice-runs/converged_gat/object_feat.npy ] || {
  echo "[converged_gat] object_feat.npy MISSING -- re-export first"; exit 1; }

echo "[converged_gat] running LATTICE, 3 seeds"
$PY scripts/run_lattice_study.py --variants converged_gat --seeds 0 1 2 --eval-cores 10 \
    || { echo "[converged_gat] LATTICE FAILED"; exit 1; }
echo "[converged_gat] COMPLETE at $(date -Is)"
