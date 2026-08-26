#!/usr/bin/env bash
# Isolated GAT-vs-GraphSAGE downstream comparison, queued behind whatever is running.
#
# `converged_gat` is `converged` with backbone=gat and nothing else changed
# (both: 1000 epochs, tau=0.5, lr=1e-3, 2 layers, repaired pipeline). That makes
# converged vs converged_gat the only clean backbone pair downstream -- `tuned`
# moves five knobs at once and cannot isolate the backbone.
#
# Intrinsic side already measured (stageD): GAT 0.8070 +/- 0.0027,
# SAGE 0.7886. This fills in R@k / P@k / NDCG@k for the same pair.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

echo "[converged_gat] exporting encoder (GAT, 1000 ep, tau=0.5, lr=1e-3, 2L)"
$PY scripts/export_lattice_feats.py \
    --variant converged_gat --config converged_gat --seed 0 \
    || { echo "[converged_gat] EXPORT FAILED"; exit 1; }

echo "[converged_gat] running LATTICE, 3 seeds"
$PY scripts/run_lattice_study.py --variants converged_gat --seeds 0 1 2 --eval-cores 10 \
    || { echo "[converged_gat] LATTICE FAILED"; exit 1; }

echo "[converged_gat] COMPLETE at $(date -Is)"
