#!/usr/bin/env bash
# Re-run the NYU-only arm end to end, AFTER the detector queue drains.
#
# Writes to variant `nyu_rerun`, not `nyu_default_fixed`:
#   * the original arm is cited in the write-up (§13.6) and stays byte-intact
#   * run_lattice_study skips (variant, seed) rows already in downstream.csv,
#     so reusing the old name would silently no-op
#   * a fresh name makes this a reproducibility check -- same corpus, same
#     recipe, same seeds should land on 0.795 intrinsic / 292 unique vectors
#     / R@20 0.0427
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for detector queue (PID $WAIT_PID) to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] detector queue done at $(date -Is)"
fi

echo "[nyu_rerun] exporting encoder (default_fixed, seed 0, corpus data/nyu-depth.json)"
$PY scripts/export_lattice_feats.py \
    --variant nyu_rerun --config default_fixed --seed 0 \
    --scenes data/nyu-depth.json || { echo "[nyu_rerun] EXPORT FAILED"; exit 1; }

echo "[nyu_rerun] running LATTICE, 3 seeds"
$PY scripts/run_lattice_study.py --variants nyu_rerun --seeds 0 1 2 --eval-cores 10 \
    || { echo "[nyu_rerun] LATTICE FAILED"; exit 1; }

echo "[nyu_rerun] COMPLETE at $(date -Is)"
