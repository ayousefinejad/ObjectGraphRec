#!/usr/bin/env bash
# LATTICE + object modality, with the pooled (soft top-3 node) object features instead of the
# argmax hard assignment -- same fix already tested for CRANE (510 -> 1815 unique vectors,
# 21.5% -> 41.5% of items reachable). There it moved R@20 +2.5% but wasn't resolvable at 3
# seeds (paired t=+0.92). LATTICE's global modal_weight fusion is a different consumer, and
# LATTICE is the one architecture where the object modality has already shown a clean, real
# gain (+7.1% with the argmax features) -- so this asks whether resolution was leaving
# something on the table specifically for the consumer that already benefits.
#
# Directory built via scripts/lattice_variant.py; object_feat.npy is the same file already
# exported for the CRANE pooled arm (data/lattice-runs/openai_mit_pooled/), not re-derived.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

echo "[lattice-pooled] verifying dataset directory..."
$PY -c "
import numpy as np
a = np.load('data/lattice-runs/openai_mit_pooled/object_feat.npy')
u = len(np.unique(a.round(6), axis=0))
assert u == 1815, f'expected 1815 unique vectors, found {u} -- wrong file?'
print(f'  object_feat.npy OK: {a.shape}, {u} unique vectors')
"

$PY scripts/run_lattice_study.py --variants openai_mit_pooled --seeds 0 1 2 --eval-cores 10 \
    || { echo "[lattice-pooled] FAILED"; exit 1; }
echo "[lattice-pooled] COMPLETE at $(date -Is)"
