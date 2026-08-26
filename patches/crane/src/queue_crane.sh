#!/usr/bin/env bash
# CRANE object-modality comparison on the MIT-only (OpenAI) encoder features.
# Both arms, 3 seeds each, one fixed hyperparameter cell -- queued behind
# whatever is already using the GPU.
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"

if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

for arm in noobject object; do
  echo "[crane] === arm=$arm ==="
  $PY run_crane_objgraph.py --arm "$arm" || { echo "[crane] ARM $arm FAILED"; exit 1; }
done

echo "[crane] COMPLETE at $(date -Is)"
