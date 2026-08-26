#!/usr/bin/env bash
# CRANE with the pooled object features (1815 unique vectors vs 510 for argmax).
# The no-object baseline is unchanged by pooling, so only the object arm is re-run;
# it is compared against the noobject arm already measured on home_v2_openai.
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi
echo "[crane-pooled] === arm=object, dataset=home_v2_pooled ==="
$PY run_crane_objgraph.py --arm object --dataset home_v2_pooled || { echo "[crane-pooled] FAILED"; exit 1; }
echo "[crane-pooled] COMPLETE at $(date -Is)"
