#!/usr/bin/env bash
# CRANE object-modality test, seed-replicated at last. The previous CRANE numbers in this study
# compared arms across dropout x reg_weight cells at a SINGLE seed (mean Delta R@20 -0.00029,
# paired t=-0.29, 3 of 8 cells moving the other way) -- indistinguishable from zero and with no
# replicate-level noise estimate. run_crane_objgraph.py was written to fix that (both arms,
# ONE fixed config = the best cell from that sweep, dropout=0.9 reg_weight=0.001, 3 seeds) but
# was never actually executed -- no log after 28 Jul, before this run.
#
# Config() resolves configs/overall.yaml relative to os.getcwd(), not the script's location, so
# this MUST run with cwd = CRANE/CRANE/src (confirmed by a smoke test that failed with
# KeyError: 'valid_metric' from the repo root and succeeded -- reaching real data loading --
# from src/).
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[crane] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[crane] predecessor done at $(date -Is)"
fi

NEED_MIB=12000
waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(( total - used ))
  [ "$free" -ge "$NEED_MIB" ] && break
  [ $(( waited % 600 )) -eq 0 ] && echo "[crane] waiting for GPU: ${free} MiB free (${waited}s)"
  sleep 60; waited=$(( waited + 60 ))
done

mkdir -p ../../../object-graph/objectgraph-eval/embeddings
OUT=../../../object-graph/objectgraph-eval/embeddings

for ARM in noobject object; do
  LOG="$OUT/crane_${ARM}.log"
  if grep -q "█████████████ BEST ████████████████" "$LOG" 2>/dev/null; then
    echo "[crane] $ARM already complete, skipping"; continue
  fi
  echo "[crane] === arm=$ARM, 3 seeds === $(date -Is)"
  $PY run_crane_objgraph.py --arm "$ARM" > "$LOG" 2>&1 \
      || { echo "[crane] $ARM FAILED -- see $LOG"; tail -20 "$LOG"; exit 1; }
  grep -A6 "█████████████ BEST" "$LOG" | tail -8
done

echo "[crane] COMPLETE at $(date -Is)"
