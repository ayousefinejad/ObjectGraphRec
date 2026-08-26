#!/usr/bin/env bash
# MMGCN (Wei et al., ACM MM'19) with vs without the object-graph modality, 3 seeds each -- the
# same seed-replicated treatment as queue_crane_seeded.sh, for the same reason: a single-seed
# comparison has no noise estimate to test a delta against.
#
# Port verified before this queue: bit-for-bit copy of enoche/MMRec's mmgcn.py plus one new
# branch (self.o_gcn, structurally identical to self.t_gcn -- see models/mmgcn.py's docstring
# for the exact treatment). Both arms smoke-tested for 2 epochs: noobject builds v_gcn+t_gcn
# only (1,418,240 params), object adds o_gcn (+49,536 params) and trains without error.
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[mmgcn] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[mmgcn] predecessor done at $(date -Is)"
fi

NEED_MIB=12000
waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(( total - used ))
  [ "$free" -ge "$NEED_MIB" ] && break
  [ $(( waited % 600 )) -eq 0 ] && echo "[mmgcn] waiting for GPU: ${free} MiB free (${waited}s)"
  sleep 60; waited=$(( waited + 60 ))
done

mkdir -p ../../../object-graph/objectgraph-eval/embeddings
OUT=../../../object-graph/objectgraph-eval/embeddings

for ARM in noobject object; do
  LOG="$OUT/mmgcn_${ARM}.log"
  if grep -q "█████████████ BEST ████████████████" "$LOG" 2>/dev/null; then
    echo "[mmgcn] $ARM already complete, skipping"; continue
  fi
  echo "[mmgcn] === arm=$ARM, 3 seeds === $(date -Is)"
  $PY run_mmgcn_objgraph.py --arm "$ARM" > "$LOG" 2>&1 \
      || { echo "[mmgcn] $ARM FAILED -- see $LOG"; tail -20 "$LOG"; exit 1; }
  grep -A6 "█████████████ BEST" "$LOG" | tail -8
done

echo "[mmgcn] COMPLETE at $(date -Is)"
