#!/usr/bin/env bash
# Try to make the object modality actually help CRANE.
#
# Baseline facts (measured, 3 seeds each):
#   noobject          R@20 0.03970 +- 0.00101
#   object (as-is)    R@20 0.03963 +- 0.00138   -> 0.1 sd, no effect
#
# The object channel is near-inert for two separable reasons:
#   1. resolution -- argmax assigns each item ONE node, so 1,815 distinct labels collapse to
#      510 distinct vectors and 510 distinct neighbourhoods (image/text have ~14,200).
#   2. reach -- top-k is directed and ties break by index, so 78.5% of items have in-degree 0
#      and, with residual=True, are untouched by the object graph entirely.
#
# Three arms separate them. Each is 3 seeds at the fixed best cell (dropout 0.9, reg 1e-3).
#
#   pooled       fixes resolution only   510 -> 1815 vectors,  reach 21.5% -> 41.5%
#   sym          fixes reach only        510 vectors,          reach 21.5% -> 100%
#   pooled_sym   fixes both              1815 vectors,         reach 100%
#
# If only pooled_sym moves, the two are complementary; if sym alone captures it, resolution was
# never the constraint. Either is a result.
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

run () {  # name, dataset, extra-json
  echo "[crane-improve] === $1 ==="
  $PY run_crane_objgraph.py --arm object --dataset "$2" --extra "$3" \
      || { echo "[crane-improve] $1 FAILED"; exit 1; }
}
run "sym         (argmax feats, symmetric graph)" home_v2_openai '{"obj_knn_sym": true}'
run "pooled_sym  (pooled feats, symmetric graph)" home_v2_pooled '{"obj_knn_sym": true}'
echo "[crane-improve] COMPLETE at $(date -Is)"
