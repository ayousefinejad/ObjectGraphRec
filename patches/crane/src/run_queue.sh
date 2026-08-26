#!/usr/bin/env bash
# Queue for the object-graph CRANE runs. Waits for whatever CRANE job is already on the GPU,
# then trains the extended architecture, then two ablations that split its two entry points.
#
#   nohup bash run_queue.sh > /tmp/crane_queue.log 2>&1 &
#
# Sequential on purpose: the runs are the only thing on this GPU and overlapping them would
# make the per-epoch times -- the one signal that the object stage is or is not cheap --
# uninterpretable. Each stage writes its own log under log/ as well as the file named here.

set -u
cd "$(dirname "$0")"
PY=/home/worker/hamedenv/bin/python
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Wait out the in-flight batch_first=1 grid rather than racing it onto the same card.
while pgrep -f 'main_diag.py .*--batch_first 1' | grep -qv "^$$\$"; do sleep 60; done

run () {  # run <logfile> <args...>
  local log=$1; shift
  echo "=== $(date '+%F %T')  $*"
  $PY main_diag.py -m CRANE -d home_v2 --batch_first 1 "$@" > "$log" 2>&1
  echo "=== $(date '+%F %T')  exit=$? -> $log"
}

# 1. The deliverable: same 8-config grid as the batch_first=1 baseline, object graph on. Same
#    grid, same selection rule, one axis changed -- so the difference is attributable.
run /tmp/crane_obj_grid.log --object 1

# 2. Ablations at (0.9, 0.001), which won validation on both prior single-config runs. These
#    split the object modality's two entry points, which the grid above conflates:
#      a. graph path only   -- object kNN propagation, no third attention token
#      b. attention only    -- third token, no propagation (n_obj_layers=0 is the identity)
run /tmp/crane_obj_graphonly.log --object 1 --object_in_attention 0 --dropout 0.9 --reg 0.001
run /tmp/crane_obj_attnonly.log  --object 1 --n_obj_layers 0      --dropout 0.9 --reg 0.001

echo "=== $(date '+%F %T')  queue done"
