#!/usr/bin/env bash
# GAT vs simple message passing on the MIT+NYU corpus (default_fixed).
#
# The existing propagation comparison (RESULTS.md) ran on openai_mit -- MIT only, 1,007 nodes --
# where GAT lost by -1.6% R@20 (t=-4.92). This repeats it on the shipped MIT+NYU graph
# (1,068 nodes), the corpus every headline number in the study uses.
#
# Only the GAT arm is run: the dense arm is the published default_fixed configuration and is
# already on disk at 3 seeds (downstream.csv: 0.04332 / 0.04306 / 0.04234).
#
# Routed through run_lattice_study.py rather than a bare main.py loop, because that runner's
# check_provenance() requires a `LATTICE_PROP mode=gat` line in the log. --item_prop changes no
# adjacency and leaves no fingerprint in LATTICE_DIAG, so a silently-ignored flag would make this
# arm numerically identical to the control and nothing else would catch it.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[gat] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[gat] predecessor done at $(date -Is)"
fi

NEED_MIB=12000
waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(( total - used ))
  [ "$free" -ge "$NEED_MIB" ] && break
  [ $(( waited % 600 )) -eq 0 ] && echo "[gat] waiting for GPU: ${free} MiB free (${waited}s)"
  sleep 60; waited=$(( waited + 60 ))
done

echo "[gat] === itemgat on default_fixed, 3 seeds === $(date -Is)"
$PY scripts/run_lattice_study.py --variants default_fixed --arms itemgat --seeds 0 1 2 \
    --eval-cores 10 || { echo "[gat] FAILED"; exit 1; }

echo "[gat] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[gat] FROZEN CHANGED"; exit 1; }
echo "[gat] COMPLETE at $(date -Is)"
