#!/usr/bin/env bash
# Neighbourhood aggregation in the object-graph encoder: 5 alternatives against the published
# mean, everything else held at the default_fixed recipe.
#
#   mean   SAGEConv default            -- the published arm, ALREADY on disk at 3 seeds
#                                          (downstream.csv default_fixed, 0.04291 +- 0.00051)
#   max    SAGEConv aggr='max'         -- GraphSAGE's pool aggregator (Hamilton et al. 3.3)
#   sum    SAGEConv aggr='add'         -- unnormalised sum; node degree leaks into the norm
#   gcn    GCNConv                     -- symmetric-normalised sum, D^-1/2 A D^-1/2
#   gat    GATConv                     -- attention-weighted sum
#   wmean  WeightedSAGEConv + weighted -- Eq. (2)'s w_ab = c_ab / sqrt(c_a c_b)
#
# `aggr='mean'` was verified to be the SAME MODULE as SAGEConv's default (identical weights and
# identical forward output under a fixed seed), so the published arm is untouched by this change
# and does not need re-running.
#
# Routed through run_lattice_study.py: these are encoder variants like openai_mit or yolo_mit,
# so they belong in downstream.csv under that runner's provenance checks, not in a bespoke loop.
set -u
cd /home/worker/arshia_yousefi/object-graph
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

NEED_MIB=12000
waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(( total - used ))
  [ "$free" -ge "$NEED_MIB" ] && break
  [ $(( waited % 600 )) -eq 0 ] && echo "[agg] waiting for GPU: ${free} MiB free (${waited}s)"
  sleep 60; waited=$(( waited + 60 ))
done

echo "[agg] === 5 aggregations x 3 seeds === $(date -Is)"
~/hamedenv/bin/python scripts/run_lattice_study.py \
    --variants agg_max agg_sum agg_gcn agg_gat agg_wmean --seeds 0 1 2 --eval-cores 10 \
    || { echo "[agg] FAILED"; exit 1; }

echo "[agg] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[agg] FROZEN CHANGED"; exit 1; }
echo "[agg] COMPLETE at $(date -Is)"
