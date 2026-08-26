#!/usr/bin/env bash
# Negative-sampling strategy for the object-graph encoder's link-prediction loss: 4 strategies
# named by the user (Uniform, Popularity-aware, Hard Negative, Dynamic/Semi-hard), tested
# downstream in LATTICE. Everything except neg_mode held at the default_fixed recipe.
#
#   uniform    any non-edge pair, equally likely           -- PUBLISHED, ALREADY on disk at 3
#                                                              seeds (downstream.csv default_fixed,
#                                                              0.04291 +- 0.00051). NOT re-run.
#   pop        degree^alpha-weighted non-edge pairs         -- neg_mode=degree, neg_alpha=1.0
#   hard       hardest (most similar) non-edge per anchor,
#              re-mined from the LIVE embedding every epoch -- neg_mode=hard
#   semihard   mid-percentile band (5-30%) of that same
#              per-epoch ranking, resampled stochastically   -- neg_mode=semihard
#              within the band each epoch ("dynamic")
#
# The 'hard'/'semihard' mining code (ObjectGraph/core.py:_mined_negatives) was unit-tested before
# any encoder training: forbidden pairs (self, positive edges) are never selected (0/200 in both
# modes), and mean cosine-to-anchor orders hard > semihard > random as it must. The three new
# encoders are already trained (CPU, ~1 min each): neg_pop AUC 0.7227, neg_hard AUC 0.6951,
# neg_semihard AUC 0.7284, all below the uniform baseline's 0.7573 -- expected, since a harder
# training distribution makes the SAME uniform-negative intrinsic eval harder, not easier.
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
  [ $(( waited % 600 )) -eq 0 ] && echo "[negmode] waiting for GPU: ${free} MiB free (${waited}s)"
  sleep 60; waited=$(( waited + 60 ))
done

echo "[negmode] === 3 negative-sampling strategies x 3 seeds === $(date -Is)"
~/hamedenv/bin/python scripts/run_lattice_study.py \
    --variants neg_pop neg_hard neg_semihard --seeds 0 1 2 --eval-cores 10 \
    || { echo "[negmode] FAILED"; exit 1; }

echo "[negmode] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[negmode] FROZEN CHANGED"; exit 1; }
echo "[negmode] COMPLETE at $(date -Is)"
