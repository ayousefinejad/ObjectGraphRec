#!/usr/bin/env bash
# Nine LATTICE runs backing the "the object modality helps *because the detected-object
# vocabulary covers the catalogue*" claim, all on the shipped default_fixed variant (encoder
# trained on data/scenes.json = MIT + NYU-Depth), all dumping best-epoch user/item embeddings so
# the gain can be decomposed per item offline:
#
#   control  +object, the published arm            -- treatment
#   noobj    --modalities image,text               -- baseline; control-noobj is the +7.1%
#   shufobj  +object with object_feat rows permuted -- placebo: same features, same graph up to
#            isomorphism, only the item<->object correspondence destroyed
#
# Ordered seed-major, not arm-major: an abort after any seed still leaves a complete paired
# triple rather than three unusable halves.
#
# BASE is copied from run_lattice_study.py's BASE (the default_fixed recipe) so these runs are
# comparable to the rows already in downstream.csv / fusion_arms.csv. Deliberately NOT via
# run_lattice_study.py: that runner owns the CSV schemas and has no --dump_embeddings axis.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
# The two embedding-dump runs earlier in this study died at the same allocation with 1.9 GiB
# reserved-but-unallocated while a second tenant held ~7 GB. Fragmentation, not capacity.
# (this torch build renamed the variable and warns on the old one; set both so it works either way)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

BASE=(--data_path data/ --verbose 5 --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
      --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]"
      --core 5 --topk 10 --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part --fusion softmax)

# This GPU is shared. A run needs ~14 GB at the epoch-boundary rebuild, so wait for headroom
# rather than start a job that will die 20 minutes in and take the batch down with it.
NEED_MIB=12000
wait_for_gpu () {
  local waited=0
  while :; do
    local used total free
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && { [ "$waited" -gt 0 ] && echo "[cov] ${free} MiB free after ${waited}s"; return 0; }
    [ $(( waited % 600 )) -eq 0 ] && echo "[cov] waiting for GPU: ${free} MiB free, need ${NEED_MIB} (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                      # run_one <arm> <seed> <extra flags...>
  local arm="$1" seed="$2"; shift 2
  local log="$OUT/cov_${arm}_seed${seed}.log"
  if [ -s "$OUT/cov_${arm}_seed${seed}.pt" ]; then
    echo "[cov] $arm seed$seed already dumped, skipping"; return 0
  fi
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[cov] === $arm seed$seed (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --seed "$seed" "$@" \
          --dump_embeddings "$OUT/cov_${arm}_seed${seed}.pt" > "$log" 2>&1; then
      grep -E "^LATTICE_PROP |^LATTICE_DIAG " "$log" | head -2
      grep "test==" "$log" | tail -1
      return 0
    fi
    # Only an OOM is worth retrying -- it is caused by the other tenant, not by this recipe.
    # Anything else is a real fault and retrying it just burns the GPU three times.
    if grep -q "OutOfMemoryError" "$log"; then
      echo "[cov] $arm seed$seed OOM on attempt $try, backing off"; sleep 300; continue
    fi
    echo "[cov] $arm seed$seed FAILED (not OOM) -- see $log"; tail -5 "$log"; return 1
  done
  echo "[cov] $arm seed$seed FAILED after 3 OOMs -- see $log"; return 1
}

for SEED in 0 1 2; do
  run_one control "$SEED" --dataset lattice-runs/default_fixed                        || exit 1
  run_one noobj   "$SEED" --dataset lattice-runs/default_fixed --modalities image,text || exit 1
  run_one shufobj "$SEED" --dataset lattice-runs/default_fixed_shufobj                || exit 1
done

echo "[cov] verifying the frozen artifacts were not touched..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[cov] FROZEN ARTIFACT CHANGED"; exit 1; }
echo "[cov] COMPLETE at $(date -Is)"
