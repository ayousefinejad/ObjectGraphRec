#!/usr/bin/env bash
# MIT-Indoors vs NYU-Depth as object-graph corpora, on the TUNED LightGCN backbone.
#
# Flags are held at tuning.csv's obj_lgn3 cell exactly (lightgcn, [64,64,64], lr 5e-4,
# --epoch 400) so these corpus rows are directly comparable to the two that already exist there:
#   lgn3      image+text          0.04340 +- 0.00069   (3 seeds)
#   obj_lgn3  +object, MIT+NYU    0.04372 +- 0.00069   (3 seeds)
# Neither is re-run here.
#
#   openai_mit         MIT only, 2,645 scenes / 1,007 nodes  -- the direct request
#   nyu_default_fixed  NYU only,   579 scenes /   360 nodes  -- the comparison
#   mit_sub579         MIT at NYU's scene count, 490 nodes   -- the size control, without which
#                      "MIT is the better corpus" is confounded with "MIT is the bigger corpus"
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[corp] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[corp] predecessor done at $(date -Is)"
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

TUNED=(--data_path data/ --verbose 5 --epoch 400 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
       --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64
       --weight_size "[64,64,64]" --core 5 --topk 10 --lambda_coeff 0.9 --n_layers 1
       --mess_dropout "[0.1, 0.1]" --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]"
       --test_flag part --fusion softmax --cf_model lightgcn --modalities image,text,graph)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && return 0
    [ $(( waited % 600 )) -eq 0 ] && echo "[corp] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                      # run_one <variant> <seed>
  local var="$1" seed="$2"
  local tag="corp_${var}_seed${seed}"
  local log="$OUT/${tag}.log"
  [ -s "$OUT/${tag}.pt" ] && { echo "[corp] $tag done, skipping"; return 0; }
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[corp] === $tag (attempt $try) === $(date -Is)"
    if $PY main.py "${TUNED[@]}" --dataset "lattice-runs/${var}" --seed "$seed" \
          --dump_embeddings "$OUT/${tag}.pt" > "$log" 2>&1; then
      grep -m1 "^LATTICE_KNN graph" "$log"
      grep "test==" "$log" | tail -1
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" \
      && { echo "[corp] $tag OOM attempt $try"; sleep 300; continue; }
    echo "[corp] $tag FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  echo "[corp] $tag FAILED after 3 OOMs"; return 1
}

for SEED in 0 1 2; do
  for VAR in openai_mit nyu_default_fixed mit_sub579; do
    run_one "$VAR" "$SEED" || exit 1
  done
done

echo "[corp] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[corp] FROZEN CHANGED"; exit 1; }
echo "[corp] COMPLETE at $(date -Is)"
