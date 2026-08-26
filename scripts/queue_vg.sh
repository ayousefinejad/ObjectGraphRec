#!/usr/bin/env bash
# Visual Genome as the object-graph corpus, size-matched to MIT+NYU, on the MF backbone.
#
# vg_sub3213 is VG subsampled to MIT+NYU's exact scene count (3,213) under the unchanged
# default_fixed recipe, so it is directly comparable to the rows already on disk:
#   image+text          0.04007 +- 0.00031   (fusion_arms.csv, noobj)
#   +object, MIT+NYU    0.04291 +- 0.00051   (downstream.csv, default_fixed)
# Neither is re-run.
#
# MF only: the object modality is measurably non-zero on MF alone (+7.1%, t=13.4); on NGCF and
# tuned LightGCN it is flat, so a corpus change cannot show up there.
#
# Registered before running: the coverage gate put VG's exact coverage at 53.8% against the
# union's 51.8%, and moved 858 items from unreached to near. Propagating F18's measured per-tier
# effects gives an expected global change of ~+0.00025 -- below the 0.0003 reproducibility floor.
# This batch is therefore expected to be null, and is run to confirm rather than to discover.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[vg] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

BASE=(--data_path data/ --verbose 5 --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
      --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]"
      --core 5 --topk 10 --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part --fusion softmax
      --dataset lattice-runs/vg_sub3213 --modalities image,text,graph)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && return 0
    [ $(( waited % 600 )) -eq 0 ] && echo "[vg] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {
  local seed="$1"
  local tag="vg_sub3213_seed${seed}"
  local log="$OUT/${tag}.log"
  # Completion is read from the LOG's terminator, never from the .pt: --dump_embeddings rewrites
  # that file on every best-validation epoch, so an interrupted run already has one and would be
  # skipped with a mid-training checkpoint standing in for a final result.
  if [ -f "$log" ] && { grep -q "Early stop! #####" "$log" || tail -3 "$log" | grep -q "^{'precision'"; }; then
    echo "[vg] $tag already complete, skipping"; return 0
  fi
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[vg] === $tag (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --seed "$seed" \
          --dump_embeddings "$OUT/${tag}.pt" > "$log" 2>&1; then
      grep -m1 "^LATTICE_KNN graph" "$log"
      grep "test==" "$log" | tail -1
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" && { echo "[vg] $tag OOM attempt $try"; sleep 300; continue; }
    echo "[vg] $tag FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  echo "[vg] $tag FAILED after 3 OOMs"; return 1
}

for SEED in 0 1 2; do run_one "$SEED" || exit 1; done

echo "[vg] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[vg] FROZEN CHANGED"; exit 1; }
echo "[vg] COMPLETE at $(date -Is)"
