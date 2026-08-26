#!/usr/bin/env bash
# The two missing cells of the modality lattice: image-only and text-only.
#
# Five of the seven non-empty subsets of {image, text, object} are measured (RESULTS.md 7.4).
# The singletons {image} and {text} are not, and one of them is load-bearing:
#
#   {text, object} = 0.04514 is the best number measured anywhere in this study, and it is
#   currently uninterpretable. If {text} alone reaches ~0.045 then the object modality is inert
#   in that pair and the headline configuration is simply text; if {text} falls short by the
#   ~0.0025 object contributes elsewhere, the pair is real. The existing table cannot tell these
#   apart, which is why this batch exists.
#
# rho=1 to match the five rows already in that table. rho=2 is the better setting (7.6) but
# rebuilding the lattice there is a separate job, and mixing rho within one table would make it
# unreadable.
#
# The object modality is spelled `graph`, never `object`: --modalities image,text,object would
# mask all three weights to zero, because `object` matches nothing and the parser only rejects an
# empty subset.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

BASE=(--data_path data/ --verbose 5 --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
      --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]"
      --core 5 --topk 10 --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part --fusion softmax
      --dataset lattice-runs/default_fixed)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && { [ "$waited" -gt 0 ] && echo "[mod] ${free} MiB free after ${waited}s"; return 0; }
    [ $(( waited % 600 )) -eq 0 ] && echo "[mod] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                       # run_one <tag> <seed> <flags...>
  local tag="$1" seed="$2"; shift 2
  local name="abl_${tag}_seed${seed}"
  local log="$OUT/${name}.log"
  # Completion from the log's terminator, never the .pt: --dump_embeddings rewrites that file on
  # every best-validation epoch, so an interrupted run already has one.
  if [ -f "$log" ] && { grep -q "Early stop! #####" "$log" || tail -3 "$log" | grep -q "^{'precision'"; }; then
    echo "[mod] $name already complete, skipping"; return 0
  fi
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[mod] === $name (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --seed "$seed" "$@" \
          --dump_embeddings "$OUT/${name}.pt" > "$log" 2>&1; then
      grep "test==" "$log" | tail -1
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" && { echo "[mod] $name OOM attempt $try"; sleep 300; continue; }
    echo "[mod] $name FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  echo "[mod] $name FAILED after 3 OOMs"; return 1
}

for SEED in 0 1 2; do
  run_one txt_only "$SEED" --modalities text  || exit 1
  run_one img_only "$SEED" --modalities image || exit 1
done

echo "[mod] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[mod] FROZEN CHANGED"; exit 1; }
echo "[mod] COMPLETE at $(date -Is)"
