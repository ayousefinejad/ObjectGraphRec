#!/usr/bin/env bash
# rho sweep: the gain on the graph-enhanced item representation in paper Eq. 8,
#
#     x_hat_i = x_bar_i + rho * h_i / ||h_i||
#
# The paper fixes rho at 1 implicitly and never tunes it (Sec 2.4 "simply enhance item embeddings
# by adding normalized item embeddings"), so this is an unexamined assumption rather than a
# tuned value. Two reasons it deserves examination here:
#
#   * It is not a mixing ratio. ||h/||h|||| is exactly 1 by construction while ||x_bar|| is
#     unconstrained; measured at init, ||x_bar|| = 0.109, so rho=1 makes the graph path ~9x the
#     CF path. rho_effective = rho/||x_bar|| is logged at every eval to track how that drifts.
#   * With --cf_model mf every modality reaches the score ONLY through h, so rho is the single
#     global gain on the whole multimodal pathway -- a more direct control than alpha, which only
#     redistributes weight among modalities.
#
# Run at BOTH arms so the question is not just "what is the best rho" but "does adding the object
# graph change the best rho". If the object modality contributed noise, its optimum should sit
# lower than image+text's.
#
# rho=1 is NOT re-run: it is the published setting and already has 3 seeds on disk
# (+object 0.04291 +- 0.00051, image+text 0.04007 +- 0.00031; seed 0 = 0.04332 / 0.04043).
# The rho=1 code path was verified bit-identical to the pre-change code before this batch.
#
# rho=0 is the arm no one has run: the item graph removed entirely, leaving the bare CF model.
# It bounds everything else in the study.
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
    [ "$free" -ge "$NEED_MIB" ] && return 0
    [ $(( waited % 600 )) -eq 0 ] && echo "[rho] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                       # run_one <rho> <arm> <seed> <modality flags...>
  local rho="$1" arm="$2" seed="$3"; shift 3
  local tag="rho${rho}_${arm}_seed${seed}"
  local log="$OUT/${tag}.log"
  # Completion from the log's terminator, never the .pt (which is rewritten every best epoch).
  if [ -f "$log" ] && { grep -q "Early stop! #####" "$log" || tail -3 "$log" | grep -q "^{'precision'"; }; then
    echo "[rho] $tag already complete, skipping"; return 0
  fi
  local try
  for try in 1 2; do
    wait_for_gpu
    echo "[rho] === $tag (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --rho "$rho" --seed "$seed" "$@" > "$log" 2>&1; then
      grep "test==" "$log" | tail -1
      grep "^LATTICE_FUSION" "$log" | tail -1 | grep -oP '"rho[^,]*|"cf_norm[^,]*' | tr '\n' ' '; echo
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" && { echo "[rho] $tag OOM"; sleep 300; continue; }
    echo "[rho] $tag FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  return 1
}

# Seed 0 first across the whole grid: the curve's SHAPE is what decides where to spend seeds,
# and an interrupted batch then leaves a complete curve rather than one arm finished and one not.
for RHO in 0 0.25 0.5 2 4; do
  run_one "$RHO" obj   0                                || exit 1
  run_one "$RHO" noobj 0 --modalities image,text        || exit 1
done

echo "[rho] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[rho] FROZEN CHANGED"; exit 1; }
echo "[rho] COMPLETE at $(date -Is)"
