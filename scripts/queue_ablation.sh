#!/usr/bin/env bash
# Ablation study, two scenarios, all on default_fixed (MIT+NYU) with the published MF recipe so
# every row differs from the control on exactly one axis. Control = default_fixed 3 seeds
# (0.04332 / 0.04306 / 0.04234), already in downstream.csv and NOT re-run.
#
# SCENARIO A -- per-modality contribution (leave-one-out).
#   The classic multimodal ablation table. image,text (-object) already exists at 3 seeds
#   (fusion_arms.csv noobj); these add the other two leave-one-outs and the object modality
#   standing alone, which is the only arm that shows what the object graph carries by itself.
#
# SCENARIO B -- adaptive fusion weighting.
#   Motivation is measured, not assumed: across 26 runs the learned weights end at
#   image 0.3449 / text 0.3326 / object 0.3225, i.e. <0.02 from the uniform initialisation, on
#   ~130 gradient steps. So the question is whether the "adaptive" weighting does anything.
#     frozen  weights pinned at uniform, projections still trainable -> does LEARNING alpha pay?
#     gated   per-item n_items x 3 gate -> does per-ITEM weighting pay where per-corpus does not?
#     lrfus   100x step size on the fusion params only -> can alpha move if it is allowed to?
#   frozen is the arm that matters: if it ties the control, the adaptive fusion is inert and the
#   paper should say so rather than claim it as a contribution.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[abl] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[abl] predecessor done at $(date -Is)"
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

BASE=(--data_path data/ --verbose 5 --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
      --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]"
      --core 5 --topk 10 --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part
      --dataset lattice-runs/default_fixed)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && return 0
    [ $(( waited % 600 )) -eq 0 ] && echo "[abl] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

# ---------------------------------------------------------------- negative control, first
# A freeze flag that does not bite would produce a clean run with control numbers and nothing
# would say so. Two 6-epoch runs: frozen must hold logit_range at exactly 0.0, unfrozen must not.
# If the flag is inert, the whole of Scenario B is meaningless, so this gate runs before the
# 15-hour queue rather than after it.
echo "[abl] === freeze negative control ==="
wait_for_gpu
$PY main.py "${BASE[@]}" --fusion softmax --seed 0 --epoch 6 \
    > "$OUT/abl_gate_unfrozen.log" 2>&1 || { echo "[abl] gate run failed"; exit 1; }
wait_for_gpu
$PY main.py "${BASE[@]}" --fusion softmax --freeze_fusion 1 --seed 0 --epoch 6 \
    > "$OUT/abl_gate_frozen.log" 2>&1 || { echo "[abl] gate run failed"; exit 1; }
UNFROZEN=$(grep "^LATTICE_FUSION" "$OUT/abl_gate_unfrozen.log" | tail -1 | grep -oP '"logit_range": \K[\d.]+')
FROZEN=$(grep "^LATTICE_FUSION" "$OUT/abl_gate_frozen.log" | tail -1 | grep -oP '"logit_range": \K[\d.]+')
grep -h "^LATTICE_FUSE" "$OUT/abl_gate_frozen.log"
echo "[abl] logit_range after 6 epochs: unfrozen=$UNFROZEN  frozen=$FROZEN"
[ "$FROZEN" = "0.0" ] || { echo "[abl] ABORT: --freeze_fusion did not freeze (got $FROZEN)"; exit 1; }
[ "$UNFROZEN" = "0.0" ] && { echo "[abl] ABORT: control did not move either -- gate proves nothing"; exit 1; }
echo "[abl] freeze flag verified: it binds, and the control moves without it"

run_one () {                      # run_one <tag> <seed> <flags...>
  local tag="$1" seed="$2"; shift 2
  local name="abl_${tag}_seed${seed}"
  local log="$OUT/${name}.log"
  # Completion is decided by the LOG, never by the .pt. --dump_embeddings rewrites that file on
  # every best-validation epoch, so a run killed at epoch 287 of 400 already has one -- skipping
  # on its existence would silently promote a mid-training checkpoint to a final result. The
  # marker is main.py's own terminator: the early-stop banner, or the dict it prints on exit.
  if [ -f "$log" ] && { grep -q "Early stop! #####" "$log" || tail -3 "$log" | grep -q "^{'precision'"; }; then
    echo "[abl] $name already complete, skipping"; return 0
  fi
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[abl] === $name (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --seed "$seed" "$@" \
          --dump_embeddings "$OUT/${name}.pt" > "$log" 2>&1; then
      grep -m1 "^LATTICE_FUSE" "$log"
      grep "test==" "$log" | tail -1
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" \
      && { echo "[abl] $name OOM attempt $try"; sleep 300; continue; }
    echo "[abl] $name FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  echo "[abl] $name FAILED after 3 OOMs"; return 1
}

for SEED in 0 1 2; do
  # Scenario B first: `frozen` is the highest-value arm in this file, so it should not be the
  # thing that gets cut if the queue is interrupted.
  run_one frozen  "$SEED" --fusion softmax --freeze_fusion 1        || exit 1
  run_one gated   "$SEED" --fusion gated                            || exit 1
  # Scenario A: leave-one-out. `image,text` (-object) already exists at 3 seeds.
  run_one no_text "$SEED" --modalities image,graph                  || exit 1
  run_one no_img  "$SEED" --modalities text,graph                   || exit 1
  run_one obj_only "$SEED" --modalities graph                       || exit 1
  # Lower priority: does alpha move at all if given a 100x step?
  run_one lrfus   "$SEED" --fusion softmax --lr_fusion 0.05         || exit 1
done

echo "[abl] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[abl] FROZEN CHANGED"; exit 1; }
echo "[abl] COMPLETE at $(date -Is)"
