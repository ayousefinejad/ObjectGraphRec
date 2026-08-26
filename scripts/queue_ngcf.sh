#!/usr/bin/env bash
# NGCF backbone, +/- object graph, 3 seeds -- the missing row of the backbone table.
#
# Two corrections over this morning's sweep, which is why those runs were discarded rather than
# extended:
#
#   DEPTH 1, not 2. The sweep inherited --weight_size [64,64] from the MF recipe, but tuning.csv
#   shows NGCF is BETTER at depth 1 than depth 2 on image+text (0.03906 vs 0.03698) -- the
#   opposite of LightGCN, which improves to depth 3. Reporting NGCF at depth 2 would understate
#   the backbone and make the object-graph comparison sit on a handicapped baseline.
#
#   --epoch 400, not 200. NGCF's best validation epoch is 220 (tuning.csv ngcf2), so a 200-epoch
#   cap truncates it mid-improvement. This morning's depth-2 run stopped at best_epoch 195
#   against a 200 cap -- almost certainly truncated, which alone could explain its weak number.
#
# Flags match tuning.csv's `ngcf1` cell exactly (--weight_size [64], --mess_dropout [0.1,0.1,0.1]
# pinned at length 3 as that cell does), so the image+text seed-0 row here should reproduce
# 0.03906 and thereby validate the whole batch. Both arms are run at all three seeds so the pair
# is protocol-matched end to end -- the flaw that made this morning's NGCF delta uninterpretable.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[ngcf] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[ngcf] predecessor done at $(date -Is)"
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

NG=(--data_path data/ --verbose 5 --epoch 400 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
    --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64
    --weight_size "[64]" --core 5 --topk 10 --lambda_coeff 0.9 --n_layers 1
    --mess_dropout "[0.1,0.1,0.1]" --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]"
    --test_flag part --fusion softmax --cf_model ngcf --dataset lattice-runs/default_fixed)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && return 0
    [ $(( waited % 600 )) -eq 0 ] && echo "[ngcf] waiting for GPU: ${free} MiB free (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                      # run_one <arm> <seed> <flags...>
  local arm="$1" seed="$2"; shift 2
  local tag="ngcf1_${arm}_seed${seed}"
  local log="$OUT/${tag}.log"
  [ -s "$OUT/${tag}.pt" ] && { echo "[ngcf] $tag done, skipping"; return 0; }
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[ngcf] === $tag (attempt $try) === $(date -Is)"
    if $PY main.py "${NG[@]}" --seed "$seed" "$@" \
          --dump_embeddings "$OUT/${tag}.pt" > "$log" 2>&1; then
      grep "test==" "$log" | tail -1
      # A run that stops at the cap was still improving; its number is a floor, not a result.
      local best cap
      best=$(grep "test==" "$log" | tail -1 | grep -oP '^Epoch \K\d+')
      cap=400
      [ "$best" -ge $(( cap - 50 )) ] && echo "[ngcf] WARNING $tag best_epoch=$best near cap $cap"
      return 0
    fi
    grep -q "OutOfMemoryError" "$log" \
      && { echo "[ngcf] $tag OOM attempt $try"; sleep 300; continue; }
    echo "[ngcf] $tag FAILED (not OOM)"; tail -5 "$log"; return 1
  done
  echo "[ngcf] $tag FAILED after 3 OOMs"; return 1
}

for SEED in 0 1 2; do
  run_one control "$SEED" --modalities image,text,graph || exit 1
  run_one noobj   "$SEED" --modalities image,text       || exit 1
done

echo "[ngcf] cross-check: seed0 image+text should reproduce tuning.csv ngcf1 = 0.03906"
grep "test==" "$OUT/ngcf1_noobj_seed0.log" | tail -1

echo "[ngcf] verifying frozen artifacts..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[ngcf] FROZEN CHANGED"; exit 1; }
echo "[ngcf] COMPLETE at $(date -Is)"
