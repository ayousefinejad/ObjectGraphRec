#!/usr/bin/env bash
# LATTICE across the three downstream CF backbones of the LATTICE paper (arXiv:2104.09036,
# Table 3 / section 3.3): MF, NGCF, LightGCN -- crossed with the object modality.
#
# The paper's ablation plugs its item graph into all three and reports LATTICE-LightGCN as the
# headline model; MF is the weakest of the three there (Sports R@20 0.0753 / 0.0856 / 0.0915).
# This whole object-graph study has run --cf_model mf, so every number in it sits on the weakest
# backbone. This batch asks the obvious follow-up: does the object modality still pay on the
# stronger ones, or was it compensating for a weak CF path?
#
#   arm noobj    --modalities image,text   (LATTICE with image+text only, per backbone)
#   arm control  (all three modalities)    (+ object graph)
#
# MF is NOT re-run here: queue_coverage_proof.sh already runs exactly these two arms at these
# three seeds with --cf_model mf on the same variant, so its rows complete the 3x2 grid.
#
# Everything except --cf_model is byte-identical to that batch's BASE, so a backbone row differs
# from the mf row on one axis only.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export LATTICE_EVAL_CORES=10

WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[bb] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[bb] predecessor done at $(date -Is)"
fi

OUT=objectgraph-eval/embeddings
mkdir -p "$OUT"

BASE=(--data_path data/ --verbose 5 --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]"
      --lr 0.0005 --model_name LATTICE --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]"
      --core 5 --topk 10 --lambda_coeff 0.9 --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part --fusion softmax
      --dataset lattice-runs/default_fixed)

NEED_MIB=12000
wait_for_gpu () {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -ge "$NEED_MIB" ] && { [ "$waited" -gt 0 ] && echo "[bb] ${free} MiB free after ${waited}s"; return 0; }
    [ $(( waited % 600 )) -eq 0 ] && echo "[bb] waiting for GPU: ${free} MiB free, need ${NEED_MIB} (${waited}s)"
    sleep 60; waited=$(( waited + 60 ))
  done
}

run_one () {                      # run_one <backbone> <arm> <seed> [extra flags...]
  local bb="$1" arm="$2" seed="$3"; shift 3
  local tag="bb_${bb}_${arm}_seed${seed}"
  local log="$OUT/${tag}.log"
  if [ -s "$OUT/${tag}.pt" ]; then echo "[bb] $tag already done, skipping"; return 0; fi
  local try
  for try in 1 2 3; do
    wait_for_gpu
    echo "[bb] === $tag (attempt $try) === $(date -Is)"
    if $PY main.py "${BASE[@]}" --cf_model "$bb" --seed "$seed" "$@" \
          --dump_embeddings "$OUT/${tag}.pt" > "$log" 2>&1; then
      grep "test==" "$log" | tail -1
      return 0
    fi
    if grep -q "OutOfMemoryError" "$log"; then
      echo "[bb] $tag OOM on attempt $try, backing off"; sleep 300; continue
    fi
    echo "[bb] $tag FAILED (not OOM) -- see $log"; tail -5 "$log"; return 1
  done
  echo "[bb] $tag FAILED after 3 OOMs"; return 1
}

# Two epochs per backbone first. Neither ngcf nor lightgcn has ever been exercised in this repo
# (every run so far is --cf_model mf), and finding out that one of them throws after 40 minutes
# of the first real run would take the whole batch down with it.
for BB in lightgcn ngcf; do
  echo "[bb] smoke $BB"
  wait_for_gpu
  $PY main.py "${BASE[@]}" --cf_model "$BB" --seed 0 --epoch 2 \
      > "$OUT/bb_smoke_${BB}.log" 2>&1 \
      || { echo "[bb] SMOKE FAILED for $BB"; tail -15 "$OUT/bb_smoke_${BB}.log"; exit 1; }
  grep -E "^Epoch 1 .*train==" "$OUT/bb_smoke_${BB}.log" | tail -1
done

for SEED in 0 1 2; do
  for BB in lightgcn ngcf; do          # lightgcn first: it is the paper's headline backbone
    run_one "$BB" control "$SEED"                          || exit 1
    run_one "$BB" noobj   "$SEED" --modalities image,text  || exit 1
  done
done

echo "[bb] verifying the frozen artifacts were not touched..."
(cd .. && md5sum -c objectgraph-eval/frozen_artifacts.md5) || { echo "[bb] FROZEN ARTIFACT CHANGED"; exit 1; }
echo "[bb] COMPLETE at $(date -Is)"
