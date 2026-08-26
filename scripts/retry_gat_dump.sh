#!/usr/bin/env bash
# Retry the itemgat embedding dump after the OOM at epoch 2 (a competing tenant process, PID
# 877123, pushed combined GPU usage over the 23.51GB cap -- not a code fault; the same config
# ran clean in the earlier 3-seed itemgat study). Queued behind lattice_pooled so it doesn't
# recreate the same contention.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[retry] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[retry] predecessor done at $(date -Is)"
fi

BASE=(--dataset lattice-runs/openai_mit --seed 0 --fusion softmax --data_path data/ --verbose 5
      --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]" --lr 0.0005 --model_name LATTICE
      --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]" --core 5 --topk 10
      --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part)

echo "[retry] === itemgat (GAT), attempt 2 ==="
$PY main.py "${BASE[@]}" --item_prop gat \
    --dump_embeddings objectgraph-eval/embeddings/gat_seed0.pt \
    > objectgraph-eval/embeddings/gat_seed0.log 2>&1 \
    || { echo "[retry] itemgat FAILED AGAIN"; exit 1; }
tail -3 objectgraph-eval/embeddings/gat_seed0.log
echo "[retry] COMPLETE at $(date -Is)"
