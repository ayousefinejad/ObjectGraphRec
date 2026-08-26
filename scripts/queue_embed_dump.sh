#!/usr/bin/env bash
# Train the control (dense) and itemgat (GAT) arms once each, dumping best-epoch user/item
# embeddings for the t-SNE comparison figure. Same recipe as the 3-seed itemgat study, seed 0
# only -- one representative trained model per arm, matching the FREEDOM-vs-CRANE figure's own
# single-model-per-panel design.
set -u
cd /home/worker/arshia_yousefi/object-graph
PY=~/hamedenv/bin/python
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[queue] waiting for PID $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[queue] predecessor done at $(date -Is)"
fi

# An array, not a string: the earlier string form word-split "[0.1, 0.1]" and "[10, 20]" on
# their internal space (two tokens each), which argparse then rejected as unrecognized. Every
# other queue_*.sh in this project passes args inline for the same reason -- this file was the
# one exception, and it broke on exactly the args with a space inside brackets.
BASE=(--dataset lattice-runs/openai_mit --seed 0 --fusion softmax --data_path data/ --verbose 5
      --epoch 200 --batch_size 1024 --regs "[1e-5,1e-5,1e-2]" --lr 0.0005 --model_name LATTICE
      --embed_size 64 --feat_embed_dim 64 --weight_size "[64,64]" --core 5 --topk 10
      --lambda_coeff 0.9 --cf_model mf --n_layers 1 --mess_dropout "[0.1, 0.1]"
      --early_stopping_patience 10 --gpu_id 0 --Ks "[10, 20]" --test_flag part)

mkdir -p objectgraph-eval/embeddings
echo "[embed-dump] === control (dense) ==="
$PY main.py "${BASE[@]}" --dump_embeddings objectgraph-eval/embeddings/dense_seed0.pt \
    > objectgraph-eval/embeddings/dense_seed0.log 2>&1 \
    || { echo "[embed-dump] control FAILED"; exit 1; }
tail -3 objectgraph-eval/embeddings/dense_seed0.log

echo "[embed-dump] === itemgat (GAT) ==="
$PY main.py "${BASE[@]}" --item_prop gat --dump_embeddings objectgraph-eval/embeddings/gat_seed0.pt \
    > objectgraph-eval/embeddings/gat_seed0.log 2>&1 \
    || { echo "[embed-dump] itemgat FAILED"; exit 1; }
tail -3 objectgraph-eval/embeddings/gat_seed0.log

echo "[embed-dump] COMPLETE at $(date -Is)"
