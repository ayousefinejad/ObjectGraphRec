#!/usr/bin/env bash
# MICRO on home_v2, published defaults, three seeds.
#
#   nohup bash run_micro.sh > /tmp/micro_queue.log 2>&1 &
#
# Everything except --seed, --gpu_id and --epoch is left at utility/parser.py's defaults, so this
# is the published model: cf_model=lightgcn, weight_size=[64,64], lr=5e-4, topk=10,
# lambda_coeff=0.9, layers=1, loss_ratio=0.03, sparse=1, norm_type=sym, verbose=5, patience=10.
#
# Seeds 0/1/2 rather than parser.py's 123, to match the seed set the LATTICE lgn3 / obj_lgn3 rows
# in docs/crane-experiments.md were collected under -- the eval path (load_data, batch_test,
# metrics) is now byte-identical between the two trees, so the rows are directly comparable and
# the seed set is the last thing that would have differed.
#
# Sequential: two other users' jobs are already on this card, and overlapping our own runs would
# make the per-epoch times unreadable.

set -u
cd "$(dirname "$0")"
PY=/home/worker/hamedenv/bin/python
export CUDA_VISIBLE_DEVICES=0
export LATTICE_EVAL_CORES=8   # pool size only; eval results are identical at any value

# --epoch 400 rather than the parser's 1000: patience 10 x verbose 5 = 50 epochs of no improvement
# ends a run long before that, and 400 is the cap the tuned LATTICE runs used.
for SEED in 0 1 2; do
  LOG=/tmp/micro_seed${SEED}.log
  echo "=== $(date '+%F %T')  seed=$SEED -> $LOG"
  $PY main.py --dataset home_v2 --gpu_id 0 --seed "$SEED" --epoch 400 > "$LOG" 2>&1
  echo "=== $(date '+%F %T')  seed=$SEED exit=$?"
  grep -E "Test_Recall@20" "$LOG" | tail -1
done

echo "=== $(date '+%F %T')  micro queue done"
