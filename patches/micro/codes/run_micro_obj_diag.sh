#!/usr/bin/env bash
# Diagnostic: which pathway makes the object modality hurt MICRO?
#
#   nohup bash run_micro_obj_diag.sh > /tmp/micro_diag_queue.log 2>&1 &
#
# MICRO feeds every modality into BOTH the softmax fusion and its own InfoNCE term, so the measured
# regression (0.05063 image+text -> 0.04274 with object, all 3 seeds) has two possible causes. These
# two arms separate them:
#
#   fusion    graph in the attention, NOT in the contrastive loss
#   contrast  graph in the contrastive loss, NOT in the attention
#
# Reference points already collected, same flags otherwise:
#   image,text          0.05041 / 0.05084 / 0.05063   mean 0.05063
#   +graph, both        0.04282 / 0.04236 / 0.04305   mean 0.04274
#
# Prior: the InfoNCE term is the likelier culprit -- it asks the fusion to be discriminative w.r.t.
# a modality whose kNN graph reaches only 2,219 distinct items out of 14,503.
#
# Seed-major ordering, so an interrupted queue still leaves a complete paired comparison at each
# seed rather than one finished arm and one empty. Sequential: two other users share this card.

set -u
cd "$(dirname "$0")"
PY=/home/worker/hamedenv/bin/python
export CUDA_VISIBLE_DEVICES=0
export LATTICE_EVAL_CORES=8

for SEED in 0 1 2; do
  for MODE in fusion contrast; do
    LOG=/tmp/micro_obj_${MODE}_seed${SEED}.log
    echo "=== $(date '+%F %T')  mode=$MODE seed=$SEED -> $LOG"
    $PY main.py --dataset home_v2 --gpu_id 0 --seed "$SEED" --epoch 400 \
        --modalities image,text,graph --graph_mode "$MODE" > "$LOG" 2>&1
    echo "=== $(date '+%F %T')  mode=$MODE seed=$SEED exit=$?"
    grep -oP "Test_Recall@20: \K[\d.]+" "$LOG" | sort -rn | head -1
  done
done

echo "=== $(date '+%F %T')  micro diagnostic queue done"
