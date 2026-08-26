#!/usr/bin/env bash
# MICRO on home_v2 with the object modality fused alongside image and text, three seeds.
#
#   nohup bash run_micro_obj.sh > /tmp/micro_obj_queue.log 2>&1 &
#
# Identical to run_micro.sh in every flag except --modalities image,text,graph, so the paired
# comparison against /tmp/micro_seed{0,1,2}.log isolates the object modality and nothing else.
#
# The object arm is a plain top-k kNN graph over object_feat.npy, built and propagated exactly the
# way image and text are, with no tiebreak handling. That is a deliberate choice rather than an
# omission: object_feat.npy has only 293 unique rows across 14,503 items, and in the LATTICE fork
# every tiebreak repair scored *below* the plain top-k arm (control 0.04332 vs selfloop 0.04258,
# tb_text 0.04138, res2_text 0.04016, tb_rand 0.03850), which itself beat no-object at 0.04007.
#
# Sequential: two other users' jobs share this card.

set -u
cd "$(dirname "$0")"
PY=/home/worker/hamedenv/bin/python
export CUDA_VISIBLE_DEVICES=0
export LATTICE_EVAL_CORES=8

for SEED in 0 1 2; do
  LOG=/tmp/micro_obj_seed${SEED}.log
  echo "=== $(date '+%F %T')  seed=$SEED -> $LOG"
  $PY main.py --dataset home_v2 --gpu_id 0 --seed "$SEED" --epoch 400 \
      --modalities image,text,graph > "$LOG" 2>&1
  echo "=== $(date '+%F %T')  seed=$SEED exit=$?"
  grep -E "Test_Recall@20" "$LOG" | tail -1
done

echo "=== $(date '+%F %T')  micro object queue done"
