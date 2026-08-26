#!/usr/bin/env bash
set -u
cd /home/worker/arshia_yousefi/CRANE/CRANE/src
PY=~/hamedenv/bin/python
run () { echo "[crane] === $1 ==="; $PY run_crane_objgraph.py --arm object --dataset "$2" ${3:+--extra "$3"} || { echo "[crane] $1 FAILED"; exit 1; }; }
run "pooled      (pooled feats, directed graph)" home_v2_pooled ''
run "sym         (argmax feats, symmetric graph)" home_v2_openai '{"obj_knn_sym": true}'
run "pooled_sym  (pooled feats, symmetric graph)" home_v2_pooled '{"obj_knn_sym": true}'
echo "[crane] COMPLETE at $(date -Is)"
