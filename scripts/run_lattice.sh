#!/bin/bash

# Script to run LATTICE model with graph modality
# Based on configuration from LATTICE-Filter-data-object-graph.ipynb

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the project root directory (parent of scripts directory)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root directory
cd "$PROJECT_ROOT"

# Run the main script
python main.py \
    --data_path 'data/' \
    --seed 123 \
    --dataset 'home_v2-2' \
    --verbose 5 \
    --epoch 200 \
    --batch_size 1024 \
    --regs '[1e-5,1e-5,1e-2]' \
    --lr 0.0005 \
    --model_name 'LATTICE' \
    --embed_size 64 \
    --feat_embed_dim 64 \
    --weight_size '[64,64]' \
    --core 5 \
    --topk 10 \
    --lambda_coeff 0.9 \
    --cf_model 'mf' \
    --n_layers 1 \
    --mess_dropout '[0.1, 0.1]' \
    --early_stopping_patience 10 \
    --gpu_id 0 \
    --Ks '[10, 20]' \
    --test_flag 'part'

