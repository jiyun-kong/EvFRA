#!/bin/bash
# Stage 1: ER-VAE training on BS-ERGB.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:-data/bs_ergb}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/er_vae}"

python train_er_vae.py \
    --train_data_path "$DATA_ROOT/train" \
    --val_data_path "$DATA_ROOT/valid" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 4 \
    --epochs 200 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --gamma_min 0.3 \
    --gamma_max 0.7 \
    --precision fp16 \
    "$@"
