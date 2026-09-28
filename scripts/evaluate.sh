#!/bin/bash
# Reproduces the EvFRA rows of Tab. 1 and Tab. 2.
#
#   bash scripts/evaluate.sh <bs_ergb|hs_ergb|gopro> [SHARD_ID NUM_SHARDS]            # VFP
#   TASK=vfi CKPT=checkpoints/evfra_vfi bash scripts/evaluate.sh <dataset> ...       # VFI
#
# Several GPUs can share one setting by running different SHARD_IDs.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:?Usage: $0 <bs_ergb|hs_ergb|gopro> [SHARD_ID NUM_SHARDS]}"
SHARD_ID="${2:-0}"
NUM_SHARDS="${3:-1}"
TASK="${TASK:-vfp}"
CKPT="${CKPT:-checkpoints/evfra}"
DATA="${DATA:-data}"
OUT="${OUT:-results}"

case "$DATASET" in
    bs_ergb) FRAMES=(1 3) ;;
    hs_ergb) FRAMES=(7) ;;
    gopro)   FRAMES=(7 15) ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 1 ;;
esac

for K in "${FRAMES[@]}"; do
    python evaluate.py \
        --task "$TASK" \
        --dataset "$DATASET" \
        --data_root "$DATA/$DATASET" \
        --num_frames "$K" \
        --output_dir "$OUT/${TASK}_${DATASET}_${K}frames" \
        --controlnet_path "$CKPT/controlnet" \
        --latent_tokenizer_path "$CKPT/latent_tokenizer.pth" \
        --er_vae_path "$CKPT/er_vae.pt" \
        --shard_id "$SHARD_ID" --num_shards "$NUM_SHARDS"
done
