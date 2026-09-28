#!/bin/bash
# Stage 2 training schedule of the released EvFRA(VFP) model.
#
#   bash scripts/train.sh phase1   # Gaussian start, ControlNet initialized from the SD2 UNet
#   bash scripts/train.sh phase2   # ER-VAE event prior start, from phase1
#   bash scripts/train.sh phase3   # + event-weighted L1, from phase2  (released VFP model)
#   bash scripts/train.sh vfi      # interpolation: forward + backward branches, from phase3
#
# Each phase warm-starts from the previous phase's checkpoint-best
# (override the VFI starting point with VFI_INIT=<checkpoint dir>).
# Effective batch size: 4 GPUs x 1 x 16 accumulation = 64.
set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:?Usage: $0 <phase1|phase2|phase3|vfi>}"
DATA_ROOT="${DATA_ROOT:-data/bs_ergb}"
ER_VAE="${ER_VAE:-experiments/er_vae/best.pt}"
EXP_ROOT="${EXP_ROOT:-experiments}"
NUM_GPUS="${NUM_GPUS:-4}"

COMMON=(
    --train_data_path "$DATA_ROOT/train"
    --val_data_path "$DATA_ROOT/train/horse_04"
    --er_vae_path "$ER_VAE"
    --per_gpu_batch_size 1
    --gradient_accumulation_steps 16
    --lr_warmup_steps 1000
    --lambda_lpips 1.0
    --lambda_pixel_l1 1.0
    --sigma_min 0.05
    --sigma_max 3.0
    --loss_weight_max 500
    --use_ema
    --mixed_precision fp16
    --seed 123
)

warm_start() {
    local ckpt="$1"
    [[ -d "$ckpt/controlnet_ema" ]] || { echo "Missing $ckpt" >&2; exit 1; }
    echo --controlnet_model_name_or_path "$ckpt/controlnet_ema" --latent_tokenizer_path "$ckpt/latent_tokenizer.pth"
}

case "$PHASE" in
    phase1)
        ARGS=(--output_dir "$EXP_ROOT/phase1" --max_train_steps 100000 --learning_rate 2e-5 --lambda_event_l1 0.0) ;;
    phase2)
        ARGS=(--output_dir "$EXP_ROOT/phase2" --max_train_steps 80000 --learning_rate 1e-5 --lambda_event_l1 0.0
              --use_event_prior $(warm_start "$EXP_ROOT/phase1/checkpoint-best")) ;;
    phase3)
        ARGS=(--output_dir "$EXP_ROOT/phase3" --max_train_steps 35000 --learning_rate 5e-6 --lambda_event_l1 0.5
              --use_event_prior $(warm_start "$EXP_ROOT/phase2/checkpoint-best")) ;;
    vfi)
        ARGS=(--output_dir "$EXP_ROOT/vfi" --task vfi --max_train_steps 30000 --learning_rate 5e-6 --lambda_event_l1 0.5
              --use_event_prior $(warm_start "${VFI_INIT:-$EXP_ROOT/phase3/checkpoint-best}")) ;;
    *)
        echo "Unknown phase: $PHASE" >&2; exit 1 ;;
esac

MULTI_GPU=()
(( NUM_GPUS > 1 )) && MULTI_GPU=(--multi_gpu)

accelerate launch --num_processes "$NUM_GPUS" "${MULTI_GPU[@]}" train.py "${COMMON[@]}" "${ARGS[@]}" "${@:2}"
