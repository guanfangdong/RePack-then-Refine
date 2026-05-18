#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG="${1:-configs/lightningdit_xl_repack_f16d32_dinov3.yaml}"
CKPTS="${2:-output/lightningdit_xl1_repack_f16d32_dinov3/checkpoints/0320000.pt}"

accelerate launch --config_file accelerate_config_inference.yaml \
  inference_batch.py \
  --config "${CONFIG}" \
  --ckpts "${CKPTS}"
