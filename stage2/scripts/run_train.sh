#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONFIG="${1:-configs/lightningdit_xl_repack_f16d32_dinov3.yaml}"

accelerate launch --config_file accelerate_config.yaml \
  train.py \
  --config "${CONFIG}"
