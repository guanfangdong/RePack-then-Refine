#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

DATA_PATH="${1:-/path/to/imagenet}"
OUTPUT_PATH="${2:-./imagenet_features}"
CONFIG="${3:-configs/repack_f16d32_dinov3.yaml}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-8}"

accelerate launch --config_file accelerate_config.yaml \
  extract_features.py \
  --config "${CONFIG}" \
  --data_path "${DATA_PATH}" \
  --data_split train \
  --output_path "${OUTPUT_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
