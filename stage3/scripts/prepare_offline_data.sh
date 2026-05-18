#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

IMAGE_ROOT="${1:-/path/to/imagenet}"
OUTPUT_DIR="${2:-./offline_refiner_data}"
REPACK_CONFIG="${3:-../stage2/configs/repack_f16d32_dinov3.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" prepare_offline_data.py \
  --image_root "${IMAGE_ROOT}" \
  --config "${REPACK_CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
