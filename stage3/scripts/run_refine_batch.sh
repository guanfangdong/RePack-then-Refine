#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CKPT_PATH="${1:-./repack_refiner.ckpt}"
LATENT_DIR="${2:-../stage2/output/lightningdit_xl1_repack_f16d32_dinov3/samples/latents}"
REC_DIR="${3:-../stage2/output/lightningdit_xl1_repack_f16d32_dinov3/samples}"
OUTPUT_DIR="${4:-./refined_results}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" refine_batch.py \
  --ckpt_path "${CKPT_PATH}" \
  --latent_dir "${LATENT_DIR}" \
  --rec_dir "${REC_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --z_dim 32 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"
