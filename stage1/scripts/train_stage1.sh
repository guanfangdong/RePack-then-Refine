#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml}
GPUS=${GPUS:-4}
PORT=${PORT:-29500}

torchrun \
  --nproc_per_node="${GPUS}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${PORT}" \
  train_stage1.py \
  --config "${CONFIG}" \
  --devices "${GPUS}"

