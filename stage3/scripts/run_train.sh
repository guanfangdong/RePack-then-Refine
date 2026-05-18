#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

CONFIG="${1:-configs/refiner_offline.yaml}"
RESUME="${2:-}"

if [[ -n "${RESUME}" ]]; then
  python train_refiner.py --config "${CONFIG}" --resume "${RESUME}"
else
  python train_refiner.py --config "${CONFIG}"
fi
