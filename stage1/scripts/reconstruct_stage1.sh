#!/usr/bin/env bash
set -euo pipefail

python reconstruct_stage1.py \
  --config configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml \
  --ckpt /path/to/repack_stage1.ckpt \
  --input-list /path/to/val_paths.txt \
  --out outputs/reconstruction
