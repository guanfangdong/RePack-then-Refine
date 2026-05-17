#!/usr/bin/env bash
set -euo pipefail

python visualize_stage1_latents.py \
  --config configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml \
  --ckpt /path/to/repack_stage1.ckpt \
  --input-list /path/to/vis_paths.txt \
  --out outputs/latent_vis \
  --limit 32
