#!/usr/bin/env bash
set -euo pipefail

IMG_DIR="${1:-./refined_results}"
REF_NPZ="${2:-/path/to/VIRTUAL_imagenet256_labeled.npz}"
OUTPUT="${3:-${IMG_DIR}/fid_results.json}"

python eval_fid.py \
  --img_dir "${IMG_DIR}" \
  --ref_npz "${REF_NPZ}" \
  --fid_num "${FID_NUM:-50000}" \
  --output "${OUTPUT}"
