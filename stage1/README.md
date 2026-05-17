# RePack Stage 1

This folder contains the Stage 1 representation-packing code for RePack:

- frozen DINOv3-B/16 feature extraction
- bias-free 1x1 projection to f16d32 packed latents
- decoder training with LPIPS/GAN, focal-frequency loss, and Watson loss
- reconstruction and latent visualization scripts

## Checkpoint

The released Stage 1 checkpoint is:

- File: `repack-dinov3b-s16-ch32.ckpt`
- Download:
  [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link)

We plan to also host this checkpoint on Hugging Face later. For now, we provide
the Google Drive link first so users can download it quickly.

## Setup

Install the project dependencies in your training environment:

```bash
pip install -r requirements.txt
```

The minimal `taming` modules required by LPIPS/GAN training are bundled in this
folder, so no separate `taming-transformers` checkout is required for Stage 1.

## Train

Edit `configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml`:

- set `data.params.train.params.txt_file` to your training image list
- set `model.params.ddconfig.model_name` to a local DINOv3-B path if you do not want Hugging Face loading
- optionally set `init_weight` when continuing from a previous stage-1 checkpoint

Run:

```bash
cd public_release/stage1
bash scripts/train_stage1.sh configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml
```

Checkpoints are saved under `logs/<run_name>/checkpoints`.

## Reconstruct

```bash
cd public_release/stage1
python reconstruct_stage1.py \
  --config configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml \
  --ckpt /path/to/repack-dinov3b-s16-ch32.ckpt \
  --input-list /path/to/val_paths.txt \
  --out outputs/reconstruction
```

This writes aligned crops to `gt/` and reconstructions to `recon/`.

## Visualize Latents

```bash
cd public_release/stage1
python visualize_stage1_latents.py \
  --config configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml \
  --ckpt /path/to/repack-dinov3b-s16-ch32.ckpt \
  --input-list /path/to/vis_paths.txt \
  --out outputs/latent_vis
```

Each output image is `[input | reconstruction | latent PCA]`.
