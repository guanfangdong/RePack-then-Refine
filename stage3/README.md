# Stage 3: Latent-Guided Refinement

Stage 3 trains and applies the Latent-Guided Refiner. The release path is **offline**: the Refiner is trained from saved triplets `(I_rec, z, x_gt)` and, during inference, reads the base images and generated latents saved by Stage 2.

The original working folder also contained an online training path that reconstructs images through the Stage-1 autoencoder during training. For public release we keep the offline path as the default because it is simpler, faster to reproduce, and matches the inference script used for the reported results.

## Stage 3 Overview

| Step | Script | Input | Output |
|---|---|---|---|
| 1. Prepare offline triplets | `scripts/prepare_offline_data.sh` | ImageNet + Stage-1 RePack checkpoint | `offline_refiner_data/{gt,rec,latent}` |
| 2. Train Refiner | `scripts/run_train.sh` | offline triplets | Refiner checkpoints |
| 3. Refine generated images | `scripts/run_refine_batch.sh` | Stage-2 PNGs + Stage-2 `latents/` | refined PNG folder |
| 4. Compute FID | `scripts/run_fid.sh` | refined PNG folder + ADM reference `.npz` | FID JSON |

## Released Checkpoint

| Name | Module | Download |
|---|---|---|
| `repack_refiner.ckpt` | Latent-Guided Refiner | [Google Drive](https://drive.google.com/file/d/1--eR0xFgJ9ivUMdb4xO4SzrzjGhzV3OD/view?usp=sharing) |

## Try the Refiner Directly

If you only want to test the Refiner, first download:

| File | Link |
|---|---|
| Stage-3 checkpoint: `repack_refiner.ckpt` | [Google Drive](https://drive.google.com/file/d/1--eR0xFgJ9ivUMdb4xO4SzrzjGhzV3OD/view?usp=sharing) |
| Stage-2 50-step samples: `lightningdit-xl-1-ckpt-0320000-euler-50-interval0.11-cfg15.00-shift0.00.zip` | [Google Drive](https://drive.google.com/file/d/1DhhvQcUjW-Hnzp2VrUVNGUyUpmGMafDq/view?usp=sharing) |

The sample zip should contain decoded PNG images and a `latents/` folder produced by `stage2/inference_batch.py`.

Run:

```bash
cd stage3
bash scripts/run_refine_batch.sh \
  /path/to/repack_refiner.ckpt \
  /path/to/stage2_samples/latents \
  /path/to/stage2_samples \
  ./refined_results
```

This writes refined images to `./refined_results`.

## FID

Stage 3 reuses the Stage 2 FID implementation through `../stage2/tools/calculate_fid.py`, so keep `stage2/` and `stage3/` as sibling folders in the released repository.

```bash
bash scripts/run_fid.sh \
  ./refined_results \
  /path/to/VIRTUAL_imagenet256_labeled.npz \
  ./refined_results/fid_results.json
```

For the provided 50-step sample set, we measured:

| Setting | FID |
|---|---:|
| RePack-DiT, no Refiner | 1.87 |
| RePack-DiT + Refiner | 1.76 |

The best reported FID of `1.65` is obtained with 250-step inference before refinement, which is slower than the 50-step preview setting.

## Offline Training

Prepare ImageNet and set the Stage-1 checkpoint path in:

```bash
../stage2/configs/repack_f16d32_dinov3.yaml
```

Then build offline training triplets:

```bash
bash scripts/prepare_offline_data.sh \
  /path/to/imagenet \
  ./offline_refiner_data \
  ../stage2/configs/repack_f16d32_dinov3.yaml
```

The expected data layout is:

```text
offline_refiner_data/
  gt/
    *.png
  rec/
    *.png
  latent/
    *.pt
```

Train:

```bash
bash scripts/run_train.sh configs/refiner_offline.yaml
```

The default config uses `z_dim=32`, `base_channels=128`, `L1 + LPIPS + PatchGAN`, and saves checkpoints under `logs_refiner/checkpoints`.

## Notes

| Item | Detail |
|---|---|
| Latent input | Use the `.pt` files saved by `stage2/inference_batch.py` for generated samples. |
| Base image input | Use the decoded PNG folder from the same Stage-2 sampling run. |
| LPIPS dependency | Training with `lpips_weight > 0` requires `pip install lpips`. |
| Output names | By default refined images keep the same basename as the Stage-2 samples, which is convenient for FID. |
