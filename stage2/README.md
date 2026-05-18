# Stage 2: RePack-DiT Generative Modeling

Stage 2 trains the Diffusion Transformer on compact RePack latents. The released code keeps this stage focused on the ImageNet-1K feature cache, DiT training, batch inference, latent saving, and FID evaluation.

## Stage 2 Overview

| Step | Script | Main Config | Output |
|---|---|---|---|
| 1. Cache RePack latents | `scripts/run_extraction.sh` | `configs/repack_f16d32_dinov3.yaml` | `imagenet_features/repack_f16d32_dinov3/train_256` |
| 2. Train RePack-DiT | `scripts/run_train.sh` | `configs/lightningdit_xl_repack_f16d32_dinov3.yaml` | `output/lightningdit_xl1_repack_f16d32_dinov3/checkpoints` |
| 3. Continue with lower LR | `scripts/run_train.sh` | `configs/lower_lr_lightningdit_xl_repack_f16d32_dinov3.yaml` | final 64-epoch checkpoint |
| 4. Batch inference + latents | `scripts/run_inference_batch.sh` | training config | PNG samples + generated latent `.pt` files |
| 5. FID evaluation | built into `inference_batch.py` or `scripts/run_fid.sh` | `fid_reference_file` | `fid_results.json` and `fid_results.jsonl` |

## 1. Prepare ImageNet

Please prepare ImageNet-1K locally. The extraction script uses `torchvision.datasets.ImageFolder`, so the training split should look like:

```text
/path/to/imagenet/
  n01440764/
    *.JPEG
  n01443537/
    *.JPEG
  ...
```

## 2. Cache RePack Features

First set the Stage-1 checkpoint path in:

```bash
configs/repack_f16d32_dinov3.yaml
```

The expected checkpoint filename is:

```text
repack-dinov3b-s16-ch32.ckpt
```

Then run:

```bash
cd stage2
bash scripts/run_extraction.sh /path/to/imagenet ./imagenet_features
```

This creates:

```text
imagenet_features/repack_f16d32_dinov3/train_256/
  latents_rank00_shard000.safetensors
  ...
  latents_stats.pt
```

The Stage-2 tokenizer wrapper reuses the Stage-1 `repack` modules from the sibling `stage1/` folder instead of duplicating the RePack autoencoder code.

## 3. Train RePack-DiT

Edit the paths in:

```bash
configs/lightningdit_xl_repack_f16d32_dinov3.yaml
```

At minimum, check:

| Key | Meaning |
|---|---|
| `data.data_path` | cached RePack latent directory |
| `data.fid_reference_file` | ImageNet 256x256 FID reference `.npz` |
| `train.output_dir` | where checkpoints and logs are saved |
| `train.exp_name` | experiment name |

Start training:

```bash
bash scripts/run_train.sh configs/lightningdit_xl_repack_f16d32_dinov3.yaml
```

For strict reproduction of the paper curve in Figure 1, the high-learning-rate phase runs to `240000` optimizer steps. The default config trains this first phase with LR `2e-4`.

Then continue with:

```bash
bash scripts/run_train.sh configs/lower_lr_lightningdit_xl_repack_f16d32_dinov3.yaml
```

The lower-LR config resumes from `0240000.pt`, switches LR to `1e-4`, and trains to `320000` steps, corresponding to the 64-epoch setting used in the paper. If you change the global batch size, adjust the step counts accordingly.

## Released Checkpoint

| Name | Model | Training | Download |
|---|---|---|---|
| `repack_dit_xl1_ep64.pt` | RePack-DiT-XL/1 | 64 epochs, high LR to 240k steps, low LR to 320k steps | [Google Drive](https://drive.google.com/file/d/14S7WXprhUVh-XlR7eaX5yRw-cms_s6BR/view?usp=sharing) |

## Preview Samples for Refiner

If you only want to inspect generated images or try the Stage-3 Refiner, you can directly download the 50-step inference results:

| File | Sampling | FID without Refiner | FID with Refiner | Download |
|---|---|---:|---:|---|
| `lightningdit-xl-1-ckpt-0320000-euler-50-interval0.11-cfg15.00-shift0.00.zip` | Euler, 50 steps, CFG 15.0, interval start 0.11, shift 0.00 | 1.87 | 1.76 | [Google Drive](https://drive.google.com/file/d/1DhhvQcUjW-Hnzp2VrUVNGUyUpmGMafDq/view?usp=sharing) |

The best reported FID of `1.65` is obtained with 250-step inference, which is slower but gives higher fidelity.

## 4. Batch Inference

Use `inference_batch.py` through the provided shell script:

```bash
bash scripts/run_inference_batch.sh \
  configs/lightningdit_xl_repack_f16d32_dinov3.yaml \
  output/lightningdit_xl1_repack_f16d32_dinov3/checkpoints/0320000.pt
```

Please use this batch inference path for evaluation. It saves both:

| Artifact | Location |
|---|---|
| decoded PNG images | the sample output folder |
| generated RePack latents | `latents/*.pt` inside the same sample folder |
| per-checkpoint FID JSON | `fid_results.json` inside the sample folder |
| all FID records | `output/<exp_name>/fid_results.jsonl` |

The saved latents are needed by Stage 3 for latent-guided refinement.

## 5. Standalone FID

If you already have generated samples and only want to recompute FID:

```bash
bash scripts/run_fid.sh \
  /path/to/generated_png_folder \
  /path/to/VIRTUAL_imagenet256_labeled.npz \
  /path/to/fid_results.json
```

The same FID implementation is used by `inference_batch.py`.
