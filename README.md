# ⚡ RePack then Refine: Efficient Diffusion Transformers with Vision Foundation Models

**Official implementation of the ICML 2026 paper _"RePack then Refine: Efficient Diffusion Transformers with Vision Foundation Models"_.**
A highly efficient DiT framework achieving **1.65 FID on ImageNet-1K in only
64 epochs** by leveraging compressed Vision Foundation Model (VFM) features and
latent-guided refinement.

**Authors:** Guanfang Dong, Luke Schultz, Negar Hassanpour, Chao Gao

> Current release status: **Stage 1, Stage 2, and Stage 3 are ready for the current release.**

| Paper | Code | Checkpoints | Status |
|:--|:--|:--|:--|
| [arXiv](https://arxiv.org/abs/2512.12083) | [`stage1`](stage1), [`stage2`](stage2), [`stage3`](stage3) | [Stage 1](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link), [Stage 2](https://drive.google.com/file/d/14S7WXprhUVh-XlR7eaX5yRw-cms_s6BR/view?usp=sharing), [Stage 3](https://drive.google.com/file/d/1--eR0xFgJ9ivUMdb4xO4SzrzjGhzV3OD/view?usp=sharing) | 🚧 Not camera-ready yet |

## ✨ Highlights

| Feature | Description |
|:--|:--|
| 🧠 VFM latents | Uses semantic-rich DINOv3-B/16 patch features as the representation source. |
| 📦 RePack compression | Projects high-dimensional VFM features into a compact **f16d32** latent space. |
| ⚡ Efficient DiT training | Trains diffusion in the packed latent space for much faster convergence. |
| 🎨 Latent-guided refinement | Restores high-frequency details from the decoded image and packed latent guidance. |
| 🏁 Strong ImageNet result | Achieves **1.65 FID** on ImageNet-1K with the Refiner after only **64 epochs**. |

## 🧩 Method Overview

RePack then Refine is organized as three independent stages:

| Stage | Module | What It Does | Release Status |
|:--:|:--|:--|:--|
| 1 | **Representation Packing** | Freezes DINOv3-B/16, compresses patch features into f16d32 latents, and trains a decoder with reconstruction, LPIPS/GAN, focal-frequency, and Watson losses. | ✅ Ready |
| 2 | **Generative Modeling** | Caches RePack latents, trains RePack-DiT, runs batch inference, saves generated latents, and evaluates FID. | ✅ Ready |
| 3 | **Latent-Guided Refinement** | Refines decoded base images using upsampled packed latents as structural guidance and supports direct FID evaluation. | ✅ Ready |

Raw VFM features are powerful but high-dimensional and redundant. RePack first
filters the redundancy by learning a compact latent manifold. DiT then learns
generation in this compact space. Finally, the Refiner restores local textures
and high-frequency details that may be weakened by compression.

## 📌 Current Release

| Item | Path / Link | Notes |
|:--|:--|:--|
| Paper draft | [`4234_RePack_then_Refine_Effici.pdf`](4234_RePack_then_Refine_Effici.pdf) | Draft version included in the release. |
| Stage 1 code | [`stage1`](stage1) | Training, reconstruction, and latent visualization. |
| Stage 1 README | [`stage1/README.md`](stage1/README.md) | Start here for representation packing usage. |
| Stage 1 checkpoint | [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link) | `repack-dinov3b-s16-ch32.ckpt` |
| Stage 2 code | [`stage2`](stage2) | Feature extraction, RePack-DiT training, batch inference, and FID evaluation. |
| Stage 2 README | [`stage2/README.md`](stage2/README.md) | Start here after downloading the Stage 1 checkpoint and preparing ImageNet. |
| Stage 2 checkpoint | [Google Drive](https://drive.google.com/file/d/14S7WXprhUVh-XlR7eaX5yRw-cms_s6BR/view?usp=sharing) | `repack_dit_xl1_ep64.pt` |
| 50-step generated samples | [Google Drive](https://drive.google.com/file/d/1DhhvQcUjW-Hnzp2VrUVNGUyUpmGMafDq/view?usp=sharing) | `lightningdit-xl-1-ckpt-0320000-euler-50-interval0.11-cfg15.00-shift0.00.zip` |
| Stage 3 code | [`stage3`](stage3) | Offline Refiner training, batch refinement, and FID evaluation. |
| Stage 3 README | [`stage3/README.md`](stage3/README.md) | Start here for using the released Refiner checkpoint. |
| Stage 3 checkpoint | [Google Drive](https://drive.google.com/file/d/1--eR0xFgJ9ivUMdb4xO4SzrzjGhzV3OD/view?usp=sharing) | `repack_refiner.ckpt` |

The checkpoint will also be mirrored on Hugging Face later. For now, we provide
the Google Drive link first so users can download it quickly.

## 🗂️ Repository Layout

The uploaded repository is organized directly by stages. Stage 1, Stage 2, and
Stage 3 are available in the current release.

| Directory / File | Purpose |
|:--|:--|
| `4234_RePack_then_Refine_Effici.pdf` | Paper draft. |
| `stage1/` | Representation packing code, configs, scripts, and checkpoint usage. |
| `stage1/configs/` | Stage 1 YAML configs. |
| `stage1/repack/` | Clean `repack.*` implementation for representation packing. |
| `stage1/scripts/` | Shell helpers for training, reconstruction, and visualization. |
| `stage1/taming/` | Minimal bundled modules needed for LPIPS/GAN training. |
| `stage1/train_stage1.py` | Stage 1 training entry point. |
| `stage1/reconstruct_stage1.py` | Reconstruction script for evaluating a Stage 1 checkpoint. |
| `stage1/visualize_stage1_latents.py` | Latent PCA visualization script. |
| `stage2/` | Generative modeling code for RePack-DiT. |
| `stage2/configs/` | Stage 2 RePack tokenizer and DiT YAML configs. |
| `stage2/scripts/` | Shell helpers for feature extraction, training, batch inference, and FID. |
| `stage2/tokenizer/` | Lightweight RePack tokenizer wrapper that reuses Stage 1 modules. |
| `stage2/train.py` | Stage 2 DiT training entry point. |
| `stage2/extract_features.py` | ImageNet RePack latent caching script. |
| `stage2/inference_batch.py` | Batch sampling script that saves both images and generated latents. |
| `stage3/` | Latent-guided refinement code. |
| `stage3/configs/` | Offline Refiner training YAML config. |
| `stage3/scripts/` | Shell helpers for offline data preparation, training, refinement, and FID. |
| `stage3/prepare_offline_data.py` | Builds Refiner training triplets from ImageNet and Stage-1 RePack. |
| `stage3/train_refiner.py` | Offline Refiner training entry point. |
| `stage3/refine_batch.py` | Batch refinement for Stage-2 generated images and latents. |

## 🚀 Quick Start

| Task | Command |
|:--|:--|
| Enter Stage 1 folder | `cd stage1` |
| Install dependencies | `pip install -r requirements.txt` |
| Train Stage 1 | `bash scripts/train_stage1.sh configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml` |
| Reconstruct images | See [`stage1/README.md`](stage1/README.md#reconstruct). |
| Visualize latents | See [`stage1/README.md`](stage1/README.md#visualize-latents). |
| Enter Stage 2 folder | `cd stage2` |
| Cache ImageNet latents | `bash scripts/run_extraction.sh /path/to/imagenet ./imagenet_features` |
| Train RePack-DiT | `bash scripts/run_train.sh configs/lightningdit_xl_repack_f16d32_dinov3.yaml` |
| Lower LR after 240k steps | `bash scripts/run_train.sh configs/lower_lr_lightningdit_xl_repack_f16d32_dinov3.yaml` |
| Batch inference + FID | `bash scripts/run_inference_batch.sh configs/lightningdit_xl_repack_f16d32_dinov3.yaml /path/to/ckpt.pt` |
| Enter Stage 3 folder | `cd stage3` |
| Apply Refiner | `bash scripts/run_refine_batch.sh /path/to/repack_refiner.ckpt /path/to/samples/latents /path/to/samples ./refined_results` |
| Refined FID | `bash scripts/run_fid.sh ./refined_results /path/to/VIRTUAL_imagenet256_labeled.npz` |

## 📦 Stage 1 Checkpoint

| Name | Backbone | Latent | Download |
|:--|:--|:--|:--|
| `repack-dinov3b-s16-ch32.ckpt` | DINOv3-B/16 | f16d32 | [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link) |

## ⚡ Stage 2 Checkpoint and Samples

| Name | Notes | Download |
|:--|:--|:--|
| `repack_dit_xl1_ep64.pt` | RePack-DiT-XL/1, 64 epochs, high LR to 240k steps and low LR to 320k steps. | [Google Drive](https://drive.google.com/file/d/14S7WXprhUVh-XlR7eaX5yRw-cms_s6BR/view?usp=sharing) |
| `lightningdit-xl-1-ckpt-0320000-euler-50-interval0.11-cfg15.00-shift0.00.zip` | 50-step inference results for preview and Stage-3 Refiner testing. FID: 1.87 without Refiner, 1.76 with Refiner. | [Google Drive](https://drive.google.com/file/d/1DhhvQcUjW-Hnzp2VrUVNGUyUpmGMafDq/view?usp=sharing) |

The best reported FID of **1.65** is obtained with 250-step inference, which is
slower than the 50-step preview setting.

## 🎨 Stage 3 Refiner Checkpoint

| Name | Notes | Download |
|:--|:--|:--|
| `repack_refiner.ckpt` | Latent-Guided Refiner trained offline on RePack reconstruction triplets. | [Google Drive](https://drive.google.com/file/d/1--eR0xFgJ9ivUMdb4xO4SzrzjGhzV3OD/view?usp=sharing) |

## 🚧 Note

This release is **not the camera-ready version yet**. We will keep the code and
the arXiv paper version synchronized as the project is updated.

Hope to see you at **ICML 2026**!

## 🙏 Acknowledgements

Our codebase mainly inherits from the VA-VAE / LightningDiT repository. We
thank the authors for their excellent work:
[hustvl/LightningDiT](https://github.com/hustvl/LightningDiT/tree/main).

| Project | Paper |
|:--|:--|
| VA-VAE / LightningDiT | **Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models** |

```bibtex
@inproceedings{yao2025vavae,
  title={Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models},
  author={Yao, Jingfeng and Yang, Bin and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## 📚 Citation

If you find this project useful, please cite:

```bibtex
@misc{dong2026repackrefineefficientdiffusion,
  title={RePack then Refine: Efficient Diffusion Transformer with Vision Foundation Model},
  author={Guanfang Dong and Luke Schultz and Negar Hassanpour and Chao Gao},
  year={2026},
  eprint={2512.12083},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.12083},
}
```
