# ⚡ RePack then Refine

**Official implementation of the ICML 2026 paper _"RePack then Refine"_.**
A highly efficient DiT framework achieving **1.65 FID on ImageNet-1K in only
64 epochs** by leveraging compressed Vision Foundation Model (VFM) features and
latent-guided refinement.

> Current release status: **Stage 1 is ready. Stage 2 and Stage 3 are being cleaned for release.**

| Paper | Code | Checkpoint | Status |
|:--|:--|:--|:--|
| [arXiv](https://arxiv.org/abs/2512.12083) | [`stage1`](stage1) | [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link) | 🚧 Not camera-ready yet |

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
| 2 | **Generative Modeling** | Trains a LightningDiT-style diffusion transformer on the packed latent space. | 🚧 Cleaning |
| 3 | **Latent-Guided Refinement** | Refines decoded base images using upsampled packed latents as structural guidance. | 🚧 Cleaning |

Raw VFM features are powerful but high-dimensional and redundant. RePack first
filters the redundancy by learning a compact latent manifold. DiT then learns
generation in this compact space. Finally, the Refiner restores local textures
and high-frequency details that may be weakened by compression.

## 📌 Current Release

| Item | Path / Link | Notes |
|:--|:--|:--|
| Paper draft | [`4234_RePack_then_Refine_Effici.pdf`](4234_RePack_then_Refine_Effici.pdf) | Draft version included in the release. |
| Stage 1 code | [`stage1`](stage1) | Training, reconstruction, and latent visualization. |
| Stage 1 README | [`stage1/README.md`](stage1/README.md) | Start here for usage instructions. |
| Stage 1 checkpoint | [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link) | `repack-dinov3b-s16-ch32.ckpt` |

The checkpoint will also be mirrored on Hugging Face later. For now, we provide
the Google Drive link first so users can download it quickly.

## 🗂️ Repository Layout

The uploaded repository is organized directly by stages. Stage 1 is available
now; Stage 2 and Stage 3 will be added as they are cleaned for release.

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
| `stage2/` | Generative modeling code. Coming soon. |
| `stage3/` | Latent-guided refinement code. Coming soon. |

## 🚀 Quick Start

| Task | Command |
|:--|:--|
| Enter Stage 1 folder | `cd stage1` |
| Install dependencies | `pip install -r requirements.txt` |
| Train Stage 1 | `bash scripts/train_stage1.sh configs/stage1_repack_dinov3b_f16d32_ffl_watson.yaml` |
| Reconstruct images | See [`stage1/README.md`](stage1/README.md#reconstruct). |
| Visualize latents | See [`stage1/README.md`](stage1/README.md#visualize-latents). |

## 📦 Stage 1 Checkpoint

| Name | Backbone | Latent | Download |
|:--|:--|:--|:--|
| `repack-dinov3b-s16-ch32.ckpt` | DINOv3-B/16 | f16d32 | [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link) |

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
