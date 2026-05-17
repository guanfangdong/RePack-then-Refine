# RePack then Refine

This folder is the clean public release for **RePack then Refine**, a
three-stage framework for efficient image generation with compact VFM latents.

Raw Vision Foundation Model features are semantic-rich but high-dimensional.
RePack compresses frozen DINOv3-B/16 patch features into an f16d32 latent space,
trains a generative model on that packed space, and then uses a latent-guided
refiner to recover high-frequency details.

The paper draft is included as
[`4234_RePack_then_Refine_Effici.pdf`](4234_RePack_then_Refine_Effici.pdf).

## Release Layout

```text
public_release/
  4234_RePack_then_Refine_Effici.pdf
  stage1/
    configs/
    repack/
    scripts/
    taming/
    train_stage1.py
    reconstruct_stage1.py
    visualize_stage1_latents.py
    README.md
```

## Current Status

The public release is currently prepared through **Stage 1**.

- Stage 1 representation packing: ready
- Stage 1 checkpoint: available through Google Drive
- Stage 2 generative modeling: being cleaned
- Stage 3 latent-guided refinement: being cleaned

Start with [`stage1/README.md`](stage1/README.md).

## Stage 1 Checkpoint

- File: `repack-dinov3b-s16-ch32.ckpt`
- Download:
  [Google Drive](https://drive.google.com/file/d/1oDd1SRUjp8-7ncyI0Tc2HSirL06Vf1f-/view?usp=drive_link)

We plan to mirror the checkpoint on Hugging Face later. It is provided through
Google Drive first so the release can be used immediately.

## Acknowledgements

Our codebase mainly inherits from the VA-VAE / LightningDiT repository. We
thank the authors for their excellent work:
[hustvl/LightningDiT](https://github.com/hustvl/LightningDiT/tree/main).

Related paper:
**Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models**

```bibtex
@inproceedings{yao2025vavae,
  title={Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models},
  author={Yao, Jingfeng and Yang, Bin and Wang, Xinggang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Citation

This release is not the camera-ready version yet. We will keep the code and the
arXiv paper version synchronized as the project is updated. We hope to see you
at ICML 2026.

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
