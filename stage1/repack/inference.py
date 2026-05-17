import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

from repack.data.txt_image import center_crop_arr
from repack.util import instantiate_from_config, pil_to_tensor, tensor_to_pil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_stage1_model(config_path, ckpt_path=None, device=None):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    if ckpt_path is None:
        ckpt_path = config.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("A checkpoint is required. Pass --ckpt or set ckpt_path in the config.")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    return model, config, torch.device(device)


def read_image_paths(input_list=None, input_dir=None, limit=None):
    paths = []
    if input_list:
        with open(input_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    paths.append(line)
    if input_dir:
        root = Path(input_dir)
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(path))
    paths = [p for p in paths if os.path.isfile(p)]
    return paths[:limit] if limit is not None else paths


def load_image_tensor(path, resolution):
    image = Image.open(path).convert("RGB")
    image = center_crop_arr(image, resolution)
    return pil_to_tensor(image), image


def image_stem(path, index):
    stem = Path(path).stem
    return f"{index:06d}_{stem}"


@torch.no_grad()
def reconstruct_batch(model, images, sample=False):
    recon, posterior, z = model(images, sample_posterior=sample)
    return recon, posterior, z


__all__ = [
    "image_stem",
    "load_image_tensor",
    "load_stage1_model",
    "read_image_paths",
    "reconstruct_batch",
    "tensor_to_pil",
]
