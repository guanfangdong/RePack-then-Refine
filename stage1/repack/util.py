import importlib

import numpy as np
import torch
from PIL import Image


def get_obj_from_str(path: str, reload: bool = False):
    module, cls = path.rsplit(".", 1)
    module_obj = importlib.import_module(module)
    if reload:
        importlib.reload(module_obj)
    return getattr(module_obj, cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Config block must contain a 'target' field.")
    params = config.get("params", {})
    return get_obj_from_str(config["target"])(**params)


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def tensor_to_pil(x: torch.Tensor) -> Image.Image:
    if x.dim() == 4:
        x = x[0]
    x = (x.detach().float().cpu().clamp(-1, 1) + 1.0) / 2.0
    x = (x * 255.0).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x, mode="RGB")


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

