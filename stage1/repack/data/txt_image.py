import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class TxtImageDataset(Dataset):
    """Image-list dataset for RePack stage-1 training."""

    def __init__(
        self,
        txt_file: str,
        resolution: int = 256,
        random_flip: bool = True,
        to_rgb: bool = True,
        skip_nonexistent: bool = True,
    ):
        if not os.path.isfile(txt_file):
            raise FileNotFoundError(f"txt_file not found: {txt_file}")

        self.paths = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                path = line.strip()
                if not path or path.startswith("#"):
                    continue
                if skip_nonexistent and not os.path.isfile(path):
                    continue
                self.paths.append(path)

        if not self.paths:
            raise RuntimeError(f"No valid image paths found in {txt_file}")

        self.transform = T.Compose(
            [
                T.Lambda(lambda img: center_crop_arr(img, resolution)),
                T.RandomHorizontalFlip(p=0.5 if random_flip else 0.0),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.to_rgb = to_rgb

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            with Image.open(path) as img:
                if self.to_rgb:
                    img = img.convert("RGB")
                image = self.transform(img).permute(1, 2, 0)
            return {"image": image, "path": path}
        except Exception:
            return self.__getitem__((idx + 1) % len(self.paths))

