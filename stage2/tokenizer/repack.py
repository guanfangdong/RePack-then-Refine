import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision import transforms


_RELEASE_ROOT = Path(__file__).resolve().parents[2]
_STAGE1_ROOT = _RELEASE_ROOT / "stage1"
if _STAGE1_ROOT.is_dir() and str(_STAGE1_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE1_ROOT))

from repack.data.txt_image import center_crop_arr  # noqa: E402
from tokenizer.repack_autoencoder import RePackAutoencoderForInference  # noqa: E402


class RePackTokenizer:
    """Encode images into compact RePack latents and decode latents to images."""

    def __init__(self, config, img_size=256, sample_posterior=True, load_encoder=True):
        self.config_path = Path(config)
        self.config = OmegaConf.load(self.config_path)
        self.img_size = img_size
        self.sample_posterior = sample_posterior
        self.load_encoder = load_encoder
        self.device = torch.device("cuda")
        self.model = self._build_model().to(self.device).eval()

    def _build_model(self):
        params = self.config.model.params
        ddconfig = OmegaConf.to_container(params.ddconfig, resolve=True)
        ckpt_path = self.config.get("ckpt_path")
        if not ckpt_path:
            raise ValueError(f"`ckpt_path` is required in {self.config_path}")

        return RePackAutoencoderForInference(
            ddconfig=ddconfig,
            embed_dim=params.embed_dim,
            ckpt_path=ckpt_path,
            sample_posterior=self.sample_posterior,
            load_encoder=self.load_encoder,
        )

    def img_transform(self, p_hflip=0.0, img_size=None):
        img_size = img_size or self.img_size
        return transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
                transforms.RandomHorizontalFlip(p=p_hflip),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    inplace=True,
                ),
            ]
        )

    @torch.no_grad()
    def encode_images(self, images):
        images = images.to(self.device, non_blocking=True)
        z, _ = self.model.encode_to_latent(images)
        return z

    @torch.no_grad()
    def decode_to_images(self, z):
        images = self.model.decode(z.to(self.device, non_blocking=True))
        images = torch.clamp(127.5 * images + 128.0, 0, 255)
        images = images.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return images
