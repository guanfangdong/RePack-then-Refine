import torch
import torch.nn as nn

from repack.modules.distributions import DiagonalGaussianDistribution
from repack.modules.repack_blocks import Decoder, DinoV3RePackEncoder


class RePackAutoencoderForInference(nn.Module):
    """Stage-2 wrapper that reuses the Stage-1 RePack encoder and decoder."""

    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        sample_posterior=True,
        load_encoder=True,
    ):
        super().__init__()
        if not ddconfig.get("double_z", True):
            raise ValueError("RePack expects ddconfig.double_z=true.")

        self.encoder = DinoV3RePackEncoder(**ddconfig) if load_encoder else None
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.sample_posterior = sample_posterior
        self.load_encoder = load_encoder

        if ckpt_path:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded RePack checkpoint from {path}")
        print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    def encode(self, x):
        if self.encoder is None:
            raise RuntimeError("This RePack tokenizer was initialized without the encoder.")
        moments = self.quant_conv(self.encoder(x))
        return DiagonalGaussianDistribution(moments)

    def decode(self, z):
        return self.decoder(self.post_quant_conv(z))

    def encode_to_latent(self, x, sample_posterior=None):
        posterior = self.encode(x)
        if sample_posterior is None:
            sample_posterior = self.sample_posterior
        z = posterior.sample() if sample_posterior else posterior.mode()
        return z, posterior
