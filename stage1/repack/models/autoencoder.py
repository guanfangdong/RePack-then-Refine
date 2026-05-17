import torch
import pytorch_lightning as pl

from repack.losses import LPIPSWithDiscriminator
from repack.modules.distributions import DiagonalGaussianDistribution
from repack.modules.repack_blocks import Decoder, DinoV3RePackEncoder
from repack.util import instantiate_from_config


class RePackAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=None,
        image_key="image",
        monitor=None,
        sample_posterior=True,
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = DinoV3RePackEncoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.embed_dim = embed_dim
        self.sample_posterior = sample_posterior
        self.automatic_optimization = False

        if not ddconfig.get("double_z", True):
            raise ValueError("RePackAutoencoder expects ddconfig.double_z=true.")
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [])

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not any(key.startswith(prefix) for prefix in ignore_keys)
        }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {path}")
        print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    def get_input(self, batch, key=None):
        key = key or self.image_key
        x = batch[key]
        if x.dim() == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    def encode(self, x):
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

    def forward(self, x, sample_posterior=None):
        z, posterior = self.encode_to_latent(x, sample_posterior)
        recon = self.decode(z)
        return recon, posterior, z

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior, z = self(inputs)
        ae_opt, disc_opt = self.optimizers()

        ae_loss, ae_logs = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
            z=z,
        )
        ae_opt.zero_grad(set_to_none=True)
        self.manual_backward(ae_loss)
        ae_opt.step()

        disc_loss, disc_logs = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        disc_opt.zero_grad(set_to_none=True)
        self.manual_backward(disc_loss)
        disc_opt.step()

        self.log("aeloss", ae_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("discloss", disc_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(ae_logs, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict(disc_logs, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior, z = self(inputs, sample_posterior=False)
        ae_loss, ae_logs = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
            z=z,
        )
        disc_loss, disc_logs = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        self.log("val/rec_loss", ae_logs.get("val/rec_loss", ae_loss), prog_bar=True, sync_dist=True)
        self.log_dict(ae_logs, prog_bar=False, logger=True, sync_dist=True)
        self.log_dict(disc_logs, prog_bar=False, logger=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.learning_rate
        ae_params = [
            p
            for p in (
                list(self.encoder.parameters())
                + list(self.decoder.parameters())
                + list(self.quant_conv.parameters())
                + list(self.post_quant_conv.parameters())
            )
            if p.requires_grad
        ]
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))

        if hasattr(self.loss, "losses") and "lpips_disc_loss" in self.loss.losses:
            disc = self.loss.losses["lpips_disc_loss"].discriminator
        elif isinstance(self.loss, LPIPSWithDiscriminator):
            disc = self.loss.discriminator
        else:
            raise RuntimeError("Stage1 training requires an LPIPSWithDiscriminator loss.")
        opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, max_images=4, **kwargs):
        inputs = self.get_input(batch)[:max_images].to(self.device)
        logs = {"inputs": inputs}
        if not only_inputs:
            reconstructions, posterior, _ = self(inputs, sample_posterior=False)
            logs["reconstructions"] = reconstructions
            logs["samples"] = self.decode(torch.randn_like(posterior.mode()))
        return logs
