import argparse
import glob
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from networks import PatchDiscriminator, RefineUNet


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class OfflineRefineDataset(Dataset):
    """Reads offline refiner triplets: latent, base reconstruction, ground truth."""

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.latent_paths = sorted((self.data_root / "latent").glob("*.pt"))
        if not self.latent_paths:
            raise FileNotFoundError(f"No .pt files found in {self.data_root / 'latent'}")

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        latent_path = self.latent_paths[idx]
        base_name = latent_path.stem
        z = load_tensor(latent_path).float()
        x_rec = self.transform(Image.open(self.data_root / "rec" / f"{base_name}.png").convert("RGB"))
        x_gt = self.transform(Image.open(self.data_root / "gt" / f"{base_name}.png").convert("RGB"))
        return {"z": z, "x_rec": x_rec, "gt": x_gt}


class OfflineDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size=8, num_workers=16):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = OfflineRefineDataset(self.data_root)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )


class RePackRefinerTask(pl.LightningModule):
    def __init__(
        self,
        z_dim=32,
        base_channels=128,
        gan_weight=0.5,
        lpips_weight=1.0,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.unet = RefineUNet(in_channels=3 + z_dim, base_channels=base_channels)
        self.discriminator = PatchDiscriminator(in_channels=3, base_channels=base_channels)
        self.perceptual_loss = self._build_lpips() if lpips_weight > 0 else None

    def _build_lpips(self):
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("Stage 3 training with lpips_weight > 0 requires `pip install lpips`.") from exc

        loss = lpips.LPIPS(net="vgg").eval()
        for param in loss.parameters():
            param.requires_grad = False
        return loss

    def forward(self, x_rec, z):
        return self.unet(x_rec, z)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        z = batch["z"]
        x_rec = batch["x_rec"]
        x_gt = batch["gt"]

        x_refined = self(x_rec, z)
        loss_l1 = F.l1_loss(x_refined, x_gt)
        loss_lpips = torch.zeros_like(loss_l1)
        if self.perceptual_loss is not None:
            loss_lpips = self.perceptual_loss(x_refined, x_gt).mean()
        logits_fake = self.discriminator(x_refined)
        loss_g_gan = -torch.mean(logits_fake)
        total_g_loss = loss_l1 + self.hparams.lpips_weight * loss_lpips + self.hparams.gan_weight * loss_g_gan

        opt_g.zero_grad(set_to_none=True)
        self.manual_backward(total_g_loss)
        opt_g.step()

        logits_real = self.discriminator(x_gt.detach())
        logits_fake_d = self.discriminator(x_refined.detach())
        loss_d_real = torch.mean(F.relu(1.0 - logits_real))
        loss_d_fake = torch.mean(F.relu(1.0 + logits_fake_d))
        total_d_loss = 0.5 * (loss_d_real + loss_d_fake)

        opt_d.zero_grad(set_to_none=True)
        self.manual_backward(total_d_loss)
        opt_d.step()

        self.log_dict(
            {
                "g/l1": loss_l1,
                "g/lpips": loss_lpips,
                "g/gan": loss_g_gan,
                "g/total": total_g_loss,
                "d/loss": total_d_loss,
            },
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.unet.parameters(), lr=lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        return [opt_g, opt_d]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/refiner_offline.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    config = OmegaConf.load(args.config)
    data = OfflineDataModule(**config.data.params)
    model = RePackRefinerTask(**config.model.params)

    checkpoint_cfg = config.checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_cfg.dirpath,
        filename=checkpoint_cfg.get("filename", "{epoch:03d}-{step:08d}"),
        save_top_k=checkpoint_cfg.get("save_top_k", -1),
        every_n_train_steps=checkpoint_cfg.get("every_n_train_steps", 5000),
    )

    logger = TensorBoardLogger(save_dir=config.logging.logdir, name=config.logging.name)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.trainer.get("devices", 1),
        strategy=config.trainer.get("strategy", "auto"),
        max_epochs=config.trainer.get("max_epochs", 100),
        precision=config.trainer.get("precision", 32),
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=config.trainer.get("log_every_n_steps", 50),
    )
    trainer.fit(model, data, ckpt_path=args.resume or None)


if __name__ == "__main__":
    main()
