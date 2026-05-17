import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from repack.util import instantiate_from_config


def resolve_path(path, config_path):
    if not path:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    config_dir = Path(config_path).resolve().parent
    candidate = (config_dir / path).resolve()
    return candidate if candidate.exists() else path.resolve()


def parse_trainer_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


class ImageLogger(Callback):
    def __init__(self, batch_frequency=1000, max_images=4):
        super().__init__()
        self.batch_frequency = batch_frequency
        self.max_images = max_images

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if trainer.global_step == 0 or trainer.global_step % self.batch_frequency != 0:
            return
        logger = trainer.logger
        if logger is None or not hasattr(logger, "experiment"):
            return
        images = pl_module.log_images(batch, max_images=self.max_images)
        for name, value in images.items():
            value = value[: self.max_images]
            grid = torchvision.utils.make_grid(
                value,
                nrow=min(self.max_images, value.shape[0]),
                normalize=True,
                value_range=(-1, 1),
            )
            logger.experiment.add_image(name, grid, trainer.global_step)


def load_initial_weights(model, config, config_path):
    init_weight = config.get("init_weight", None)
    if init_weight is None:
        return
    init_weight = resolve_path(init_weight, config_path)
    if not init_weight.exists():
        print(f"Initial weight not found, training from scratch: {init_weight}")
        return

    checkpoint = torch.load(init_weight, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded initial weight: {init_weight}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def build_trainer(config, args, run_name):
    lightning_cfg = config.get("lightning", OmegaConf.create())
    trainer_cfg = OmegaConf.to_container(lightning_cfg.get("trainer", {}), resolve=True)

    overrides = {
        "accelerator": parse_trainer_value(args.accelerator),
        "devices": parse_trainer_value(args.devices),
        "num_nodes": args.num_nodes,
        "strategy": parse_trainer_value(args.strategy),
        "max_epochs": args.max_epochs,
        "precision": parse_trainer_value(args.precision),
    }
    for key, value in overrides.items():
        if value is not None:
            trainer_cfg[key] = value

    trainer_cfg.setdefault("accelerator", "gpu")
    trainer_cfg.setdefault("devices", 1)
    trainer_cfg.setdefault("num_nodes", 1)
    trainer_cfg.setdefault("max_epochs", 99)
    trainer_cfg.setdefault("precision", 32)

    logdir = Path(args.logdir) / run_name
    ckptdir = logdir / "checkpoints"
    logger = TensorBoardLogger(save_dir=str(logdir), name="tensorboard")

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=str(ckptdir / "trainstep_checkpoints"),
            filename="{epoch:06}-{step:09}",
            every_n_train_steps=args.save_every_n_steps,
            save_top_k=-1,
            save_weights_only=True,
        ),
        ModelCheckpoint(
            dirpath=str(ckptdir),
            filename="last",
            save_last=True,
            save_top_k=0,
        ),
    ]
    if not args.no_image_log:
        callbacks.append(ImageLogger(args.image_log_every_n_steps, args.image_log_max_images))

    return pl.Trainer(**trainer_cfg, logger=logger, callbacks=callbacks), logdir


def parse_args():
    parser = argparse.ArgumentParser(description="Train RePack stage1.")
    parser.add_argument("-c", "--config", required=True, help="Path to the stage1 YAML config.")
    parser.add_argument("--name", default=None, help="Run name. Defaults to the config filename.")
    parser.add_argument("--logdir", default="logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-every-n-steps", type=int, default=20000)
    parser.add_argument("--image-log-every-n-steps", type=int, default=1000)
    parser.add_argument("--image-log-max-images", type=int, default=4)
    parser.add_argument("--no-image-log", action="store_true")
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.num_workers is not None:
        config.data.params.num_workers = args.num_workers

    run_name = args.name or Path(args.config).stem
    model = instantiate_from_config(config.model)
    load_initial_weights(model, config, args.config)
    model.learning_rate = config.model.base_learning_rate

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    for split, dataset in data.datasets.items():
        print(f"{split}: {dataset.__class__.__name__}, {len(dataset)} images")

    trainer, logdir = build_trainer(config, args, run_name)
    print(f"Logging to: {logdir}")
    trainer.fit(model, datamodule=data, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
