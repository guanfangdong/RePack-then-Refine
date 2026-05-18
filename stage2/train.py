"""
Accelerate-native training loop for LightningDiT on RePack latents.
- Fully controlled by `accelerate` (no manual DDP / torch.distributed calls)
- Works with bf16/fp16/no mixed precision via Accelerate config or CLI
- Safe checkpointing with `accelerator.save` and `accelerator.unwrap_model`
- Optional validation pass

Usage (example):
  accelerate launch --config_file accelerate_config.yaml train_accelerate.py \
      --config configs/your_config.yaml
"""

from __future__ import annotations

import os
import math
import json
import yaml
import time
import argparse
import logging
from glob import glob
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator
from accelerate.utils import set_seed

# project imports
from models.lightningdit import LightningDiT_models
from transport import create_transport
from local_datasets.img_latent_dataset import ImgLatentDataset

import torch._dynamo

# -------------------------------
# Utilities
# -------------------------------

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_logger(logging_dir: str, is_main: bool) -> logging.Logger:
    os.makedirs(logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format='[\x1b[34m%(asctime)s\x1b[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logging_dir, "log.txt")),
            ],
        )
    else:
        logger.addHandler(logging.NullHandler())
    return logger


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, src_model: torch.nn.Module, decay: float = 0.9999):
    """EMA: ema = decay * ema + (1-decay) * src"""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(src_model.named_parameters())
    for name, param in model_params.items():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def load_weights_with_shape_check(model: torch.nn.Module, checkpoint: dict, log_fn=print):
    model_state_dict = model.state_dict()
    for name, param in checkpoint['model'].items():
        if name in model_state_dict:
            if param.shape == model_state_dict[name].shape:
                model_state_dict[name].copy_(param)
            elif name == 'x_embedder.proj.weight':
                # special case for proj weights when in_chans differs
                weight = torch.zeros_like(model_state_dict[name])
                # keep first 16 channels (for f16)
                ch = min(weight.shape[1], param.shape[1])
                weight[:, :ch] = param[:, :ch]
                model_state_dict[name] = weight
            else:
                log_fn(f"[load] skip '{name}' shape {tuple(param.shape)} -> {tuple(model_state_dict[name].shape)}")
        else:
            log_fn(f"[load] missing in model: '{name}'")
    model.load_state_dict(model_state_dict, strict=False)
    return model


@torch.no_grad()
def evaluate(model: torch.nn.Module, valid_loader: DataLoader, device: torch.device, transport, clamp_range=(0.0, 1.0)) -> float:
    model.eval()
    losses = []
    for x, y in valid_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        loss_dict = transport.training_losses(model, x, dict(y=y))
        if 'cos_loss' in loss_dict:
            mse_loss = loss_dict["loss"].mean()
            total = (loss_dict["cos_loss"].mean() + mse_loss).detach()
        else:
            total = loss_dict["loss"].mean().detach()
        losses.append(total)
    if len(losses) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


# -------------------------------
# Training
# -------------------------------

def do_train(train_config: dict, accelerator: Accelerator):
    device = accelerator.device

    # seeding
    seed = int(train_config['train'].get('global_seed', 0))
    set_seed(seed, device_specific=True)

    # experiment dirs (deterministic by exp_name)
    out_dir = train_config['train']['output_dir']
    exp_name = train_config['train'].get('exp_name') or 'exp'
    experiment_dir = os.path.join(out_dir, exp_name)
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')

    # logging (main only)
    logger = create_logger(experiment_dir, accelerator.is_main_process)
    writer = None
    if accelerator.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        tb_dir = os.path.join('tensorboard_logs', exp_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text('training configs', json.dumps(train_config, indent=2))
        logger.info(f"Experiment dir: {experiment_dir}")

    # ---------------- models ----------------
    # RePack downsample ratio -> latent size.
    downsample_ratio = int(train_config['repack'].get('downsample_ratio', 16))
    assert train_config['data']['image_size'] % downsample_ratio == 0
    latent_size = train_config['data']['image_size'] // downsample_ratio

    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', 4),
        use_checkpoint=train_config['model'].get('use_checkpoint', False),
    )


    torch.backends.cudnn.benchmark = True

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # optional pretrained init
    if 'weight_init' in train_config.get('train', {}):
        ckpt_path = train_config['train']['weight_init']
        if accelerator.is_main_process:
            logger.info(f"Loading pretrained weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # strip 'module.' if present
        ckpt['model'] = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
        log_fn = (lambda *a, **k: logger.info(*a, **k)) if accelerator.is_main_process else (lambda *a, **k: None)
        model = load_weights_with_shape_check(model, ckpt, log_fn)
        ema = load_weights_with_shape_check(ema, ckpt, log_fn)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['optimizer']['lr'],
        weight_decay=0.0,
        betas=(0.9, train_config['optimizer']['beta2']),
    )

    # ---------------- resume (optional) ----------------
    train_steps = 0
    if train_config['train'].get('resume', False):
        # Prefer an explicit resume path; otherwise continue from the latest local checkpoint.
        manual_path = train_config['train'].get('resume_from', None)
        
        if manual_path and os.path.exists(manual_path):
            latest = manual_path
        else:
            ckpts = sorted(glob(os.path.join(checkpoint_dir, '*.pt')))
            latest = ckpts[-1] if ckpts else None

        if latest:
            if accelerator.is_main_process:
                logger.info(f"Loading specific checkpoint: {latest}")
            
            payload = torch.load(latest, map_location='cpu')
            
            accelerator.unwrap_model(model).load_state_dict(payload['model'])
            ema.load_state_dict(payload['ema'])
            opt.load_state_dict(payload['opt'])
            
            # Keep the LR from the active config, which enables the strict 42-epoch LR drop.
            new_lr = train_config['optimizer']['lr']
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr
            
            # Restore the training step.
            try:
                train_steps = int(os.path.basename(latest).split('.')[0])
            except:
                train_steps = payload.get('global_step', 0)

            if accelerator.is_main_process:
                logger.info(f"Successfully resumed from step {train_steps} with LR={new_lr}")
            
            accelerator.wait_for_everyone()
        else:
            if accelerator.is_main_process:
                logger.info("No checkpoint found. Starting from scratch.")

    # --- after optional pretrained init, before optimizer/prepare ---
    # model.y_embedder = torch._dynamo.disable()(model.y_embedder)
    # use_compile = bool(train_config['train'].get('compile', False))
    # if use_compile:
    #     if not hasattr(torch, "compile"):
    #         if accelerator.is_main_process:
    #             logger.warning("torch.compile is unavailable. Continuing without compilation.")
    #     else:
    #         backend = train_config['train'].get('compile_backend', 'inductor')
    #         mode = train_config['train'].get('compile_mode', 'max-autotune')
    #         fullgraph = bool(train_config['train'].get('compile_fullgraph', False))
    #         dynamic = bool(train_config['train'].get('compile_dynamic', True))

    #         # Compile the trainable model only; keep EMA outside graph capture.
    #         if accelerator.is_main_process:
    #             logger.info(f"Compiling model with torch.compile(backend='{backend}', mode='{mode}', "
    #                         f"fullgraph={fullgraph}, dynamic={dynamic})")
    #         model = torch.compile(model,
    #                               backend=backend,
    #                               mode=mode,
    #                               fullgraph=fullgraph,
    #                               dynamic=dynamic)


    # transport & optimizer
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss=train_config['transport'].get('use_cosine_loss', False),
        use_lognorm=train_config['transport'].get('use_lognorm', False),
    )


    # ---------------- data ----------------
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data'].get('latent_norm', False),
        latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215),
    )

    # per-device batch size considering gradient accumulation
    grad_accum = int(train_config['train'].get('gradient_accumulation_steps',
                                            getattr(accelerator, "gradient_accumulation_steps", 1)))
    per_device_bs = max(1, int(round(
        train_config['train']['global_batch_size'] / (accelerator.num_processes * grad_accum)
    )))
    global_bs = per_device_bs * accelerator.num_processes * grad_accum  # effective global batch

    if accelerator.is_main_process:
        logger.info(f"Per-device batch: {per_device_bs} | Accum: {grad_accum} | World size: {accelerator.num_processes}")
        logger.info(f"Effective global batch size: {global_bs} (target={train_config['train']['global_batch_size']})")


    loader = DataLoader(
        dataset,
        batch_size=per_device_bs,
        shuffle=True,
        num_workers=train_config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = None
    if 'valid_path' in train_config['data']:
        valid_dataset = ImgLatentDataset(
            data_dir=train_config['data']['valid_path'],
            latent_norm=train_config['data'].get('latent_norm', False),
            latent_multiplier=train_config['data'].get('latent_multiplier', 0.18215),
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=per_device_bs,
            shuffle=False,
            num_workers=train_config['data']['num_workers'],
            pin_memory=True,
            drop_last=True,
        )

    # -------------- prepare with accelerate --------------
    # IMPORTANT: do NOT wrap DDP manually; let Accelerate handle it.
    model, opt, loader, valid_loader = accelerator.prepare(model, opt, loader, valid_loader)

    # init EMA from (possibly sharded) model
    update_ema(ema, accelerator.unwrap_model(model), decay=0.0)
    model.train()
    ema.eval()

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in accelerator.unwrap_model(model).parameters()) / 1e6
        logger.info(f"LightningDiT Parameters: {n_params:.2f}M")
        logger.info(f"Optimizer: AdamW, lr={train_config['optimizer']['lr']}, beta2={train_config['optimizer']['beta2']}")
        logger.info(f"Dataset: {len(dataset):,} samples from {train_config['data']['data_path']}")
        logger.info(f"Batch size per device: {per_device_bs}, global: {global_bs}")
        logger.info(f"Use lognorm: {train_config['transport'].get('use_lognorm', False)} | Cosine loss: {train_config['transport'].get('use_cosine_loss', False)}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")


    # ---------------- training loop ----------------
    log_every = int(train_config['train']['log_every'])
    ckpt_every = int(train_config['train']['ckpt_every'])
    max_steps = int(train_config['train']['max_steps'])

    running_loss = 0.0
    micro_steps = 0
    start_time = time.time()

    while train_steps < max_steps:
        for x, y in loader:
            # move to device; dtype handled by autocast via Accelerator
            if accelerator.mixed_precision == 'no':
                x = x.to(device, dtype=torch.float32, non_blocking=True)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with accelerator.accumulate(model):
                loss_dict = transport.training_losses(model, x, dict(y=y))
                if 'cos_loss' in loss_dict:
                    mse_loss = loss_dict["loss"].mean()
                    loss = loss_dict["cos_loss"].mean() + mse_loss
                    tracked = mse_loss
                else:
                    loss = loss_dict["loss"].mean()
                    tracked = loss

                accelerator.backward(loss)

                # Clip and update only when gradient accumulation reaches a sync step.
                if 'max_grad_norm' in train_config['optimizer'] and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_config['optimizer']['max_grad_norm'])

                if accelerator.sync_gradients:
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    update_ema(ema, accelerator.unwrap_model(model))
                    train_steps += 1

            running_loss += tracked.item()
            micro_steps += 1

            # Log after optimizer updates, not after every micro step.
            if accelerator.sync_gradients and (train_steps % log_every == 0):
                avg_loss = torch.tensor(running_loss / max(1, micro_steps), device=device)
                avg_loss = accelerator.reduce(avg_loss, reduction='mean')

                elapsed = time.time() - start_time
                updates_per_sec = log_every / max(1e-6, elapsed)

                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) loss={avg_loss.item():.4f} | {updates_per_sec:.2f} updates/s")
                    if writer:
                        writer.add_scalar('Loss/train', avg_loss.item(), train_steps)

                running_loss = 0.0
                micro_steps = 0
                start_time = time.time()

            # Save checkpoints and run validation after optimizer updates.
            if accelerator.sync_gradients and (train_steps % ckpt_every == 0 or train_steps >= max_steps):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    to_save = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'ema': ema.state_dict(),
                        'opt': opt.state_dict(),
                        'config': train_config,
                        'global_step': train_steps,
                    }
                    ckpt_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
                    accelerator.save(to_save, ckpt_path)
                    logger.info(f"[ckpt] saved: {ckpt_path}")

                if valid_loader is not None:
                    val = evaluate(model, valid_loader, device, transport)
                    val = accelerator.reduce(val, reduction='mean')
                    if accelerator.is_main_process:
                        logger.info(f"[valid] step={train_steps} loss={val.item():.4f}")
                        if writer:
                            writer.add_scalar('Loss/validation', val.item(), train_steps)
                    model.train()

            if train_steps >= max_steps:
                break

        # end for loader
    # end while

    if accelerator.is_main_process:
        logger.info("Training complete.")
        if writer:
            writer.flush(); writer.close()


# -------------------------------
# main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to training YAML config')
    args = parser.parse_args()

    accelerator = Accelerator()  # precision & dist controlled by accelerate config/CLI
    cfg = load_config(args.config)
    do_train(cfg, accelerator)
