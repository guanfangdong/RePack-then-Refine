"""Batch inference for RePack-DiT checkpoints.

This entrypoint saves both decoded images and generated RePack latents.
"""

import os, math, json, pickle, logging, argparse, yaml, torch, numpy as np
from time import time, strftime
from glob import glob
from copy import deepcopy
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torchvision

# local imports
from tokenizer import RePackTokenizer
from models.lightningdit import LightningDiT_models
from transport import create_transport, Sampler
from local_datasets.img_latent_dataset import ImgLatentDataset


# -------------------------------
# utils
# -------------------------------
def print_with_prefix(*messages):
    prefix = f"\033[34m[RePack-DiT Sampling {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = ' '.join(map(str, messages))
    print(f"{prefix}: {combined_message}")

def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# -------------------------------
# sampling core
# -------------------------------
def do_sample(
    train_config,
    accelerator,
    ckpt_path=None,
    cfg_scale=None,
    model=None,
    repack_tokenizer=None,
    demo_sample_mode=False,
    latent_mean=None,
    latent_std=None,
):
    """
    Run sampling for one checkpoint.
    """

    folder_name = f"{train_config['model']['model_type'].replace('/', '-')}-ckpt-{ckpt_path.split('/')[-1].split('.')[0]}-{train_config['sample']['sampling_method']}-{train_config['sample']['num_sampling_steps']}".lower()
    if cfg_scale is None:
        cfg_scale = train_config['sample']['cfg_scale']
    cfg_interval_start = train_config['sample']['cfg_interval_start'] if 'cfg_interval_start' in train_config['sample'] else 0
    timestep_shift = train_config['sample']['timestep_shift'] if 'timestep_shift' in train_config['sample'] else 0
    if cfg_scale > 1.0:
        folder_name += f"-interval{cfg_interval_start:.2f}"+f"-cfg{cfg_scale:.2f}"
        folder_name += f"-shift{timestep_shift:.2f}"

    if demo_sample_mode:
        cfg_interval_start = 0
        timestep_shift = 0
        cfg_scale = 9.0

    sample_folder_dir = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], folder_name)
    if accelerator.process_index == 0:
        if not demo_sample_mode:
            print_with_prefix('Sample_folder_dir=', sample_folder_dir)
        print_with_prefix('ckpt_path=', ckpt_path)
        print_with_prefix('cfg_scale=', cfg_scale)
        print_with_prefix('cfg_interval_start=', cfg_interval_start)
        print_with_prefix('timestep_shift=', timestep_shift)

    if not os.path.exists(sample_folder_dir):
        if accelerator.process_index == 0:
            os.makedirs(sample_folder_dir, exist_ok=True)
    else:
        png_files = [f for f in os.listdir(sample_folder_dir) if f.endswith('.png')]
        png_count = len(png_files)
        if png_count > train_config['sample']['fid_num']:
            if accelerator.process_index == 0:
                print_with_prefix(f"Found {png_count} PNG files in {sample_folder_dir}, skip sampling.")
            return sample_folder_dir

    torch.backends.cuda.matmul.allow_tf32 = True  # fast, small numerical diffs possible
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup accelerator / device / seed
    device = accelerator.device
    seed = train_config['train']['global_seed'] * accelerator.num_processes + accelerator.process_index
    torch.manual_seed(seed)
    print_with_prefix(f"Starting rank={accelerator.local_process_index}, seed={seed}, world_size={accelerator.num_processes}.")
    rank = accelerator.local_process_index

    # Latent size
    downsample_ratio = train_config['repack'].get('downsample_ratio', 16)
    latent_size = train_config['data']['image_size'] // downsample_ratio

    # Load model weights for this ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # Transport / sampler
    transport = create_transport(
        train_config['transport']['path_type'],
        train_config['transport']['prediction'],
        train_config['transport']['loss_weight'],
        train_config['transport']['train_eps'],
        train_config['transport']['sample_eps'],
        use_cosine_loss = train_config['transport']['use_cosine_loss'] if 'use_cosine_loss' in train_config['transport'] else False,
        use_lognorm = train_config['transport']['use_lognorm'] if 'use_lognorm' in train_config['transport'] else False,
    )
    sampler = Sampler(transport)
    mode = train_config['sample']['mode']
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=train_config['sample']['sampling_method'],
            num_steps=train_config['sample']['num_sampling_steps'],
            atol=train_config['sample']['atol'],
            rtol=train_config['sample']['rtol'],
            reverse=train_config['sample']['reverse'],
            timestep_shift=timestep_shift,
        )
    else:
        raise NotImplementedError(f"Sampling mode {mode} is not supported.")

    # RePack tokenizer / decoder.
    if repack_tokenizer is None:
        repack_tokenizer = RePackTokenizer(train_config['repack']['config_path'], load_encoder=False)
        if accelerator.process_index == 0:
            print_with_prefix('Loaded RePack tokenizer')

    using_cfg = cfg_scale > 1.0
    if using_cfg and accelerator.process_index == 0:
        print_with_prefix('Using cfg:', using_cfg)

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        if accelerator.process_index == 0 and not demo_sample_mode:
            print_with_prefix(f"Saving .png samples at {sample_folder_dir}")
    accelerator.wait_for_everyone()

    # How many samples per GPU / iterations
    n = train_config['sample']['per_proc_batch_size']
    global_batch_size = n * accelerator.num_processes
    num_samples = len([name for name in os.listdir(sample_folder_dir) if (os.path.isfile(os.path.join(sample_folder_dir, name)) and ".png" in name)])
    total_samples = int(math.ceil(train_config['sample']['fid_num'] / global_batch_size) * global_batch_size)
    if rank == 0 and accelerator.process_index == 0:
        print_with_prefix(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    done_iterations = int( int(num_samples // accelerator.num_processes) // n)
    pbar = range(iterations)
    if not demo_sample_mode:
        pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    # latent stats (cached preferred)
    latent_multiplier = train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215
    if latent_mean is None or latent_std is None:
        if accelerator.process_index == 0:
            print_with_prefix("Using latent normalization (computed here)")
        dataset = ImgLatentDataset(
            data_dir=train_config['data']['data_path'],
            latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
            latent_multiplier=latent_multiplier,
        )
        latent_mean, latent_std = dataset.get_latent_stats()

    # move to device
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    # demo mode
    if demo_sample_mode:
        if accelerator.process_index == 0:
            images = []
            for label in tqdm([975, 3, 207, 387, 388, 88, 979, 279], desc="Generating Demo Samples"):
                z = torch.randn(1, model.in_channels, latent_size, latent_size, device=device)
                y = torch.tensor([label], device=device)
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * 1, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=False, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
                samples = sample_fn(z, model_fn, **model_kwargs)[-1]
                samples = (samples * latent_std) / latent_multiplier + latent_mean
                samples = repack_tokenizer.decode_to_images(samples)
                images.append(samples)
            os.makedirs('demo_images', exist_ok=True)
            all_images = np.stack([img[0] for img in images])
            h, w = all_images.shape[1:3]
            grid = np.zeros((2 * h, 4 * w, 3), dtype=np.uint8)
            for idx, image in enumerate(all_images):
                i, j = divmod(idx, 4)
                grid[i*h:(i+1)*h, j*w:(j+1)*w] = image
            Image.fromarray(grid).save('demo_images/demo_samples.png')
            return None
    else:
        # normal sampling loop
        for i in pbar:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, train_config['data']['num_classes'], (n,), device=device)

            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=cfg_interval_start)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(z, model_fn, **model_kwargs)[-1]
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            latent_save_dir = os.path.join(sample_folder_dir, "latents")
            
            if accelerator.is_main_process:
                os.makedirs(latent_save_dir, exist_ok=True)
            
            accelerator.wait_for_everyone()

            # Save generated latents before denormalization/decoding for Stage-3 refinement.
            for i_lat, lat_tensor in enumerate(samples):
                index = i_lat * accelerator.num_processes + accelerator.process_index + total
                torch.save(lat_tensor.cpu(), f"{latent_save_dir}/{index:06d}.pt")

            samples = (samples * latent_std) / latent_multiplier + latent_mean
            samples = repack_tokenizer.decode_to_images(samples)

            for i_img, sample in enumerate(samples):
                index = i_img * accelerator.num_processes + accelerator.process_index + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += global_batch_size
            accelerator.wait_for_everyone()

    return sample_folder_dir


# -------------------------------
# main
# -------------------------------
if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lightningdit_xl_repack_f16d32_dinov3.yaml')
    parser.add_argument('--demo', action='store_true', default=False)
    # NEW: multi-ckpt options
    parser.add_argument('--ckpt_glob', type=str, default=None,
                        help="Glob pattern for multiple ckpts, e.g. '/path/checkpoints/*.pt'")
    parser.add_argument('--ckpts', type=str, default=None,
                        help="Comma-separated list of ckpt paths.")
    parser.add_argument('--no_fid', action='store_true', default=False,
                        help="Skip FID calculation for speed.")
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    # build ckpt list
    ckpt_list = []
    if args.ckpt_glob is not None:
        ckpt_list = sorted(glob(args.ckpt_glob))
    if args.ckpts is not None:
        ckpt_list.extend([p.strip() for p in args.ckpts.split(',') if p.strip()])

    if not ckpt_list:
        assert 'ckpt_path' in train_config, "ckpt_path must be specified in config or provide --ckpt_glob/--ckpts"
        ckpt_list = [train_config['ckpt_path']]

    if accelerator.process_index == 0:
        print_with_prefix(f'Found {len(ckpt_list)} checkpoints:')
        for p in ckpt_list:
            print_with_prefix('  -', p)

    # latent size
    latent_size = train_config['data']['image_size'] // train_config['repack'].get('downsample_ratio', 16)

    # build model once
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model']['use_swiglu'] if 'use_swiglu' in train_config['model'] else False,
        use_rope=train_config['model']['use_rope'] if 'use_rope' in train_config['model'] else False,
        use_rmsnorm=train_config['model']['use_rmsnorm'] if 'use_rmsnorm' in train_config['model'] else False,
        wo_shift=train_config['model']['wo_shift'] if 'wo_shift' in train_config['model'] else False,
        in_channels=train_config['model']['in_chans'] if 'in_chans' in train_config['model'] else 4,
        # learn_sigma=train_config['model']['learn_sigma'] if 'learn_sigma' in train_config['model'] else False,
    )

    # Build RePack once and reuse it for all checkpoints.
    repack_tokenizer = RePackTokenizer(train_config['repack']['config_path'], load_encoder=False)
    if accelerator.process_index == 0:
        print_with_prefix('Loaded RePack tokenizer (shared across ckpts)')

    # precompute latent stats once
    dataset = ImgLatentDataset(
        data_dir=train_config['data']['data_path'],
        latent_norm=train_config['data']['latent_norm'] if 'latent_norm' in train_config['data'] else False,
        latent_multiplier=train_config['data']['latent_multiplier'] if 'latent_multiplier' in train_config['data'] else 0.18215,
    )
    latent_mean, latent_std = dataset.get_latent_stats()

    # loop over ckpts
    for one_ckpt in ckpt_list:
        if accelerator.process_index == 0:
            print_with_prefix('Using ckpt:', one_ckpt)

        sample_folder_dir = do_sample(
            train_config,
            accelerator,
            ckpt_path=one_ckpt,
            model=model,
            repack_tokenizer=repack_tokenizer,
            demo_sample_mode=args.demo,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )

        if args.demo:
            # demo mode only produces a grid image per ckpt; no FID
            continue

        # optional FID per ckpt
        if (not args.no_fid) and accelerator.process_index == 0:
            from tools.calculate_fid import calculate_fid_given_paths
            print_with_prefix('Calculating FID with {} number of samples'.format(train_config['sample']['fid_num']))
            assert 'fid_reference_file' in train_config['data'], "fid_reference_file must be specified in config"
            fid_reference_file = train_config['data']['fid_reference_file']
            fid = calculate_fid_given_paths(
                [fid_reference_file, sample_folder_dir],
                batch_size=50,
                dims=2048,
                device='cuda',
                num_workers=8,
                sp_len = train_config['sample']['fid_num']
            )
            print_with_prefix(f'fid (ckpt={os.path.basename(one_ckpt)}) = {fid}')
            result = {
                "checkpoint": one_ckpt,
                "sample_dir": sample_folder_dir,
                "fid_reference_file": fid_reference_file,
                "fid_num": train_config['sample']['fid_num'],
                "fid": float(fid),
            }
            result_json = os.path.join(sample_folder_dir, "fid_results.json")
            result_jsonl = os.path.join(train_config['train']['output_dir'], train_config['train']['exp_name'], "fid_results.jsonl")
            with open(result_json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            with open(result_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            print_with_prefix(f"Saved FID results to {result_json}")
