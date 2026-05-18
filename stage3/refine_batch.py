import argparse
import glob
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from networks import RefineUNet


def load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class RefineDataset(Dataset):
    def __init__(self, latent_dir, rec_dir, transform=None):
        self.latent_files = sorted(glob.glob(os.path.join(latent_dir, "*.pt")))
        if not self.latent_files:
            raise FileNotFoundError(f"No .pt files found in {latent_dir}")
        self.rec_dir = rec_dir
        self.transform = transform

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        latent_path = self.latent_files[idx]
        base_name = Path(latent_path).stem
        rec_path = os.path.join(self.rec_dir, f"{base_name}.png")
        if not os.path.exists(rec_path):
            raise FileNotFoundError(f"Missing base image for latent {latent_path}: {rec_path}")

        z = load_tensor(latent_path).float()
        x_rec = Image.open(rec_path).convert("RGB")
        if self.transform is not None:
            x_rec = self.transform(x_rec)
        return x_rec, z, base_name


def extract_unet_state_dict(checkpoint):
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if any(key.startswith("unet.") for key in state_dict):
        return {key.replace("unet.", "", 1): value for key, value in state_dict.items() if key.startswith("unet.")}
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Apply the RePack latent-guided Refiner")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--latent_dir", type=str, required=True)
    parser.add_argument("--rec_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="refined_results")
    parser.add_argument("--z_dim", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    if "RANK" in os.environ:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Processing with {world_size} GPU(s).")

    model = RefineUNet(in_channels=3 + args.z_dim, base_channels=args.base_channels)
    checkpoint = load_checkpoint(args.ckpt_path)
    model.load_state_dict(extract_unet_state_dict(checkpoint), strict=True)
    model.to(device).eval()

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    dataset = RefineDataset(args.latent_dir, args.rec_dir, transform=transform)
    sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with torch.no_grad():
        iterator = tqdm(dataloader) if local_rank == 0 else dataloader
        for x_rec, z, base_names in iterator:
            x_rec = x_rec.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            x_refined = model(x_rec, z)

            for idx, base_name in enumerate(base_names):
                out_img = torch.clamp(x_refined[idx] * 0.5 + 0.5, 0, 1)
                save_path = os.path.join(args.output_dir, f"{base_name}{args.suffix}.png")
                save_image(out_img, save_path)

    if world_size > 1:
        torch.distributed.destroy_process_group()

    if local_rank == 0:
        print(f"Finished. Refined images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
