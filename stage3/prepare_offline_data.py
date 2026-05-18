import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from tqdm import tqdm


_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_STAGE2_ROOT = _RELEASE_ROOT / "stage2"
if _STAGE2_ROOT.is_dir() and str(_STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE2_ROOT))

from tokenizer import RePackTokenizer  # noqa: E402


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path


def save_tensor_image(tensor, path):
    image = (tensor + 1.0) / 2.0
    image = image.clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def safe_stem(path, root):
    rel = Path(path).resolve().relative_to(Path(root).resolve())
    return f"{rel.parent.name}_{rel.stem}"


def main():
    parser = argparse.ArgumentParser(description="Build offline training triplets for the RePack Refiner")
    parser.add_argument("--image_root", type=str, required=True, help="ImageFolder-style ImageNet root")
    parser.add_argument("--config", type=str, default="../stage2/configs/repack_f16d32_dinov3.yaml")
    parser.add_argument("--output_dir", type=str, default="./offline_refiner_data")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        for name in ["gt", "rec", "latent"]:
            (output_dir / name).mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    tokenizer = RePackTokenizer(args.config, sample_posterior=False, load_encoder=True)
    dataset = ImageFolderWithPath(args.image_root, transform=tokenizer.img_transform(p_hflip=0.0))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    iterator = tqdm(dataloader, desc="Preparing refiner data") if rank == 0 else dataloader
    with torch.no_grad():
        for images, _, paths in iterator:
            images = images.to(device, non_blocking=True)
            z = tokenizer.encode_images(images)
            rec_images = tokenizer.decode_to_images(z)

            for idx, source_path in enumerate(paths):
                name = safe_stem(source_path, args.image_root)
                torch.save(z[idx].cpu(), output_dir / "latent" / f"{name}.pt")
                save_tensor_image(images[idx], output_dir / "gt" / f"{name}.png")
                Image.fromarray(rec_images[idx]).save(output_dir / "rec" / f"{name}.png")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"Offline refiner data saved to: {output_dir}")


if __name__ == "__main__":
    main()
