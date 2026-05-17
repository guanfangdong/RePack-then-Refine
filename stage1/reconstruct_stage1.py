import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from repack.inference import (
    image_stem,
    load_image_tensor,
    load_stage1_model,
    read_image_paths,
    reconstruct_batch,
    tensor_to_pil,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstruct images with a trained RePack stage1 model.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-list", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample", action="store_true", help="Sample the posterior instead of using its mode.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_list is None and args.input_dir is None:
        raise ValueError("Pass --input-list or --input-dir.")

    model, _, device = load_stage1_model(args.config, args.ckpt, args.device)
    paths = read_image_paths(args.input_list, args.input_dir, args.limit)
    if not paths:
        raise RuntimeError("No input images found.")

    out = Path(args.out)
    gt_dir = out / "gt"
    recon_dir = out / "recon"
    gt_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, len(paths), args.batch_size)):
        batch_paths = paths[start : start + args.batch_size]
        tensors = []
        for path in batch_paths:
            tensor, image = load_image_tensor(path, args.resolution)
            tensors.append(tensor)
            tensor_to_pil(tensor).save(gt_dir / f"{image_stem(path, start + len(tensors) - 1)}.png")

        images = torch.stack(tensors, dim=0).to(device)
        recon, _, _ = reconstruct_batch(model, images, sample=args.sample)
        for offset, path in enumerate(batch_paths):
            name = image_stem(path, start + offset)
            tensor_to_pil(recon[offset]).save(recon_dir / f"{name}.png")

    print(f"Saved ground truth crops to: {gt_dir}")
    print(f"Saved reconstructions to: {recon_dir}")


if __name__ == "__main__":
    main()

