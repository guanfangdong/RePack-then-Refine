import argparse
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from repack.inference import (
    image_stem,
    load_image_tensor,
    load_stage1_model,
    read_image_paths,
    reconstruct_batch,
    tensor_to_pil,
)


def latent_to_rgb(z):
    if z.dim() == 4:
        z = z[0]
    channels, height, width = z.shape
    flat = z.detach().float().permute(1, 2, 0).reshape(-1, channels).cpu()
    flat = flat - flat.mean(dim=0, keepdim=True)
    if channels >= 3:
        _, _, v = torch.pca_lowrank(flat, q=3)
        rgb = flat @ v[:, :3]
    else:
        rgb = torch.nn.functional.pad(flat, (0, 3 - channels))
    rgb = rgb.reshape(height, width, 3)
    rgb = rgb - rgb.amin(dim=(0, 1), keepdim=True)
    rgb = rgb / rgb.amax(dim=(0, 1), keepdim=True).clamp_min(1e-8)
    rgb = (rgb * 255).round().byte().numpy()
    return Image.fromarray(rgb, mode="RGB")


def make_triplet(input_img, recon_img, latent_img, resolution):
    latent_img = latent_img.resize((resolution, resolution), Image.NEAREST)
    canvas = Image.new("RGB", (resolution * 3, resolution), "white")
    canvas.paste(input_img, (0, 0))
    canvas.paste(recon_img, (resolution, 0))
    canvas.paste(latent_img, (resolution * 2, 0))
    return canvas


def parse_args():
    parser = argparse.ArgumentParser(description="Save input/reconstruction/latent PCA visualizations.")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-list", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--limit", type=int, default=32)
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
    out.mkdir(parents=True, exist_ok=True)

    for index, path in enumerate(tqdm(paths)):
        tensor, input_img = load_image_tensor(path, args.resolution)
        images = tensor.unsqueeze(0).to(device)
        recon, _, z = reconstruct_batch(model, images, sample=args.sample)
        recon_img = tensor_to_pil(recon[0])
        latent_img = latent_to_rgb(z[0])
        triplet = make_triplet(input_img, recon_img, latent_img, args.resolution)
        triplet.save(out / f"{image_stem(path, index)}_triplet.png")

    print(f"Saved visualizations to: {out}")


if __name__ == "__main__":
    main()
