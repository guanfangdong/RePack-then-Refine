import argparse
import json
import os
from time import strftime


def print_with_prefix(*messages):
    prefix = f"\033[34m[RePack-DiT FID {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    combined_message = " ".join(map(str, messages))
    print(f"{prefix}: {combined_message}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FID for RePack-DiT generated images")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory with generated PNG images")
    parser.add_argument("--ref_npz", type=str, required=True, help="Reference ImageNet .npz file")
    parser.add_argument("--fid_num", type=int, default=50000, help="Number of generated samples used for FID")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--dims", type=int, default=2048, help="Inception feature dimension")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--output", type=str, default=None, help="Optional path for saving a JSON result")
    args = parser.parse_args()

    from tools.calculate_fid import calculate_fid_given_paths

    print_with_prefix(f"Calculating FID with {args.fid_num} samples")
    print_with_prefix("Reference NPZ:", args.ref_npz)
    print_with_prefix("Image Directory:", args.img_dir)

    fid = calculate_fid_given_paths(
        [args.ref_npz, args.img_dir],
        batch_size=args.batch_size,
        dims=args.dims,
        device="cuda",
        num_workers=args.num_workers,
        sp_len=args.fid_num,
    )

    print_with_prefix("fid=", fid)
    if args.output is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        result = {
            "img_dir": args.img_dir,
            "ref_npz": args.ref_npz,
            "fid_num": args.fid_num,
            "fid": float(fid),
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print_with_prefix("Saved FID result:", args.output)
