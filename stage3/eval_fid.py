import argparse
import json
import os
import sys
from pathlib import Path
from time import strftime


_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_STAGE2_ROOT = _RELEASE_ROOT / "stage2"
if _STAGE2_ROOT.is_dir() and str(_STAGE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE2_ROOT))

from tools.calculate_fid import calculate_fid_given_paths  # noqa: E402


def print_with_prefix(*messages):
    prefix = f"\033[34m[RePack Refiner FID {strftime('%Y-%m-%d %H:%M:%S')}]\033[0m"
    print(f"{prefix}: {' '.join(map(str, messages))}")


def main():
    parser = argparse.ArgumentParser(description="Compute FID for refined images")
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--ref_npz", type=str, required=True)
    parser.add_argument("--fid_num", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print_with_prefix(f"Calculating FID with {args.fid_num} samples")
    fid = calculate_fid_given_paths(
        [args.ref_npz, args.img_dir],
        batch_size=args.batch_size,
        dims=args.dims,
        device="cuda",
        num_workers=args.num_workers,
        sp_len=args.fid_num,
    )
    print_with_prefix("fid=", fid)

    if args.output:
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


if __name__ == "__main__":
    main()
