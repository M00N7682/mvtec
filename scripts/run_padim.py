import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

from rdi.datasets.build import build_samples
from rdi.datasets.mvtec_split import MVTecSplit
from rdi.baselines.padim import run_padim
from rdi.utils import RunPaths, ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--train_normals", type=int, default=200)
    args = ap.parse_args()

    rp = RunPaths(root=REPO_ROOT)
    split_path = rp.splits_dir / "mvtec_ad" / args.category / f"k{args.k}.json"
    split = MVTecSplit.from_json(split_path)
    built = build_samples(split, train_normals_cap=args.train_normals, seed=args.seed)

    train_normals = [s.image_path for s in built.train_samples if s.label == 0]
    test_paths = [(s.image_path, s.label) for s in built.test_samples]

    out_dir = rp.outputs_dir / "padim_lite" / args.category / f"k{args.k}"
    ensure_dir(out_dir)
    run_padim(
        train_normal_paths=train_normals,
        test_paths=test_paths,
        out_dir=out_dir,
        image_size=args.image_size,
        seed=args.seed,
        device="mps",
    )


if __name__ == "__main__":
    main()


