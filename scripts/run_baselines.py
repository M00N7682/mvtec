import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

from rdi.config import default_config
from rdi.datasets.build import build_samples
from rdi.datasets.mvtec_split import MVTecSplit
from rdi.utils import RunPaths, ensure_dir, save_json
from rdi.train.clf_resnet18 import train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--method", default="copy_paste", choices=["no_synth", "copy_paste", "poisson_like"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--train_normals", type=int, default=200)
    ap.add_argument("--synth_defects", type=int, default=200)
    ap.add_argument("--clf_epochs", type=int, default=10)
    args = ap.parse_args()

    cfg = default_config()
    rp = RunPaths(root=Path(__file__).resolve().parents[1])

    split_path = rp.splits_dir / "mvtec_ad" / args.category / f"k{args.k}.json"
    split = MVTecSplit.from_json(split_path)
    built = build_samples(split, train_normals_cap=args.train_normals, seed=args.seed)

    out_dir = rp.outputs_dir / "baselines" / args.method / args.category / f"k{args.k}"
    ensure_dir(out_dir)

    # synthesize defects if requested
    synth_dir = out_dir / "synthetic"
    ensure_dir(synth_dir)
    rng = random.Random(args.seed)

    synth_index = []
    if args.method != "no_synth":
        from rdi.augment.copy_paste import build_patch_bank, synthesize_one
        from PIL import Image
        import numpy as np

        patch_bank = build_patch_bank(built.real_defect_samples)
        if not patch_bank:
            raise SystemExit("No patch bank available (missing defect masks?).")

        normals = [s.image_path for s in built.train_samples if s.label == 0]
        rng.shuffle(normals)
        normals = normals[: min(args.synth_defects, len(normals))]

        mode = "alpha" if args.method == "copy_paste" else "poisson_like"
        for i, npath in enumerate(normals):
            rgb, used_mask = synthesize_one(npath, patch_bank, args.image_size, rng, mode=mode)
            out_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            out_mask = (np.clip(used_mask, 0, 1) * 255).astype(np.uint8)
            img_path = synth_dir / f"synth_{i:04d}.png"
            mask_path = synth_dir / f"synth_{i:04d}_mask.png"
            Image.fromarray(out_img).save(img_path)
            Image.fromarray(out_mask).save(mask_path)
            synth_index.append({"img": str(img_path), "mask": str(mask_path)})

    save_json(out_dir / "synth_index.json", {"method": args.method, "items": synth_index})
    print(f"[OK] wrote {out_dir / 'synth_index.json'}")

    # Build classifier training samples (append synth as defect)
    train_samples = list(built.train_samples)
    if synth_index:
        from rdi.datasets.torch_dataset import Sample

        for it in synth_index:
            train_samples.append(Sample(Path(it["img"]), 1, Path(it["mask"])))

    train_and_eval(
        train_samples=train_samples,
        test_samples=built.test_samples,
        out_dir=out_dir,
        image_size=args.image_size,
        epochs=int(args.clf_epochs),
        batch_size=32,
        lr=3e-4,
        weight_decay=1e-4,
        seed=args.seed,
        device="mps",
    )


if __name__ == "__main__":
    main()


