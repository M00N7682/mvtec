import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

from rdi.const import DEFAULT_CATEGORIES
from rdi.utils import RunPaths, save_json


def list_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    out: list[Path] = []
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return sorted(out)


def mvtec_category_root(mvtec_root: Path, category: str) -> Path:
    # tar layout is typically mvtec_anomaly_detection/<category>/...
    # We support either:
    # - mvtec_root/mvtec_anomaly_detection/<category>
    # - mvtec_root/<category>
    if (mvtec_root / "mvtec_anomaly_detection" / category).exists():
        return mvtec_root / "mvtec_anomaly_detection" / category
    return mvtec_root / category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mvtec_root", required=True, help="Extracted MVTec root dir")
    ap.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--protocol",
        default="resplit_test_defects",
        choices=["resplit_test_defects"],
        help="We re-split by moving K defect samples from the original test defects into train_defect.",
    )
    args = ap.parse_args()

    rp = RunPaths(root=Path(__file__).resolve().parents[1])
    mvtec_root = Path(args.mvtec_root).expanduser().resolve()

    rng = random.Random(args.seed)

    for cat in args.categories:
        cat_root = mvtec_category_root(mvtec_root, cat)
        train_good = list_images(cat_root / "train" / "good")

        test_good = list_images(cat_root / "test" / "good")

        # original defects: test/<defect_type>/*.png  (excluding 'good')
        defect_imgs: list[Path] = []
        defect_mask_imgs: dict[str, Path] = {}  # image_path -> mask_path (if exists)
        for defect_type_dir in (cat_root / "test").iterdir():
            if not defect_type_dir.is_dir():
                continue
            if defect_type_dir.name == "good":
                continue
            for img in list_images(defect_type_dir):
                defect_imgs.append(img)
                # ground_truth/<defect_type>/<img_name>_mask.png (varies)
                gt_dir = cat_root / "ground_truth" / defect_type_dir.name
                # Common MVTec mask naming: <img_name>_mask.png
                mask = gt_dir / f"{img.stem}_mask.png"
                if mask.exists():
                    defect_mask_imgs[str(img)] = mask

        defect_imgs = sorted(defect_imgs)
        if not train_good or not test_good or not defect_imgs:
            print(f"[WARN] category {cat}: missing expected dirs/images. Skipping.")
            continue

        for k in args.ks:
            k = int(k)
            if k > len(defect_imgs):
                print(f"[WARN] category {cat} k={k}: not enough defect imgs ({len(defect_imgs)}). Skipping.")
                continue

            chosen = rng.sample(defect_imgs, k)
            chosen_set = {str(p) for p in chosen}

            # Test set excludes the K moved samples
            test_defects_remaining = [p for p in defect_imgs if str(p) not in chosen_set]

            split = {
                "dataset": "mvtec_ad",
                "category": cat,
                "protocol": args.protocol,
                "seed": args.seed,
                "k_defects": k,
                "train": {
                    "normal": [str(p) for p in train_good],
                    "defect": [str(p) for p in chosen],
                    "defect_masks": {str(p): str(defect_mask_imgs[str(p)]) for p in chosen if str(p) in defect_mask_imgs},
                },
                "test": {
                    "normal": [str(p) for p in test_good],
                    "defect": [str(p) for p in test_defects_remaining],
                    "defect_masks": {
                        str(p): str(defect_mask_imgs[str(p)])
                        for p in test_defects_remaining
                        if str(p) in defect_mask_imgs
                    },
                },
            }

            out_path = rp.splits_dir / "mvtec_ad" / cat / f"k{k}.json"
            save_json(out_path, split)
            print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()


