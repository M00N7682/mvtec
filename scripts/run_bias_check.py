import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.env312 import activate_local_deps

activate_local_deps()

import numpy as np

from rdi.datasets.mvtec_split import MVTecSplit
from rdi.metrics import auroc_score
from rdi.utils import RunPaths, ensure_dir, load_json, save_json


def _crop_mask_bbox(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return img[y0:y1, x0:x1, :]


def _load_rgb(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(img).astype(np.float32) / 255.0


def _load_mask(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    m = Image.open(path).convert("L").resize((size, size))
    return (np.asarray(m).astype(np.float32) / 255.0) > 0.5


def _to_tensor(batch: list[np.ndarray]) -> "Tensor":
    import torch

    x = np.stack(batch, axis=0)  # [B,H,W,3]
    return torch.from_numpy(x).permute(0, 3, 1, 2).float()


def _extract_feats(patches: list[np.ndarray], device: str = "mps") -> np.ndarray:
    import torch
    import torchvision
    import torch.nn as nn

    dev = torch.device(device if device == "mps" and torch.backends.mps.is_available() else "cpu")
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(m.children())[:-1]).to(dev).eval()  # [B,512,1,1]
    norm = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    feats = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(patches), bs):
            xb = _to_tensor(patches[i : i + bs]).to(dev)
            xb = norm(xb)
            fb = backbone(xb).squeeze(-1).squeeze(-1).detach().cpu().numpy()
            feats.append(fb)
    return np.concatenate(feats, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--method", default="copy_paste", choices=["copy_paste", "poisson_like"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--max_real", type=int, default=50)
    ap.add_argument("--max_synth", type=int, default=50)
    ap.add_argument(
        "--real_source",
        default="both",
        choices=["train", "test", "both"],
        help="Where to sample real defect patches from. Use 'both' for small K.",
    )
    args = ap.parse_args()

    rp = RunPaths(root=REPO_ROOT)
    split_path = rp.splits_dir / "mvtec_ad" / args.category / f"k{args.k}.json"
    split = MVTecSplit.from_json(split_path)

    out_dir = rp.outputs_dir / "bias_check" / args.method / args.category / f"k{args.k}"
    ensure_dir(out_dir)

    rng = random.Random(args.seed)

    # Real patches
    real_items = []
    if args.real_source in ("train", "both"):
        real_items.extend(list(split.train_defect_masks.items()))
    if args.real_source in ("test", "both"):
        real_items.extend(list(split.test_defect_masks.items()))
    rng.shuffle(real_items)
    real_items = real_items[: min(args.max_real, len(real_items))]

    real_patches = []
    for img_path, mask_path in real_items:
        img = _load_rgb(img_path, args.image_size)
        mask = _load_mask(mask_path, args.image_size)
        crop = _crop_mask_bbox(img, mask)
        if crop is None:
            continue
        from PIL import Image

        crop = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((args.patch_size, args.patch_size)))
        real_patches.append(crop.astype(np.float32) / 255.0)

    # Synth patches
    synth_index_path = rp.outputs_dir / "baselines" / args.method / args.category / f"k{args.k}" / "synth_index.json"
    if not synth_index_path.exists():
        raise SystemExit(f"missing synth_index: {synth_index_path} (run baselines first)")
    synth_index = load_json(synth_index_path)["items"]
    rng.shuffle(synth_index)
    synth_index = synth_index[: min(args.max_synth, len(synth_index))]

    synth_patches = []
    for it in synth_index:
        img = _load_rgb(Path(it["img"]), args.image_size)
        mask = _load_mask(Path(it["mask"]), args.image_size)
        crop = _crop_mask_bbox(img, mask)
        if crop is None:
            continue
        from PIL import Image

        crop = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((args.patch_size, args.patch_size)))
        synth_patches.append(crop.astype(np.float32) / 255.0)

    if len(real_patches) < 5 or len(synth_patches) < 5:
        raise SystemExit(f"not enough patches: real={len(real_patches)} synth={len(synth_patches)}")

    X = real_patches + synth_patches
    y = [0] * len(real_patches) + [1] * len(synth_patches)  # 1 = synth

    feats = _extract_feats(X, device="mps")

    # Logistic regression on top
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    probs = np.zeros(len(y), dtype=np.float32)
    y_arr = np.asarray(y)

    for tr, te in skf.split(feats, y_arr):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(feats[tr], y_arr[tr])
        probs[te] = clf.predict_proba(feats[te])[:, 1]

    auroc = auroc_score(y_arr.tolist(), probs.tolist())
    save_json(
        out_dir / "metrics.json",
        {
            "auroc": auroc,
            "label_1": "synth",
            "n_real": int(len(real_patches)),
            "n_synth": int(len(synth_patches)),
            "real_source": args.real_source,
        },
    )
    print(f"[bias] AUROC(real_vs_synth)={auroc:.4f} -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()


