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
from rdi.datasets.torch_dataset import Sample
from rdi.train.clf_resnet18 import train_and_eval
from rdi.train.rdi_net_train import RDINetConfig, synthesize_with_rdi_net, train_rdi_net
from rdi.utils import RunPaths, ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--train_normals", type=int, default=200)
    ap.add_argument("--synth_defects", type=int, default=200)
    ap.add_argument("--clf_epochs", type=int, default=3)
    ap.add_argument("--rdi_steps", type=int, default=800)
    ap.add_argument("--rdi_batch", type=int, default=4)
    args = ap.parse_args()

    rp = RunPaths(root=REPO_ROOT)
    split_path = rp.splits_dir / "mvtec_ad" / args.category / f"k{args.k}.json"
    split = MVTecSplit.from_json(split_path)
    built = build_samples(split, train_normals_cap=args.train_normals, seed=args.seed)

    out_dir = rp.outputs_dir / "rdi_net" / args.category / f"k{args.k}"
    ensure_dir(out_dir)

    # Train RDI-Net on K-shot real defects
    cfg = RDINetConfig(image_size=args.image_size, steps=int(args.rdi_steps), batch_size=int(args.rdi_batch))
    wpath = train_rdi_net(
        real_defect_samples=built.real_defect_samples,
        out_dir=out_dir,
        cfg=cfg,
        seed=args.seed,
        device="mps",
    )

    # Synthesize defects on normal images
    normal_paths = [p for p in split.train_normal][: min(args.train_normals, len(split.train_normal))]
    mask_paths = list(split.train_defect_masks.values())
    synth_index = synthesize_with_rdi_net(
        weights_path=wpath,
        normal_paths=normal_paths,
        mask_paths=mask_paths,
        out_dir=out_dir,
        image_size=args.image_size,
        n_synth=int(args.synth_defects),
        seed=args.seed,
        device="mps",
    )
    save_json(out_dir / "synth_index.json", {"method": "rdi_net", "items": synth_index})

    # Classifier train samples = normals + real K defects + synthesized defects
    train_samples = list(built.train_samples)
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


