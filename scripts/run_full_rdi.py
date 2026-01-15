import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--categories",
        nargs="*",
        default=["bottle", "cable", "capsule", "hazelnut", "metal_nut"],
    )
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 5, 10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--train_normals", type=int, default=200)
    ap.add_argument("--synth_defects", type=int, default=200)
    ap.add_argument("--clf_epochs", type=int, default=3)
    ap.add_argument("--rdi_steps", type=int, default=800)
    ap.add_argument("--rdi_batch", type=int, default=4)
    args = ap.parse_args()

    py = "/opt/homebrew/bin/python3.12"
    for cat in args.categories:
        for k in args.ks:
            expected = REPO_ROOT / "outputs" / "rdi_net" / cat / f"k{k}" / "metrics.json"
            if _exists(expected):
                print(f"[SKIP] exists {expected}")
                continue
            _run(
                [
                    py,
                    str(REPO_ROOT / "scripts" / "run_rdi_net.py"),
                    "--category",
                    cat,
                    "--k",
                    str(k),
                    "--seed",
                    str(args.seed),
                    "--image_size",
                    str(args.image_size),
                    "--train_normals",
                    str(args.train_normals),
                    "--synth_defects",
                    str(args.synth_defects),
                    "--clf_epochs",
                    str(args.clf_epochs),
                    "--rdi_steps",
                    str(args.rdi_steps),
                    "--rdi_batch",
                    str(args.rdi_batch),
                ]
            )

    print("\n[OK] full RDI-Net run finished.")


if __name__ == "__main__":
    main()


